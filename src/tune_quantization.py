"""Optuna multi-objective hyperparameter optimization for model quantization.

Searches across quantization methods (dense/no quantization, MoMos2D, QAT, v-fold
/ Hierarchical MoMos2D, Fold MoMos, Soft MoMos) to construct the Pareto front maximizing
Validation Accuracy and Compression Rate.

Includes automated K-fold cross-validation verification for Pareto-optimal trials.
"""

import copy
import gc
import os
import statistics
import hydra
import optuna
from omegaconf import DictConfig, OmegaConf, open_dict
from optuna.samplers import NSGAIISampler
import torch

from data import ImageDataModule
from model import build_backbone
from train import build_datamodule, run_training
from quantizers.soft_momos import SOFT_MOMOS_METHODS
from utils.quant_bits import compute_quantization_bits


def _suggest_block(trial: optuna.Trial, space: DictConfig, prefix: str) -> dict:
    """Sample a rows/cols/capacity block under `prefix_*` param names.

    force_zero is always True (not part of the search space): a dedicated
    zero motif must always be available so blocks can be pruned to zero.
    """
    cap_spec = space.capacity
    return {
        "rows": int(trial.suggest_categorical(f"{prefix}_rows", list(space.rows))),
        "cols": int(trial.suggest_categorical(f"{prefix}_cols", list(space.cols))),
        "capacity": float(
            trial.suggest_float(
                f"{prefix}_capacity",
                float(cap_spec.low),
                float(cap_spec.high),
                log=bool(cap_spec.log),
            )
        ),
        "force_zero": True,
    }


def _suggest_quant_config(
    trial: optuna.Trial, search_space: DictConfig, methods: list[str]
) -> dict:
    """Sample a quantization configuration from the search space."""
    method = trial.suggest_categorical("method", methods)

    if method == "none":
        # Dense baseline: no quantization at all. compression_rate is exactly
        # 1.0 (compute_quantization_bits' enabled=False fallback) and
        # build_callbacks skips QuantizationCallback entirely, so this trains
        # like a plain unquantized run -- an anchor point on the accuracy axis
        # for the rest of the Pareto front to be compared against.
        return {
            "enabled": False,
            "method": "none",
        }

    elif method == "momos2d":
        return {
            "enabled": True,
            "method": "momos2d",
            **_suggest_block(trial, search_space.momos2d, "momos2d"),
        }

    elif method == "qat":
        space = search_space.qat
        q = trial.suggest_categorical("qat_q", list(space.q))
        return {
            "enabled": True,
            "method": "qat",
            "q": int(q),
            "exclude_layers": [],
        }

    elif method == "hierarchical_momos2d":
        space = search_space.hierarchical_momos2d
        return {
            "enabled": True,
            "method": "hierarchical_momos2d",
            "switch_fraction": float(space.get("switch_fraction", 0.0)),
            "primary": _suggest_block(trial, space.primary, "hm_p"),
            "secondary": _suggest_block(trial, space.secondary, "hm_s"),
        }

    elif method == "fold_momos":
        space = search_space.fold_momos
        return {
            "enabled": True,
            "method": "fold_momos",
            "switch_fraction": float(space.get("switch_fraction", 0.0)),
            "primary": _suggest_block(trial, space.primary, "fm_p"),
            "secondary": _suggest_block(trial, space.secondary, "fm_s"),
        }

    elif method in SOFT_MOMOS_METHODS:
        return _suggest_soft_momos(trial, search_space[method], method)

    else:
        raise ValueError(f"Unsupported quantization method for sampling: {method}")


# Optuna parameter-name prefix per soft-MoMos method (shared with eval_pareto_cv).
SOFT_MOMOS_PREFIX = {"soft_momos": "sm", "soft_momos_plus": "smp"}


def _suggest_soft_momos(trial: optuna.Trial, space: DictConfig, method: str) -> dict:
    """Sample a soft_momos / soft_momos_plus config.

    Block shape and the penalty weight are searched; the codebook size is fixed
    (k = 256 -> 8 bits per block; soft_momos also has final_k = 256, soft_momos_plus has none: its trained codebook is final), so compression is set purely by the
    block size and lam only trades accuracy for distortion. `k: null` with a
    `capacity` block falls back to searching capacity. soft_momos_plus also turns
    warm start on and takes `freeze_fraction` from the space: a {low, high} spec
    is searched, a plain number is fixed.
    """
    pre = SOFT_MOMOS_PREFIX[method]
    lam_spec = space.lam
    out = {
        "enabled": True,
        "method": method,
        "rows": int(trial.suggest_categorical(f"{pre}_rows", list(space.rows))),
        "cols": int(trial.suggest_categorical(f"{pre}_cols", list(space.cols))),
        "force_zero": True,
        "lam": float(
            trial.suggest_float(
                f"{pre}_lam",
                float(lam_spec.low),
                float(lam_spec.high),
                log=bool(lam_spec.log),
            )
        ),
    }
    # Optional growing-L1 term: l1_final is searched ({low, high, log}) or fixed.
    if space.get("l1_final") is not None:
        l1_final = space.l1_final
        if hasattr(l1_final, "low"):
            l1_final = trial.suggest_float(
                f"{pre}_l1_final",
                float(l1_final.low),
                float(l1_final.high),
                log=bool(l1_final.get("log", False)),
            )
        out["l1_final"] = float(l1_final)
        for key in ("l1_init", "l1_power"):
            if space.get(key) is not None:
                out[key] = float(space[key])
        if space.get("l1_schedule") is not None:
            out["l1_schedule"] = str(space.l1_schedule)
    if method == "soft_momos":
        out["final_k"] = int(space.final_k)
    if method == "soft_momos_plus":
        freeze = space.freeze_fraction
        if hasattr(freeze, "low"):
            freeze = trial.suggest_float(
                f"{pre}_freeze_fraction", float(freeze.low), float(freeze.high)
            )
        out.update(
            warm_start=True,
            reseed_dead=True,
            lloyd_iters=int(space.get("lloyd_iters", 1)),
            freeze_fraction=float(freeze),
        )
    if space.get("k") is not None:
        out["k"] = int(space.k)
        out["capacity"] = None
    else:
        cap_spec = space.capacity
        out["k"] = None
        out["capacity"] = float(
            trial.suggest_float(
                f"{pre}_capacity",
                float(cap_spec.low),
                float(cap_spec.high),
                log=bool(cap_spec.log),
            )
        )
    return out


def _build_datamodule(cfg: DictConfig, fold: int) -> ImageDataModule:
    """Build a fresh ImageDataModule for `fold`.

    A brand-new instance is created on every call so each Optuna trial (and
    each Pareto K-fold verification run) owns its own DataLoaders and
    ``CombinedLoader`` state. Reusing a single instance across trials left a
    stale ``_iterator=None`` ``CombinedLoader`` behind after Lightning's
    teardown (``combined_loader.reset()``), which then raised
    ``RuntimeError: Please call `iter(combined_loader)` first.`` on the
    following trial. Building fresh per call sidesteps that shared-state
    leak. The dataset content only depends on ``fold`` (not on
    quantization/seed/prefix), so the reload cost is a single dataset
    construction per distinct fold.
    """
    fold_cfg = copy.deepcopy(cfg)
    with open_dict(fold_cfg):
        fold_cfg.fold = int(fold)
    return build_datamodule(fold_cfg)


def _bump_fd_limit(target: int = 65536) -> None:
    """Raise the process open-file (RLIMIT_NOFILE) soft limit.

    The multi-trial Optuna study spawns many DataLoader worker processes
    across 25+ trials; combined with ``file_system`` tensor sharing and W&B,
    the default soft limit (often 1024) is eventually exhausted -- ``errno 24,
    Too many open files`` -- and the next worker spawn's ``os.pipe()`` fails.
    This is a safety net; also raise the limit at launch (e.g.
    ``ulimit -n 65536`` in the srun/sbatch wrapper).
    """
    try:
        import resource
    except ImportError:  # non-Unix platforms
        return
    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    except (ValueError, OSError) as exc:
        print(f"[tune] could not read RLIMIT_NOFILE: {exc}")
        return
    new_soft = min(max(soft, target), hard) if hard > 0 else target
    if new_soft <= soft:
        return
    try:
        resource.setrlimit(resource.RLIMIT_NOFILE, (new_soft, hard))
        print(
            f"[tune] raised RLIMIT_NOFILE soft limit {soft} -> {new_soft} (hard {hard})"
        )
    except (ValueError, OSError) as exc:
        print(f"[tune] could not raise RLIMIT_NOFILE to {new_soft}: {exc}")


def _objective_factory(cfg: DictConfig, backbone_sample: torch.nn.Module):
    """Build the Optuna objective function."""
    opt_cfg = cfg.optuna
    search_space = opt_cfg.search_space
    methods = list(opt_cfg.methods)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        quant_cfg = _suggest_quant_config(trial, search_space, methods)

        # --- Exact-parameter reuse -------------------------------------------------
        # If a previously *completed* trial already evaluated this exact parameter
        # set, reuse its stored accuracy + compression rate instead of re-training.
        # This only runs in the search phase
        if bool(opt_cfg.get("reuse_completed_trials", True)):
            existing = trial.study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
            for past in existing:
                if past.params == trial.params and past.values is not None:
                    val_acc, comp_rate = float(past.values[0]), float(past.values[1])
                    trial.set_user_attr("val_acc", val_acc)
                    trial.set_user_attr("compression_rate", comp_rate)
                    trial.set_user_attr("reused_trial", int(past.number))
                    return val_acc, comp_rate

        # Theoretical bit calculation before full training
        bit_stats = compute_quantization_bits(backbone_sample, quant_cfg)
        compression_rate = float(bit_stats["compression_rate"])
        bpp = float(bit_stats["bpp"])

        trial.set_user_attr("compression_rate", compression_rate)
        trial.set_user_attr("compressed_bits", float(bit_stats["compressed_bits"]))
        trial.set_user_attr("bpp", bpp)
        trial.set_user_attr("method", quant_cfg["method"])

        # Create config copy for this trial
        trial_cfg = copy.deepcopy(cfg)
        with open_dict(trial_cfg):
            trial_cfg.quantization = OmegaConf.create(quant_cfg)
            trial_cfg.prefix = f"{cfg.prefix}/trial-{trial.number}"

        # Evaluate on search folds (default fold 0)
        search_folds = list(opt_cfg.get("eval_folds_search", [0]))
        val_accs = []
        for fold in search_folds:
            fold_cfg = copy.deepcopy(trial_cfg)
            with open_dict(fold_cfg):
                fold_cfg.fold = int(fold)
                if not cfg.get("wandb", {}).get("enabled", False):
                    fold_cfg.wandb.enabled = False
                else:
                    fold_cfg.wandb.name = (
                        f"trial-{trial.number}_{quant_cfg['method']}_f{fold}"
                    )

            dm = None
            res = None
            val_acc = None
            try:
                dm = _build_datamodule(cfg, fold)
                res = run_training(fold_cfg, optuna_trial=trial, datamodule=dm)
                val_acc = res["val_acc"]
            except optuna.TrialPruned:
                raise
            except (RuntimeError, ValueError) as exc:
                msg = str(exc).lower()
                if (
                    "out of memory" in msg
                    or "cuda" in msg
                    or "too many open files" in msg
                    or "errno 24" in msg
                ):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    raise optuna.TrialPruned() from exc
                raise
            finally:
                # Release DataLoader workers / GPU refs promptly so open file
                # descriptors don't accumulate across the long study
                # (errno 24, "Too many open files").
                del dm, res
                gc.collect()
            val_accs.append(val_acc)

        mean_val_acc = statistics.mean(val_accs) if val_accs else 0.0
        trial.set_user_attr("val_acc", mean_val_acc)

        return mean_val_acc, compression_rate

    return objective


@hydra.main(config_path="configs", config_name="tune_quantization", version_base="1.3")
def main(cfg: DictConfig):
    opt_cfg = cfg.optuna
    storage_dir = str(opt_cfg.storage_dir)
    os.makedirs(storage_dir, exist_ok=True)

    study_name = str(opt_cfg.study_name)
    storage_url = f"sqlite:///{os.path.join(storage_dir, study_name + '.db')}"

    # Build reference backbone for bit estimation
    img_size = cfg.model.get("img_size", None) or cfg.dataset.img_size
    backbone_sample = build_backbone(
        cfg.model,
        in_channels=cfg.dataset.in_channels,
        img_size=img_size,
        num_classes=cfg.dataset.num_classes,
    )

    # Feasibility floor: trials below min_val_accuracy are marked infeasible
    # and excluded from the Pareto front. This drops the degenerate
    # ~random-guess points (CIFAR-10 -> 10% acc) that otherwise dominate only
    # because of extreme compression, while keeping the full 2-D trade-off.
    min_val_accuracy = float(opt_cfg.get("min_val_accuracy", 0.20))

    def _constraints_func(trial: optuna.trial.FrozenTrial) -> list[float]:
        acc = float(trial.user_attrs.get("val_acc", 0.0))
        # constraint <= 0 means feasible; > 0 means infeasible
        return [min_val_accuracy - acc]

    sampler = NSGAIISampler(
        seed=int(opt_cfg.sampler.seed),
        constraints_func=_constraints_func,
    )

    # Multi-objective study: Maximize Accuracy and Maximize Compression Rate.
    # Native Optuna pruning (trial.report/should_prune, optuna.pruners.*) is
    # unsupported for multi-objective studies, so pruning is instead handled
    # manually by OptunaMultiObjectiveEpochPruning     
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        directions=["maximize", "maximize"],
        sampler=sampler,
        pruner=optuna.pruners.NopPruner(),
        load_if_exists=True,
    )
    study.set_user_attr("min_val_accuracy", min_val_accuracy)

    print("\n========================================================")
    print(" Starting Optuna Multi-Objective Quantization Search")
    print(f" Study Name : {study_name}")
    print(f" Storage    : {storage_url}")
    print(f" Trials     : {opt_cfg.n_trials}")
    print(" Targets    : [Maximize Accuracy, Maximize Compression Rate]")
    print("========================================================\n")

    _bump_fd_limit()

    objective = _objective_factory(cfg, backbone_sample)
    study.optimize(objective, n_trials=int(opt_cfg.n_trials), gc_after_trial=True)

    # Save full history to CSV
    csv_path = os.path.join(storage_dir, f"{study_name}.csv")
    df = study.trials_dataframe()
    df.to_csv(csv_path, index=False)
    print(f"\nFull trial history written to: {csv_path}")

    # Extract Pareto Front
    pareto_trials = study.best_trials
    print("\n========================================================")
    print(f" PARETO FRONT SUMMARY ({len(pareto_trials)} Pareto-optimal trials)")
    print("========================================================")

    pareto_summary = []
    for t in pareto_trials:
        val_acc, comp_rate = t.values[0], t.values[1]  # type: ignore
        bpp = t.user_attrs.get("bpp", 0.0)
        method = t.user_attrs.get("method", "unknown")
        print(
            f"Trial #{t.number:03d} | Method: {method:20s} | Val Acc: {val_acc:.4f} | Comp Rate: {comp_rate:.2f}x | BPP: {bpp:.2f}"
        )
        pareto_summary.append(
            {
                "trial_number": t.number,
                "method": method,
                "val_acc": float(val_acc),
                "compression_rate": float(comp_rate),
                "bpp": float(bpp),
                "params": t.params,
            }
        )

    pareto_yaml_path = os.path.join(storage_dir, f"{study_name}_pareto.yaml")
    with open(pareto_yaml_path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create({"pareto_trials": pareto_summary})))
    print(f"Pareto summary written to: {pareto_yaml_path}")

    # Phase 2: K-Fold Verification for Pareto-best configurations
    if opt_cfg.get("pareto_kfold_eval", False) and cfg.dataset.get("n_folds", 1) > 1:
        n_folds = int(cfg.dataset.n_folds)
        seeds = list(opt_cfg.get("pareto_seeds", [cfg.seed]))
        print("\n========================================================")
        print(
            f" Starting Pareto K-Fold Verification Across {n_folds} Folds & {len(seeds)} Seeds"
        )
        print("========================================================\n")

        pareto_kfold_results = []
        for t in pareto_trials:
            method = t.user_attrs.get("method", "unknown")
            print(
                f"--- Evaluating Pareto Trial #{t.number} ({method}) across {n_folds} folds x {len(seeds)} seeds ---"
            )

            # Reconstruct quant_cfg from trial params
            sample_space = opt_cfg.search_space
            dummy_trial = optuna.trial.FixedTrial(t.params)
            quant_cfg = _suggest_quant_config(
                dummy_trial,  # type: ignore
                sample_space,
                list(opt_cfg.methods),
            )

            fold_accs = []
            for seed in seeds:
                for fold_idx in range(n_folds):
                    eval_cfg = copy.deepcopy(cfg)
                    with open_dict(eval_cfg):
                        eval_cfg.seed = int(seed)
                        eval_cfg.fold = int(fold_idx)
                        eval_cfg.quantization = OmegaConf.create(quant_cfg)
                        eval_cfg.prefix = f"{cfg.prefix}/pareto_eval/trial-{t.number}/seed-{seed}_fold-{fold_idx}"

                        if eval_cfg.get("wandb", {}).get("enabled", False):
                            eval_cfg.wandb.name = (
                                f"pareto-t{t.number}_s{seed}_f{fold_idx}"
                            )

                    dm = None
                    res = None
                    try:
                        dm = _build_datamodule(cfg, fold_idx)
                        res = run_training(eval_cfg, datamodule=dm)
                        fold_accs.append(res["val_acc"])
                    finally:
                        # run_training now drops trainer/model/datamodule from its
                        # return dict and GCs in its own finally; this extra
                        # release keeps the per-(seed,fold) FD count flat so the
                        # long Pareto loop doesn't accumulate workers either.
                        del dm, res
                        gc.collect()

            mean_acc = statistics.mean(fold_accs)
            std_acc = statistics.pstdev(fold_accs)

            print(
                f"Trial #{t.number} ({method}) -> Overall K-Fold Val Acc: {mean_acc:.4f} ± {std_acc:.4f}"
            )
            pareto_kfold_results.append(
                {
                    "trial_number": t.number,
                    "method": method,
                    "kfold_val_acc_mean": mean_acc,
                    "kfold_val_acc_std": std_acc,
                    "compression_rate": float(t.values[1]),  # type: ignore
                    "bpp": float(t.user_attrs.get("bpp", 0.0)),
                    "params": t.params,
                }
            )

        kfold_yaml_path = os.path.join(storage_dir, f"{study_name}_pareto_kfold.yaml")
        with open(kfold_yaml_path, "w") as f:
            f.write(
                OmegaConf.to_yaml(
                    OmegaConf.create(
                        {"pareto_kfold_verification": pareto_kfold_results}
                    )
                )
            )
        print(f"\nPareto K-Fold verification results written to: {kfold_yaml_path}")


if __name__ == "__main__":
    main()
