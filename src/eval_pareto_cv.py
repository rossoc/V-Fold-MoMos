"""5-fold cross-validation + held-out test verification for Optuna Pareto trials.

Consumes a Pareto-summary YAML written by `tune_quantization.py` (top-level
key `pareto_trials`, each entry with `trial_number`/`method`/`val_acc`/
`compression_rate`/`bpp`/`params`) and, for every *distinct* hyperparameter
configuration, retrains it across all `dataset.n_folds` folds and reports
mean +/- std validation and test accuracy.

The `val_acc` in the source file comes from a single fold (search-time
shortcut in `tune_quantization.py`), so it's a noisy point estimate; this
script exists to get a robust per-config estimate before committing to a
final configuration.

Duplicate trials (identical `params`, different `trial_number` -- including
ones NOT tagged `method: unknown`) are collapsed to a single configuration
so they're only trained once. See `_dedupe_trials`.
"""

import copy
import gc
from pathlib import Path
import statistics

import hydra
from omegaconf import DictConfig, OmegaConf, open_dict
import optuna
import torch

from model import build_backbone
from train import build_datamodule, run_training
from tune_quantization import _bump_fd_limit, _suggest_quant_config
from utils.quant_bits import compute_quantization_bits

# Real (non-baseline) methods `_suggest_quant_config` knows how to rebuild.
_ALL_METHODS = ["momos2d", "qat", "hierarchical_momos2d", "soft_momos"]

_NO_QUANT_METHOD_NAMES = {None, "none", "None", "null", ""}


def _params_key(params: dict) -> tuple:
    return tuple(sorted(params.items()))


def _load_pareto_trials(path: str) -> list[dict]:
    data = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    trials = data.get("pareto_trials") or []
    if not trials:
        raise ValueError(f"No 'pareto_trials' entries found in {path}")
    return sorted(trials, key=lambda t: int(t["trial_number"]))


def _dedupe_trials(trials: list[dict]) -> list[dict]:
    """Collapse trials that share an identical `params` dict.

    Duplicates come from Optuna's exact-parameter reuse short-circuit in
    `tune_quantization.py::_objective_factory` (it skips retraining and
    copies a prior COMPLETE trial's metrics). The `method: unknown` tag is a
    symptom of that path (it returns before `trial.set_user_attr("method",
    ...)` runs) but not the full signal -- duplicates also occur between two
    normally-tagged trials (e.g. `qat_q=4` sampled twice, both labeled
    `qat`). So dedup on `params` content itself, not the `method` label.
    """
    grouped: dict[tuple, dict] = {}
    for t in trials:
        key = _params_key(t["params"])
        if key not in grouped:
            grouped[key] = {**t, "source_trial_numbers": [int(t["trial_number"])]}
        else:
            grouped[key]["source_trial_numbers"].append(int(t["trial_number"]))
    return list(grouped.values())


def _ensure_categorical_choices(base_choices, value):
    """Widen a categorical choice list so it's guaranteed to contain `value`.

    Optuna's `FixedTrial` handles out-of-range `suggest_float` values by
    only *warning* (range-checking happens after a lossless log/linear
    transform) -- but `suggest_categorical` looks the fixed value up by
    *index* in the choices list, which raises `ValueError` outright if it
    isn't present. `cfg.quant_search_space`'s row/col/q lists are just a
    static default, not a hard ceiling on what a real trial may have used,
    so every categorical suggestion in `_suggest_quant_config` (method,
    qat_q, momos2d_rows/cols, hm_p/hm_s rows/cols) needs its actual value
    unioned in before reconstruction -- otherwise a config using a
    row/col/q outside the configured defaults hard-fails instead of just
    reconstructing correctly.
    """
    choices = list(base_choices)
    return choices if value in choices else [value, *choices]


def _reconstruct_quant_cfg(params: dict, search_space: DictConfig) -> dict:
    """Rebuild a full quantization config from a trial's flat `params` dict.

    Mirrors the Pareto K-fold verification phase already in
    `tune_quantization.py::main` (`optuna.trial.FixedTrial(t.params)` +
    `_suggest_quant_config`), just lifted out into a standalone script, plus
    the "no quantization" baseline case tune_quantization.py never needed
    (see `_NO_QUANT_METHOD_NAMES`) and the choice-widening in
    `_ensure_categorical_choices`.
    """
    method = params.get("method")
    if method in _NO_QUANT_METHOD_NAMES:
        return {"enabled": False, "method": None}

    space = copy.deepcopy(search_space)
    with open_dict(space):
        if method == "momos2d":
            space.momos2d.rows = _ensure_categorical_choices(space.momos2d.rows, params["momos2d_rows"])
            space.momos2d.cols = _ensure_categorical_choices(space.momos2d.cols, params["momos2d_cols"])
        elif method == "qat":
            space.qat.q = _ensure_categorical_choices(space.qat.q, params["qat_q"])
        elif method == "hierarchical_momos2d":
            p, s = space.hierarchical_momos2d.primary, space.hierarchical_momos2d.secondary
            p.rows = _ensure_categorical_choices(p.rows, params["hm_p_rows"])
            p.cols = _ensure_categorical_choices(p.cols, params["hm_p_cols"])
            s.rows = _ensure_categorical_choices(s.rows, params["hm_s_rows"])
            s.cols = _ensure_categorical_choices(s.cols, params["hm_s_cols"])
        elif method == "soft_momos":
            space.soft_momos.rows = _ensure_categorical_choices(space.soft_momos.rows, params["sm_rows"])
            space.soft_momos.cols = _ensure_categorical_choices(space.soft_momos.cols, params["sm_cols"])
        else:
            raise ValueError(
                f"Unsupported quantization method in pareto entry params: {method!r} "
                f"(expected one of {_ALL_METHODS} or a no-quantization marker "
                f"{sorted({str(m) for m in _NO_QUANT_METHOD_NAMES})})"
            )

    dummy_trial = optuna.trial.FixedTrial(dict(params))
    methods = _ensure_categorical_choices(_ALL_METHODS, method)
    return _suggest_quant_config(dummy_trial, space, methods)


def _run_fold(
    cfg: DictConfig,
    quant_cfg: dict,
    method: str,
    trial_number: int,
    fold_idx: int,
    seed: int,
) -> dict:
    """Train + validate + test one (config, fold) pair. Never raises on OOM."""
    eval_cfg = copy.deepcopy(cfg)
    with open_dict(eval_cfg):
        eval_cfg.quantization = OmegaConf.create(quant_cfg)
        eval_cfg.fold = int(fold_idx)
        eval_cfg.seed = int(seed)
        eval_cfg.prefix = f"{cfg.prefix}/{method}_t{trial_number}/fold{fold_idx}"
        if eval_cfg.get("wandb", {}).get("enabled", False):
            eval_cfg.wandb.name = f"{method}_t{trial_number}_f{fold_idx}"

    dm = None
    res = None
    try:
        dm = build_datamodule(eval_cfg)
        res = run_training(eval_cfg, datamodule=dm)
        return {
            "fold": fold_idx,
            "seed": int(seed),
            "val_acc": res["val_acc"],
            "val_loss": res["val_loss"],
            "test_acc": res["test_acc"],
            "test_loss": res["test_loss"],
            "metrics": res.get("metrics", {}),
            "metrics_history": res.get("metrics_history", []),
        }
    except (RuntimeError, ValueError) as exc:
        msg = str(exc).lower()
        if (
            "out of memory" in msg
            or "cuda" in msg
            or "too many open files" in msg
            or "errno 24" in msg
        ):
            print(
                f"    [eval_pareto_cv] SKIPPED trial {trial_number} ({method}) "
                f"fold {fold_idx}: {exc}"
            )
            return {
                "fold": fold_idx,
                "seed": int(seed),
                "val_acc": None,
                "val_loss": None,
                "test_acc": None,
                "test_loss": None,
                "metrics": {},
                "metrics_history": [],
                "error": str(exc),
            }
        raise
    finally:
        del dm, res
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _aggregate(per_fold: list[dict]) -> dict:
    def _stats(vals: list[float]) -> tuple[float | None, float | None]:
        if not vals:
            return None, None
        return statistics.mean(vals), statistics.pstdev(vals)

    def _field_stats(key: str) -> tuple[float | None, float | None]:
        return _stats([f[key] for f in per_fold if f.get(key) is not None])

    val_acc_mean, val_acc_std = _field_stats("val_acc")
    val_loss_mean, val_loss_std = _field_stats("val_loss")
    test_acc_mean, test_acc_std = _field_stats("test_acc")
    test_loss_mean, test_loss_std = _field_stats("test_loss")

    # cfg.metrics is fixed for the whole script run, so every fold's
    # `metrics` dict has the same keys -- union them anyway in case a fold
    # was skipped (OOM) or resumed from an older run with a different
    # cfg.metrics list.
    metric_names = sorted({name for f in per_fold for name in f.get("metrics", {})})
    cv_metrics_mean: dict[str, float | None] = {}
    cv_metrics_std: dict[str, float | None] = {}
    for name in metric_names:
        vals = [
            f["metrics"][name]
            for f in per_fold
            if f.get("metrics", {}).get(name) is not None
        ]
        cv_metrics_mean[name], cv_metrics_std[name] = _stats(vals)

    return {
        "cv_val_acc_mean": val_acc_mean,
        "cv_val_acc_std": val_acc_std,
        "cv_val_loss_mean": val_loss_mean,
        "cv_val_loss_std": val_loss_std,
        "cv_test_acc_mean": test_acc_mean,
        "cv_test_acc_std": test_acc_std,
        "cv_test_loss_mean": test_loss_mean,
        "cv_test_loss_std": test_loss_std,
        "cv_metrics_mean": cv_metrics_mean,
        "cv_metrics_std": cv_metrics_std,
        "n_folds_completed": sum(1 for f in per_fold if f.get("val_acc") is not None),
    }


def _load_existing_results(path: Path) -> dict[tuple, dict]:
    if not path.exists():
        return {}
    try:
        data = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True)
    except Exception as exc:  # noqa: BLE001 - best-effort resume, never fatal
        print(f"[eval_pareto_cv] could not parse existing {path} ({exc}); starting fresh")
        return {}
    return {_params_key(entry["params"]): entry for entry in data.get("pareto_cv_results") or []}


def _write_results(path: Path, results: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(OmegaConf.to_yaml(OmegaConf.create({"pareto_cv_results": results})))


def _source_metric(trial: dict, key: str) -> float | None:
    value = trial.get(key)
    return float(value) if value is not None else None


@hydra.main(config_path="configs", config_name="eval_pareto_cv", version_base="1.3")
def main(cfg: DictConfig) -> None:
    _bump_fd_limit()

    pareto_path = str(cfg.pareto_yaml)
    trials = _load_pareto_trials(pareto_path)
    unique_configs = _dedupe_trials(trials)

    n_folds = int(cfg.dataset.n_folds)
    seeds = list(cfg.cv.seeds)
    if len(seeds) < n_folds:
        raise ValueError(
            f"cv.seeds has {len(seeds)} entries but dataset.n_folds={n_folds}; "
            "provide at least one seed per fold."
        )

    output_path = (
        Path(str(cfg.output_yaml))
        if cfg.get("output_yaml")
        else Path(cfg.log_dir) / "pareto_cv" / f"{Path(pareto_path).stem}_cv_test.yaml"
    )
    existing = _load_existing_results(output_path) if bool(cfg.cv.resume) else {}

    print("\n========================================================")
    print(" Pareto CV + Test Verification")
    print(f" Input        : {pareto_path}")
    print(f" Trials       : {len(trials)} -> {len(unique_configs)} unique configurations")
    print(f" Folds        : {n_folds}  (seeds={seeds[:n_folds]})")
    print(f" Output       : {output_path}")
    print(f" Resume       : {bool(cfg.cv.resume)} ({len(existing)} configs found on disk)")
    print("========================================================\n")

    # Reference backbone for fresh (model-aware) compression_rate/bpp accounting.
    img_size = cfg.model.get("img_size", None) or cfg.dataset.img_size
    backbone_sample = build_backbone(
        cfg.model,
        in_channels=cfg.dataset.in_channels,
        img_size=img_size,
        num_classes=cfg.dataset.num_classes,
    )

    results: list[dict] = []
    for cfg_idx, trial in enumerate(unique_configs):
        params = dict(trial["params"])
        # `.get(..., default)` only falls back when the key is *missing*, not
        # when it's explicitly `null` -- a genuine no-quantization baseline
        # entry (see _NO_QUANT_METHOD_NAMES) needs the `or` chain so it
        # displays as "none" rather than the string "None".
        method = str(params.get("method") or trial.get("method") or "none")
        trial_number = int(trial["trial_number"])
        key = _params_key(params)

        quant_cfg = _reconstruct_quant_cfg(params, cfg.quant_search_space)
        bit_stats = compute_quantization_bits(backbone_sample, quant_cfg)

        prior = existing.get(key, {})
        prior_by_fold = {
            int(f["fold"]): f
            for f in prior.get("per_fold", [])
            if f.get("val_acc") is not None
        }

        print(
            f"--- [{cfg_idx + 1}/{len(unique_configs)}] trial {trial_number} ({method}) "
            f"params={params} [from trials {trial['source_trial_numbers']}] ---"
        )

        per_fold: list[dict] = []
        for fold_idx in range(n_folds):
            if fold_idx in prior_by_fold:
                print(f"    fold {fold_idx}: already completed, skipping")
                per_fold.append(prior_by_fold[fold_idx])
                continue

            seed = seeds[fold_idx]
            print(f"    fold {fold_idx}: training with seed={seed}")
            per_fold.append(_run_fold(cfg, quant_cfg, method, trial_number, fold_idx, seed))

            # Incremental save after every fold so an interrupted sweep can resume.
            partial_entry = {
                "trial_number": trial_number,
                "source_trial_numbers": trial["source_trial_numbers"],
                "method": method,
                "params": params,
                "compression_rate": float(bit_stats["compression_rate"]),
                "bpp": float(bit_stats["bpp"]),
                "source_val_acc": _source_metric(trial, "val_acc"),
                "source_compression_rate": _source_metric(trial, "compression_rate"),
                "source_bpp": _source_metric(trial, "bpp"),
                **_aggregate(per_fold),
                "per_fold": per_fold,
            }
            _write_results(output_path, results + [partial_entry])

        entry = {
            "trial_number": trial_number,
            "source_trial_numbers": trial["source_trial_numbers"],
            "method": method,
            "params": params,
            "compression_rate": float(bit_stats["compression_rate"]),
            "bpp": float(bit_stats["bpp"]),
            "source_val_acc": _source_metric(trial, "val_acc"),
            "source_compression_rate": _source_metric(trial, "compression_rate"),
            "source_bpp": _source_metric(trial, "bpp"),
            **_aggregate(per_fold),
            "per_fold": per_fold,
        }
        results.append(entry)
        _write_results(output_path, results)

        if entry["cv_val_acc_mean"] is not None:
            print(
                f"    -> val_acc {entry['cv_val_acc_mean']:.4f} +/- {entry['cv_val_acc_std']:.4f} | "
                f"test_acc {entry['cv_test_acc_mean']:.4f} +/- {entry['cv_test_acc_std']:.4f}"
            )
        else:
            print("    -> no completed folds")
        if entry["cv_metrics_mean"]:
            metrics_str = " | ".join(
                f"{name}={mean:.4f}+/-{entry['cv_metrics_std'][name]:.4f}"
                if mean is not None
                else f"{name}=n/a"
                for name, mean in entry["cv_metrics_mean"].items()
            )
            print(f"    -> {metrics_str}")

    print("\n========================================================")
    print(f" PARETO CV+TEST SUMMARY ({len(results)} configurations)")
    print("========================================================")
    for entry in sorted(results, key=lambda e: e["compression_rate"]):
        v_s = (
            f"{entry['cv_val_acc_mean']:.4f}+/-{entry['cv_val_acc_std']:.4f}"
            if entry["cv_val_acc_mean"] is not None
            else "n/a"
        )
        t_s = (
            f"{entry['cv_test_acc_mean']:.4f}+/-{entry['cv_test_acc_std']:.4f}"
            if entry["cv_test_acc_mean"] is not None
            else "n/a"
        )
        print(
            f"Trial #{entry['trial_number']:03d} | {entry['method']:20s} | "
            f"comp={entry['compression_rate']:.2f}x | bpp={entry['bpp']:.2f} | "
            f"val_acc={v_s} | test_acc={t_s} | params={entry['params']}"
        )
    print(f"\nResults written to: {output_path}")


if __name__ == "__main__":
    main()
