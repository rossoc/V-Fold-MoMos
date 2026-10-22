"""Training entry point using PyTorch Lightning and Hydra."""

import hydra
from omegaconf import DictConfig, OmegaConf
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from coolname import generate_slug
import torch.multiprocessing
import gc

from data import ImageDataModule
from model import LitClassifier, build_backbone
from utils.init import resolve_runtime, setup_checkpoint_dir, configure_cuda_fast_path
from utils.callbacks import (
    build_callbacks,
    QuantizationCallback,
    MetricsHistoryCallback,
)

torch.multiprocessing.set_sharing_strategy("file_system")


def build_datamodule(cfg: DictConfig) -> ImageDataModule:
    """Construct the ImageDataModule for a given config.

    Only depends on `cfg.dataset.*`, `cfg.fold`, `cfg.batch_size`, `cfg.model`
    (img_size), and `cfg.accelerator` (runtime) — not `cfg.quantization`,
    `cfg.seed`, or `cfg.prefix` — so callers can build one instance per
    distinct fold and reuse it across runs that share those fields.
    """
    _, runtime_cfg = resolve_runtime(cfg.accelerator)
    img_size = cfg.model.get("img_size", None) or cfg.dataset.img_size
    return ImageDataModule(
        dataset_name=cfg.dataset.name,
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        img_size=img_size,
        val_pct=cfg.dataset.val_pct,
        test_pct=cfg.dataset.test_pct,
        n_folds=cfg.dataset.n_folds,
        fold=cfg.get("fold", None),
        kfold_seed=cfg.dataset.kfold_seed,
        runtime=runtime_cfg,
    )


def run_training(
    cfg: DictConfig, optuna_trial=None, datamodule: ImageDataModule | None = None
) -> dict:
    """Run training with PyTorch Lightning and Hydra config management.

    Args:
        cfg: DictConfig hydra configuration.
        optuna_trial: Optional Optuna trial for pruning callbacks.
        datamodule: Optional pre-built ImageDataModule to reuse (e.g. across
            trials that share the same fold). Built from `cfg` when omitted.

    Returns:
        Dict with metrics (val_acc, val_loss, test_acc, test_loss, best_model_path).
    """
    # Set seed for reproducibility
    run_name = generate_slug()
    L.seed_everything(cfg.seed, workers=True)

    accelerator, _ = resolve_runtime(cfg.accelerator)

    # Whether the model opts into torch.compile; needed up-front because it
    # gates both the CUDA conv autotuner and the deterministic baseline.
    compile_model = bool(cfg.model.get("compile", False))

    # Enable CUDA fast-path (TF32 + high matmul precision). cudnn.benchmark is
    # nondeterministic, so only enable it when compiling (deterministic is off
    # in that case). No-op when CUDA is unavailable.
    configure_cuda_fast_path(enable_benchmark=compile_model)

    # Mixed precision: auto-pick by device when unset. bf16-mixed needs Ampere+;
    # override with precision=16-mixed on older GPUs (V100/T4).
    precision = cfg.get("precision", None)
    if precision is None:
        precision = "bf16-mixed" if accelerator == "cuda" else "32-true"

    # Effective input size: a model may require a fixed resolution (e.g. timm
    # TinyViT expects 224), overriding the dataset's native size. The datamodule
    # resizes to this size, so the backbone receives the geometry it expects.
    img_size = cfg.model.get("img_size", None) or cfg.dataset.img_size

    if datamodule is None:
        datamodule = build_datamodule(cfg)

    # Setup checkpoint directory
    checkpoint_dir, unique_run_name, init_ckpt_path = setup_checkpoint_dir(
        log_dir=cfg.log_dir,
        dataset_name=cfg.dataset.name,
        prefix=cfg.prefix,
        run_name=run_name,
        model_name=cfg.model.name,
    )

    # Build the backbone (mlp / timm TinyViT / ...) and wrap it
    backbone = build_backbone(
        cfg.model,
        in_channels=cfg.dataset.in_channels,
        img_size=img_size,
        num_classes=cfg.dataset.num_classes,
    )
    model = LitClassifier(
        backbone=backbone,
        num_classes=cfg.dataset.num_classes,
        learning_rate=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
        epochs=cfg.epochs,
        optimizer=str(cfg.model.get("optimizer", "adamw")),
        momentum=float(cfg.model.get("momentum", 0.9)),
        save_init_path=init_ckpt_path if cfg.get("save_init_ckpt", True) else None,
        compile_model=compile_model,
    )

    # Setup callbacks via factory
    has_logger = cfg.get("wandb", {}).get("enabled", False)
    callbacks = build_callbacks(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        unique_run_name=unique_run_name,
        has_logger=has_logger,
    )

    if optuna_trial is not None:
        study = optuna_trial.study
        is_multi_objective = (
            len(study.directions) > 1
            or getattr(study, "_is_multi_objective", lambda: False)()
        )

        pruner_cfg = cfg.get("optuna", {}).get("pruner", {})
        pruner_enabled = bool(pruner_cfg.get("enabled", True))

        if is_multi_objective:
            if pruner_enabled:
                from utils.optuna_callback import OptunaMultiObjectiveEpochPruning

                callbacks.append(
                    OptunaMultiObjectiveEpochPruning(
                        optuna_trial,
                        monitor=str(pruner_cfg.get("monitor", "val/acc")),
                        n_startup_trials=int(pruner_cfg.get("n_startup_trials", 5)),
                        n_warmup_steps=int(pruner_cfg.get("n_warmup_steps", 3)),
                        percentile=float(pruner_cfg.get("percentile", 50.0)),
                        min_history=int(pruner_cfg.get("min_history", 3)),
                        stratify_by=pruner_cfg.get("stratify_by", None),
                        stratify_rtol=float(pruner_cfg.get("stratify_rtol", 0.05)),
                    )
                )
            # else: pruning explicitly disabled via config — attach nothing.
        else:
            from utils.optuna_callback import OptunaTrainEpochPruning

            callbacks.append(OptunaTrainEpochPruning(optuna_trial, monitor="val/acc"))

    # Extract checkpoint callback for later reference
    checkpoint_callback = next(
        (cb for cb in callbacks if isinstance(cb, L.pytorch.callbacks.ModelCheckpoint)),
        None,
    )
    quant_callback = next(
        (cb for cb in callbacks if isinstance(cb, QuantizationCallback)), None
    )
    # Per-epoch metrics/* trajectory (see MetricsHistoryCallback); present
    # whenever cfg.metrics is set, independent of quant_callback.
    metrics_history_callback = next(
        (cb for cb in callbacks if isinstance(cb, MetricsHistoryCallback)), None
    )

    # Setup W&B logger (if enabled)
    # When disabled, use False to prevent Lightning from creating default lightning_logs/ directory
    if cfg.get("wandb", {}).get("enabled", False):
        wandb_cfg = cfg.wandb
        wandb_run_name = wandb_cfg.get("name") or unique_run_name
        logger = WandbLogger(
            project=wandb_cfg.get("project", "momos-reproduction"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_run_name,
            group=wandb_cfg.get("group", None),
            tags=wandb_cfg.get("tags", []),
            log_model=wandb_cfg.get("log_model", False),
            save_dir=cfg.log_dir,
            reinit=True,
        )
        # Log full Hydra config to W&B
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    else:
        # Explicitly disable logging to prevent lightning_logs/ directory creation
        logger = False

    # Create Trainer
    trainer = L.Trainer(
        max_epochs=cfg.epochs,
        accelerator=accelerator,
        devices=cfg.devices,
        precision=precision,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=50,
        enable_progress_bar=True,
        enable_model_summary=True,
        # torch.compile (Inductor) can conflict with deterministic algorithms;
        # keep the deterministic baseline unless the model opts into compiling.
        deterministic=not compile_model,
    )

    try:
        # Train
        trainer.fit(model, datamodule=datamodule)

        best_val_acc = 0.0
        best_val_loss = float("inf")

        # Report best-checkpoint metrics
        if checkpoint_callback and checkpoint_callback.best_model_path:
            print(f"\nLoading best checkpoint: {checkpoint_callback.best_model_path}")
            best_val_loss = float(checkpoint_callback.best_model_score or float("inf"))
            print(f"Best validation loss: {best_val_loss:.4f}")
            val_results = trainer.validate(
                model,
                datamodule=datamodule,
                ckpt_path=checkpoint_callback.best_model_path,
            )
            if val_results:
                best_val_acc = float(val_results[0].get("val/acc", 0.0))
                print(f"Best validation accuracy: {best_val_acc:.4f}")
        else:
            best_val_acc = float(trainer.callback_metrics.get("val/acc", 0.0))
            best_val_loss = float(
                trainer.callback_metrics.get("val/loss", float("inf"))
            )

        # Final test evaluation
        test_results = None
        if cfg.get("evaluate_test", True):
            print("\nEvaluating on test set...")
            test_results = trainer.test(model, datamodule=datamodule)

        test_acc = float(test_results[0].get("test/acc", 0.0)) if test_results else None
        test_loss = (
            float(test_results[0].get("test/loss", 0.0)) if test_results else None
        )

        metrics = {}
        metrics_history = []
        if metrics_history_callback:
            metrics = {
                k: v
                for k, v in metrics_history_callback.history[-1].items()
                if k != "epoch"
            }

            metrics_history = metrics_history_callback.history

        return {
            "val_acc": best_val_acc,
            "val_loss": best_val_loss,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "metrics": metrics,
            "metrics_history": metrics_history,
            "best_model_path": checkpoint_callback.best_model_path
            if checkpoint_callback
            else None,
        }
    finally:
        if logger:
            try:
                import wandb

                if wandb.run is not None:
                    wandb.finish()
            except Exception:
                pass
        try:
            del trainer
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    """Run training with PyTorch Lightning and Hydra config management."""
    run_training(cfg)


if __name__ == "__main__":
    main()
