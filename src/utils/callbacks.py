"""Lightning callbacks for quantization-aware training and MoMos projection."""

import math
import os

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf

from quantizers import quantize_qat, k_from_capacity, quantize
from quantizers.block_utils import iter_trainable_params
from quantizers.soft_momos import (
    draw_mosaic,
    project_fixed_mosaic,
    soft_momos_penalty,
)
from utils.metrics import compute_metrics, resolve_block_shape


# Keys returned by ``quantize`` that are not scalar metrics to log under quant/.
_QUANT_LOG_SKIP = {
    "motif_counts",
    "method",
    "iota",
    "bucket_sizes",
    "K_primary",
    "s_prime",
    "k_per_bucket",
    "_motifs",
    "_nearest",
    "_layer_specs",
}


def _hierarchical_bit_estimate(n, k, s, s_prime, k_per_bucket, q):
    """Bit cost of the {Z, iota, M'} description (rounded up per integer index).

    motifs: k * s * q
    iota:   s' * k_per_bucket * ceil(log2 k)
    mosaic: ceil(n / s) ... wait — n/(s*s') big-block indices, each ceil(log2 k')
            with k' = k_per_bucket ** s', so ceil(s' * log2 k_per_bucket) bits.
    """
    motif_bits = k * s * q
    idx_bits = math.ceil(math.log2(max(2, k)))
    iota_bits = s_prime * k_per_bucket * idx_bits
    n_big = math.ceil(n / (s * s_prime))
    big_idx_bits = math.ceil(s_prime * math.log2(max(2, k_per_bucket)))
    mosaic_bits = n_big * big_idx_bits
    total = motif_bits + iota_bits + mosaic_bits
    return {
        "motif_bits": motif_bits,
        "iota_bits": iota_bits,
        "mosaic_bits": mosaic_bits,
        "total_bits": total,
    }


def _fold_bit_estimate(n, k, s, q):
    """Bit cost estimate for fold_momos: same shape as plain momos2d, since the
    fold/k-means machinery only refines the shared codebook -- it does not add a
    separate per-fold storage term (unlike hierarchical's iota).

    motifs: k * s * q
    mosaic: ceil(n / s) block indices, each ceil(log2 k) bits.

    ``k`` here is an *upper bound* (min(primary.k, s' * k_per_bucket)): the real
    k-means-refined codebook size is only known once the fold pass actually runs.
    """
    motif_bits = k * s * q
    idx_bits = math.ceil(math.log2(max(2, k)))
    n_blocks = math.ceil(n / s)
    mosaic_bits = n_blocks * idx_bits
    total = motif_bits + mosaic_bits
    return {
        "motif_bits": motif_bits,
        "mosaic_bits": mosaic_bits,
        "total_bits": total,
    }


def _log_bits_hyperparams(pl_module, est, baseline):
    """Log a ``*_bit_estimate`` dict (``<name>_bits`` keys) as ``bits/<name>`` hparams."""
    logger = getattr(pl_module, "logger", None)
    if logger is None or not hasattr(logger, "log_hyperparams"):
        return
    params = {f"bits/{key.removesuffix('_bits')}": v for key, v in est.items()}
    params["bits/dense"] = baseline
    params["bits/ratio"] = est["total_bits"] / baseline
    try:
        logger.log_hyperparams(params)
    except Exception:
        pass


class QuantizationCallback(L.Callback):
    """Callback that handles QAT setup and MoMos projection during training.

    This callback integrates quantization into the Lightning training lifecycle:
    - QAT: Attaches fake-quant parametrizations before training starts
    - MoMos: Applies motif-based projection at the end of each training epoch

    Args:
        quant_cfg: Quantization configuration dictionary. Must contain ``method`` key
            with value ``"qat"`` or ``"momos"``. Other keys depend on the method:
            - QAT: ``q`` (bit-width), ``exclude_layers`` (optional)
            - MoMos: ``s`` (block size), ``k`` (motif count), ``force_zero``, etc.
        metric_names: Optional list of metric names to compute and log after each
            epoch. Uses ``utils.metrics.compute_metrics``. Examples: ``sparsity``,
            ``l2``, ``gzip``, ``bz2``, ``lzma``, ``bdm``.
        compression_binarized: Passed to metric computation for compression payloads.
    """

    def __init__(
        self,
        quant_cfg: dict,
        metric_names: list[str] | None = None,
        compression_binarized: bool = False,
    ):
        super().__init__()
        self.quant_cfg = quant_cfg
        self.metric_names = metric_names or []
        self.compression_binarized = compression_binarized
        self.method = str(quant_cfg.get("method", "")).lower()
        # Stats from the most recent MoMos projection, logged in
        # on_validation_epoch_end (projection runs in on_validation_epoch_start).
        self._last_quant_stats: dict | None = None
        # soft_momos state: the current epoch's mosaic (and its bincount, which
        # only depends on M), the penalty weight in effect, whether the final
        # k-means++ snap has run, and the raw weights stashed while validating
        # on the projection.
        self.M = None
        self.finalized = False
        self.current_lam = float(quant_cfg.get("lam", 0.0))
        self._counts = None
        self._raw_params: list | None = None
        self._aux_fn = None

    def on_fit_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Set up QAT parametrizations before training begins."""
        if self.method == "soft_momos":
            self._setup_soft_momos(pl_module)
            return

        if self.method == "hierarchical_momos2d":
            self._log_hierarchical_bit_estimate(pl_module)
            return

        if self.method == "fold_momos":
            self._log_fold_bit_estimate(pl_module)
            return

        if self.method != "qat":
            return

        model = pl_module.model
        stats = quantize_qat(model, self.quant_cfg)

        qat_enabled = stats.get("qat_enabled", False)
        attached = stats.get("attached_modules", 0)
        updated = stats.get("updated_modules", 0)

        if qat_enabled:
            print(
                f"QAT enabled: attached={attached}, updated={updated}, "
                f"bitwidth={stats.get('q_bits', '?')}"
            )
        else:
            print(
                f"QAT disabled (bitwidth >= 32): disabled_modules={stats.get('disabled_modules', 0)}"
            )

    def _setup_soft_momos(self, pl_module: L.LightningModule):
        """Resolve k, register the penalty (once) and log the final bit estimate."""
        model = pl_module.model
        cfg = self.quant_cfg
        s = int(cfg["s"])
        if cfg.get("k") is None:
            cfg["k"] = k_from_capacity(model, s, cfg["capacity"])
        rows, cols = int(cfg["rows"]), int(cfg["cols"])
        k, force_zero = int(cfg["k"]), bool(cfg.get("force_zero", True))

        if self._aux_fn is None:

            def penalty(m):
                return self.current_lam * soft_momos_penalty(
                    m, self.M, rows, cols, k, force_zero,
                    counts=self._counts, with_stats=False,
                )[0]

            self._aux_fn = penalty
        if self._aux_fn not in pl_module.aux_losses:
            pl_module.aux_losses.append(self._aux_fn)

        q = int(cfg.get("q_bits", 32))
        n = sum(p.numel() for p in iter_trainable_params(model))
        est = _fold_bit_estimate(n, int(cfg["final_k"]), s, q)
        baseline = n * q
        print(
            f"Soft MoMos bit estimate (q={q}, n={n}, k={k}, final_k={cfg['final_k']}, "
            f"s={s}): motif={est['motif_bits']}, mosaic={est['mosaic_bits']}, "
            f"total={est['total_bits']} "
            f"({est['total_bits'] / baseline:.4%} of dense {baseline} bits)"
        )
        _log_bits_hyperparams(pl_module, est, baseline)

    def on_train_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """soft_momos: draw a new mosaic from the current weights, update lam."""
        if self.method != "soft_momos":
            return
        cfg = self.quant_cfg
        self.M = draw_mosaic(
            pl_module.model,
            cfg["rows"],
            cfg["cols"],
            cfg["k"],
            cfg.get("force_zero", True),
            cfg.get("chunk_size"),
        )[0]
        self._counts = torch.bincount(self.M, minlength=int(cfg["k"]))
        warmup = float(cfg.get("lam_warmup_fraction", 0.0)) * (trainer.max_epochs or 1)
        ramp = min(1.0, trainer.current_epoch / warmup) if warmup > 0 else 1.0
        self.current_lam = float(cfg["lam"]) * ramp

    def _soft_validation_start(self, pl_module: L.LightningModule):
        """Record penalty stats on the raw weights, then optionally project to Z[M]."""
        if self.M is None:
            return
        cfg = self.quant_cfg
        rows, cols, k = int(cfg["rows"]), int(cfg["cols"]), int(cfg["k"])
        force_zero = bool(cfg.get("force_zero", True))
        with torch.no_grad():
            _, stats = soft_momos_penalty(
                pl_module.model, self.M, rows, cols, k, force_zero, counts=self._counts
            )
        stats = {name: float(v) for name, v in stats.items()}
        stats["lam"] = float(self.current_lam)
        if cfg.get("validate_projected", True):
            self._raw_params = [
                p.detach().clone() for p in iter_trainable_params(pl_module.model)
            ]
            stats.update(
                project_fixed_mosaic(pl_module.model, self.M, rows, cols, k, force_zero)
            )
        self._last_quant_stats = stats

    def _restore_raw_params(self, pl_module: L.LightningModule):
        if self._raw_params is None:
            return
        with torch.no_grad():
            for p, raw in zip(
                iter_trainable_params(pl_module.model), self._raw_params
            ):
                p.copy_(raw)
        self._raw_params = None

    def _finalize_soft_momos(self, trainer, pl_module, fn: TrainerFn):
        """Snap the model to final_k motifs once, when entering validate/test."""
        if self.method != "soft_momos" or self.finalized or trainer.state.fn != fn:
            return
        stats = quantize(pl_module.model, self.quant_cfg)
        self.finalized = True
        pl_module.log("quant/final_distortion", stats["distortion"])
        pl_module.log("quant/final_num_motifs_used", float(stats["num_motifs_used"]))
        print(
            f"Soft MoMos finalized: distortion={stats['distortion']:.4f}, "
            f"motifs used={stats['num_motifs_used']}"
        )
        dirpath = getattr(trainer.checkpoint_callback, "dirpath", None)
        if dirpath:
            self._save_final_motifs(pl_module, stats, os.path.join(dirpath, "final_motifs.pt"))

    def _save_final_motifs(self, pl_module, stats, path):
        """Save the finalized codebook/mosaic in ``extract_motifs.py``'s format
        (unused centroids dropped so every motif id is referenced)."""
        counts = stats["motif_counts"]
        used = counts > 0
        remap = used.long().cumsum(0) - 1
        names = {id(p): n for n, p in pl_module.model.named_parameters()}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(
            {
                "motifs": stats["_motifs"][used].cpu(),
                "assignments": remap[stats["_nearest"]].cpu(),
                "rows": int(self.quant_cfg["rows"]),
                "cols": int(self.quant_cfg["cols"]),
                "num_blocks": int(stats["_nearest"].numel()),
                "num_motifs": int(used.sum()),
                "layer_specs": [
                    (names[id(p)], nb, tuple(shape))
                    for p, nb, _, shape in stats["_layer_specs"]
                ],
            },
            path,
        )

    def on_validation_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._finalize_soft_momos(trainer, pl_module, TrainerFn.VALIDATING)

    def on_test_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        self._finalize_soft_momos(trainer, pl_module, TrainerFn.TESTING)

    def state_dict(self):
        if self.method != "soft_momos":
            return {}
        return {"M": self.M, "finalized": self.finalized}

    def load_state_dict(self, state_dict):
        if self.method == "soft_momos":
            self.M = state_dict.get("M")
            self.finalized = bool(state_dict.get("finalized", False))

    def _log_hierarchical_bit_estimate(self, pl_module: L.LightningModule):
        model = pl_module.model
        primary = self.quant_cfg["primary"]
        secondary = self.quant_cfg["secondary"]

        s = int(primary["s"])
        s_prime = int(secondary["s"])
        q = int(self.quant_cfg.get("q_bits", 32))
        c = float(secondary["capacity"])

        if primary.get("k") is None and primary.get("capacity") is not None:
            primary["k"] = k_from_capacity(model, s, primary["capacity"])
        k = int(primary["k"])

        n = sum(p.numel() for p in iter_trainable_params(model))
        n_big = math.ceil(n / (s * s_prime))
        # Bucket size is relative to K_primary (capacity * K), matching the
        # quantizer in momos_hierarchy.py; capped at the big-block count.
        k_per_bucket = max(1, min(int(round(k * c)), n_big))

        est = _hierarchical_bit_estimate(n, k, s, s_prime, k_per_bucket, q)
        baseline = n * q
        print(
            f"Hierarchical MoMos2D bit estimate (q={q}, n={n}, k={k}, s={s}, "
            f"s'={s_prime}, k_per_bucket={k_per_bucket}): "
            f"motif={est['motif_bits']}, iota={est['iota_bits']}, "
            f"mosaic={est['mosaic_bits']}, total={est['total_bits']} "
            f"({est['total_bits'] / baseline:.4%} of dense {baseline} bits)"
        )

        _log_bits_hyperparams(pl_module, est, baseline)

    def _log_fold_bit_estimate(self, pl_module: L.LightningModule):
        model = pl_module.model
        primary = self.quant_cfg["primary"]
        secondary = self.quant_cfg["secondary"]

        s = int(primary["s"])
        s_prime = int(secondary["s"])
        q = int(self.quant_cfg.get("q_bits", 32))
        c = float(secondary.get("capacity") or 1.0)

        if primary.get("k") is None and primary.get("capacity") is not None:
            primary["k"] = k_from_capacity(model, s, primary["capacity"])
        k_primary = int(primary["k"])

        n = sum(p.numel() for p in iter_trainable_params(model))
        n_blocks = math.ceil(n / s)
        k_per_bucket = max(1, min(int(round(k_primary * c)), n_blocks))
        # Upper bound: the pooled k-means training set has at most
        # s_prime * k_per_bucket points (with repeats), and k-means never
        # returns more centroids than that or than k_primary.
        k_est = max(1, min(k_primary, s_prime * k_per_bucket))

        est = _fold_bit_estimate(n, k_est, s, q)
        baseline = n * q
        print(
            f"Fold MoMos bit estimate (q={q}, n={n}, k<={k_est}, s={s}, "
            f"s'={s_prime}, k_per_bucket={k_per_bucket}): "
            f"motif={est['motif_bits']}, mosaic={est['mosaic_bits']}, "
            f"total={est['total_bits']} "
            f"({est['total_bits'] / baseline:.4%} of dense {baseline} bits)"
        )

        _log_bits_hyperparams(pl_module, est, baseline)

    def on_validation_epoch_start(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Apply MoMos projection before validation runs.

        Lightning nests the validation loop inside the training epoch, so this
        hook fires after the epoch's training batches but before the validation
        batches (and before ``on_train_epoch_end``). Projecting here means
        validation, the post-validation metrics, and the ``ModelCheckpoint``
        saved during validation all reflect the quantized model.
        """
        if self.method not in [
            "momos",
            "momos2d",
            "static_momos2d",
            "hierarchical_momos2d",
            "fold_momos",
            "soft_momos",
        ]:
            return

        # Skip the pre-training sanity-check validation and any standalone
        # trainer.validate()/test() pass at the end of main: project only during
        # the fit-time validation that follows a real training epoch.
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return

        if self.method == "soft_momos":
            self._soft_validation_start(pl_module)
            return

        model = pl_module.model

        if self.method in ("hierarchical_momos2d", "fold_momos"):
            switch_fraction = float(self.quant_cfg.get("switch_fraction", 0.5))
            max_epochs = max(1, int(trainer.max_epochs or 1))
            progress = trainer.current_epoch / max_epochs
            self.quant_cfg["apply_secondary"] = progress >= switch_fraction

            # Only primary uses k_from_capacity: secondary `capacity` is interpreted
            # relative to K_primary inside the hierarchy/fold module.
            primary = self.quant_cfg["primary"]
            if primary.get("k") is None and primary.get("capacity") is not None:
                primary["k"] = k_from_capacity(
                    model, primary["s"], primary["capacity"]
                )
        else:
            # Resolve k from capacity if needed
            if (
                self.quant_cfg.get("k") is None
                and self.quant_cfg.get("capacity") is not None
            ):
                self.quant_cfg["k"] = k_from_capacity(
                    model, self.quant_cfg["s"], self.quant_cfg["capacity"]
                )

        stats = quantize(model, self.quant_cfg)

        # Stash for logging in on_validation_epoch_end; logging is cleaner from
        # the epoch-end hook than from epoch-start.
        self._last_quant_stats = stats

        report = ""
        for k, v in stats.items():
            if k in _QUANT_LOG_SKIP:
                continue
            report += f"{k}={v:.4f}"
        print("MoMos applied:", report)

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        """Log MoMos projection stats and compute quantization metrics."""
        # Log the stats from this epoch's projection (set in
        # on_validation_epoch_start) so they land on the same step as val metrics.
        if self._last_quant_stats is not None:
            for k, v in self._last_quant_stats.items():
                if k in _QUANT_LOG_SKIP:
                    continue
                pl_module.log("quant/" + k, v, on_epoch=True, prog_bar=False)
            self._last_quant_stats = None

        # Metrics are computed on the weights as validated (the projection for
        # soft_momos); the raw weights are restored afterwards. Note that
        # MetricsHistoryCallback runs after this hook and so sees the raw weights.
        if self.metric_names:
            self._log_metrics(pl_module)
        self._restore_raw_params(pl_module)

    def on_exception(self, trainer, pl_module, exception):
        self._restore_raw_params(pl_module)

    def _log_metrics(self, pl_module: L.LightningModule):
        try:
            rows, cols = resolve_block_shape(self.method, self.quant_cfg)
            metrics = compute_metrics(
                pl_module.model,
                self.metric_names,
                self.compression_binarized,
                rows=rows,
                cols=cols,
            )
            for name, value in metrics.items():
                if value is not None:
                    pl_module.log(
                        f"metrics/{name}", value, on_epoch=True, prog_bar=False
                    )
        except Exception as e:
            print(f"Warning: Failed to compute metrics: {e}")


class MetricsHistoryCallback(L.Callback):
    """Records `utils.metrics.compute_metrics` results once per training epoch.

    `QuantizationCallback.on_validation_epoch_end` already logs the *latest*
    `metrics/*` values to `pl_module.log` (visible in wandb / a single
    snapshot in `trainer.callback_metrics`), but two things limit that for
    inspecting how a metric evolves over training: it only runs when
    `cfg.quantization.enabled` is true, and nothing keeps the full per-epoch
    trajectory in memory for a caller to read back programmatically. This
    callback is decoupled from `quantization.enabled` (so it also covers
    unquantized/baseline runs) and appends every epoch's values to
    `self.history`, which `run_training` surfaces as `metrics_history`.

    Args:
        metric_names: Metric names to compute each epoch (see
            `utils.metrics.compute_metrics`'s registry).
        compression_binarized: Passed through to `compute_metrics`.
        quant_callback: The run's `QuantizationCallback`, if any -- reused
            only so the block shape (`resolve_block_shape`) matches exactly
            what quantization is actually using (e.g. a hierarchical
            method's live primary/secondary switch). `rows=cols=1` when
            `None` (no quantization, or a method with no block structure).
    """

    def __init__(
        self,
        metric_names: list[str] | None = None,
        compression_binarized: bool = False,
        quant_callback: "QuantizationCallback | None" = None,
    ):
        super().__init__()
        self.metric_names = metric_names or []
        self.compression_binarized = compression_binarized
        self.quant_callback = quant_callback
        self.history: list[dict] = []

    def on_validation_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        if not self.metric_names:
            return
        # Skip the pre-training sanity-check validation and any standalone
        # trainer.validate()/test() pass after fit (e.g. run_training's
        # best-checkpoint reload) -- only real training epochs belong in the
        # trajectory.
        if trainer.sanity_checking or trainer.state.fn != TrainerFn.FITTING:
            return

        if self.quant_callback is not None:
            rows, cols = resolve_block_shape(
                self.quant_callback.method, self.quant_callback.quant_cfg
            )
        else:
            rows, cols = 1, 1

        try:
            metrics = compute_metrics(
                pl_module.model,
                self.metric_names,
                self.compression_binarized,
                rows=rows,
                cols=cols,
            )
        except Exception as e:
            print(f"Warning: Failed to compute metrics history: {e}")
            return

        self.history.append({"epoch": int(trainer.current_epoch), **metrics})

        # Log to wandb when no QuantizationCallback is present (it handles
        # logging for quantized runs via its own on_validation_epoch_end).
        if self.quant_callback is None:
            for name, value in metrics.items():
                if value is not None:
                    pl_module.log(
                        f"metrics/{name}", value, on_epoch=True, prog_bar=False
                    )


def _best_checkpoint(checkpoint_dir: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="best",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        save_last=False,
    )


def _periodic_checkpoint(checkpoint_dir: str) -> ModelCheckpoint:
    return ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="epoch-{epoch:02d}",
        every_n_epochs=20,
        save_top_k=-1,
    )


def _early_stopping(patience: int | None) -> EarlyStopping | None:
    if patience is None or patience <= 0:
        return None
    return EarlyStopping(
        monitor="val/loss", mode="min", patience=patience, verbose=True
    )


def _build_quantization_callback(cfg, quant_cfg) -> QuantizationCallback:
    """Convert a per-method `quantization` config into a QuantizationCallback.

    The YAML file selected by Hydra's `quantization=<method>` group already
    contains exactly the keys the method consumes, so we forward the dict
    as-is (minus the config-only `enabled` flag).
    """
    full = OmegaConf.to_container(quant_cfg, resolve=True)
    assert isinstance(full, dict)
    full.pop("enabled", None)

    method = full.get("method")
    if method in ("momos2d", "static_momos2d", "soft_momos"):
        full["s"] = int(full["rows"]) * int(full["cols"])

    if method in ("momos", "momos2d", "static_momos2d", "soft_momos"):
        if full.get("k") is None and full.get("capacity") is None:
            raise ValueError(f"{method} requires either k or capacity in config")

    if method in ("hierarchical_momos2d", "fold_momos"):
        for sub_key in ("primary", "secondary"):
            sub = full.get(sub_key)
            if not isinstance(sub, dict):
                raise ValueError(
                    f"{method} requires a `{sub_key}` block in the quantization config"
                )
            sub["s"] = int(sub["rows"]) * int(sub["cols"])
            if sub.get("k") is None and sub.get("capacity") is None:
                raise ValueError(
                    f"{method}.{sub_key} requires either k or capacity in config"
                )

    return QuantizationCallback(
        quant_cfg=full,
        metric_names=cfg.get("metrics", []),
        compression_binarized=cfg.get("all_compression_metrics_binarized", False),
    )


def build_callbacks(
    cfg,
    checkpoint_dir: str,
    unique_run_name: str,
    has_logger: bool = True,
) -> list[L.Callback]:
    """Build the complete list of callbacks for training.

    Includes standard callbacks (checkpointing, early stopping, LR monitoring)
    plus a quantization callback if `cfg.quantization.enabled` is true. The
    per-method quantization schema lives in `src/configs/quantization/`.
    """
    callbacks: list[L.Callback] = []

    if has_logger:
        callbacks.append(LearningRateMonitor(logging_interval="epoch"))

    callbacks.append(_best_checkpoint(checkpoint_dir))

    if cfg.get("periodic_checkpoint"):
        callbacks.append(_periodic_checkpoint(checkpoint_dir))

    if (es := _early_stopping(cfg.get("patience"))) is not None:
        callbacks.append(es)

    quant_cfg = cfg.get("quantization", {})
    quant_callback = None
    if quant_cfg.get("enabled", False):
        quant_callback = _build_quantization_callback(cfg, quant_cfg)
        callbacks.append(quant_callback)

    # Built whenever cfg.metrics is set, regardless of quantization.enabled,
    # so unquantized/baseline runs get a metrics trajectory too -- see
    # MetricsHistoryCallback's docstring.
    if cfg.get("metrics"):
        callbacks.append(
            MetricsHistoryCallback(
                metric_names=cfg.get("metrics", []),
                compression_binarized=cfg.get("all_compression_metrics_binarized", False),
                quant_callback=quant_callback,
            )
        )

    return callbacks
