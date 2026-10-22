"""Optuna pruning callbacks that report after each training epoch.

The default ``PyTorchLightningPruningCallback`` from ``optuna-integration``
hooks into ``on_validation_end``. ``LitMamba`` has no ``validation_step`` —
its ``val/acc`` is logged from inside ``on_train_epoch_end`` — so these
custom callbacks report metrics on the same hook the model uses.

Two callbacks are provided:

- ``OptunaTrainEpochPruning``: uses Optuna's native ``trial.report()`` /
  ``trial.should_prune()`` + ``optuna.pruners.*``. Only valid for
  single-objective studies (used by ``tune_hypernet.py``).
- ``OptunaMultiObjectiveEpochPruning``: a manual percentile-based pruning
  mechanism for multi-objective studies (used by ``tune_quantization.py``),
  where Optuna's native report/should_prune API is unsupported. It stores
  per-epoch metric values as trial user attributes and compares the current
  trial's value against a percentile of other trials' values recorded at
  the same epoch — the same comparison ``MedianPruner`` makes internally,
  just built on APIs that aren't restricted to single-objective studies.
"""

from __future__ import annotations

import lightning as L
import optuna


class OptunaTrainEpochPruning(L.Callback):
    """Report ``monitor`` to an Optuna trial after every training epoch.

    Raises ``optuna.TrialPruned`` if the trial should be pruned.

    Only active for single-objective studies; native Optuna
    report/should_prune is unsupported for multi-objective studies, so this
    callback is a no-op there. See ``OptunaMultiObjectiveEpochPruning`` for
    the multi-objective equivalent.
    """

    def __init__(self, trial: optuna.Trial, monitor: str = "val/acc"):
        super().__init__()
        self.trial = trial
        self.monitor = monitor

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        # Optuna trial reporting and pruning are only supported for single-objective studies
        study = self.trial.study
        if len(study.directions) > 1 or getattr(study, "_is_multi_objective", lambda: False)():
            return
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.item() if hasattr(metric, "item") else metric)
        epoch = int(trainer.current_epoch)
        self.trial.report(value, step=epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} ({self.monitor}={value:.4f})"
            )


def _percentile(values: list[float], pct: float) -> float:
    """Linear-interpolation percentile (no numpy dependency needed here)."""
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    if pct <= 0:
        return ordered[0]
    if pct >= 100:
        return ordered[-1]
    rank = (pct / 100.0) * (len(ordered) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(ordered) - 1)
    frac = rank - lo
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


class OptunaMultiObjectiveEpochPruning(L.Callback):
    """Manual epoch-level pruning for multi-objective Optuna studies.

    Optuna's ``trial.report()``/``trial.should_prune()`` and built-in pruners
    (``MedianPruner``, etc.) are unsupported when ``len(study.directions) >
    1``. This callback reimplements a ``MedianPruner``-style comparison on
    top of the unrestricted ``set_user_attr``/``get_trials`` APIs:

    1. After every training epoch, record ``value`` under a per-epoch
       user_attr key on the current trial.
    2. Once at least ``n_startup_trials`` historical trials exist (state
       ``COMPLETE`` or ``PRUNED``) and the current epoch is >=
       ``n_warmup_steps``, gather the same per-epoch key from those trials
       and compute the ``percentile``-th percentile of their values.
    3. If the current trial's value is below that percentile, raise
       ``optuna.TrialPruned`` directly (bypassing ``should_prune()``, which
       is unavailable here).

    Stratification: in a multi-objective accuracy-vs-compression study a
    high-compression trial is *expected* to have lower accuracy than a
    low-compression one, so comparing every trial against the study-wide
    median prunes the whole high-compression side of the front (observed:
    88 of 102 soft_momos trials pruned, almost all with block size >= 4).
    With ``stratify_by`` set to a user_attr name (e.g. ``"compression_rate"``,
    which ``tune_quantization.py`` records before training starts), the
    history is restricted to trials whose attribute matches the current
    trial's within ``stratify_rtol`` (relative), i.e. trials at the same
    point on the compression axis.

    Parameters
    ----------
    trial:
        The active Optuna trial.
    monitor:
        Metric name to read from ``trainer.callback_metrics``.
    n_startup_trials:
        Minimum number of historical trials (COMPLETE/PRUNED) required
        before pruning decisions are made at all.
    n_warmup_steps:
        Minimum epoch (0-indexed) before pruning decisions are made for a
        given trial.
    percentile:
        Percentile (0-100) of historical values at the same epoch used as
        the pruning threshold. 50 = median (default), matching
        ``MedianPruner`` semantics.
    min_history:
        Minimum number of *other* trials that must have a recorded value at
        this exact epoch for the comparison to be considered meaningful
        (separate from ``n_startup_trials``, which only gates on total
        trial count regardless of per-epoch coverage).
    key_prefix:
        Prefix used for the per-epoch user_attr key, so multiple monitors
        or callback instances don't collide.
    stratify_by:
        Optional user_attr name; when set, only historical trials whose
        value of this attribute is within ``stratify_rtol`` (relative) of
        the current trial's are used for the comparison. ``None`` keeps the
        study-wide comparison.
    stratify_rtol:
        Relative tolerance for ``stratify_by`` matching.
    """

    def __init__(
        self,
        trial: optuna.Trial,
        monitor: str = "val/acc",
        n_startup_trials: int = 5,
        n_warmup_steps: int = 3,
        percentile: float = 50.0,
        min_history: int = 3,
        key_prefix: str = "epoch",
        stratify_by: str | None = None,
        stratify_rtol: float = 0.05,
    ):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.n_startup_trials = int(n_startup_trials)
        self.n_warmup_steps = int(n_warmup_steps)
        self.percentile = float(percentile)
        self.min_history = int(min_history)
        self.key_prefix = key_prefix
        self.stratify_by = stratify_by or None
        self.stratify_rtol = float(stratify_rtol)

    def _same_stratum(self, other: optuna.trial.FrozenTrial) -> bool:
        """True if ``other`` belongs to the current trial's comparison stratum."""
        if self.stratify_by is None:
            return True
        mine = self.trial.user_attrs.get(self.stratify_by)
        theirs = other.user_attrs.get(self.stratify_by)
        if mine is None or theirs is None:
            return False
        try:
            mine_f, theirs_f = float(mine), float(theirs)
        except (TypeError, ValueError):
            return mine == theirs
        return abs(mine_f - theirs_f) <= self.stratify_rtol * max(abs(mine_f), 1e-12)

    def _epoch_key(self, epoch: int) -> str:
        return f"{self.key_prefix}_{epoch}_{self.monitor.replace('/', '_')}"

    def on_train_epoch_end(self, trainer: L.Trainer, pl_module: L.LightningModule):
        metric = trainer.callback_metrics.get(self.monitor)
        if metric is None:
            return
        value = float(metric.item() if hasattr(metric, "item") else metric)
        epoch = int(trainer.current_epoch)
        key = self._epoch_key(epoch)

        # Always record, regardless of whether we prune this epoch — future
        # trials use this as comparison history.
        self.trial.set_user_attr(key, value)

        # Per-trial warmup gate.
        if epoch < self.n_warmup_steps:
            return

        study = self.trial.study
        history_trials = [
            t
            for t in study.get_trials(
                deepcopy=False,
                states=(optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED),
            )
            if t.number != self.trial.number and self._same_stratum(t)
        ]

        # Study-wide startup gate: need enough trials overall to trust the
        # comparison, matching MedianPruner's n_startup_trials semantics.
        if len(history_trials) < self.n_startup_trials:
            return

        historical_values = [
            t.user_attrs[key] for t in history_trials if key in t.user_attrs
        ]
        # Per-epoch coverage gate: enough *other* trials specifically reached
        # (and recorded) this exact epoch, distinct from total trial count.
        if len(historical_values) < self.min_history:
            return

        threshold = _percentile(historical_values, self.percentile)
        if value < threshold:
            raise optuna.TrialPruned(
                f"Trial pruned at epoch {epoch} ({self.monitor}={value:.4f} < "
                f"p{self.percentile:g}={threshold:.4f} over "
                f"{len(historical_values)} historical trials"
                + (f" with matching {self.stratify_by})" if self.stratify_by else ")")
            )
