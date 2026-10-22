"""Unit tests for the Optuna pruning callbacks.

Covers ``OptunaMultiObjectiveEpochPruning`` (the manual percentile-based
pruner used for multi-objective studies, since native Optuna report/
should_prune is unsupported there) using a lightweight fake trial/study
harness — no Lightning trainer or real Optuna storage needed. Also includes
a regression check that ``OptunaTrainEpochPruning`` (the native single-
objective path used by the hypernet study) is unaffected.
"""

import optuna
import pytest

from utils.optuna_callback import (
    OptunaMultiObjectiveEpochPruning,
    OptunaTrainEpochPruning,
    _percentile,
)


class FakeTrainer:
    """Stand-in for a Lightning Trainer exposing just what the callbacks read."""

    def __init__(self, metrics, epoch):
        self.callback_metrics = metrics
        self.current_epoch = epoch


class FakeFrozenTrial:
    """Stand-in for a historical (COMPLETE/PRUNED) trial."""

    def __init__(self, number, user_attrs):
        self.number = number
        self.user_attrs = user_attrs


class FakeStudy:
    def __init__(self, trials):
        self._trials = trials

    def get_trials(self, deepcopy=False, states=None):
        return list(self._trials)


class FakeTrial:
    """Stand-in for the live ``optuna.Trial`` the callback is attached to."""

    def __init__(self, number, study):
        self.number = number
        self.study = study
        self.user_attrs = {}

    def set_user_attr(self, key, value):
        self.user_attrs[key] = value


def _callback(trial, **kwargs):
    return OptunaMultiObjectiveEpochPruning(trial, monitor="val/acc", **kwargs)


def test_percentile_helper_known_values():
    assert _percentile([1, 2, 3, 4, 5], 50) == 3
    assert _percentile([1, 2, 3, 4], 50) == 2.5
    assert _percentile([1, 2, 3, 4, 5], 0) == 1
    assert _percentile([1, 2, 3, 4, 5], 100) == 5
    assert _percentile([42.0], 50) == 42.0


def test_percentile_helper_rejects_empty():
    with pytest.raises(ValueError):
        _percentile([], 50)


def test_records_value_but_never_prunes_before_warmup():
    study = FakeStudy(
        [FakeFrozenTrial(i, {"epoch_0_val_acc": 0.9}) for i in range(10)]
    )
    trial = FakeTrial(number=99, study=study)
    cb = _callback(trial, n_startup_trials=1, n_warmup_steps=3, min_history=1)

    # Epoch 0, well below any history, but still inside warmup.
    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.01}, epoch=0), pl_module=None)

    assert trial.user_attrs["epoch_0_val_acc"] == 0.01


def test_no_prune_before_startup_trial_count_reached():
    study = FakeStudy([FakeFrozenTrial(0, {"epoch_3_val_acc": 0.9})])
    trial = FakeTrial(number=1, study=study)
    cb = _callback(trial, n_startup_trials=5, n_warmup_steps=0, min_history=1)

    # Only 1 historical trial exists; n_startup_trials=5 not met.
    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.01}, epoch=3), pl_module=None)

    assert trial.user_attrs["epoch_3_val_acc"] == 0.01


def test_no_prune_when_too_few_trials_have_data_at_this_epoch():
    # Plenty of historical trials overall, but none recorded THIS epoch key
    # (e.g. they were all pruned earlier).
    study = FakeStudy(
        [FakeFrozenTrial(i, {"epoch_0_val_acc": 0.9}) for i in range(10)]
    )
    trial = FakeTrial(number=99, study=study)
    cb = _callback(trial, n_startup_trials=1, n_warmup_steps=0, min_history=3)

    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.01}, epoch=5), pl_module=None)

    # No exception raised (missing comparison data => can't decide, not a
    # trivial "below threshold"), and the value was still recorded.
    assert trial.user_attrs["epoch_5_val_acc"] == 0.01


def test_prunes_trial_below_percentile():
    study = FakeStudy(
        [
            FakeFrozenTrial(0, {"epoch_5_val_acc": 0.5}),
            FakeFrozenTrial(1, {"epoch_5_val_acc": 0.6}),
            FakeFrozenTrial(2, {"epoch_5_val_acc": 0.7}),
            FakeFrozenTrial(3, {"epoch_5_val_acc": 0.8}),
            FakeFrozenTrial(4, {"epoch_5_val_acc": 0.9}),
        ]
    )
    trial = FakeTrial(number=99, study=study)
    cb = _callback(trial, n_startup_trials=5, n_warmup_steps=0, min_history=3, percentile=50.0)

    with pytest.raises(optuna.TrialPruned):
        cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.11}, epoch=5), pl_module=None)


def test_survives_trial_above_percentile():
    study = FakeStudy(
        [
            FakeFrozenTrial(0, {"epoch_5_val_acc": 0.5}),
            FakeFrozenTrial(1, {"epoch_5_val_acc": 0.6}),
            FakeFrozenTrial(2, {"epoch_5_val_acc": 0.7}),
            FakeFrozenTrial(3, {"epoch_5_val_acc": 0.8}),
            FakeFrozenTrial(4, {"epoch_5_val_acc": 0.9}),
        ]
    )
    trial = FakeTrial(number=99, study=study)
    cb = _callback(trial, n_startup_trials=5, n_warmup_steps=0, min_history=3, percentile=50.0)

    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.85}, epoch=5), pl_module=None)

    assert trial.user_attrs["epoch_5_val_acc"] == 0.85


def test_excludes_current_trial_from_its_own_history():
    # Even if the current trial's own number somehow appears in get_trials
    # (defensive case), it must not count toward n_startup_trials/history.
    study = FakeStudy([FakeFrozenTrial(99, {"epoch_0_val_acc": 0.5})])
    trial = FakeTrial(number=99, study=study)
    cb = _callback(trial, n_startup_trials=1, n_warmup_steps=0, min_history=1)

    # No *other* trials exist, so this must not prune despite a low value.
    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.01}, epoch=0), pl_module=None)


def test_missing_monitor_metric_is_ignored():
    study = FakeStudy([])
    trial = FakeTrial(number=0, study=study)
    cb = _callback(trial, n_startup_trials=0, n_warmup_steps=0, min_history=0)

    # No exception, no recorded attr, since the monitored metric is absent.
    cb.on_train_epoch_end(FakeTrainer({}, epoch=0), pl_module=None)

    assert trial.user_attrs == {}


class _FakeSingleObjectiveStudy:
    directions = [optuna.study.StudyDirection.MAXIMIZE]

    def _is_multi_objective(self):
        return False


class _RecordingTrial:
    """Minimal stand-in for optuna.Trial to check OptunaTrainEpochPruning
    still calls report/should_prune for single-objective studies."""

    def __init__(self):
        self.study = _FakeSingleObjectiveStudy()
        self.reported = []
        self._should_prune = False

    def report(self, value, step):
        self.reported.append((value, step))

    def should_prune(self):
        return self._should_prune


def test_single_objective_callback_still_reports_and_can_prune():
    trial = _RecordingTrial()
    cb = OptunaTrainEpochPruning(trial, monitor="val/acc")

    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.42}, epoch=2), pl_module=None)
    assert trial.reported == [(0.42, 2)]

    trial._should_prune = True
    with pytest.raises(optuna.TrialPruned):
        cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.42}, epoch=3), pl_module=None)


def _stratified_study():
    # Five high-compression trials (16x) with low accuracy and five
    # low-compression ones (8x) with high accuracy, all recorded at epoch 5.
    return FakeStudy(
        [FakeFrozenTrial(i, {"epoch_5_val_acc": 0.30 + 0.01 * i, "compression_rate": 16.0}) for i in range(5)]
        + [FakeFrozenTrial(10 + i, {"epoch_5_val_acc": 0.60 + 0.01 * i, "compression_rate": 8.0}) for i in range(5)]
    )


def test_stratified_pruning_compares_only_within_same_compression_rate():
    trial = FakeTrial(number=99, study=_stratified_study())
    trial.set_user_attr("compression_rate", 16.0 * 1.01)  # within the 5% tolerance
    cb = _callback(
        trial, n_startup_trials=5, n_warmup_steps=0, min_history=3,
        percentile=50.0, stratify_by="compression_rate",
    )
    # 0.40 is below the study-wide median (~0.45) but above the 16x median (0.32):
    # it must survive when stratified ...
    cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.40}, epoch=5), pl_module=None)
    # ... and be pruned without stratification.
    cb_global = _callback(trial, n_startup_trials=5, n_warmup_steps=0, min_history=3, percentile=50.0)
    with pytest.raises(optuna.TrialPruned):
        cb_global.on_train_epoch_end(FakeTrainer({"val/acc": 0.40}, epoch=5), pl_module=None)


def test_stratified_pruning_still_prunes_within_stratum_and_skips_unmatched():
    trial = FakeTrial(number=99, study=_stratified_study())
    trial.set_user_attr("compression_rate", 16.0)
    cb = _callback(
        trial, n_startup_trials=5, n_warmup_steps=0, min_history=3,
        percentile=50.0, stratify_by="compression_rate",
    )
    with pytest.raises(optuna.TrialPruned):
        cb.on_train_epoch_end(FakeTrainer({"val/acc": 0.25}, epoch=5), pl_module=None)

    # A trial at a compression rate nobody else has: no history -> never pruned.
    lonely = FakeTrial(number=98, study=_stratified_study())
    lonely.set_user_attr("compression_rate", 32.0)
    cb_lonely = _callback(
        lonely, n_startup_trials=5, n_warmup_steps=0, min_history=3,
        percentile=50.0, stratify_by="compression_rate",
    )
    cb_lonely.on_train_epoch_end(FakeTrainer({"val/acc": 0.01}, epoch=5), pl_module=None)
