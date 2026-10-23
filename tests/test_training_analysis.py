"""Tests for src/view/training_analysis.py."""

import sys
import os
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from src.view.training_analysis import (
    _run_metric_history,
    aggregate_metric_across_runs,
    build_compression_breakdown,
    build_reference_backbone,
    compute_quantization_bits,
    fetch_grouped_runs,
    fetch_named_runs,
    format_group_label,
    load_pareto_cv_results,
    load_pareto_cv_results_from_wandb,
    parse_pareto_cv_run_name,
    pareto_cv_run_names,
)

# Built once and shared across format_group_label tests - k_from_capacity
# just needs the architecture (param shapes), so a single reference backbone
# is safe to reuse across every test in this module.
_BACKBONE = build_reference_backbone("resnet20")


def test_pareto_cv_run_names_matches_eval_pareto_cv_naming():
    """Must exactly match src/eval_pareto_cv.py::_run_fold's
    f"{method}_t{trial_number}_f{fold_idx}" naming."""
    names = pareto_cv_run_names("momos2d", 14, n_folds=5)
    assert names == [
        "momos2d_t14_f0",
        "momos2d_t14_f1",
        "momos2d_t14_f2",
        "momos2d_t14_f3",
        "momos2d_t14_f4",
    ]
    print("✓ test_pareto_cv_run_names_matches_eval_pareto_cv_naming passed")


def test_parse_pareto_cv_run_name_roundtrips_with_pareto_cv_run_names():
    for method, trial_number in [("momos2d", 14), ("hierarchical_momos2d", 3), ("fold_momos", 100), ("none", 0)]:
        for name in pareto_cv_run_names(method, trial_number, n_folds=3):
            parsed = parse_pareto_cv_run_name(name)
            assert parsed is not None
            parsed_method, parsed_trial, _fold = parsed
            assert parsed_method == method
            assert parsed_trial == trial_number

    assert parse_pareto_cv_run_name("some-unrelated-run") is None
    assert parse_pareto_cv_run_name("momos2d_t14_f0_extra") is None
    print("✓ test_parse_pareto_cv_run_name_roundtrips_with_pareto_cv_run_names passed")


def test_format_group_label_qat():
    assert format_group_label("qat", {"q": 4}, backbone=_BACKBONE) == "QAT4"
    assert format_group_label("qat", {"q": 8}, backbone=_BACKBONE) == "QAT8"
    print("✓ test_format_group_label_qat passed")


def test_format_group_label_momos2d():
    label = format_group_label(
        "momos2d", {"rows": 2, "cols": 4, "capacity": 0.05}, backbone=_BACKBONE
    )
    # "(rows, cols)\n$k=..$" - k is a positive integer, not the raw 0.05 capacity.
    lines = label.split("\n")
    assert len(lines) == 2
    assert lines[0] == "(2, 4)"
    assert lines[1].startswith("$k=") and lines[1].endswith("$")
    k_str = lines[1].removeprefix("$k=").removesuffix("$")
    assert k_str.isdigit() and int(k_str) > 0
    print("✓ test_format_group_label_momos2d passed")


def test_format_group_label_hierarchical_and_fold_momos_share_format():
    quant_cfg = {
        "primary": {"rows": 2, "cols": 2, "capacity": 0.01},
        "secondary": {"rows": 16, "cols": 16, "capacity": 0.2},
    }
    hier = format_group_label("hierarchical_momos2d", quant_cfg, backbone=_BACKBONE)
    fold = format_group_label("fold_momos", quant_cfg, backbone=_BACKBONE)
    # Same params -> same label (hierarchical/fold_momos share the exact
    # same primary/secondary block-shape format).
    assert hier == fold
    # Every piece on its own line: primary shape, primary k, secondary
    # shape, secondary v.
    lines = hier.split("\n")
    assert len(lines) == 4
    assert lines[0] == "(2, 2)"
    assert lines[1].startswith("$k=") and lines[1].endswith("$")
    assert lines[2] == "(16, 16)"
    assert lines[3].startswith("$v=") and lines[3].endswith("$")
    print("✓ test_format_group_label_hierarchical_and_fold_momos_share_format passed")


def test_format_group_label_falls_back_to_bare_name_for_baseline():
    # "It's enough the name" - no trial-number disambiguator for methods
    # without hyperparameters to show (there's only ever one baseline).
    assert format_group_label("none", {}, backbone=_BACKBONE) == "Baseline"
    print("✓ test_format_group_label_falls_back_to_bare_name_for_baseline passed")


def test_load_pareto_cv_results(tmp_path):
    yaml_path = tmp_path / "study_cv_test.yaml"
    yaml_path.write_text(
        OmegaConf.to_yaml(
            OmegaConf.create(
                {
                    "pareto_cv_results": [
                        {
                            "trial_number": 1,
                            "method": "qat",
                            "compression_rate": 2.0,
                            "cv_val_acc_mean": 0.9,
                            "cv_val_acc_std": 0.01,
                        },
                        {
                            "trial_number": 2,
                            "method": "momos2d",
                            "compression_rate": 4.0,
                            "cv_val_acc_mean": None,  # e.g. 0 completed folds so far
                            "cv_val_acc_std": None,
                        },
                    ]
                }
            )
        )
    )

    df = load_pareto_cv_results(str(yaml_path))
    assert len(df) == 2
    assert set(df["method"]) == {"qat", "momos2d"}

    complete = df[df["cv_val_acc_mean"].notna()]
    assert len(complete) == 1
    assert complete.iloc[0]["trial_number"] == 1
    print("✓ test_load_pareto_cv_results passed")


def test_load_pareto_cv_results_raises_on_empty(tmp_path):
    yaml_path = tmp_path / "empty.yaml"
    yaml_path.write_text(OmegaConf.to_yaml(OmegaConf.create({"pareto_cv_results": []})))
    try:
        load_pareto_cv_results(str(yaml_path))
        raise AssertionError("expected ValueError for empty pareto_cv_results")
    except ValueError:
        pass
    print("✓ test_load_pareto_cv_results_raises_on_empty passed")


class _FakeRunWithHistory:
    def __init__(self, history_df):
        self._history_df = history_df

    def history(self, samples=500):
        return self._history_df


def test_run_metric_history_skips_columns_missing_from_this_runs_history():
    """QAT/baseline runs never log quant/* (only the MoMos-family projection
    callback does) - a metric list that includes quant/distortion must not
    crash for those runs, just omit that column."""
    history = pd.DataFrame(
        {
            "epoch": [0.0, 1.0],
            "val/acc": [0.5, 0.6],
            # no "quant/distortion" column at all, e.g. a QAT run
        }
    )
    run = _FakeRunWithHistory(history)

    result = _run_metric_history(run, metrics=["val/acc", "quant/distortion"])

    assert not result.empty
    assert "val/acc" in result.columns
    assert "quant/distortion" not in result.columns
    print("✓ test_run_metric_history_skips_columns_missing_from_this_runs_history passed")


def test_aggregate_metric_across_runs_buckets_by_rounded_epoch():
    # Two "fold runs" whose epoch floats are close but not bit-identical -
    # round(epoch, 4) must still bucket them together.
    frame_a = pd.DataFrame({"epoch": [0.0, 1.00004], "val/acc": [0.5, 0.6]})
    frame_b = pd.DataFrame({"epoch": [0.0, 0.99996], "val/acc": [0.7, 0.8]})

    result = aggregate_metric_across_runs([frame_a, frame_b], metrics=["val/acc", "val/loss"])

    epochs, means, stds = result["val/acc"]
    assert list(epochs) == [0.0, 1.0]
    assert means[0] == 0.6  # mean(0.5, 0.7)
    assert means[1] == 0.7  # mean(0.6, 0.8)
    assert stds[0] > 0

    # A metric present in neither frame gets the empty-triple fallback.
    empty_epochs, empty_means, empty_stds = result["val/loss"]
    assert len(empty_epochs) == 0 and len(empty_means) == 0 and len(empty_stds) == 0
    print("✓ test_aggregate_metric_across_runs_buckets_by_rounded_epoch passed")


class _FakeRun:
    def __init__(self, name):
        self.name = name


class _FakeApi:
    """Stands in for wandb.Api(): only knows about `existing_names`."""

    def __init__(self, existing_names):
        self._existing = set(existing_names)

    def runs(self, path, filters=None):
        name = filters["display_name"]
        return [_FakeRun(name)] if name in self._existing else []


def test_fetch_named_runs_warns_and_skips_missing():
    api = _FakeApi(existing_names=["qat_t1_f0", "qat_t1_f1"])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = fetch_named_runs(
            "entity", "project", ["qat_t1_f0", "qat_t1_f1", "qat_t1_f2"], api=api
        )

    assert set(found.keys()) == {"qat_t1_f0", "qat_t1_f1"}
    assert any("qat_t1_f2" in str(w.message) for w in caught)
    print("✓ test_fetch_named_runs_warns_and_skips_missing passed")


class _FakeRunWithConfig:
    def __init__(self, name, val_acc, quant_cfg):
        self.name = name
        self.summary = {"val/acc": val_acc}
        self.config = {"quantization": quant_cfg}


class _FakeProjectApi:
    """Stands in for wandb.Api() for load_pareto_cv_results_from_wandb: only
    `.runs(path)` (no filters) is used, returning every run in the project."""

    def __init__(self, runs):
        self._runs = runs

    def runs(self, path, filters=None):
        return self._runs


def test_load_pareto_cv_results_from_wandb_groups_and_recomputes_compression():
    qat_cfg = {"enabled": True, "method": "qat", "q": 4}  # 32/4 = 8x
    none_cfg = {"enabled": False, "method": "none"}  # 1.0x by construction
    runs = [
        _FakeRunWithConfig("qat_t1_f0", 0.90, qat_cfg),
        _FakeRunWithConfig("qat_t1_f1", 0.92, qat_cfg),
        _FakeRunWithConfig("none_t2_f0", 0.95, none_cfg),
        _FakeRunWithConfig("unrelated-run-not-matching-convention", 0.5, {}),
    ]
    api = _FakeProjectApi(runs)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df = load_pareto_cv_results_from_wandb("entity", "project", "resnet20", api=api)

    assert set(df["method"]) == {"qat", "none"}
    qat_row = df[df["method"] == "qat"].iloc[0]
    assert qat_row["trial_number"] == 1
    assert qat_row["n_folds_completed"] == 2
    assert abs(qat_row["cv_val_acc_mean"] - 0.91) < 1e-9
    assert abs(qat_row["compression_rate"] - 8.0) < 1e-9
    assert abs(qat_row["bpp"] - 4.0) < 1e-9

    none_row = df[df["method"] == "none"].iloc[0]
    assert abs(none_row["compression_rate"] - 1.0) < 1e-9
    assert abs(none_row["bpp"] - 32.0) < 1e-9

    # quant_cfg is retained verbatim - format_group_label needs it downstream.
    assert qat_row["quant_cfg"] == qat_cfg
    assert none_row["quant_cfg"] == none_cfg

    assert any("don't match eval_pareto_cv.py's naming convention" in str(w.message) for w in caught)
    print("✓ test_load_pareto_cv_results_from_wandb_groups_and_recomputes_compression passed")


def test_load_pareto_cv_results_from_wandb_raises_when_nothing_matches():
    api = _FakeProjectApi([_FakeRunWithConfig("not-a-pareto-cv-run", 0.5, {})])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            load_pareto_cv_results_from_wandb("entity", "project", "resnet20", api=api)
        raise AssertionError("expected ValueError when no run matches the naming convention")
    except ValueError:
        pass
    print("✓ test_load_pareto_cv_results_from_wandb_raises_when_nothing_matches passed")


_QAT_CFG = {"enabled": True, "method": "qat", "q": 8}
_NONE_CFG = {"enabled": False, "method": "none"}
_MOMOS2D_CFG = {"enabled": True, "method": "momos2d", "rows": 2, "cols": 4, "capacity": 0.05}
_HIER_SECONDARY = {"rows": 16, "cols": 16, "capacity": 0.2}
_HIER_PRIMARY = {"rows": 2, "cols": 2, "capacity": 0.01}
_HIERARCHICAL_CFG = {
    "enabled": True,
    "method": "hierarchical_momos2d",
    "primary": _HIER_PRIMARY,
    "secondary": _HIER_SECONDARY,
}
_FOLD_MOMOS_CFG = {
    "enabled": True,
    "method": "fold_momos",
    "primary": _HIER_PRIMARY,
    "secondary": _HIER_SECONDARY,
}


def _breakdown_fixture_df():
    return pd.DataFrame(
        [
            {"method": "qat", "trial_number": 1, "quant_cfg": _QAT_CFG, "is_pareto_optimal": True},
            {"method": "none", "trial_number": 2, "quant_cfg": _NONE_CFG, "is_pareto_optimal": True},
            {"method": "momos2d", "trial_number": 3, "quant_cfg": _MOMOS2D_CFG, "is_pareto_optimal": True},
            # dominated - must be excluded even though it's momos-family
            {"method": "momos2d", "trial_number": 4, "quant_cfg": _MOMOS2D_CFG, "is_pareto_optimal": False},
            {
                "method": "hierarchical_momos2d",
                "trial_number": 5,
                "quant_cfg": _HIERARCHICAL_CFG,
                "is_pareto_optimal": True,
            },
            {
                "method": "fold_momos",
                "trial_number": 6,
                "quant_cfg": _FOLD_MOMOS_CFG,
                "is_pareto_optimal": True,
            },
        ]
    )


def test_build_compression_breakdown_filters_to_pareto_optimal_momos_family():
    breakdown = build_compression_breakdown(_breakdown_fixture_df(), _BACKBONE)
    # qat/none excluded (no component split / no compression), dominated
    # momos2d row (trial 4) excluded too.
    assert set(breakdown["method"]) == {"momos2d", "hierarchical_momos2d", "fold_momos"}
    assert 4 not in set(breakdown["trial_number"])
    print("✓ test_build_compression_breakdown_filters_to_pareto_optimal_momos_family passed")


def test_build_compression_breakdown_component_counts_per_method():
    breakdown = build_compression_breakdown(_breakdown_fixture_df(), _BACKBONE)

    def components_for(method):
        return set(breakdown[breakdown["method"] == method]["component"])

    assert components_for("momos2d") == {"Motifs", "Mosaic"}
    assert components_for("fold_momos") == {"Motifs", "Mosaic"}  # NOT "Folds" - see quant_bits.py
    assert components_for("hierarchical_momos2d") == {"Motifs", "Mosaic", "Folds"}
    print("✓ test_build_compression_breakdown_component_counts_per_method passed")


def test_build_compression_breakdown_fractions_sum_to_inverse_compression_rate():
    breakdown = build_compression_breakdown(_breakdown_fixture_df(), _BACKBONE)
    cfgs = {"momos2d": _MOMOS2D_CFG, "hierarchical_momos2d": _HIERARCHICAL_CFG, "fold_momos": _FOLD_MOMOS_CFG}
    for method, cfg in cfgs.items():
        rows = breakdown[breakdown["method"] == method]
        total = rows["bits_fraction"].sum()
        expected = compute_quantization_bits(_BACKBONE, cfg)["bpp"] / 32.0
        assert abs(total - expected) < 1e-9, f"{method}: {total} != {expected}"
    print("✓ test_build_compression_breakdown_fractions_sum_to_inverse_compression_rate passed")


def test_build_compression_breakdown_empty_when_no_momos_family_pareto_rows():
    df = pd.DataFrame(
        [
            {"method": "qat", "trial_number": 1, "quant_cfg": _QAT_CFG, "is_pareto_optimal": True},
            {"method": "momos2d", "trial_number": 2, "quant_cfg": _MOMOS2D_CFG, "is_pareto_optimal": False},
        ]
    )
    breakdown = build_compression_breakdown(df, _BACKBONE)
    assert breakdown.empty
    assert list(breakdown.columns) == ["method", "trial_number", "bar_label", "component", "bits_fraction"]
    print("✓ test_build_compression_breakdown_empty_when_no_momos_family_pareto_rows passed")


def test_build_compression_breakdown_disambiguates_hierarchical_vs_fold_momos_labels():
    # Same primary/secondary params -> format_group_label alone ties them,
    # but build_compression_breakdown's bar_label must not.
    breakdown = build_compression_breakdown(_breakdown_fixture_df(), _BACKBONE)
    hier_label = breakdown[breakdown["method"] == "hierarchical_momos2d"]["bar_label"].iloc[0]
    fold_label = breakdown[breakdown["method"] == "fold_momos"]["bar_label"].iloc[0]
    assert hier_label != fold_label
    print("✓ test_build_compression_breakdown_disambiguates_hierarchical_vs_fold_momos_labels passed")


def test_fetch_grouped_runs_drops_group_with_zero_matches():
    """This is the code path that lets the still-running mlp study's
    incomplete fold sets degrade gracefully instead of crashing the report."""
    api = _FakeApi(existing_names=["qat_t1_f0", "qat_t1_f1", "qat_t1_f2", "qat_t1_f3", "qat_t1_f4"])
    groups = {
        "QAT (t1)": pareto_cv_run_names("qat", 1, n_folds=5),  # all 5 exist
        "MoMos (t2)": pareto_cv_run_names("momos2d", 2, n_folds=5),  # none exist yet
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = fetch_grouped_runs("entity", "project", groups, api=api)

    assert list(grouped.keys()) == ["QAT (t1)"]
    assert len(grouped["QAT (t1)"]) == 5
    print("✓ test_fetch_grouped_runs_drops_group_with_zero_matches passed")


class _FakeDuplicateNameApi:
    """Stands in for a project like fold-momos-tinyvit, where the sweep names
    every fold of a config the same thing (`wandb.name="cap${s_cap}"`, no fold
    suffix) so one display name legitimately matches several runs."""

    def __init__(self, counts):
        self._counts = counts  # {display_name: how many runs carry it}

    def runs(self, path, filters=None):
        name = filters["display_name"]
        return [_FakeRun(name) for _ in range(self._counts.get(name, 0))]


def test_fetch_named_runs_keeps_every_run_sharing_a_display_name():
    """The variance-band bug: `matches[0]` kept 1 of the 5 folds, so every
    per-epoch bucket held identical values and `np.std` collapsed to 0."""
    api = _FakeDuplicateNameApi({"cap0.25": 5})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = fetch_named_runs("entity", "project", ["cap0.25"], api=api)

    assert len(found["cap0.25"]) == 5
    assert any("share the display name 'cap0.25'" in str(w.message) for w in caught)
    print("✓ test_fetch_named_runs_keeps_every_run_sharing_a_display_name passed")


def test_fetch_named_runs_does_not_warn_when_each_name_is_unique():
    """pareto_cv_run_names' names carry the fold index, so the ambiguity
    warning must stay quiet for the mlp/resnet notebooks."""
    api = _FakeApi(existing_names=pareto_cv_run_names("qat", 1, n_folds=5))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        found = fetch_named_runs("entity", "project", pareto_cv_run_names("qat", 1), api=api)

    assert all(len(runs) == 1 for runs in found.values())
    assert [str(w.message) for w in caught] == []
    print("✓ test_fetch_named_runs_does_not_warn_when_each_name_is_unique passed")


def test_fetch_grouped_runs_flattens_shared_display_names():
    """One name -> 5 runs must reach build_runs_summary as 5 runs, not 1."""
    api = _FakeDuplicateNameApi({"cap0.25": 5, "cap0.125": 5})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = fetch_grouped_runs(
            "entity", "project", {"V-Fold cap=1/4": ["cap0.25"], "V-Fold cap=1/8": ["cap0.125"]},
            api=api,
        )

    assert len(grouped["V-Fold cap=1/4"]) == 5
    assert len(grouped["V-Fold cap=1/8"]) == 5
    print("✓ test_fetch_grouped_runs_flattens_shared_display_names passed")


def test_fetch_grouped_runs_treats_bare_str_group_value_as_one_name():
    """`str` is itself a `Sequence[str]`; without the guard "cap0.25" would be
    iterated character by character and looked up as 7 separate run names."""
    api = _FakeDuplicateNameApi({"cap0.25": 5})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grouped = fetch_grouped_runs("entity", "project", {"V-Fold cap=1/4": "cap0.25"}, api=api)

    assert len(grouped["V-Fold cap=1/4"]) == 5
    print("✓ test_fetch_grouped_runs_treats_bare_str_group_value_as_one_name passed")


if __name__ == "__main__":
    import tempfile
    from pathlib import Path

    test_pareto_cv_run_names_matches_eval_pareto_cv_naming()
    test_parse_pareto_cv_run_name_roundtrips_with_pareto_cv_run_names()
    test_format_group_label_qat()
    test_format_group_label_momos2d()
    test_format_group_label_hierarchical_and_fold_momos_share_format()
    test_format_group_label_falls_back_to_bare_name_for_baseline()
    with tempfile.TemporaryDirectory() as d:
        test_load_pareto_cv_results(Path(d))
    with tempfile.TemporaryDirectory() as d:
        test_load_pareto_cv_results_raises_on_empty(Path(d))
    test_run_metric_history_skips_columns_missing_from_this_runs_history()
    test_aggregate_metric_across_runs_buckets_by_rounded_epoch()
    test_load_pareto_cv_results_from_wandb_groups_and_recomputes_compression()
    test_load_pareto_cv_results_from_wandb_raises_when_nothing_matches()
    test_build_compression_breakdown_filters_to_pareto_optimal_momos_family()
    test_build_compression_breakdown_component_counts_per_method()
    test_build_compression_breakdown_fractions_sum_to_inverse_compression_rate()
    test_build_compression_breakdown_empty_when_no_momos_family_pareto_rows()
    test_build_compression_breakdown_disambiguates_hierarchical_vs_fold_momos_labels()
    test_fetch_named_runs_warns_and_skips_missing()
    test_fetch_grouped_runs_drops_group_with_zero_matches()
    test_fetch_named_runs_keeps_every_run_sharing_a_display_name()
    test_fetch_named_runs_does_not_warn_when_each_name_is_unique()
    test_fetch_grouped_runs_flattens_shared_display_names()
    test_fetch_grouped_runs_treats_bare_str_group_value_as_one_name()
    print("\n✅ All tests passed!")
