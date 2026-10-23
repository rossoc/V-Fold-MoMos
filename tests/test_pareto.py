"""Tests for src/view/pareto.py: compute_pareto_mask and plot_pareto_front's
`connect` option."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")

from matplotlib.legend import Legend
import pandas as pd

from src.view.pareto import compute_pareto_mask, plot_compression_breakdown, plot_pareto_front


def test_compute_pareto_mask_basic():
    """(1, 1) is dominated by (2, 2); (2, 2) and (3, 1) are mutually
    non-dominated; an exact duplicate of (2, 2) stays optimal too (ties never
    dominate each other)."""
    df = pd.DataFrame(
        {
            "x": [1, 2, 3, 2],
            "y": [1, 2, 1, 2],
        }
    )
    mask = compute_pareto_mask(df, x="x", y="y")

    assert mask.tolist() == [False, True, True, True]
    assert list(mask.index) == list(df.index)
    print("✓ test_compute_pareto_mask_basic passed")


def test_compute_pareto_mask_all_optimal_when_no_domination():
    """A strictly decreasing x/y trade-off: every point is Pareto-optimal."""
    df = pd.DataFrame({"x": [1, 2, 3], "y": [30, 20, 10]})
    mask = compute_pareto_mask(df, x="x", y="y")
    assert mask.tolist() == [True, True, True]
    print("✓ test_compute_pareto_mask_all_optimal_when_no_domination passed")


def _two_group_df():
    return pd.DataFrame(
        {
            "x": [3, 1, 2, 1, 2, 3],
            "y": [30, 10, 20, 10, 20, 30],
            "g": ["a", "a", "a", "b", "b", "b"],
        }
    )


def test_plot_pareto_front_connect_true_draws_sorted_lines():
    df = _two_group_df()
    fig = plot_pareto_front(df, x="x", y="y", group="g", connect=True)
    lines = fig.ax[0][0].get_lines()

    assert len(lines) == 2  # one connecting line per group ("a", "b")
    for line in lines:
        xdata = list(line.get_xdata())
        assert xdata == sorted(xdata), "connecting line must be sorted by x"
    print("✓ test_plot_pareto_front_connect_true_draws_sorted_lines passed")


def test_plot_pareto_front_connect_false_default_draws_no_lines():
    """Default behavior (connect=False) is unaffected - this is what
    notebook/pareto_front_analysis.py's existing call site relies on."""
    df = _two_group_df()
    fig = plot_pareto_front(df, x="x", y="y", group="g")
    assert fig.ax[0][0].get_lines() == []
    print("✓ test_plot_pareto_front_connect_false_default_draws_no_lines passed")


def test_plot_pareto_front_connect_with_optimal_still_works():
    """connect=True combined with the optimal/dominated split (the actual
    usage in the new pareto-cv notebooks) doesn't raise and still draws one
    line per group."""
    df = _two_group_df()
    df["is_opt"] = [True, False, True, True, False, True]
    fig = plot_pareto_front(df, x="x", y="y", group="g", optimal="is_opt", connect=True)
    assert len(fig.ax[0][0].get_lines()) == 2
    print("✓ test_plot_pareto_front_connect_with_optimal_still_works passed")


def test_plot_pareto_front_y_err_draws_error_bars():
    df = _two_group_df()
    df["yerr"] = [1.0, 2.0, 0.5, 1.5, 2.5, 0.2]
    fig = plot_pareto_front(df, x="x", y="y", group="g", y_err="yerr")
    ax = fig.ax[0][0]

    assert len(ax.containers) == 2  # one errorbar container per group
    for container in ax.containers:
        assert container.has_yerr
        assert not container.has_xerr
    print("✓ test_plot_pareto_front_y_err_draws_error_bars passed")


def test_plot_pareto_front_no_err_by_default():
    df = _two_group_df()
    fig = plot_pareto_front(df, x="x", y="y", group="g")
    assert fig.ax[0][0].containers == []
    print("✓ test_plot_pareto_front_no_err_by_default passed")


def _find_legend_by_title(ax, title: str) -> Legend | None:
    for child in ax.get_children():
        if isinstance(child, Legend) and child.get_title().get_text() == title:
            return child
    return None


def test_plot_pareto_front_method_legend_is_lower_left():
    df = _two_group_df()
    df["is_opt"] = [True, False, True, True, False, True]
    fig = plot_pareto_front(df, x="x", y="y", group="g", optimal="is_opt")
    ax = fig.ax[0][0]

    method_legend = _find_legend_by_title(ax, "Method")
    assert method_legend is not None
    assert method_legend._loc == Legend.codes["lower left"]
    print("✓ test_plot_pareto_front_method_legend_is_lower_left passed")


def _breakdown_long_df():
    return pd.DataFrame(
        [
            # 2-component bar ("momos2d"-shaped): fraction sums to 0.3
            {"bar_label": "MoMos A", "component": "Motifs", "bits_fraction": 0.2},
            {"bar_label": "MoMos A", "component": "Mosaic", "bits_fraction": 0.1},
            # 3-component bar ("hierarchical_momos2d"-shaped): sums to 0.6
            {"bar_label": "V-Fold B", "component": "Motifs", "bits_fraction": 0.3},
            {"bar_label": "V-Fold B", "component": "Mosaic", "bits_fraction": 0.2},
            {"bar_label": "V-Fold B", "component": "Folds", "bits_fraction": 0.1},
        ]
    )


def test_plot_compression_breakdown_y_axis_fixed_0_1():
    fig = plot_compression_breakdown(_breakdown_long_df())
    assert fig.ax[0][0].get_ylim() == (0.0, 1.0)
    print("✓ test_plot_compression_breakdown_y_axis_fixed_0_1 passed")


def test_plot_compression_breakdown_bars_sorted_ascending_by_total_height():
    # "MoMos A" totals 0.3, "V-Fold B" totals 0.6 -> A should come first.
    fig = plot_compression_breakdown(_breakdown_long_df())
    xticklabels = [t.get_text() for t in fig.ax[0][0].get_xticklabels()]
    assert xticklabels == ["MoMos A", "V-Fold B"]
    print("✓ test_plot_compression_breakdown_bars_sorted_ascending_by_total_height passed")


def test_plot_compression_breakdown_stacks_two_vs_three_segments():
    fig = plot_compression_breakdown(_breakdown_long_df())
    ax = fig.ax[0][0]
    # 2 segments for "MoMos A" + 3 segments for "V-Fold B" = 5 bar patches.
    assert len(ax.patches) == 5
    total_height = sum(p.get_height() for p in ax.patches)
    assert abs(total_height - (0.3 + 0.6)) < 1e-9
    print("✓ test_plot_compression_breakdown_stacks_two_vs_three_segments passed")


def test_plot_compression_breakdown_legend_deduplicated_and_ordered():
    fig = plot_compression_breakdown(_breakdown_long_df())
    ax = fig.ax[0][0]
    legend = _find_legend_by_title(ax, "Component")
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    # 3 distinct components present, each exactly once, in Motifs/Mosaic/Folds order.
    assert texts == ["Motifs", "Mosaic", "Folds"]
    print("✓ test_plot_compression_breakdown_legend_deduplicated_and_ordered passed")


def test_plot_compression_breakdown_raises_on_empty_data():
    empty = pd.DataFrame(columns=["bar_label", "component", "bits_fraction"])
    try:
        plot_compression_breakdown(empty)
        raise AssertionError("expected ValueError on empty data")
    except ValueError:
        pass
    print("✓ test_plot_compression_breakdown_raises_on_empty_data passed")


def test_plot_compression_breakdown_raises_on_unknown_component():
    df = pd.DataFrame([{"bar_label": "X", "component": "Unknown", "bits_fraction": 0.5}])
    try:
        plot_compression_breakdown(df)
        raise AssertionError("expected ValueError on unrecognized component")
    except ValueError:
        pass
    print("✓ test_plot_compression_breakdown_raises_on_unknown_component passed")


if __name__ == "__main__":
    test_compute_pareto_mask_basic()
    test_compute_pareto_mask_all_optimal_when_no_domination()
    test_plot_pareto_front_connect_true_draws_sorted_lines()
    test_plot_pareto_front_connect_false_default_draws_no_lines()
    test_plot_pareto_front_connect_with_optimal_still_works()
    test_plot_pareto_front_y_err_draws_error_bars()
    test_plot_pareto_front_no_err_by_default()
    test_plot_pareto_front_method_legend_is_lower_left()
    test_plot_compression_breakdown_y_axis_fixed_0_1()
    test_plot_compression_breakdown_bars_sorted_ascending_by_total_height()
    test_plot_compression_breakdown_stacks_two_vs_three_segments()
    test_plot_compression_breakdown_legend_deduplicated_and_ordered()
    test_plot_compression_breakdown_raises_on_empty_data()
    test_plot_compression_breakdown_raises_on_unknown_component()
    print("\nAll tests passed!")
