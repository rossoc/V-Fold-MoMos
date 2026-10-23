"""Generic Pareto-front scatter, decoupled from any specific data source.

Consumes a plain DataFrame - it has no knowledge of Optuna, YAML, or
quantization - so callers translate their own data into columns first
(see `notebook/pareto_front_analysis.py` for the Optuna-specific reader).
"""

from typing import Literal, Sequence

import pandas as pd

from .figure import Figure

# Matplotlib marker codes, assigned to groups in sorted order. Extend this
# list if a study ever uses more distinct groups than it has entries.
MARKERS: list[str] = ["*", "o", "s", "^", "D", "v", "P", "X", "h", "<", ">", "8"]

# Raw quantization.method strings (as stored by tune_quantization.py /
# eval_pareto_cv.py) -> presentation labels for plot legends. Shared so
# every notebook that plots a method-grouped Pareto front relabels
# identically instead of each inlining its own copy that can drift.
DEFAULT_METHOD_LABELS: dict[str, str] = {
    "none": "Baseline",
    "momos2d": "MoMos",
    "qat": "QAT",
    "hierarchical_momos2d": "V-Fold",
    "fold_momos": "Fold MoMos",
}


def compute_pareto_mask(data: pd.DataFrame, x: str, y: str) -> pd.Series:
    """Boolean mask of rows in `data` that are Pareto-optimal, maximizing
    both `data[x]` and `data[y]` (e.g. compression rate and accuracy).

    A row is optimal iff no other row is >= it on both `x` and `y` with a
    strict > on at least one (standard non-dominated-set definition). Two
    rows with identical (x, y) never dominate each other, so exact ties are
    both kept as optimal.

    Args:
        data: rows to evaluate.
        x, y: numeric column names to maximize; should have no NaNs (filter
            those out before calling, e.g. `df[df[y].notna()]`).

    Returns:
        A `bool` `pd.Series` aligned to `data.index`, assignable directly as
        a new column and passable as `plot_pareto_front`'s `optimal=` arg.
    """
    xs = data[x].to_numpy(dtype=float)
    ys = data[y].to_numpy(dtype=float)
    # dominates[i, j]: does row j dominate row i? (>= on both axes, > on at
    # least one). The diagonal is automatically False (no strict
    # improvement over itself), so no explicit fill_diagonal is needed.
    ge_x = xs[None, :] >= xs[:, None]
    ge_y = ys[None, :] >= ys[:, None]
    gt_x = xs[None, :] > xs[:, None]
    gt_y = ys[None, :] > ys[:, None]
    dominated_by_someone = (ge_x & ge_y & (gt_x | gt_y)).any(axis=1)
    return pd.Series(~dominated_by_someone, index=data.index)


def plot_pareto_front(
    data: pd.DataFrame,
    x: str,
    y: str,
    group: str,
    optimal: str | None = None,
    connect: bool = False,
    y_err: str | None = None,
    x_err: str | None = None,
    x_label: str = "x",
    y_label: str = "y",
    title: str | None = None,
    markers: Sequence[str] = MARKERS,
    optimal_color: str = "crimson",
    dominated_color: str = "steelblue",
    dominated_alpha: float = 0.45,
    optimal_alpha: float = 0.95,
    connect_color: str = "gray",
    connect_alpha: float = 0.5,
    connect_linewidth: float = 1.2,
    error_capsize: float = 3.0,
    markersize: float = 70.0,
    style: Literal["sci", "scientific", "plain"] = "plain",
    figsize: tuple[float, float] = (9, 7),
    fontsize: int = 13,
) -> Figure:
    """Scatter `data[x]` vs `data[y]`, one marker shape per distinct value of
    `data[group]` (cycled from `markers`). If `optimal` names a boolean
    column, Pareto-optimal rows are drawn solid/large and the rest
    faded/small, with a second color legend; otherwise every point is drawn
    the same way and only the method-shape legend is shown.

    Args:
        data: rows to plot - any DataFrame with at least `x`, `y`, `group`
            columns (and `optimal` if given). Not tied to any particular
            source; a hand-built DataFrame works just as well as one read
            from an Optuna study.
        x, y: column names for the scatter coordinates.
        group: column name whose distinct values each get their own marker
            shape (e.g. a quantization method).
        optimal: optional boolean column name marking Pareto-optimal rows.
        connect: if True, points sharing a `group` value are additionally
            joined by a thin line (sorted by `x`), so a trade-off curve per
            group is visible. The line sits behind the markers (single
            neutral `connect_color` for every group - marker *shape*
            already disambiguates group via the legend, so the line doesn't
            need its own per-group color too).
        y_err, x_err: optional column names giving a symmetric error bar
            half-width per row (e.g. std across CV folds) - drawn as a plain
            error bar (no marker of its own) in the same color/alpha as that
            point's own marker, so a dominated point's error bar fades along
            with it.
        x_label, y_label, title: passed through to the underlying `Figure`.
        markers: marker codes assigned to `sorted(data[group].unique())` in
            order; must have at least as many entries as distinct groups.

    Returns:
        A `Figure` ready to `.save()` standalone or hand to a `Report` via
        `report.append_figures([fig])`.

    Raises:
        ValueError: more distinct `group` values than entries in `markers`.
    """
    groups = sorted(data[group].unique())
    if len(groups) > len(markers):
        raise ValueError(
            f"{len(groups)} distinct '{group}' values but only {len(markers)} "
            "markers in MARKERS - extend src/view/pareto.py:MARKERS."
        )
    marker_of = dict(zip(groups, markers))

    fig = Figure(fontsize=fontsize, figSize=figsize)
    ax = fig._ax()

    # (is_optimal, alpha, size, edgecolor) passes to draw per group. Without an
    # `optimal` column every row is drawn once, using the "optimal" styling.
    passes = (
        [(True, optimal_alpha, markersize, "black")]
        if optimal is None
        else [
            (True, optimal_alpha, markersize, "black"),
            (False, dominated_alpha, markersize * 0.6, "none"),
        ]
    )
    for g in groups:
        g_rows = data[data[group] == g]
        if connect and len(g_rows) > 1:
            g_sorted = g_rows.sort_values(x)
            ax.plot(
                g_sorted[x],
                g_sorted[y],
                color=connect_color,
                alpha=connect_alpha,
                linewidth=connect_linewidth,
                zorder=1,
            )
        for is_opt, alpha, size, edge in passes:
            subset = g_rows if optimal is None else g_rows[g_rows[optimal] == is_opt]
            if subset.empty:
                continue
            if y_err is not None or x_err is not None:
                ax.errorbar(
                    subset[x],
                    subset[y],
                    yerr=subset[y_err] if y_err is not None else None,
                    xerr=subset[x_err] if x_err is not None else None,
                    fmt="none",
                    ecolor=(optimal_color if is_opt else dominated_color),
                    alpha=alpha,
                    capsize=error_capsize,
                    linewidth=1.0,
                    zorder=1.5,  # above the connecting line, below the markers
                )
            ax.scatter(
                subset[x],
                subset[y],
                marker=marker_of[g],
                s=size,
                alpha=alpha,
                color=(optimal_color if is_opt else dominated_color),
                edgecolors=edge,
                linewidths=0.8,
                zorder=2,
            )

    # Legend #1: marker shape -> group. Proxy handles (empty scatters) so the
    # legend key shows the shape regardless of how many points were actually
    # drawn for that group.
    method_handles = [
        ax.scatter([], [], marker=marker_of[g], color="black", label=str(g))
        for g in groups
    ]
    method_legend = ax.legend(
        handles=method_handles, title="Method", loc="lower left", framealpha=1.0
    )

    # Legend #2 (only when `optimal` is given): color -> Pareto status. Added
    # via `add_artist` since a second `ax.legend()` call would otherwise
    # replace the first one.
    if optimal is not None:
        ax.add_artist(method_legend)
        status_handles = [
            ax.scatter([], [], marker="o", color=optimal_color, label="Pareto-optimal"),
            ax.scatter(
                [], [], marker="o", color=dominated_color, alpha=dominated_alpha,
                label="Dominated",
            ),
        ]
        ax.legend(handles=status_handles, loc="upper right", framealpha=1.0)

    fig._default_settings(x_label, y_label, title or "", style, False, False, grid=True)
    return fig


# Component display name -> bar-segment color, and stacking order
# (bottom-to-top). Every MoMos-family bar has "Motifs"; "Mosaic" is the
# index/lookup layer every method has (single- or outer-level); "Folds" is
# hierarchical_momos2d's extra per-fold index layer, absent for
# momos2d/fold_momos (simply skipped for those bars, not a zero segment).
COMPONENT_COLORS: dict[str, str] = {
    "Motifs": "steelblue",
    "Mosaic": "coral",
    "Folds": "darkseagreen",
}
COMPONENT_ORDER: list[str] = ["Motifs", "Mosaic", "Folds"]


def plot_compression_breakdown(
    data: pd.DataFrame,
    label: str = "bar_label",
    component: str = "component",
    fraction: str = "bits_fraction",
    component_colors: dict[str, str] = COMPONENT_COLORS,
    component_order: Sequence[str] = COMPONENT_ORDER,
    x_label: str = "",
    y_label: str = r"$1 / r_\mathrm{\phi}$",
    title: str = "",
    bar_width: float = 0.6,
    ylim: tuple[float, float] | None = (0.0, 1.0),
    figsize: tuple[float, float] = (11, 7),
    fontsize: int = 13,
) -> Figure:
    """Stacked bar chart of `data[fraction]` (e.g. bits spent per
    compression component, as a fraction of dense/FP32 size), one bar per
    distinct `data[label]` value, ordered ascending by each bar's total
    height (best-compressed / smallest first, so the trend reads
    left-to-right). Y-axis defaults to [0, 1]: 1.0 is an uncompressed (FP32)
    model, 0.0 an unreachable zero-bit one.

    `data` is any long-form DataFrame with (at least) `label`, `component`,
    `fraction` columns, one row per (bar, component) pair - no
    WandB/backbone/quantization knowledge here;
    `training_analysis.build_compression_breakdown` is what shapes the
    pareto-cv DataFrame into this form.

    Args:
        data: long-form rows, one per (bar, component) pair. Must not be
            empty - caller decides whether to skip building this figure at
            all (e.g. zero Pareto-optimal MoMos-family configs).
        label: column naming each bar (x-axis category).
        component: column naming each bar's stacked segment.
        fraction: column giving each segment's height; must be additive -
            segments sharing a `label` stack to that bar's total.
        component_colors: maps every distinct `data[component]` value to a
            fixed color.
        component_order: stacking order, bottom to top; a bar missing one
            of these components (e.g. momos2d has no "Folds") simply skips
            that segment, no gap left behind.
        ylim: y-axis range, or None to autoscale to the bars. The [0, 1]
            default keeps "fraction of the dense budget" readable at a
            glance, but buries the detail when every bar is small - a study
            whose bars all sit below ~0.2 leaves its smallest components
            (e.g. the motif codebook, or hierarchical's per-fold term)
            indistinguishable from the axis. Pass None there.

    Returns:
        A `Figure` ready to `.save()` standalone or hand to a `Report` via
        `report.append_figures([fig])`.

    Raises:
        ValueError: `data` is empty, or contains a `component` value absent
            from `component_colors`.
    """
    if data.empty:
        raise ValueError("`data` is empty - nothing to plot; caller should skip this figure.")
    unknown = set(data[component].unique()) - set(component_colors)
    if unknown:
        raise ValueError(f"component(s) {sorted(unknown)} missing from component_colors.")

    labels_sorted = list(data.groupby(label)[fraction].sum().sort_values().index)

    fig = Figure(fontsize=fontsize, figSize=figsize)
    ax = fig._ax()

    seen = set()
    for x, bar_label in enumerate(labels_sorted):
        bar_rows = data[data[label] == bar_label].set_index(component)[fraction]
        bottom = 0.0
        for comp in component_order:
            if comp not in bar_rows.index:
                continue
            height = float(bar_rows[comp])
            ax.bar(
                x,
                height,
                width=bar_width,
                bottom=bottom,
                color=component_colors[comp],
                label=comp if comp not in seen else None,
            )
            seen.add(comp)
            bottom += height

    handles, labels_ = ax.get_legend_handles_labels()
    order = [labels_.index(c) for c in component_order if c in labels_]
    ax.legend([handles[i] for i in order], [labels_[i] for i in order], title="Component")

    # _default_settings must run BEFORE set_xticks: it calls
    # ax.ticklabel_format(axis="both", ...), which raises "AttributeError:
    # This method only works with the ScalarFormatter" once the x-axis has
    # categorical/string tick labels installed (verified empirically) - so
    # labels/grid/title first, then replace the x-axis ticks.
    fig._default_settings(x_label, y_label, title, "plain", False, False, axis=None, grid=True)
    ax.set_xticks(range(len(labels_sorted)), labels_sorted)
    if ylim is not None:
        ax.set_ylim(*ylim)
    return fig
