# %% [markdown]
# 5-fold CV-verified Pareto front + training-dynamics report for the
# `resent-pareto-verification` WandB project (resnet20 / cifar10 study,
# complete).
#
# Everything is read straight from WandB (like notebook/analysis.ipynb) -
# `eval_pareto_cv.py` never writes a combined per-config summary to WandB
# itself, only one run per (method, trial_number, fold), so
# `load_pareto_cv_results_from_wandb` reconstructs the CV-verified summary
# (val/acc mean/std across folds, recomputed compression_rate/bpp) directly
# from those runs. Pareto-optimality is then recomputed fresh from those
# CV-verified numbers (NOT reused from the source Optuna study's
# `best_trials`, since the noisy single-fold val_acc that produced those may
# no longer hold after real 5-fold verification). Finally builds a
# training-dynamics + per-metric report for 4 hand-picked configs - styled
# like notebook/momos2d_analysis.py, but skipping its first "capacity" plot
# (doesn't apply across a mix of different quantization methods) and adding
# two metrics momos2d_analysis.py didn't have: QuBD and Entropy.
# %%
import wandb

import numpy as np
import pandas as pd

from src.view import Report
from src.view.pareto import (
    DEFAULT_METHOD_LABELS,
    compute_pareto_mask,
    plot_compression_breakdown,
    plot_pareto_front,
)
from src.view.training_analysis import (
    PARETO_CV_METRICS,
    PARETO_CV_METRICS_OVERVIEW,
    build_compression_breakdown,
    build_reference_backbone,
    build_runs_summary,
    fetch_grouped_runs,
    format_group_label,
    load_pareto_cv_results_from_wandb,
    parse_pareto_cv_run_name,
    pareto_cv_run_names,
)

ENTITY = "danesinoo-university-of-copenhagen"
PROJECT = "resent-pareto-verification"  # exact spelling as registered on WandB
MODEL_NAME = "resnet20"  # for compute_quantization_bits' reference backbone
OUTPUT_PDF = "resnet_pareto_cv_report.pdf"
N_FOLDS = 5

api = wandb.Api()

# %%
df = load_pareto_cv_results_from_wandb(ENTITY, PROJECT, MODEL_NAME, api=api)
# Keep the raw `method` (needed to reconstruct WandB run names below)
# alongside a separate presentation `method_label` column for the plot/report.
df["method_label"] = df["method"].replace(DEFAULT_METHOD_LABELS)

# %%
# Aggregate the per-fold `test/acc` (logged by every run) into the same
# CV-verified mean/std shape as `cv_val_acc_*`, so the Pareto front can be
# drawn on Test Accuracy instead of Validation Accuracy. Re-groups the
# project's runs by (method, trial_number) via `parse_pareto_cv_run_name`
# (mirroring load_pareto_cv_results_from_wandb) and pulls `test/acc` from
# each fold's summary.
def _add_cv_test_acc(df: pd.DataFrame, entity: str, project: str, api) -> pd.DataFrame:
    from collections import defaultdict

    grouped = defaultdict(list)
    for run in api.runs(f"{entity}/{project}"):
        parsed = parse_pareto_cv_run_name(run.name)
        if parsed is None:
            continue
        method, trial_number, _fold = parsed
        grouped[(method, trial_number)].append(run)

    test_means, test_stds = {}, {}
    for (method, trial_number), runs in grouped.items():
        accs = [float(run.summary["test/acc"]) for run in runs if run.summary.get("test/acc") is not None]
        if accs:
            test_means[(method, trial_number)] = float(np.mean(accs))
            test_stds[(method, trial_number)] = float(np.std(accs))
    df["cv_test_acc_mean"] = df.apply(
        lambda r: test_means.get((r["method"], r["trial_number"]), float("nan")), axis=1
    )
    df["cv_test_acc_std"] = df.apply(
        lambda r: test_stds.get((r["method"], r["trial_number"]), float("nan")), axis=1
    )
    n_missing = int(df["cv_test_acc_mean"].isna().sum())
    if n_missing:
        print(f"WARNING: {n_missing} config(s) missing test/acc summary - left as NaN")
    return df


df = _add_cv_test_acc(df, ENTITY, PROJECT, api)
# %%
# Recompute Pareto-optimality fresh from the CV-verified numbers (confirmed
# design choice - see module docstring above).
df["is_pareto_optimal"] = compute_pareto_mask(df, x="compression_rate", y="cv_val_acc_mean")
n_opt = int(df["is_pareto_optimal"].sum())
print(f"Loaded {len(df)} CV-verified configs from {ENTITY}/{PROJECT}")
print(f"  {n_opt} Pareto-optimal, {len(df) - n_opt} dominated (post-CV)")
print(
    df[["trial_number", "method_label", "cv_val_acc_mean", "cv_val_acc_std", "compression_rate", "is_pareto_optimal"]]
    .sort_values("compression_rate")
    .to_string(index=False)
)

# %%
# Shared reference backbone (architecture only, no trained weights) - reused
# below both for the compression-breakdown chart and the detail-run group
# labels, so there's a single source of truth for "the backbone used to
# interpret this study's configs".
label_backbone = build_reference_backbone(MODEL_NAME)

# %%
# The one "fundamental" plot: CV-verified Test Accuracy vs. compression
# rate, one marker shape per method, points of the same method connected by
# a line (sorted by compression rate). No "capacity" plot is built here -
# it doesn't apply across a mix of heterogeneous quantization methods.
fig_pareto = plot_pareto_front(
    df,
    x="compression_rate",
    y="cv_test_acc_mean",
    group="method_label",
    optimal="is_pareto_optimal",
    connect=True,
    y_err="cv_test_acc_std",  # 5-fold variance - very relevant, worth showing
    x_label=r"$r_\mathrm{\phi}$",
    y_label="Test Accuracy",
)

report = Report()
report.append_figures([fig_pareto])

# %%
# Stacked bar breakdown of each Pareto-optimal MoMos-family config's
# compressed size (motif / mosaic / iota), from this CV-verified DataFrame's
# own quant_cfg column - no synthetic sweep like notebook/complexity.ipynb.
# Dominated runs and non-MoMos methods (qat, the dense baseline) are dropped
# internally by build_compression_breakdown.
breakdown_df = build_compression_breakdown(df, label_backbone)
if breakdown_df.empty:
    print("No Pareto-optimal MoMos-family configs found - skipping compression breakdown plot.")
else:
    tallest = breakdown_df.groupby('bar_label')['bits_fraction'].sum().max()
    print(f'Tallest breakdown bar: {tallest:.4f} of dense (y-axis capped at 0.4)')
    # Fixed to [0, 0.4] rather than the default [0, 1] or an autoscale: the
    # mlp and resnet reports are read side by side, so their bars have to be
    # measured against the same ruler. Anything taller than 0.4 would be
    # clipped silently - the print below is the guard.
    fig_breakdown = plot_compression_breakdown(breakdown_df, ylim=(0.0, 0.4))
    report.append_figures([fig_breakdown])

# %%
# Hand-picked detail configs for the training-dynamics + metric report (all
# plots after the first two "all-runs" figures: the Pareto front and the
# compression breakdown). Selected directly by `method` + `trial_number`
# (resolve these from the WandB `resent-pareto-verification` project for the
# runs you actually want to inspect), so the report shows exactly the trials
# requested rather than whatever random config happens to match a set of
# hyperparameters from an earlier study snapshot.
_DETAIL_SPECS = [
    {"method": "momos2d", "trial_number": 20},
    {"method": "momos2d", "trial_number": 29},
    {"method": "none", "trial_number": 1},  # dense baseline (enabled=false, method=none)
    {"method": "momos2d", "trial_number": 28},
]


def _resolve_trial_number(spec: dict) -> int:
    """Return the requested `trial_number`, validating it exists in `df`."""
    tn = int(spec["trial_number"])
    hits = df[df["method"].eq(spec["method"]) & df["trial_number"].eq(tn)]
    if hits.empty:
        raise ValueError(
            f"No config in df matches detail spec {spec} "
            f"(method={spec['method']}, trial_number={tn})"
        )
    return tn


DETAIL_TRIAL_NUMBERS = [_resolve_trial_number(s) for s in _DETAIL_SPECS]
detail_df = df[df["trial_number"].isin(DETAIL_TRIAL_NUMBERS)]
print(f"Detail configs (trial_number): {DETAIL_TRIAL_NUMBERS}")
for s, tn in zip(_DETAIL_SPECS, DETAIL_TRIAL_NUMBERS):
    row = df[df["trial_number"] == tn].iloc[0]
    print(
        f"  t{tn} {s['method']}: "
        f"{format_group_label(row['method'], row['quant_cfg'], label_backbone)}"
    )

# %%
# Group labels reveal each config's hyperparameters at a glance instead of
# just "<method> (t<trial>)" - see format_group_label's docstring for the
# exact per-method format (QAT{q} / (rows, cols), k / hierarchical & fold_momos'
# (r, c), k, (r', c'), k'). The WandB run names themselves still come from
# the raw `method` (eval_pareto_cv.py names each fold's run
# f"{method}_t{trial_number}_f{fold_idx}"). Those names carry the fold index,
# so each is expected to resolve to exactly one run - if fetch_named_runs warns
# that a name matched several, a fold was logged twice and is now counted twice.
groups = {
    format_group_label(row.method, row.quant_cfg, label_backbone): pareto_cv_run_names(
        row.method, int(row.trial_number), n_folds=N_FOLDS
    )
    for row in detail_df.itertuples()
}
grouped_runs = fetch_grouped_runs(ENTITY, PROJECT, groups, api=api)
runs_summary = build_runs_summary(grouped_runs, PARETO_CV_METRICS)

# %%
# For the Training Overview only, collapse the verbose per-config group labels
# down to just the Method (+ trial_number, so repeated methods stay distinct):
#   Baseline / MoMs / V-Fold / QAT4|QAT8|QAT16
# metrics_vs_accuracy below keeps the full hyperparameter labels unchanged.
def _overview_method_label(method: str, trial_number: int, quant_cfg) -> str:
    label = DEFAULT_METHOD_LABELS.get(method, method)
    if method == "qat":
        return f"QAT{int(quant_cfg.get('q', ''))}"
    if method == "none":
        return "Baseline"
    return f"{label} t{trial_number}"

runs_summary_overview = {
    _overview_method_label(row.method, int(row.trial_number), row.quant_cfg): runs_summary[
        format_group_label(row.method, row.quant_cfg, label_backbone)
    ]
    for row in detail_df.itertuples()
}
report.training_overview(runs_summary_overview, PARETO_CV_METRICS_OVERVIEW)
# %%
report.metrics_vs_accuracy(runs_summary, PARETO_CV_METRICS_OVERVIEW)
# %%
report.save(OUTPUT_PDF)
print(f"Saved {report.output_dir / OUTPUT_PDF}")
