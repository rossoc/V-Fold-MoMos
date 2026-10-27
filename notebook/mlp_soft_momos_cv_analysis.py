# %% [markdown]
# 5-fold CV-verified Pareto front + training-dynamics report for the
# `soft_momos_plus` study on the mlp / cifar10 backbone, trained for 200
# epochs. Adapted from notebook/mlp_pareto_cv_analysis.py for a *single*
# quantization method (soft_momos_plus) whose CV results were written to
# local YAML files rather than only logged to WandB.
#
# Two data sources, mirroring how the study was actually run:
#   * The CV-verified Pareto summary (val/test acc mean/std across 5 folds,
#     compression_rate, bpp, per-config params) is read straight from the
#     local `outputs/pareto_cv/mlp_soft_200ep/trial*_cv_test.yaml` files
#     written by `src/eval_pareto_cv.py` (`load_pareto_cv_results`). This is
#     the authoritative summary - it already carries `cv_test_acc_*`, so no
#     re-derivation from per-fold runs is needed (unlike the WandB-only
#     mlp/resnet notebooks, which had to reconstruct it).
#   * The per-epoch training dynamics (val/acc, val/loss, train/acc,
#     train/loss + the compression metrics) are read from the WandB project
#     `mlp-soft-momos-cv-200ep` (one run per (config, fold), named
#     `soft_momos_plus_t<trial>_f<fold>`), because the local YAML only stores
#     the compression metrics per epoch, not the accuracy/loss curves.
#
# Pareto-optimality is recomputed fresh from the CV-verified numbers (NOT
# reused from the source Optuna study's `best_trials`), exactly as in the
# mlp/resnet notebooks. The report then builds the same figures as
# mlp_pareto_cv_analysis.py - Pareto front, compression breakdown, training
# overview, metrics-vs-accuracy - but adapted for soft_momos_plus:
#   * `soft_momos_plus` is a single method, so the Pareto front has one
#     group (no per-method marker shapes needed).
#   * The compression breakdown is built inline for soft_momos_plus (it is
#     not in `training_analysis.MOMOS_FAMILY_METHODS`, so the shared
#     `build_compression_breakdown` would drop it). `compute_quantization_bits`
#     returns a motif/index split for it (no iota/folds term).
#   * Group labels show the searched hyperparameters: block shape (rows,
#     cols) and the freeze fraction (k is fixed at 256 for this study).
# %%
import glob

import numpy as np
import pandas as pd
import wandb

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
    build_reference_backbone,
    build_runs_summary,
    fetch_grouped_runs,
    load_pareto_cv_results,
    pareto_cv_run_names,
)
from src.utils.quant_bits import compute_quantization_bits

ENTITY = "danesinoo-university-of-copenhagen"
PROJECT = "mlp-soft-momos-cv-200ep"  # training-dynamics runs (one per config x fold)
MODEL_NAME = "mlp"  # for compute_quantization_bits' reference backbone
RESULTS_DIR = "outputs/pareto_cv/mlp_soft_200ep"  # local eval_pareto_cv.py output
OUTPUT_PDF = "mlp_soft_momos_cv_report.pdf"
N_FOLDS = 5
K_FIXED = 256  # soft_momos_plus codebook size is fixed for this study

# soft_momos_plus is not in pareto.DEFAULT_METHOD_LABELS - add a presentation
# label for it (merged so any other method still falls back to the shared map).
METHOD_LABELS = {**DEFAULT_METHOD_LABELS, "soft_momos_plus": "Soft MoMos+"}

api = wandb.Api()

# %%
# Load + concatenate the per-config CV summaries from the local YAML files
# written by eval_pareto_cv.py. Each file holds one `pareto_cv_results` entry.
cv_files = sorted(glob.glob(f"{RESULTS_DIR}/trial*_cv_test.yaml"))
if not cv_files:
    raise FileNotFoundError(f"No trial*_cv_test.yaml files found in {RESULTS_DIR}")
df = pd.concat([load_pareto_cv_results(f) for f in cv_files], ignore_index=True)

# The YAML `params` dict stores the Optuna-sampled names (smp_rows, smp_cols,
# smp_lam, smp_freeze_fraction). Reconstruct the resolved quantization config
# that compute_quantization_bits / the group labels expect (rows, cols, lam,
# freeze_fraction, k=256, ...) - the same shape tune_quantization.py builds.
def _to_quant_cfg(params: dict) -> dict:
    return {
        "enabled": True,
        "method": params["method"],
        "rows": int(params["smp_rows"]),
        "cols": int(params["smp_cols"]),
        "lam": float(params["smp_lam"]),
        "freeze_fraction": float(params["smp_freeze_fraction"]),
        "k": K_FIXED,
        "capacity": None,
        "force_zero": True,
        "warm_start": True,
        "reseed_dead": True,
        "lloyd_iters": 1,
    }


df["quant_cfg"] = df["params"].apply(_to_quant_cfg)
df["method_label"] = df["method"].replace(METHOD_LABELS)

# %%
# Recompute Pareto-optimality fresh from the CV-verified numbers (confirmed
# design choice - see module docstring above).
df["is_pareto_optimal"] = compute_pareto_mask(
    df, x="compression_rate", y="cv_val_acc_mean"
)
n_opt = int(df["is_pareto_optimal"].sum())
print(f"Loaded {len(df)} CV-verified configs from {RESULTS_DIR}")
print(f"  {n_opt} Pareto-optimal, {len(df) - n_opt} dominated (post-CV)")
print(
    df[
        [
            "trial_number",
            "method_label",
            "cv_val_acc_mean",
            "cv_val_acc_std",
            "cv_test_acc_mean",
            "cv_test_acc_std",
            "compression_rate",
            "is_pareto_optimal",
        ]
    ]
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
# The one "fundamental" plot: CV-verified Test Accuracy vs. compression rate.
# All configs are soft_momos_plus, so there is a single group (one marker
# shape) and the connecting line traces the method's accuracy/compression
# trade-off curve. No "capacity" plot is built here - it doesn't apply to a
# single quantization method.
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
# Stacked bar breakdown of each Pareto-optimal config's compressed size
# (motif / mosaic), from this DataFrame's own quant_cfg column.
#
# soft_momos_plus is NOT in training_analysis.MOMOS_FAMILY_METHODS, so the
# shared build_compression_breakdown would drop it - build the long-form
# breakdown inline instead, using the same compute_quantization_bits that
# populated compression_rate/bpp (so results stay self-consistent). For
# soft_momos_plus it returns a motif/index split (no iota/folds term), mapped
# to the same "Motifs"/"Mosaic" display names as the shared helper.
# Compact group label for soft_momos_plus: block shape (rows, cols) and the
# freeze fraction - the two searched hyperparameters that set the trade-off
# (k is fixed at 256 for this study, so it carries no information).
def _group_label(quant_cfg: dict) -> str:
    rows, cols = int(quant_cfg["rows"]), int(quant_cfg["cols"])
    ff = float(quant_cfg["freeze_fraction"])
    return f"({rows}, {cols})\n$f_f={ff:.2f}$"


def _build_soft_breakdown(df: pd.DataFrame, backbone) -> pd.DataFrame:
    columns = ["method", "trial_number", "bar_label", "component", "bits_fraction"]
    opt = df[df["is_pareto_optimal"]]
    if opt.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for r in opt.itertuples():
        stats = compute_quantization_bits(backbone, r.quant_cfg)
        dense_bits = stats["dense_bits"]
        bar_label = f"{r.method_label}\n{_group_label(r.quant_cfg)}"
        for key, display in (("motif_bits", "Motifs"), ("index_bits", "Mosaic")):
            if key in stats:
                rows.append(
                    {
                        "method": r.method,
                        "trial_number": int(r.trial_number),
                        "bar_label": bar_label,
                        "component": display,
                        "bits_fraction": stats[key] / dense_bits,
                    }
                )
    return pd.DataFrame(rows, columns=columns)


breakdown_df = _build_soft_breakdown(df, label_backbone)
if breakdown_df.empty:
    print("No Pareto-optimal configs found - skipping compression breakdown plot.")
else:
    tallest = breakdown_df.groupby("bar_label")["bits_fraction"].sum().max()
    print(f"Tallest breakdown bar: {tallest:.4f} of dense (y-axis capped at 0.4)")
    # Fixed to [0, 0.4] so the mlp and resnet reports are read against the
    # same ruler (see mlp_pareto_cv_analysis.py). Anything taller than 0.4
    # would be clipped silently - the print above is the guard.
    fig_breakdown = plot_compression_breakdown(breakdown_df, ylim=(0.0, 0.4))
    report.append_figures([fig_breakdown])

# %%
# Detail configs for the training-dynamics + metric report. The 4
# Pareto-optimal configs are selected (the dominated t27 is excluded - it is
# dominated by t33 and not part of the front's trade-off story). This also
# matches the 2x2 grid Report.metrics_vs_accuracy draws (one subplot per
# config), which the original mlp/resnet notebooks sized for 4 configs.
DETAIL_TRIAL_NUMBERS = sorted(
    int(t) for t in df.loc[df["is_pareto_optimal"], "trial_number"].unique()
)
detail_df = df[df["trial_number"].isin(DETAIL_TRIAL_NUMBERS)]
print(f"Detail configs (trial_number): {DETAIL_TRIAL_NUMBERS}")

for row in detail_df.itertuples():
    print(f"  t{row.trial_number} {row.method_label}: {_group_label(row.quant_cfg)}")

# %%
# Group labels reveal each config's hyperparameters at a glance. The WandB
# run names come from the raw `method` (eval_pareto_cv.py names each fold's
# run f"{method}_t{trial_number}_f{fold_idx}"), so each is expected to resolve
# to exactly one run. All 25 runs exist (study complete) - fetch_grouped_runs
# warns and drops any that don't.
groups = {
    _group_label(row.quant_cfg): pareto_cv_run_names(
        row.method, int(row.trial_number), n_folds=N_FOLDS
    )
    for row in detail_df.itertuples()
}
grouped_runs = fetch_grouped_runs(ENTITY, PROJECT, groups, api=api)
runs_summary = build_runs_summary(grouped_runs, PARETO_CV_METRICS)

# %%
# For the Training Overview only, collapse the verbose per-config group
# labels down to just the block shape "(rows, cols)". All configs are the
# same algorithm (soft_momos_plus), so the method name doesn't identify a
# run - the block shape does (and every shape here is unique). metrics_vs_accuracy
# below keeps the full hyperparameter labels unchanged.
def _overview_method_label(quant_cfg: dict) -> str:
    rows, cols = int(quant_cfg["rows"]), int(quant_cfg["cols"])
    return f"({rows}, {cols})"


runs_summary_overview = {
    _overview_method_label(row.quant_cfg): runs_summary[_group_label(row.quant_cfg)]
    for row in detail_df.itertuples()
}
report.training_overview(runs_summary_overview, PARETO_CV_METRICS_OVERVIEW)
# %%
# The shared PARETO_CV_METRICS_OVERVIEW titles the BDM/QuBD panels with
# \mathcal{\hat{K}}; the user prefers plain K. Override just those two labels
# locally so the shared module (used by the mlp/resnet notebooks) is untouched.
METRICS_OVERVIEW = {**PARETO_CV_METRICS_OVERVIEW}
METRICS_OVERVIEW["metrics/bdm_complexity"] = r"$K_\mathrm{BDM}$"
METRICS_OVERVIEW["metrics/qbdm_complexity"] = r"$K_\mathrm{QuBD}$"
report.metrics_vs_accuracy(runs_summary, METRICS_OVERVIEW)
# %%
report.save(OUTPUT_PDF)
print(f"Saved {report.output_dir / OUTPUT_PDF}")
