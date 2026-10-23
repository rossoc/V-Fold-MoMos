# %% [markdown]
# # V-Fold MoMos × TinyViT analysis
#
# Run cell-by-cell in ipy/nvim. Each `# %%` is a cell.
#
# `USE_CACHED_CSV` reads the repo-root `fold-momos-tinyvit.csv` (written by a
# previous live pull) instead of hitting wandb - flip it to `False` (or delete
# the csv) to re-pull.
# %%
import ast
from fractions import Fraction
from pathlib import Path

import pandas as pd
import wandb
from tqdm import tqdm

from src.view import (
    Report,
    build_compression_breakdown,
    build_reference_backbone,
    build_runs_summary,
    compute_pareto_mask,
    fetch_grouped_runs,
    PARETO_CV_METRICS_OVERVIEW,
    plot_compression_breakdown,
    plot_pareto_front,
)

# Importing `src.view` is what puts `src/` itself on sys.path (see the
# comment in src/view/training_analysis.py), which is what makes this bare
# import resolve. It is the same function `src/train.py` logs `bits/*` with,
# so every compression number below matches what the runs recorded.
from utils.quant_bits import compute_quantization_bits  # noqa: E402

PROJECT = "fold-momos-tinyvit"
ENTITY = "danesinoo-university-of-copenhagen"
MODEL_NAME = "tinyvit"
CSV_PATH = Path(f"../{PROJECT}.csv")
USE_CACHED_CSV = True

# %% [markdown]
# ## 1. Load runs (cached csv by default)
# %%
api = wandb.Api()

if USE_CACHED_CSV and CSV_PATH.exists():
    runs_df = pd.read_csv(CSV_PATH)
else:
    runs = api.runs(f"{ENTITY}/{PROJECT}")
    summary_list, config_list, name_list = [], [], []
    for run in tqdm(runs):
        summary_list.append(run.summary._json_dict)
        config_list.append(
            {k: v for k, v in run.config.items() if not k.startswith("_")}
        )
        name_list.append(run.name)
    runs_df = pd.DataFrame(
        {"summary": summary_list, "config": config_list, "name": name_list}
    )
    runs_df.to_csv(CSV_PATH, index=False)

runs_df

# %% [markdown]
# ## 2. Reference backbone for the bit accounting
#
# Every compression number in this report - the RAS axis of the Pareto
# front and every bar of the breakdown - now comes from
# `utils.quant_bits.compute_quantization_bits`, the single function that
# also produced the `bits/*` values the training runs logged.
#
# **Breakdown bug fixed here:** this notebook used to re-derive the bit
# budget by hand from a hardcoded `N_PARAMS`, treating each hierarchical
# config as if it were a flat momos2d over its *secondary* block. That
# ignored the primary codebook and the per-fold (iota) term completely, and
# put V-Fold RAS out by up to 3x - it reported 29.5 at capacity 1/32 where
# the run itself logged 9.97.
#
# `build_reference_backbone` needs only the architecture, never trained
# weights: the accounting is a function of parameter *shapes*.
# %%
backbone = build_reference_backbone(MODEL_NAME)
n_params = compute_quantization_bits(backbone, None)["num_parameters"]
print(f"{MODEL_NAME}: {n_params:,} trainable parameters")

# %% [markdown]
# ## 3. Build the per-run dataframe
#
# One row per wandb run (5 CV folds per config). Three families, keyed off
# each run's own `config.quantization`:
#   - `baseline`        -> dense, quantization disabled
#   - `momos-baseline`  -> flat momos2d at the top level
#   - everything else   -> hierarchical (V-Fold) momos2d, whose only sweep
#                          dimension is `quantization.secondary.capacity`
#
# `quant_cfg` is carried through verbatim so every compression number
# downstream is computed from the config the run actually trained with,
# rather than reconstructed from a handful of scraped scalars.
# %%
DISPLAY_NAMES = {"baseline": "Baseline", "momos-baseline": "MoMos Baseline"}

momos2d_data = []
for r in runs_df.itertuples():
    summary = ast.literal_eval(r.summary) if isinstance(r.summary, str) else r.summary
    config = ast.literal_eval(r.config) if isinstance(r.config, str) else r.config
    quant_cfg = config.get("quantization") or {}
    secondary = quant_cfg.get("secondary")
    momos2d_data.append(
        dict(
            name=DISPLAY_NAMES.get(r.name, "V-Fold MoMos"),
            run_name=r.name,
            method=quant_cfg.get("method"),
            quant_cfg=quant_cfg,
            capacity=float(secondary["capacity"]) if secondary else float("nan"),
            val_acc=summary["val/acc"],
            val_loss=summary["val/loss"],
            train_acc=summary["train/acc"],
            train_loss=summary["train/loss"],
            test_acc=summary.get("test/acc", None),
        )
    )

momos2d_df = pd.DataFrame(momos2d_data)
momos2d_df

# %% [markdown]
# ## 4. Per-config summary + Pareto front
#
# One row per config: `val/acc` mean +/- std over its 5 CV folds, and RAS
# from `compute_quantization_bits`. For the V-Fold configs that RAS
# reproduces the `bits/ratio` each run logged during training to the digit
# (5.7427 / 6.5569 / 7.4690 / 8.5552 / 9.9739) - the old hand-rolled
# formula did not.
#
# The old "Validation Accuracy vs Capacity" figure is gone. Capacity is a
# per-method knob rather than a shared axis - `Baseline` and `MoMos
# Baseline` were both parked at a fake `capacity=1.0` purely to have
# somewhere to sit - and RAS already carries the same trade-off in units
# that compare across methods. Same reason
# notebook/mlp_pareto_cv_analysis.py builds no capacity plot.
# %%
config_stats = (
    momos2d_df.groupby("run_name", sort=False)
    .agg(
        name=("name", "first"),
        method=("method", "first"),
        quant_cfg=("quant_cfg", "first"),
        capacity=("capacity", "first"),
        val_acc_mean=("val_acc", "mean"),
        val_acc_std=("val_acc", "std"),
        n_folds=("val_acc", "size"),
    )
    .reset_index()
)
config_stats["ras"] = [
    compute_quantization_bits(backbone, cfg)["compression_rate"]
    for cfg in config_stats["quant_cfg"]
]
config_stats["is_pareto_optimal"] = compute_pareto_mask(
    config_stats, x="ras", y="val_acc_mean"
)
# build_compression_breakdown labels its output rows by `trial_number`; this
# study has no Optuna trials behind it, so hand each config a stable ordinal.
config_stats["trial_number"] = range(len(config_stats))

n_opt = int(config_stats["is_pareto_optimal"].sum())
print(f"{n_opt} / {len(config_stats)} configs are Pareto-optimal")
print(
    config_stats[
        [
            "run_name",
            "name",
            "ras",
            "val_acc_mean",
            "val_acc_std",
            "n_folds",
            "is_pareto_optimal",
        ]
    ]
    .sort_values("ras")
    .to_string(index=False)
)

# %%
report = Report()

fig_pareto = plot_pareto_front(
    config_stats,
    x="ras",
    y="val_acc_mean",
    group="name",
    # Every config in this study happens to sit on the front, which would
    # leave the optimal/dominated colour legend describing nothing - only
    # switch it on once something is actually dominated.
    optimal=(
        "is_pareto_optimal" if not config_stats["is_pareto_optimal"].all() else None
    ),
    connect=True,
    y_err="val_acc_std",  # spread across the 5 CV folds
    x_label="RAS",
    y_label="Validation Accuracy",
)
report.append_figures([fig_pareto])

# %% [markdown]
# ## 5. Compression breakdown
#
# The share of the dense 32-bit budget each Pareto-optimal MoMos-family
# config actually spends, stacked by component - straight from
# `compute_quantization_bits` via `build_compression_breakdown`, the same
# path notebook/mlp_pareto_cv_analysis.py uses. The V-Fold bars now carry
# all three components:
#
#   Motifs - the primary codebook, `k * s1 * 32` bits
#   Mosaic - the outer index layer, one entry per fold
#   Folds  - hierarchical's per-fold index (`iota`) term, which the old
#            hand-rolled breakdown left out entirely
#
# Bar labels come from `format_group_label`, so a bar reads
# "V-Fold / (1, 2) k=2537 / (4, 32) k'=1268" - method, then the primary
# block and its codebook, then the fold shape and the number of those
# motifs each fold keeps. `k'` is notebook/complexity.py's notation, where
# `v = r' * c'` is the fold count (already carried by the "(4, 32)" line)
# and `k'` the per-fold codebook. The dense `Baseline` has no compression
# to decompose and is dropped internally.
# %%
breakdown_df = build_compression_breakdown(config_stats, backbone)
if breakdown_df.empty:
    print("No Pareto-optimal MoMos-family configs - skipping compression breakdown.")
else:
    # Autoscale rather than the [0, 1] default: every bar here sits below
    # 0.19, which would squash the whole chart into the bottom fifth and
    # hide the Motifs and Folds segments entirely.
    fig_breakdown = plot_compression_breakdown(
        breakdown_df, y_label=r"$1 / \mathrm{RAS}$", ylim=None
    )
    report.append_figures([fig_breakdown])

# %% [markdown]
# ## 6. Detail runs for the training / metrics overview
#
# **Variance bug fixed here:** every CV fold of a config shares a single
# wandb display name - the sweep sets `wandb.name="cap${cap}"` and never
# suffixes the fold - so the old
# `api.runs(..., filters={"display_name": ...})[0]` resolved all 5 folds of
# a config to the *same* run. The per-epoch buckets then held 5 identical
# values, `np.std` collapsed to 0, and `fill_between(mean, mean)` drew a
# zero-height band. Keeping the whole match set (no `[0]`) is what puts the
# real fold spread back into the training-dynamics figures; that fix now
# lives in `fetch_grouped_runs`/`fetch_named_runs`, so this notebook uses
# the shared helper again instead of its own inline `api.runs` loop.
#
# `capacity=1/2` is dropped to make room for the MoMos baseline:
# `metrics_vs_accuracy` lays out one subplot per series on a fixed 2x2
# grid, so 4 series is the hard cap.
# %%
MOMOS2D_METRICS = [
    "metrics/qbdm_complexity",
    "metrics/bdm_complexity",
    "metrics/gzip_compression_rate",
    "metrics/bz2_compression_rate",
    "metrics/lzma_compression_rate",
    "metrics/weight_l2",
    "metrics/sparsity",
    "quant/distortion",
    "quant/num_changed_weights",
    "val/loss",
    "val/acc",
    "train/loss",
    "train/acc",
]

DETAIL_CAPACITIES = [1 / 4, 1 / 8, 1 / 16]

# %%
vfold_names = (
    momos2d_df[momos2d_df["name"] == "V-Fold MoMos"]
    .groupby("capacity")["run_name"]
    .first()
)
detail_groups = {
    f"V-Fold cap={Fraction(c).limit_denominator()}": vfold_names[c]
    for c in DETAIL_CAPACITIES
}
detail_groups["MoMos Baseline"] = momos2d_df.loc[
    momos2d_df["name"] == "MoMos Baseline", "run_name"
].iloc[0]

# This sweep names all 5 folds of a config the same thing
# (scripts/fold-momos-sweep-tinyvit.sh passes `wandb.name="cap${s_cap}"` with no
# fold suffix), so each label here is *one* display name that must resolve to 5
# runs. `fetch_grouped_runs` keeps every match per name - it used to keep only
# the first, which collapsed each group to a single run and flattened every
# variance band in the report below.
grouped_runs = fetch_grouped_runs(ENTITY, PROJECT, detail_groups, api=api)
# Each line must read 5 runs (one per fold) - a 1 here means the runs
# collapsed by display name again and every variance band will be flat.
for label, runs in grouped_runs.items():
    print(f"{label}: {len(runs)} runs ({detail_groups[label]})")

# %% [markdown]
# ## 7. Per-epoch metric histories (mean +/- std across the folds)
#
# `build_runs_summary` is the library equivalent of the hand-rolled bucketing
# loop this cell used to carry: it fetches each run's history, keeps only the
# metric columns that run actually logged (so a method-specific metric can't
# `KeyError` the whole detail set) and reduces each group to
# `{metric: (epochs, means, stds)}`.
# %%
runs_summary = build_runs_summary(grouped_runs, MOMOS2D_METRICS)

# %% [markdown]
# ## 8. Assemble the report
# %%
tr_overview = {
    "val/acc": "Validation Accuracy",
    "val/loss": "Validation Loss",
    "train/acc": "Training Accuracy",
    "train/loss": "Training Loss",
}
report.training_overview(runs_summary, tr_overview, alpha=0.25)

# %% [markdown]
# Metric titles come from the shared `PARETO_CV_METRICS_OVERVIEW`
# (src/view/training_analysis.py) instead of this notebook's own prose
# titles, so this report and the pareto-cv ones name the same quantity the
# same way - and so the long ones stop crowding the axis:
#
#   "Gzip Compression Rate"     -> $r_\mathrm{Gzip}$
#   "QBDM Complexity"           -> $\hat{K}_\mathrm{QuBD}$
#   "BDM Complexity"            -> $\hat{K}_\mathrm{BDM}$
#   "Number of Changed Weights" -> # Changed Weights
#
# Filtered to the metrics this sweep actually logged:
# `Report.metrics_vs_accuracy` opens a figure per entry *before* checking
# whether any series has data, so an unlogged one (this project never
# records `metrics/weight_entropy`) would add a blank page to the PDF.
# %%
def _logged(metric: str) -> bool:
    return any(
        len(series[metric][0]) for series in runs_summary.values() if metric in series
    )


metrics_overview = {
    m: t for m, t in PARETO_CV_METRICS_OVERVIEW.items() if _logged(m)
}
print("Metric pages:", list(metrics_overview))
report.metrics_vs_accuracy(runs_summary, metrics_overview, style="sci")

# %%
report.save("vfold_momos_vit.pdf")
report.close()
print("Saved assets/vfold_momos_vit.pdf")

# %% [markdown]
# ## 9. The actual sweep
#
# The literal sweep definitions every run in this report came from.
# %%
SWEEP_SCRIPTS = [
    Path("scripts/fold-momos-sweep-tinyvit.sh"),
    Path("scripts/baseline-fold-momos-tinyvit.sh"),
]
for script in SWEEP_SCRIPTS:
    print(f"{'=' * 70}\n{script}\n{'=' * 70}")
    print(script.read_text())
