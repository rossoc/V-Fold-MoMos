# %% [markdown]
# Combined 5-fold CV-verified Pareto front + training-dynamics report for the
# mlp / cifar10 backbone, merging two studies:
#
#   * `mlp_pareto_verification` (WandB): momos2d / hierarchical_momos2d / qat
#     configs. Read via `load_pareto_cv_results_from_wandb`, with `test/acc`
#     per-fold means re-aggregated from the project's runs (same as
#     notebook/mlp_pareto_cv_analysis.py).
#   * `soft_momos_plus` 200-epoch study. CV summary read from the local
#     `outputs/pareto_cv/mlp_soft_200ep/trial*_cv_test.yaml` files; training
#     dynamics from the WandB project `mlp-soft-momos-cv-200ep`.
#
# Pareto-optimality is recomputed fresh across the *combined* set of
# CV-verified numbers (not reused from either Optuna study's `best_trials`),
# so Soft MoMos runs may end up dominated by MoMos/QAT configs from the
# verification study - expected and fine.
#
# The training-dynamics report covers 4 hand-picked configs (the two Soft
# MoMos shapes that survive, plus one MoMos and one QAT reference), labelled
# by method + block shape so runs are identifiable:
#   Soft (2, 2)  = soft_momos_plus t4
#   Soft (2, 5)  = soft_momos_plus t11
#   MoMos (2, 2) = momos2d t133
#   QAT4         = qat t9
# The Soft MoMos (1, 3) and (2, 3) configs are intentionally left out of the
# detail report.
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
    MOMOS_FAMILY_METHODS,
    PARETO_CV_METRICS,
    PARETO_CV_METRICS_OVERVIEW,
    build_reference_backbone,
    build_runs_summary,
    fetch_grouped_runs,
    format_group_label,
    load_pareto_cv_results,
    load_pareto_cv_results_from_wandb,
    pareto_cv_run_names,
)
from src.utils.quant_bits import compute_quantization_bits

ENTITY = "danesinoo-university-of-copenhagen"
PROJECT_MLP = "mlp_pareto_verification"  # momos2d / hierarchical / qat
PROJECT_SOFT = "mlp-soft-momos-cv-200ep"  # soft_momos_plus training dynamics
MODEL_NAME = "mlp"  # for compute_quantization_bits' reference backbone
RESULTS_DIR = "outputs/pareto_cv/mlp_soft_200ep"  # local soft CV results
OUTPUT_PDF = "mlp_combined_cv_report.pdf"
N_FOLDS = 5
K_FIXED = 256  # soft_momos_plus codebook size is fixed for this study

# soft_momos_plus is not in pareto.DEFAULT_METHOD_LABELS - add a presentation
# label (merged so any other method still falls back to the shared map).
METHOD_LABELS = {**DEFAULT_METHOD_LABELS, "soft_momos_plus": "Soft MoMos+"}

api = wandb.Api()

# %%
# Aggregate the per-fold `test/acc` (logged by every run of the verification
# project) into the same CV-verified mean/std shape as `cv_val_acc_*` - the
# soft YAML already carries cv_test_acc_*, the wandb-read verification df
# does not. Re-groups the project's runs by (method, trial_number) via
# `parse_pareto_cv_run_name` and pulls `test/acc` from each fold's summary.
def _add_cv_test_acc(df: pd.DataFrame, entity: str, project: str, api) -> pd.DataFrame:
    from collections import defaultdict
    from src.view.training_analysis import parse_pareto_cv_run_name

    grouped = defaultdict(list)
    for run in api.runs(f"{entity}/{project}"):
        parsed = parse_pareto_cv_run_name(run.name)
        if parsed is None:
            continue
        method, trial_number, _fold = parsed
        grouped[(method, trial_number)].append(run)

    test_means, test_stds = {}, {}
    for (method, trial_number), runs in grouped.items():
        accs = [
            float(run.summary["test/acc"])
            for run in runs
            if run.summary.get("test/acc") is not None
        ]
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


# Reconstruct the resolved soft_momos_plus quantization config from the
# Optuna-sampled `params` dict (smp_rows/smp_cols/smp_lam/smp_freeze_fraction)
# - the same shape tune_quantization.py builds.
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


# %%
# Study 1: the verification project (WandB). Study 2: the soft 200ep study
# (local YAMLs). Concatenate into one df; (method, trial_number) is unique
# across both, so no key collision.
df_mlp = load_pareto_cv_results_from_wandb(ENTITY, PROJECT_MLP, MODEL_NAME, api=api)
df_mlp = _add_cv_test_acc(df_mlp, ENTITY, PROJECT_MLP, api)

cv_files = sorted(glob.glob(f"{RESULTS_DIR}/trial*_cv_test.yaml"))
if not cv_files:
    raise FileNotFoundError(f"No trial*_cv_test.yaml files found in {RESULTS_DIR}")
df_soft = pd.concat([load_pareto_cv_results(f) for f in cv_files], ignore_index=True)
df_soft["quant_cfg"] = df_soft["params"].apply(_to_quant_cfg)

df = pd.concat([df_mlp, df_soft], ignore_index=True)
df["method_label"] = df["method"].replace(METHOD_LABELS)

# %%
# Recompute Pareto-optimality fresh from the combined CV-verified numbers.
df["is_pareto_optimal"] = compute_pareto_mask(
    df, x="compression_rate", y="cv_val_acc_mean"
)
n_opt = int(df["is_pareto_optimal"].sum())
print(f"Loaded {len(df)} CV-verified configs ({len(df_mlp)} verification + {len(df_soft)} soft)")
print(f"  {n_opt} Pareto-optimal, {len(df) - n_opt} dominated (post-CV, combined)")
print(
    df[
        [
            "method",
            "trial_number",
            "method_label",
            "cv_val_acc_mean",
            "cv_test_acc_mean",
            "compression_rate",
            "is_pareto_optimal",
        ]
    ]
    .sort_values(["method", "compression_rate"])
    .to_string(index=False)
)

# %%
# Shared reference backbone (architecture only, no trained weights) - reused
# below both for the compression-breakdown chart and the detail-run labels.
label_backbone = build_reference_backbone(MODEL_NAME)

# %%
# Compact, method-specific display label for a config - used for the Pareto
# front group (via method_label), the compression-breakdown bars, and the
# training-dynamics group keys/legends. Method + block shape identify a run:
#   Soft (r, c) / MoMos (r, c) / QAT{q} / V-Fold / Baseline
def _display_label(method: str, quant_cfg: dict) -> str:
    if method in ("soft_momos_plus", "momos2d"):
        rows, cols = int(quant_cfg["rows"]), int(quant_cfg["cols"])
        prefix = "Soft" if method == "soft_momos_plus" else "MoMos"
        return f"{prefix} ({rows}, {cols})"
    if method == "qat":
        return f"QAT{int(quant_cfg['q'])}"
    if method == "none":
        return "Baseline"
    return DEFAULT_METHOD_LABELS.get(method, method)


# %%
# The one "fundamental" plot: CV-verified Test Accuracy vs. compression rate,
# one marker shape per method, points of the same method connected (sorted by
# compression rate). Pareto-optimality is from the combined set, so Soft
# MoMos points may be marked dominated by the verification study's configs.
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
# Stacked bar breakdown of each Pareto-optimal MoMos-family / Soft MoMos
# config's compressed size, from this DataFrame's own quant_cfg column.
# Handles both the verification-study methods (motif / mosaic / folds via
# compute_quantization_bits) and soft_momos_plus (motif + index only - no
# iota/folds term). Dominated runs and non-MoMos methods (qat, baseline) are
# dropped. Skip the figure rather than crash if nothing survives.
def _build_combined_breakdown(df: pd.DataFrame, backbone) -> pd.DataFrame:
    columns = ["method", "trial_number", "bar_label", "component", "bits_fraction"]
    methods = set(MOMOS_FAMILY_METHODS) | {"soft_momos_plus"}
    opt = df[df["is_pareto_optimal"] & df["method"].isin(methods)]
    if opt.empty:
        return pd.DataFrame(columns=columns)
    rows = []
    for r in opt.itertuples():
        stats = compute_quantization_bits(backbone, r.quant_cfg)
        dense_bits = stats["dense_bits"]
        if r.method == "soft_momos_plus":
            bar_label = _display_label(r.method, r.quant_cfg)  # "Soft (r, c)"
        else:
            # MoMos-family: append the hyperparameter label so each config gets
            # its own bar (format_group_label carries shape + k, distinct per
            # config; the bare method label alone would collide, e.g. two
            # Pareto-optimal V-Fold configs).
            bar_label = (
                f"{DEFAULT_METHOD_LABELS.get(r.method, r.method)}\n"
                f"{format_group_label(r.method, r.quant_cfg, backbone)}"
            )
        for key, display in (
            ("motif_bits", "Motifs"),
            ("index_bits", "Mosaic"),
            ("mosaic_bits", "Mosaic"),
            ("iota_bits", "Folds"),
        ):
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


breakdown_df = _build_combined_breakdown(df, label_backbone)
if breakdown_df.empty:
    print("No Pareto-optimal MoMos-family / Soft configs found - skipping breakdown plot.")
else:
    tallest = breakdown_df.groupby("bar_label")["bits_fraction"].sum().max()
    print(f"Tallest breakdown bar: {tallest:.4f} of dense (y-axis capped at 0.4)")
    # Fixed to [0, 0.4] so the mlp and resnet reports are read against the
    # same ruler. Anything taller than 0.4 would be clipped - print is the guard.
    fig_breakdown = plot_compression_breakdown(breakdown_df, ylim=(0.0, 0.4))
    report.append_figures([fig_breakdown])

# %%
# Hand-picked detail configs for the training-dynamics + metric report: the
# two Soft MoMos shapes that survive (2, 2) / (2, 5), plus a MoMos and a QAT
# reference from the verification study. The Soft MoMos (1, 3) and (2, 3)
# configs are deliberately left out. Selected by (method, trial_number) so the
# report shows exactly the runs requested.
_DETAIL_SPECS = [
    {"method": "soft_momos_plus", "trial_number": 4},  # Soft (2, 2)
    {"method": "soft_momos_plus", "trial_number": 11},  # Soft (2, 5)
    {"method": "momos2d", "trial_number": 133},  # MoMos (2, 2)
    {"method": "qat", "trial_number": 9},  # QAT4
]


def _resolve(spec: dict):
    hits = df[df["method"].eq(spec["method"]) & df["trial_number"].eq(spec["trial_number"])]
    if hits.empty:
        raise ValueError(
            f"No config in df matches detail spec {spec} "
            f"(method={spec['method']}, trial_number={spec['trial_number']})"
        )
    return hits.iloc[0]


detail_rows = [_resolve(s) for s in _DETAIL_SPECS]
for s, r in zip(_DETAIL_SPECS, detail_rows):
    print(f"  t{r.trial_number} {r.method}: {_display_label(r.method, r.quant_cfg)}")

# %%
# Group labels (the display labels above) key each config's WandB fold runs.
# The detail configs span two projects, so build the groups per project and
# fetch+merge separately: soft_momos_plus lives in `mlp-soft-momos-cv-200ep`,
# momos2d/qat in `mlp_pareto_verification`. Run names are
# f"{method}_t{trial_number}_f{fold_idx}" (eval_pareto_cv.py's convention),
# so each resolves to exactly one run per fold.
groups_soft, groups_mlp = {}, {}
for r in detail_rows:
    label = _display_label(r.method, r.quant_cfg)
    names = pareto_cv_run_names(r.method, int(r.trial_number), n_folds=N_FOLDS)
    (groups_soft if r.method == "soft_momos_plus" else groups_mlp)[label] = names

grouped_runs = fetch_grouped_runs(ENTITY, PROJECT_SOFT, groups_soft, api=api)
grouped_runs.update(fetch_grouped_runs(ENTITY, PROJECT_MLP, groups_mlp, api=api))
runs_summary = build_runs_summary(grouped_runs, PARETO_CV_METRICS)

# %%
# The shared PARETO_CV_METRICS_OVERVIEW titles the BDM/QuBD panels with
# \mathcal{\hat{K}}; the user prefers plain K. Override just those two labels
# locally so the shared module (used by the mlp/resnet notebooks) is untouched.
METRICS_OVERVIEW = {**PARETO_CV_METRICS_OVERVIEW}
METRICS_OVERVIEW["metrics/bdm_complexity"] = r"$K_\mathrm{BDM}$"
METRICS_OVERVIEW["metrics/qbdm_complexity"] = r"$K_\mathrm{QuBD}$"

# The runs_summary keys ARE the display labels ("Soft (2, 2)", "MoMos (2, 2)",
# "QAT4", ...), so both the training overview and metrics-vs-accuracy use them
# directly - no separate collapsed-label pass needed.
report.training_overview(runs_summary, METRICS_OVERVIEW)
# %%
report.metrics_vs_accuracy(runs_summary, METRICS_OVERVIEW)
# %%
report.save(OUTPUT_PDF)
print(f"Saved {report.output_dir / OUTPUT_PDF}")
