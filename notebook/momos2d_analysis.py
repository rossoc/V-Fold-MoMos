# %% [markdown]
#
# %%
import pandas as pd
import wandb
from src.view import (
    Report,
    compute_pareto_mask,
    extract_columns,
    merge_dfs,
    plot_compression_breakdown,
)
import ast
import numpy as np
from tqdm import tqdm
from collections import defaultdict

project = "momos2d-remake"  # "momos-reproduction"
api = wandb.Api()


def momos2d_k(n, cap, s):
    """Realized motif/codebook count for a momos2d config - the `k` shown in
    the compression-breakdown bar label, matching
    `src.view.training_analysis.format_group_label`'s convention (the actual
    integer motif count, not the raw `capacity` fraction)."""
    m = np.ceil(n / s)
    return int(np.maximum(1, np.floor(m * cap)))


def momos_complexity_breakdown(n, cap, s, q=32):
    """Same formula as `momos_complexity`, split into its two components -
    for the compression-breakdown bar chart (`plot_compression_breakdown`),
    which needs each component's bit count separately rather than summed."""
    m = np.ceil(n / s)
    k = momos2d_k(n, cap, s)
    motif_complexity = k * s * q
    mosaic_complexity = m * np.ceil(np.log2(k))
    # "Motifs" (plural) is the key plot_compression_breakdown expects -
    # COMPONENT_COLORS in src/view/pareto.py keys off it, and a singular
    # "Motif" raises ValueError before the figure is ever drawn.
    return {"Motifs": motif_complexity, "Mosaic": mosaic_complexity}


def momos_complexity(n, cap, s, q=32):
    parts = momos_complexity_breakdown(n, cap, s, q=q)
    return parts["Motifs"] + parts["Mosaic"]


# %%

# Project is specified by <entity/project-name>
# runs = api.runs(f"danesinoo-university-of-copenhagen/{project}")
#
# summary_list, config_list, name_list = [], [], []
# for run in tqdm(runs):
#     summary_list.append(run.summary._json_dict)
#     config_list.append({k: v for k, v in run.config.items() if not k.startswith("_")})
#
#     name_list.append(run.name)
#
# runs_df = pd.DataFrame(
#     {"summary": summary_list, "config": config_list, "name": name_list}
# )
#
# # %%
# runs_df.to_csv(f"{project}.csv")
runs_df = pd.read_csv(f"{project}.csv")


# %%
report = Report()
# %%
momos2d_runs = []
for i, r in enumerate(runs_df.iterrows()):
    try:
        config = ast.literal_eval(r[1]["config"])
    except Exception as _:
        config = r[1]["config"]

    if (
        not config.get("quantization", None)
        or config["quantization"].get("method", None) != "momos2d"
    ):
        continue
    rows = config["quantization"]["rows"]
    cols = config["quantization"]["cols"]

    try:
        momos2d_runs.append(r[1])
    except Exception as e:
        print(e)
# %%
momos2d_data = []
n = 986634
for r in momos2d_runs:
    try:
        summary = ast.literal_eval(r["summary"])
        config = ast.literal_eval(r["config"])
    except Exception as _:
        summary = r["summary"]
        config = r["config"]

    rows = config["quantization"]["rows"]
    cols = config["quantization"]["cols"]
    cap = config["quantization"]["capacity"]
    name = f"rows: {config['quantization']['rows']}"
    name += f" cols: {config['quantization']['cols']}"
    run_name = r["name"]
    momos2d_data.append(
        {
            "name": name,
            "run_name": r["name"],
            "val_acc": summary["val/acc"],
            "val_loss": summary["val/loss"],
            "train_acc": summary["train/acc"],
            "train_loss": summary["train/loss"],
            "test_acc": summary["test/acc"],
            "rows": rows,
            "cols": cols,
            "capacity": cap,
            "ras": n * 32 / momos_complexity(n, cap, rows * cols),
        }
    )
# %%
momos2d_df = pd.DataFrame(momos2d_data)
stats = momos2d_df.groupby("name")[
    ["val_acc", "val_loss", "train_acc", "train_loss", "test_acc", "test_acc"]
].agg(["mean", "std"])
# %%
stats = (
    momos2d_df.groupby(["name", "capacity"])["test_acc"]
    .agg(["mean", "std"])
    .reset_index()
)
stats = stats.sort_values(["name", "capacity"])

stats_ras = (
    momos2d_df.groupby(["name", "ras"])["test_acc"].agg(["mean", "std"]).reset_index()
)
stats_ras = stats_ras.sort_values(["name", "ras"])

# 3. Create the dictionary mapping name to ([capacities], [means], [stds])
result_dict = (
    stats.groupby("name")
    .apply(
        lambda x: (
            np.array(x["capacity"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)

result_dict_ras = (
    stats_ras.groupby("name")
    .apply(
        lambda x: (
            np.array(x["ras"].tolist()),
            np.array(x["mean"].tolist()),
            np.array(x["std"].tolist()),
        )
    )
    .to_dict()
)

# %%
fig = report.new_figure()

fig.axhline(
    data={"Baseline": 0.564},
    linestyle="--",
    color="black",
)


fig.plot_with_var(
    result_dict,
    "",
    symbol="o-",
    x_label="Capacity",
    y_label="Test Accuracy",
    logx=True,
)
# fig.show()
# %%
fig_ras = report.new_figure()

fig_ras.axhline(
    data={"Baseline": 0.564},
    linestyle="--",
    color="black",
)

fig_ras.plot_with_var(
    result_dict_ras,
    "",
    symbol="o-",
    x_label="Compression Ratio",
    y_label="Test Accuracy",
)
# fig_ras.show()

# %%
# Compression breakdown, picking only the Pareto-optimal momos2d configs -
# unlike the pareto-cv report (notebook/resnet_pareto_cv_analysis.py), there
# are many more runs here (repeated per (rows, cols, capacity) combo, several
# combos dominated), so aggregate to one point per config first and filter
# to the Pareto front before decomposing each surviving config's compressed
# size into motif/mosaic bits. Uses this notebook's own local
# momos_complexity_breakdown formula (n is a fixed constant here, not a live
# model) rather than compute_quantization_bits/build_compression_breakdown,
# which need an actual backbone - plot_compression_breakdown itself is the
# same generic function either way, no knowledge of where the bits come from.
config_stats = (
    momos2d_df.groupby(["rows", "cols", "capacity"])
    .agg(ras=("ras", "first"), test_acc_mean=("test_acc", "mean"))
    .reset_index()
)
config_stats["is_pareto_optimal"] = compute_pareto_mask(
    config_stats, x="ras", y="test_acc_mean"
)
pareto_momos2d = config_stats[config_stats["is_pareto_optimal"]]
print(f"{len(pareto_momos2d)} / {len(config_stats)} momos2d configs are Pareto-optimal")

breakdown_rows = []
for r in pareto_momos2d.itertuples():
    parts = momos_complexity_breakdown(n, r.capacity, r.rows * r.cols)
    dense_bits = n * 32
    k = momos2d_k(n, r.capacity, r.rows * r.cols)
    bar_label = f"MoMos\n({r.rows}, {r.cols})\n$k={k}$"
    for component, bits in parts.items():
        breakdown_rows.append(
            {
                "bar_label": bar_label,
                "component": component,
                "bits_fraction": bits / dense_bits,
            }
        )

breakdown_df = pd.DataFrame(breakdown_rows)
if breakdown_df.empty:
    print(
        "No Pareto-optimal momos2d configs found - skipping compression breakdown plot."
    )
else:
    tallest = breakdown_df.groupby('bar_label')['bits_fraction'].sum().max()
    print(f'Tallest breakdown bar: {tallest:.4f} of dense (y-axis capped at 0.4)')
    # Fixed to [0, 0.4], matching notebook/mlp_pareto_cv_analysis.py and
    # notebook/resnet_pareto_cv_analysis.py, so the breakdowns across all
    # three reports are measured against the same ruler. Anything taller
    # than 0.4 would be clipped silently - the print above is the guard.
    fig_breakdown = plot_compression_breakdown(breakdown_df, ylim=(0.0, 0.4))
    report.append_figures([fig_breakdown])

# %%
# report.save("momos2d_brief.pdf")

# %%
hyperparams = ["rows", "cols", "capacity"]
detail_params = pd.DataFrame({"rows": [2, 2, 2, 1], "cols": [1, 2, 4, 8]})
detail_params["capacity"] = 0.001
detail_run_df = momos2d_df.merge(detail_params, on=hyperparams, how="inner")
# %%
detail_runs = [
    api.runs(
        f"danesinoo-university-of-copenhagen/{project}",
        filters={"display_name": run["run_name"]},
    )[0]
    for run in tqdm(detail_run_df.to_dict("records"))
]
# %%
MOMOS2D_METRICS = [
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


# %%
grouped_runs = detail_run_df.groupby(hyperparams)["run_name"].apply(list).reset_index()
# %%
momos2d_best_runs_data = []
for run in tqdm(detail_runs):
    history = run.history(samples=500)
    assert type(history) is pd.DataFrame

    if history.empty:
        continue

    try:
        metrics = []
        metrics += [history[["epoch"]].dropna().drop_duplicates()]
        metrics += extract_columns(history, MOMOS2D_METRICS)

    except Exception as e:
        print(history.columns)
        print(e)
        break

    metrics = merge_dfs(metrics)
    momos2d_best_runs_data.append(
        {
            "name": run.name,
            "metrics": metrics,
            "config": run.config,
        }
    )
# %%
runs_summary = {}
for r in grouped_runs.to_dict("records"):
    metrics_by_epoch = defaultdict(lambda: defaultdict(list))
    for run in momos2d_best_runs_data:
        if run["name"] in r["run_name"]:
            for metric in MOMOS2D_METRICS:
                if metric in run["metrics"].columns:
                    clean_df = run["metrics"][["epoch", metric]].dropna()
                    for epoch, val in zip(clean_df["epoch"], clean_df[metric]):
                        metrics_by_epoch[metric][round(epoch, 4)].append(val)

    result = {}
    for metric in MOMOS2D_METRICS:
        if metric in metrics_by_epoch and metrics_by_epoch[metric]:
            epoch_dict = metrics_by_epoch[metric]
            sorted_epochs = sorted(epoch_dict.keys())
            means = []
            stds = []
            for epoch in sorted_epochs:
                vals = epoch_dict[epoch]
                means.append(np.mean(vals))
                stds.append(np.std(vals) if len(vals) > 1 else 0.0)
            result[metric] = (np.array(sorted_epochs), np.array(means), np.array(stds))
        else:
            result[metric] = (np.array([]), np.array([]), np.array([]))

    runs_summary[f"rows: {r['rows']} cols: {r['cols']}"] = result
# %%
tr_overview = {
    "val/acc": "Validation Accuracy",
    "val/loss": "Validation Loss",
    "train/acc": "Training Accuracy",
    "train/loss": "Training Loss",
}
report.training_overview(runs_summary, tr_overview)
# %%
metrics_overview = {
    "metrics/bdm_complexity": r"$\mathcal{K}_\mathrm{BDM}$",
    "metrics/gzip_compression_rate": r"$\mathcal{r}_{Gzip}$",
    "metrics/bz2_compression_rate": r"$\mathcal{r}_\mathrm{BZ2}$",
    "metrics/lzma_compression_rate": r"$\mathcal{r}_\mathrm{LZMA}$",
    "metrics/weight_l2": r"$L_2$ Norm",
    "metrics/sparsity": "Sparsity",
    "quant/distortion": "Distortion",
    "quant/num_changed_weights": "# Changed Weights",
}

report.metrics_vs_accuracy(runs_summary, metrics_overview)
# %%
report.save("momos2_overview.pdf")
