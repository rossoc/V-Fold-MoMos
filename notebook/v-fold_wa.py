# %% [markdown]
# Weight-distribution ("motif") report for a hand-picked `eval_pareto_cv.py`
# config, across its `dataset.n_folds` CV folds - the fold-wise counterpart
# to `momos2d_wa.py` (which instead walks one run's checkpoint versions over
# training). Reuses `plot_weights_2d` unchanged per fold checkpoint (it only
# needs a ckpt path + rows/cols, same as momos2d_wa.py), then adds one extra
# plot per config once every fold is in: "Motifs by number of folds" -
# analogous to `plot_weights_2d`'s own "Motifs by number of layers", but
# counting how many of the *folds'* (not layers') final weights each unique
# motif shows up in.
# %%
import os

import torch
import wandb
from tqdm import tqdm

from src.view import Report
from src.view.training_analysis import pareto_cv_run_names
from src.view.weight_distribution import (
    extract_blocks_2d,
    load_model,
    plot_motifs_by_fold_spread,
    plot_weights_2d,
)

ENTITY = "danesinoo-university-of-copenhagen"
PROJECT = "mlp_pareto_verification"
N_FOLDS = 5

api = wandb.Api()

# Placeholder pick - inspect notebook/mlp_pareto_cv_analysis.py's Pareto
# front first, then hand-edit to the (method, trial_number) config(s) you
# actually want a weight report for. `method` must be the raw method string
# (e.g. "momos2d", "hierarchical_momos2d", "fold_momos") - only these three
# have the rows/cols/capacity a 2D motif plot needs; "qat"/"none" don't.
CONFIGS = [
    ("hierarchical_momos2d", 0),
]

_2D_METHODS = {"momos2d", "hierarchical_momos2d", "fold_momos"}

for method, trial_number in CONFIGS:
    if method not in _2D_METHODS:
        print(f"Skipping {method!r} (t{trial_number}): no 2D rows/cols to plot")
        continue

    report = Report()
    fold_names = pareto_cv_run_names(method, trial_number, n_folds=N_FOLDS)
    rows = cols = cap = None
    fold_all_blocks = []  # one all-layers `all_blocks` tensor per fold, for the plot below

    for fold_idx, run_name in enumerate(tqdm(fold_names)):
        matches = api.runs(f"{ENTITY}/{PROJECT}", filters={"display_name": run_name})
        if not matches:
            print(f"  fold {fold_idx}: no WandB run named {run_name!r} yet - skipping")
            continue
        run = matches[0]

        q_cfg = run.config.get("quantization", {})
        # hierarchical_momos2d/fold_momos nest under "primary" (+ "secondary");
        # momos2d has rows/cols/capacity directly at the top level.
        primary = q_cfg.get("primary", q_cfg)
        rows, cols, cap = primary["rows"], primary["cols"], primary["capacity"]

        root = os.path.join("artifacts", run_name)
        ckpt_path = None
        if os.path.isdir(root):
            existing = [f for f in os.listdir(root) if f.endswith(".ckpt")]
            if existing:
                ckpt_path = os.path.join(root, sorted(existing)[0])

        if ckpt_path is None:
            artifacts = list(run.logged_artifacts())
            model_artifacts = [a for a in artifacts if a.type == "model"] or artifacts
            if not model_artifacts:
                print(
                    f"  fold {fold_idx}: no checkpoint artifact logged for {run_name!r} - skipping"
                )
                continue
            artifact_dir = model_artifacts[-1].download(root=root)
            ckpts = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
            if not ckpts:
                print(
                    f"  fold {fold_idx}: downloaded artifact has no .ckpt file - skipping"
                )
                continue
            ckpt_path = os.path.join(artifact_dir, sorted(ckpts)[0])

        report.append_figures(
            plot_weights_2d(ckpt_path, rows, cols, f"fold {fold_idx}")
        )

        model = load_model(ckpt_path)
        blocks, _ = extract_blocks_2d(model, rows, cols)
        fold_all_blocks.append(torch.cat(blocks, dim=0))

    for figure in report.figures:
        figure.fontsize = 13

    if fold_all_blocks:
        report.append_figures(
            [plot_motifs_by_fold_spread(fold_all_blocks, f"{method} t{trial_number}")]
        )

    if rows is not None:
        report.save(f"v-fold_wa_{method}_t{trial_number}_r{rows}_c{cols}_cap{cap}.pdf")
        print(
            f"Saved {report.output_dir / f'v-fold_wa_{method}_t{trial_number}_r{rows}_c{cols}_cap{cap}.pdf'}"
        )

    # Release every figure before the next config's weight analysis starts,
    # so matplotlib's figure registry doesn't accumulate live figures (and
    # their weight tensors) across configs and exhaust RAM.
    report.close()
