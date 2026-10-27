# %% [markdown]
# Weight-distribution ("motif") report for the `soft_momos_plus` CV-verified
# Pareto configs, across their CV folds - the soft-MoMos counterpart to
# `v-fold_wa.py`. Unlike that notebook it needs no WandB: `eval_pareto_cv.py`
# keeps each fold's finalized (k-means-snapped) `best.ckpt` locally under
# `outputs/<prefix>/soft_momos_plus_t<trial>/fold<i>/*/checkpoints/`, and
# rows/cols come from the `*_cv_test.yaml` written next to the Pareto summary.
#
# Per config: `plot_weights_2d` on every fold checkpoint, plus a final
# "Motifs by number of folds" plot (how many folds' weights contain each
# unique motif).
# %%
import glob
import os

import torch
import yaml
from tqdm import tqdm

from src.view import Report
from src.view.weight_distribution import (
    extract_blocks_2d,
    load_model,
    plot_motifs_by_fold_spread,
    plot_weights_2d,
)

OUTPUTS = "outputs"
# (results dir with trial*_cv_test.yaml, eval_pareto_cv `prefix` template)
STUDIES = [
    ("outputs/pareto_cv/mlp_soft_200ep", "pareto_cv_200ep_trial{t}"),
    ("outputs/pareto_cv/mlp_soft_l1_200ep", "pareto_cv_l1_200ep_trial{t}"),
]
METHOD = "soft_momos_plus"
N_FOLDS = 5
# Restrict to some trial numbers, e.g. {11, 21}; None = every config found.
ONLY_TRIALS = None

for results_dir, prefix_tmpl in STUDIES:
    for cv_file in sorted(glob.glob(f"{results_dir}/trial*_cv_test.yaml")):
        with open(cv_file) as f:
            entry = yaml.safe_load(f)["pareto_cv_results"][0]
        trial = entry["trial_number"]
        if ONLY_TRIALS is not None and trial not in ONLY_TRIALS:
            continue
        rows, cols = int(entry["params"]["smp_rows"]), int(entry["params"]["smp_cols"])
        study = os.path.basename(results_dir)

        report = Report()
        fold_all_blocks = []  # one all-layers `all_blocks` tensor per fold
        for fold_idx in tqdm(range(N_FOLDS), desc=f"{study} t{trial}"):
            ckpts = glob.glob(
                f"{OUTPUTS}/{prefix_tmpl.format(t=trial)}/{METHOD}_t{trial}"
                f"/fold{fold_idx}/*/checkpoints/best.ckpt"
            )
            if not ckpts:
                print(f"  fold {fold_idx}: no best.ckpt for t{trial} - skipping")
                continue
            ckpt_path = sorted(ckpts)[0]
            report.append_figures(
                plot_weights_2d(ckpt_path, rows, cols, f"fold {fold_idx}")
            )
            blocks, _ = extract_blocks_2d(load_model(ckpt_path), rows, cols)
            fold_all_blocks.append(torch.cat(blocks, dim=0))

        for figure in report.figures:
            figure.fontsize = 13
        if fold_all_blocks:
            report.append_figures(
                [plot_motifs_by_fold_spread(fold_all_blocks, f"{METHOD} t{trial}")]
            )
            name = f"soft_momos_v-fold_wa_{study}_t{trial}_r{rows}_c{cols}.pdf"
            report.save(name)
            print(f"Saved {report.output_dir / name}")
        report.close()
