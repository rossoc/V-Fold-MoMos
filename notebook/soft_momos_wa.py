# %% [markdown]
# Weight-distribution ("motif") report along the training of the long soft-MoMos
# runs (`soft-momos` WandB project) - the soft-MoMos counterpart to `momos2d_wa.py`.
# Needs no WandB: the runs live in `outputs/cifar10_mlp/<id>_*/checkpoints/`.
#
# The periodic `epoch-epoch=N.ckpt` hold the *raw* weights, which are not on the
# mosaic (every block is ~unique), so the motifs are recomputed from
# `epoch_motifs-epoch=N.pt` instead: its `weights` are the weights as validated,
# i.e. the projection `Z[M]` onto that epoch's mosaic. Frequencies are the unique
# blocks of those weights, counted afresh (padded blocks are snapped back to their
# motif through the stored mosaic `M`, see `plot_weights_2d_from_tensors`).
# The last page is the final codebook (`final_motifs.pt`: motifs[assignments]).
# %%
import glob
import re

import torch
from tqdm import tqdm

from src.view import Report
from src.view.weight_distribution import plot_weights_2d_from_tensors

OUTPUTS = "outputs/cifar10_mlp"
RUN_IDS = [12, 13]


def final_weights(final):
    """Rebuild the per-tensor weights from `final_motifs.pt` (motifs[assignments])."""
    blocks = final["motifs"][final["assignments"]]
    weights, start = [], 0
    for _, n_blocks, shape in final["layer_specs"]:
        chunk = blocks[start : start + n_blocks].reshape(-1)
        weights.append(chunk[: shape[0] * shape[1]].reshape(shape))
        start += n_blocks
    return weights


for run_id in RUN_IDS:
    (run_dir,) = glob.glob(f"{OUTPUTS}/{run_id}_*")
    ckpts = sorted(
        glob.glob(f"{run_dir}/checkpoints/epoch_motifs-epoch=*.pt"),
        key=lambda p: int(re.search(r"epoch=(\d+)", p).group(1)),
    )
    report = Report()
    rows = cols = None
    for path in tqdm(ckpts, desc=f"run {run_id}"):
        m = torch.load(path, weights_only=False, map_location="cpu")
        rows, cols = int(m["rows"]), int(m["cols"])
        figs = plot_weights_2d_from_tensors(
            m["weights"], rows, cols, m["epoch"], ckpt_path=path, M=m["M"]
        )
        report.append_figures(figs)

    final_path = f"{run_dir}/checkpoints/final_motifs.pt"
    final = torch.load(final_path, weights_only=False, map_location="cpu")
    report.append_figures(
        plot_weights_2d_from_tensors(
            final_weights(final),
            final["rows"],
            final["cols"],
            "final",
            final_path,
            M=final["assignments"],
        )
    )
    for figure in report.figures:
        figure.fontsize = 20
    name = f"soft_momos_wa_run{run_id}_r{rows}_c{cols}.pdf"
    report.save(name)
    print(f"Saved {report.output_dir / name}")
    report.close()
