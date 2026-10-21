"""Soft MoMos: a fixed-per-epoch mosaic enforced by a differentiable penalty.

Training: every epoch a mosaic ``M`` is drawn from the CURRENT weights (random
blocks as motifs, nearest assignment; the weights are not modified). Each batch
adds ``lam * (1/n) * sum_i ||b_i - Z[M_i]||^2`` to the loss, where ``Z`` are the
(detached) group means of the blocks. After training, :func:`finalize_soft_momos`
runs k-means++ with ``final_k`` clusters on all blocks and snaps every block to
its centroid, giving the stored ``{Z, M}`` description.

Padding rule: zero padding added by ``tensor2D_to_blocks`` participates in the
group means and penalty like any block entry; the normaliser ``n`` is the number
of REAL weights.
"""

import torch
from sklearn.cluster import KMeans, MiniBatchKMeans

from .block_utils import (
    _assign_blocks,
    _initialize_motifs,
    _nearest_motifs_chunked,
)
from .momos2d import _get_model_blocks2D, _reconstruct_and_apply


def draw_mosaic(model, rows, cols, k, force_zero=True, chunk_size=None):
    """Draw ``Z0`` from the current blocks and assign every block to its nearest.

    Returns ``(M, Z0, layer_specs)``; the model weights are left untouched.
    """
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        k_eff = max(1, min(int(k), blocks.size(0)))
        Z0 = _initialize_motifs(blocks, k_eff, rows * cols, force_zero)
        M, _ = _assign_blocks(blocks, Z0, chunk_size, False, "soft_momos", None, None)
    return M, Z0, layer_specs


def group_means(blocks, M, k, force_zero=True, counts=None):
    """Per-motif mean of ``blocks`` in O(m) (``index_add_`` + ``bincount``).

    Empty groups give zero rows; row 0 is pinned to 0 when ``force_zero``.
    ``counts`` (``bincount(M, minlength=k)``) may be passed to skip recomputing it
    (it only depends on ``M``).
    """
    if counts is None:
        counts = torch.bincount(M, minlength=k)
    Z = torch.zeros(k, blocks.size(1), dtype=blocks.dtype, device=blocks.device)
    Z.index_add_(0, M, blocks)
    Z /= counts.clamp(min=1).unsqueeze(1).to(Z.dtype)
    if force_zero:
        Z[0] = 0.0
    return Z


def soft_momos_penalty(
    model, M, rows, cols, k, force_zero=True, counts=None, with_stats=True
):
    """``(1/n) * sum_i ||b_i - Z[M_i]||^2`` with ``Z`` the detached group means.

    Differentiable wrt the live parameters (gradient ``2 (w - Z[M]) / n``; detaching
    ``Z`` is exact since deviations from a group mean sum to zero). Returns
    ``(penalty, stats)``; stats values are 0-dim tensors (no device sync) and
    ``stats`` is ``{}`` when ``with_stats`` is False.
    """
    blocks, layer_specs = _get_model_blocks2D(model, int(rows), int(cols), detach=False)
    n = sum(spec[2] for spec in layer_specs)
    if counts is None:
        counts = torch.bincount(M, minlength=k)
    with torch.no_grad():
        Z = group_means(blocks.detach(), M, k, force_zero, counts)
    sq = (blocks - Z[M]).square().sum()
    stats = {}
    if with_stats:
        with torch.no_grad():
            stats = {
                "rel_distortion": sq.detach() / blocks.detach().square().sum(),
                "active_motifs": (counts > 0).sum(),
            }
    return sq / n, stats


def project_fixed_mosaic(model, M, rows, cols, k, force_zero=True):
    """Write ``Z[M]`` (group means of the current blocks) into the params in place."""
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        counts = torch.bincount(M, minlength=k)
        Z = group_means(blocks, M, k, force_zero, counts)
        distortion, changed = _reconstruct_and_apply(
            layer_specs, blocks, Z[M], rows, cols
        )
    return {
        "distortion": float(distortion),
        "num_changed_weights": int(changed),
        "num_motifs_used": int((counts > 0).sum()),
    }


def finalize_soft_momos(
    model,
    rows,
    cols,
    final_k,
    force_zero=True,
    seed=0,
    max_iter=100,
    n_init="auto",
    backend="minibatch",
    chunk_size=None,
):
    """k-means++ with ``final_k`` clusters on all blocks, then snap blocks in place."""
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        k_eff = max(1, min(int(final_k), blocks.size(0)))

        common = dict(
            n_clusters=k_eff,
            init="k-means++",
            n_init=n_init,
            max_iter=int(max_iter),
            random_state=int(seed),
        )
        if backend == "exact":
            km = KMeans(**common)
        elif backend == "minibatch":
            km = MiniBatchKMeans(batch_size=max(1024, k_eff), **common)
        else:
            raise ValueError(f"Unknown finalize backend: {backend!r}")
        km.fit(blocks.cpu().numpy())
        motifs = torch.from_numpy(km.cluster_centers_).to(blocks)
        del km

        if force_zero:
            motifs[motifs.norm(dim=1).argmin()] = 0.0

        nearest = _nearest_motifs_chunked(blocks, motifs, chunk_size=chunk_size)
        distortion, changed = _reconstruct_and_apply(
            layer_specs, blocks, motifs[nearest], rows, cols
        )
        counts = torch.bincount(nearest, minlength=k_eff)

    return {
        "distortion": float(distortion),
        "num_changed_weights": int(changed),
        "num_motifs_used": int((counts > 0).sum()),
        "motif_counts": counts,
        "_motifs": motifs,
        "_nearest": nearest,
        "_layer_specs": layer_specs,
    }


def quantize_soft_momos(model, quant_cfg):
    """Registry entry: the post-training finalisation step of soft_momos."""
    fin = quant_cfg.get("finalize") or {}
    return finalize_soft_momos(
        model,
        rows=quant_cfg["rows"],
        cols=quant_cfg["cols"],
        final_k=quant_cfg["final_k"],
        force_zero=quant_cfg.get("force_zero", True),
        seed=fin.get("seed", 0),
        max_iter=fin.get("max_iter", 100),
        n_init=fin.get("n_init", "auto"),
        backend=fin.get("backend", "minibatch"),
        chunk_size=quant_cfg.get("chunk_size"),
    )
