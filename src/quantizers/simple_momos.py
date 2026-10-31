"""Simple MoMos: a persistent codebook refined by one Lloyd step per epoch.

Sits between ``momos2d`` and ``soft_momos``:

* like ``momos2d`` the weights are hard-projected onto the mosaic once per epoch
  and there is no per-batch cost (no penalty, plain CE training);
* unlike ``momos2d`` the motifs are not redrawn at random every epoch: the codebook
  ``Z`` persists and is updated as the group means of the partition induced by
  the mosaic (a k-means/Lloyd step), so it tracks the weights the way
  ``soft_momos``'s group means do, but without the per-batch L2 penalty.

Algorithm::

    Z <- random blocks of the initial weights (row 0 = 0 under force_zero)
    for epoch:
        train the network
        M <- nearest(blocks(w), Z)        # recompute the mosaic from the motifs
        Z <- group_means(blocks(w), M)    # update the motifs (empty groups kept/reseeded)
        w <- Z[M]                         # hard projection

After the last epoch the weights are exactly ``Z[M]``, so the stored codebook is
the trained ``Z`` (``k`` motifs) and no k-means finalisation is needed.
"""

import torch

from .block_utils import _initialize_motifs, _nearest_motifs_chunked
from .momos2d import _get_model_blocks2D, _reconstruct_and_apply
from .soft_momos import group_means

SIMPLE_MOMOS_METHODS = ("simple_momos",)


def init_codebook(model, rows, cols, k, force_zero=True):
    """Initial codebook: ``k`` distinct random blocks of the current weights.

    Row 0 is the zero motif under ``force_zero``. ``k`` is clamped to the number
    of blocks. Weights are left untouched.
    """
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, _ = _get_model_blocks2D(model, rows, cols)
        k_eff = max(1, min(int(k), blocks.size(0)))
        return _initialize_motifs(blocks, k_eff, rows * cols, force_zero).clone()


def _nearest(blocks, Z, chunk_size):
    if Z.size(0) == 1:
        return torch.zeros(blocks.size(0), dtype=torch.long, device=blocks.device)
    return _nearest_motifs_chunked(blocks, Z, chunk_size=chunk_size)


def simple_momos_step(
    model,
    Z,
    rows,
    cols,
    force_zero=True,
    reseed_dead=True,
    chunk_size=None,
    M_prev=None,
):
    """One Lloyd step from codebook ``Z``, then project ``w <- Z[M]``.

    Assigns every block to its nearest motif (``M``), replaces every motif by the
    mean of its group and writes ``Z[M]`` into the weights: a single nearest-motif
    search per call. A motif with no members keeps its previous value, or is
    reseeded with a random block when ``reseed_dead`` (motif 0 is never reseeded
    under ``force_zero``); dead motifs are not referenced by ``M``, so reseeding
    does not affect the projection and the weights are exactly ``Z[M]`` on return.
    A reseeded motif can win blocks from the next call on.

    Returns ``(Z, M, stats)``. ``stats`` has the ``momos2D`` keys (``distortion``,
    ``num_changed_weights``, ``num_motifs_used``, ``motif_counts``, ``_motifs``,
    ``_nearest``, ``_layer_specs``) plus ``dead_motifs`` (empty groups, before
    reseeding) and ``mosaic_changed`` (fraction of blocks whose assignment differs
    from ``M_prev``; 1.0 when ``M_prev`` is None).
    """
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        Z = Z.detach().to(blocks).clone()
        k = Z.size(0)
        if force_zero:
            Z[0] = 0.0
        M = _nearest(blocks, Z, chunk_size)
        counts = torch.bincount(M, minlength=k)
        means = group_means(blocks, M, k, force_zero, counts)
        dead = counts == 0
        if force_zero:
            dead[0] = False
        n_dead = int(dead.sum())
        # Live groups move to their mean; dead groups keep their old motif.
        Z = torch.where(dead.unsqueeze(1), Z, means)
        if reseed_dead and n_dead:
            pick = torch.randperm(blocks.size(0), device=blocks.device)[:n_dead]
            Z[dead] = blocks[pick]
        distortion, changed = _reconstruct_and_apply(
            layer_specs, blocks, Z[M], rows, cols
        )
        if M_prev is None or M_prev.shape != M.shape:
            mosaic_changed = 1.0
        else:
            mosaic_changed = float((M != M_prev.to(M.device)).float().mean())

    stats = {
        "distortion": float(distortion),
        "num_changed_weights": int(changed),
        "num_motifs_used": int((counts > 0).sum()),
        "dead_motifs": n_dead,
        "mosaic_changed": mosaic_changed,
        "motif_counts": counts,
        "_motifs": Z,
        "_nearest": M,
        "_layer_specs": layer_specs,
    }
    return Z, M, stats


def quantize_simple_momos(model, quant_cfg):
    """Registry entry: one stateless step (fresh random codebook + Lloyd step + projection).

    During training the ``QuantizationCallback`` keeps ``Z`` across epochs and
    calls :func:`simple_momos_step` directly; this entry is for standalone use.
    """
    rows, cols = quant_cfg["rows"], quant_cfg["cols"]
    force_zero = bool(quant_cfg.get("force_zero", True))
    Z = init_codebook(model, rows, cols, quant_cfg["k"], force_zero)
    _, _, stats = simple_momos_step(
        model,
        Z,
        rows,
        cols,
        force_zero=force_zero,
        reseed_dead=bool(quant_cfg.get("reseed_dead", True)),
        chunk_size=quant_cfg.get("chunk_size"),
    )
    return stats


def codebook_from_weights(model, rows, cols):
    """Recover ``{Z, M}`` exactly from weights that are already a mosaic.

    The distinct blocks are the codebook and ``M`` indexes them. This is used after
    training, when the loaded (best) checkpoint may come from an earlier epoch than
    the callback's in-memory ``Z``. Returns the stats dict of
    :func:`simple_momos_step`. Weights are not modified.
    """
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        Z, M = torch.unique(blocks, dim=0, return_inverse=True)
        counts = torch.bincount(M, minlength=Z.size(0))
    return {
        "distortion": 0.0,
        "num_changed_weights": 0,
        "num_motifs_used": int(Z.size(0)),
        "motif_counts": counts,
        "_motifs": Z,
        "_nearest": M,
        "_layer_specs": layer_specs,
    }
