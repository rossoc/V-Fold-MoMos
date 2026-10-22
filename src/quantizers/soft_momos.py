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

Two optional extensions (``soft_momos_plus``) live on the same code path:

* warm start (:func:`warm_start_mosaic`): the per-epoch mosaic is a Lloyd step
  from the previous epoch's mosaic instead of a fresh random draw;
* frozen phase (:func:`freeze_codebook`): the mosaic is fixed and the weights are
  *parametrized* as ``Z[M]``, so the codebook ``Z`` is the only trainable tensor.
  While parametrized, ``iter_trainable_params`` yields ``Z``, not the weights;
  read the weights through :func:`materialized_weights`.
"""

from contextlib import contextmanager

import torch
import torch.nn as nn
from torch.nn.utils import parametrize
from sklearn.cluster import KMeans, MiniBatchKMeans

from .block_utils import (
    _assign_blocks,
    _initialize_motifs,
    _nearest_motifs_chunked,
)
from .momos2d import _get_model_blocks2D, _reconstruct_and_apply, blocks_to_tensor2D

# ``soft_momos_plus`` is ``soft_momos`` with ``warm_start``/``freeze_fraction`` on.
SOFT_MOMOS_METHODS = ("soft_momos", "soft_momos_plus")


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


def finalize_from_codebook(model, Z, M, rows, cols, force_zero=True, chunk_size=None):
    """Finalize a model whose weights are ``Z[M]`` (after a frozen phase) without k-means.

    The codebook and mosaic already describe the weights exactly, so they are
    adopted as is (row 0 zeroed under ``force_zero``, as in the parametrization).
    Returns the stats dict of :func:`finalize_soft_momos`; ``distortion`` is
    measured against the current weights (0 when they are bit-exactly ``Z[M]``).
    """
    rows, cols = int(rows), int(cols)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        motifs = Z.detach().to(blocks).clone()
        if force_zero:
            motifs[0] = 0.0
        nearest = M.to(blocks.device)
        distortion, changed = _reconstruct_and_apply(
            layer_specs, blocks, motifs[nearest], rows, cols
        )
        counts = torch.bincount(nearest, minlength=motifs.size(0))
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


def warm_start_mosaic(
    model,
    M_prev,
    rows,
    cols,
    k,
    force_zero=True,
    reseed_dead=True,
    lloyd_iters=1,
    chunk_size=None,
):
    """``lloyd_iters`` Lloyd steps on the CURRENT blocks, starting from ``M_prev``.

    Each step: group means under the current mosaic, optional reseeding of dead
    motifs (no member; motif 0 excluded under ``force_zero``) with distinct random
    blocks, then nearest assignment. Weights are left untouched. Returns
    ``(M, Z, layer_specs, stats)`` with ``Z`` the codebook ``M`` was assigned
    against and python-number ``stats`` (``mosaic_changed`` relative to
    ``M_prev``; ``dead_motifs`` of the first step, before reseeding).
    """
    rows, cols, k = int(rows), int(cols), int(k)
    with torch.no_grad():
        blocks, layer_specs = _get_model_blocks2D(model, rows, cols)
        M, dead_first = M_prev, None
        for _ in range(int(lloyd_iters)):
            counts = torch.bincount(M, minlength=k)
            Z = group_means(blocks, M, k, force_zero, counts)
            dead = counts == 0
            if force_zero:
                dead[0] = False
            n_dead = int(dead.sum())
            dead_first = n_dead if dead_first is None else dead_first
            if reseed_dead and n_dead:
                pick = torch.randperm(blocks.size(0), device=blocks.device)[:n_dead]
                Z[dead] = blocks[pick]
            M = _nearest_motifs_chunked(blocks, Z, chunk_size=chunk_size)
        changed = float((M != M_prev).float().mean())
    return M, Z, layer_specs, {"mosaic_changed": changed, "dead_motifs": dead_first}


class CodebookTensor(nn.Module):
    """Parametrization ``weight = blocks_to_tensor2D(Z[M_slice])`` (ignores the original).

    ``Z`` is the ``nn.Parameter`` shared by all parametrized tensors; the gather's
    backward scatter-adds the weight gradients into ``Z`` (O(n)). Under
    ``force_zero`` row 0 is masked in the (small) codebook, not in the gathered
    blocks, so ``Z[0]`` gets no gradient and never reaches the weights.
    """

    is_codebook_tensor = True  # duck-typed: ``quantizers`` and ``src.quantizers`` coexist

    def __init__(self, Z, M_slice, shape, rows, cols, force_zero, out_shape):
        super().__init__()
        self.Z = Z
        self.register_buffer("M", M_slice)
        keep = torch.ones(Z.size(0), 1, dtype=Z.dtype, device=Z.device)
        if force_zero:
            keep[0] = 0.0
        self.register_buffer("keep", keep)
        self.shape, self.rows, self.cols = tuple(shape), rows, cols
        self.out_shape = tuple(out_shape)

    def forward(self, _original):
        blocks = (self.Z * self.keep)[self.M]
        return blocks_to_tensor2D(blocks, self.shape, self.rows, self.cols).reshape(
            self.out_shape
        )


def _param_owners(model):
    """``{id(param): (module, attribute name)}`` for every direct parameter."""
    return {
        id(p): (mod, name)
        for mod in model.modules()
        for name, p in mod._parameters.items()
        if p is not None
    }


def freeze_codebook(model, M, rows, cols, k, force_zero=True):
    """Make ``Z`` (group means of the current blocks under ``M``) the only trainable
    tensor: every quantized weight becomes ``Z[M]`` through a :class:`CodebookTensor`.

    Returns ``(Z, handles)``; pass ``handles`` to :func:`unfreeze_codebook`.
    """
    rows, cols, k = int(rows), int(cols), int(k)
    with torch.no_grad():
        blocks, specs = _get_model_blocks2D(model, rows, cols)
        Z = nn.Parameter(group_means(blocks, M, k, force_zero))
    owners = _param_owners(model)
    handles, offset = [], 0
    for param, n_blocks, _, shape in specs:
        mod, name = owners[id(param)]
        param.requires_grad_(False)
        parametrize.register_parametrization(
            mod,
            name,
            CodebookTensor(Z, M[offset : offset + n_blocks], shape, rows, cols,
                           force_zero, param.shape),
            unsafe=True,
        )
        mod.parametrizations[name].original.requires_grad_(False)
        handles.append((mod, name))
        offset += n_blocks
    return Z, handles


def unfreeze_codebook(model, handles):
    """Inverse of :func:`freeze_codebook` (idempotent): plain weights ``== Z[M]``."""
    for mod, name in handles:
        if parametrize.is_parametrized(mod, name):
            parametrize.remove_parametrizations(mod, name, leave_parametrized=True)
            getattr(mod, name).requires_grad_(True)


def _codebook_sites(model):
    """Yield ``(qualified module name, module, param name, ParametrizationList)``."""
    for mod_name, mod in model.named_modules():
        if parametrize.is_parametrized(mod):
            for name, plist in mod.parametrizations.items():
                if getattr(plist[0], "is_codebook_tensor", False):
                    yield mod_name, mod, name, plist


def materialized_state(model):
    """``{plain state_dict key: materialized weight}`` of the parametrized tensors."""
    with torch.no_grad():
        return {
            f"{mod_name}.{name}" if mod_name else name: getattr(mod, name).clone()
            for mod_name, mod, name, _ in _codebook_sites(model)
        }


@contextmanager
def materialized_weights(model):
    """Read-only view of a parametrized model: inside, ``iter_trainable_params``
    yields one tensor per quantized tensor holding its materialized ``Z[M]``
    values (the originals, temporarily rebound; ``Z`` is hidden). Everything is
    restored on exit, also on exceptions; a no-op for plain models. Writes made
    inside are not persisted.
    """
    sites = list(_codebook_sites(model))
    if not sites:
        yield model
        return
    saved, codebooks = [], {}
    try:
        with torch.no_grad():
            for _, mod, name, plist in sites:
                orig = plist.original
                saved.append((orig, orig.data, orig.requires_grad))
                orig.data = getattr(mod, name)  # one materialisation per tensor
                orig.requires_grad_(True)
                Z = plist[0].Z
                codebooks.setdefault(id(Z), (Z, Z.requires_grad))
        for Z, _ in codebooks.values():
            Z.requires_grad_(False)
        yield model
    finally:
        for orig, data, requires_grad in saved:
            orig.data = data
            orig.requires_grad_(requires_grad)
        for Z, requires_grad in codebooks.values():
            Z.requires_grad_(requires_grad)
