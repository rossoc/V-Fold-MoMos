"""Tests that define the API of the ``soft_momos`` quantization method.

These tests were written before the implementation (TDD). They are the single
source of truth for the API; the implementer should make exactly these pass.

================================================================================
API CONTRACT
================================================================================

Module ``src/quantizers/soft_momos.py``
---------------------------------------

Block layout everywhere: blocks are the concatenation, in
``iter_trainable_params(model)`` order, of ``tensor2D_to_blocks(param, rows, cols)``
for every trainable param -- i.e. exactly ``_get_model_blocks2D(model, rows, cols)``
(zero padding on the right/bottom of every 2D slice; 1D params are treated as
``(1, n)``). ``s = rows * cols``. ``m`` = total number of blocks.
``layer_specs`` is the list of ``(param, n_blocks, n_params, shape)`` tuples
returned by ``_get_model_blocks2D``.

PADDING RULE (decided here): padded positions are treated *as-is* -- the zero
padding introduced by ``tensor2D_to_blocks`` participates in the group means and
in the penalty sum like any other block entry (consistent with ``momos2D``,
which clusters padded blocks too). The normaliser ``n`` is the number of REAL
trainable weights (``sum(p.numel())``), *not* the padded count. The gradient wrt
every real weight is still exactly ``2 * (w - Z[M]) / n`` because padded entries
are constants.

``draw_mosaic(model, rows, cols, k, force_zero=True, chunk_size=None)
    -> (M, Z0, layer_specs)``
    * ``k_eff = min(k, m)``. ``Z0`` is ``(k_eff, s)``: ``Z0[0] == 0`` when
      ``force_zero`` and the remaining rows are blocks of the CURRENT weights drawn
      uniformly without replacement (``_initialize_motifs`` semantics, uses the
      global torch RNG so ``torch.manual_seed`` makes it deterministic).
    * ``M`` is a ``torch.long`` tensor of shape ``(m,)`` with ``M[i] = argmin_j
      ||b_i - Z0[j]||^2``. The assignment MUST go through the existing chunked
      helper ``block_utils._nearest_motifs_chunked`` (directly or via
      ``_assign_blocks``) so peak memory is bounded by ``chunk_size`` (MB, forwarded
      verbatim) instead of a full ``m x k`` distance matrix.
    * Must NOT modify the model weights. No gradient tracking needed.

``group_means(blocks, M, k, force_zero=True) -> Z``
    * ``Z`` is ``(k, s)``, same dtype/device as ``blocks``.
      ``Z[j] = mean{ blocks[i] : M[i] == j }``; a ``j`` with no members gives a
      zero row (never NaN); ``Z[0]`` is pinned to exactly 0 when ``force_zero``
      even if group 0 has members. ``k`` may exceed ``M.max() + 1`` (extra rows
      are zero).
    * Must be O(m) (``index_add_`` / ``bincount``): no python loop over motifs,
      no ``k x m`` matrix. 200k blocks with k=50k must run well under 1 s on CPU.

``soft_momos_penalty(model, M, rows, cols, k, force_zero=True) -> (penalty, stats)``
    * ``penalty = (1 / n) * sum_i ||b_i - Z[M_i]||^2`` with ``Z = group_means(...)``
      of the CURRENT blocks, detached; ``b_i`` built from the live params (with
      grad). ``penalty`` is a 0-dim differentiable tensor; ``d penalty / d w ==
      2 * (w - Z[M]) / n`` at every real weight position.
    * ``stats`` is a dict containing at least ``"rel_distortion"``
      (= ``sum_i ||b_i - Z[M_i]||^2 / sum_i ||b_i||^2``) and ``"active_motifs"``
      (= number of ``j`` with at least one member). Values are python numbers or
      0-dim tensors (a tensor is allowed so a per-batch call need not force a
      device sync; the caller converts with ``float()`` when logging).
    * Must not allocate a ``k x m`` matrix: 200k blocks with k=50k, forward +
      backward, must run in well under 2 s on CPU.

``project_fixed_mosaic(model, M, rows, cols, k, force_zero=True) -> stats``
    * Writes ``Z[M]`` (group means of the current blocks) into the params in place
      through ``blocks_to_tensor2D`` (padding dropped).
    * ``stats`` = ``{"distortion": sum over real weights of (before - after)^2,
      "num_changed_weights": count(before != after),
      "num_motifs_used": number of motif ids with members}`` (python numbers).
    * Applying it twice with the same ``M`` is idempotent up to float rounding
      of the group means (not required to be bit-exact).

``finalize_soft_momos(model, rows, cols, final_k, force_zero=True, seed=0,
                      max_iter=100, n_init="auto", backend="minibatch") -> stats``
    * k-means++ (``backend="exact"`` -> ``sklearn.cluster.KMeans``,
      ``"minibatch"`` -> ``MiniBatchKMeans``, ``random_state=seed``) on ALL blocks
      with ``k_eff = n_clusters = min(final_k, m)``. If ``force_zero``, the
      centroid nearest to 0 is snapped to exact 0. Each block is replaced in
      place by its nearest centroid (``_nearest_motifs_chunked``), written back
      through ``blocks_to_tensor2D``. Extra keyword arguments with defaults (e.g.
      ``chunk_size=None``) are allowed.
    * Returns ``{"distortion", "num_changed_weights", "num_motifs_used" (<= k_eff,
      number of centroids with >= 1 block), "motif_counts" (long, shape
      ``(k_eff,)``, sums to m), "_motifs" (``(k_eff, s)`` centroids), "_nearest"
      (long ``(m,)``), "_layer_specs"}``. ``param == blocks_to_tensor2D(
      _motifs[_nearest][slice])`` bit-exactly. All keys not in
      ``utils.callbacks._QUANT_LOG_SKIP`` are python numbers.
    * Same ``seed`` + same weights -> identical ``_motifs`` / ``_nearest``.

``quantize_soft_momos(model, quant_cfg) -> stats``
    * Registered as ``METHODS["soft_momos"]`` (and exported from
      ``src.quantizers``). Reads ``rows, cols, final_k, force_zero`` and the
      optional ``finalize: {backend, max_iter, n_init, seed}`` sub-dict (all with
      the defaults above) and calls ``finalize_soft_momos``.

``LitClassifier`` (``src/model/lit_module.py``)
-----------------------------------------------
    * New attribute ``aux_losses: list[Callable[[nn.Module], Tensor]]`` (default
      ``[]``). ``training_step`` returns ``CE + sum(f(self.model) for f in
      aux_losses)`` and logs the aux sum as ``"train/aux_loss"`` (at least
      whenever ``aux_losses`` is non-empty). With no aux losses the returned loss
      is exactly the CE loss.

``QuantizationCallback`` (``src/utils/callbacks.py``), ``method == "soft_momos"``
---------------------------------------------------------------------------------
Config keys: ``rows, cols, k | capacity, final_k, force_zero, lam,
lam_warmup_fraction (default 0), validate_projected (default True),
finalize: {...}, chunk_size``.
    * ``_build_quantization_callback`` sets ``quant_cfg["s"] = rows * cols``,
      pops ``enabled`` and raises ``ValueError`` when neither ``k`` nor
      ``capacity`` is given.
    * Attributes: ``callback.M`` (current mosaic, ``None`` before the first draw),
      ``callback.finalized`` (bool, ``False`` initially), ``callback.current_lam``
      (float; set by ``on_train_epoch_start``, initial value unspecified).
    * ``on_fit_start``: resolves ``quant_cfg["k"]`` from ``capacity`` via
      ``k_from_capacity(model, s, capacity)`` when ``k`` is None, and appends
      EXACTLY ONE closure to ``pl_module.aux_losses`` (calling ``on_fit_start``
      twice must not register a second one). The closure returns
      ``current_lam * soft_momos_penalty(model, M, rows, cols, k, force_zero)[0]``.
    * ``on_train_epoch_start``: ``self.M = draw_mosaic(...)[0]`` on the CURRENT
      weights (weights untouched; a new draw every epoch) and sets
      ``current_lam = lam * min(1, current_epoch / (lam_warmup_fraction *
      max_epochs))`` (``current_lam = lam`` when ``lam_warmup_fraction == 0``).
    * ``on_validation_epoch_start`` (only when ``not trainer.sanity_checking`` and
      ``trainer.state.fn == TrainerFn.FITTING``): if ``validate_projected``, stash
      the raw weights and write ``Z[M]`` (``project_fixed_mosaic``).
      ``on_validation_epoch_end``: restore the raw weights bit-exactly (no-op
      when nothing was stashed, e.g. during the sanity check).
    * ``on_validation_start`` (``trainer.state.fn == TrainerFn.VALIDATING``) and
      ``on_test_start`` (``TrainerFn.TESTING``): when ``not self.finalized``,
      run ``finalize_soft_momos`` in place and set ``finalized = True``. Any later
      call does nothing. During fit (``FITTING``) these hooks do nothing. With
      ``trainer.checkpoint_callback is None`` nothing is written to disk.
    * ``state_dict()`` / ``load_state_dict()`` round-trip ``M`` and ``finalized``.
    * The fake trainer used below exposes only: ``state.fn, current_epoch,
      max_epochs, sanity_checking, checkpoint_callback (None), callback_metrics,
      logger (None), default_root_dir (None)``. ``pl_module.log`` is called
      outside a Trainer in these tests (Lightning only warns), so logging in the
      hooks must not assume a logger.

``compute_quantization_bits`` (``src/utils/quant_bits.py``): see
``tests/test_quant_bits.py`` -- ``method == "soft_momos"`` uses ``final_k``
(required, ``ValueError`` otherwise) as the codebook size: ``motif_bits =
final_k * s * q_bits``, ``index_bits = n_blocks * ceil(log2 final_k)`` with
``n_blocks = sum(ceil(numel / s))`` exactly as the ``momos2d`` branch counts it.

``resolve_block_shape("soft_momos", cfg) == (rows, cols)``.
"""

import copy
import time
import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import src.quantizers as quantizers
import src.quantizers.block_utils as block_utils
import src.quantizers.soft_momos as soft_momos_mod
from src.quantizers.block_utils import iter_trainable_params, k_from_capacity
from src.quantizers.momos2d import _get_model_blocks2D, blocks_to_tensor2D
from src.quantizers.soft_momos import (
    draw_mosaic,
    finalize_soft_momos,
    group_means,
    project_fixed_mosaic,
    quantize_soft_momos,
    soft_momos_penalty,
)
from model.lit_module import LitClassifier
from utils.callbacks import (
    QuantizationCallback,
    _QUANT_LOG_SKIP,
    _build_quantization_callback,
)
from utils.metrics import resolve_block_shape


# -----------------------------------------------------------------------------
# Helpers / fixtures
# -----------------------------------------------------------------------------


class PaddedLinearModel(nn.Module):
    """nn.Linear(5, 3) with bias: with rows=cols=2 both the (3, 5) weight and the
    (1, 3) bias need zero padding (6 + 2 = 8 blocks, 18 real weights)."""

    def __init__(self, seed=0):
        super().__init__()
        self.fc = nn.Linear(5, 3, bias=True)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(3, 5, generator=g))
            self.fc.bias.copy_(torch.randn(3, generator=g))


# A mosaic for PaddedLinearModel's 8 blocks over 3 motifs; group 0 (the zero
# motif) has members too, so force_zero pinning is exercised.
PADDED_M = torch.tensor([0, 1, 1, 2, 2, 0, 1, 2], dtype=torch.long)


class FourValueModel(nn.Module):
    """A (16, 16) weight whose 2x2 blocks take exactly four distinct values
    (16 blocks each, one of them the zero block). No padding with rows=cols=2.
    Values are small dyadic rationals so float32 sums/means are exact."""

    VALUES = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 4.0],
            [-1.0, 0.5, 2.0, -2.0],
            [3.0, 3.0, 3.0, 3.0],
        ],
        dtype=torch.float32,
    )

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16, bias=False)
        idx = torch.arange(64) % 4
        blocks = self.VALUES[idx]
        with torch.no_grad():
            self.fc.weight.copy_(blocks_to_tensor2D(blocks, (16, 16), 2, 2))


class RandomLinearModel(nn.Module):
    """A (16, 16) random weight without bias: 64 distinct 2x2 blocks."""

    def __init__(self, seed=0):
        super().__init__()
        self.fc = nn.Linear(16, 16, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(16, 16, generator=g))


def _snapshot(model):
    return [p.detach().clone() for p in model.parameters()]


def _assert_params_equal(model, snapshot):
    for p, q in zip(model.parameters(), snapshot):
        assert torch.equal(p.detach(), q)


def _diff_stats(before, after):
    """(sum of squared differences, number of changed entries) over two snapshots."""
    distortion = sum((b - a).square().sum().item() for b, a in zip(before, after))
    changed = sum((b != a).sum().item() for b, a in zip(before, after))
    return distortion, changed


def _unique_blocks(model, rows, cols):
    blocks, _ = _get_model_blocks2D(model, rows, cols)
    return torch.unique(blocks, dim=0)


def _blocks_to_param_tensors(blocks, layer_specs, rows, cols):
    """Un-block a ``(m, s)`` matrix into one tensor per param (padding dropped)."""
    out = []
    offset = 0
    for param, n_blocks, _n_params, shape in layer_specs:
        out.append(
            blocks_to_tensor2D(blocks[offset : offset + n_blocks], shape, rows, cols)
            .reshape(param.shape)
        )
        offset += n_blocks
    return out


def _as_number(value):
    """Stats values may be python numbers or 0-dim tensors."""
    if isinstance(value, torch.Tensor):
        assert value.dim() == 0, "stats tensors must be 0-dim"
        return value.item()
    assert isinstance(value, (int, float)), f"{value!r} is not a number"
    return value


def _loop_group_means(blocks, M, k, force_zero):
    """Reference: explicit python loop over motifs (deliberately NOT the
    index_add_ formulation the implementation is expected to use)."""
    Z = torch.zeros(k, blocks.shape[1], dtype=blocks.dtype)
    for j in range(k):
        mask = M == j
        if mask.any():
            Z[j] = blocks[mask].mean(dim=0)
    if force_zero:
        Z[0] = 0.0
    return Z


def _reference_penalty(model, M, rows, cols, k, force_zero):
    """Reference penalty with the documented padding rule (padded blocks as-is,
    normalised by the number of REAL weights)."""
    blocks, _specs = _get_model_blocks2D(model, rows, cols)
    Z = _loop_group_means(blocks, M, k, force_zero)
    n_real = sum(p.numel() for p in iter_trainable_params(model))
    sq = (blocks - Z[M]).square().sum()
    return sq / n_real, Z, blocks, n_real


def _tiny_backbone():
    # Shapes chosen so that rows=1, cols=2 never pads: weights (8, 16) and
    # (4, 8), biases (1, 8) and (1, 4) -> 64 + 4 + 16 + 2 = 86 blocks.
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))


def _lit(seed=0, epochs=2):
    torch.manual_seed(seed)
    return LitClassifier(backbone=_tiny_backbone(), num_classes=4, epochs=epochs)


TINY_ROWS, TINY_COLS, TINY_K, TINY_FINAL_K, TINY_BLOCKS = 1, 2, 8, 4, 86


def _soft_cfg(**overrides):
    cfg = {
        "enabled": True,
        "method": "soft_momos",
        "rows": TINY_ROWS,
        "cols": TINY_COLS,
        "k": TINY_K,
        "capacity": None,
        "final_k": TINY_FINAL_K,
        "force_zero": True,
        "lam": 1.0,
        "lam_warmup_fraction": 0.0,
        "validate_projected": True,
        "finalize": {"backend": "exact", "max_iter": 50, "n_init": 1, "seed": 0},
        "chunk_size": None,
    }
    cfg.update(overrides)
    return cfg


def _build_cb(**overrides):
    """Build the callback the same way ``build_callbacks`` does."""
    return _build_quantization_callback({}, OmegaConf.create(_soft_cfg(**overrides)))


class FakeTrainer:
    """Stand-in for a Lightning Trainer exposing only what the callback may read."""

    def __init__(
        self,
        fn=TrainerFn.FITTING,
        current_epoch=0,
        max_epochs=10,
        sanity_checking=False,
    ):
        self.state = SimpleNamespace(fn=fn)
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.sanity_checking = sanity_checking
        self.checkpoint_callback = None
        self.callback_metrics = {}
        self.logger = None
        self.default_root_dir = None


def _fitted_cb(seed, **overrides):
    """Callback + LitClassifier after on_fit_start and one epoch-0 mosaic draw."""
    cb = _build_cb(**overrides)
    pl = _lit(seed=seed)
    cb.on_fit_start(FakeTrainer(), pl)
    cb.on_train_epoch_start(FakeTrainer(current_epoch=0), pl)
    return cb, pl


# -----------------------------------------------------------------------------
# group_means
# -----------------------------------------------------------------------------


class TestGroupMeans(unittest.TestCase):
    def test_matches_explicit_loop_and_pins_zero_motif(self):
        torch.manual_seed(0)
        blocks = torch.randn(40, 4) + 1.0  # strictly non-zero on average
        M = torch.randint(0, 6, (40,))
        M[:5] = 0  # group 0 has members -> must still be pinned to 0
        Z = group_means(blocks, M, 6, force_zero=True)

        self.assertEqual(Z.shape, (6, 4))
        self.assertEqual(Z.dtype, blocks.dtype)
        self.assertTrue(torch.allclose(Z, _loop_group_means(blocks, M, 6, True), atol=1e-6))
        self.assertTrue(torch.equal(Z[0], torch.zeros(4)))

    def test_without_force_zero_group_zero_is_a_plain_mean(self):
        torch.manual_seed(1)
        blocks = torch.randn(30, 2)
        M = torch.randint(0, 3, (30,))
        M[:4] = 0
        Z = group_means(blocks, M, 3, force_zero=False)
        self.assertTrue(torch.allclose(Z[0], blocks[M == 0].mean(dim=0), atol=1e-6))
        self.assertTrue(torch.allclose(Z, _loop_group_means(blocks, M, 3, False), atol=1e-6))

    def test_empty_groups_are_zero_not_nan(self):
        blocks = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        M = torch.tensor([2, 2, 2])
        Z = group_means(blocks, M, 5, force_zero=True)
        self.assertEqual(Z.shape, (5, 2))
        self.assertTrue(torch.isfinite(Z).all())
        self.assertTrue(torch.allclose(Z[2], torch.tensor([3.0, 4.0])))
        for j in (0, 1, 3, 4):
            self.assertTrue(torch.equal(Z[j], torch.zeros(2)), f"row {j}")

    def test_is_linear_time_for_many_motifs(self):
        # 200k blocks, 50k motifs: a k x m one-hot/distance matrix would be 10^10
        # floats (40 GB) and a python loop over motifs 50k mask evaluations of a
        # 200k vector -- both far above 1 s; index_add_/bincount is ~1 ms.
        torch.manual_seed(0)
        m, k, s = 200_000, 50_000, 2
        blocks = torch.randn(m, s)
        M = torch.randint(0, k, (m,))
        group_means(blocks[:100], M[:100], k, force_zero=True)  # warm-up

        start = time.perf_counter()
        Z = group_means(blocks, M, k, force_zero=True)
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 1.0, f"group_means took {elapsed:.2f}s")

        counts = torch.bincount(M, minlength=k).clamp(min=1).unsqueeze(1)
        ref = torch.zeros(k, s).index_add_(0, M, blocks) / counts
        ref[0] = 0.0
        self.assertEqual(Z.shape, (k, s))
        self.assertTrue(torch.allclose(Z, ref, atol=1e-5))


# -----------------------------------------------------------------------------
# draw_mosaic
# -----------------------------------------------------------------------------


class TestDrawMosaic(unittest.TestCase):
    def test_returns_nearest_assignment_and_leaves_weights_untouched(self):
        model = RandomLinearModel(seed=3)
        before = _snapshot(model)
        blocks, specs_ref = _get_model_blocks2D(model, 2, 2)

        torch.manual_seed(0)
        M, Z0, layer_specs = draw_mosaic(model, 2, 2, 8, force_zero=True)

        _assert_params_equal(model, before)
        self.assertEqual(M.dtype, torch.long)
        self.assertEqual(M.shape, (64,))
        self.assertEqual(Z0.shape, (8, 4))
        self.assertTrue(torch.equal(Z0[0], torch.zeros(4)))
        self.assertGreaterEqual(int(M.min()), 0)
        self.assertLess(int(M.max()), 8)

        # Every non-zero motif is one of the current blocks.
        for j in range(1, 8):
            self.assertTrue((blocks == Z0[j]).all(dim=1).any(), f"Z0[{j}] not a block")

        # M is the nearest-motif assignment (up to ties). Exact broadcast
        # distances (64 x 8 x 4), not torch.cdist's matmul path.
        d_all = (blocks[:, None, :] - Z0[None, :, :]).square().sum(dim=2)
        d_assigned = (blocks - Z0[M]).square().sum(dim=1)
        self.assertTrue(torch.allclose(d_assigned, d_all.min(dim=1).values, atol=1e-6))

        # layer_specs has the _get_model_blocks2D layout.
        self.assertEqual(len(layer_specs), len(specs_ref))
        for (p, n_blocks, n_params, shape), (p_ref, nb_ref, np_ref, shape_ref) in zip(
            layer_specs, specs_ref
        ):
            self.assertIs(p, p_ref)
            self.assertEqual(n_blocks, nb_ref)
            self.assertEqual(n_params, np_ref)
            self.assertEqual(tuple(shape), tuple(shape_ref))
        self.assertEqual(sum(spec[1] for spec in layer_specs), M.numel())

    def test_without_force_zero_all_motifs_are_blocks(self):
        model = RandomLinearModel(seed=4)
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        torch.manual_seed(0)
        M, Z0, _ = draw_mosaic(model, 2, 2, 5, force_zero=False)
        self.assertEqual(Z0.shape, (5, 4))
        for j in range(5):
            self.assertTrue((blocks == Z0[j]).all(dim=1).any())

    def test_k_is_clamped_to_total_blocks(self):
        model = RandomLinearModel(seed=5)  # 64 blocks at 2x2
        M, Z0, _ = draw_mosaic(model, 2, 2, 1000, force_zero=True)
        self.assertEqual(Z0.shape[0], 64)
        self.assertLess(int(M.max()), 64)

    def test_is_deterministic_under_manual_seed_and_random_otherwise(self):
        model = RandomLinearModel(seed=6)
        torch.manual_seed(123)
        M_a, Z_a, _ = draw_mosaic(model, 2, 2, 8)
        torch.manual_seed(123)
        M_b, Z_b, _ = draw_mosaic(model, 2, 2, 8)
        self.assertTrue(torch.equal(M_a, M_b))
        self.assertTrue(torch.equal(Z_a, Z_b))

        torch.manual_seed(456)
        M_c, _, _ = draw_mosaic(model, 2, 2, 8)
        self.assertFalse(torch.equal(M_a, M_c))

    def test_multi_param_model_concatenates_in_trainable_param_order(self):
        model = PaddedLinearModel()
        M, Z0, layer_specs = draw_mosaic(model, 2, 2, 4)
        self.assertEqual(M.shape, (8,))
        self.assertEqual([spec[1] for spec in layer_specs], [6, 2])
        self.assertIs(layer_specs[0][0], model.fc.weight)
        self.assertIs(layer_specs[1][0], model.fc.bias)

    def test_uses_chunked_nearest_motifs_with_forwarded_chunk_size(self):
        # Peak memory of the assignment cannot be measured reliably on CPU (no
        # per-tensor allocator stats, and a naive 200k x 50k matrix would OOM
        # the test process rather than fail), so bounded memory is enforced
        # structurally: the assignment must go through the existing chunked
        # helper, with the caller's chunk_size (MB budget) forwarded verbatim.
        model = RandomLinearModel(seed=7)
        calls = []
        real = block_utils._nearest_motifs_chunked

        def spy(blocks, motifs, *args, **kwargs):
            calls.append((args, kwargs))
            return real(blocks, motifs, *args, **kwargs)

        # Patch the helper where _assign_blocks resolves it (block_utils globals)
        # and, if soft_momos imported it by name, there as well.
        targets = [block_utils]
        if hasattr(soft_momos_mod, "_nearest_motifs_chunked"):
            targets.append(soft_momos_mod)
        with ExitStack() as stack:
            for target in targets:
                stack.enter_context(patch.object(target, "_nearest_motifs_chunked", spy))
            M, Z0, _ = draw_mosaic(model, 2, 2, 8, force_zero=True, chunk_size=7)

        self.assertEqual(len(calls), 1, "draw_mosaic must go through _nearest_motifs_chunked")
        args, kwargs = calls[0]
        chunk = kwargs.get("chunk_size", args[0] if args else None)
        self.assertEqual(chunk, 7)
        self.assertEqual(M.shape, (64,))


# -----------------------------------------------------------------------------
# soft_momos_penalty
# -----------------------------------------------------------------------------


class TestSoftMomosPenalty(unittest.TestCase):
    ROWS, COLS, K = 2, 2, 3

    def test_value_matches_reference_with_padding_rule(self):
        model = PaddedLinearModel(seed=1)
        penalty, stats = soft_momos_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, force_zero=True
        )
        ref, Z, blocks, n_real = _reference_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, True
        )
        self.assertEqual(n_real, 18)
        self.assertEqual(penalty.dim(), 0)
        self.assertTrue(torch.allclose(penalty.detach(), ref, atol=1e-6))

        self.assertIsInstance(stats, dict)
        for value in stats.values():
            _as_number(value)
        rel_ref = (blocks - Z[PADDED_M]).square().sum() / blocks.square().sum()
        self.assertAlmostEqual(_as_number(stats["rel_distortion"]), float(rel_ref), places=5)
        self.assertEqual(int(_as_number(stats["active_motifs"])), 3)

    def test_is_differentiable_with_analytic_gradient(self):
        model = PaddedLinearModel(seed=2)
        model.zero_grad()
        penalty, _ = soft_momos_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, force_zero=True
        )
        self.assertTrue(penalty.requires_grad)
        self.assertIsNotNone(penalty.grad_fn)
        penalty.backward()

        _ref, Z, blocks, n_real = _reference_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, True
        )
        # Analytic gradient 2 (b - Z[M]) / n at real positions (padding dropped).
        grad_blocks = 2.0 * (blocks - Z[PADDED_M]) / n_real
        _blocks, layer_specs = _get_model_blocks2D(model, self.ROWS, self.COLS)
        expected = _blocks_to_param_tensors(grad_blocks, layer_specs, self.ROWS, self.COLS)
        for (param, *_), exp in zip(layer_specs, expected):
            self.assertIsNotNone(param.grad)
            self.assertTrue(
                torch.allclose(param.grad, exp, atol=1e-6),
                f"gradient mismatch for param of shape {tuple(param.shape)}",
            )

    def test_zero_when_blocks_already_equal_their_group_means(self):
        model = FourValueModel()
        M = torch.arange(64) % 4  # each group is exactly one value
        penalty, stats = soft_momos_penalty(model, M, 2, 2, 4, force_zero=True)
        self.assertAlmostEqual(penalty.detach().item(), 0.0, places=6)
        self.assertAlmostEqual(_as_number(stats["rel_distortion"]), 0.0, places=6)
        self.assertEqual(int(_as_number(stats["active_motifs"])), 4)

    def test_k_larger_than_used_motifs_does_not_change_value(self):
        model = PaddedLinearModel(seed=3)
        p_small, _ = soft_momos_penalty(model, PADDED_M, 2, 2, 3, force_zero=True)
        p_big, stats = soft_momos_penalty(model, PADDED_M, 2, 2, 10, force_zero=True)
        self.assertTrue(torch.allclose(p_small.detach(), p_big.detach(), atol=1e-7))
        self.assertEqual(int(_as_number(stats["active_motifs"])), 3)

    def test_scales_to_many_blocks_and_motifs(self):
        # 400k weights -> 200k blocks at 1x2, 50k motifs: forward + backward must
        # stay far below the budget (a k x m distance/one-hot matrix would need
        # 40 GB; the O(m) path is a few ms).
        torch.manual_seed(0)
        model = nn.Linear(400, 1000, bias=False)
        M = torch.randint(0, 50_000, (200_000,))
        # Warm-up on a tiny model so one-off allocator/autograd setup is excluded.
        warm_p, _ = soft_momos_penalty(
            nn.Linear(4, 4, bias=False), torch.zeros(8, dtype=torch.long), 1, 2, 4
        )
        warm_p.backward()

        start = time.perf_counter()
        penalty, stats = soft_momos_penalty(model, M, 1, 2, 50_000, force_zero=True)
        penalty.backward()
        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0, f"penalty forward+backward took {elapsed:.2f}s")
        self.assertIsNotNone(model.weight.grad)
        self.assertGreater(penalty.detach().item(), 0.0)


# -----------------------------------------------------------------------------
# project_fixed_mosaic
# -----------------------------------------------------------------------------


class TestProjectFixedMosaic(unittest.TestCase):
    def test_writes_group_means_in_place_with_stats(self):
        model = PaddedLinearModel(seed=4)
        before = _snapshot(model)
        blocks, layer_specs = _get_model_blocks2D(model, 2, 2)
        Z = _loop_group_means(blocks, PADDED_M, 3, True)

        stats = project_fixed_mosaic(model, PADDED_M, 2, 2, 3, force_zero=True)

        expected = _blocks_to_param_tensors(Z[PADDED_M], layer_specs, 2, 2)
        for (param, *_), exp in zip(layer_specs, expected):
            self.assertTrue(torch.allclose(param.detach(), exp, atol=1e-6))

        distortion, changed = _diff_stats(before, _snapshot(model))
        self.assertAlmostEqual(stats["distortion"], distortion, places=5)
        self.assertEqual(stats["num_changed_weights"], changed)
        self.assertEqual(stats["num_motifs_used"], 3)
        for key in ("distortion", "num_changed_weights", "num_motifs_used"):
            self.assertIsInstance(stats[key], (int, float))

    def test_blocks_in_zero_group_become_exactly_zero(self):
        model = RandomLinearModel(seed=8)
        M = torch.zeros(64, dtype=torch.long)
        M[32:] = 1
        project_fixed_mosaic(model, M, 2, 2, 2, force_zero=True)
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        self.assertTrue(torch.equal(blocks[:32], torch.zeros(32, 4)))
        self.assertEqual(torch.unique(blocks[32:], dim=0).shape[0], 1)

    def test_is_idempotent_up_to_rounding(self):
        # After one projection every group is constant, so re-projecting with
        # the same M moves nothing beyond float rounding of mean(x, x, ..., x).
        model = RandomLinearModel(seed=9)
        torch.manual_seed(0)
        M, _, _ = draw_mosaic(model, 2, 2, 8)
        project_fixed_mosaic(model, M, 2, 2, 8, force_zero=True)
        snap = _snapshot(model)
        stats = project_fixed_mosaic(model, M, 2, 2, 8, force_zero=True)
        for p, q in zip(model.parameters(), snap):
            self.assertTrue(torch.allclose(p.detach(), q, atol=1e-6, rtol=0))
        self.assertLess(stats["distortion"], 1e-9)


# -----------------------------------------------------------------------------
# finalize_soft_momos / quantize_soft_momos / METHODS registry
# -----------------------------------------------------------------------------


class TestFinalizeSoftMomos(unittest.TestCase):
    def _check_exact_recovery(self, backend):
        model = FourValueModel()
        stats = finalize_soft_momos(
            model, 2, 2, final_k=4, force_zero=True, seed=0, backend=backend
        )

        self.assertAlmostEqual(stats["distortion"], 0.0, places=5)
        self.assertEqual(stats["num_motifs_used"], 4)

        uniques = _unique_blocks(model, 2, 2)
        self.assertLessEqual(uniques.shape[0], 4)
        for u in uniques:
            dist = (FourValueModel.VALUES - u).square().sum(dim=1).min()
            self.assertLess(float(dist), 1e-8, f"block {u.tolist()} is not an original value")

        # Codebook / mosaic internals reconstruct the written weights bit-exactly.
        motifs, nearest, layer_specs = stats["_motifs"], stats["_nearest"], stats["_layer_specs"]
        self.assertEqual(motifs.shape, (4, 4))
        self.assertEqual(nearest.dtype, torch.long)
        self.assertEqual(nearest.shape, (64,))
        self.assertTrue((motifs == 0).all(dim=1).any(), "force_zero must keep an exact 0 motif")
        rebuilt = _blocks_to_param_tensors(motifs[nearest], layer_specs, 2, 2)
        for (param, *_), exp in zip(layer_specs, rebuilt):
            self.assertTrue(torch.equal(param.detach(), exp))

        counts = stats["motif_counts"]
        self.assertEqual(counts.dtype, torch.long)
        self.assertEqual(counts.shape, (4,))
        self.assertEqual(int(counts.sum()), 64)
        self.assertTrue(torch.equal(counts, torch.bincount(nearest, minlength=4)))

    def test_exact_backend_recovers_collapsed_blocks_with_zero_distortion(self):
        self._check_exact_recovery("exact")

    def test_minibatch_backend_recovers_collapsed_blocks_with_zero_distortion(self):
        self._check_exact_recovery("minibatch")

    def test_random_weights_end_up_with_at_most_final_k_blocks_and_zero_motif(self):
        model = RandomLinearModel(seed=10)
        before = _snapshot(model)
        stats = finalize_soft_momos(model, 2, 2, final_k=8, force_zero=True, seed=1)

        uniques = _unique_blocks(model, 2, 2)
        self.assertLessEqual(uniques.shape[0], 8)
        self.assertLessEqual(stats["num_motifs_used"], 8)
        self.assertGreater(stats["num_motifs_used"], 1)
        self.assertGreater(stats["distortion"], 0.0)

        distortion, changed = _diff_stats(before, _snapshot(model))
        self.assertAlmostEqual(stats["distortion"], distortion, places=4)
        self.assertEqual(stats["num_changed_weights"], changed)

        motifs = stats["_motifs"]
        self.assertEqual(motifs.shape, (8, 4))
        self.assertTrue((motifs == 0).all(dim=1).any(), "force_zero must snap one centroid to 0")

    def test_stats_are_loggable_except_skipped_keys(self):
        model = RandomLinearModel(seed=11)
        stats = finalize_soft_momos(model, 2, 2, final_k=8)
        for key in ("distortion", "num_changed_weights", "num_motifs_used", "motif_counts",
                    "_motifs", "_nearest", "_layer_specs"):
            self.assertIn(key, stats)
        for key, value in stats.items():
            if key in _QUANT_LOG_SKIP:
                continue
            self.assertIsInstance(value, (int, float), f"stats[{key!r}] is not loggable")

    def test_same_seed_is_deterministic(self):
        model_a = RandomLinearModel(seed=12)
        model_b = RandomLinearModel(seed=12)
        stats_a = finalize_soft_momos(model_a, 2, 2, final_k=8, seed=3)
        stats_b = finalize_soft_momos(model_b, 2, 2, final_k=8, seed=3)
        self.assertTrue(torch.equal(stats_a["_motifs"], stats_b["_motifs"]))
        self.assertTrue(torch.equal(stats_a["_nearest"], stats_b["_nearest"]))
        self.assertTrue(torch.equal(model_a.fc.weight, model_b.fc.weight))

    def test_registered_in_methods_and_dispatcher(self):
        self.assertIn("soft_momos", quantizers.METHODS)
        self.assertIs(quantizers.METHODS["soft_momos"], quantize_soft_momos)
        self.assertIn("soft_momos", quantizers.available_methods())
        self.assertIs(quantizers.quantize_soft_momos, quantize_soft_momos)

        model = FourValueModel()
        cfg = {
            "method": "soft_momos",
            "rows": 2,
            "cols": 2,
            "k": 8,  # training-time k: irrelevant for finalize
            "final_k": 4,
            "force_zero": True,
            "finalize": {"backend": "exact", "max_iter": 50, "n_init": 1, "seed": 0},
        }
        stats = quantizers.quantize(model, cfg)
        self.assertEqual(stats["method"], "soft_momos")
        self.assertIn("q_time", stats)
        self.assertAlmostEqual(stats["distortion"], 0.0, places=5)
        self.assertEqual(stats["num_motifs_used"], 4)
        self.assertLessEqual(_unique_blocks(model, 2, 2).shape[0], 4)

    def test_dispatcher_works_without_finalize_subdict(self):
        model = RandomLinearModel(seed=13)
        stats = quantize_soft_momos(
            model, {"rows": 2, "cols": 2, "final_k": 8, "force_zero": True}
        )
        self.assertLessEqual(stats["num_motifs_used"], 8)
        self.assertLessEqual(_unique_blocks(model, 2, 2).shape[0], 8)


# -----------------------------------------------------------------------------
# LitClassifier.aux_losses
# -----------------------------------------------------------------------------


def test_lit_classifier_aux_losses_default_empty_and_loss_is_ce():
    module = _lit(seed=0)
    assert module.aux_losses == []
    x = torch.randn(6, 16)
    y = torch.randint(0, 4, (6,))
    with patch.object(module, "log"):
        loss = module.training_step((x, y), 0)
    ce = module.criterion(module(x), y)
    assert torch.allclose(loss, ce)


def test_lit_classifier_training_step_adds_aux_losses_and_logs_them():
    module = _lit(seed=1)
    module.aux_losses.append(lambda m: 0.5 * m[1].weight.square().sum())
    module.aux_losses.append(lambda m: torch.tensor(0.25))
    x = torch.randn(6, 16)
    y = torch.randint(0, 4, (6,))

    with patch.object(module, "log") as log:
        loss = module.training_step((x, y), 0)

    ce = module.criterion(module(x), y)
    aux = 0.5 * module.model[1].weight.square().sum() + 0.25
    assert torch.allclose(loss, ce + aux)
    assert loss.requires_grad

    logged = {}
    for call in log.call_args_list:
        name = call.args[0] if call.args else call.kwargs["name"]
        value = call.args[1] if len(call.args) > 1 else call.kwargs["value"]
        logged[name] = value
    assert "train/aux_loss" in logged
    assert torch.allclose(torch.as_tensor(logged["train/aux_loss"]).float(), aux.detach())
    assert "train/loss" in logged


# -----------------------------------------------------------------------------
# QuantizationCallback (soft_momos) with a fake trainer
# -----------------------------------------------------------------------------


class TestSoftMomosCallbackBuild(unittest.TestCase):
    def test_build_sets_block_size_and_method(self):
        cb = _build_cb()
        self.assertIsInstance(cb, QuantizationCallback)
        self.assertEqual(cb.method, "soft_momos")
        self.assertEqual(cb.quant_cfg["s"], TINY_ROWS * TINY_COLS)
        self.assertNotIn("enabled", cb.quant_cfg)
        self.assertEqual(cb.quant_cfg["final_k"], TINY_FINAL_K)
        self.assertIsNone(cb.M)
        self.assertFalse(cb.finalized)

    def test_build_requires_k_or_capacity(self):
        with self.assertRaises(ValueError):
            _build_cb(k=None, capacity=None)
        _build_cb(k=None, capacity=0.1)  # ok
        _build_cb(k=8, capacity=None)  # ok

    def test_resolve_block_shape(self):
        cb = _build_cb(rows=2, cols=3)
        self.assertEqual(resolve_block_shape(cb.method, cb.quant_cfg), (2, 3))
        self.assertEqual(resolve_block_shape("soft_momos", {"rows": 4, "cols": 1}), (4, 1))


class TestSoftMomosCallbackHooks(unittest.TestCase):
    def test_on_fit_start_registers_exactly_one_closure_and_resolves_capacity(self):
        cb = _build_cb(k=None, capacity=0.1)
        pl = _lit(seed=0)
        self.assertEqual(pl.aux_losses, [])

        cb.on_fit_start(FakeTrainer(), pl)
        self.assertEqual(len(pl.aux_losses), 1)
        self.assertTrue(callable(pl.aux_losses[0]))
        expected_k = k_from_capacity(pl.model, TINY_ROWS * TINY_COLS, 0.1)
        self.assertEqual(cb.quant_cfg["k"], expected_k)

        cb.on_fit_start(FakeTrainer(), pl)
        self.assertEqual(len(pl.aux_losses), 1, "re-running on_fit_start must not double the penalty")

    def test_on_train_epoch_start_draws_new_mosaic_without_touching_weights(self):
        cb = _build_cb()
        pl = _lit(seed=1)
        cb.on_fit_start(FakeTrainer(), pl)
        before = _snapshot(pl.model)

        cb.on_train_epoch_start(FakeTrainer(current_epoch=0), pl)
        _assert_params_equal(pl.model, before)
        self.assertIsInstance(cb.M, torch.Tensor)
        self.assertEqual(cb.M.dtype, torch.long)
        self.assertEqual(cb.M.shape, (TINY_BLOCKS,))
        self.assertLess(int(cb.M.max()), TINY_K)
        M0 = cb.M.clone()

        with torch.no_grad():
            for p in pl.model.parameters():
                p.add_(0.1 * torch.randn_like(p))
        before = _snapshot(pl.model)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=1), pl)
        _assert_params_equal(pl.model, before)
        self.assertFalse(torch.equal(cb.M, M0), "a new mosaic must be drawn every epoch")

    def test_on_train_epoch_start_draws_from_current_not_stale_weights(self):
        # k=2 -> Z0 = [0, one sampled block]. After overwriting every weight
        # with 1.0 all 86 blocks equal (1, 1), so the sampled motif is (1, 1)
        # and the nearest assignment is 1 for every block, with no ties. A draw
        # from the epoch-0 (random) weights would give a mixed mosaic.
        cb, pl = _fitted_cb(seed=2, k=2)
        with torch.no_grad():
            for p in pl.model.parameters():
                p.fill_(1.0)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=1), pl)
        self.assertTrue(torch.equal(cb.M, torch.ones(TINY_BLOCKS, dtype=torch.long)))
        for p in pl.model.parameters():
            self.assertTrue(torch.equal(p.detach(), torch.ones_like(p)))

    def test_registered_closure_returns_lam_times_penalty(self):
        cb = _build_cb(lam=0.7, lam_warmup_fraction=0.0)
        pl = _lit(seed=2)
        cb.on_fit_start(FakeTrainer(), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=3), pl)
        self.assertAlmostEqual(cb.current_lam, 0.7)

        value = pl.aux_losses[0](pl.model)
        ref, _ = soft_momos_penalty(
            pl.model, cb.M, TINY_ROWS, TINY_COLS, cb.quant_cfg["k"], force_zero=True
        )
        self.assertIsInstance(value, torch.Tensor)
        self.assertTrue(value.requires_grad)
        self.assertTrue(torch.allclose(value.detach(), 0.7 * ref.detach(), atol=1e-7))

    def test_lam_warmup_schedule(self):
        # lam * min(1, epoch / (0.5 * 10)): 0, 0.4, 1.6, 2.0, 2.0.
        cb = _build_cb(lam=2.0, lam_warmup_fraction=0.5)
        pl = _lit(seed=3)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        expected = {0: 0.0, 1: 0.4, 4: 1.6, 5: 2.0, 9: 2.0}
        for epoch, lam in expected.items():
            cb.on_train_epoch_start(FakeTrainer(current_epoch=epoch, max_epochs=10), pl)
            self.assertAlmostEqual(cb.current_lam, lam, places=6, msg=f"epoch {epoch}")

        cb_const = _build_cb(lam=2.0, lam_warmup_fraction=0.0)
        pl_const = _lit(seed=3)
        cb_const.on_fit_start(FakeTrainer(max_epochs=10), pl_const)
        cb_const.on_train_epoch_start(FakeTrainer(current_epoch=0, max_epochs=10), pl_const)
        self.assertAlmostEqual(cb_const.current_lam, 2.0)

    def test_validation_projects_to_group_means_and_restores_raw_weights(self):
        cb, pl = _fitted_cb(seed=4, validate_projected=True)
        raw = _snapshot(pl.model)

        expected = copy.deepcopy(pl.model)
        project_fixed_mosaic(
            expected, cb.M, TINY_ROWS, TINY_COLS, cb.quant_cfg["k"], force_zero=True
        )

        trainer = FakeTrainer(current_epoch=0)
        cb.on_validation_epoch_start(trainer, pl)
        for p, q in zip(pl.model.parameters(), expected.parameters()):
            self.assertTrue(torch.allclose(p.detach(), q.detach(), atol=1e-6))
        self.assertLessEqual(_unique_blocks(pl.model, TINY_ROWS, TINY_COLS).shape[0], TINY_K)

        cb.on_validation_epoch_end(trainer, pl)
        _assert_params_equal(pl.model, raw)

    def test_validation_without_projection_leaves_weights_alone(self):
        cb, pl = _fitted_cb(seed=5, validate_projected=False)
        raw = _snapshot(pl.model)
        trainer = FakeTrainer(current_epoch=0)
        cb.on_validation_epoch_start(trainer, pl)
        _assert_params_equal(pl.model, raw)
        cb.on_validation_epoch_end(trainer, pl)
        _assert_params_equal(pl.model, raw)

    def test_sanity_check_and_standalone_validation_do_not_project(self):
        cb, pl = _fitted_cb(seed=6, validate_projected=True)
        raw = _snapshot(pl.model)

        # Lightning's sanity check runs with fn=FITTING and sanity_checking=True
        # (before any mosaic exists in a real run); trainer.validate() runs with
        # fn=VALIDATING. Neither may project, and epoch_end must be a no-op.
        for trainer in (
            FakeTrainer(sanity_checking=True),
            FakeTrainer(fn=TrainerFn.VALIDATING),
        ):
            cb.on_validation_epoch_start(trainer, pl)
            _assert_params_equal(pl.model, raw)
            cb.on_validation_epoch_end(trainer, pl)
            _assert_params_equal(pl.model, raw)

    def test_state_dict_round_trips_mosaic_and_finalized_flag(self):
        cb, pl = _fitted_cb(seed=7)

        state = cb.state_dict()
        self.assertIsInstance(state, dict)
        cb2 = _build_cb()
        cb2.load_state_dict(state)
        self.assertTrue(torch.equal(cb2.M, cb.M))
        self.assertFalse(cb2.finalized)

        cb.on_validation_start(FakeTrainer(fn=TrainerFn.VALIDATING, current_epoch=2), pl)
        self.assertTrue(cb.finalized)
        cb3 = _build_cb()
        cb3.load_state_dict(cb.state_dict())
        self.assertTrue(cb3.finalized)

    def test_on_validation_start_finalizes_once_when_validating(self):
        cb, pl = _fitted_cb(seed=8)
        self.assertGreater(_unique_blocks(pl.model, TINY_ROWS, TINY_COLS).shape[0], TINY_FINAL_K)

        # During fit nothing happens on on_validation_start.
        raw = _snapshot(pl.model)
        cb.on_validation_start(FakeTrainer(fn=TrainerFn.FITTING, current_epoch=0), pl)
        self.assertFalse(cb.finalized)
        _assert_params_equal(pl.model, raw)

        trainer = FakeTrainer(fn=TrainerFn.VALIDATING, current_epoch=2)
        cb.on_validation_start(trainer, pl)
        self.assertTrue(cb.finalized)
        self.assertLessEqual(_unique_blocks(pl.model, TINY_ROWS, TINY_COLS).shape[0], TINY_FINAL_K)

        # Already finalized: a later validate or test pass leaves the model alone
        # (run_training calls trainer.validate(ckpt_path=best) then trainer.test).
        finalized = _snapshot(pl.model)
        cb.on_validation_start(trainer, pl)
        _assert_params_equal(pl.model, finalized)
        cb.on_test_start(FakeTrainer(fn=TrainerFn.TESTING, current_epoch=2), pl)
        _assert_params_equal(pl.model, finalized)

    def test_on_test_start_finalizes_when_testing_without_prior_validate(self):
        # trainer.test() straight after fit must also evaluate the finalized
        # (final_k-motif) model, not the raw weights.
        cb, pl = _fitted_cb(seed=9)
        raw = _snapshot(pl.model)
        cb.on_test_start(FakeTrainer(fn=TrainerFn.FITTING, current_epoch=0), pl)
        self.assertFalse(cb.finalized)
        _assert_params_equal(pl.model, raw)

        cb.on_test_start(FakeTrainer(fn=TrainerFn.TESTING, current_epoch=2), pl)
        self.assertTrue(cb.finalized)
        self.assertLessEqual(_unique_blocks(pl.model, TINY_ROWS, TINY_COLS).shape[0], TINY_FINAL_K)


# -----------------------------------------------------------------------------
# End-to-end with a real lightning.Trainer (CPU, tiny)
# -----------------------------------------------------------------------------


class _MosaicRecorder(L.Callback):
    """Records the mosaic and lam in effect at the end of every training epoch
    (on_train_epoch_end runs after the in-epoch validation loop)."""

    def __init__(self, quant_cb):
        super().__init__()
        self.quant_cb = quant_cb
        self.mosaics = []
        self.lams = []

    def on_train_epoch_end(self, trainer, pl_module):
        self.mosaics.append(self.quant_cb.M.clone())
        self.lams.append(float(self.quant_cb.current_lam))


def test_end_to_end_training_and_finalize_with_real_trainer():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    val_loader = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16)

    model = _lit(seed=0, epochs=2)
    quant_cb = _build_cb(lam=1.0, lam_warmup_fraction=0.0, validate_projected=True)
    recorder = _MosaicRecorder(quant_cb)
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[quant_cb, recorder],
    )
    # Sanity check (M is None, must be skipped) -> 2 epochs of train + projected
    # validation + restore.
    trainer.fit(model, train_loader, val_loader)

    assert len(model.aux_losses) == 1
    assert "train/aux_loss" in trainer.callback_metrics
    assert len(recorder.mosaics) == 2
    assert recorder.mosaics[0].shape == (TINY_BLOCKS,)
    assert not torch.equal(recorder.mosaics[0], recorder.mosaics[1])
    assert recorder.lams == [1.0, 1.0]
    # After fit the module holds the RAW weights (validation projection
    # restored): far more distinct blocks than the TINY_K-motif projection has.
    assert _unique_blocks(model.model, TINY_ROWS, TINY_COLS).shape[0] > TINY_K
    assert not quant_cb.finalized

    results = trainer.validate(model, val_loader)
    assert quant_cb.finalized
    assert _unique_blocks(model.model, TINY_ROWS, TINY_COLS).shape[0] <= TINY_FINAL_K
    assert results and "val/acc" in results[0]

    # A second standalone validation does not re-finalize.
    finalized = _snapshot(model.model)
    trainer.validate(model, val_loader)
    _assert_params_equal(model.model, finalized)


if __name__ == "__main__":
    unittest.main()
