"""Tests that define the API of ``soft_momos_plus`` (TDD, written before the code).

``soft_momos_plus`` extends ``soft_momos`` (see the contract in
``tests/test_soft_momos.py``, which must keep passing unchanged) with two
independent flags that live on the SAME callback / quantizer code path:

* ``warm_start``  -- the per-epoch mosaic is a Lloyd step from the previous
  epoch's mosaic instead of a fresh random draw;
* ``freeze_fraction`` -- for the last ``ceil(freeze_fraction * max_epochs)``
  epochs the mosaic is frozen and only the codebook ``Z`` is trained.

``method == "soft_momos_plus"`` is an alias of ``"soft_momos"`` whose yaml turns
both on; the flags must ALSO work under ``method == "soft_momos"``. With the
flags off (``warm_start=False``, ``freeze_fraction=0``, the in-code defaults)
behaviour is bit-for-bit that of the base method.

NOTE on imports: the production code imports its own modules as top-level
packages (``quantizers``, ``utils``, ``model``), so this file does too -- never
``src.quantizers`` for classes (``CodebookTensor`` identity!).

================================================================================
API CONTRACT (additions)
================================================================================

Module ``quantizers.soft_momos``
--------------------------------
``warm_start_mosaic(model, M_prev, rows, cols, k, force_zero=True,
                    reseed_dead=True, lloyd_iters=1, chunk_size=None)
    -> (M, Z, layer_specs, stats)``
    * ``blocks`` = ``_get_model_blocks2D(model, rows, cols)`` of the CURRENT
      weights (weights are never modified). One Lloyd iteration:
      ``Z = group_means(blocks, M_prev, k, force_zero)``; "dead" motifs (no member
      in ``M_prev``, motif 0 excluded under ``force_zero``) are, when
      ``reseed_dead``, overwritten with distinct blocks drawn uniformly without
      replacement from the current blocks (global torch RNG), otherwise left as
      zero rows; then ``M = nearest(Z)`` through
      ``_nearest_motifs_chunked(..., chunk_size=chunk_size)``. ``lloyd_iters=n``
      repeats (means -> reseed -> assign) n times, feeding each ``M`` into the
      next; the n-th ``M`` and the ``Z`` it was assigned against are returned.
      ``M`` is ``long (m,)``, ``Z`` is ``(k, s)`` (the codebook USED for the final
      assignment, i.e. including reseeded rows), ``layer_specs`` as in
      ``draw_mosaic``. Rows other than 0 are never pinned.
    * ``stats``: python ``int``/``float`` (not tensors) with at least
      ``"mosaic_changed"`` = fraction of blocks with ``M != M_prev`` (always
      relative to the ``M_prev`` argument) and ``"dead_motifs"`` = number of dead
      motifs of the first iteration, counted BEFORE reseeding.
    * Everything except the chunked assignment is O(m) (no python loop over
      motifs, no ``k x m`` matrix).

``class CodebookTensor(nn.Module)``
    A ``torch.nn.utils.parametrize`` parametrization. Its forward IGNORES the
    original tensor and returns ``blocks_to_tensor2D(Z[M_slice] * keep, shape,
    rows, cols).reshape(param.shape)``, where ``keep`` zeroes motif row 0 when
    ``force_zero`` (so ``Z[0]`` has zero gradient and never leaks into the
    weights even if perturbed). One shared ``nn.Parameter`` ``Z`` serves all
    parametrizations; gather's backward makes ``dL/dZ_j`` = sum of the weight
    gradients over group ``j`` (O(n)).

``freeze_codebook(model, M, rows, cols, k, force_zero=True) -> (Z, handles)``
    ``Z = nn.Parameter(group_means(current blocks, M, k, force_zero))``
    (``requires_grad=True``; shape ``(k, s)``, ``Z[0] == 0`` under force_zero).
    Registers a ``CodebookTensor`` on every (module, name) covered by
    ``iter_trainable_params`` and sets ``original.requires_grad_(False)``;
    afterwards the only trainable parameter in ``model.parameters()`` is ``Z``.
    The values left in ``parametrizations.<n>.original`` are never read (an
    implementation may even free them); ``module.<n>`` is the source of truth.
    ``handles`` is opaque (list).
``unfreeze_codebook(model, handles)``
    ``remove_parametrizations(..., leave_parametrized=True)``: model is plain
    again, weights == materialized ``Z[M]`` bit-exactly, ``requires_grad=True`` on
    all weights, no ``parametrizations`` keys in ``state_dict()``, ``Z`` gone.

``materialized_weights(model)``  (context manager; the FROZEN-PHASE WEIGHT-ACCESS RULE)
    While a model is parametrized, ``iter_trainable_params(model)`` yields ``Z``
    (the only ``requires_grad`` tensor), NOT the weights, so every helper that
    walks it (``_get_model_blocks2D`` and thus ``soft_momos_penalty`` /
    ``project_fixed_mosaic`` / ``finalize_soft_momos`` / ``warm_start_mosaic``,
    ``flatten_weights``, ``count_trainable_parameters``, ``k_from_capacity``,
    ``compute_quantization_bits``) is only valid on a PLAIN model. The rule:
      * ``with materialized_weights(model):`` makes ``iter_trainable_params`` see
        the weights for READING: inside it yields one tensor per quantized
        tensor (in ``model.parameters()`` order) holding the materialized
        ``Z[M]`` values with ``requires_grad=True``, and ``Z`` is not yielded
        (``Z.requires_grad`` is False). On exit (also on exceptions) everything
        is restored exactly: ``Z`` and its ``requires_grad``, the originals'
        flags. Writes made inside are not persisted. It is a no-op yielding ``model`` for a plain model, and it
        detects parametrizations by duck typing (class attribute
        ``is_codebook_tensor = True`` on ``CodebookTensor``) because production
        imports both ``quantizers.*`` and ``src.quantizers.*`` (two module
        copies, two ``CodebookTensor`` classes).
      * ``utils.metrics.compute_metrics(model, ...)`` applies it itself, hence
        ``QuantizationCallback`` per-epoch ``metrics/*`` and
        ``MetricsHistoryCallback`` both see the materialized weights while
        frozen, whatever the callback order, and post-hoc scripts need nothing.
      * Anything that WRITES weights (projection, finalize) must run on a plain
        model: the callback never calls them while parametrized (the penalty
        closure returns 0 while frozen, the validation projection is skipped) and
        ``_finalize_soft_momos`` (validate/test start) unfreezes first if the model
        is still parametrized.
      * ``k`` / ``capacity`` are resolved at ``on_fit_start`` (plain model), bit
        accounting runs after ``fit`` (plain model again).

``QuantizationCallback`` (``src/utils/callbacks.py``), methods ``soft_momos`` /
``soft_momos_plus``
------------------------------------------------------------------------------
cfg keys (besides the soft_momos ones): ``warm_start`` (bool, default False),
``reseed_dead`` (bool, default True), ``lloyd_iters`` (int, default 1),
``freeze_fraction`` (float, default 0.0). ``_build_quantization_callback``
treats ``soft_momos_plus`` like ``soft_momos`` (sets ``s``, requires k/capacity).

New public attributes / methods:
    ``callback.frozen``       bool; True from the freeze start on (stays True
                              after ``on_fit_end``; it means "mosaic frozen, no
                              more penalty/draws/projection").
    ``callback.Z``            the shared ``nn.Parameter`` (None before freeze).
    ``callback.freeze_epoch(max_epochs) -> int | None``
        ``max_epochs - ceil(freeze_fraction * max_epochs)`` clamped to >= 0;
        ``None`` when ``freeze_fraction == 0``.

Hooks (``utils.callbacks`` imports ``warm_start_mosaic`` (and ``quantize``) BY NAME,
like ``draw_mosaic``; two tests patch them there to observe call arguments. Whether
a model is parametrized is read off the model, there is no ``callback.parametrized``
requirement):
    * ``on_train_epoch_start`` at epoch ``e``: (1) the usual draw -- random at
      ``e == 0`` or when ``warm_start`` is off, ``warm_start_mosaic(model, M_prev
      = self.M, ...)`` otherwise (and also random if ``self.M is None``); the
      ``counts`` cache used by the penalty is refreshed; ``quant/mosaic_changed``
      and ``quant/dead_motifs`` are logged for warm-start draws; (2) lambda
      schedule as before; (3) if ``e == freeze_epoch`` (and not yet frozen),
      FREEZE START. Whether the freeze-start epoch still draws is UNSPECIFIED
      (drawing first is a Lloyd step under warm start, so projecting onto its
      group means is the right codebook; reusing the previous ``M`` is also
      valid); ``M`` must exist (a draw happens if it is ``None``), and from the
      epoch after the freeze start on there are no draws at all (``M`` unchanged).
    * FREEZE START: the weights become ``Z[M]`` with ``Z`` the group means of the
      weights as they are then (``freeze_codebook`` computes ``Z`` itself and the
      parametrized weights are ``Z[M]``; an explicit ``project_fixed_mosaic`` of
      the now-unread originals first is allowed but unobservable); ``frozen =
      True``; ``callback.Z`` is the shared param. If ``trainer.optimizers`` is non-empty: ONE new param
      group ``{"params": [Z], "lr": <current lr of group 0>, "weight_decay": 0}``
      is added to ``trainer.optimizers[0]``, every scheduler in
      ``trainer.lr_scheduler_configs`` (``cfg.scheduler``) gets matching entries
      in ``base_lrs`` / ``_last_lr`` so it keeps stepping group 1 in lockstep with
      group 0, and the optimizer state of the now-frozen original params is
      dropped (``optimizer.state.pop(p)``). A trainer without optimizers works.
    * penalty closure: returns a 0-dim tensor equal to 0.0 while ``frozen``.
    * ``on_validation_epoch_start/end``: no projection / restore while frozen
      (weights, ``Z``, ``M`` untouched).
    * ``on_save_checkpoint(trainer, pl_module, checkpoint)``: while the model is
      parametrized, ``checkpoint["state_dict"]`` is rewritten to the PLAIN
      keys (``*.parametrizations.<n>.original`` and ``Z`` entries dropped) with
      the materialized ``Z[M]`` values; the live model is unchanged. No-op when
      not parametrized.
    * ``on_fit_end`` / ``on_exception``: ``unfreeze_codebook`` (idempotent; the model is
      plain again, ``frozen`` stays True).
    * ``state_dict()`` additionally carries ``"Z"`` (tensor) and ``"frozen"``;
      ``load_state_dict`` restores them (``callback.Z`` / ``callback.frozen``).
      Resuming a fit from a checkpoint written in the frozen phase is NOT part of
      this contract (the saved optimizer has the extra param group).
    * ``finalize`` (validate/test): when ``frozen`` (from training or restored by
      ``load_state_dict``) NO k-means runs: the callback unfreezes if needed and
      calls ``finalize_from_codebook(model, Z, M, rows, cols, force_zero)``
      (``quantizers.soft_momos``), which adopts the codebook ``Z`` (row 0 zeroed
      under force_zero) and mosaic ``M`` and returns the stats keys of
      ``finalize_soft_momos`` (distortion computed against the weights, exactly 0.0;
      ``num_changed_weights`` 0; ``num_motifs_used`` <= k; ``motif_counts``). The same
      ``final_motifs.pt`` writer and ``quant/final_*`` logs are used. This is by
      flags (``freeze_fraction > 0``), also for ``method == "soft_momos"``. Without
      a frozen phase the k-means++ finalize (``final_k``) is unchanged;
      ``finalize_soft_momos`` itself is the committed one (no unique-blocks shortcut).

Registry / bits / tuning
------------------------
``quantizers.METHODS["soft_momos_plus"] is quantize_soft_momos``;
``compute_quantization_bits(method="soft_momos_plus")`` == the soft_momos result
when ``freeze_fraction == 0`` (``final_k`` required); with ``freeze_fraction > 0``
the stored codebook is the trained one, so bits use ``k`` (or capacity) and ignore
``final_k`` (optional, unused; the same holds for the on_fit_start bit estimate).
``METHODS["soft_momos_plus"]`` stays ``quantize_soft_momos`` (standalone k-means call,
needs ``final_k``; the callback does not use it after a frozen phase).
``resolve_block_shape("soft_momos_plus", cfg)`` ==
``(rows, cols)``. ``tune_quantization._suggest_quant_config`` for
``"soft_momos_plus"`` samples exactly as ``soft_momos`` but with param prefix
``smp_`` (``smp_rows``, ``smp_cols``, ``smp_lam``, optionally
``smp_capacity``), returns ``method="soft_momos_plus"``, ``warm_start=True``,
``reseed_dead=True``, ``lloyd_iters`` (space value, default 1) and
``freeze_fraction`` read from ``search_space.soft_momos_plus.freeze_fraction``:
a ``{low, high}`` spec -> ``trial.suggest_float("smp_freeze_fraction", low, high)``,
a plain number -> that fixed value (no trial parameter).
Config files: ``src/configs/quantization/soft_momos_plus.yaml`` (method
soft_momos_plus, warm_start true, reseed_dead true, lloyd_iters 1,
freeze_fraction 0.1, no ``final_k``, no ``finalize`` block); ``soft_momos.yaml`` gains ``warm_start: false``,
``freeze_fraction: 0.0``.
"""

import copy
import inspect
import os
import tempfile
import time
import unittest
from contextlib import ExitStack
from functools import reduce
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import lightning as L
import torch
import torch.nn as nn
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf
from torch.nn.utils import parametrize
from torch.utils.data import DataLoader, TensorDataset

import quantizers
import quantizers.block_utils as block_utils
import quantizers.soft_momos as sm
import utils.callbacks as callbacks_mod
from quantizers.block_utils import iter_trainable_params
from quantizers.momos2d import _get_model_blocks2D
from quantizers.soft_momos import (
    CodebookTensor,
    draw_mosaic,
    freeze_codebook,
    group_means,
    materialized_weights,
    quantize_soft_momos,
    soft_momos_penalty,
    unfreeze_codebook,
    warm_start_mosaic,
)
from tests.test_soft_momos import (
    PADDED_M,
    FakeTrainer,
    FourValueModel,
    PaddedLinearModel,
    RandomLinearModel,
    TINY_COLS,
    TINY_ROWS,
    _assert_params_equal,
    _blocks_to_param_tensors,
    _loop_group_means,
    _snapshot,
    _soft_cfg,
    _tiny_backbone,
    _unique_blocks,
    _lit,
)
from utils.callbacks import MetricsHistoryCallback, _build_quantization_callback
from utils.metrics import compute_metrics, resolve_block_shape
from utils.quant_bits import compute_quantization_bits

R, C = TINY_ROWS, TINY_COLS
K = 8  # training mosaic size (= stored codebook size after a frozen phase)
REPO = Path(__file__).resolve().parent.parent


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _forbid_kmeans(stack, spy):
    """Enter patches (on ``stack``) so any k-means path fails the test, and spy on
    ``finalize_from_codebook`` (results appended to ``spy``)."""
    def boom(*a, **kw):
        raise AssertionError("k-means must not run after a frozen phase")

    real = callbacks_mod.finalize_from_codebook

    def wrapped(*a, **kw):
        spy.append(real(*a, **kw))
        return spy[-1]

    for obj, name, new in (
        (callbacks_mod, "quantize", boom),
        (sm, "finalize_soft_momos", boom),
        (sm, "KMeans", boom),
        (sm, "MiniBatchKMeans", boom),
        (callbacks_mod, "finalize_from_codebook", wrapped),
    ):
        stack.enter_context(patch.object(obj, name, new))


def _plus_cfg(**overrides):
    cfg = _soft_cfg(
        method="soft_momos_plus",
        k=K,
        final_k=K,
        warm_start=True,
        reseed_dead=True,
        lloyd_iters=1,
        freeze_fraction=0.2,
    )
    cfg.update(overrides)
    return cfg


def _build_plus(**overrides):
    return _build_quantization_callback({}, OmegaConf.create(_plus_cfg(**overrides)))


class PlusTrainer(FakeTrainer):
    """FakeTrainer that also exposes what Lightning exposes for optimizers:
    ``trainer.optimizers`` (list of Optimizer) and ``trainer.lr_scheduler_configs``
    (list of LRSchedulerConfig, ``cfg.scheduler`` is the scheduler)."""

    def __init__(self, optimizers=None, schedulers=None, **kw):
        super().__init__(**kw)
        self.optimizers = list(optimizers or [])
        self.lr_scheduler_configs = [SimpleNamespace(scheduler=s) for s in (schedulers or [])]


def _get_attr(model, dotted):
    return reduce(getattr, dotted.split("."), model)


def _plain_copy(model):
    """A plain tiny backbone holding the (materialized) weights of ``model``."""
    plain = _tiny_backbone()
    with torch.no_grad():
        for name, p in plain.named_parameters():
            p.copy_(_get_attr(model, name))
    return plain


def _materialized(model):
    return [_get_attr(model, n).detach().clone() for n, _ in _tiny_backbone().named_parameters()]


def _expected_from(plain, Z, M, rows=R, cols=C):
    """Per-param tensors equal to un-blocked ``Z[M]`` using ``plain``'s layout."""
    _, specs = _get_model_blocks2D(plain, rows, cols)
    return _blocks_to_param_tensors(Z.detach()[M], specs, rows, cols)


def _is_parametrized(model):
    return any(parametrize.is_parametrized(m) for m in model.modules())


def _no_parametrizations(model):
    return not any(parametrize.is_parametrized(m) for m in model.modules()) and not any(
        "parametrizations" in k for k in model.state_dict()
    )


def _batch(seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(16, 16, generator=g), torch.randint(0, 4, (16,), generator=g)


def _train_steps(pl, opt, n, sched=None):
    x, y = _batch()
    for _ in range(n):
        opt.zero_grad()
        pl.criterion(pl(x), y).backward()
        opt.step()
    if sched is not None:
        sched.step()


def _perturb(model, scale=0.1, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            p.add_(scale * torch.randn(p.shape, generator=g))


def _setup(seed=0, max_epochs=10, **cfg):
    """Callback + LitClassifier + real AdamW/CosineAnnealingLR on a PlusTrainer,
    after on_fit_start, with a populated Adam state. No epoch hook run yet."""
    cb = _build_plus(**cfg)
    pl = _lit(seed=seed, epochs=max_epochs)
    out = pl.configure_optimizers()
    opt, sched = out["optimizer"], out["lr_scheduler"]["scheduler"]
    trainer = PlusTrainer([opt], [sched], max_epochs=max_epochs)
    cb.on_fit_start(trainer, pl)
    _train_steps(pl, opt, 2)  # populate the Adam moments of every param
    return SimpleNamespace(cb=cb, pl=pl, opt=opt, sched=sched, trainer=trainer, max_epochs=max_epochs)


def _epoch(s, e):
    s.trainer.current_epoch = e
    s.cb.on_train_epoch_start(s.trainer, s.pl)


def _run_until_frozen(s):
    """Drive epochs 0..freeze_epoch (moving the weights a bit and stepping the
    scheduler between epochs, like a real run). Returns the pre-freeze plain
    copy (layout reference) taken right before the freeze-start epoch."""
    fe = s.cb.freeze_epoch(s.max_epochs)
    plain = None
    for e in range(fe + 1):
        if e == fe:
            plain = copy.deepcopy(s.pl.model)
        _epoch(s, e)
        if e < fe:
            _perturb(s.pl.model, 0.05, seed=e)
            s.sched.step()
    return plain


# -----------------------------------------------------------------------------
# warm_start_mosaic
# -----------------------------------------------------------------------------


def _full_use_prev(m=64, k=8):
    return torch.arange(m) % k  # every motif has members


class TestWarmStartMosaic(unittest.TestCase):
    def test_is_one_lloyd_step_and_leaves_weights_untouched(self):
        model = RandomLinearModel(seed=1)
        before = _snapshot(model)
        blocks, specs_ref = _get_model_blocks2D(model, 2, 2)
        M_prev = _full_use_prev()

        M, Z, specs, stats = warm_start_mosaic(model, M_prev, 2, 2, 8, force_zero=True)

        _assert_params_equal(model, before)
        Z_ref = _loop_group_means(blocks, M_prev, 8, True)
        d = (blocks[:, None, :] - Z_ref[None]).square().sum(2)
        self.assertEqual(M.dtype, torch.long)
        self.assertEqual(M.shape, (64,))
        self.assertTrue(torch.allclose(Z, Z_ref, atol=1e-6))
        self.assertTrue(torch.equal(Z[0], torch.zeros(4)))
        d_assigned = (blocks - Z[M]).square().sum(1)
        self.assertTrue(torch.allclose(d_assigned, d.min(1).values, atol=1e-5))
        self.assertEqual([s[1] for s in specs], [s[1] for s in specs_ref])
        self.assertIs(specs[0][0], model.fc.weight)
        self.assertEqual(stats["dead_motifs"], 0)
        self.assertAlmostEqual(stats["mosaic_changed"], float((M != M_prev).float().mean()))
        for key in ("mosaic_changed", "dead_motifs"):
            self.assertIsInstance(stats[key], (int, float), key)
            self.assertNotIsInstance(stats[key], (bool, torch.Tensor), key)

    def test_nonzero_motif_zero_group_is_not_dead_and_stays_zero(self):
        # No block was assigned to motif 0: under force_zero it is the pinned
        # zero motif, not a dead one (no reseed, not counted).
        model = RandomLinearModel(seed=2)
        M_prev = 1 + torch.arange(64) % 7  # motifs 1..7, never 0
        torch.manual_seed(0)
        M, Z, _, stats = warm_start_mosaic(model, M_prev, 2, 2, 8, force_zero=True)
        self.assertTrue(torch.equal(Z[0], torch.zeros(4)))
        self.assertEqual(stats["dead_motifs"], 0)

    def test_dead_motifs_are_reseeded_from_current_blocks(self):
        model = RandomLinearModel(seed=3)
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        M_prev = torch.arange(64) % 3  # motifs 0,1,2 used; 3,4,5 dead (k=6)
        torch.manual_seed(7)
        M, Z, _, stats = warm_start_mosaic(
            model, M_prev, 2, 2, 6, force_zero=True, reseed_dead=True
        )
        self.assertEqual(stats["dead_motifs"], 3)
        Z_ref = _loop_group_means(blocks, M_prev, 6, True)
        self.assertTrue(torch.allclose(Z[:3], Z_ref[:3], atol=1e-6))
        self.assertTrue(torch.equal(Z[0], torch.zeros(4)))
        for j in (3, 4, 5):
            self.assertTrue((blocks == Z[j]).all(dim=1).any(), f"Z[{j}] is not a current block")
        self.assertEqual(torch.unique(Z[3:], dim=0).shape[0], 3, "reseeds must be distinct")
        # Each reseeded motif is at distance 0 from its own source block, hence
        # it must be used by the new mosaic.
        counts = torch.bincount(M, minlength=6)
        self.assertTrue((counts[3:] >= 1).all(), counts)

    def test_dead_motifs_stay_zero_without_reseed(self):
        model = RandomLinearModel(seed=3)
        M_prev = torch.arange(64) % 3
        M, Z, _, stats = warm_start_mosaic(
            model, M_prev, 2, 2, 6, force_zero=True, reseed_dead=False
        )
        self.assertEqual(stats["dead_motifs"], 3)
        self.assertTrue(torch.equal(Z[3:], torch.zeros(3, 4)))
        # zero rows tie with motif 0 (which id wins is up to the assignment): the
        # chosen motif must be a nearest one, and a zero row is never better than
        # motif 0, so no block is strictly closer to a dead motif than to 0.
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        d = (blocks[:, None, :] - Z[None]).square().sum(2)
        self.assertTrue(torch.allclose((blocks - Z[M]).square().sum(1), d.min(1).values, atol=1e-5))

    def test_without_force_zero_motif_zero_can_be_dead_and_is_reseeded(self):
        model = RandomLinearModel(seed=4)
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        M_prev = 1 + torch.arange(64) % 3  # k=4: motif 0 has no members
        torch.manual_seed(1)
        _, Z, _, stats = warm_start_mosaic(model, M_prev, 2, 2, 4, force_zero=False)
        self.assertEqual(stats["dead_motifs"], 1)
        self.assertTrue((blocks == Z[0]).all(dim=1).any())
        self.assertGreater(float(Z[0].abs().sum()), 0.0)

    def test_reseeding_is_deterministic_under_the_global_seed(self):
        model = RandomLinearModel(seed=5)
        M_prev = torch.arange(64) % 3
        torch.manual_seed(11)
        a = warm_start_mosaic(model, M_prev, 2, 2, 6)
        torch.manual_seed(11)
        b = warm_start_mosaic(model, M_prev, 2, 2, 6)
        self.assertTrue(torch.equal(a[0], b[0]))
        self.assertTrue(torch.equal(a[1], b[1]))

    def test_two_lloyd_iters_equal_applying_one_iteration_twice(self):
        model = RandomLinearModel(seed=6)
        M_prev = _full_use_prev()
        M1, _, _, _ = warm_start_mosaic(model, M_prev, 2, 2, 8, reseed_dead=False, lloyd_iters=1)
        M2, Z2, _, _ = warm_start_mosaic(model, M1, 2, 2, 8, reseed_dead=False, lloyd_iters=1)
        Mn, Zn, _, stats = warm_start_mosaic(
            model, M_prev, 2, 2, 8, reseed_dead=False, lloyd_iters=2
        )
        self.assertFalse(torch.equal(M1, M_prev), "test needs a mosaic that actually moves")
        self.assertTrue(torch.equal(Mn, M2))
        self.assertTrue(torch.allclose(Zn, Z2, atol=1e-6))
        # mosaic_changed is always relative to the M_prev argument
        self.assertAlmostEqual(stats["mosaic_changed"], float((Mn != M_prev).float().mean()))

    def test_fixed_point_reports_no_change(self):
        # Weights that are exactly 4 distinct block values with M_prev the true
        # classes: means == values, nearest == classes -> nothing changes, even
        # with spare (dead) motifs, which are reseeded with duplicates of existing
        # values or left zero (a duplicate ties with the true id, so only the
        # quantized blocks, not the ids, are pinned when there are spare motifs).
        model = FourValueModel()
        blocks, _ = _get_model_blocks2D(model, 2, 2)
        M_prev = torch.arange(64) % 4
        for k, reseed in ((4, True), (6, False), (6, True)):
            torch.manual_seed(0)
            M, Z, _, stats = warm_start_mosaic(model, M_prev, 2, 2, k, reseed_dead=reseed)
            self.assertTrue(torch.equal(Z[M], blocks), (k, reseed))
            self.assertEqual(stats["dead_motifs"], k - 4)
            if k == 4:
                self.assertTrue(torch.equal(M, M_prev))
                self.assertEqual(stats["mosaic_changed"], 0.0)

    def test_goes_through_chunked_assignment_with_forwarded_chunk_size(self):
        model = RandomLinearModel(seed=7)
        calls = []
        real = block_utils._nearest_motifs_chunked

        def spy(blocks, motifs, *args, **kwargs):
            calls.append((args, kwargs))
            return real(blocks, motifs, *args, **kwargs)

        targets = [block_utils]
        if hasattr(sm, "_nearest_motifs_chunked"):
            targets.append(sm)
        with ExitStack() as stack:
            for t in targets:
                stack.enter_context(patch.object(t, "_nearest_motifs_chunked", spy))
            warm_start_mosaic(model, _full_use_prev(), 2, 2, 8, lloyd_iters=2, chunk_size=7)
        self.assertEqual(len(calls), 2, "one chunked assignment per Lloyd iteration")
        for args, kwargs in calls:
            self.assertEqual(kwargs.get("chunk_size", args[0] if args else None), 7)

    def test_everything_but_the_assignment_is_linear_time(self):
        # 400k blocks (rows=1, cols=2), k=200k, 199k motifs dead and reseeded.
        # The assignment (m x k distances) is stubbed out; what remains (group
        # means, dead detection, reseeding) must not loop over motifs in python
        # (vectorised ~0.2 s, a python loop over the dead motifs ~2 s).
        m, k = 400_000, 200_000
        model = nn.Module()
        model.w = nn.Parameter(torch.randn(2 * m))
        M_prev = torch.arange(m) % 1000  # motifs 0..999 used, all others dead
        stub = lambda blocks, motifs, *a, **kw: torch.zeros(blocks.size(0), dtype=torch.long)
        with ExitStack() as stack:
            stack.enter_context(patch.object(block_utils, "_nearest_motifs_chunked", stub))
            if hasattr(sm, "_nearest_motifs_chunked"):
                stack.enter_context(patch.object(sm, "_nearest_motifs_chunked", stub))
            t0 = time.perf_counter()
            _, Z, _, stats = warm_start_mosaic(model, M_prev, 1, 2, k, reseed_dead=True)
            elapsed = time.perf_counter() - t0
        self.assertEqual(Z.shape, (k, 2))
        self.assertEqual(stats["dead_motifs"], k - 1000)
        self.assertLess(elapsed, 1.0)


# -----------------------------------------------------------------------------
# CodebookTensor / freeze_codebook / unfreeze_codebook
# -----------------------------------------------------------------------------


def _keys(model):
    return list(model.state_dict().keys())


class TestFreezeCodebook(unittest.TestCase):
    def test_materialized_weights_are_Z_of_M_with_padding_dropped(self):
        model = PaddedLinearModel()
        plain = copy.deepcopy(model)
        blocks, specs = _get_model_blocks2D(plain, 2, 2)

        Z, handles = freeze_codebook(model, PADDED_M, 2, 2, 3, force_zero=True)

        Z_ref = group_means(blocks, PADDED_M, 3, True)
        self.assertIsInstance(Z, nn.Parameter)
        self.assertTrue(Z.requires_grad)
        self.assertEqual(Z.shape, (3, 4))
        self.assertTrue(torch.equal(Z.detach(), Z_ref))
        self.assertTrue(torch.equal(Z[0].detach(), torch.zeros(4)))
        w_ref, b_ref = _blocks_to_param_tensors(Z_ref[PADDED_M], specs, 2, 2)
        self.assertTrue(torch.equal(model.fc.weight.detach(), w_ref))
        self.assertTrue(torch.equal(model.fc.bias.detach(), b_ref))
        self.assertEqual(model.fc.weight.shape, (3, 5))
        self.assertTrue(parametrize.is_parametrized(model.fc, "weight"))
        self.assertTrue(parametrize.is_parametrized(model.fc, "bias"))
        self.assertIsInstance(model.fc.parametrizations.weight[0], CodebookTensor)
        self.assertGreater(len(handles), 0)

    def test_Z_is_the_only_trainable_parameter_and_forward_ignores_originals(self):
        model = PaddedLinearModel()
        Z, _ = freeze_codebook(model, PADDED_M, 2, 2, 3)
        trainable = [p for p in model.parameters() if p.requires_grad]
        self.assertEqual(len(trainable), 1)
        self.assertIs(trainable[0], Z)
        orig = model.fc.parametrizations.weight.original
        self.assertFalse(orig.requires_grad)
        self.assertFalse(model.fc.parametrizations.bias.original.requires_grad)

        w = model.fc.weight.detach().clone()
        with torch.no_grad():
            orig.add_(5.0)
        self.assertTrue(torch.equal(model.fc.weight.detach(), w))

    def test_zero_motif_row_is_pinned(self):
        model = PaddedLinearModel()
        Z, _ = freeze_codebook(model, PADDED_M, 2, 2, 3, force_zero=True)
        with torch.no_grad():
            Z[0] = 7.0  # even a perturbed Z[0] must not leak into the weights
        blocks_of_group0 = (PADDED_M == 0).nonzero().flatten()
        self.assertEqual(blocks_of_group0.numel(), 2)
        w = model.fc.weight.detach()
        # block 0 is the top-left 2x2 patch of the weight and belongs to motif 0
        self.assertTrue(torch.equal(w[:2, :2], torch.zeros(2, 2)))
        # ... and receives no gradient
        loss = (model.fc.weight * torch.randn(3, 5)).sum() + model.fc.bias.sum()
        loss.backward()
        self.assertTrue(torch.equal(Z.grad[0], torch.zeros(4)))

    def test_without_force_zero_row_zero_is_a_regular_motif(self):
        model = PaddedLinearModel()
        blocks, _ = _get_model_blocks2D(copy.deepcopy(model), 2, 2)
        Z, _ = freeze_codebook(model, PADDED_M, 2, 2, 3, force_zero=False)
        self.assertTrue(torch.allclose(Z[0].detach(), blocks[PADDED_M == 0].mean(0), atol=1e-6))
        self.assertGreater(float(model.fc.weight.detach()[:2, :2].abs().sum()), 0.0)

    def test_grad_of_Z_is_index_add_of_weight_grads(self):
        # Correctness of the O(n) backward: dL/dZ_j = sum of the per-weight grads
        # of every block in group j (padded positions contribute nothing).
        model = PaddedLinearModel(seed=2)
        plain = copy.deepcopy(model)
        coef_w, coef_b = torch.randn(3, 5), torch.randn(3)
        Z, _ = freeze_codebook(model, PADDED_M, 2, 2, 3, force_zero=True)
        ((model.fc.weight * coef_w).sum() + (model.fc.bias * coef_b).sum()).backward()

        with torch.no_grad():  # per-weight grads laid out like the params
            plain.fc.weight.copy_(coef_w)
            plain.fc.bias.copy_(coef_b)
        gblocks, _ = _get_model_blocks2D(plain, 2, 2)
        expected = torch.zeros(3, 4).index_add_(0, PADDED_M, gblocks)
        expected[0] = 0.0
        self.assertTrue(torch.allclose(Z.grad, expected, atol=1e-6))

    def test_a_step_on_Z_moves_weights_only_through_Z(self):
        model = PaddedLinearModel(seed=3)
        _, specs = _get_model_blocks2D(copy.deepcopy(model), 2, 2)
        Z, _ = freeze_codebook(model, PADDED_M, 2, 2, 3)
        originals = [
            model.fc.parametrizations.weight.original.detach().clone(),
            model.fc.parametrizations.bias.original.detach().clone(),
        ]
        Z_before = Z.detach().clone()
        opt = torch.optim.SGD([Z], lr=0.1)
        (model.fc(torch.randn(4, 5)).square().sum()).backward()
        opt.step()
        self.assertFalse(torch.equal(Z.detach(), Z_before))
        self.assertTrue(torch.equal(Z[0].detach(), torch.zeros(4)))
        for p, o in zip((model.fc.parametrizations.weight.original,
                         model.fc.parametrizations.bias.original), originals):
            self.assertTrue(torch.equal(p.detach(), o))
        # group-0 blocks are masked to zero by the pinned row
        keep = (PADDED_M != 0).float().unsqueeze(1)
        w_ref, b_ref = _blocks_to_param_tensors(Z.detach()[PADDED_M] * keep, specs, 2, 2)
        self.assertTrue(torch.equal(model.fc.weight.detach(), w_ref))
        self.assertTrue(torch.equal(model.fc.bias.detach(), b_ref))

    def test_unfreeze_leaves_plain_model_with_bit_exact_materialized_weights(self):
        model = PaddedLinearModel()
        plain_keys = _keys(model)
        Z, handles = freeze_codebook(model, PADDED_M, 2, 2, 3)
        self.assertNotEqual(_keys(model), plain_keys)
        w, b = model.fc.weight.detach().clone(), model.fc.bias.detach().clone()

        unfreeze_codebook(model, handles)

        self.assertFalse(parametrize.is_parametrized(model.fc))
        self.assertEqual(_keys(model), plain_keys)
        self.assertTrue(torch.equal(model.fc.weight.detach(), w))
        self.assertTrue(torch.equal(model.fc.bias.detach(), b))
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        self.assertTrue(all(p is not Z for p in model.parameters()))
        self.assertIsInstance(model.fc.weight, nn.Parameter)


# -----------------------------------------------------------------------------
# Callback: warm start
# -----------------------------------------------------------------------------


class TestWarmStartCallback(unittest.TestCase):
    def test_build_treats_plus_like_soft_momos(self):
        cb = _build_plus()
        self.assertEqual(cb.method, "soft_momos_plus")
        self.assertEqual(cb.quant_cfg["s"], R * C)
        self.assertNotIn("enabled", cb.quant_cfg)
        with self.assertRaises(ValueError):
            _build_plus(k=None, capacity=None)

    def test_fit_start_registers_one_closure_for_plus(self):
        cb, pl = _build_plus(), _lit()
        cb.on_fit_start(PlusTrainer(), pl)
        cb.on_fit_start(PlusTrainer(), pl)
        self.assertEqual(len(pl.aux_losses), 1)

    def test_epoch_zero_draw_is_identical_to_soft_momos(self):
        for flags in ({"warm_start": True}, {"warm_start": False}):
            base = _build_quantization_callback(
                {}, OmegaConf.create(_soft_cfg(k=K, final_k=K))
            )
            plus = _build_plus(freeze_fraction=0.0, **flags)
            pl_a, pl_b = _lit(seed=3), _lit(seed=3)
            base.on_fit_start(PlusTrainer(), pl_a)
            plus.on_fit_start(PlusTrainer(), pl_b)
            torch.manual_seed(5)
            base.on_train_epoch_start(PlusTrainer(current_epoch=0), pl_a)
            torch.manual_seed(5)
            plus.on_train_epoch_start(PlusTrainer(current_epoch=0), pl_b)
            self.assertTrue(torch.equal(base.M, plus.M), flags)

    def test_later_epoch_is_nearest_of_group_means_of_previous_mosaic(self):
        for method in ("soft_momos_plus", "soft_momos"):  # flags work on both
            cb = _build_plus(method=method, freeze_fraction=0.0)
            pl = _lit(seed=4)
            cb.on_fit_start(PlusTrainer(), pl)
            torch.manual_seed(0)
            cb.on_train_epoch_start(PlusTrainer(current_epoch=0), pl)
            M0 = cb.M.clone()
            _perturb(pl.model, 0.1, seed=1)
            before = _snapshot(pl.model)
            blocks, _ = _get_model_blocks2D(pl.model, R, C)
            Z = _loop_group_means(blocks, M0, K, True)
            counts = torch.bincount(M0, minlength=K)
            self.assertTrue((counts[1:] > 0).all(), "precondition: no dead motif")
            expected = (blocks[:, None, :] - Z[None]).square().sum(2).argmin(1)

            cb.on_train_epoch_start(PlusTrainer(current_epoch=1), pl)

            _assert_params_equal(pl.model, before)
            self.assertTrue(torch.equal(cb.M, expected), method)

    def test_warm_start_off_keeps_the_random_draw(self):
        cb = _build_plus(warm_start=False, freeze_fraction=0.0)
        pl = _lit(seed=4)
        cb.on_fit_start(PlusTrainer(), pl)
        for e in (0, 1):
            torch.manual_seed(9 + e)
            cb.on_train_epoch_start(PlusTrainer(current_epoch=e), pl)
            torch.manual_seed(9 + e)
            ref = draw_mosaic(pl.model, R, C, K, True)[0]
            self.assertTrue(torch.equal(cb.M, ref), e)

    def test_penalty_uses_the_refreshed_mosaic_counts_after_warm_start(self):
        cb = _build_plus(freeze_fraction=0.0, lam=0.7)
        pl = _lit(seed=5)
        cb.on_fit_start(PlusTrainer(), pl)
        cb.on_train_epoch_start(PlusTrainer(current_epoch=0), pl)
        _perturb(pl.model, 0.2, seed=3)
        cb.on_train_epoch_start(PlusTrainer(current_epoch=1), pl)
        value = pl.aux_losses[0](pl.model)
        ref, _ = soft_momos_penalty(pl.model, cb.M, R, C, K, True)
        self.assertTrue(torch.allclose(value.detach(), 0.7 * ref.detach(), atol=1e-7))

    def test_lloyd_iters_and_reseed_flags_are_forwarded(self):
        pl = _lit(seed=6)
        for iters, reseed in ((2, False), (1, True)):
            cb = _build_plus(freeze_fraction=0.0, lloyd_iters=iters, reseed_dead=reseed)
            cb.on_fit_start(PlusTrainer(), pl)
            cb.on_train_epoch_start(PlusTrainer(current_epoch=0), pl)
            real = sm.warm_start_mosaic
            with patch.object(callbacks_mod, "warm_start_mosaic", wraps=real) as spy:
                cb.on_train_epoch_start(PlusTrainer(current_epoch=1), pl)
            self.assertEqual(spy.call_count, 1)
            bound = inspect.signature(real).bind(*spy.call_args.args, **spy.call_args.kwargs)
            self.assertEqual(bound.arguments.get("lloyd_iters"), iters)
            self.assertEqual(bound.arguments.get("reseed_dead"), reseed)

    def test_flags_off_by_default_for_plain_soft_momos(self):
        cb = _build_quantization_callback({}, OmegaConf.create(_soft_cfg()))
        self.assertIsNone(cb.freeze_epoch(10))
        self.assertFalse(cb.frozen)

    def test_plus_validation_projects_and_finalizes_like_soft_momos(self):
        cb = _build_plus(freeze_fraction=0.0, final_k=4)
        pl = _lit(seed=7)
        cb.on_fit_start(PlusTrainer(), pl)
        cb.on_train_epoch_start(PlusTrainer(current_epoch=0), pl)
        raw = _snapshot(pl.model)
        tr = PlusTrainer(current_epoch=0)
        cb.on_validation_epoch_start(tr, pl)
        self.assertLessEqual(_unique_blocks(pl.model, R, C).shape[0], K)
        cb.on_validation_epoch_end(tr, pl)
        _assert_params_equal(pl.model, raw)
        cb.on_validation_start(PlusTrainer(fn=TrainerFn.VALIDATING), pl)
        self.assertTrue(cb.finalized)
        self.assertLessEqual(_unique_blocks(pl.model, R, C).shape[0], 4)


# -----------------------------------------------------------------------------
# Callback: freeze schedule
# -----------------------------------------------------------------------------


class TestFreezeSchedule(unittest.TestCase):
    def test_freeze_epoch_formula(self):
        table = [
            (10, 0.1, 9),
            (10, 0.25, 7),  # ceil(2.5) = 3 frozen epochs
            (10, 0.2, 8),
            (10, 1.0, 0),
            (10, 0.0, None),
            (1, 0.5, 0),
            (3, 0.01, 2),  # at least one frozen epoch when the fraction is > 0
            (100, 0.1, 90),
        ]
        for max_epochs, frac, expected in table:
            cb = _build_plus(freeze_fraction=frac)
            self.assertEqual(cb.freeze_epoch(max_epochs), expected, (max_epochs, frac))

    def test_freezes_exactly_from_the_freeze_epoch_on(self):
        for method in ("soft_momos_plus", "soft_momos"):
            s = _setup(method=method, freeze_fraction=0.25)  # epochs 7, 8, 9
            seen = []
            for e in range(10):
                _epoch(s, e)
                seen.append((s.cb.frozen, _is_parametrized(s.pl.model)))
                if e < 9:
                    s.sched.step()
            self.assertEqual(seen, [(False, False)] * 7 + [(True, True)] * 3, method)

    def test_fraction_zero_never_freezes(self):
        s = _setup(freeze_fraction=0.0)
        for e in range(10):
            _epoch(s, e)
        self.assertFalse(s.cb.frozen)
        self.assertEqual(len(s.opt.param_groups), 1)
        self.assertTrue(_no_parametrizations(s.pl.model))

    def test_mosaic_and_codebook_are_fixed_after_freeze_start(self):
        s = _setup(freeze_fraction=0.25)
        _run_until_frozen(s)
        M, Z = s.cb.M.clone(), s.cb.Z.detach().clone()
        _train_steps(s.pl, s.opt, 1)  # moves Z
        Z_trained = s.cb.Z.detach().clone()
        for e in (8, 9):
            _epoch(s, e)
        self.assertTrue(torch.equal(s.cb.M, M), "no draws once frozen")
        self.assertTrue(torch.equal(s.cb.Z.detach(), Z_trained), "no re-freeze / re-init of Z")
        self.assertFalse(torch.equal(Z, Z_trained))
        self.assertEqual(len(s.opt.param_groups), 2, "exactly one group is ever added")


# -----------------------------------------------------------------------------
# Callback: frozen phase
# -----------------------------------------------------------------------------


class TestFrozenPhase(unittest.TestCase):
    def setUp(self):
        self.s = _setup(seed=1, freeze_fraction=0.2)
        self.plain = _run_until_frozen(self.s)  # taken just before freeze start
        self.cb, self.pl, self.opt = self.s.cb, self.s.pl, self.s.opt

    def test_weights_are_Z_of_M_and_Z_is_the_shared_trainable_parameter(self):
        cb, pl = self.cb, self.pl
        self.assertIsInstance(cb.Z, nn.Parameter)
        self.assertTrue(cb.Z.requires_grad)
        self.assertEqual(cb.Z.shape, (K, R * C))
        self.assertTrue(torch.equal(cb.Z[0].detach(), torch.zeros(R * C)))
        # weights == Z[M] exactly, with group means of the weights as they were
        # after the freeze-start draw (projection happens at freeze start)
        expected = _expected_from(self.plain, cb.Z, cb.M)
        for got, want in zip(_materialized(pl.model), expected):
            self.assertTrue(torch.equal(got, want))
        self.assertLessEqual(_unique_blocks(_plain_copy(pl.model), R, C).shape[0], K)
        self.assertEqual([p for p in pl.model.parameters() if p.requires_grad], [cb.Z])
        for m in (pl.model[1], pl.model[3]):
            self.assertTrue(parametrize.is_parametrized(m, "weight"))
            self.assertTrue(parametrize.is_parametrized(m, "bias"))
            self.assertIsInstance(m.parametrizations.weight[0], CodebookTensor)

    def test_group_Z_equals_group_means_of_the_projected_weights(self):
        # Projection at freeze start is group-mean projection: Z must equal the
        # group means of the (pre-freeze) weights under M.
        blocks, _ = _get_model_blocks2D(self.plain, R, C)
        self.assertTrue(
            torch.allclose(self.cb.Z.detach(), group_means(blocks, self.cb.M, K, True), atol=1e-6)
        )

    def test_optimizer_gets_exactly_one_new_group_with_Z(self):
        opt, cb = self.opt, self.cb
        self.assertEqual(len(opt.param_groups), 2)
        g0, g1 = opt.param_groups
        self.assertEqual(len(g1["params"]), 1)
        self.assertIs(g1["params"][0], cb.Z)
        self.assertEqual(g1["weight_decay"], 0.0)
        self.assertGreater(g0["weight_decay"], 0.0)
        self.assertAlmostEqual(g1["lr"], g0["lr"], places=12)

    def test_adam_state_of_frozen_params_is_dropped(self):
        for m in (self.pl.model[1], self.pl.model[3]):
            for name in ("weight", "bias"):
                orig = getattr(m.parametrizations, name).original
                self.assertNotIn(orig, self.opt.state)
                self.assertFalse(orig.requires_grad)

    def test_scheduler_keeps_stepping_both_groups_in_lockstep(self):
        sched, opt = self.s.sched, self.opt
        self.assertEqual(len(sched.base_lrs), 2)
        self.assertEqual(len(sched.get_last_lr()), 2)
        lr_before = opt.param_groups[1]["lr"]
        for _ in range(2):
            sched.step()  # must not raise
        g0, g1 = opt.param_groups
        self.assertLess(g1["lr"], lr_before)
        self.assertAlmostEqual(g1["lr"], g0["lr"], places=10)
        self.assertEqual(len(sched.get_last_lr()), 2)

    def test_training_step_changes_only_Z_and_stays_on_the_mosaic(self):
        cb, pl, opt = self.cb, self.pl, self.opt
        originals = [
            getattr(m.parametrizations, n).original.detach().clone()
            for m in (pl.model[1], pl.model[3])
            for n in ("weight", "bias")
        ]
        Z_before = cb.Z.detach().clone()
        w_before = _materialized(pl.model)
        _train_steps(pl, opt, 1)
        self.assertFalse(torch.equal(cb.Z.detach(), Z_before))
        self.assertTrue(torch.equal(cb.Z[0].detach(), torch.zeros(R * C)))
        now = [
            getattr(m.parametrizations, n).original.detach()
            for m in (pl.model[1], pl.model[3])
            for n in ("weight", "bias")
        ]
        for a, b in zip(originals, now):
            self.assertTrue(torch.equal(a, b))
        w_after = _materialized(pl.model)
        self.assertTrue(any(not torch.equal(a, b) for a, b in zip(w_before, w_after)))
        for got, want in zip(w_after, _expected_from(self.plain, cb.Z, cb.M)):
            self.assertTrue(torch.equal(got, want))
        self.assertLessEqual(_unique_blocks(_plain_copy(pl.model), R, C).shape[0], K)

    def test_Z_grad_is_index_add_of_per_weight_grads_on_the_real_model(self):
        cb, pl = self.cb, self.pl
        x, y = _batch(3)
        self.opt.zero_grad()
        pl.criterion(pl(x), y).backward()
        gZ = cb.Z.grad.clone()

        ref = _plain_copy(pl.model)
        for p in ref.parameters():
            p.requires_grad_(True)
        nn.CrossEntropyLoss()(ref(x), y).backward()
        gm = copy.deepcopy(ref)
        with torch.no_grad():
            for p, q in zip(gm.parameters(), ref.parameters()):
                p.copy_(q.grad)
        gblocks, _ = _get_model_blocks2D(gm, R, C)
        expected = torch.zeros_like(gZ).index_add_(0, cb.M, gblocks)
        expected[0] = 0.0
        self.assertTrue(torch.allclose(gZ, expected, atol=1e-6))

    def test_penalty_closure_is_zero_when_frozen(self):
        value = self.pl.aux_losses[0](self.pl.model)
        self.assertIsInstance(value, torch.Tensor)
        self.assertEqual(value.dim(), 0)
        self.assertEqual(float(value), 0.0)

    def test_validation_is_a_noop_when_frozen(self):
        cb, pl = self.cb, self.pl
        w, Z, M = _materialized(pl.model), cb.Z.detach().clone(), cb.M.clone()
        tr = PlusTrainer(current_epoch=8)
        cb.on_validation_epoch_start(tr, pl)
        for a, b in zip(_materialized(pl.model), w):
            self.assertTrue(torch.equal(a, b))
        cb.on_validation_epoch_end(tr, pl)
        for a, b in zip(_materialized(pl.model), w):
            self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(cb.Z.detach(), Z))
        self.assertTrue(torch.equal(cb.M, M))
        self.assertTrue(_is_parametrized(pl.model))

    def test_on_save_checkpoint_writes_plain_keys_with_materialized_values(self):
        cb, pl = self.cb, self.pl
        _train_steps(pl, self.opt, 1)
        ckpt = {"state_dict": pl.state_dict(), "epoch": 8}
        self.assertTrue(any("parametrizations" in k for k in ckpt["state_dict"]))
        weights = _materialized(pl.model)

        cb.on_save_checkpoint(PlusTrainer(current_epoch=8), pl, ckpt)

        sd = ckpt["state_dict"]
        self.assertFalse(any("parametrizations" in k for k in sd))
        self.assertEqual(set(sd), set(_lit().state_dict()))
        for (name, _), want in zip(_tiny_backbone().named_parameters(), weights):
            self.assertTrue(torch.equal(sd["model." + name], want), name)
        self.assertTrue(_is_parametrized(pl.model), "the live model must stay parametrized")
        fresh = _lit(seed=99)
        fresh.load_state_dict(sd, strict=True)
        for got, want in zip(fresh.model.parameters(), weights):
            self.assertTrue(torch.equal(got.detach(), want))

    def test_state_dict_round_trips_M_Z_and_frozen(self):
        cb = self.cb
        state = cb.state_dict()
        self.assertTrue(state["frozen"])
        self.assertTrue(torch.equal(state["M"], cb.M))
        self.assertTrue(torch.equal(torch.as_tensor(state["Z"]), cb.Z.detach()))
        cb2 = _build_plus()
        cb2.load_state_dict(state)
        self.assertTrue(cb2.frozen)
        self.assertTrue(torch.equal(cb2.M, cb.M))
        self.assertTrue(torch.equal(cb2.Z.detach(), cb.Z.detach()))

    def test_state_dict_before_freeze_is_not_frozen(self):
        cb, pl = _build_plus(), _lit()
        cb.on_fit_start(PlusTrainer(), pl)
        cb.on_train_epoch_start(PlusTrainer(current_epoch=0), pl)
        cb2 = _build_plus()
        cb2.load_state_dict(cb.state_dict())
        self.assertFalse(cb2.frozen)
        self.assertIsNone(cb2.Z)

    def test_on_fit_end_returns_a_plain_model_with_bit_exact_weights(self):
        cb, pl = self.cb, self.pl
        _train_steps(pl, self.opt, 1)
        weights = _materialized(pl.model)
        cb.on_fit_end(PlusTrainer(current_epoch=9), pl)
        self.assertTrue(cb.frozen)
        self.assertTrue(_no_parametrizations(pl.model))
        self.assertEqual(set(pl.state_dict()), set(_lit().state_dict()))
        for p, w in zip(pl.model.parameters(), weights):
            self.assertTrue(torch.equal(p.detach(), w))
            self.assertTrue(p.requires_grad)
        cb.on_fit_end(PlusTrainer(current_epoch=9), pl)  # idempotent

    def test_on_exception_also_unfreezes(self):
        weights = _materialized(self.pl.model)
        self.cb.on_exception(PlusTrainer(), self.pl, RuntimeError("boom"))
        self.assertTrue(_no_parametrizations(self.pl.model))
        for p, w in zip(self.pl.model.parameters(), weights):
            self.assertTrue(torch.equal(p.detach(), w))
        self.cb.on_fit_end(PlusTrainer(), self.pl)  # then fit_end is harmless

    def test_save_checkpoint_is_a_noop_when_not_parametrized(self):
        cb, pl = _build_plus(), _lit()
        cb.on_fit_start(PlusTrainer(), pl)
        sd = pl.state_dict()
        ckpt = {"state_dict": dict(sd)}
        cb.on_save_checkpoint(PlusTrainer(), pl, ckpt)
        self.assertEqual(list(ckpt["state_dict"]), list(sd))
        for k in sd:
            self.assertTrue(torch.equal(ckpt["state_dict"][k], sd[k]))

    def test_finalize_after_frozen_phase_adopts_codebook_without_kmeans(self):
        cb, pl = self.cb, self.pl
        cb.on_fit_end(PlusTrainer(current_epoch=9), pl)
        weights = _materialized(pl.model)
        seen = []
        with ExitStack() as stack:
            _forbid_kmeans(stack, seen)
            cb.on_validation_start(PlusTrainer(fn=TrainerFn.VALIDATING, current_epoch=9), pl)
        self.assertTrue(cb.finalized)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["distortion"], 0.0)
        self.assertEqual(seen[0]["num_changed_weights"], 0)
        self.assertLessEqual(seen[0]["num_motifs_used"], K)
        for p, w in zip(pl.model.parameters(), weights):
            self.assertTrue(torch.equal(p.detach(), w))


    def test_final_motifs_file_reconstructs_the_weights(self):
        cb, pl = self.cb, self.pl
        cb.on_fit_end(PlusTrainer(current_epoch=9), pl)
        with tempfile.TemporaryDirectory() as tmp:
            trainer = PlusTrainer(fn=TrainerFn.VALIDATING, current_epoch=9)
            trainer.checkpoint_callback = SimpleNamespace(dirpath=tmp)
            with ExitStack() as stack:
                _forbid_kmeans(stack, [])
                cb.on_validation_start(trainer, pl)
            blob = torch.load(os.path.join(tmp, "final_motifs.pt"), weights_only=False)
        self.assertLessEqual(blob["num_motifs"], K)
        self.assertEqual(blob["motifs"].shape[0], blob["num_motifs"])
        self.assertEqual(int(blob["assignments"].max()), blob["num_motifs"] - 1)
        blocks, _ = _get_model_blocks2D(pl.model, R, C)
        self.assertTrue(torch.equal(blob["motifs"][blob["assignments"]], blocks.cpu()))


class TestFrozenWeightAccess(unittest.TestCase):
    """The frozen-phase weight-access rule (see the contract): while parametrized,
    ``iter_trainable_params`` yields ``Z``; consumers of the weights go through
    ``materialized_weights`` (``compute_metrics`` does it itself)."""

    METRICS = ["sparsity", "l2"]

    def setUp(self):
        self.s = _setup(seed=1, freeze_fraction=0.2)
        _run_until_frozen(self.s)
        self.cb, self.pl = self.s.cb, self.s.pl
        # Move Z so that the stale values left in the originals differ from Z[M].
        _train_steps(self.pl, self.s.opt, 3)
        self.assertTrue(_is_parametrized(self.pl.model))
        self.ref = _materialized(self.pl.model)
        self.plain = _plain_copy(self.pl.model)

    def _plain_metrics(self):
        return compute_metrics(self.plain, self.METRICS, rows=R, cols=C)

    def test_without_the_context_the_trainable_tensor_is_Z_not_the_weights(self):
        self.assertEqual(list(iter_trainable_params(self.pl.model)), [self.cb.Z])

    def test_context_exposes_materialized_weights_and_restores_everything(self):
        model, Z = self.pl.model, self.cb.Z
        Z_before = Z.detach().clone()
        with materialized_weights(model):
            seen = list(iter_trainable_params(model))
            self.assertEqual(len(seen), len(self.ref))
            self.assertFalse(any(p is Z for p in seen))
            for p, want in zip(seen, self.ref):
                self.assertTrue(torch.equal(p.detach(), want))
            blocks, _ = _get_model_blocks2D(model, R, C)  # what every block consumer reads
            want_blocks, _ = _get_model_blocks2D(self.plain, R, C)
            self.assertTrue(torch.equal(blocks, want_blocks))
        self.assertEqual(list(iter_trainable_params(model)), [Z])
        self.assertTrue(Z.requires_grad)
        self.assertTrue(torch.equal(Z.detach(), Z_before))
        for got, want in zip(_materialized(model), self.ref):
            self.assertTrue(torch.equal(got, want))
        self.assertTrue(_is_parametrized(model))

    def test_context_restores_on_exception(self):
        model, Z = self.pl.model, self.cb.Z
        with self.assertRaises(RuntimeError):
            with materialized_weights(model):
                raise RuntimeError("boom")
        self.assertEqual(list(iter_trainable_params(model)), [Z])
        self.assertTrue(Z.requires_grad)

    def test_context_is_a_noop_on_a_plain_model(self):
        model = _tiny_backbone()
        before = list(iter_trainable_params(model))
        with materialized_weights(model):
            self.assertEqual(list(iter_trainable_params(model)), before)
        self.assertEqual(list(iter_trainable_params(model)), before)
        self.assertTrue(all(p.requires_grad for p in before))

    def test_compute_metrics_on_the_frozen_model_equals_a_plain_copy(self):
        got = compute_metrics(self.pl.model, self.METRICS, rows=R, cols=C)
        want = self._plain_metrics()
        self.assertEqual(set(got), set(want))
        for key in want:
            self.assertAlmostEqual(float(got[key]), float(want[key]), places=6, msg=key)
        self.assertTrue(_is_parametrized(self.pl.model))

    def test_quantization_callback_logs_metrics_of_the_materialized_weights(self):
        self.cb.metric_names = self.METRICS
        logged = {}
        pl = self.pl
        with patch.object(pl, "log", MagicMock(side_effect=lambda k, v, **kw: logged.update({k: v}))):
            tr = PlusTrainer(current_epoch=8)
            self.cb.on_validation_epoch_start(tr, pl)
            self.cb.on_validation_epoch_end(tr, pl)
        want = self._plain_metrics()
        for key, value in want.items():
            self.assertIn("metrics/" + key, logged)
            self.assertAlmostEqual(float(logged["metrics/" + key]), float(value), places=6, msg=key)

    def test_metrics_history_callback_after_the_quant_callback_sees_the_same_weights(self):
        self.cb.metric_names = self.METRICS
        hist = MetricsHistoryCallback(self.METRICS, quant_callback=self.cb)
        tr = PlusTrainer(current_epoch=8)
        with patch.object(self.pl, "log", MagicMock()):
            self.cb.on_validation_epoch_start(tr, self.pl)
            self.cb.on_validation_epoch_end(tr, self.pl)
            hist.on_validation_epoch_end(tr, self.pl)
        self.assertEqual(len(hist.history), 1)
        want = self._plain_metrics()
        for key, value in want.items():
            self.assertAlmostEqual(float(hist.history[0][key]), float(value), places=6, msg=key)

    def test_finalize_on_a_still_parametrized_model_unfreezes_first(self):
        seen = []
        real = callbacks_mod.finalize_from_codebook

        def spy(model, *a, **kw):
            seen.append(_is_parametrized(model))
            return real(model, *a, **kw)

        with patch.object(callbacks_mod, "finalize_from_codebook", spy), patch.object(
            callbacks_mod, "quantize", side_effect=AssertionError("no k-means")
        ):
            self.cb.on_validation_start(
                PlusTrainer(fn=TrainerFn.VALIDATING, current_epoch=9), self.pl
            )
        self.assertEqual(seen, [False], "finalize must run on a plain model")
        self.assertTrue(self.cb.finalized)
        self.assertTrue(_no_parametrizations(self.pl.model))
        self.assertLessEqual(_unique_blocks(self.pl.model, R, C).shape[0], K)
        for p, w in zip(self.pl.model.parameters(), self.ref):
            self.assertTrue(torch.allclose(p.detach(), w, atol=1e-6))


class TestFreezeEdgeCases(unittest.TestCase):
    def test_freeze_works_without_any_optimizer_on_the_trainer(self):
        cb, pl = _build_plus(freeze_fraction=1.0), _lit(seed=2)
        tr = PlusTrainer(current_epoch=0)
        cb.on_fit_start(tr, pl)
        cb.on_train_epoch_start(tr, pl)  # freeze_epoch == 0: first draw + freeze
        self.assertTrue(cb.frozen and _is_parametrized(pl.model))
        self.assertEqual(float(pl.aux_losses[0](pl.model)), 0.0)

    def test_freeze_with_warm_start_off_still_projects_and_freezes(self):
        s = _setup(warm_start=False, freeze_fraction=0.2)
        plain = _run_until_frozen(s)
        self.assertTrue(s.cb.frozen)
        for got, want in zip(_materialized(s.pl.model), _expected_from(plain, s.cb.Z, s.cb.M)):
            self.assertTrue(torch.equal(got, want))

    @unittest.skipUnless(
        os.environ.get("MOMOS_TEST_COMPILE") == "1" and hasattr(torch, "compile"),
        "torch.compile test is slow; set MOMOS_TEST_COMPILE=1 to run it",
    )
    def test_compiled_backbone_runs_a_frozen_step(self):
        # Pinned behaviour: parametrize + in-place ``model.compile()`` works.
        cb = _build_plus(freeze_fraction=1.0)
        torch.manual_seed(0)
        from model.lit_module import LitClassifier

        pl = LitClassifier(backbone=_tiny_backbone(), num_classes=4, epochs=2, compile_model=True)
        out = pl.configure_optimizers()
        tr = PlusTrainer([out["optimizer"]], [out["lr_scheduler"]["scheduler"]], max_epochs=2)
        cb.on_fit_start(tr, pl)
        cb.on_train_epoch_start(tr, pl)
        _train_steps(pl, out["optimizer"], 1)
        self.assertLessEqual(_unique_blocks(_plain_copy(pl.model), R, C).shape[0], K)


# -----------------------------------------------------------------------------
# End-to-end with a real lightning.Trainer
# -----------------------------------------------------------------------------


class _Recorder(L.Callback):
    """Placed AFTER the quantization and metrics-history callbacks."""

    METRICS = ["sparsity", "l2"]

    def __init__(self, quant_cb):
        super().__init__()
        self.q = quant_cb
        self.mosaics, self.Z_end, self.frozen_unique = [], {}, {}
        self.lrs, self.expected_metrics = {}, {}

    def on_train_epoch_start(self, trainer, pl_module):
        self.mosaics.append(self.q.M.clone())

    def on_train_epoch_end(self, trainer, pl_module):
        if _is_parametrized(pl_module.model):
            e = trainer.current_epoch
            self.Z_end[e] = self.q.Z.detach().clone()
            self.frozen_unique[e] = _unique_blocks(_plain_copy(pl_module.model), R, C).shape[0]
            groups = trainer.optimizers[0].param_groups
            sched = trainer.lr_scheduler_configs[0].scheduler
            self.lrs[e] = ([g["lr"] for g in groups], len(sched.get_last_lr()))

    def on_validation_epoch_end(self, trainer, pl_module):
        if _is_parametrized(pl_module.model) and not trainer.sanity_checking:
            self.expected_metrics[trainer.current_epoch] = compute_metrics(
                _plain_copy(pl_module.model), self.METRICS, rows=R, cols=C
            )


def test_end_to_end_warm_start_then_frozen_codebook_with_real_trainer():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=16, shuffle=False)
    val_loader = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16)

    model = _lit(seed=0, epochs=10)
    quant_cb = _build_plus(freeze_fraction=0.2, warm_start=True, validate_projected=True)
    quant_cb.metric_names = _Recorder.METRICS
    history = MetricsHistoryCallback(_Recorder.METRICS, quant_callback=quant_cb)
    rec = _Recorder(quant_cb)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = L.Trainer(
            max_epochs=10,
            accelerator="cpu",
            devices=1,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            callbacks=[quant_cb, history, rec, ModelCheckpoint(dirpath=tmp, save_last=True)],
        )
        trainer.fit(model, train_loader, val_loader)

        # --- fit ran, schedule: freeze start at epoch 8, epochs 8 and 9 frozen
        assert len(rec.mosaics) == 10
        assert quant_cb.frozen and _no_parametrizations(model)
        assert sorted(rec.Z_end) == [8, 9]
        assert torch.equal(rec.mosaics[8], rec.mosaics[9]), "no draw in the last frozen epoch"
        assert not torch.equal(rec.Z_end[8], rec.Z_end[9]), "Z must train while frozen"
        assert all(n <= K for n in rec.frozen_unique.values()), rec.frozen_unique
        assert "quant/mosaic_changed" in trainer.callback_metrics
        assert "quant/dead_motifs" in trainer.callback_metrics

        # --- one extra optimizer group (Z), kept in lockstep by the real scheduler
        for e, (lrs, n_last) in rec.lrs.items():
            assert len(lrs) == 2 and n_last == 2, (e, lrs, n_last)
            assert abs(lrs[0] - lrs[1]) <= 1e-12 * max(1.0, abs(lrs[0])), (e, lrs)

        # --- per-epoch metrics are those of the materialized weights, in both callbacks
        hist = {h["epoch"]: h for h in history.history}
        assert sorted(rec.expected_metrics) == [8, 9]
        for e, want in rec.expected_metrics.items():
            for key, value in want.items():
                assert abs(float(hist[e][key]) - float(value)) < 1e-6, (e, key)
        assert "metrics/sparsity" in trainer.callback_metrics

        # --- warm start: mosaic ids are stable (a fresh random draw changes ~7/8 of
        # them every epoch; one Lloyd step per epoch changes far fewer)
        changed = [
            float((rec.mosaics[t] != rec.mosaics[t - 1]).float().mean()) for t in range(1, 8)
        ]
        assert max(changed) < 0.5, changed

        # --- after fit the in-memory model is plain, quantized, no finalize needed
        assert all(p.requires_grad for p in model.parameters())
        assert _unique_blocks(model.model, R, C).shape[0] <= K
        assert not quant_cb.finalized

        # --- last.ckpt (written while parametrized) is loadable into a plain model
        ckpt_path = os.path.join(tmp, "last.ckpt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        assert not any("parametrizations" in k for k in ckpt["state_dict"])
        fresh = _lit(seed=123, epochs=10)
        fresh.load_state_dict(ckpt["state_dict"], strict=True)
        assert _unique_blocks(fresh.model, R, C).shape[0] <= K
        for a, b in zip(fresh.model.parameters(), model.model.parameters()):
            assert torch.equal(a.detach(), b.detach())
        states = [v for v in ckpt["callbacks"].values() if isinstance(v, dict) and "frozen" in v]
        assert len(states) == 1 and states[0]["frozen"] is True
        assert tuple(torch.as_tensor(states[0]["Z"]).shape) == (K, R * C)

        # --- validate finalizes from the codebook, no k-means (in-memory model) ...
        seen = []
        with ExitStack() as stack:
            _forbid_kmeans(stack, seen)
            results = trainer.validate(model, val_loader)
        assert quant_cb.finalized and len(seen) == 1
        assert seen[0]["distortion"] == 0.0
        assert seen[0]["num_motifs_used"] <= K
        assert results and "val/acc" in results[0]

        # --- ... and so does a fresh trainer restoring model + callback state from
        # the checkpoint written in the frozen phase
        fresh_cb = _build_plus(freeze_fraction=0.2, warm_start=True)
        trainer2 = L.Trainer(
            accelerator="cpu", devices=1, logger=False, enable_progress_bar=False,
            enable_model_summary=False, enable_checkpointing=False, callbacks=[fresh_cb],
        )
        seen.clear()
        with ExitStack() as stack:
            _forbid_kmeans(stack, seen)
            trainer2.validate(_lit(seed=7, epochs=10), val_loader, ckpt_path=ckpt_path)
        assert fresh_cb.frozen and fresh_cb.finalized and _no_parametrizations(trainer2.lightning_module)
        assert torch.equal(fresh_cb.M, quant_cb.M)
        assert len(seen) == 1 and seen[0]["distortion"] == 0.0 and seen[0]["num_motifs_used"] <= K


# -----------------------------------------------------------------------------
# Registry / bits / tuning / configs
# -----------------------------------------------------------------------------


class _Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 8)


class TestRegistryBitsTune(unittest.TestCase):
    def test_methods_registry_aliases_soft_momos(self):
        self.assertIs(quantizers.METHODS["soft_momos_plus"], quantize_soft_momos)
        self.assertIs(quantizers.METHODS["soft_momos"], quantize_soft_momos)

    def test_bits_equal_soft_momos_without_freeze(self):
        base = {"enabled": True, "rows": 2, "cols": 2, "k": 8, "final_k": 16, "q_bits": 32}
        a = compute_quantization_bits(_Net(), {**base, "method": "soft_momos"})
        b = compute_quantization_bits(
            _Net(), {**base, "method": "soft_momos_plus", "freeze_fraction": 0.0}
        )
        self.assertEqual(a, b)
        self.assertEqual(b["motif_bits"], 16 * 4 * 32.0)
        with self.assertRaises(ValueError):
            compute_quantization_bits(
                _Net(), {"enabled": True, "method": "soft_momos_plus", "rows": 2, "cols": 2, "k": 8}
            )

    def test_bits_with_frozen_phase_use_k_and_ignore_final_k(self):
        base = {"enabled": True, "rows": 2, "cols": 2, "k": 8, "q_bits": 32, "freeze_fraction": 0.1}
        want = compute_quantization_bits(_Net(), {**base, "method": "soft_momos_plus"})
        self.assertEqual(want["motif_bits"], 8 * 4 * 32.0)  # no final_k needed
        for method in ("soft_momos_plus", "soft_momos"):  # by flags, not by name
            got = compute_quantization_bits(_Net(), {**base, "method": method, "final_k": 16})
            self.assertEqual(got, want)

    def test_resolve_block_shape(self):
        self.assertEqual(resolve_block_shape("soft_momos_plus", {"rows": 2, "cols": 4}), (2, 4))

    def test_yaml_configs(self):
        qdir = REPO / "src" / "configs" / "quantization"
        plus = OmegaConf.load(qdir / "soft_momos_plus.yaml")
        self.assertEqual(plus.method, "soft_momos_plus")
        self.assertIs(plus.warm_start, True)
        self.assertIs(plus.reseed_dead, True)
        self.assertEqual(plus.lloyd_iters, 1)
        self.assertEqual(plus.freeze_fraction, 0.1)
        for key in ("rows", "cols", "k", "lam"):
            self.assertIn(key, plus)
        for key in ("final_k", "finalize"):
            self.assertNotIn(key, plus)
        base = OmegaConf.load(qdir / "soft_momos.yaml")
        self.assertIs(base.warm_start, False)
        self.assertEqual(base.freeze_fraction, 0.0)


class _Trial:
    """Minimal optuna.Trial stand-in: categorical -> forced value or first choice,
    float -> forced value or ``low``; records requested names."""

    def __init__(self, method, forced=None):
        self.method, self.forced, self.names = method, forced or {}, []

    def suggest_categorical(self, name, choices):
        self.names.append(name)
        if name == "method":
            return self.method
        return self.forced.get(name, list(choices)[0])

    def suggest_float(self, name, low, high, log=False):
        self.names.append(name)
        return self.forced.get(name, low)


def _space(freeze_fraction):
    return OmegaConf.create(
        {
            "soft_momos_plus": {
                "rows": [1, 2],
                "cols": [2, 4],
                "k": 256,
                "lam": {"low": 0.1, "high": 1000.0, "log": True},
                "freeze_fraction": freeze_fraction,
            }
        }
    )


class TestTuneSuggest(unittest.TestCase):
    def setUp(self):
        import tune_quantization

        self.suggest = tune_quantization._suggest_quant_config

    def test_range_freeze_fraction_is_searched_with_smp_prefix(self):
        trial = _Trial(
            "soft_momos_plus",
            {"smp_rows": 2, "smp_cols": 4, "smp_lam": 5.0, "smp_freeze_fraction": 0.15},
        )
        cfg = self.suggest(trial, _space({"low": 0.05, "high": 0.3}), ["soft_momos_plus"])
        self.assertEqual(cfg["method"], "soft_momos_plus")
        self.assertTrue(cfg["enabled"])
        self.assertEqual((cfg["rows"], cfg["cols"]), (2, 4))
        self.assertTrue(cfg["warm_start"])
        self.assertTrue(cfg["reseed_dead"])
        self.assertEqual(cfg["lloyd_iters"], 1)
        self.assertAlmostEqual(cfg["freeze_fraction"], 0.15)
        self.assertEqual(cfg["lam"], 5.0)
        self.assertEqual(cfg["k"], 256)
        self.assertNotIn("final_k", cfg)
        self.assertTrue(cfg["force_zero"])
        for name in ("smp_rows", "smp_cols", "smp_lam", "smp_freeze_fraction"):
            self.assertIn(name, trial.names)
        self.assertFalse([n for n in trial.names if n.startswith("sm_")])

    def test_scalar_freeze_fraction_is_fixed(self):
        trial = _Trial("soft_momos_plus")
        cfg = self.suggest(trial, _space(0.1), ["soft_momos_plus"])
        self.assertAlmostEqual(cfg["freeze_fraction"], 0.1)
        self.assertNotIn("smp_freeze_fraction", trial.names)

    def test_soft_momos_branch_is_unchanged(self):
        space = OmegaConf.create(
            {"soft_momos": {"rows": [1], "cols": [2], "k": 256, "final_k": 256,
                            "lam": {"low": 1.0, "high": 10.0, "log": False}}}
        )
        cfg = self.suggest(_Trial("soft_momos"), space, ["soft_momos"])
        self.assertEqual(cfg["method"], "soft_momos")
        self.assertNotIn("warm_start", cfg)
        self.assertNotIn("freeze_fraction", cfg)


if __name__ == "__main__":
    unittest.main()
