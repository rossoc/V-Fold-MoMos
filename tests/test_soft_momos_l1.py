"""Tests that define the API of the growing-L1 extension of ``soft_momos`` (TDD).

The soft penalty becomes ``lam * L2 + mu(epoch) * L1`` where, with ``Z[M]`` the
detached group means (as in the L2 term) and ``n`` the number of REAL weights,

    L2 = (1/n) * sum_i ||b_i - Z[M_i]||_2^2        (``soft_momos_penalty``)
    L1 = (1/n) * sum_i ||b_i - Z[M_i]||_1          (``soft_momos_l1_penalty``)

L1 has a constant-magnitude gradient ``sign(w - Z[M]) / n``, so unlike L2 it
keeps pulling a weight onto its group mean all the way (no vanishing force near
the target): ramping ``mu`` up through training introduces the frozen mosaic
gradually instead of switching to it in one step.

Module ``src/quantizers/soft_momos.py``
---------------------------------------
``soft_momos_l1_penalty(model, M, rows, cols, k, force_zero=True, counts=None)``
    -> 0-dim tensor, same conventions (padding rule, normaliser, detached Z) as
    ``soft_momos_penalty`` but with ``abs`` instead of ``square``.

``l1_ramp(epoch, ramp_epochs, init, final, shape="cosine", power=2.0)`` -> float
    ``t = clip(epoch / ramp_epochs, 0, 1)`` (``t = 1`` when ``ramp_epochs <= 0``),
    ``mu = init + (final - init) * g(t)`` with ``g(0) = 0`` and ``g(1) = 1``:
      * ``"linear"``: ``t``
      * ``"cosine"``: ``1 - cos(pi * t / 2)``  (zero slope at 0, accelerating)
      * ``"power"``:  ``t ** power``
    Unknown ``shape`` raises ``ValueError``.

Callback config (all optional; defaults keep today's behaviour bit-for-bit)
--------------------------------------------------------------------------
``l1_init`` (0.0), ``l1_final`` (0.0), ``l1_schedule`` ("cosine"), ``l1_power``
(2.0). ``cb.current_l1`` holds ``mu`` for the running epoch, updated in
``on_train_epoch_start``. The ramp spans ``freeze_epoch`` epochs when
``freeze_fraction > 0`` (so ``mu`` reaches ``l1_final`` exactly when the mosaic
is frozen), else ``max_epochs``. The registered aux closure returns
``current_lam * L2 + current_l1 * L1`` (no L1 graph is built when both
``l1_init`` and ``l1_final`` are 0); once frozen it still returns the zero
penalty. Validation stats gain ``"l1"`` (= ``current_l1``).

Search space (``tune_quantization._suggest_soft_momos``)
--------------------------------------------------------
An ``l1_final`` ``{low, high, log}`` block in the space is sampled as
``{pre}_l1_final``; ``l1_init`` / ``l1_schedule`` / ``l1_power`` are copied from
the space when present. Without those keys the suggested config is unchanged.
"""

import math
import unittest

import optuna
import torch
from omegaconf import OmegaConf

from quantizers.momos2d import _get_model_blocks2D
from quantizers.soft_momos import (
    l1_ramp,
    soft_momos_l1_penalty,
    soft_momos_penalty,
)
from tests.test_soft_momos import (
    PADDED_M,
    FakeTrainer,
    FourValueModel,
    PaddedLinearModel,
    TINY_COLS,
    TINY_ROWS,
    _blocks_to_param_tensors,
    _build_cb,
    _loop_group_means,
    _lit,
    _reference_penalty,
)
from tune_quantization import _suggest_soft_momos


class TestL1Ramp(unittest.TestCase):
    def test_endpoints_for_every_shape(self):
        for shape in ("linear", "cosine", "power"):
            self.assertAlmostEqual(l1_ramp(0, 10, 0.5, 4.0, shape), 0.5, places=9)
            self.assertAlmostEqual(l1_ramp(10, 10, 0.5, 4.0, shape), 4.0, places=9)
            self.assertAlmostEqual(l1_ramp(25, 10, 0.5, 4.0, shape), 4.0, places=9, msg="clipped")

    def test_shapes_at_the_midpoint(self):
        self.assertAlmostEqual(l1_ramp(5, 10, 0.0, 1.0, "linear"), 0.5)
        self.assertAlmostEqual(l1_ramp(5, 10, 0.0, 1.0, "cosine"), 1 - math.cos(math.pi / 4))
        self.assertAlmostEqual(l1_ramp(5, 10, 0.0, 1.0, "power", power=3.0), 0.125)
        self.assertAlmostEqual(l1_ramp(5, 10, 2.0, 6.0, "linear"), 4.0)

    def test_cosine_is_monotone_and_accelerating(self):
        mu = [l1_ramp(e, 20, 0.1, 3.0, "cosine") for e in range(21)]
        steps = [b - a for a, b in zip(mu, mu[1:])]
        self.assertTrue(all(s > 0 for s in steps))
        self.assertTrue(all(b > a for a, b in zip(steps, steps[1:])), "increments must grow")

    def test_constant_when_init_equals_final_and_degenerate_ramp_is_final(self):
        for e in range(6):
            self.assertAlmostEqual(l1_ramp(e, 5, 0.3, 0.3, "cosine"), 0.3)
        self.assertAlmostEqual(l1_ramp(0, 0, 0.0, 2.0), 2.0)

    def test_unknown_shape_raises(self):
        with self.assertRaises(ValueError):
            l1_ramp(1, 10, 0.0, 1.0, "exponential")


class TestL1Penalty(unittest.TestCase):
    ROWS, COLS, K = 2, 2, 3

    def test_value_matches_reference_with_padding_rule(self):
        model = PaddedLinearModel(seed=1)
        value = soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K)
        _, Z, blocks, n_real = _reference_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, True
        )
        ref = (blocks - Z[PADDED_M]).abs().sum() / n_real
        self.assertEqual(value.dim(), 0)
        self.assertTrue(torch.allclose(value.detach(), ref, atol=1e-6))

    def test_gradient_is_sign_over_n(self):
        model = PaddedLinearModel(seed=2)
        model.zero_grad()
        value = soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K)
        self.assertTrue(value.requires_grad)
        value.backward()
        _, Z, blocks, n_real = _reference_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, True
        )
        grad_blocks = torch.sign(blocks - Z[PADDED_M]) / n_real
        _b, specs = _get_model_blocks2D(model, self.ROWS, self.COLS)
        expected = _blocks_to_param_tensors(grad_blocks, specs, self.ROWS, self.COLS)
        for (param, *_), exp in zip(specs, expected):
            self.assertTrue(torch.allclose(param.grad, exp, atol=1e-6))

    def test_zero_when_blocks_equal_their_group_means(self):
        model = FourValueModel()
        M = torch.arange(64) % 4
        value = soft_momos_l1_penalty(model, M, 2, 2, 4, force_zero=False)
        self.assertAlmostEqual(float(value), 0.0, places=7)

    def test_never_exceeds_sqrt_of_block_size_times_l2_scale(self):
        # |x|_1 <= sqrt(s) |x|_2 per block; a cheap sanity link between the terms.
        model = PaddedLinearModel(seed=3)
        l1 = soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K)
        l2, _ = soft_momos_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K)
        _, _, blocks, n_real = _reference_penalty(
            model, PADDED_M, self.ROWS, self.COLS, self.K, True
        )
        _, Z, _, _ = _reference_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K, True)
        bound = (blocks - Z[PADDED_M]).norm(dim=1).sum() / n_real * math.sqrt(blocks.size(1))
        self.assertLessEqual(float(l1), float(bound) + 1e-6)
        self.assertGreater(float(l2), 0.0)

    def test_gradient_descent_collapses_blocks_onto_group_means(self):
        # L1 alone, fixed mosaic: weights reach (numerically) their group means.
        model = PaddedLinearModel(seed=4)
        opt = torch.optim.SGD(model.parameters(), lr=0.05 * 18)  # step ~0.05 per weight
        first = float(soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K))
        for _ in range(400):
            opt.zero_grad()
            soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K).backward()
            opt.step()
        last = float(soft_momos_l1_penalty(model, PADDED_M, self.ROWS, self.COLS, self.K))
        self.assertLess(last, 0.05 * first)


def _closure_value(cb, pl):
    return float(pl.aux_losses[0](pl.model))


class TestCallbackL1Schedule(unittest.TestCase):
    def test_defaults_leave_the_penalty_exactly_l2(self):
        cb = _build_cb(lam=0.7)
        pl = _lit(seed=2)
        cb.on_fit_start(FakeTrainer(), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=3), pl)
        self.assertEqual(cb.current_l1, 0.0)
        ref, _ = soft_momos_penalty(pl.model, cb.M, TINY_ROWS, TINY_COLS, cb.quant_cfg["k"])
        self.assertTrue(torch.allclose(pl.aux_losses[0](pl.model).detach(), 0.7 * ref.detach(), atol=1e-7))

    def test_current_l1_follows_the_cosine_ramp_over_max_epochs(self):
        cb = _build_cb(l1_init=0.1, l1_final=2.0, l1_schedule="cosine")
        pl = _lit(seed=3)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        for epoch in (0, 1, 5, 9, 10):
            cb.on_train_epoch_start(FakeTrainer(current_epoch=epoch, max_epochs=10), pl)
            self.assertAlmostEqual(cb.current_l1, l1_ramp(epoch, 10, 0.1, 2.0, "cosine"), places=9)
        self.assertAlmostEqual(cb.current_l1, 2.0)

    def test_ramp_ends_at_the_freeze_epoch_when_freezing(self):
        # freeze_fraction=0.2 of 10 epochs -> freeze at epoch 8 -> mu hits final there.
        cb = _build_cb(method="soft_momos_plus", l1_init=0.0, l1_final=3.0,
                       l1_schedule="linear", freeze_fraction=0.2, warm_start=True)
        pl = _lit(seed=4)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        for epoch, expected in {0: 0.0, 4: 1.5, 7: 3.0 * 7 / 8}.items():
            cb.on_train_epoch_start(FakeTrainer(current_epoch=epoch, max_epochs=10), pl)
            self.assertAlmostEqual(cb.current_l1, expected, places=9, msg=f"epoch {epoch}")

    def test_closure_is_lam_l2_plus_mu_l1(self):
        cb = _build_cb(lam=0.7, l1_init=0.4, l1_final=0.4)
        pl = _lit(seed=5)
        cb.on_fit_start(FakeTrainer(), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=2), pl)
        k = cb.quant_cfg["k"]
        l2, _ = soft_momos_penalty(pl.model, cb.M, TINY_ROWS, TINY_COLS, k)
        l1 = soft_momos_l1_penalty(pl.model, cb.M, TINY_ROWS, TINY_COLS, k)
        value = pl.aux_losses[0](pl.model)
        self.assertTrue(value.requires_grad)
        self.assertTrue(torch.allclose(value.detach(), 0.7 * l2.detach() + 0.4 * l1.detach(), atol=1e-6))

    def test_l1_only_when_lam_is_zero(self):
        cb = _build_cb(lam=0.0, l1_init=1.5, l1_final=1.5)
        pl = _lit(seed=6)
        cb.on_fit_start(FakeTrainer(), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=0), pl)
        l1 = soft_momos_l1_penalty(pl.model, cb.M, TINY_ROWS, TINY_COLS, cb.quant_cfg["k"])
        self.assertAlmostEqual(_closure_value(cb, pl), 1.5 * float(l1), places=6)

    def test_validation_stats_report_l1(self):
        cb = _build_cb(l1_init=0.0, l1_final=1.0, l1_schedule="linear")
        pl = _lit(seed=7)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=5, max_epochs=10), pl)
        cb._soft_validation_start(pl)
        self.assertAlmostEqual(cb._last_quant_stats["l1"], 0.5)


class TestSearchSpace(unittest.TestCase):
    SPACE = {
        "rows": [1], "cols": [2], "k": 8, "final_k": 4,
        "lam": {"low": 1.0, "high": 10.0, "log": True},
    }

    def _suggest(self, space, method="soft_momos"):
        study = optuna.create_study()
        return _suggest_soft_momos(study.ask(), OmegaConf.create(space), method)

    def test_unchanged_without_l1_keys(self):
        out = self._suggest(self.SPACE)
        self.assertFalse(any(key.startswith("l1") for key in out))

    def test_l1_final_is_sampled_and_the_rest_is_copied(self):
        space = dict(self.SPACE, l1_init=0.05, l1_schedule="power", l1_power=3.0,
                     l1_final={"low": 0.1, "high": 100.0, "log": True})
        out = self._suggest(space)
        self.assertTrue(0.1 <= out["l1_final"] <= 100.0)
        self.assertEqual((out["l1_init"], out["l1_schedule"], out["l1_power"]),
                         (0.05, "power", 3.0))

    def test_l1_final_scalar_is_fixed(self):
        out = self._suggest(dict(self.SPACE, l1_final=2.0))
        self.assertEqual(out["l1_final"], 2.0)


if __name__ == "__main__":
    unittest.main()
