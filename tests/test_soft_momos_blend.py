"""Tests for the blend-toward-mosaic extension of ``soft_momos_plus``.

Replaces the dropped growing-L1 term. Instead of a loss penalty, each epoch the
weights are blended toward their mosaic projection,

    w <- (1 - alpha) w + alpha Z[M],

with ``alpha`` ramping 0 -> 1 over the ``blend_fraction`` of epochs ending at
``freeze_epoch``. At ``alpha=1`` the weights are already on the mosaic, so the
hard freeze changes nothing and there is no val-acc bump / train-acc drop.

Module ``src/quantizers/soft_momos.py``
--------------------------------------
``blend_toward_mosaic(model, M, rows, cols, k, alpha, force_zero=True)``
    -> stats dict ``{distortion, num_changed_weights, num_motifs_used}``.
    ``alpha=0`` leaves the weights untouched; ``alpha=1`` is the hard projection
    (identical to ``project_fixed_mosaic``); intermediate ``alpha`` gives the
    element-wise convex combination ``(1-alpha) w + alpha Z[M]``.

Callback config (``src/utils/callbacks.py``)
--------------------------------------------
``blend_fraction`` (0.0 = off, preserving the hard-freeze jump). When > 0 and
``freeze_fraction > 0``, ``on_train_epoch_start`` blends the weights toward the
mosaic over the last ``blend_fraction * max_epochs`` epochs before
``freeze_epoch``; ``cb.blend_alpha`` holds the running alpha (0 -> 1).

Search space (``tune_quantization._suggest_soft_momos``)
--------------------------------------------------------
A ``blend_fraction`` ``{low, high}`` block in the space is sampled as
``{pre}_blend_fraction``; a plain scalar is fixed. Without the key the
suggested config is unchanged.
"""

import unittest

import optuna
import torch
from omegaconf import OmegaConf

from quantizers.soft_momos import blend_toward_mosaic, project_fixed_mosaic
from tests.test_soft_momos import (
    PADDED_M,
    FakeTrainer,
    PaddedLinearModel,
    _build_cb,
    _lit,
)
from tune_quantization import _suggest_soft_momos


class TestBlendTowardMosaic(unittest.TestCase):
    ROWS, COLS, K = 2, 2, 3

    def _weights(self, model):
        return [p.detach().clone() for p in model.parameters()]

    def test_alpha_zero_leaves_weights_unchanged(self):
        model = PaddedLinearModel(seed=1)
        before = self._weights(model)
        blend_toward_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K, 0.0)
        for b, a in zip(before, self._weights(model)):
            self.assertTrue(torch.equal(b, a))

    def test_alpha_one_equals_hard_projection(self):
        model = PaddedLinearModel(seed=2)
        before = self._weights(model)
        blend_toward_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K, 1.0)
        blended = self._weights(model)
        for p, b in zip(model.parameters(), before):
            p.data.copy_(b)
        project_fixed_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K)
        projected = self._weights(model)
        for a, b in zip(blended, projected):
            self.assertTrue(torch.allclose(a, b, atol=1e-6))

    def test_alpha_half_is_the_midpoint(self):
        model = PaddedLinearModel(seed=3)
        before = self._weights(model)
        blend_toward_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K, 1.0)
        proj = self._weights(model)
        for p, b in zip(model.parameters(), before):
            p.data.copy_(b)
        blend_toward_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K, 0.5)
        half = self._weights(model)
        for b, h, p in zip(before, half, proj):
            self.assertTrue(torch.allclose(h, 0.5 * (b + p), atol=1e-6))

    def test_returns_stats_dict(self):
        model = PaddedLinearModel(seed=4)
        stats = blend_toward_mosaic(model, PADDED_M, self.ROWS, self.COLS, self.K, 0.5)
        self.assertIn("distortion", stats)
        self.assertIn("num_changed_weights", stats)
        self.assertIn("num_motifs_used", stats)


class TestCallbackBlend(unittest.TestCase):
    def test_blend_off_by_default(self):
        cb = _build_cb(method="soft_momos_plus", freeze_fraction=0.2, warm_start=True)
        pl = _lit(seed=2)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        before = [p.detach().clone() for p in pl.model.parameters()]
        cb.on_train_epoch_start(FakeTrainer(current_epoch=5, max_epochs=10), pl)
        for b, a in zip(before, [p.detach().clone() for p in pl.model.parameters()]):
            self.assertTrue(torch.equal(b, a), "blend_fraction=0 must not touch weights")
        self.assertEqual(cb.blend_alpha, 0.0)

    def test_alpha_ramps_to_one_by_freeze_epoch(self):
        # freeze_fraction=0.2 of 10 epochs -> freeze at epoch 8; blend_fraction=0.5
        # -> blend over the last 5 epochs (3..8), alpha = (epoch-3)/5.
        cb = _build_cb(method="soft_momos_plus", freeze_fraction=0.2, warm_start=True,
                       blend_fraction=0.5)
        pl = _lit(seed=3)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=3, max_epochs=10), pl)
        self.assertAlmostEqual(cb.blend_alpha, 0.0, places=9)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=5, max_epochs=10), pl)
        self.assertAlmostEqual(cb.blend_alpha, 0.4, places=9)
        cb.on_train_epoch_start(FakeTrainer(current_epoch=8, max_epochs=10), pl)
        self.assertAlmostEqual(cb.blend_alpha, 1.0, places=9)

    def test_blend_actually_moves_weights_toward_mosaic(self):
        cb = _build_cb(method="soft_momos_plus", freeze_fraction=0.2, warm_start=True,
                       blend_fraction=0.5)
        pl = _lit(seed=4)
        cb.on_fit_start(FakeTrainer(max_epochs=10), pl)
        before = [p.detach().clone() for p in pl.model.parameters()]
        cb.on_train_epoch_start(FakeTrainer(current_epoch=8, max_epochs=10), pl)
        after = [p.detach().clone() for p in pl.model.parameters()]
        # alpha=1 at freeze -> weights are now the projection; at least one changed.
        changed = any(not torch.equal(b, a) for b, a in zip(before, after))
        self.assertTrue(changed, "blend at alpha=1 must project the weights")


class TestSearchSpace(unittest.TestCase):
    SPACE = {
        "rows": [1], "cols": [2], "k": 8, "final_k": 4,
        "lam": {"low": 1.0, "high": 10.0, "log": True},
    }

    def _suggest(self, space, method="soft_momos"):
        study = optuna.create_study()
        return _suggest_soft_momos(study.ask(), OmegaConf.create(space), method)

    def test_unchanged_without_blend_fraction(self):
        out = self._suggest(self.SPACE)
        self.assertNotIn("blend_fraction", out)

    def test_blend_fraction_is_sampled(self):
        out = self._suggest(dict(self.SPACE, blend_fraction={"low": 0.05, "high": 0.3}))
        self.assertTrue(0.05 <= out["blend_fraction"] <= 0.3)

    def test_blend_fraction_scalar_is_fixed(self):
        out = self._suggest(dict(self.SPACE, blend_fraction=0.1))
        self.assertEqual(out["blend_fraction"], 0.1)


if __name__ == "__main__":
    unittest.main()
