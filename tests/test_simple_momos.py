"""Tests for the ``simple_momos`` quantization method.

simple_momos keeps a persistent codebook ``Z``; once per epoch it runs a Lloyd
step (``M <- nearest(w, Z)``, ``Z <- group means under M``) and hard-projects the
weights to ``Z[M]``. No per-batch penalty is registered.
"""

from types import SimpleNamespace

import lightning as L
import optuna
import torch
import torch.nn as nn
from lightning.pytorch.trainer.states import TrainerFn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, TensorDataset

import src.quantizers as quantizers
from src.quantizers.momos2d import _get_model_blocks2D
from src.quantizers.simple_momos import (
    codebook_from_weights,
    init_codebook,
    simple_momos_step,
)
from model.lit_module import LitClassifier
from tune_quantization import _suggest_quant_config
from utils.callbacks import _build_quantization_callback
from utils.quant_bits import compute_quantization_bits

ROWS, COLS, K = 1, 2, 8


class RandomLinearModel(nn.Module):
    def __init__(self, seed=0):
        super().__init__()
        self.fc = nn.Linear(16, 16, bias=False)
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            self.fc.weight.copy_(torch.randn(16, 16, generator=g))


def _tiny_backbone():
    return nn.Sequential(nn.Flatten(), nn.Linear(16, 8), nn.ReLU(), nn.Linear(8, 4))


def _lit(seed=0, epochs=2):
    torch.manual_seed(seed)
    return LitClassifier(backbone=_tiny_backbone(), num_classes=4, epochs=epochs)


def _cfg(**overrides):
    cfg = {
        "enabled": True,
        "method": "simple_momos",
        "rows": ROWS,
        "cols": COLS,
        "k": K,
        "capacity": None,
        "force_zero": True,
        "reseed_dead": True,
        "chunk_size": None,
    }
    cfg.update(overrides)
    return cfg


def _build_cb(**overrides):
    return _build_quantization_callback({}, OmegaConf.create(_cfg(**overrides)))


class FakeTrainer:
    def __init__(self, fn=TrainerFn.FITTING, current_epoch=0, max_epochs=10):
        self.state = SimpleNamespace(fn=fn)
        self.current_epoch = current_epoch
        self.max_epochs = max_epochs
        self.sanity_checking = False
        self.checkpoint_callback = None
        self.callback_metrics = {}
        self.logger = None


def _blocks(model):
    return _get_model_blocks2D(model, ROWS, COLS)[0]


def _distortion(model, Z):
    """Squared distance of every block to its nearest motif in ``Z``."""
    b = _blocks(model)
    return torch.cdist(b, Z).square().min(dim=1).values.sum().item()


# -----------------------------------------------------------------------------
# simple_momos_step
# -----------------------------------------------------------------------------


def test_step_projects_weights_exactly_onto_codebook():
    torch.manual_seed(0)
    model = RandomLinearModel()
    Z0 = init_codebook(model, ROWS, COLS, K, force_zero=True)
    Z, M, stats = simple_momos_step(model, Z0, ROWS, COLS, force_zero=True)

    b = _blocks(model)
    assert torch.equal(b, Z[M])
    assert torch.unique(b, dim=0).size(0) <= K
    assert torch.equal(Z[0], torch.zeros(ROWS * COLS))
    assert stats["num_motifs_used"] <= K
    assert stats["mosaic_changed"] == 1.0  # no previous mosaic


def test_step_runs_a_single_nearest_motif_search():
    import src.quantizers.simple_momos as simple_mod

    torch.manual_seed(5)
    model = RandomLinearModel(seed=5)
    Z0 = init_codebook(model, ROWS, COLS, K, force_zero=True)
    calls = []
    real = simple_mod._nearest_motifs_chunked

    def counting(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)

    simple_mod._nearest_motifs_chunked = counting
    try:
        simple_momos_step(model, Z0, ROWS, COLS, force_zero=True)
    finally:
        simple_mod._nearest_motifs_chunked = real
    assert len(calls) == 1


def test_lloyd_steps_do_not_increase_distortion():
    torch.manual_seed(1)
    model = RandomLinearModel(seed=1)
    raw = model.fc.weight.detach().clone()
    Z = init_codebook(model, ROWS, COLS, K, force_zero=True)
    prev = _distortion(model, Z)
    for _ in range(5):
        with torch.no_grad():
            model.fc.weight.copy_(raw)  # fixed weights: pure k-means
        Z, _, _ = simple_momos_step(model, Z, ROWS, COLS, force_zero=True)
        with torch.no_grad():
            model.fc.weight.copy_(raw)
        cur = _distortion(model, Z)
        assert cur <= prev + 1e-5
        prev = cur


def test_dead_motifs_are_kept_or_reseeded_never_nan():
    torch.manual_seed(2)
    model = RandomLinearModel(seed=2)
    # A motif far from every block: it gets no members.
    Z0 = init_codebook(model, ROWS, COLS, K, force_zero=True)
    Z0[-1] = torch.tensor([1e3, 1e3])

    Z_keep, _, stats = simple_momos_step(
        model, Z0.clone(), ROWS, COLS, force_zero=True, reseed_dead=False
    )
    assert stats["dead_motifs"] >= 1
    assert torch.equal(Z_keep[-1], Z0[-1])
    assert torch.isfinite(Z_keep).all()

    model2 = RandomLinearModel(seed=2)
    Z_re, _, _ = simple_momos_step(
        model2, Z0.clone(), ROWS, COLS, force_zero=True, reseed_dead=True
    )
    assert torch.isfinite(Z_re).all()
    assert not torch.equal(Z_re[-1], Z0[-1])
    assert torch.equal(Z_re[0], torch.zeros(ROWS * COLS))


def test_codebook_from_weights_recovers_mosaic():
    torch.manual_seed(3)
    model = RandomLinearModel(seed=3)
    Z0 = init_codebook(model, ROWS, COLS, K, force_zero=True)
    simple_momos_step(model, Z0, ROWS, COLS, force_zero=True)
    stats = codebook_from_weights(model, ROWS, COLS)
    assert torch.equal(_blocks(model), stats["_motifs"][stats["_nearest"]])
    assert stats["num_motifs_used"] <= K


def test_registry_dispatch():
    torch.manual_seed(4)
    model = RandomLinearModel(seed=4)
    out = quantizers.quantize(model, _cfg())
    assert out["method"] == "simple_momos"
    assert torch.unique(_blocks(model), dim=0).size(0) <= K


# -----------------------------------------------------------------------------
# Callback
# -----------------------------------------------------------------------------


def test_callback_keeps_codebook_across_epochs_and_registers_no_aux_loss():
    cb = _build_cb()
    pl = _lit()
    cb.on_fit_start(FakeTrainer(), pl)
    assert len(pl.aux_losses) == 0
    Z_init = cb.Z.clone()

    cb.on_validation_epoch_start(FakeTrainer(current_epoch=0), pl)
    Z1 = cb.Z.clone()
    assert torch.equal(_get_model_blocks2D(pl.model, ROWS, COLS)[0], Z1[cb.M])

    # Next epoch: the codebook evolves from Z1 (Lloyd), it is not redrawn.
    with torch.no_grad():
        for p in pl.model.parameters():
            p.add_(0.01 * torch.randn_like(p))
    cb.on_validation_epoch_start(FakeTrainer(current_epoch=1), pl)
    assert cb.Z.shape == Z_init.shape
    assert (cb.Z - Z1).abs().max() < 0.5  # small drift, not a random redraw
    assert "mosaic_changed" in cb._last_quant_stats


def test_callback_state_dict_roundtrip():
    cb = _build_cb()
    pl = _lit()
    cb.on_fit_start(FakeTrainer(), pl)
    cb.on_validation_epoch_start(FakeTrainer(), pl)
    state = cb.state_dict()

    cb2 = _build_cb()
    cb2.load_state_dict(state)
    assert torch.equal(cb2.Z, cb.Z)
    assert torch.equal(cb2.M, cb.M)
    cb2.on_fit_start(FakeTrainer(), _lit(seed=1))
    assert torch.equal(cb2.Z, cb.Z)  # resumed codebook is not redrawn


def test_capacity_resolves_k():
    cb = _build_cb(k=None, capacity=0.1)
    pl = _lit()
    cb.on_fit_start(FakeTrainer(), pl)
    assert cb.quant_cfg["k"] >= 1
    assert cb.Z.size(0) == cb.quant_cfg["k"]


def test_end_to_end_with_real_trainer():
    torch.manual_seed(0)
    x = torch.randn(64, 16)
    y = torch.randint(0, 4, (64,))
    train_loader = DataLoader(TensorDataset(x, y), batch_size=16)
    val_loader = DataLoader(TensorDataset(x[:32], y[:32]), batch_size=16)

    model = _lit(epochs=2)
    cb = _build_cb()
    trainer = L.Trainer(
        max_epochs=2,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[cb],
    )
    trainer.fit(model, train_loader, val_loader)
    assert len(model.aux_losses) == 0
    # The last projection leaves the weights on the mosaic.
    assert torch.unique(_get_model_blocks2D(model.model, ROWS, COLS)[0], dim=0).size(0) <= K

    results = trainer.validate(model, val_loader)
    assert cb.finalized
    assert results and "val/acc" in results[0]


# -----------------------------------------------------------------------------
# Bit accounting and tuning
# -----------------------------------------------------------------------------


def test_bits_match_momos2d():
    model = RandomLinearModel()
    simple = compute_quantization_bits(model, _cfg(k=16, rows=2, cols=2))
    momos = compute_quantization_bits(
        model, {"enabled": True, "method": "momos2d", "k": 16, "rows": 2, "cols": 2}
    )
    assert simple["compressed_bits"] == momos["compressed_bits"]


def test_suggest_quant_config_builds_simple_momos():
    space = OmegaConf.create(
        {"simple_momos": {"rows": [1, 2], "cols": [2, 4], "k": 256}}
    )
    trial = optuna.trial.FixedTrial(
        {"method": "simple_momos", "simple_rows": 2, "simple_cols": 4}
    )
    cfg = _suggest_quant_config(trial, space, ["simple_momos"])
    assert cfg["method"] == "simple_momos"
    assert (cfg["rows"], cfg["cols"], cfg["k"]) == (2, 4, 256)
    assert "lloyd_iters" not in cfg
    assert cfg["capacity"] is None
