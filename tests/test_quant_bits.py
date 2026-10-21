"""Unit tests for theoretical bit calculation and compression rate utilities."""

import pytest
import torch
import torch.nn as nn
from utils.quant_bits import count_trainable_parameters, compute_quantization_bits


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(16, 16, bias=False)  # 256 params
        self.fc2 = nn.Linear(16, 16, bias=False)  # 256 params
        # total params = 512


def test_count_trainable_parameters():
    model = DummyModel()
    assert count_trainable_parameters(model) == 512


def test_compute_quantization_bits_none():
    model = DummyModel()
    res = compute_quantization_bits(model, {"enabled": False})
    assert res["num_parameters"] == 512
    assert res["dense_bits"] == 512 * 32
    assert res["compressed_bits"] == 512 * 32
    assert res["compression_rate"] == 1.0
    assert res["bpp"] == 32.0


def test_compute_quantization_bits_qat():
    model = DummyModel()
    res = compute_quantization_bits(model, {"enabled": True, "method": "qat", "q": 4})
    assert res["compressed_bits"] == 512 * 4
    assert res["compression_rate"] == 8.0
    assert res["bpp"] == 4.0


def test_compute_quantization_bits_momos2d():
    model = DummyModel()
    # 512 params. block_size = 4x4 = 16. Total blocks = 512 / 16 = 32 blocks.
    # capacity = 0.25 => k = 8 motifs.
    # motif bits = 8 * 16 * 32 = 4096 bits.
    # index bits = 32 blocks * ceil(log2(8)) = 32 * 3 = 96 bits.
    # compressed_bits = 4096 + 96 = 4192 bits.
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "momos2d",
            "rows": 4,
            "cols": 4,
            "capacity": 0.25,
            "q_bits": 32,
        },
    )
    assert res["motif_bits"] == 4096.0
    assert res["index_bits"] == 96.0
    assert res["compressed_bits"] == 4192.0
    assert res["compression_rate"] == (512 * 32) / 4192.0


def test_compute_quantization_bits_hierarchical():
    model = DummyModel()
    # Primary: 4x4 = 16 size, 32 blocks. capacity 0.5 => k1 = 16 motifs.
    # Secondary: 2x2 = 4 primary motifs per big block. n_big = ceil(32 / 4) = 8.
    # Secondary capacity 0.5 => k_per_bucket = round(16 * 0.5) = 8.
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "hierarchical_momos2d",
            "primary": {"rows": 4, "cols": 4, "capacity": 0.5},
            "secondary": {"rows": 2, "cols": 2, "capacity": 0.5},
            "q_bits": 32,
        },
    )
    assert res["compressed_bits"] > 0
    assert res["compression_rate"] > 1.0


def test_compute_quantization_bits_fold_momos_matches_plain_momos2d_shape():
    model = DummyModel()
    # Primary: 4x4 = 16 size, 32 blocks. capacity 0.5 => k1 = 16 motifs.
    # Secondary (fold) grid: 2x2 = 4 folds. n_big = ceil(32 / 4) = 8.
    # Secondary capacity 0.5 => k_per_bucket = round(16 * 0.5) = 8, so the pooled
    # k-means training set has up to 4 * 8 = 32 points -- more than k1=16, so the
    # upper-bound k_eff clamps back down to k1=16 (unlike hierarchical, there is
    # no separate iota term).
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "fold_momos",
            "primary": {"rows": 4, "cols": 4, "capacity": 0.5},
            "secondary": {"rows": 2, "cols": 2, "capacity": 0.5},
            "q_bits": 32,
        },
    )
    # Same formula shape as plain momos2d with k=k_eff=16 (no iota_bits/mosaic-only
    # restriction term): motif_bits = 16*16*32 = 8192, index_bits = 32*ceil(log2(16))
    # = 32*4 = 128.
    assert res["motif_bits"] == 8192.0
    assert res["index_bits"] == 128.0
    assert res["compressed_bits"] == 8320.0
    assert "iota_bits" not in res
    assert res["compression_rate"] == (512 * 32) / 8320.0


def test_compute_quantization_bits_fold_momos_requires_primary_k_or_capacity():
    model = DummyModel()
    with pytest.raises(ValueError):
        compute_quantization_bits(
            model,
            {
                "enabled": True,
                "method": "fold_momos",
                "primary": {"rows": 4, "cols": 4},
                "secondary": {"rows": 2, "cols": 2, "capacity": 0.5},
                "q_bits": 32,
            },
        )


def test_compute_quantization_bits_soft_momos_uses_final_k_codebook():
    model = DummyModel()
    # 512 params, 4x4 = 16-size blocks -> 32 blocks, counted with the same flat
    # sum(ceil(numel / s)) rule as the momos2d branch so the two methods are
    # directly comparable on the Pareto plot. The stored codebook is the
    # post-training k-means++ one with final_k entries -- the training-time k
    # (mosaic draw) is NOT stored.
    # motif bits = 256 * 16 * 32 = 131072; index bits = 32 * ceil(log2 256) = 256.
    res = compute_quantization_bits(
        model,
        {
            "enabled": True,
            "method": "soft_momos",
            "rows": 4,
            "cols": 4,
            "k": 8,
            "final_k": 256,
            "q_bits": 32,
        },
    )
    assert res["num_parameters"] == 512
    assert res["motif_bits"] == 131072.0
    assert res["index_bits"] == 256.0
    assert res["compressed_bits"] == 131328.0
    assert res["compression_rate"] == (512 * 32) / 131328.0
    assert res["bpp"] == 131328.0 / 512
    assert "iota_bits" not in res


def test_compute_quantization_bits_soft_momos_ignores_training_k_and_capacity():
    model = DummyModel()
    # final_k = 16 -> motif bits 16 * 16 * 32 = 8192; index bits 32 * 4 = 128,
    # regardless of the (much larger) training k / capacity.
    for extra in ({"k": 1000}, {"capacity": 1.0}, {}):
        res = compute_quantization_bits(
            model,
            {
                "enabled": True,
                "method": "soft_momos",
                "rows": 4,
                "cols": 4,
                "final_k": 16,
                "q_bits": 32,
                **extra,
            },
        )
        assert res["motif_bits"] == 8192.0
        assert res["index_bits"] == 128.0
        assert res["compressed_bits"] == 8320.0


def test_compute_quantization_bits_soft_momos_q_bits_scales_motif_bits_only():
    model = DummyModel()
    res = compute_quantization_bits(
        model,
        {"enabled": True, "method": "soft_momos", "rows": 4, "cols": 4, "final_k": 16, "q_bits": 8},
    )
    assert res["motif_bits"] == 16 * 16 * 8
    assert res["index_bits"] == 128.0


def test_compute_quantization_bits_soft_momos_requires_final_k():
    model = DummyModel()
    with pytest.raises(ValueError):
        compute_quantization_bits(
            model,
            {"enabled": True, "method": "soft_momos", "rows": 4, "cols": 4, "k": 8, "q_bits": 32},
        )
