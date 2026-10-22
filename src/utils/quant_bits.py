"""Theoretical bit accounting and compression rate calculation utilities.

Computes the compressed bit count, bits-per-parameter (bpp), and compression
ratio for dense models, QAT, MoMos2D, and Hierarchical (v-fold) MoMos2D.
"""

import math
import torch.nn as nn
from quantizers.block_utils import iter_trainable_params
from quantizers.soft_momos import SOFT_MOMOS_METHODS


def _soft_keeps_trained_codebook(quant_cfg) -> bool:
    """Soft MoMos with a frozen phase: the trained codebook (size k) is the final one
    (no k-means, ``final_k`` unused)."""
    return float(quant_cfg.get("freeze_fraction", 0.0) or 0.0) > 0


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the total number of trainable parameter elements in `model`."""
    return sum(p.numel() for p in iter_trainable_params(model))


def compute_quantization_bits(
    model: nn.Module, quant_cfg: dict | None
) -> dict[str, float]:
    """Compute theoretical storage in bits and compression rate for a model.

    Args:
        model: PyTorch model instance.
        quant_cfg: Quantization configuration dict (e.g. from Hydra).

    Returns:
        dict containing:
            - 'num_parameters': total trainable parameters
            - 'dense_bits': float (params * 32)
            - 'compressed_bits': float total compressed bits
            - 'compression_rate': float (dense_bits / compressed_bits)
            - 'bpp': float bits per parameter
            - 'motif_bits': float (if applicable)
            - 'index_bits': float (if applicable)
    """
    n_params = count_trainable_parameters(model)
    if n_params <= 0:
        return {
            "num_parameters": 0,
            "dense_bits": 0.0,
            "compressed_bits": 0.0,
            "compression_rate": 1.0,
            "bpp": 0.0,
        }

    dense_bits = float(n_params * 32)

    if not quant_cfg or not quant_cfg.get("enabled", False):
        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": dense_bits,
            "compression_rate": 1.0,
            "bpp": 32.0,
        }

    method = str(quant_cfg.get("method", "none")).lower()
    q_motifs = int(quant_cfg.get("q_bits", 32))

    if method == "qat":
        q_bits = int(quant_cfg.get("q", 32))
        compressed_bits = float(n_params * q_bits)
        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": compressed_bits,
            "compression_rate": dense_bits / max(1.0, compressed_bits),
            "bpp": float(q_bits),
        }

    elif method in ("momos2d", "static_momos2d", "momos", *SOFT_MOMOS_METHODS):
        rows = int(quant_cfg.get("rows") or quant_cfg.get("s") or 1)
        cols = int(quant_cfg.get("cols") or 1)
        s = rows * cols

        n_blocks = sum((p.numel() + s - 1) // s for p in iter_trainable_params(model))

        if method in SOFT_MOMOS_METHODS and not _soft_keeps_trained_codebook(quant_cfg):
            # The stored codebook is the post-training k-means++ one (final_k);
            # the training-time k / capacity only shape the penalty mosaic.
            if quant_cfg.get("final_k") is None:
                raise ValueError("soft_momos requires 'final_k' in quant_cfg")
            k = int(quant_cfg["final_k"])
        elif quant_cfg.get("k") is not None:
            k = int(quant_cfg["k"])
        elif quant_cfg.get("capacity") is not None:
            c = float(quant_cfg["capacity"])
            k = max(1, min(int(c * n_blocks), n_blocks))
        else:
            raise ValueError(f"{method} requires either 'k' or 'capacity' in quant_cfg")

        motif_bits = float(k * s * q_motifs)
        idx_bits_per_block = math.ceil(math.log2(max(2, k)))
        index_bits = float(n_blocks * idx_bits_per_block)
        compressed_bits = motif_bits + index_bits

        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": compressed_bits,
            "compression_rate": dense_bits / max(1.0, compressed_bits),
            "bpp": compressed_bits / max(1, n_params),
            "motif_bits": motif_bits,
            "index_bits": index_bits,
        }

    elif method == "hierarchical_momos2d":
        primary = quant_cfg.get("primary", {})
        secondary = quant_cfg.get("secondary", {})

        s1 = int(primary.get("rows", 1)) * int(primary.get("cols", 1))
        n_blocks1 = sum(
            (p.numel() + s1 - 1) // s1 for p in iter_trainable_params(model)
        )

        if primary.get("k") is not None:
            k1 = int(primary["k"])
        elif primary.get("capacity") is not None:
            c1 = float(primary["capacity"])
            k1 = max(1, min(int(c1 * n_blocks1), n_blocks1))
        else:
            raise ValueError("hierarchical_momos2d primary requires 'k' or 'capacity'")

        s2 = int(secondary.get("rows", 1)) * int(secondary.get("cols", 1))
        n_big = math.ceil(n_blocks1 / s2)
        c2 = float(secondary.get("capacity", 1.0))
        k_per_bucket = max(1, min(int(round(k1 * c2)), n_big))

        motif_bits = float(k1 * s1 * q_motifs)
        idx_bits = math.ceil(math.log2(max(2, k1)))
        iota_bits = float(s2 * k_per_bucket * idx_bits)

        big_idx_bits = math.ceil(s2 * math.log2(max(2, k_per_bucket)))
        mosaic_bits = float(n_big * big_idx_bits)

        compressed_bits = motif_bits + iota_bits + mosaic_bits

        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": compressed_bits,
            "compression_rate": dense_bits / max(1.0, compressed_bits),
            "bpp": compressed_bits / max(1, n_params),
            "motif_bits": motif_bits,
            "iota_bits": iota_bits,
            "mosaic_bits": mosaic_bits,
        }

    elif method == "fold_momos":
        primary = quant_cfg.get("primary", {})
        secondary = quant_cfg.get("secondary", {})

        s1 = int(primary.get("rows", 1)) * int(primary.get("cols", 1))
        n_blocks1 = sum(
            (p.numel() + s1 - 1) // s1 for p in iter_trainable_params(model)
        )

        if primary.get("k") is not None:
            k1 = int(primary["k"])
        elif primary.get("capacity") is not None:
            c1 = float(primary["capacity"])
            k1 = max(1, min(int(c1 * n_blocks1), n_blocks1))
        else:
            raise ValueError("fold_momos primary requires 'k' or 'capacity'")

        s2 = int(secondary.get("rows", 1)) * int(secondary.get("cols", 1))
        n_big = math.ceil(n_blocks1 / s2)
        c2 = float(secondary.get("capacity", 1.0))
        k_per_bucket = max(1, min(int(round(k1 * c2)), n_big))
        # fold_momos has no per-fold storage term (unlike hierarchical_momos2d's
        # iota): reconstruction always references one shared, k-means-refined
        # codebook. k_eff is an upper bound on that codebook's size -- the pooled
        # k-means training set has at most s2 * k_per_bucket points (with
        # repeats), and k-means never returns more centroids than k1 or that.
        k_eff = max(1, min(k1, s2 * k_per_bucket))

        motif_bits = float(k_eff * s1 * q_motifs)
        idx_bits_per_block = math.ceil(math.log2(max(2, k_eff)))
        index_bits = float(n_blocks1 * idx_bits_per_block)
        compressed_bits = motif_bits + index_bits

        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": compressed_bits,
            "compression_rate": dense_bits / max(1.0, compressed_bits),
            "bpp": compressed_bits / max(1, n_params),
            "motif_bits": motif_bits,
            "index_bits": index_bits,
        }

    else:
        return {
            "num_parameters": n_params,
            "dense_bits": dense_bits,
            "compressed_bits": dense_bits,
            "compression_rate": 1.0,
            "bpp": 32.0,
        }
