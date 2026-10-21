"""MoMos quantization package."""

import time

import torch

from .fake_quant import (
    RoundSTE,
    UniformSymmetric,
    FakeQuantParametrization,
    attach_weight_quantizers,
    toggle_quantization,
    quantize_qat,
)

from .block_utils import (
    iter_trainable_params,
    tensor_to_blocks,
    blocks_to_tensor,
    count_total_blocks,
    k_from_capacity,
)

from .momos import (
    momos,
    quantize_momos,
)

from .momos2d import quantize_momos2D, momos2D, tensor2D_to_blocks
from .momos_hierarchy import quantize_hierarchical_momos2D
from .fold_momos import quantize_fold_momos
from .soft_momos import quantize_soft_momos
from .static_momos import (
    quantize_static_momos2D,
    static_momos2D,
    g_exp,
    g_exp_inverse,
    g_sr_rational,
    build_codebook,
)

__all__ = [
    # Fake quantization
    "RoundSTE",
    "UniformSymmetric",
    "FakeQuantParametrization",
    "attach_weight_quantizers",
    "toggle_quantization",
    "quantize_qat",
    # Block utilities
    "iter_trainable_params",
    "tensor_to_blocks",
    "blocks_to_tensor",
    "count_total_blocks",
    "k_from_capacity",
    # MoMos core
    "momos",
    "quantize_momos",
    "METHODS",
    "available_methods",
    "quantize",
    # MoMos2D core
    "momos2D",
    "quantize_momos2D",
    "tensor2D_to_blocks",
    # Hierarchical (doubled) MoMos2D
    "quantize_hierarchical_momos2D",
    # Fold MoMos (k-means++ refined codebook via fold-stratified sampling)
    "quantize_fold_momos",
    # Soft MoMos (penalised training + k-means++ finalisation)
    "quantize_soft_momos",
    # Static MoMos2D (G-function initialised)
    "static_momos2D",
    "quantize_static_momos2D",
    "g_exp",
    "g_exp_inverse",
    "g_sr_rational",
    "build_codebook",
]

METHODS = {
    "qat": quantize_qat,
    "momos": quantize_momos,
    "momos2d": quantize_momos2D,
    "static_momos2d": quantize_static_momos2D,
    "hierarchical_momos2d": quantize_hierarchical_momos2D,
    "fold_momos": quantize_fold_momos,
    "soft_momos": quantize_soft_momos,
}


def available_methods():
    """Return supported quantization method names."""
    return sorted(METHODS.keys())


def quantize(model, quant_cfg):
    """Dispatch quantization by method and attach elapsed time.

    Args:
        model: Model to quantize.
        quant_cfg: Quantization config dict with ``method`` key.

    Returns:
        Stats dict with ``method`` and ``q_time``, or ``None``.
    """
    if not quant_cfg:
        return None

    start = time.perf_counter()
    method = str(quant_cfg.get("method", "qat")).lower()
    fn = METHODS.get(method)
    if fn is None:
        raise ValueError(
            f"Unsupported quantization method: {method}. Available: {', '.join(available_methods())}"
        )
    out = fn(model, quant_cfg) or {}
    out["method"] = method
    out["q_time"] = time.perf_counter() - start
    return out
