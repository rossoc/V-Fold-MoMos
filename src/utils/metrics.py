import bz2
import gzip
import lzma
import numpy as np
import torch
import warnings
from src.quantizers import iter_trainable_params, tensor2D_to_blocks
from src.quantizers.soft_momos import materialized_weights

with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
        module=r"pybdm\.utils",
    )
    from pybdm import BDM


def flatten_weights(model, rows=1, cols=2):
    """Flatten trainable params into a block matrix via tensor2D_to_blocks.

    Returns a 2D float32 NumPy array [n_blocks, rows*cols].
    Pass rows=r, cols=c for MoMos2D.
    If both are None, falls back to flatten_weights (single-row matrix).
    """
    arrays = [
        tensor2D_to_blocks(param.detach().cpu().float(), rows, cols)[0].numpy()
        for param in iter_trainable_params(model)
    ]
    if not arrays:
        return np.empty((0, rows * cols), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def real_weight_mask(model, rows=1, cols=2):
    """Boolean mask aligned with `flatten_weights(model, rows, cols)`.

    True where a position holds an actual model weight; False where it's
    zero-padding `tensor2D_to_blocks` introduces so every parameter's last
    block is full-sized (`quantizers/momos2d.py::pad_to_multiple`, which pads
    the last two dims independently via `F.pad(..., value=0.0)` -- a real
    2D grid pad, not a flat "round numel up to the next multiple of
    rows*cols" pad, so the padding count isn't derivable from numel alone).

    Computed by running the *same* `tensor2D_to_blocks` pad/reshape/permute
    path on an all-ones tensor of each parameter's shape: padding always
    comes out exactly 0 there (it's literally what gets padded in), real
    positions come out 1 -- exact, because it's the identical
    geometry-producing function `flatten_weights` uses for the real values,
    not a hand-derived reshape that could drift out of sync with it.
    """
    arrays = [
        tensor2D_to_blocks(torch.ones(param.shape, dtype=torch.float32), rows, cols)[0]
        .numpy()
        .astype(bool)
        for param in iter_trainable_params(model)
    ]
    if not arrays:
        return np.empty((0, rows * cols), dtype=bool)
    return np.concatenate(arrays, axis=0)


def _is_bias_name(name: str) -> bool:
    """True if a dotted parameter name is a bias.

    Matches the standard nn.Linear/nn.Conv2d/norm-layer convention (the
    bias parameter's name always ends in "bias") used throughout this
    codebase's models -- not a dimensionality check, since a norm layer's
    1D *scale* parameter (e.g. `nn.LayerNorm.weight`) is not a bias.
    """
    return name.rsplit(".", 1)[-1] == "bias"


def flatten_weights_excluding_bias(model, rows=1, cols=2):
    """Like `flatten_weights`, but skips bias parameters (see `_is_bias_name`).

    Used only by `WeightAnalyzer.bdm_complexity`: a bias is a 1D tensor, so
    `tensor2D_to_blocks` unsqueezes it to a `[1, N]` "matrix" before
    blocking; whenever `rows > 1` that forces padding up to `rows` in
    height on *every* call, and because of how the block reshape
    interleaves rows/cols, that padding lands mixed *within* every block
    the bias touches (measured: a 4-element bias with rows=cols=2 gives 4
    real + 4 padding values, with both resulting blocks half-padding) --
    not as separable padding blocks. Unlike sparsity/entropy (see
    `real_weight_mask`), BDM can't mask that out of its 2D grid without
    leaving a non-rectangular hole, so biases are excluded from BDM's
    weight view at the source instead. Weight tensors still pad and still
    contribute that padding to BDM when their own shape isn't divisible by
    (rows, cols), same as before -- only bias parameters are excluded.
    """
    arrays = [
        tensor2D_to_blocks(param.detach().cpu().float(), rows, cols)[0].numpy()
        for name, param in model.named_parameters()
        if param.requires_grad and param.numel() > 0 and not _is_bias_name(name)
    ]
    if not arrays:
        return np.empty((0, rows * cols), dtype=np.float32)
    return np.concatenate(arrays, axis=0).astype(np.float32)


def get_compression_payload_from_weights(weights, compression_binarized):
    """Serialize an already flattened weight vector to bytes."""
    if weights.size == 0:
        return b""
    if compression_binarized:
        bits = np.ascontiguousarray((weights > 0).astype(np.uint8))
        return bits.tobytes()
    values = np.ascontiguousarray(weights.astype(np.float32))
    return values.tobytes()


def compression_rate(payload, compressed_payload):
    """Compute compression ratio as raw_size/compressed_size.

    Args:
        payload: Uncompressed bytes.
        compressed_payload: Compressed bytes.

    Returns:
        Float compression ratio (0.0 for empty payload).
    """
    raw_size = len(payload)
    if raw_size == 0:
        return 0.0
    compressed_size = max(1, len(compressed_payload))
    return float(raw_size / compressed_size)


def resolve_block_shape(method, quant_cfg):
    """Resolve the (rows, cols) block shape metrics should flatten weights into.

    Mirrors the branch `QuantizationCallback.on_validation_epoch_end` uses
    when logging per-epoch `metrics/*` during training -- factored out here
    so a post-hoc caller (e.g. a standalone eval script computing metrics
    once on a loaded checkpoint) can reproduce the exact same shape without
    duplicating the branch.

    For `hierarchical_momos2d` the active block depends on whether training
    had progressed past `switch_fraction` by the time `quant_cfg` was read
    (`quant_cfg["apply_secondary"]`, mutated in place by
    `QuantizationCallback` during `on_validation_epoch_start`) -- pass that
    same, possibly-mutated `quant_cfg` dict, not a fresh copy, to get the
    shape actually in effect.

    Args:
        method: Quantization method name (e.g. "momos2d", "qat", "none").
        quant_cfg: The method's config dict (plain dict, not DictConfig).

    Returns:
        (rows, cols) tuple of ints; (1, 1) for methods with no block shape.
    """
    if method == "hierarchical_momos2d":
        active = (
            quant_cfg.get("secondary") or {}
            if quant_cfg.get("apply_secondary", False)
            else quant_cfg.get("primary") or {}
        )
        return int(active.get("rows") or 1), int(active.get("cols") or 1)
    if method == "fold_momos":
        # fold_momos never changes block geometry -- only the codebook -- so the
        # primary block shape is always the one in effect.
        primary = quant_cfg.get("primary") or {}
        return int(primary.get("rows") or 1), int(primary.get("cols") or 1)
    return int(quant_cfg.get("rows") or 1), int(quant_cfg.get("cols") or 1)


def compute_metrics(model, names, compression_binarized=False, rows=1, cols=1):
    """Compute selected metrics on the current model weights.

    Uses WeightAnalyzer internally for efficient caching.

    Args:
        model: PyTorch model to inspect.
        names: List of metric names to compute.
        compression_binarized: Global flag for compression metric payload format.

    Returns:
        Dict of merged metric outputs.
    """
    with materialized_weights(model):  # frozen soft_momos_plus: read Z[M], not Z
        return _compute_metrics(model, names, compression_binarized, rows, cols)


def _compute_metrics(model, names, compression_binarized, rows, cols):
    analyzer = WeightAnalyzer(
        model,
        compression_binarized=compression_binarized,
        rows=rows,
        cols=cols,
    )
    registry = {
        "sparsity": ("sparsity", analyzer.sparsity),
        "l2": ("weight_l2", analyzer.l2_norm),
        "entropy": ("weight_entropy", analyzer.entropy),
        "bdm": ("bdm_complexity", analyzer.bdm_complexity),
        "qbdm": ("qbdm_complexity", analyzer.qbdm),
        "gzip": ("gzip_compression_rate", analyzer.gzip_compress),
        "bz2": ("bz2_compression_rate", analyzer.bz2_compress),
        "lzma": ("lzma_compression_rate", analyzer.lzma_compress),
    }

    unknown = [n for n in names if n not in registry]
    if unknown:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown metric(s): {', '.join(sorted(set(str(n) for n in unknown)))}. Available: {available}"
        )

    return {
        output_key: compute_fn()
        for name in names
        for output_key, compute_fn in [registry[name]]
    }


class WeightAnalyzer:
    """Analyzes model weights with efficient caching.

    Flattens weights once and caches payloads for repeated access.

    Args:
        model: PyTorch model to analyze.
        compression_binarized: If True, use binarized payload for compression metrics.
    """

    def __init__(self, model, compression_binarized=False, rows=1, cols=1):
        blocks = flatten_weights(model, rows=rows, cols=cols)
        self.weights = blocks.ravel()
        self.weights2d = blocks
        # Kept for metrics (e.g. qbdm) that need per-layer weight matrices and
        # parametrized (quantized) weights rather than the flattened block view.
        self._model = model

        # True where a position in self.weights/self.weights2d is a real
        # model weight, False where tensor2D_to_blocks padded it with 0.0 --
        # see real_weight_mask. Used to exclude padding from metrics that
        # treat weight *values* as a distribution (sparsity, entropy); a 1x1
        # block shape (rows=cols=1, qat/no-quantization's default) never
        # pads, so this is an all-True no-op there.
        self.real_mask = real_weight_mask(model, rows=rows, cols=cols).ravel()

        self._payload_cache = {}
        self._compression_binarized = compression_binarized

        # A genuine 2D block layout (momos2d / hierarchical_momos2d, i.e.
        # rows>1 or cols>1 -- see resolve_block_shape) has real 2D spatial
        # structure worth preserving for BDM, matching qbdm.py's own
        # BDM(ndim=2, nsymbols=2) precedent for bit-planes; qat/no-quantization
        # (always rows=cols=1) or a degenerate 1x1 "block" don't, so BDM stays
        # 1D over the flat weight vector there, as before.
        self.rows = int(rows)
        self.cols = int(cols)
        self._bdm_ndim = 2 if (self.rows > 1 or self.cols > 1) else 1
        self._bdm_engine = (
            BDM(ndim=self._bdm_ndim, nsymbols=2) if BDM is not None else None
        )

        # BDM's own weight-only view (see flatten_weights_excluding_bias):
        # biases are excluded here specifically -- everything else above
        # (self.weights/self.weights2d/self.real_mask) still includes them.
        bdm_blocks = flatten_weights_excluding_bias(model, rows=rows, cols=cols)
        self._bdm_weights = bdm_blocks.ravel()
        self._bdm_weights2d = bdm_blocks

    def set_compression_binarized(self, value):
        """Set the binarized flag for compression payloads.

        Args:
            value: Boolean flag.
        """
        self._compression_binarized = value

    def get_payload(self, binarized):
        """Get compression payload, computing and caching if needed.

        Args:
            binarized: If True, use binarized payload (sign bits only).

        Returns:
            Byte payload for compression.
        """
        if binarized not in self._payload_cache:
            self._payload_cache[binarized] = get_compression_payload_from_weights(
                self.weights, compression_binarized=binarized
            )
        return self._payload_cache[binarized]

    def sparsity(self):
        """Compute exact-zero fraction over real (non-padding) trainable weights.

        tensor2D_to_blocks zero-pads each parameter's block region so its
        last block is full-sized; those padding zeros aren't real model
        weights and would inflate this fraction, so self.real_mask excludes
        them (a no-op whenever rows=cols=1, since 1x1 blocks never pad).
        """
        real = self.weights[self.real_mask]
        if real.size == 0:
            return 0.0
        return float((real == 0).mean())

    def l2_norm(self):
        """Compute L2 norm over trainable weights."""
        if self.weights.size == 0:
            return 0.0
        return float(np.linalg.norm(self.weights))

    def entropy(self, bins=256):
        """Compute the Shannon entropy (bits) of the weight value distribution.

        Generic, method-agnostic companion to bdm/qbdm/gzip/bz2/lzma: those
        approximate algorithmic complexity or actual codec-achieved
        compression, this instead treats the (continuous) weight values as a
        distribution -- discretized into `bins` equal-width histogram bins
        over their observed range -- and computes the classic
        H = -sum(p_i * log2(p_i)) over bins with nonzero mass. Higher means
        the weights are spread more uniformly across their range (harder to
        compress); lower means mass concentrated in few bins (e.g. sparse or
        heavily quantized weights), consistent with the sparsity/compression
        metrics above.

        Excludes tensor2D_to_blocks's zero-padding (see self.real_mask,
        set alongside sparsity's identical need to exclude it) -- the same
        padding zeros that would inflate sparsity would also distort this
        distribution's zero-bin mass.

        Returns:
            Float entropy in bits (0.0 for an empty or single-valued model).
        """
        real = self.weights[self.real_mask]
        if real.size == 0:
            return 0.0
        counts, _ = np.histogram(real, bins=bins)
        counts = counts[counts > 0]
        if counts.size == 0:
            return 0.0
        probs = counts / counts.sum()
        return float(-np.sum(probs * np.log2(probs)))

    def bdm_complexity(self):
        """Compute BDM complexity on binarized model *weight* tensors.

        Uses BDM(ndim=2, nsymbols=2) over the block-shaped weight matrix
        (self._bdm_weights2d, shape [n_blocks, rows*cols]) when the model
        uses a genuine 2D block layout (self._bdm_ndim == 2, i.e. rows>1 or
        cols>1 -- momos2d / hierarchical_momos2d), preserving the 2D
        adjacency those methods actually operate on; otherwise BDM(ndim=1)
        over the flat weight vector, as before.

        Excludes bias parameters (see flatten_weights_excluding_bias /
        WeightAnalyzer.__init__'s self._bdm_weights[2d]): forced into a
        [1, N] shape by tensor2D_to_blocks, a bias's padding (whenever
        rows>1) mixes into every block it touches and, unlike
        sparsity/entropy, can't be cleanly masked out of BDM's 2D grid --
        so it's excluded at the source instead, unlike self.weights/
        self.weights2d (which still include biases for every other metric).
        Weight-tensor-only padding (e.g. a conv kernel with a spatial size
        not divisible by rows/cols) is still included here, same as before.

        Returns:
            Float complexity value, or None if BDM unavailable.
        """
        if self._bdm_engine is None:
            return None
        if self._bdm_weights.size == 0:
            return 0.0
        source = self._bdm_weights2d if self._bdm_ndim == 2 else self._bdm_weights
        bits = np.ascontiguousarray((source > 0).astype(np.uint8))
        try:
            return float(self._bdm_engine.bdm(bits))
        except Exception as e:
            print(e)
            return None

    def qbdm(self, bit_depth=8):
        """Compute quantized bit-plane BDM complexity over weight matrices.

        Unlike :meth:`bdm_complexity` (sign bits only), this uniformly quantizes
        each trainable weight matrix to ``bit_depth`` levels and sums BDM over
        all binary bit-planes. Heavier than ``bdm`` (multiprocessing over many
        planes); intended for periodic (e.g. per-epoch) evaluation.

        Returns:
            Float complexity value, or None on failure.
        """
        if self.weights.size == 0:
            return 0.0
        try:
            from src.utils.qbdm import qbdm_complexity

            return qbdm_complexity(self._model, bit_depth=bit_depth)
        except Exception as e:
            print(e)
            return None

    def _compress_payload(self, backend):
        """Compress the configured payload and return the compression rate."""
        payload = self.get_payload(binarized=self._compression_binarized)
        if backend == "gzip":
            compressed = gzip.compress(payload, compresslevel=9, mtime=0)
        elif backend == "bz2":
            compressed = bz2.compress(payload, compresslevel=9)
        elif backend == "lzma":
            compressed = lzma.compress(payload, preset=9)
        else:
            raise ValueError(f"Unknown compression backend: {backend}")
        return compression_rate(payload, compressed)

    def gzip_compress(self):
        """Compute gzip compression rate."""
        return self._compress_payload("gzip")

    def bz2_compress(self):
        """Compute bz2 compression rate."""
        return self._compress_payload("bz2")

    def lzma_compress(self):
        """Compute lzma compression rate."""
        return self._compress_payload("lzma")
