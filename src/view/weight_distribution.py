import torch
import numpy as np
from sklearn.decomposition import PCA
import scipy.stats
from typing import Literal
from matplotlib.colors import LinearSegmentedColormap

from src.view.figure import Figure
from src.quantizers import iter_trainable_params, tensor_to_blocks
from src.quantizers.momos2d import tensor2D_to_blocks
from src.model import MLP

# Low frequency -> blue, mid -> grey, high frequency -> red.
_FREQ_CMAP = LinearSegmentedColormap.from_list("blue_grey_red", ["blue", "grey", "red"])


def extract_blocks(model, block_size):
    # differ from src.quantizers.momos._get_model_blocks because doesn't
    # concatenate the blocks
    block_size = int(block_size)
    with torch.no_grad():
        layer_specs = []
        all_blocks = []

        for param in iter_trainable_params(model):
            blocks, n_params, shape = tensor_to_blocks(param.detach(), block_size)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)

    return all_blocks, layer_specs


def load_model(checkpoint_path):
    # Use weights_only=False fallback because PL checkpoints contain non-tensor fields.
    try:
        weights = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    except Exception:
        weights = torch.load(checkpoint_path, weights_only=False, map_location="cpu")

    try:
        state_dict = weights["state_dict"]
    except Exception:
        state_dict = weights
    new_state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model = MLP(3 * 32 * 32, 10)
    try:
        model.load_state_dict(new_state_dict)
    except Exception:
        model.load_state_dict(state_dict["model"])
    return model


def extract_blocks_2d(model, rows, cols):
    with torch.no_grad():
        layer_specs = []
        all_blocks = []
        for param in iter_trainable_params(model):
            blocks, n_params, shape = tensor2D_to_blocks(param.detach(), rows, cols)
            layer_specs.append((param, int(blocks.size(0)), int(n_params), shape))
            all_blocks.append(blocks)
    return all_blocks, layer_specs


def correlation_data(blocks: list[torch.Tensor]) -> dict:
    """Compute per-block Pearson and Spearman correlation between x_1 and x_2 columns."""
    pearson_r, pearson_p, spearman_r, spearman_p = [], [], [], []
    for block in blocks:
        x1 = block[:, 0].numpy()
        x2 = block[:, 1].numpy()
        pr, pp = scipy.stats.pearsonr(x1, x2)
        sr, sp = scipy.stats.spearmanr(x1, x2)
        pearson_r.append(float(pr))
        pearson_p.append(float(pp))
        spearman_r.append(float(sr))
        spearman_p.append(float(sp))
    return {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
    }


def report_weight_distribution(
    run,
    frequencies=None,
    norms=None,
    scatter=None,
    scatter_layer=None,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> list[Figure]:
    res = []
    if scatter:
        fig = Figure(fontsize=13)
        fig.plot(
            scatter,
            f"Motifs per layer - {run[2]}",
            symbol="o",
            axis=None,
            x_label="$X_1$",
            y_label="$X_2$",
            style=style,
        )
        res.append(fig)

    if frequencies:
        fig = Figure(fontsize=13)
        fig.plot(
            frequencies,
            "",  # title removed; should be f"Motifs frequencies - {run[2]}"
            x_label="Motifs",
            y_label="Frequency",
            colors=["red", "green", "pink", "coral", "purple"],
            axis=None,
            style=style,
            linewidth=2.5,
        )
        res.append(fig)

    if norms:
        fig = Figure(fontsize=13)
        fig.plot(
            norms,
            f"Motifs' norms per layer - {run[2]}",
            x_label="Norms",
            y_label="Frequency",
            logy=True,
            axis=None,
            symbol="x",
            style=style,
        )
        res.append(fig)

    if scatter_layer:
        fig = Figure(
            f"Motifs' norms per layer - {run[2]}",
            nrows=2,
            ncols=2,
            fontsize=13,
        )
        for (n, d), c in zip(scatter_layer.items(), ["red", "green", "pink", "coral"]):
            _ = fig.plot(
                d,
                n,
                legend=False,
                symbol="o",
                axis=None,
                x_label="$X_1$",
                y_label="$X_2$",
                colors=[c],
                style=style,
            )
        res.append(fig)

    return res


def _print_correlation(corr: dict, epoch: str) -> None:
    """Print mean Pearson and Spearman r across blocks."""
    mean_pearson = np.mean(corr["pearson_r"])
    mean_spearman = np.mean(corr["spearman_r"])
    print(
        f"[epoch {epoch}] mean Pearson r={mean_pearson:.4f}  "
        f"mean Spearman r={mean_spearman:.4f}"
    )


def _size_scale(
    counts: torch.Tensor, min_size=20.0, max_size=400.0, scale="linear", power=2.0
):
    """Fit a min-max size mapping (area, for matplotlib's `s=`) to `counts`.

    scale="linear": size grows proportionally with count.
    scale="log":    size grows with log(count) - compresses the gap between big counts and
                     spreads out the small ones (more size variation among rare motifs).
    scale="power":  size grows with count**power - the opposite: a handful of frequent
                     motifs blow up disproportionately, while everything else stays near
                     min_size.

    Returns `(to_size, to_norm, legend_counts)`: `to_size(c)` maps counts to marker sizes,
    `to_norm(c)` maps counts to a 0..1 fraction across that same fitted range (handy for
    driving a colormap alongside the size), and `legend_counts(n)` returns `n` representative
    counts - evenly spaced in the same transformed space, so their sizes/colors are evenly
    spaced too - for building a legend.
    """
    counts = counts.float()

    if scale == "log":
        forward, backward = torch.log, torch.exp
    elif scale == "power":
        forward, backward = (lambda c: c**power), (lambda c: c ** (1 / power))
    else:
        forward = backward = lambda c: c

    space = forward(counts)
    lo, hi = space.min(), space.max()

    def to_norm(c):
        c = forward(c.float())
        if hi == lo:
            return torch.full_like(c, 0.5)
        return (c - lo) / (hi - lo)

    def to_size(c):
        return min_size + to_norm(c) * (max_size - min_size)

    def legend_counts(n=4):
        ticks = backward(torch.linspace(lo, hi, n))
        return torch.unique(ticks.round().clamp(min=1))

    return to_size, to_norm, legend_counts


def _size_legend(ax, to_size, to_color, legend_counts, cmap, title, n=4):
    """Add a small legend with a handful of representative dot sizes and colors, labeled by
    the frequency they stand for. `to_size`/`to_color` may come from different scales."""
    handles = [
        ax.scatter(
            [],
            [],
            s=float(to_size(c)),
            color=cmap(float(to_color(c))),
            alpha=0.6,
            label=f"{int(c)}",
        )
        for c in legend_counts(n)
    ]
    ax.legend(
        handles=handles,
        title=title,
        loc="best",
        labelspacing=1.4,
        borderpad=1.1,
        framealpha=0.9,
    )


def _plot_motifs_by_total_frequency(
    all_blocks: torch.Tensor,
    epoch: str,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> Figure:
    """Scatter of unique motifs; dot size = how often each motif occurs across all layers on
    a count**2 scale (a handful of dominant motifs blow up, the rest stay small), while dot
    color follows the actual (linear) count instead (blue=low, grey=mid, red=high)."""
    motifs, counts = all_blocks.unique(dim=0, return_counts=True)
    to_size, _, legend_counts = _size_scale(counts, scale="power", power=2.0)
    _, to_color, _ = _size_scale(counts, scale="linear")
    sizes = to_size(counts)

    fig = Figure(fontsize=13)
    ax = fig._ax()
    ax.scatter(
        motifs[:, 0].numpy(),
        motifs[:, 1].numpy(),
        s=sizes.numpy(),
        c=to_color(counts).numpy(),
        cmap=_FREQ_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.6,
    )
    _size_legend(ax, to_size, to_color, legend_counts, _FREQ_CMAP, "Frequency")
    fig._default_settings(
        "$w_1$",
        "$w_2$",
        "",  # title removed; should be f"Motifs by total frequency - {epoch}"
        style,
        False,
        False,
        axis=None,
    )
    return fig


def _plot_motifs_by_unit_spread(
    blocks_per_unit: list[torch.Tensor],
    epoch: str,
    unit_label: str,
    legend_title: str,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> Figure:
    """Scatter of unique motifs; dot size = number of distinct entries of `blocks_per_unit`
    containing that motif on a count**2 scale (each entry contributing at most 1, regardless
    of how many times it repeats within it), while dot color follows the actual (linear)
    count instead (blue=few, grey=mid, red=many).

    Generic over what a "unit" is - `_plot_motifs_by_layer_spread` calls this once per layer
    (`blocks_per_unit` = one run's per-layer blocks) and `plot_motifs_by_fold_spread` calls it
    once per CV fold (`blocks_per_unit` = one all-layers `all_blocks` tensor per fold)."""
    per_unit_unique = [b.unique(dim=0) for b in blocks_per_unit]
    motifs, unit_counts = torch.cat(per_unit_unique, dim=0).unique(
        dim=0, return_counts=True
    )
    to_size, _, legend_counts = _size_scale(unit_counts, scale="power", power=2.0)
    _, to_color, _ = _size_scale(unit_counts, scale="linear")
    sizes = to_size(unit_counts)

    fig = Figure(fontsize=13)
    ax = fig._ax()
    ax.scatter(
        motifs[:, 0].numpy(),
        motifs[:, 1].numpy(),
        s=sizes.numpy(),
        c=to_color(unit_counts).numpy(),
        cmap=_FREQ_CMAP,
        vmin=0.0,
        vmax=1.0,
        alpha=0.6,
    )
    _size_legend(ax, to_size, to_color, legend_counts, _FREQ_CMAP, legend_title)
    fig._default_settings(
        "$w_1$",
        "$w_2$",
        f"Motifs by number of {unit_label} - {epoch}",
        style,
        False,
        False,
        axis=None,
    )
    return fig


def _plot_motifs_by_layer_spread(
    blocks: list[torch.Tensor],
    epoch: str,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> Figure:
    return _plot_motifs_by_unit_spread(blocks, epoch, "layers", "# Layers", style=style)


def plot_motifs_by_fold_spread(
    blocks_per_fold: list[torch.Tensor],
    epoch: str,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> Figure:
    """Scatter of unique motifs across a config's CV folds; dot size = number of distinct
    folds whose weights (all layers concatenated - see `extract_blocks_2d`) contain that
    motif at least once, on a count**2 scale, dot color follows the linear count (blue=few,
    grey=mid, red=many). `blocks_per_fold` is one `all_blocks` tensor per fold. Mirrors
    `_plot_motifs_by_layer_spread` at fold granularity instead of layer granularity - see
    `notebook/v-fold_wa.py`."""
    return _plot_motifs_by_unit_spread(
        blocks_per_fold, epoch, "folds", "# Folds", style=style
    )


def _plot_all_blocks(
    run,
    all_blocks,
    blocks,
    layers_specs,
    style: Literal["sci", "scientific", "plain"] = "sci",
) -> list[Figure]:
    if all_blocks.shape[1] == 2:
        _, counts = all_blocks.unique(dim=0, return_counts=True)

        sort_idx = torch.argsort(counts, descending=True)
        frequencies = {"All layers": (range(len(sort_idx)), counts[sort_idx])}

        modules = module_data(blocks)

        figures = report_weight_distribution(
            run,
            frequencies=frequencies,
            norms=modules,
            style=style,
        )
        figures.append(_plot_motifs_by_total_frequency(all_blocks, run[2], style=style))
        figures.append(_plot_motifs_by_layer_spread(blocks, run[2], style=style))

        corr = correlation_data(blocks)
        _print_correlation(corr, run[2])

        return figures

    elif all_blocks.shape[1] > 2:
        pca = PCA(n_components=0.9999)
        pca = pca.fit(all_blocks)
        block_transformed = [pca.transform(b) for b in blocks]

        figures = []
        start = 0
        for end in range(2, all_blocks.shape[1] + 1, 2):
            blocks_cut = [b[:, start:end] for b in block_transformed]
            figures += plot_blocks(
                (run[0], run[1], str(run[2]) + f" Dim={end - 1, end}"),
                blocks_cut,
                layers_specs,
                style=style,
            )
            start = end

        frequencies = frequency_data(all_blocks, blocks)
        modules = module_data(blocks)
        figures += report_weight_distribution(
            run, frequencies=frequencies, norms=modules, style=style
        )

        return figures
    else:
        raise ValueError(
            f"What did you mean to do? In PlotWeight, shape is < 2: {all_blocks.shape}"
        )


def plot_weights(
    run, style: Literal["sci", "scientific", "plain"] = "sci"
) -> list[Figure]:
    model = load_model(run[0])
    blocks, layers_specs = extract_blocks(model, run[1])
    all_blocks = torch.cat(blocks, dim=0)
    return _plot_all_blocks(run, all_blocks, blocks, layers_specs, style=style)


def plot_weights_2d(
    ckpt_path, rows, cols, epoch, style: Literal["sci", "scientific", "plain"] = "sci"
) -> list[Figure]:
    # rows/cols must be supplied by the caller (fetched from wandb run config).
    model = load_model(ckpt_path)
    rows, cols = int(rows), int(cols)
    blocks, layers_specs = extract_blocks_2d(model, rows, cols)
    all_blocks = torch.cat(blocks, dim=0)
    run_display = (ckpt_path, f"rows={rows},cols={cols}", f"epoch={epoch}")
    return _plot_all_blocks(run_display, all_blocks, blocks, layers_specs, style=style)


def plot_blocks(
    run, blocks, layers_specs, style: Literal["sci", "scientific", "plain"] = "sci"
):
    scatter, scatter_layer = scatter_data_numpy(blocks, layers_specs, run[2])
    return report_weight_distribution(
        run, scatter=scatter, scatter_layer=scatter_layer, style=style
    )


def scatter_data_numpy(blocks, layers_specs, capacity):
    scatter_per_layer = {}
    scatter_motifs_per_layer = {}

    for i, (b, _) in enumerate(zip(blocks, layers_specs)):
        uniques_motifs_l = np.unique(b, axis=1)
        scatter_per_layer[f"Layer {i + 1}"] = uniques_motifs_l.T
        if i % 2 == 0 and i < 7:
            scatter_motifs_per_layer[f"Layer {i + 1} with n_params={capacity}"] = {
                "scatter": uniques_motifs_l.T
            }

    return (
        scatter_per_layer,
        scatter_motifs_per_layer,
    )


def frequency_data(all_blocks, blocks):
    motifs, inverse_indices, counts = all_blocks.unique(
        dim=0, return_inverse=True, return_counts=True
    )

    sort_idx = torch.argsort(counts, descending=True)

    remapper = torch.zeros(len(motifs), dtype=torch.long)
    remapper[sort_idx] = torch.arange(len(motifs))
    sorted_inverse_indices = remapper[inverse_indices]

    # 4. Split the sorted inverse indices back into layers
    layer_sizes = [b.size(0) for b in blocks]
    per_layer_inverse = torch.split(sorted_inverse_indices, layer_sizes)

    # 5. Compute counts for each layer
    frequencies = {}
    num_motifs = len(motifs)

    n = 0
    for i, l_inv in enumerate(per_layer_inverse[:-3]):
        if i % 2 == 0:
            layer_histogram = torch.bincount(l_inv, minlength=num_motifs)

            n += layer_histogram.sum()

            frequencies[f"Layer {i + 1}"] = (
                range(len(layer_histogram)),
                layer_histogram / layer_histogram.sum() * 100,
            )

    frequencies["All layers"] = (
        range(len(counts)),
        counts[sort_idx] / counts.sum() * 100,
    )

    return frequencies


def module_data(blocks):
    blocks_norm = [torch.linalg.vector_norm(b, ord=2, dim=1) for b in blocks]

    norms = {}
    for i, b in enumerate(blocks_norm):
        modules, counts = b.unique(dim=0, return_counts=True)

        norms[f"Layer {i + 1}"] = (
            modules,
            counts / counts.sum() * 100,
        )

    return norms
