from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from pathlib import Path

from .figure import Figure


def report(filename, experiments, momos_data, group_by, momos_runs, show=True):
    # %% Training overview
    metrics = {
        "Validation Accuracy": "val/acc",
        "Training Accuracy": "train/acc",
        "Validation Loss": "val/loss",
        "Training Loss": "train/loss",
    }

    runs = experiments[:5]

    fig_training = Figure(title="Training Overview", ncols=2, nrows=2)
    for name, metric in metrics.items():
        _ = fig_training.plot(
            data={r["name"]: r["metrics"][["epoch", metric]] for r in runs},
            exp_name=name,
        )

    # %% 3D group by
    fig_3D = Figure()
    fig_3D.plot3D(
        (momos_data["capacity"], momos_data["s"], momos_data["val_acc"]),
        x_label="Capacity",
        y_label="Block size",
        z_label="Validation accuracy",
        exp_name="MoMos: Accuracy vs. block size and capacity",
        cmap="magma",
        logy=2,
    )

    # %% Group by
    fig_group_by = Figure(title="MoMos: Capacity vs. Accuracy grouped by block Size")
    fig_group_by.plot(
        {f"S={s}": (d["capacity"], d["val_acc"]) for s, d in group_by.items()},
        exp_name=None,  # type: ignore
        symbol="o-",
    )
    fig_group_by.plot_index -= 1
    fig_group_by.plot(
        {"baseline": ([0, 0.3], [np.max(experiments[0]["metrics"]["val/acc"])] * 2)},
        x_label="Capacity",
        y_label="Validation Accuracy",
        exp_name="",
        symbol="--",
        colors=["black"],
    )

    figures = [
        fig_training,
        fig_3D,
        fig_group_by,
    ]
    metrics = {
        "Distortion": "quant/distortion",
        "BZ2 Compression Rate": "metrics/bz2_compression_rate",
        "LZMA Compression Rate": "metrics/lzma_compression_rate",
        "BDM Complexity": "metrics/bdm_complexity",
        "GZip Compression Rate": "metrics/gzip_compression_rate",
        "L2": "metrics/weight_l2",
    }
    for name, key in metrics.items():
        fig = Figure(f"{name} vs Validation Loss", ncols=2, nrows=2)
        for i in range(4):
            _ = fig.plot_twinx(
                (
                    momos_runs[i]["metrics"][["epoch", key]],
                    momos_runs[i]["metrics"][["epoch", "val/loss"]],
                ),
                f"S = {momos_runs[i]['s']}, capacity = {momos_runs[i]['capacity']}",
                x_label="Epochs",
                y1_label=name,
                y2_label="Val. Loss",
                # first column (i=0,2): drop the right y-axis label ("Val. Loss")
                # second column (i=1,3): drop the left y-axis label (name)
                show_y1_label=(i % 2 == 0),
                show_y2_label=(i % 2 == 1),
            )
        figures.append(fig)

    if filename is not None:
        with PdfPages(filename + ".pdf") as pdf:
            for fig in figures:
                fig.save(pdf=pdf)

    if show:
        for fig in figures:
            fig.show()


class Report:
    # Metrics that get divided by their own per-run max in metrics_vs_accuracy, so each
    # stream is rescaled to peak at 1.0 - keyed by the underlying wandb metric string, so
    # this applies automatically to any notebook's metrics_overview dict that uses one of
    # these keys, without needing to be threaded through as a parameter.
    _NORMALIZE_BY_MAX = {
        "metrics/bdm_complexity",
        "metrics/qbdm_complexity",
        "metrics/weight_entropy",
    }

    def __init__(self, output_dir="assets"):
        self.figures: list[Figure] = []
        self.output_dir = Path(output_dir)

    def new_figure(
        self, title=None, figSize=(10, 7), ncols=1, nrows=1, fontsize=13
    ) -> Figure:
        # Redundant safety net: close any figures we already hold before
        # creating a new one, so matplotlib's global figure registry doesn't
        # accumulate live figures (each keeps its weight tensors) across a
        # long report build. Harmless if they were already closed by
        # save()/close() - closing an already-closed figure is a no-op. Any
        # not-yet-saved figures are intentionally discarded here (callers
        # that want them must append+save first).
        for fig in self.figures:
            fig.close()
        fig = Figure(title, figSize, ncols, nrows, fontsize)
        self.figures.append(fig)
        return fig

    def save(self, filename):
        with PdfPages(self.output_dir / filename) as pdf:
            for fig in self.figures:
                fig.save(pdf=pdf)

    @staticmethod
    def _normalize_stream(stream):
        """Divide a metric's means (and stds, to keep the variance band consistent under the
        rescaling: std(x/c) = std(x)/c) by the stream's own max, so it peaks at 1.0. `stream`
        is either `(means, stds)` or `(x, means, stds)` - same shape in, same shape out."""
        *rest, means, stds = stream
        means = np.asarray(means, dtype=float)
        stds = np.asarray(stds, dtype=float)
        peak = means.max() if means.size else 0.0
        peak = peak if peak else 1.0
        return (*rest, means / peak, stds / peak)

    def metrics_vs_accuracy(self, run_data, metrics, style="sci", show=False):
        for m, t in metrics.items():
            fig = self.new_figure(t + " vs Validation Loss", nrows=2, ncols=2)
            y1_label = f"{t} (normalized)" if m in self._NORMALIZE_BY_MAX else t

            i = 0  # tracks grid position; only advances on an actual plot (matches _next_plot)
            for sub_t, data in run_data.items():
                if m not in data or "val/loss" not in data:
                    continue
                if len(data[m][0]) == 0 or len(data["val/loss"][0]) == 0:
                    continue
                stream = data[m]
                if m in self._NORMALIZE_BY_MAX:
                    stream = self._normalize_stream(stream)
                if len(stream) == 3:
                    d = (
                        (stream[0], stream[1], stream[2]),
                        (
                            data["val/loss"][0],
                            data["val/loss"][1],
                            data["val/loss"][2],
                        ),
                    )
                else:
                    d = (
                        (range(len(stream[0])), stream[0], stream[1]),
                        (
                            range(len(data["val/loss"][0])),
                            data["val/loss"][0],
                            data["val/loss"][1],
                        ),
                    )
                fig.plot_twinx_with_var(
                    d,
                    sub_t,
                    x_label="Epochs" if i > 1 else "",
                    y1_label=y1_label,
                    style=style,
                    # first column (i=0,2): drop "Val. Loss" from the legend
                    # second column (i=1,3): drop the metric name from the legend
                    show_y1_legend=(i % 2 == 0),
                    show_y2_legend=(i % 2 == 1),
                )
                i += 1
                if show:
                    fig.show()

    def training_overview(self, run_data, metrics, show=False, alpha=0.12):
        fig = self.new_figure("Training Overview", nrows=2, ncols=2)
        tr_overview = {
            "val/acc": "Validation Accuracy",
            "val/loss": "Validation Loss",
            "train/acc": "Training Accuracy",
            "train/loss": "Training Loss",
        }

        for m, t in tr_overview.items():
            data = {}
            for k, v in run_data.items():
                if m not in v or len(v[m][0]) == 0:
                    continue
                if len(v[m]) == 3:
                    data[k] = (v[m][0], v[m][1], v[m][2])
                else:
                    data[k] = (range(len(v[m][0])), v[m][0], v[m][1])
            if data:
                fig.plot_with_var(data, "", y_label=t, alpha=alpha)

        if show:
            fig.show()

    def append_figures(self, figures: list):
        self.figures += figures

    def close(self):
        """Close every figure this report holds and clear the list, freeing
        the underlying matplotlib memory. Call after `save()` (or once a
        config's report is no longer needed) so figures from one weight-
        analysis config don't accumulate in matplotlib's global registry
        while the next config's analysis runs."""
        for fig in self.figures:
            fig.close()
        self.figures.clear()
