import matplotlib.pyplot as plt
from typing import Literal, Any
from matplotlib import font_manager
from matplotlib.legend import Legend
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

_FONT = font_manager.FontProperties(fname="note/fonts/HKGrotesk-Regular.ttf")

_COLORS = [
    "red",
    "blue",
    "green",
    "orange",
    "pink",
    "teal",
    "coral",
    "purple",
    "gold",
    "navy",
    "crimson",
    "turquoise",
    "maroon",
    "olive",
    "lavender",
    "indigo",
    "salmon",
    "sienna",
    "orchid",
    "khaki",
    "steelblue",
    "darkseagreen",
    "lime",
]


class Figure:
    def __init__(self, title=None, figSize=(10, 7), ncols=1, nrows=1, fontsize=12):
        self.figSize = figSize
        self.fontsize = fontsize
        self.ncols = ncols
        self.nrows = nrows

        # Adjust subplot parameters to prevent title/label overlap in multi-row figures
        if nrows > 1:
            plt.rcParams["figure.subplot.hspace"] = 0.35
            plt.rcParams["figure.subplot.bottom"] = 0.12

        self.fig, self.ax = plt.subplots(
            nrows=nrows, ncols=ncols, figsize=figSize, squeeze=False
        )
        self.plot_index = -1

        if (self.ncols > 1 or self.nrows > 1) and title is not None:
            assert isinstance(title, str)
            self.fig.suptitle(title, fontproperties=_FONT, fontsize=self.fontsize * 1.5)

    def _ax(self):
        row, col = divmod(self.plot_index, self.ncols)
        return self.ax[row][col]

    def _next_plot(self):
        self.plot_index += 1
        if self.plot_index >= self.ncols * self.nrows:
            raise RuntimeError(
                f"Too many plots on the current figure. Available axes: {self.ncols * self.nrows}, Current plot index: {self.plot_index}"
            )

    def _default_settings(
        self,
        x_label,
        y_label,
        exp_name,
        style,
        logx,
        logy,
        limits=None,
        ax=None,
        axis=None,
        grid=True,
    ):
        ax = ax or self._ax()
        ax.set_xlabel(x_label, fontproperties=_FONT, fontsize=self.fontsize * 4)
        ax.set_ylabel(y_label, fontproperties=_FONT, fontsize=self.fontsize * 4)

        # scilimits defaults to (0, 0) - like plot3D/plot_twinx_with_var below - so
        # style="sci" actually forces scientific notation on every magnitude; matplotlib's
        # own default of (-5, 6) only kicks in for very large/small values and silently
        # leaves normal-range data in plain notation regardless of `style`.
        ax.ticklabel_format(
            axis="both",
            useMathText=True,
            useOffset=True,
            style=style,
            scilimits=limits if limits is not None else (0, 0),
        )

        if self.ncols == 1 and self.nrows == 1:
            self.fig.suptitle(
                exp_name, fontproperties=_FONT, fontsize=self.fontsize * 5
            )
        else:
            ax.set_title(exp_name, fontproperties=_FONT, fontsize=self.fontsize * 5)

        ax.grid(grid)
        if axis:
            ax.set_xlim(left=axis[0])
            ax.set_ylim(bottom=axis[0])

        if logx:
            ax.set_xscale("log")

        if logy:
            ax.set_yscale("log")

    def plot(
        self,
        data: dict[str, tuple | pd.DataFrame],
        exp_name: str | None,
        logx=False,
        logy=False,
        x_label="Epochs",
        y_label="Loss",
        symbol="-",
        style: Literal["sci", "scientific", "plain"] = "plain",
        markersize=5,
        skip_n=0,
        pop_n=0,
        colors=_COLORS,
        legend=True,
        limits=None,
        axis=(0, 0),
        grid=True,
        linewidth: float | None = None,
    ):
        self._next_plot()

        for i, (name, points) in enumerate(data.items()):
            if isinstance(points, pd.DataFrame):
                points = points.dropna().to_numpy().T
            self._ax().plot(
                points[0][skip_n : len(points[0]) - pop_n],
                points[1][skip_n : len(points[0]) - pop_n],
                symbol,
                markersize=markersize,
                label=name,
                color=colors[i % len(colors)],
                linewidth=linewidth,
            )

        self._default_settings(
            x_label,
            y_label,
            exp_name,
            style,
            logx,
            logy,
            limits=limits,
            axis=axis,
            grid=grid,
        )
        if legend:
            self._ax().legend()

    def plot_with_var(
        self,
        data: dict[str, tuple[list[float], Any, Any]],  # Any stand for np.Array
        exp_name: str,
        logx=False,
        logy=False,
        x_label="Epochs",
        y_label="Loss",
        symbol="-",
        alpha=0.12,
        style: Literal["sci", "scientific", "plain"] = "plain",
        markersize=5,
        legend=True,
        grid=True,
    ):
        """
        data is of the kind:
        "run_name": (X, [[means...], [stds...]]), such that X are the epochs for
        example and the second array is a numpy array of shape (2, |X|)

        """
        self._next_plot()
        ax = self._ax()
        for i, (name, points) in enumerate(data.items()):
            ax.plot(
                points[0],
                points[1],
                symbol,
                markersize=markersize,
                label=name,
                color=_COLORS[i % len(_COLORS)],
            )
            ax.fill_between(
                points[0],
                points[1] - points[2],
                points[1] + points[2],
                alpha=alpha,
                color=_COLORS[i % len(_COLORS)],
            )

        self._default_settings(x_label, y_label, exp_name, style, logx, logy, grid=grid)
        if legend:
            ax.legend()

    def plot3D(
        self,
        data: tuple[list[float], list[float], list[float]],
        x_label: str,
        y_label: str,
        z_label: str,
        exp_name: str,
        cmap="viridis",
        levels=20,
        logx=1,
        logy=1,
        style: Literal["sci", "scientific", "plain"] = "plain",
        grid=True,
    ):
        """
        Create a 3D plot from list of tuples (x, y, z), such that the z is
        indicated with a color.

        Args:
            data: List of tuples (x, y, z) where:
                x: x-axis value
                y: y-axis value
                z: z-axis value (height/accuracy)
            exp_name: Title for the plot
            cmap: Colormap for the contour plot (default: "viridis")
            levels: Number of contour levels (default: 20)
            logx: If >0, use log scale for x-axis with logx as base
            logy: If >0, use log scale for y-axis with logy as base
        """
        self._next_plot()
        ax = self._ax()

        assert len(data) == 3

        cntr = ax.tricontourf(data[0], data[1], data[2], cmap=cmap, levels=levels)

        cbar = plt.colorbar(cntr, ax=ax)
        cbar.set_label(z_label, fontproperties=_FONT, size=self.fontsize)

        if self.ncols == 1 and self.nrows == 1:
            self.fig.suptitle(
                exp_name, fontproperties=_FONT, fontsize=self.fontsize * 1.5
            )

        else:
            ax.set_title(exp_name, fontproperties=_FONT, fontsize=self.fontsize)

        ax.set_xlabel(x_label, fontproperties=_FONT, fontsize=self.fontsize)
        ax.set_ylabel(y_label, fontproperties=_FONT, fontsize=self.fontsize)
        ax.ticklabel_format(
            axis="both", useMathText=True, useOffset=True, style=style, scilimits=(0, 0)
        )
        ax.grid(grid)

        if logx > 1:
            ax.set_xscale("log", base=logx)

        if logy > 1:
            ax.set_yscale("log", base=logy)

    def plot_twinx(
        self,
        data: Any,
        exp_name: str,
        y1_label="Loss",
        y2_label="Val. Loss",
        logx=False,
        logy=False,
        x_label="Epochs",
        symbol="-",
        style: Literal["sci", "scientific", "plain"] = "plain",
        markersize=5,
        skip_n=0,
        pop_n=0,
        colors=_COLORS,
        grid=True,
        show_y1_label=True,
        show_y2_label=True,
    ):
        self._next_plot()

        ax1 = self._ax()

        if isinstance(data[0], pd.DataFrame):
            points = data[0].dropna().to_numpy().T
        else:
            points = data[0]

        ax1.plot(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n],
            symbol,
            markersize=markersize,
            label=y1_label,
            color=colors[0],
        )

        ax1.grid(grid)

        if logx:
            ax1.set_xscale("log")

        if logy:
            ax1.set_yscale("log")

        ax2 = ax1.twinx()
        if isinstance(data[1], pd.DataFrame):
            points = data[1].dropna().to_numpy().T
        else:
            points = data[0]

        ax2.plot(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n],
            symbol,
            markersize=markersize,
            label=y2_label,
            color=colors[1],
        )

        ax1.set_xlabel(x_label, fontproperties=_FONT, fontsize=self.fontsize)
        if show_y1_label:
            ax1.set_ylabel(y1_label, fontproperties=_FONT, fontsize=self.fontsize)
        ax1.ticklabel_format(axis="both", useMathText=True, useOffset=True, style=style)
        if show_y2_label:
            ax2.set_ylabel(y2_label, fontproperties=_FONT, fontsize=self.fontsize)
        ax2.ticklabel_format(axis="y", useMathText=True, useOffset=True, style=style)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        ax2.legend([], [], title=exp_name, loc="lower left")

    def plot_twinx_with_var(
        self,
        data: Any,
        exp_name: str,
        y1_label="Loss",
        y2_label="Val. Loss",
        logx=False,
        logy=False,
        x_label="Epochs",
        symbol="-",
        style: Literal["sci", "scientific", "plain"] = "plain",
        markersize=5,
        linewidth=2.5,
        skip_n=0,
        pop_n=0,
        colors=_COLORS,
        alpha=0.12,
        grid=True,
        show_y1_legend=True,
        show_y2_legend=True,
    ):
        self._next_plot()

        ax1 = self._ax()

        if isinstance(data[0], pd.DataFrame):
            points = data[0].dropna().to_numpy().T
        else:
            points = data[0]

        ax1.plot(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n],
            symbol,
            markersize=markersize,
            linewidth=linewidth,
            label=y1_label,
            color=colors[0],
        )

        ax1.fill_between(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n]
            - points[2][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n]
            + points[2][skip_n : len(points[0]) - pop_n],
            alpha=alpha,
            color=colors[0],
        )

        ax1.grid(grid)

        if logx:
            ax1.set_xscale("log")

        if logy:
            ax1.set_yscale("log")

        ax2 = ax1.twinx()
        if isinstance(data[1], pd.DataFrame):
            points = data[1].dropna().to_numpy().T
        else:
            points = data[1]

        ax2.plot(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n],
            symbol,
            markersize=markersize,
            linewidth=linewidth,
            label=y2_label,
            color=colors[1],
        )

        ax2.fill_between(
            points[0][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n]
            - points[2][skip_n : len(points[0]) - pop_n],
            points[1][skip_n : len(points[0]) - pop_n]
            + points[2][skip_n : len(points[0]) - pop_n],
            alpha=alpha,
            color=colors[1],
        )

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()

        ax1.set_xlabel(x_label, fontproperties=_FONT)
        if show_y1_legend:
            ax1.set_ylabel(y1_label, fontproperties=_FONT)
        ax1.ticklabel_format(
            axis="y", useMathText=True, useOffset=True, style=style, scilimits=(0, 0)
        )
        if show_y2_legend:
            ax2.set_ylabel(y2_label, fontproperties=_FONT, fontsize=self.fontsize)
        ax2.ticklabel_format(
            axis="y", useMathText=True, useOffset=True, style=style, scilimits=(0, 0)
        )

        legend_lines = (lines1 if show_y1_legend else []) + (
            lines2 if show_y2_legend else []
        )
        legend_labels = (labels1 if show_y1_legend else []) + (
            labels2 if show_y2_legend else []
        )
        ax1.legend(legend_lines, legend_labels, loc="upper right")
        ax2.legend([], [], title=exp_name, loc="lower left")

    def _style_text(self, text, fontsize=None):
        """Apply `_FONT` and a size to a Text object, in the order that actually sticks:
        `set_fontproperties` resets size to the FontProperties' own default (10pt, since
        `_FONT` doesn't specify one), so it must run *before* `set_fontsize`, not after."""
        text.set_fontproperties(_FONT)
        text.set_fontsize(self.fontsize if fontsize is None else fontsize)

    def _enforce_fontsize(self):
        """Force every piece of text on this figure - titles, axis labels, tick labels,
        sci-notation offset text, and legends (entries + title) - to self.fontsize, so a
        single knob rescales everything consistently. Walks `self.fig.axes` rather than the
        `self.ax` grid so twin axes (from `twinx()`) are covered too."""
        suptitle = getattr(self.fig, "_suptitle", None)
        if suptitle is not None:
            self._style_text(suptitle, fontsize=2 * self.fontsize)

        for ax in self.fig.axes:
            self._style_text(ax.title, fontsize=1.5 * self.fontsize)
            self._style_text(ax.xaxis.label, fontsize=1.5 * self.fontsize)
            self._style_text(ax.yaxis.label, fontsize=1.5 * self.fontsize)
            ax.tick_params(axis="both", which="major", labelsize=self.fontsize)
            ax.xaxis.get_offset_text().set_fontsize(self.fontsize)
            ax.yaxis.get_offset_text().set_fontsize(self.fontsize)

            # `ax.get_legend()` only returns the *most recently created* legend
            # (`ax.legend_`); a second `ax.legend()` call replaces that reference
            # even though the first legend is still drawn (kept alive via
            # `ax.add_artist()` - the pattern used for e.g. a marker-shape legend
            # plus a separate color legend on the same axes). Walk every `Legend`
            # artist actually attached to the axes instead, so all of them end up
            # at the same font size.
            for legend in ax.get_children():
                if not isinstance(legend, Legend):
                    continue
                for text in legend.get_texts():
                    text.set_fontsize(self.fontsize)
                title = legend.get_title()
                if title is not None:
                    title.set_fontsize(self.fontsize)

    def save(self, figure_name: str = "", pdf: PdfPages = None):
        """Save figure to PDF.

        Args:
            figure_name: Filename for standalone save (used if pdf is None)
            pdf: PdfPages object to save to (used if figure_name is empty)
        """
        self._enforce_fontsize()
        for ax in self.fig.axes:
            for line in ax.get_lines():
                line.set_rasterized(True)

        self.fig.tight_layout()

        if pdf is not None:
            pdf.savefig(self.fig, bbox_inches="tight")
        else:
            self.fig.savefig(f"{figure_name}.pdf", bbox_inches="tight")

        # Release the underlying matplotlib figure immediately after saving,
        # so it doesn't accumulate in matplotlib's global figure registry
        # (which keeps every plt.subplots figure alive until explicitly
        # closed) and exhaust RAM across many weight-analysis configs.
        self.close()

    def show(self):
        self._enforce_fontsize()
        self.fig.tight_layout()
        self.fig.show()

    def close(self):
        """Release the underlying matplotlib figure and free its memory.

        Needed because matplotlib keeps every `plt.subplots` figure alive in
        its global figure registry until explicitly closed - dropping the
        Python `Figure` wrapper alone does NOT free it. Call this after the
        figure is saved/rendered to avoid unbounded accumulation across many
        weight-analysis configs (which can otherwise exhaust RAM)."""
        plt.close(self.fig)

    def axhline(self, data, color, linestyle):
        for label, y in data.items():
            self._ax().axhline(
                y=y,
                linestyle=linestyle,
                color=color,
            )
            self._ax().text(
                x=1.01,
                y=y,
                s=label,
                color=color,
                va="center",
                ha="left",
                transform=self._ax().get_yaxis_transform(),
                fontproperties=_FONT,
                fontsize=self.fontsize,
            )

    def axvline(self, data, color, linestyle):
        for label, x in data.items():
            self._ax().axvline(
                x=x,
                linestyle=linestyle,
                color=color,
            )
            self._ax().text(
                x=x,
                y=1.02,
                s=label,
                rotation=270,
                color=color,
                va="bottom",
                ha="center",
                transform=self._ax().get_xaxis_transform(),
                fontproperties=_FONT,
                fontsize=self.fontsize,
            )
