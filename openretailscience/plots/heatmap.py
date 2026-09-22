"""Generic 2D heatmap: index → y-axis, columns → x-axis, cell values shown as-is.

cbar_format controls the in-cell text (no percentage formatting is applied); cell text
auto-contrasts (black/white) at the normalized midpoint; the y-axis is inverted so row 0
is on top; x labels rotate 45 past 10 chars; colors come from the option-driven sequential
colormap.
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_sequential_cmap
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

_LABEL_ROTATION_THRESHOLD = 10
_DISCRETE_BIN_COUNT = 5


def _resolve_data_range(df: pd.DataFrame) -> tuple[np.ndarray, float, float, bool]:
    """Return the underlying array, its finite (vmin, vmax), and a uniform-range flag.

    When every finite value is identical there is no magnitude variation to colour-encode;
    the range is widened symmetrically so ``Normalize`` doesn't emit a ``vmin == vmax``
    UserWarning, and the returned flag lets the caller collapse the cmap to a single tone.
    """
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")
    data = df.to_numpy()
    # np.nanmin/nanmax emit RuntimeWarning on all-NaN input; check up-front so the
    # ValueError is the only thing the caller sees.
    if not np.isfinite(data).any():
        raise ValueError("Heatmap data contains no finite values")
    vmin = float(np.nanmin(data))
    vmax = float(np.nanmax(data))
    is_uniform = vmin == vmax
    if is_uniform:
        vmin -= 0.5
        vmax += 0.5
    return data, vmin, vmax, is_uniform


def plot(
    df: pd.DataFrame,
    cbar_label: str,
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    figsize: tuple[int, int] | None = None,
    cbar_format: str = "{x:g}",
    colormap_style: Literal["discrete", "continuous"] = "discrete",
    x_labels_position: Literal["top", "bottom"] = "bottom",
) -> SubplotBase:
    """Create a generic heatmap: index → y-axis, columns → x-axis, cell values as text.

    NaN cells are skipped (no patch or text).

    Args:
        df (pd.DataFrame): Frame to visualize; the index becomes the y-axis and the
            columns the x-axis.
        cbar_label (str): Colorbar label.
        x_label (str, optional): X-axis label.
        y_label (str, optional): Y-axis label.
        title (str, optional): Plot title.
        eyebrow (str, optional): Uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        ax (Axes, optional): Axes to plot on.
        source_text (str, optional): Source attribution rendered at the bottom.
        figsize (tuple[int, int], optional): Figure size, used only when ax is None.
        cbar_format (str, optional): Format for in-cell text; in continuous mode it is also
            applied to the colorbar ticks, but discrete mode ignores it and uses fixed
            "Low"/"High" anchors.
        colormap_style ("discrete", "continuous", optional): A stepped 5-bin colorbar
            (discrete) or a smooth gradient (continuous).
        x_labels_position ("top", "bottom", optional): Draw x tick labels above or below the
            matrix; cohort charts use "top" so the chronology reads top-to-bottom.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If df is empty or contains no finite values.

    """
    data, vmin, vmax, is_uniform = _resolve_data_range(df)

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    cmap = get_sequential_cmap()
    if is_uniform:
        # No magnitude variation to encode — render every cell and every cbar bin
        # at the cmap midpoint so the visual matches the data.
        cmap = ListedColormap([cmap(0.5)] * _DISCRETE_BIN_COUNT)
    elif colormap_style == "discrete":
        cmap = ListedColormap(cmap(np.linspace(0.05, 0.95, _DISCRETE_BIN_COUNT)))

    norm = Normalize(vmin=vmin, vmax=vmax)
    mappable = ScalarMappable(norm=norm, cmap=cmap)

    # Background-colored edge stroke (in points) gives a fixed-pixel inter-cell
    # gap independent of axes aspect; adjacent edges overlap to form a uniform separator.
    style = PlotStyleHelper()
    corner_radius = style.cell_corner_radius
    cell_gap = style.cell_gap
    background = style.background_color
    cell_font = get_font_properties(style.data_label_font)
    # Normalize maps [vmin, vmax] -> [0, 1]; 0.5 is the midpoint that flips text contrast.
    threshold = 0.5
    textcolors = ("black", "white")

    for (i, j), value in np.ndenumerate(data):
        if np.isnan(value):
            continue
        cell = FancyBboxPatch(
            (j - 0.5, i - 0.5),
            width=1.0,
            height=1.0,
            boxstyle=f"round,pad=0,rounding_size={corner_radius}",
            facecolor=cmap(norm(value)),
            edgecolor=background,
            linewidth=cell_gap,
        )
        ax.add_patch(cell)
        ax.text(
            j,
            i,
            cbar_format.format(x=value),
            ha="center",
            va="center_baseline",
            color=textcolors[int(norm(value) > threshold)],
            fontsize=style.data_label_size,
            fontproperties=cell_font,
        )

    ax.set_xlim(-0.5, df.shape[1] - 0.5)
    # Invert the y-axis so row 0 of the DataFrame appears at the top, matching imshow.
    ax.set_ylim(df.shape[0] - 0.5, -0.5)
    ax.set_aspect("auto")

    if colormap_style == "discrete":
        cbar = ax.figure.colorbar(
            mappable,
            ax=ax,
            fraction=0.03,
            pad=0.02,
            shrink=0.4,
            anchor=(0.0, 1.0),
        )
    else:
        cbar = ax.figure.colorbar(mappable, ax=ax, format=cbar_format, fraction=0.03, pad=0.02, shrink=0.85)
    cbar.outline.set_visible(False)
    cbar.ax.set_ylabel(
        cbar_label,
        rotation=-90,
        va="bottom",
        fontsize=style.label_size,
        fontproperties=get_font_properties(style.label_font),
    )
    if colormap_style == "discrete":
        # Discrete bins make precise tick numbers misleading.
        cbar.set_ticks([vmin, vmax])
        cbar.set_ticklabels(["Low", "High"])
        cbar.ax.tick_params(length=0)
        cbar.ax.set_box_aspect(_DISCRETE_BIN_COUNT)
        cbar.solids.set_edgecolor(background)
        cbar.solids.set_linewidth(3)
        for spine in cbar.ax.spines.values():
            spine.set_visible(False)

    # Set up ticks and labels
    ax.set_xticks(np.arange(df.shape[1]))
    ax.set_yticks(np.arange(df.shape[0]))

    # Handle long labels with rotation and proper alignment
    x_labels = df.columns.astype(str).to_list()
    y_labels = df.index.astype(str).to_list()

    # Determine if we need rotation based on label length
    max_x_label_length = max(map(len, x_labels))
    rotation_angle = 45 if max_x_label_length > _LABEL_ROTATION_THRESHOLD else 0

    ax.set_xticklabels(x_labels, rotation=rotation_angle, ha="right" if rotation_angle > 0 else "center")
    ax.set_yticklabels(y_labels)

    label_on_top = x_labels_position == "top"
    ax.tick_params(
        top=label_on_top,
        bottom=not label_on_top,
        labeltop=label_on_top,
        labelbottom=not label_on_top,
    )

    if rotation_angle > 0:
        ax.tick_params(axis="x", which="major", pad=10)

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        source_text=source_text,
        grid_axis="none",
        hide_spines=True,
    )
