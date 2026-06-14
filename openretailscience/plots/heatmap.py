"""This module provides functionality for creating generic heatmap plots from pandas DataFrames.

This module is designed to create flexible heatmap visualizations suitable for various use cases
including migration matrices, confusion matrices, correlation matrices, and other 2D data
visualizations. It provides a clean, reusable interface without domain-specific assumptions.

### Core Features

- **Generic Design**: No domain-specific assumptions or hardcoded elements
- **Color Mapping**: Uses Tailwind green colormap for consistent visualization
- **Auto-contrast Text**: Text color automatically switches between black and white based on cell intensity
- **Customizable Labels**: Supports custom labels for x-axis, y-axis, title, and colorbar
- **Flexible Data**: Displays values as-is without formatting assumptions

### Use Cases

- **Migration Matrices**: Visualize customer movement between segments
- **Correlation Matrices**: Show relationships between variables
- **Confusion Matrices**: Display classification results
- **Any 2D Data**: Generic support for any tabular data visualization

### Design Principles

- Display values as-is from the DataFrame (no percentage or other formatting assumptions)
- Consistent with existing OpenRetailScience plotting modules (line.py, bar.py)
- Minimal parameters with **kwargs for advanced customization
- Match visual style of existing plots while remaining generic
"""

from typing import TYPE_CHECKING, Literal, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

if TYPE_CHECKING:
    from matplotlib.collections import QuadMesh
    from matplotlib.spines import Spine

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
) -> Axes:
    """Creates a generic heatmap visualization from a pandas DataFrame.

    This function creates a color-coded heatmap with cell values displayed as text. It is suitable
    for visualizing any 2D data structure including migration matrices, confusion matrices,
    correlation matrices, or cohort analysis data.

    Args:
        df (pd.DataFrame): DataFrame to visualize. Index becomes y-axis, columns become x-axis.
        cbar_label (str): Label for the colorbar.
        x_label (str, optional): Label for x-axis.
        y_label (str, optional): Label for y-axis.
        title (str, optional): Title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): Additional source text annotation.
        figsize (tuple[int, int], optional): The size of the plot. Defaults to None.
        cbar_format (str, optional): Format string applied to in-cell text. In
            ``colormap_style="continuous"`` it is also applied to colorbar tick labels; discrete mode
            labels the colorbar with fixed ``"Low"``/``"High"`` anchors and ignores ``cbar_format`` for
            the bar. Defaults to ``"{x:g}"`` which renders whole numbers without trailing zeros (8, not
            8.00) and keeps fractional values readable.
        colormap_style (Literal["discrete", "continuous"], optional): Render the colorbar as a
            stepped 5-bin scale ("discrete", default — matches the design system) or a smooth
            gradient ("continuous"). Discrete bins lose precision but read more cleanly when
            cell values are annotated; continuous gives a finer-grained sense of magnitude.
        x_labels_position (Literal["top", "bottom"], optional): Whether x-axis tick labels render above
            or below the matrix. Cohort charts conventionally use ``"top"`` so the chronology reads
            top-to-bottom alongside the row labels. Defaults to ``"bottom"``.

    Returns:
        Axes: The matplotlib axes object.
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
    # matplotlib stubs type Colorbar.outline as a Spines collection, but at runtime it is a single Spine.
    cast("Spine", cbar.outline).set_visible(False)
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
        # A discrete colorbar always renders a QuadMesh of solids; the stub types it as optional.
        solids = cast("QuadMesh", cbar.solids)
        solids.set_edgecolor(background)
        solids.set_linewidth(3)
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
