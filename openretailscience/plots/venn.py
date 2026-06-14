"""This module provides functionality for creating Venn and Euler diagrams from pandas DataFrames.

It is designed to visualize relationships between sets, highlighting intersections and differences between them.

### Core Features

- **Supports 2-set and 3-set Diagrams**: Allows visualization of up to three overlapping sets.
- **Venn and Euler Diagrams**: Uses Venn diagrams by default; switches to Euler diagrams when `vary_size=True`.
- **Customizable Colors and Labels**: Automatically assigns colors and labels for subset representation.
- **Dynamic Sizing**: Adjusts circle sizes for Euler diagrams to reflect proportions.
- **Title and Source Attribution**: Optionally adds a title and source text.

### Use Cases

- **Set Comparisons**: Identify shared and unique elements across two or three sets.
- **Proportional Representation**: Euler diagrams ensure area-accurate representation.
- **Data Overlap Visualization**: Helps in understanding relationships within categorical data.

### Limitations and Warnings

- **Only Supports 2 or 3 Sets**: Does not extend to Venn diagrams with more than three sets.
- **Pre-Aggregated Data Required**: The module does not perform data aggregation; input data
should already be structured correctly.

"""

from collections.abc import Callable
from typing import Any

import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib_set_diagrams import EulerDiagram, VennDiagram

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import apply_chart_chrome

MAX_SUPPORTED_SETS = 3
MIN_SUPPORTED_SETS = 2

_VENN_FILL_ALPHA = 0.55


def _tighten_to_artists(ax: Axes, padding: float = 0.02) -> None:
    """Shrink the axes data limits to wrap around the rendered patches and texts.

    The Venn library sets data limits with extra padding around the circles, so
    on a wide figure (with ``aspect='equal'`` preserving circle shape) the diagram
    floats in horizontal whitespace. Recomputing the limits from the actual
    artist bounding boxes — including the set labels outside the circles —
    pulls the data window in tightly so the diagram fills the available area.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    xs: list[float] = []
    ys: list[float] = []
    for artist in (*ax.patches, *ax.texts):
        if artist in ax.texts and not artist.get_text():
            continue
        bb_disp = artist.get_window_extent(renderer=renderer)
        bb_data = bb_disp.transformed(ax.transData.inverted())
        xs.extend([bb_data.x0, bb_data.x1])
        ys.extend([bb_data.y0, bb_data.y1])
    if len(xs) == 0 or len(ys) == 0:
        return
    xspan = max(xs) - min(xs)
    yspan = max(ys) - min(ys)
    ax.set_xlim(min(xs) - padding * xspan, max(xs) + padding * xspan)
    ax.set_ylim(min(ys) - padding * yspan, max(ys) + padding * yspan)
    # adjustable="box" shrinks the axes rectangle to match the data aspect
    # rather than expanding the data limits to fit the rectangle. Without this,
    # equal-aspect would re-pad the data window and undo the tightening above.
    # anchor="W" pins the resulting square to the left of the chart area so the
    # venn aligns with the title/eyebrow rather than floating in the middle of a
    # wider figure (square figures are still recommended for venns).
    ax.set_aspect("equal", adjustable="box", anchor="W")


def plot(
    df: pd.DataFrame,
    labels: list[str],
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    source_text: str | None = None,
    vary_size: bool = False,
    figsize: tuple[int, int] | None = None,
    ax: Axes | None = None,
    subset_label_formatter: Callable | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plots a Venn or Euler diagram using subset sizes extracted from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame with 'groups' and 'percent' columns.
        labels (list[str]): Labels for the sets in the diagram.
        title (str, optional): Title of the plot. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        source_text (str, optional): Source text for attribution. Defaults to None.
        vary_size (bool, optional): Whether to vary circle size based on subset sizes. Defaults to False.
        figsize (tuple[int, int], optional): Size of the plot. Defaults to None.
        ax (Axes, optional): Matplotlib axes object to plot on. Defaults to None.
        subset_label_formatter (callable, optional): Function to format subset labels. Defaults to None.
        **kwargs: Additional keyword arguments.

    Returns:
        Axes: The matplotlib axes object with the plotted diagram.

    Raises:
        ValueError: If the number of sets is not 2 or 3.
    """
    num_sets = len(labels)
    if num_sets not in {MIN_SUPPORTED_SETS, MAX_SUPPORTED_SETS}:
        raise ValueError("Only 2-set or 3-set Venn diagrams are supported")

    default_colors = get_plot_colors(num_sets)
    colors = kwargs.pop("color", default_colors)

    zero_group = (0, 0) if num_sets == MIN_SUPPORTED_SETS else (0, 0, 0)
    percent_s = df.loc[df["groups"] != zero_group, ["groups", "percent"]].set_index("groups")["percent"]
    subset_sizes = percent_s.to_dict()

    subset_labels = {
        k: subset_label_formatter(v) if subset_label_formatter else str(v) for k, v in subset_sizes.items()
    }

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    diagram_class = EulerDiagram if vary_size else VennDiagram
    diagram = diagram_class(
        set_labels=labels,
        subset_sizes=subset_sizes,
        subset_labels=subset_labels,
        set_colors=colors,
        ax=ax,
        **kwargs,
    )

    for patch in ax.patches:
        patch.set_alpha(_VENN_FILL_ALPHA)

    center_x, center_y, displacement = 0.5, 0.5, 0.1
    style = PlotStyleHelper()
    for text in diagram.set_label_artists:
        text.set_fontproperties(get_font_properties(style.legend_font))
        text.set_fontsize(style.legend_size)
        if num_sets == MAX_SUPPORTED_SETS and not vary_size:
            x, y = text.get_position()
            direction_x, direction_y = x - center_x, y - center_y
            scale = displacement / (direction_x**2 + direction_y**2) ** 0.5
            text.set_position((x + scale * direction_x, y + scale * direction_y))

    for subset_id in subset_sizes:
        if subset_id not in diagram.subset_label_artists:
            continue
        text = diagram.subset_label_artists[subset_id]
        text.set_fontproperties(get_font_properties(style.data_label_font))
        text.set_fontsize(style.data_label_size)

    ax.set_xticklabels([], visible=False)
    ax.set_yticklabels([], visible=False)

    # The library leaves generous padding in the data limits; shrink the window
    # to the real artist extents so the diagram fills the chart area.
    _tighten_to_artists(ax)

    apply_chart_chrome(
        ax,
        eyebrow=eyebrow,
        title=title,
        subtitle=subtitle,
        source_text=source_text,
        warn_stacklevel=3,
    )

    return ax
