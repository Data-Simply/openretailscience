"""This module provides functionality for creating area plots from pandas DataFrames.

It is designed to visualize data distributions over time or across categories using filled area charts. These plots
help highlight trends and comparisons between different groups by stacking or overlaying areas.

While this module supports datetime values on the x-axis, the **plots.time_area** module is better suited for
explicitly time-based visualizations, offering features like resampling and time-based aggregation.

### Core Features

- **Flexible X-Axis Handling**: Uses an index or a specified x-axis column (**`x_col`**) for plotting.
- **Multiple Area Support**: Allows plotting multiple columns (**`value_col`**) or groups (**`group_col`**).
- **Dynamic Color Mapping**: Automatically selects a colormap based on the number of groups.
- **Legend Customization**: Supports custom legend titles and the option to move the legend outside the plot.
- **Source Text**: Provides an option to add source attribution to the plot.

### Use Cases

- **Time Series Visualization**: Show trends in a metric over time (e.g., revenue by month).
- **Stacked Area Charts**: Compare contributions of different groups over time.
- **Category-Based Area Plots**: Visualize distributions of data across categories.

### Limitations and Warnings

- **Handling of Datetime Data**: If a datetime column is passed as **`x_col`**, a warning suggests using
  the **plots.time_area** module for better handling.
- **Pre-Aggregated Data Required**: The module does not perform data aggregation; data should be pre-aggregated
  before being passed to the function.
"""

from typing import Any, Literal, cast

import pandas as pd
from matplotlib.axes import Axes

from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


def plot(
    df: pd.DataFrame,
    value_col: str | list[str],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_col: str | None = None,
    group_col: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    legend_style: Literal["box", "end_of_line"] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plots an area chart for the given `value_col` over `x_col` or index, with optional grouping by `group_col`.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str or list of str): The column(s) to plot.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_col (str, optional): The column to be used as the x-axis. If None, the index is used.
        group_col (str, optional): The column used to define different areas in the plot.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        legend_style (Literal["box", "end_of_line"], optional): How series are labelled. ``"box"`` renders the
            standard legend; ``"end_of_line"`` suppresses the legend and places a colored series label at the
            right end of each line.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        Axes: The matplotlib axes object.

    Raises:
        ValueError: If `value_col` is a list and `group_col` is provided (which causes ambiguity in plotting).
        ValueError: If `legend_style` is not one of ``None``, ``"box"``, or ``"end_of_line"``.
    """
    if legend_style not in (None, "box", "end_of_line"):
        msg = f"legend_style must be one of (None, 'box', 'end_of_line'); got {legend_style!r}"
        raise ValueError(msg)

    if isinstance(df, pd.Series):
        df = df.to_frame()

    if isinstance(value_col, list) and group_col:
        raise ValueError("Cannot use both a list for `value_col` and a `group_col`. Choose one.")

    if group_col is None:
        pivot_df = df.set_index(x_col if x_col is not None else df.index)[
            [value_col] if isinstance(value_col, str) else value_col
        ]
    else:
        pivot_df = df.pivot(index=x_col if x_col is not None else None, columns=group_col, values=value_col)

    is_multi_area = (group_col is not None) or (isinstance(value_col, list) and len(value_col) > 1)

    num_colors = len(pivot_df.columns) if is_multi_area else 1
    default_colors = get_plot_colors(num_colors)
    alpha = kwargs.pop("alpha", 0.7)
    color = kwargs.pop("color", default_colors)
    # pandas-stubs types DataFrame.plot(kind="area") as a wide union; it returns a single Axes here.
    ax = cast(
        "Axes",
        pivot_df.plot(
            ax=ax,
            kind="area",
            alpha=alpha,
            color=color,
            legend=is_multi_area,
            **kwargs,
        ),
    )

    # Drop the stroke so the swatch reads as a single translucent block matching the fill.
    for collection in ax.collections:
        collection.set_linewidth(0)

    # pandas labels each band's top-edge Line2D with "_childN" so it stays out
    # of the auto legend in favour of the PolyCollection. Mirror the collection
    # label onto the line so draw_end_of_line_labels can pick the series up.
    if legend_style == "end_of_line":
        collection_labels = [c.get_label() for c in ax.collections]
        lines = ax.get_lines()
        if len(lines) == len(collection_labels):
            for line, label in zip(lines, collection_labels, strict=True):
                line.set_label(label)

    # pandas stacks bottom-up but lists series in column order; legend_reverse
    # routes the rebuild through apply_legend so chrome's tight_layout sees the
    # final ordering. No-op under end-of-line, which suppresses the box legend.
    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=is_multi_area,
        legend_style=legend_style,
        legend_reverse=is_multi_area,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )
