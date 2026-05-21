"""This module provides functionality for creating timeline plots.

Which are essential for visualizing transactional data over time.
By aggregating data by specified periods (e.g., daily, weekly, monthly), timeline plots help to identify
trends, seasonal patterns, and performance variations across different timeframes. These plots are valuable tools for
retail analysis, sales tracking, and customer behavior insights.

### Features

- **Timeline Plot Creation**: Plot a value column (e.g., sales, transactions) over time,
  aggregated by a specific period (e.g., daily, weekly).
- **Customizable Aggregation**: Supports different aggregation functions (e.g., sum, average)
  to compute the value column's metrics.
- **Grouping by Categories**: Optionally group data by a specific category (e.g., product,
  region, store) and compare performance over time.
- **Time Period Handling**: The `period` parameter allows data aggregation by different time
  periods, such as days, weeks, or months.
- **Graph Styling**: Customize the appearance of the plot with options to adjust titles, axis
  labels, legend placement, and more.
- **Color Mapping**: Use linear color gradients for category-based groupings to visually
  differentiate between groups in the timeline.

### Use Cases

- **Sales and Revenue Analysis**: Track sales performance over time, either as a total or by
  group (e.g., product category or store).
- **Seasonal Trend Analysis**: Visualize how sales or transaction values fluctuate across
  different periods, helping to identify seasonal trends or promotional impacts.
- **Customer Behavior Tracking**: Examine changes in customer behavior (e.g., purchase
  frequency, average transaction value) over time.
- **Comparative Performance**: Compare multiple categories (e.g., different products or regions)
  on the same timeline to evaluate relative performance.

### Limitations and Handling of Data

- **Time Period Grouping**: Data is aggregated by a time period defined by the `period`
  argument, which can be adjusted to daily, weekly, monthly, etc.
- **Grouping by Categories**: If `group_col` is specified, the plot will display performance
  across different categories, with color differentiation for each group.
- **Flexible Aggregation**: The aggregation function (e.g., sum, average) can be customized to
  calculate the desired value for each period.

### Functionality Details

- **plot()**: Generates a timeline plot of a specified value column over time, with
  customization options for grouping, aggregation, and styling.
- **Helper functions**: Utilizes utility functions from the `openretailscience` package to handle
  styling, formatting, and other plot adjustments.
"""

from typing import Any, Literal

import pandas as pd
from matplotlib.axes import Axes, SubplotBase
from pandas.tseries.offsets import BaseOffset

from openretailscience.options import get_option
from openretailscience.plots.styles.colors import get_named_color, get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles


def plot(
    df: pd.DataFrame,
    value_col: str,
    period: str | BaseOffset = "D",
    agg_func: str = "sum",
    group_col: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    move_legend_outside: bool = False,
    legend_style: Literal["box", "end_of_line"] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Plots the value_col over time.

    Timeline plots are a fundamental tool for interpreting transactional data within a temporal context. By presenting
    data in a chronological sequence, these visualizations reveal patterns and trends that might otherwise remain hidden
    in raw numbers, making them essential for both historical analysis and forward-looking insights. They are
    particularly useful for:

    - Tracking sales performance across different periods (e.g., daily, weekly, monthly)
    - Identifying seasonal patterns or promotional impacts on sales
    - Comparing the performance of different product categories or store locations over time
    - Visualizing customer behavior trends, such as purchase frequency or average transaction value

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str): The column to plot.
        period (str | BaseOffset): The period to group the data by.
        agg_func (str, optional): The aggregation function to apply to the value_col. Defaults to "sum".
        group_col (str, optional): The column to group the data by. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None (no title rendered).
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None (no x-axis label rendered).
        y_label (str, optional): The y-axis label. Defaults to None (no y-axis label rendered).
        legend_title (str, optional): The title of the legend. Defaults to None. When None the legend title is set to
            the title case of `group_col`
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot. Defaults to True.
        legend_style (Literal["box", "end_of_line"], optional): How series are labelled. ``"box"`` renders the
            standard legend; ``"end_of_line"`` suppresses the legend and places a colored series label at the
            right end of each line. When ``"end_of_line"``, ``move_legend_outside`` and ``legend_title`` are
            ignored and a warning is emitted if either is supplied.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If `legend_style` is not one of ``None``, ``"box"``, or ``"end_of_line"``.
    """
    if legend_style not in (None, "box", "end_of_line"):
        msg = f"legend_style must be one of (None, 'box', 'end_of_line'); got {legend_style!r}"
        raise ValueError(msg)

    df["transaction_period"] = df[get_option("column.transaction_date")].dt.to_period(
        period,
    )

    if group_col is None:
        default_colors = get_named_color("primary")
        df = df.groupby("transaction_period")[value_col].agg(agg_func)
        show_legend = False
    else:
        df = (
            df.groupby([group_col, "transaction_period"])[value_col]
            .agg(agg_func)
            .reset_index()
            .pivot(index="transaction_period", columns=group_col, values=value_col)
        )
        default_colors = get_plot_colors(df.shape[1])
        show_legend = True

    linewidth = kwargs.pop("linewidth", 3)
    color = kwargs.pop("color", default_colors)
    ax = df.plot(
        linewidth=linewidth,
        color=color,
        legend=show_legend,
        ax=ax,
        **kwargs,
    )

    return standard_graph_styles(
        ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=show_legend,
        legend_style=legend_style,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )
