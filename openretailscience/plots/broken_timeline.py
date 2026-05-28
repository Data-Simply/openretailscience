"""This module provides functionality for creating broken timeline plots from pandas DataFrames.

A broken timeline plot visualizes data availability across categories over time, showing periods where
data is available as horizontal bars, with gaps indicating missing data periods.

### Features

- **Multiple Categories**: Support for displaying multiple categories with different colors
- **Customizable Periods**: Aggregate data by different time periods (daily, weekly)
- **Threshold Filtering**: Filter out values below a specified threshold
- **Date Formatting**: Uses matplotlib's ConciseDateFormatter for clean date axis labels

### Use Cases

- **Data Quality Assessment**: Visualize data availability gaps across categories/segments over time
- **Product Availability Analysis**: Identify periods with stock outs by store/category
- **Seasonality Analysis**: Assess to look for period of low sales that may indicate seasonality or other trends
"""

from typing import Any

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import ensure_data_has_columns, ensure_value_choice
from openretailscience.options import get_option
from openretailscience.plots.styles.colors import get_named_color
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

# Map period aliases (short and long forms) to the canonical pandas frequency code
PERIOD_ALIASES = {
    "D": "D",
    "day": "D",
    "W": "W",
    "week": "W",
}

# Gap threshold and bar duration (in days) keyed by canonical pandas frequency code
PERIOD_GAP_DAYS = {
    "D": 1,
    "W": 7,
}


def _validate_inputs(df: pd.DataFrame, category_col: str, value_col: str, date_col: str) -> None:
    """Validate DataFrame contents for the plot function.

    Args:
        df: Input DataFrame
        category_col: Category column name
        value_col: Value column name
        date_col: Date column name

    Raises:
        ValueError: If DataFrame is empty, or if required columns don't exist.
    """
    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")

    ensure_data_has_columns(df, [date_col, category_col, value_col])


def plot(
    df: pd.DataFrame,
    category_col: str,
    value_col: str,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    ax: Axes | None = None,
    source_text: str | None = None,
    period: str = "D",
    agg_func: str = "sum",
    threshold_value: float | None = None,
    bar_height: float = 0.8,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Creates a broken timeline plot showing data availability across categories over time.

    Shows periods where data is available as horizontal bars, with gaps indicating missing data periods.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data to be plotted.
        category_col (str): The column containing categories to display on y-axis.
        value_col (str): The column containing values to determine data availability.
        title (str, optional): The title of the plot. Defaults to None.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The label for the x-axis. Defaults to None.
        y_label (str, optional): The label for the y-axis. Defaults to None.
        ax (Axes, optional): The Matplotlib Axes object to plot on. Defaults to None.
        source_text (str, optional): Text to be displayed as a source at the bottom of the plot. Defaults to None.
        period (str, optional): Period for aggregating data. Accepts "D"/"day" or "W"/"week"
            (case-insensitive); resolved to the corresponding pandas frequency code internally.
            Defaults to "D".
        agg_func (str, optional): The aggregation function to apply to the value_col when grouping by period.
            Defaults to "sum".
        threshold_value (float, optional): Values below this threshold are considered gaps. Defaults to None.
        bar_height (float, optional): Height of timeline bars as fraction of available space. Defaults to 0.8.
        figsize: tuple[int, int] | None = None,
        **kwargs (Any): Additional keyword arguments for matplotlib broken_barh function.

    Returns:
        SubplotBase: The Matplotlib Axes object with the generated plot.

    Raises:
        ValueError: If DataFrame is empty, required columns are missing, or invalid period specified.
    """
    date_col = get_option("column.transaction_date")

    _validate_inputs(df, category_col, value_col, date_col)
    period = PERIOD_ALIASES[ensure_value_choice(period, list(PERIOD_ALIASES.keys()), "period")]

    # Create a copy of the data and ensure date column is datetime
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])

    # Apply threshold filter if specified
    if threshold_value is not None:
        df_copy = df_copy[df_copy[value_col] >= threshold_value]

    df_copy["period"] = df_copy[date_col].dt.to_period(period)
    df_copy = df_copy.groupby([category_col, "period"]).agg({value_col: agg_func}).reset_index()
    df_copy[date_col] = df_copy["period"].dt.start_time

    # Sort by date once for all categories
    df_copy = df_copy.sort_values(date_col)

    # Get unique categories and create y-axis mapping
    categories = sorted(df_copy[category_col].unique())
    category_to_y = {cat: i for i, cat in enumerate(categories)}

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    period_days = PERIOD_GAP_DAYS[period]
    bar_color = kwargs.pop("color", get_named_color("primary"))
    bar_offset = bar_height / 2

    for category in categories:
        dates = df_copy[df_copy[category_col] == category][date_col].to_numpy()
        dates_num = mdates.date2num(dates)
        gaps = np.diff(dates_num) > period_days
        date_segments = np.split(dates_num, np.where(gaps)[0] + 1)

        segments = [(seg[0], len(seg) * period_days) for seg in date_segments if len(seg) > 0]
        ax.broken_barh(
            segments,
            (category_to_y[category] - bar_offset, bar_height),
            facecolors=bar_color,
            **kwargs,
        )

    # Configure y-axis
    ax.set_yticks(range(len(categories)))
    ax.set_yticklabels(categories)
    ax.invert_yaxis()

    # Configure x-axis for dates
    ax.xaxis_date()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))

    # Apply standard graph styles
    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        source_text=source_text,
        grid_axis="x",
    )
