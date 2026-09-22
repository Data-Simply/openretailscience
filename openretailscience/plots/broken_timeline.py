"""Broken timeline of data availability: horizontal bars per category over time.

Each bar spans periods with data; gaps mark missing periods. Values are aggregated per
period (period D/W via agg_func) rather than plotted as raw rows; threshold filtering is
supported.
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
    """Plot data availability as horizontal bars per category, with gaps for missing periods.

    Values are aggregated per period (groupby category + period, then agg_func on value_col)
    before plotting. The date column is not a parameter; it is read from the
    ``column.transaction_date`` option and must be present in df.

    Args:
        df (pd.DataFrame): Input frame; must include the ``column.transaction_date`` column.
        category_col (str): Column of categories, one bar row per category.
        value_col (str): Column of values used to determine availability.
        title (str, optional): Plot title.
        eyebrow (str, optional): Uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): X-axis label.
        y_label (str, optional): Y-axis label.
        ax (Axes, optional): Axes to plot on.
        source_text (str, optional): Source attribution rendered at the bottom.
        period (str, optional): Aggregation period; "D"/"day" or "W"/"week" (case-insensitive).
        agg_func (str, optional): Aggregation applied to value_col within each period.
        threshold_value (float, optional): Rows below this value are dropped before period
            aggregation; a period that still has data above the threshold gets a shorter bar,
            not a gap.
        bar_height (float, optional): Bar thickness as a fraction of the available row space.
        figsize (tuple[int, int], optional): Figure size, used only when ax is None.
        **kwargs: Forwarded to matplotlib broken_barh.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If df is empty, a required column is missing, or period is invalid.

    """
    date_col = get_option("column.transaction_date")

    if df.empty:
        raise ValueError("Cannot plot with empty DataFrame")
    ensure_data_has_columns(df, [date_col, category_col, value_col])
    period = PERIOD_ALIASES[ensure_value_choice(period, PERIOD_ALIASES, "period")]

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
