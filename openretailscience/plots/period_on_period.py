"""Period on period module.

This module provides functionality for plotting multiple overlapping time periods
from the same time series on a single line chart using matplotlib.

The `plot` function is useful for visual comparisons of temporal trends
across different time windows, with each time window plotted as a separate line
but aligned to a common starting point.

Example use case: Comparing sales data across multiple promotional weeks or seasonal periods.
"""

from datetime import datetime
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from openretailscience.plots.styles.colors import get_sequential_cmap
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

LINE_STYLES = [
    "-",  # solid
    "--",  # dashed
    ":",  # dotted
    "-.",  # dashdot
    (0, (5, 10)),  # long dash with offset
    (0, (6, 6)),  # loosely dashed
    (0, (3, 5, 1, 5)),  # loosely dashdotted
    (0, (3, 5, 1, 5, 1, 5)),  # loosely dashdotdotted
]

SEQUENTIAL_SAMPLE_DARKEST = 0.85
SEQUENTIAL_SAMPLE_LIGHTEST = 0.5

LINEWIDTH_NEWEST = 2.5
LINEWIDTH_OLDEST = 1.25


def plot(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    periods: list[tuple[str | datetime, str | datetime]],
    x_label: str | None = None,
    y_label: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    source_text: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    legend_style: Literal["box", "end_of_line"] | None = None,
    ax: Axes | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plot multiple overlapping periods from a single time series as individual lines.

    This function is used to align and overlay several time intervals from the same
    dataset to facilitate visual comparison. Each period is realigned to the reference
    start date and plotted as a separate line using a distinct linestyle.

    Note:
        The `periods` argument accepts a list of (start_date, end_date) tuples,
        which define the time windows to overlay. Each element in the tuple can be either
        a string (e.g., "2022-01-01") or a `datetime` object. You can use
        `find_overlapping_periods` from `openretailscience.utils.date` to generate
        the `periods` input automatically.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        x_col (str): Name of the column representing datetime values.
        value_col (str): Name of the column representing the y-axis values (e.g. sales, counts).
        periods (List[Tuple[Union[str, datetime], Union[str, datetime]]]):
            A list of (start_date, end_date) tuples representing the periods to plot.
        x_label (Optional[str]): Custom label for the x-axis.
        y_label (Optional[str]): Custom label for the y-axis.
        title (Optional[str]): Title for the plot.
        eyebrow (Optional[str]): Small uppercase label rendered above the title.
        subtitle (Optional[str]): Supporting copy rendered below the title.
        source_text (Optional[str]): Text to show below the plot as a data source.
        legend_title (Optional[str]): Title for the plot legend.
        move_legend_outside (bool): Whether to place the legend outside the plot area.
        legend_style (Literal["box", "end_of_line"], optional): How periods are labelled. ``"box"`` renders the
            standard legend; ``"end_of_line"`` suppresses the legend and places a colored period label at the
            right end of each line.
        ax (Optional[Axes]): Matplotlib Axes object to draw on. If None, a new one is created.
        figsize (tuple[int, int], optional): Size of the new figure when ``ax`` is None. Defaults to None.
        **kwargs: Additional keyword arguments passed to the base line plot function.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object with the completed plot.

    Raises:
        ValueError: The 'periods' list must contain at least two (start, end) tuples for comparison.
        ValueError: If `legend_style` is not one of ``None``, ``"box"``, or ``"end_of_line"``.
    """
    if legend_style not in (None, "box", "end_of_line"):
        msg = f"legend_style must be one of (None, 'box', 'end_of_line'); got {legend_style!r}"
        raise ValueError(msg)

    min_period_length = 2
    if len(periods) < min_period_length:
        raise ValueError("The 'periods' list must contain at least two (start, end) tuples for comparison")

    parsed_periods = [(pd.to_datetime(start), pd.to_datetime(end)) for start, end in periods]
    start_ref = parsed_periods[0][0]

    sorted_periods = sorted(parsed_periods, reverse=True, key=lambda x: x[0])

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    period_styles = {period: LINE_STYLES[idx % len(LINE_STYLES)] for idx, period in enumerate(sorted_periods)}

    cmap = get_sequential_cmap()
    color_samples = np.linspace(SEQUENTIAL_SAMPLE_DARKEST, SEQUENTIAL_SAMPLE_LIGHTEST, len(sorted_periods))
    period_colors = {period: cmap(t) for period, t in zip(sorted_periods, color_samples, strict=True)}
    # Newer periods draw thicker; uniform user-supplied linewidth wins if provided.
    user_linewidth = kwargs.pop("linewidth", None)
    if user_linewidth is None:
        linewidth_samples = np.linspace(LINEWIDTH_NEWEST, LINEWIDTH_OLDEST, len(sorted_periods))
        period_linewidths = dict(zip(sorted_periods, linewidth_samples, strict=True))
    else:
        period_linewidths = dict.fromkeys(sorted_periods, user_linewidth)
    # Newer periods draw on top of older ones regardless of caller-supplied period order.
    period_zorder = {period: len(sorted_periods) - idx + 2 for idx, period in enumerate(sorted_periods)}

    df = df.copy()
    df[x_col] = pd.to_datetime(df[x_col])

    start_ref_year = start_ref.year

    for start, end in parsed_periods:
        period_key = (start, end)
        linestyle = period_styles[period_key]
        color = period_colors[period_key]
        linewidth = period_linewidths[period_key]
        zorder = period_zorder[period_key]
        period_df = df[(df[x_col] >= start) & (df[x_col] <= end)].copy()

        if period_df.empty:
            continue

        year_diff = start.year - start_ref_year

        period_df["realigned_date"] = period_df[x_col] - pd.DateOffset(years=year_diff)

        ax.plot(
            period_df["realigned_date"],
            period_df[value_col],
            linestyle=linestyle,
            color=color,
            linewidth=linewidth,
            zorder=zorder,
            label=f"{start.date()} to {end.date()}",
            **kwargs,
        )

    # ax.plot() only labels artists; without an explicit ax.legend() call no Legend
    # is attached, so standard_graph_styles' legend gate (which requires
    # ax.get_legend() is not None) silently skips legend styling and the periods
    # render without a key. end_of_line styling removes this legend downstream.
    ax.legend()

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=True,
        legend_style=legend_style,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )
