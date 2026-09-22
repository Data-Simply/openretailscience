"""Overlay several time windows of one series on a single line chart.

Each window is shifted by whole years onto the first period's year. Use
``utils.date.find_overlapping_periods`` to build the periods. Use ``plots.time`` for one
continuous resampled series; use this module to compare multiple windows.
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

    Each period is shifted by ``start.year - first_period.start.year`` whole years, so the x-axis
    is calendar position within the year of the first tuple; windows whose in-year start dates
    differ do not line up at x=0. Periods with no matching rows are silently skipped.

    Newer periods are drawn darker, thicker, and on top; a user-supplied ``linewidth`` overrides
    the thickness gradient.

    Args:
        df (pd.DataFrame): Input DataFrame containing the time series data.
        x_col (str): Name of the column representing datetime values.
        value_col (str): Name of the column representing the y-axis values (e.g. sales, counts).
        periods (list[tuple[str | datetime, str | datetime]]):
            A list of at least two (start_date, end_date) tuples, each element a string or
            datetime. Use ``find_overlapping_periods`` from ``openretailscience.utils.date`` to
            generate them automatically.
        x_label (str, optional): Custom label for the x-axis.
        y_label (str, optional): Custom label for the y-axis.
        title (str, optional): Title for the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        source_text (str, optional): Text to show below the plot as a data source.
        legend_title (str, optional): Title for the plot legend.
        move_legend_outside (bool, optional): Whether to place the legend outside the plot area.
        legend_style (Literal["box", "end_of_line"], optional): How periods are labelled. ``"box"``
            (default when None) renders the standard legend; ``"end_of_line"`` suppresses the
            legend and places a colored period label at the right end of each line.
        ax (Axes, optional): Matplotlib Axes object to draw on. If None, a new one is created.
        figsize (tuple[int, int], optional): Size of the new figure when ``ax`` is None.
        **kwargs: Additional keyword arguments passed to ``ax.plot``.

    Returns:
        matplotlib.axes.Axes: The matplotlib Axes object with the completed plot.

    Raises:
        ValueError: If periods contains fewer than two (start, end) tuples.
        ValueError: If legend_style is not one of ``None``, ``"box"``, or ``"end_of_line"``.
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
