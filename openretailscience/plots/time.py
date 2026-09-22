"""Timeline plots of a value column resampled and aggregated by period from raw transactional data.

Aggregation happens here (``line.plot`` does not aggregate), over the configured transaction-date
column (``column.transaction_date``). Use this module for an actual datetime x-axis; use
``plots.line`` for relative or sequential x-values.
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
    """Plot ``value_col`` over time, aggregated by ``period``.

    ``df`` must contain the configured transaction-date column (``column.transaction_date``) as a
    datetime column. Data is aggregated here via ``groupby(to_period(period)).agg(agg_func)``;
    ``line.plot`` does not aggregate. With ``group_col`` the result is a pivoted multi-line plot
    with a legend; without it, a single line with no legend.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str): The column to plot.
        period (str | BaseOffset): The period to group the data by (e.g., "D", "W", "M", "Q", "Y",
            or a pandas ``BaseOffset``).
        agg_func (str, optional): The aggregation function to apply to ``value_col``.
        group_col (str, optional): The column to group the data by.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): The matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot.
        legend_style (Literal["box", "end_of_line"], optional): How series are labelled. ``"box"``
            (default when None) renders the standard legend; ``"end_of_line"`` suppresses the
            legend and places a colored series label at the right end of each line. When
            ``"end_of_line"``, ``move_legend_outside`` and ``legend_title`` are ignored and a
            warning is emitted if either is supplied.
        **kwargs: Additional keyword arguments for pandas' ``plot`` function. ``linewidth``
            (default 3) and ``color`` (default the primary color, or the palette with
            ``group_col``) are popped and consumed.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If legend_style is not one of ``None``, ``"box"``, or ``"end_of_line"``.
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
