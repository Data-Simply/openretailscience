"""Histograms of one or more value columns, optionally split by a group column.

Plots raw rows directly (no pre-aggregation). ``clip_range=(lower, upper)`` clamps out-of-range
values to the edge bins; matplotlib's native ``range`` kwarg drops them instead. A Series is
plotted as a single histogram; ``group_col`` cannot be used with one.
"""

from typing import Any

import pandas as pd
from matplotlib.axes import Axes, SubplotBase

import openretailscience.plots.styles.graph_utils as gu
from openretailscience.plots.styles.colors import get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

CLIP_RANGE_LENGTH = 2


def plot(
    df: pd.DataFrame | pd.Series,
    value_col: str | list[str] | None = None,
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
    clip_range: tuple[float | None, float | None] | None = None,
    use_hatch: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Plot a histogram of ``value_col``, optionally split by ``group_col``.

    With ``group_col``, one histogram is drawn per unique group value (via a pivot); a list
    ``value_col`` cannot be combined with it. When ``df`` is a Series, ``value_col`` (if given)
    becomes the single column name and the Series' own ``name`` is discarded; ``group_col``
    cannot be used with a Series and will error.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe (or series) to plot.
        value_col (str or list of str, optional): The column(s) to plot.
        group_col (str, optional): The column used to define different histograms.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        clip_range (tuple[float | None, float | None], optional): ``(lower, upper)`` bounds that
            clamp out-of-range values to the nearest boundary so they pile up at the edge bins.
            Pass ``None`` on either side for one-sided clipping. Mutually exclusive with
            matplotlib's ``range`` kwarg, which drops out-of-range values instead.
        use_hatch (bool, optional): Whether to use hatching for the bars.
        **kwargs: Additional keyword arguments for pandas' ``plot`` function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If both clip_range and matplotlib's range kwarg are specified.
        ValueError: If clip_range is not a 2-tuple or has lower greater than upper.
        ValueError: If value_col is a list and group_col is also provided.
        ValueError: If df is a DataFrame and value_col is None.
    """
    if isinstance(value_col, list) and group_col is not None:
        raise ValueError("`value_col` cannot be a list when `group_col` is provided. Please choose one or the other.")

    if clip_range is not None and "range" in kwargs:
        raise ValueError(
            "Cannot specify both `range` and `clip_range`. Use `clip_range` to clamp outliers to the edge bins, "
            "or `range` to drop out-of-range values entirely.",
        )

    value_col = _prepare_value_col(df=df, value_col=value_col)

    if isinstance(df, pd.Series):
        df = df.to_frame(name=value_col[0])

    if clip_range is not None and len(clip_range) != CLIP_RANGE_LENGTH:
        msg = f"clip_range must be a 2-tuple of (lower, upper); got length {len(clip_range)}"
        raise ValueError(msg)

    if clip_range is not None:
        clip_lower, clip_upper = clip_range
        if clip_lower is not None and clip_upper is not None and clip_lower > clip_upper:
            msg = f"clip_range lower ({clip_lower}) must be <= upper ({clip_upper})"
            raise ValueError(msg)
        df = df.assign(**{col: df[col].clip(lower=clip_lower, upper=clip_upper) for col in value_col})

    num_histograms = _get_num_histograms(df=df, value_col=value_col, group_col=group_col)

    default_colors = get_plot_colors(num_histograms)
    colors = kwargs.pop("color", default_colors)

    ax = _plot_histogram(
        df=df,
        value_col=value_col,
        group_col=group_col,
        ax=ax,
        colors=colors,
        num_histograms=num_histograms,
        **kwargs,
    )

    if use_hatch:
        ax = gu.apply_hatches(ax=ax, num_segments=num_histograms)

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=num_histograms > 1,
        source_text=source_text,
        grid_axis="y",
        x_margin=0,
    )


def _prepare_value_col(df: pd.DataFrame | pd.Series, value_col: str | list[str] | None) -> list[str]:
    """Normalize ``value_col`` to a list of column names.

    For a Series, ``value_col`` is not a lookup: it becomes the column name (defaulting to
    ``"value"``). For a DataFrame it is required, and a single string is promoted to a list.

    Args:
        df (pd.DataFrame | pd.Series): The input dataframe or series.
        value_col (str or list of str, optional): The column(s) to plot.

    Returns:
        list[str]: The processed value_col as a list of strings.

    Raises:
        ValueError: If df is a DataFrame and value_col is None.
    """
    if isinstance(df, pd.Series):
        return ["value"] if value_col is None else [value_col]

    if value_col is None:
        raise ValueError("Please provide a value column to plot")

    if isinstance(value_col, str):
        value_col = [value_col]

    return value_col


def _get_num_histograms(df: pd.DataFrame, value_col: list[str], group_col: str | None) -> int:
    """Calculate the number of histograms to be plotted.

    ``max(len(value_col), nunique(group_col))`` — the color list is sized for the wider of the
    two so both multi-column and multi-group plots get one color per histogram.

    Args:
        df (pd.DataFrame): The dataframe being plotted.
        value_col (list of str): The column(s) being plotted.
        group_col (str, optional): The column used for grouping data into histograms.

    Returns:
        int: The number of histograms to plot.
    """
    num_histograms = len(value_col)

    if group_col is not None:
        num_histograms = max(num_histograms, df[group_col].nunique())

    return num_histograms


def _plot_histogram(
    df: pd.DataFrame,
    value_col: list[str],
    group_col: str | None,
    ax: Axes | None,
    colors: list[str],
    num_histograms: int,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plot histograms for the provided dataframe.

    With ``group_col`` the data is pivoted and all group columns are plotted at once. Alpha
    defaults to 0.7 for multiple histograms and None for a single one.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (list of str): The column(s) to plot.
        group_col (str, optional): The column used to group data into multiple histograms.
        ax (Axes, optional): Matplotlib axes object to plot on.
        colors (list[str]): The list of colors used for the plot.
        num_histograms (int): The number of histograms being plotted.
        **kwargs: Additional keyword arguments for pandas' ``plot`` function.

    Returns:
        Axes: The matplotlib axes object with the plotted histogram.
    """
    is_multi_histogram = num_histograms > 1

    alpha = kwargs.pop("alpha", 0.7) if is_multi_histogram else kwargs.pop("alpha", None)

    if group_col is None:
        return df[value_col].plot(
            kind="hist",
            ax=ax,
            legend=is_multi_histogram,
            color=colors,
            alpha=alpha,
            **kwargs,
        )

    # if group_col is provided, only use a single value_col
    df_pivot = df.pivot(columns=group_col, values=value_col[0])

    # Plot all columns at once
    return df_pivot.plot(
        kind="hist",
        ax=ax,
        legend=is_multi_histogram,
        alpha=alpha,
        color=colors,
        **kwargs,
    )
