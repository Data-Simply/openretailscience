"""This module provides flexible functionality for creating histograms from pandas DataFrames or Series.

It allows you to visualize distributions of one or more value columns and optionally group them by a categorical column.
The module is designed to handle both DataFrames and Series, allowing you to create simple histograms or compare
distributions across categories by splitting the data into multiple histograms.

### Core Features

- **Single or Multiple Histograms**: Plot one or more value columns (**`value_col`**) as histograms.
For example, visualize the distribution of a single metric or compare multiple metrics simultaneously.
- **Grouped Histograms**: Create separate histograms for each unique value in **`group_col`** (e.g., product categories
or regions), allowing for easy comparison of distributions across groups.
- **Outlier-Preserving Clipping**: Use **`clip_range=(lower, upper)`** to clamp out-of-range values to the boundary
so they pile up at the edge bins instead of being dropped. Pass `None` on either side for one-sided clipping
(e.g., `clip_range=(0, None)` to clamp negatives only). To drop out-of-range values instead, pass matplotlib's
native `range=(lower, upper)` through `**kwargs`.
- **Comprehensive Customization**: Customize plot titles, axis labels, and legends, with the option to
move the legend outside the plot.

### Use Cases

- **Distribution Analysis**: Visualize the distribution of key metrics like revenue, sales, or user activity using
single or multiple histograms.
- **Group Comparisons**: Compare distributions across different groups, such as product categories,
geographic regions, or customer segments. For instance, plot histograms to show how sales vary across
different product categories.
- **Outlier Visibility**: Use **`clip_range`** to keep extreme values visible at the edge bins rather than
dropping them, so the shape of the central mass is readable without hiding how much sits beyond it.

### Limitations and Handling of Data

- **Pre-Aggregated Data Required**: This module does not perform any data aggregation, so all data must
be pre-aggregated before being passed in for plotting.
- **Grouped Histograms**: If **`group_col`** is provided, the data will be pivoted so that each unique value in
**`group_col`** becomes a separate histogram. Otherwise, a single histogram is plotted.
- **Series Support**: The module can also handle pandas Series, though **`group_col`** cannot be provided
when plotting a Series.

### Additional Features

- **Outlier-Preserving Clipping**: `clip_range=(lower, upper)` clamps values outside the bounds to the nearest
boundary so the edge bins absorb the outlier mass. This differs from matplotlib's native `range`, which drops
out-of-range values entirely. The two are mutually exclusive.
- **Legend Customization**: For multiple histograms, you can add legends, including the option to move the
legend outside the plot for clarity.

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
    """Plots a histogram of `value_col`, optionally split by `group_col`.

    Args:
        df (pd.DataFrame | pd.Series): The dataframe (or series) to plot.
        value_col (str or list of str, optional): The column(s) to plot.
            Can be a list of columns for multiple histograms.
        group_col (str, optional): The column used to define different histograms.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        ax (Axes, optional): Matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        move_legend_outside (bool, optional): Move the legend outside the plot.
        clip_range (tuple[float | None, float | None], optional): `(lower, upper)` bounds that clamp out-of-range
            values to the nearest boundary so they pile up at the edge bins. Pass `None` on either side for
            one-sided clipping. Mutually exclusive with matplotlib's `range` kwarg, which drops out-of-range
            values instead.
        use_hatch (bool, optional): Whether to use hatching for the bars.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If both `clip_range` and matplotlib's `range` kwarg are specified; if `clip_range` is not a
            2-tuple or has lower greater than upper; or if `value_col` is a list and `group_col` is also provided.
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
    """Ensures that value_col is properly handled and returned as a list.

    Args:
        df (pd.DataFrame | pd.Series): The input dataframe or series.
        value_col (str or list of str, optional): The column(s) to plot. If a single string, it is converted to a list.

    Returns:
        list[str]: The processed value_col as a list of strings.
    """
    if isinstance(df, pd.Series):
        return ["value"] if value_col is None else [value_col]

    if value_col is None:
        raise ValueError("Please provide a value column to plot")

    if isinstance(value_col, str):
        value_col = [value_col]

    return value_col


def _get_num_histograms(df: pd.DataFrame, value_col: list[str], group_col: str | None) -> int:
    """Calculates the number of histograms to be plotted.

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
    """Plots histograms for the provided dataframe.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (list of str): The column(s) to plot.
        group_col (str, optional): The column used to group data into multiple histograms.
        ax (Axes, optional): Matplotlib axes object to plot on.
        colors: The list of colors use for the plot.
        num_histograms (int): The number of histograms being plotted.
        **kwargs: Additional keyword arguments for Pandas' `plot` function.

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
