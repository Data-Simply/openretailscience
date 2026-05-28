"""This module provides functionality for creating index plots in retail analytics.

Index plots are useful for comparing the performance of different categories or segments against
a baseline or average, typically set at 100. The module supports customization of the plot's
appearance, sorting of data, and filtering by specific groups, offering valuable insights into
retail operations.


### Features

- **Index Plot Creation**: Visualize how categories or segments perform relative to a baseline
  value, typically set at 100. Useful for comparing performance across products, regions, or
  customer segments.
- **Flexible Sorting**: Sort data by either group or value to highlight specific trends.
- **Data Filtering**: Filter data based on specified groups to focus on specific categories
  or exclude unwanted data.
- **Highlighting Range**: Highlight specific ranges of values (e.g., performance range between
  80-120) to focus on performance.
- **Series Support**: Optionally include a `series_col` for plotting multiple series (e.g.,
  time periods) within the same plot.
- **Graph Customization**: Adjust titles, axis labels, legend titles, and styling to match the
  specific context of the analysis.

### Use Cases

- **Retail Performance Comparison**: Compare product or regional performance to the company
  average or baseline using an index plot.
- **Customer Segment Analysis**: Evaluate customer segment behavior against overall performance,
  helping identify high-performing segments.
- **Operational Insights**: Identify areas of concern or opportunity by comparing store, region,
  or product performance against the baseline.
- **Visualizing Retail Strategy**: Support decision-making by visualizing which categories or
  products overperform or underperform relative to a baseline.

### Limitations and Handling of Data

- **Data Grouping and Aggregation**: Supports aggregation functions such as sum, average, etc.,
  for calculating the index.
- **Sorting**: Sorting can be applied by group or value, allowing analysts to focus on specific
  trends. If `series_col` is provided, sorting by `group` is applied.
- **Group Filtering**: Users can exclude or include specific groups for focused analysis, with
  error handling to ensure conflicting options are not used simultaneously.

### Functionality Details

- **plot()**: Generates the index plot, which can be customized with multiple options such as
  sorting, filtering, and styling.
- **get_indexes()**: Helper function for calculating the index of the value column for a given
  subset of the dataframe based on filters and aggregation.

"""

from typing import Any, Literal

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import ensure_value_choice
from openretailscience.plots.styles.colors import get_named_color, get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

BASELINE_INDEX = 100
DEFAULT_HIGHLIGHT_RANGE = (80, 120)

VALID_SORT_BY = ("group", "value")
VALID_SORT_ORDERS = ("asc", "ascending", "desc", "descending")
VALID_AGG_FUNCS = ("sum", "mean", "max", "min", "nunique")


def filter_by_groups(
    df: pd.DataFrame,
    group_col: str,
    exclude_groups: list[Any] | None = None,
    include_only_groups: list[Any] | None = None,
) -> pd.DataFrame:
    """Filter dataframe by groups.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        group_col (str): The column name for grouping.
        exclude_groups (list[Any], optional): Groups to exclude. Defaults to None.
        include_only_groups (list[Any], optional): Groups to include. Defaults to None.

    Returns:
        pd.DataFrame: The filtered dataframe.
    """
    result_df = df.copy()
    if exclude_groups is not None:
        result_df = result_df[~result_df[group_col].isin(exclude_groups)]
    if include_only_groups is not None:
        result_df = result_df[result_df[group_col].isin(include_only_groups)]
    return result_df


def filter_by_value_thresholds(
    df: pd.DataFrame,
    filter_above: float | None = None,
    filter_below: float | None = None,
) -> pd.DataFrame:
    """Filter dataframe by index value thresholds.

    Thresholds are expressed in raw-index units (where 100 is the baseline). For example,
    ``filter_above=120`` keeps only rows whose raw index is greater than 120. Internally,
    the dataframe's ``index`` column has the baseline subtracted out, so the comparison
    re-adds the baseline before testing.

    Args:
        df (pd.DataFrame): The dataframe to filter. Its ``index`` column is expected to be
            in delta-from-baseline form (raw index minus 100).
        filter_above (float, optional): Only keep rows whose raw index exceeds this value.
            Defaults to None.
        filter_below (float, optional): Only keep rows whose raw index is below this value.
            Defaults to None.

    Returns:
        pd.DataFrame: The filtered dataframe.

    Raises:
        ValueError: If ``filter_above`` is not strictly less than ``filter_below`` (the
            thresholds form an empty open interval).
        ValueError: If filtering results in an empty dataset.
    """
    if filter_above is not None and filter_below is not None and filter_above >= filter_below:
        error_msg = (
            f"filter_above ({filter_above}) must be < filter_below ({filter_below}); "
            f"otherwise the filter excludes every row."
        )
        raise ValueError(error_msg)

    result_df = df.copy()
    if filter_above is not None:
        result_df = result_df[result_df["index"] + BASELINE_INDEX > filter_above]
    if filter_below is not None:
        result_df = result_df[result_df["index"] + BASELINE_INDEX < filter_below]

    # Check if filtering resulted in an empty dataframe
    if len(result_df) == 0:
        raise ValueError(
            "Filtering resulted in an empty dataset. Consider adjusting filter parameters.",
        )

    return result_df


def filter_top_bottom_n(df: pd.DataFrame, top_n: int | None = None, bottom_n: int | None = None) -> pd.DataFrame:
    """Filter dataframe to include only top N and/or bottom N rows by index value.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        top_n (int, optional): Number of top items to include. Defaults to None.
        bottom_n (int, optional): Number of bottom items to include. Defaults to None.

    Returns:
        pd.DataFrame: The filtered dataframe.

    Raises:
        ValueError: If top_n or bottom_n exceed the available groups.
        ValueError: If the sum of top_n and bottom_n exceeds the total number of groups.
        ValueError: If filtering results in an empty dataset.
    """
    top_n = None if top_n == 0 else top_n
    bottom_n = None if bottom_n == 0 else bottom_n

    if (top_n is None and bottom_n is None) or len(df) == 0:
        return df

    # Check if top_n or bottom_n exceed the dataframe length
    df_length = len(df)
    if top_n is not None and top_n > df_length:
        error_msg = f"top_n ({top_n}) cannot exceed the number of available groups ({df_length})"
        raise ValueError(error_msg)
    if bottom_n is not None and bottom_n > df_length:
        error_msg = f"bottom_n ({bottom_n}) cannot exceed the number of available groups ({df_length})"
        raise ValueError(error_msg)

    # Check if top_n + bottom_n exceeds total groups
    if top_n is not None and bottom_n is not None and top_n + bottom_n > df_length:
        error_msg = (
            f"The sum of top_n ({top_n}) and bottom_n ({bottom_n}) cannot exceed"
            f" the total number of groups ({df_length})"
        )
        raise ValueError(error_msg)

    # Create a temporary dataframe sorted by index value
    temp_df = df.copy().sort_values(by="index", ascending=False)

    selected_rows = pd.DataFrame()
    if top_n is not None:
        selected_rows = pd.concat([selected_rows, temp_df.head(top_n)])
    if bottom_n is not None:
        selected_rows = pd.concat([selected_rows, temp_df.tail(bottom_n)])

    return selected_rows


def plot(  # noqa: C901, PLR0913
    df: pd.DataFrame,
    value_col: str,
    group_col: str,
    index_col: str,
    value_to_index: str,
    agg_func: str = "sum",
    series_col: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    legend_title: str | None = None,
    move_legend_outside: bool = False,
    highlight_range: Literal["default"] | tuple[float, float] | None = "default",
    sort_by: Literal["group", "value"] | None = "group",
    sort_order: Literal["asc", "ascending", "desc", "descending"] = "ascending",
    ax: Axes | None = None,
    source_text: str | None = None,
    exclude_groups: list[Any] | None = None,
    include_only_groups: list[Any] | None = None,
    drop_na: bool = False,
    top_n: int | None = None,
    bottom_n: int | None = None,
    filter_above: float | None = None,
    filter_below: float | None = None,
    color_by_threshold: bool = False,
    **kwargs: Any,  # noqa: ANN401
) -> SubplotBase:
    """Creates an index plot.

    Index plots are visual tools used in retail analytics to compare different categories or segments against a
    baseline or average value, typically set at 100. Index plots allow analysts to:

    - Quickly identify which categories over- or underperform relative to the average
    - Compare performance across diverse categories on a standardized scale
    - Highlight areas of opportunity or concern in retail operations
    - Easily communicate relative performance to stakeholders without revealing sensitive absolute numbers

    In retail contexts, index plots are valuable for:

    - Comparing sales performance across product categories
    - Analyzing customer segment behavior against the overall average
    - Evaluating store or regional performance relative to company-wide metrics
    - Identifying high-potential areas for growth or investment

    By normalizing data to an index, these plots facilitate meaningful comparisons and help focus attention on
    significant deviations from expected performance, supporting more informed decision-making in retail strategy and
    operations.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str): The column to plot.
        group_col (str): The column to group the data by.
        index_col (str): The column to calculate the index on (e.g., "category").
        value_to_index (str): The baseline category or value to index against (e.g., "A").
        agg_func (str, optional): The aggregation function to apply to the value_col. Defaults to "sum".
        series_col (str, optional): The column to use as the series. Defaults to None.
        title (str, optional): The title of the plot. Defaults to None (no title rendered).
        eyebrow (str, optional): Small uppercase label rendered above the title. Defaults to None.
        subtitle (str, optional): Supporting copy rendered below the title. Defaults to None.
        x_label (str, optional): The x-axis label. Defaults to None (no x-axis label rendered).
        y_label (str, optional): The y-axis label. Defaults to None (no y-axis label rendered).
        legend_title (str, optional): The title of the legend. Defaults to None. When None the legend title is set to
            the title case of `group_col`
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area. Defaults to False.
        highlight_range (Literal["default"] | tuple[float, float] | None, optional): The range to highlight. Defaults
            to "default". When "default" the range is set to (80, 120). When None no range is highlighted.
        sort_by (Literal["group", "value"] | None, optional): The column to sort by. Defaults to "group". When None the
            data is not sorted. When "group" the data is sorted by group_col. When "value" the data is sorted by
            the value_col. When series_col is not None this option is ignored.
        sort_order (Literal["asc", "ascending", "desc", "descending"], optional): The order to sort the data. Accepts
            short or long forms, case-insensitive. Defaults to "ascending".
        ax (Axes, optional): The matplotlib axes object to plot on. Defaults to None.
        source_text (str, optional): The source text to add to the plot. Defaults to None.
        exclude_groups (list[Any], optional): The groups to exclude from the plot. Defaults to None.
        include_only_groups (list[Any], optional): The groups to include in the plot. Defaults to None. When None all
            groups are included. When not None only the groups in the list are included. Can not be used with
            exclude_groups.
        drop_na (bool, optional): Whether to drop NA index values. Defaults to False.
        top_n (int, optional): Display only the top N indexes by value. Only applicable
            when series_col is None. Defaults to None.
        bottom_n (int, optional): Display only the bottom N indexes by value. Only applicable
            when series_col is None. Defaults to None.
        filter_above (float, optional): Only display groups whose raw index exceeds this value
            (e.g., ``filter_above=120`` keeps groups indexing above 120). Only applicable when
            series_col is None. Defaults to None.
        filter_below (float, optional): Only display groups whose raw index is below this value
            (e.g., ``filter_below=80`` keeps groups indexing below 80). Only applicable when
            series_col is None. Defaults to None.
        color_by_threshold (bool, optional): Color bars based on highlight_range thresholds using configurable option
            colors. Values >= the upper threshold use the ``plot.color.positive`` option, values <= the lower
            threshold use the ``plot.color.negative`` option, and values between use the ``plot.color.neutral`` option.
            Requires highlight_range to be set (not None). Only applicable when series_col is None. Defaults to False.
        **kwargs: Additional keyword arguments to pass to the Pandas plot function. When
            ``color_by_threshold`` is True, kwargs are passed to matplotlib's ``Axes.barh()`` instead
            and pandas-specific kwargs (figsize, stacked, legend, subplots, layout) are filtered out.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If sort_by is not either "group" or "value" or None.
        ValueError: If sort_order is not one of "asc", "ascending", "desc", or "descending".
        ValueError: If exclude_groups and include_only_groups are used together.
        ValueError: If both top_n and bottom_n are provided but their sum exceeds the total number of groups.
        ValueError: If top_n or bottom_n exceed the number of available groups.
        ValueError: If top_n, bottom_n, filter_above, or filter_below are used when series_col is provided.
        ValueError: If color_by_threshold is True but highlight_range is None.
        ValueError: If color_by_threshold is True when series_col is provided.
        ValueError: If ``filter_above`` is not strictly less than ``filter_below``.
        ValueError: If filtering results in an empty dataset.
    """
    if sort_by is not None:
        ensure_value_choice(sort_by, VALID_SORT_BY, "sort_by")
    if series_col is not None and sort_by == "value":
        raise ValueError("sort_by cannot be 'value' when series_col is provided")
    ensure_value_choice(sort_order, VALID_SORT_ORDERS, "sort_order")
    if exclude_groups is not None and include_only_groups is not None:
        raise ValueError("exclude_groups and include_only_groups cannot be used together")
    if series_col is not None and (
        top_n is not None or bottom_n is not None or filter_above is not None or filter_below is not None
    ):
        raise ValueError(
            "top_n, bottom_n, filter_above, and filter_below cannot be used when series_col is provided",
        )
    if color_by_threshold:
        if highlight_range is None:
            raise ValueError("color_by_threshold requires highlight_range to be set (not None)")
        if series_col is not None:
            raise ValueError("color_by_threshold cannot be used when series_col is provided")

    if highlight_range == "default":
        highlight_range = DEFAULT_HIGHLIGHT_RANGE

    index_df = get_indexes(
        df=df,
        index_col=index_col,
        value_to_index=value_to_index,
        index_subgroup_col=series_col,
        value_col=value_col,
        agg_func=agg_func,
        offset=BASELINE_INDEX,
        group_col=group_col,
    )

    if drop_na:
        index_df = index_df.dropna(subset=["index"])

    index_df = filter_by_groups(
        df=index_df,
        group_col=group_col,
        exclude_groups=exclude_groups,
        include_only_groups=include_only_groups,
    )

    if series_col is None:
        index_df = filter_by_value_thresholds(
            df=index_df,
            filter_above=filter_above,
            filter_below=filter_below,
        )

        default_colors = get_named_color("primary")
        show_legend = False
        index_df = index_df[[group_col, "index"]].set_index(group_col)
        index_df = filter_top_bottom_n(
            df=index_df,
            top_n=top_n,
            bottom_n=bottom_n,
        )

        if sort_by in ["group", "value"]:
            index_df = index_df.sort_values(
                by=group_col if sort_by == "group" else "index",
                ascending=sort_order in ("asc", "ascending"),
            )

    else:
        show_legend = True
        default_colors = get_plot_colors(int(df[series_col].nunique()))

        if sort_by == "group":
            index_df = index_df.sort_values(by=[group_col, series_col], ascending=sort_order in ("asc", "ascending"))

        index_df = index_df.pivot_table(
            index=group_col,
            columns=series_col,
            values="index",
            sort=False,
        )

    width = kwargs.pop("width", 0.8)
    color = kwargs.pop("color", default_colors)

    if color_by_threshold:
        positive_color = get_named_color("positive")
        negative_color = get_named_color("negative")
        neutral_color = get_named_color("neutral")

        values = index_df["index"].to_numpy() + BASELINE_INDEX
        bar_colors = np.select(
            [values >= highlight_range[1], values <= highlight_range[0]],
            [positive_color, negative_color],
            default=neutral_color,
        )
        if ax is None:
            figsize = kwargs.get("figsize")
            _, ax = plt.subplots(figsize=figsize)
        _pandas_only_kwargs = {"figsize", "stacked", "legend", "subplots", "layout"}
        mpl_kwargs = {k: v for k, v in kwargs.items() if k not in _pandas_only_kwargs}
        ax.barh(
            y=index_df.index,
            width=index_df["index"].to_numpy(),
            left=BASELINE_INDEX,
            color=bar_colors,
            height=width,
            zorder=2,
            **mpl_kwargs,
        )
    else:
        ax = index_df.plot.barh(
            left=BASELINE_INDEX,
            legend=show_legend,
            ax=ax,
            color=color,
            width=width,
            zorder=2,
            **kwargs,
        )

    ax.axvline(BASELINE_INDEX, color="black", linewidth=1, alpha=0.5)
    if highlight_range is not None:
        ax.axvline(highlight_range[0], color="black", linewidth=0.25, alpha=0.1, zorder=-1)
        ax.axvline(highlight_range[1], color="black", linewidth=0.25, alpha=0.1, zorder=-1)
        ax.axvspan(highlight_range[0], highlight_range[1], color="black", alpha=0.1, zorder=-1)

    return standard_graph_styles(
        ax=ax,
        title=title,
        eyebrow=eyebrow,
        subtitle=subtitle,
        x_label=x_label,
        y_label=y_label,
        legend_title=legend_title,
        move_legend_outside=move_legend_outside,
        show_legend=show_legend,
        source_text=source_text,
        # Index plots are horizontal bars (`barh`/`hlines`); gridlines belong on the value (x) axis.
        grid_axis="x",
    )


def get_indexes(
    df: pd.DataFrame | ibis.Table,
    value_to_index: str,
    index_col: str,
    value_col: str,
    group_col: str,
    index_subgroup_col: str | None = None,
    agg_func: str = "sum",
    offset: int = 0,
) -> pd.DataFrame:
    """Calculates the index of the value_col using Ibis for efficient computation at scale.

    Args:
        df (pd.DataFrame | ibis.Table): The dataframe or Ibis table to calculate the index on. Can be a pandas
            dataframe or an Ibis table.
        value_to_index (str): The baseline category or value to index against (e.g., "A").
        index_col (str): The column to calculate the index on (e.g., "category").
        value_col (str): The column to calculate the index on (e.g., "sales").
        group_col (str): The column to group the data by (e.g., "region").
        index_subgroup_col (str, optional): The column to subgroup the index by (e.g., "store_type"). Defaults to None.
        agg_func (str, optional): The aggregation function to apply to the `value_col`. Valid options are "sum", "mean",
            "max", "min", or "nunique". Defaults to "sum".
        offset (int, optional): The offset value to subtract from the index. This allows for adjustments to the index
            values. Defaults to 0.

    Returns:
        pd.DataFrame: The calculated index values with grouping columns.
    """
    if isinstance(df, pd.DataFrame):
        df = df.copy()
        table = ibis.memtable(df)
    else:
        table = df

    agg_func = ensure_value_choice(agg_func, VALID_AGG_FUNCS, "agg_func")

    agg_fn = lambda x: getattr(x, agg_func)()  # noqa: E731

    group_cols = [group_col] if index_subgroup_col is None else [index_subgroup_col, group_col]

    overall_agg = table.group_by(group_cols).aggregate(value=agg_fn(table[value_col]))

    if index_subgroup_col is None:
        overall_total = overall_agg.value.sum()
        overall_props = overall_agg.mutate(proportion_overall=overall_agg.value / overall_total.nullif(0))
    else:
        overall_total = overall_agg.group_by(index_subgroup_col).aggregate(total=lambda t: t.value.sum())
        overall_props = (
            overall_agg.join(overall_total, index_subgroup_col)
            .mutate(proportion_overall=lambda t: t.value / t.total.nullif(0))
            .drop("total")
        )

    table = table.filter(table[index_col] == value_to_index)
    subset_agg = table.group_by(group_cols).aggregate(value=agg_fn(table[value_col]))

    if index_subgroup_col is None:
        subset_total = subset_agg.value.sum().name("total")
        subset_props = subset_agg.mutate(proportion=subset_agg.value / subset_total.nullif(0))
    else:
        subset_total = subset_agg.group_by(index_subgroup_col).aggregate(total=lambda t: t.value.sum())
        subset_props = (
            subset_agg.join(subset_total, index_subgroup_col)
            .filter(lambda t: t.total != 0)
            .mutate(proportion=lambda t: t.value / t.total)
            .drop("total")
        )

    result = (
        subset_props.join(overall_props, group_cols)
        .mutate(
            index=lambda t: (t.proportion / t.proportion_overall.nullif(0) * 100) - offset,
        )
        .order_by(group_cols)
    )

    return result[[*group_cols, "index"]].execute()
