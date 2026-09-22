"""Index plots comparing group performance against a baseline, where a raw index of 100 is parity.

Use ``index.plot`` for relative performance against a reference category; ``plots.time`` for raw
values over time; ``plots.bar`` for categorical comparison. ``plot()`` draws the bars;
``get_indexes()`` computes the index values.
"""

from typing import Any, Literal

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes, SubplotBase

from openretailscience.core.validation import VALID_SORT_ORDERS, ensure_ibis_table, ensure_value_choice
from openretailscience.plots.styles.colors import get_named_color, get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

BASELINE_INDEX = 100
DEFAULT_HIGHLIGHT_RANGE = (80, 120)

VALID_SORT_BY = ("group", "value")
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
    """Filter a dataframe by index value thresholds.

    The ``index`` column is in delta-from-baseline form (raw index minus 100); thresholds are in
    raw-index units, so the baseline is re-added before comparing (``filter_above=120`` keeps rows
    whose raw index exceeds 120).

    Args:
        df (pd.DataFrame): The dataframe to filter.
        filter_above (float, optional): Only keep rows whose raw index exceeds this value.
        filter_below (float, optional): Only keep rows whose raw index is below this value.

    Returns:
        pd.DataFrame: The filtered dataframe.

    Raises:
        ValueError: If ``filter_above`` is not strictly less than ``filter_below``.
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
    """Filter a dataframe to the top N and/or bottom N rows by index value.

    Rows are ranked by the ``index`` column, descending. ``0`` is treated as ``None`` for both
    parameters; an empty ``df`` is returned unchanged.

    Args:
        df (pd.DataFrame): The dataframe to filter.
        top_n (int, optional): Number of top rows to include.
        bottom_n (int, optional): Number of bottom rows to include.

    Returns:
        pd.DataFrame: The filtered dataframe.

    Raises:
        ValueError: If top_n or bottom_n exceeds the number of available groups.
        ValueError: If the sum of top_n and bottom_n exceeds the number of groups.
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
    """Create a horizontal bar index plot comparing each group's performance against a baseline.

    The index for each group is
    ``100 * (group share of the aggregated value_col among rows where index_col ==
    value_to_index) / (group share of the aggregated value_col overall) - 100``,
    so a delta of 0 is parity; bars are drawn from the 100 baseline outward, with a vertical
    baseline line at 100.

    Args:
        df (pd.DataFrame): The dataframe to plot.
        value_col (str): The column to plot.
        group_col (str): The column to group the data by.
        index_col (str): The column to calculate the index on (e.g., "category").
        value_to_index (str): The baseline category or value to index against (e.g., "A").
        agg_func (str, optional): The aggregation function to apply to ``value_col``.
        series_col (str, optional): The column to use as the series; when set, a legend is rendered
            and each series value gets its own bar per group. ``top_n``, ``bottom_n``,
            ``filter_above``, ``filter_below``, and ``color_by_threshold`` cannot be combined with it.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): The x-axis label.
        y_label (str, optional): The y-axis label.
        legend_title (str, optional): The title of the legend.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area.
        highlight_range (Literal["default"] | tuple[float, float] | None, optional): The range to
            highlight with a shaded span and vlines at the bounds. "default" means (80, 120);
            None disables it.
        sort_by (Literal["group", "value"] | None, optional): "group" sorts by ``group_col`` (and by
            ``series_col`` as a second key when it is set); "value" sorts by the index column; None
            skips sorting. Cannot be "value" when ``series_col`` is set.
        sort_order (Literal["asc", "ascending", "desc", "descending"], optional): The sort order.
            Accepts short or long forms, case-insensitive.
        ax (Axes, optional): The matplotlib axes object to plot on.
        source_text (str, optional): The source text to add to the plot.
        exclude_groups (list[Any], optional): The groups to exclude from the plot.
        include_only_groups (list[Any], optional): The groups to include in the plot. Cannot be used
            with ``exclude_groups``.
        drop_na (bool, optional): Whether to drop NA index values.
        top_n (int, optional): Display only the top N indexes. Only applicable when ``series_col``
            is None.
        bottom_n (int, optional): Display only the bottom N indexes. Only applicable when
            ``series_col`` is None.
        filter_above (float, optional): Only display groups whose raw index exceeds this value.
            Only applicable when ``series_col`` is None.
        filter_below (float, optional): Only display groups whose raw index is below this value.
            Only applicable when ``series_col`` is None.
        color_by_threshold (bool, optional): Color bars by the ``highlight_range`` thresholds: raw
            index >= the upper bound uses the ``plot.color.positive`` option, <= the lower bound
            uses ``plot.color.negative``, and values in between use ``plot.color.neutral``. Requires
            ``highlight_range`` to be set (not None); cannot be used with ``series_col``.
        **kwargs: Additional keyword arguments, normally passed to pandas' ``barh``. When
            ``color_by_threshold`` is True they are passed to matplotlib's ``Axes.barh()`` instead,
            excluding the pandas-only kwargs (``figsize``, ``stacked``, ``legend``, ``subplots``,
            ``layout``). ``figsize`` still sizes the figure when ``ax`` is None. ``width`` (default
            0.8) and ``color`` are popped and consumed.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If sort_by is not "group", "value", or None.
        ValueError: If sort_by is "value" and series_col is provided.
        ValueError: If sort_order is not one of "asc", "ascending", "desc", or "descending".
        ValueError: If exclude_groups and include_only_groups are used together.
        ValueError: If top_n, bottom_n, filter_above, or filter_below are used when series_col is provided.
        ValueError: If color_by_threshold is True but highlight_range is None.
        ValueError: If color_by_threshold is True when series_col is provided.
        ValueError: If filter_above is not strictly less than filter_below.
        ValueError: If filtering results in an empty dataset.
        ValueError: If top_n or bottom_n exceeds the number of available groups.
        ValueError: If top_n and bottom_n together exceed the number of available groups.
    """
    if sort_by is not None:
        sort_by = ensure_value_choice(sort_by, VALID_SORT_BY, "sort_by")
    if series_col is not None and sort_by == "value":
        raise ValueError("sort_by cannot be 'value' when series_col is provided")
    sort_order = ensure_value_choice(sort_order, VALID_SORT_ORDERS, "sort_order")
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
    """Calculate each group's index of ``value_col`` against a baseline category.

    Index = ``100 * (group share of the aggregated value_col among rows where index_col ==
    value_to_index) / (group share of the aggregated value_col overall) - offset``. When
    ``index_subgroup_col`` is set, shares are computed within each subgroup instead of
    overall.

    Args:
        df (pd.DataFrame | ibis.Table): The dataframe or Ibis table to calculate the index on.
        value_to_index (str): The baseline category or value to index against (e.g., "A").
        index_col (str): The column to calculate the index on (e.g., "category").
        value_col (str): The column to calculate the index of (e.g., "sales").
        group_col (str): The column to group the data by (e.g., "region").
        index_subgroup_col (str, optional): The column to subgroup the index by (e.g., "store_type").
        agg_func (str, optional): The aggregation function to apply to ``value_col``. Valid options
            are "sum", "mean", "max", "min", or "nunique".
        offset (int, optional): Subtracted from the computed index. Defaults to 0; ``plot`` passes
            100 so its result is delta-from-baseline.

    Returns:
        pd.DataFrame: The group columns plus an ``index`` column.

    Raises:
        ValueError: If agg_func is not one of "sum", "mean", "max", "min", or "nunique".
    """
    table = ensure_ibis_table(df)

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
