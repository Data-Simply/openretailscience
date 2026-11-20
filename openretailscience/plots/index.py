"""Index plots comparing group performance against a baseline, where an index of 100 is parity.

Use ``index.plot`` for relative performance against a reference category; ``plots.time`` for raw
values over time; ``plots.bar`` for categorical comparison. For the counts and shares behind each
index as a table, use ``openretailscience.analysis.IndexAnalysis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openretailscience.analysis.index_analysis import BASELINE_INDEX, IndexAnalysis
from openretailscience.core.validation import (
    VALID_SORT_ORDERS,
    ensure_columns,
    ensure_ibis_table,
    ensure_value_choice,
)
from openretailscience.plots.styles.colors import get_named_color, get_plot_colors
from openretailscience.plots.styles.styling_helpers import standard_graph_styles

if TYPE_CHECKING:
    import ibis
    from matplotlib.axes import Axes, SubplotBase

DEFAULT_HIGHLIGHT_RANGE = (80, 120)
DEFAULT_BAR_WIDTH = 0.8

VALID_SORT_BY = ("group", "value")


def plot(  # noqa: PLR0913, PLR0917
    df: pd.DataFrame | ibis.Table,
    value_col: str,
    group_col: str,
    index_col: str,
    value_to_index: str | int,
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

    Each group's index is ``100 * (its share of the aggregated value_col among rows where
    index_col == value_to_index) / (its share of the aggregated value_col over all rows)``; bars run
    from a baseline line at 100 to the index. Every group in the data gets a bar, at 0 when no
    ``value_to_index`` row has it. Rows with a null ``group_col`` are left out of both totals.

    Args:
        df (pd.DataFrame | ibis.Table): The data to plot.
        value_col (str): The column to aggregate.
        group_col (str): The column whose values get a bar each.
        index_col (str): The column that defines the subset (e.g., "brand").
        value_to_index (str | int): The ``index_col`` value whose rows form the subset (e.g., "Meadow Farm").
        agg_func (str, optional): How to aggregate ``value_col``: "sum", "mean", "max", "min" or "nunique".
        series_col (str, optional): The column to use as the series; when set, a legend is rendered
            and each series value gets its own bar per group. ``top_n``, ``bottom_n``,
            ``filter_above``, ``filter_below``, and ``color_by_threshold`` cannot be combined with it.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): X-axis label; omitted means no label.
        y_label (str, optional): Y-axis label; omitted means no label.
        legend_title (str, optional): The title of the legend.
        move_legend_outside (bool, optional): Whether to move the legend outside the plot area.
        highlight_range (Literal["default"] | tuple[float, float] | None, optional): The range to
            highlight with a shaded span and vertical lines at the bounds. "default" means (80, 120);
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
        filter_above (float, optional): Only display groups whose index exceeds this value.
            Only applicable when ``series_col`` is None.
        filter_below (float, optional): Only display groups whose index is below this value.
            Only applicable when ``series_col`` is None.
        color_by_threshold (bool, optional): Color bars by the ``highlight_range`` thresholds: an
            index >= the upper bound uses the ``plot.color.positive`` option, <= the lower bound
            uses ``plot.color.negative``, and values in between use ``plot.color.neutral``. Requires
            ``highlight_range`` to be set (not None); cannot be used with ``series_col``.
        **kwargs: Additional keyword arguments, normally passed to pandas' ``barh``. When
            ``color_by_threshold`` is True they are passed to matplotlib's ``Axes.barh()`` instead,
            excluding the pandas-only kwargs (``figsize``, ``stacked``, ``legend``, ``subplots``,
            ``layout``). ``figsize`` still sizes the figure when ``ax`` is None. ``width`` defaults to
            0.8. ``color`` defaults to the primary color (a palette with ``series_col``) and is ignored
            when ``color_by_threshold`` is True.

    Returns:
        SubplotBase: The matplotlib axes object.

    Raises:
        ValueError: If sort_by is not "group", "value", or None.
        TypeError: If df is not a pandas DataFrame or an Ibis Table.
        ValueError: If a column is missing or series_col is group_col.
        ValueError: If agg_func is not one of "sum", "mean", "max", "min", or "nunique", or has no numeric
            result for value_col.
        ValueError: If sort_by is "value" and series_col is provided.
        ValueError: If sort_order is not one of "asc", "ascending", "desc", or "descending".
        ValueError: If exclude_groups and include_only_groups are used together.
        ValueError: If top_n, bottom_n, filter_above, or filter_below are used when series_col is provided.
        ValueError: If color_by_threshold is True but highlight_range is None.
        ValueError: If color_by_threshold is True when series_col is provided.
        ValueError: If filter_above is not strictly less than filter_below.
        ValueError: If no row with a non-null group has index_col == value_to_index.
        ibis.common.exceptions.IbisTypeError: If value_to_index can't be compared with index_col.
        ValueError: If filtering results in an empty dataset.
        ValueError: If top_n or bottom_n is negative or exceeds the number of available groups.
        ValueError: If top_n and bottom_n together exceed the number of available groups.

    Examples:
        >>> from openretailscience.plots import index
        >>> ax = index.plot(
        ...     df=transactions, value_col="unit_spend", group_col="category", index_col="store_id", value_to_index=101,
        ...     eyebrow="Store 101", title="Category mix against the chain", subtitle="Index, 100 = chain average",
        ... )
    """
    table = ensure_ibis_table(df)
    sort_by, sort_order = _validate_display_args(
        table=table,
        group_col=group_col,
        series_col=series_col,
        sort_by=sort_by,
        sort_order=sort_order,
        exclude_groups=exclude_groups,
        include_only_groups=include_only_groups,
        top_n=top_n,
        bottom_n=bottom_n,
        filter_above=filter_above,
        filter_below=filter_below,
        highlight_range=highlight_range,
        color_by_threshold=color_by_threshold,
    )
    ascending = sort_order in ("asc", "ascending")

    if highlight_range == "default":
        highlight_range = DEFAULT_HIGHLIGHT_RANGE

    index_df = IndexAnalysis(
        table,
        value_col=value_col,
        group_col=group_col,
        index_col=index_col,
        value_to_index=value_to_index,
        agg_func=agg_func,
        subgroup_col=series_col,
    ).df
    if drop_na:
        index_df = index_df.dropna(subset=["index"])
    index_df = _filter_by_groups(
        df=index_df,
        group_col=group_col,
        exclude_groups=exclude_groups,
        include_only_groups=include_only_groups,
    )
    index_df = _filter_by_value_thresholds(df=index_df, filter_above=filter_above, filter_below=filter_below)
    if len(index_df) == 0:
        raise ValueError("Filtering resulted in an empty dataset. Consider adjusting filter parameters.")

    if series_col is None:
        index_df = index_df[[group_col, "index"]].set_index(group_col)
        index_df = _filter_top_bottom_n(df=index_df, top_n=top_n, bottom_n=bottom_n)
        if sort_by is not None:
            index_df = index_df.sort_values(by=group_col if sort_by == "group" else "index", ascending=ascending)
        default_colors = get_named_color("primary")
        show_legend = False
    else:
        if sort_by == "group":
            index_df = index_df.sort_values(by=[group_col, series_col], ascending=ascending)
        index_df = index_df.pivot_table(index=group_col, columns=series_col, values="index", sort=False)
        default_colors = get_plot_colors(len(index_df.columns))
        show_legend = True

    bar_widths = index_df - BASELINE_INDEX
    width = kwargs.pop("width", DEFAULT_BAR_WIDTH)
    color = kwargs.pop("color", default_colors)

    if color_by_threshold:
        values = index_df["index"].to_numpy()
        bar_colors = np.select(
            [values >= highlight_range[1], values <= highlight_range[0]],
            [get_named_color("positive"), get_named_color("negative")],
            default=get_named_color("neutral"),
        )
        if ax is None:
            _, ax = plt.subplots(figsize=kwargs.get("figsize"))
        pandas_only_kwargs = {"figsize", "stacked", "legend", "subplots", "layout"}
        mpl_kwargs = {k: v for k, v in kwargs.items() if k not in pandas_only_kwargs}
        ax.barh(
            # Strings make matplotlib place numeric groups (store ids) as categories, like pandas' barh
            y=bar_widths.index.astype(str),
            width=bar_widths["index"].to_numpy(),
            left=BASELINE_INDEX,
            color=bar_colors,
            height=width,
            zorder=2,
            **mpl_kwargs,
        )
    else:
        ax = bar_widths.plot.barh(
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
        # Index plots are horizontal bars; gridlines belong on the value (x) axis.
        grid_axis="x",
    )


def _validate_display_args(
    *,
    table: ibis.Table,
    group_col: str,
    series_col: str | None,
    sort_by: str | None,
    sort_order: str,
    exclude_groups: list[Any] | None,
    include_only_groups: list[Any] | None,
    top_n: int | None,
    bottom_n: int | None,
    filter_above: float | None,
    filter_below: float | None,
    highlight_range: Literal["default"] | tuple[float, float] | None,
    color_by_threshold: bool,
) -> tuple[str | None, str]:
    """Check plot's display arguments before any query runs, and return the canonical sort options.

    Args:
        table (ibis.Table): The plot's data, for checking that ``series_col`` exists.
        group_col (str): plot's ``group_col``.
        series_col (str | None): plot's ``series_col``.
        sort_by (str | None): plot's ``sort_by``, any case.
        sort_order (str): plot's ``sort_order``, any case or form.
        exclude_groups (list[Any] | None): plot's ``exclude_groups``.
        include_only_groups (list[Any] | None): plot's ``include_only_groups``.
        top_n (int | None): plot's ``top_n``.
        bottom_n (int | None): plot's ``bottom_n``.
        filter_above (float | None): plot's ``filter_above``.
        filter_below (float | None): plot's ``filter_below``.
        highlight_range (Literal["default"] | tuple[float, float] | None): plot's ``highlight_range``.
        color_by_threshold (bool): plot's ``color_by_threshold``.

    Returns:
        tuple[str | None, str]: ``sort_by`` and ``sort_order`` as spelled in their valid choices.

    Raises:
        ValueError: For the invalid or conflicting arguments listed in plot's Raises section.
    """
    sort_by = None if sort_by is None else ensure_value_choice(sort_by, VALID_SORT_BY, "sort_by")
    sort_order = ensure_value_choice(sort_order, VALID_SORT_ORDERS, "sort_order")
    if series_col is not None:
        ensure_columns(table, series_col, "series_col")
        if series_col == group_col:
            raise ValueError("series_col must differ from group_col")
        if sort_by == "value":
            raise ValueError("sort_by cannot be 'value' when series_col is provided")
        if any(value is not None for value in (top_n, bottom_n, filter_above, filter_below)):
            raise ValueError(
                "top_n, bottom_n, filter_above, and filter_below cannot be used when series_col is provided",
            )
        if color_by_threshold:
            raise ValueError("color_by_threshold cannot be used when series_col is provided")
    if exclude_groups is not None and include_only_groups is not None:
        raise ValueError("exclude_groups and include_only_groups cannot be used together")
    if (top_n is not None and top_n < 0) or (bottom_n is not None and bottom_n < 0):
        error_msg = f"top_n ({top_n}) and bottom_n ({bottom_n}) must not be negative"
        raise ValueError(error_msg)
    if filter_above is not None and filter_below is not None and filter_above >= filter_below:
        error_msg = (
            f"filter_above ({filter_above}) must be < filter_below ({filter_below}); "
            f"otherwise the filter excludes every row."
        )
        raise ValueError(error_msg)
    if color_by_threshold and highlight_range is None:
        raise ValueError("color_by_threshold requires highlight_range to be set (not None)")
    return sort_by, sort_order


def _filter_by_groups(
    df: pd.DataFrame,
    group_col: str,
    exclude_groups: list[Any] | None = None,
    include_only_groups: list[Any] | None = None,
) -> pd.DataFrame:
    """Drop the excluded groups, or keep only the included ones.

    Args:
        df (pd.DataFrame): The index table.
        group_col (str): The column holding the groups.
        exclude_groups (list[Any], optional): Groups to drop.
        include_only_groups (list[Any], optional): The only groups to keep.

    Returns:
        pd.DataFrame: The filtered table.
    """
    if exclude_groups is not None:
        df = df[~df[group_col].isin(exclude_groups)]
    if include_only_groups is not None:
        df = df[df[group_col].isin(include_only_groups)]
    return df


def _filter_by_value_thresholds(
    df: pd.DataFrame,
    filter_above: float | None = None,
    filter_below: float | None = None,
) -> pd.DataFrame:
    """Keep the rows whose ``index`` is strictly above ``filter_above`` and below ``filter_below``.

    Args:
        df (pd.DataFrame): The index table.
        filter_above (float, optional): Keep only rows whose index exceeds this value.
        filter_below (float, optional): Keep only rows whose index is below this value.

    Returns:
        pd.DataFrame: The filtered table.
    """
    if filter_above is not None:
        df = df[df["index"] > filter_above]
    if filter_below is not None:
        df = df[df["index"] < filter_below]
    return df


def _filter_top_bottom_n(df: pd.DataFrame, top_n: int | None = None, bottom_n: int | None = None) -> pd.DataFrame:
    """Keep the top N and/or bottom N rows by ``index``, ranking only rows whose index is not null.

    ``0`` is treated as None for both parameters.

    Args:
        df (pd.DataFrame): The index table.
        top_n (int, optional): Number of highest-index rows to keep.
        bottom_n (int, optional): Number of lowest-index rows to keep.

    Returns:
        pd.DataFrame: The selected rows, highest index first.

    Raises:
        ValueError: If top_n or bottom_n, or their sum, exceeds the number of rows.
    """
    top_n = None if top_n == 0 else top_n
    bottom_n = None if bottom_n == 0 else bottom_n

    if top_n is None and bottom_n is None:
        return df

    sorted_df = df.dropna(subset=["index"]).sort_values(by="index", ascending=False)
    df_length = len(sorted_df)
    if top_n is not None and top_n > df_length:
        error_msg = f"top_n ({top_n}) cannot exceed the number of available groups ({df_length})"
        raise ValueError(error_msg)
    if bottom_n is not None and bottom_n > df_length:
        error_msg = f"bottom_n ({bottom_n}) cannot exceed the number of available groups ({df_length})"
        raise ValueError(error_msg)
    if top_n is not None and bottom_n is not None and top_n + bottom_n > df_length:
        error_msg = (
            f"The sum of top_n ({top_n}) and bottom_n ({bottom_n}) cannot exceed"
            f" the total number of groups ({df_length})"
        )
        raise ValueError(error_msg)

    selected = []
    if top_n is not None:
        selected.append(sorted_df.head(top_n))
    if bottom_n is not None:
        selected.append(sorted_df.tail(bottom_n))
    return pd.concat(selected)
