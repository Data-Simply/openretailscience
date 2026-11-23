"""Gain loss (switching) analysis of customer movement between a focus and a comparison group across two periods."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import ibis

from openretailscience.core.validation import (
    ensure_agg_func,
    ensure_columns,
    ensure_data_has_columns,
    ensure_excludes_customer_col,
    ensure_ibis_table,
    ensure_value_choice,
    unknown_customer_flag,
)
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots import bar
from openretailscience.plots.styles.colors import COLORS

if TYPE_CHECKING:
    import datetime as dt

    import pandas as pd
    from matplotlib.axes import Axes, SubplotBase

# Additive per customer only: the summary sums customer values, and switching assumes value moves between groups
VALID_AGG_FUNCS = ("sum", "count", "nunique")

VALUE_COLS = [
    "focus_p1",
    "comparison_p1",
    "total_p1",
    "focus_p2",
    "comparison_p2",
    "total_p2",
    "focus_diff",
    "comparison_diff",
    "total_diff",
]
COMPONENTS = [
    "new",
    "lost",
    "increased_focus",
    "decreased_focus",
    "switch_from_comparison",
    "switch_to_comparison",
]
# Rows matching each selected value, carried into the summary so df can tell a misspelt value from a zero total
_ROW_COUNTS = {
    "p1_value": "_p1_rows",
    "p2_value": "_p2_rows",
    "focus_value": "_focus_rows",
    "comparison_value": "_comparison_rows",
}


class GainLoss:
    """Gain loss (switching) analysis between a focus group and a comparison group.

    ``customer_table`` has one row per customer, or per ``group_col`` value and customer when grouped:
    ``customer_id`` (after ``group_col``), then ``focus``/``comparison``/``total`` for ``_p1``,
    ``_p2`` and ``_diff``, then the six components ``new``, ``lost``, ``increased_focus``,
    ``decreased_focus``, ``switch_from_comparison`` and ``switch_to_comparison``, which sum to
    ``focus_diff``. It is lazy and unordered; filter it before executing on large data.

    ``table`` and ``df`` hold the same value columns summed over customers: one row when
    ungrouped, one row per ``group_col`` value (with ``group_col`` first, null as its own group) otherwise.

    Rows with a null customer ID, or the ``data.unknown_customer_value`` ID when that option is set, are left out.

    Examples:
        >>> from openretailscience.analysis import GainLoss
        >>> gl = GainLoss(
        ...     df=transactions, period_col="period", p1_value="2025Q1", p2_value="2025Q2",
        ...     compare_col="brand", focus_value="Meadow Farm", comparison_value="Greenvale",
        ... )
        >>> ax = gl.plot(
        ...     eyebrow="Brand switching", title="Where Meadow Farm spend came from and went", subtitle="Q2 vs Q1 2025"
        ... )
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str | int | dt.date,
        p2_value: str | int | dt.date,
        compare_col: str,
        focus_value: str | int,
        comparison_value: str | int,
        group_col: str | None = None,
        value_col: str | None = None,
        agg_func: str = "sum",
    ) -> None:
        """Build the gain loss queries; they run when ``df`` is read or a table is executed.

        To compare against more than one value (e.g. every other brand in the category), or to show
        display names in place of codes, label a column first with
        ``openretailscience.utils.filter_and_label_by_condition`` and select on that column. Label date
        ranges for ``period_col`` with ``openretailscience.utils.filter_and_label_by_periods``.

        Args:
            df (pd.DataFrame | ibis.Table): Transactions with the option-derived customer_id column,
                ``value_col``, ``period_col`` and ``compare_col``.
            period_col (str): Column holding a period label per row (e.g. "2025Q1").
            p1_value (str | int | dt.date): Label of the first (base) period.
            p2_value (str | int | dt.date): Label of the second period.
            compare_col (str): Column holding the focus and comparison groups (e.g. "brand").
            focus_value (str | int): Value of ``compare_col`` for the focus group; also its plot label.
            comparison_value (str | int): Value of ``compare_col`` for the comparison group; also its plot label.
            group_col (str | None): Optional column to break the analysis down by.
            value_col (str | None): Value column to aggregate; defaults to option ``column.unit_spend``.
            agg_func (str): "sum", "count" or "nunique". Defaults to "sum".

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table, if ``agg_func`` is not a string, or
                if the ``data.unknown_customer_value`` option can't be compared with customer_id.
            ValueError: If ``p1_value`` equals ``p2_value``, ``focus_value`` equals ``comparison_value``, or
                ``period_col`` equals ``compare_col``.
            ValueError: If the option-derived customer_id, ``value_col``, ``period_col`` or ``compare_col``
                columns are missing, or if a given ``group_col`` is not a single column of ``df``, or is the
                customer column, ``period_col`` or ``compare_col``.
            ValueError: If ``agg_func`` is not one of the valid options or has no numeric result for ``value_col``.
            ibis.common.exceptions.IbisTypeError: If a period value can't be compared with ``period_col``, or a
                group value with ``compare_col`` (e.g. a str label on an int column).
        """
        cols = ColumnHelper()
        value_col = cols.unit_spend if value_col is None else value_col
        table = ensure_ibis_table(df)

        if p1_value == p2_value:
            msg = f"p1_value and p2_value must be different, got {p1_value!r} for both."
            raise ValueError(msg)
        if focus_value == comparison_value:
            msg = f"focus_value and comparison_value must be different, got {focus_value!r} for both."
            raise ValueError(msg)
        if period_col == compare_col:
            msg = f"period_col and compare_col must be different, got {period_col!r} for both."
            raise ValueError(msg)
        if group_col is not None:
            group_cols = ensure_columns(table, group_col, "group_col")
            if len(group_cols) > 1:
                msg = f"group_col must be a single column, got {group_cols}."
                raise ValueError(msg)
            group_col = group_cols[0]
            ensure_excludes_customer_col(group_cols, "group_col")
            if group_col in (period_col, compare_col):
                msg = f"group_col must differ from period_col and compare_col, got {group_col!r}."
                raise ValueError(msg)
        ensure_data_has_columns(table, [cols.customer_id, value_col, period_col, compare_col])
        agg_func = ensure_value_choice(agg_func, VALID_AGG_FUNCS, "agg_func")
        ensure_agg_func(table, value_col, agg_func, numeric=True)

        self.focus_value = focus_value
        self.comparison_value = comparison_value
        self.group_col = group_col
        self.value_col = value_col
        self._selection = {
            "p1_value": (p1_value, period_col),
            "p2_value": (p2_value, period_col),
            "focus_value": (focus_value, compare_col),
            "comparison_value": (comparison_value, compare_col),
        }

        unknown_customer_value = get_option("data.unknown_customer_value")
        identified = (
            table[cols.customer_id].notnull()  # noqa: PD004 (ibis API)
            if unknown_customer_value is None
            else ~unknown_customer_flag(table, unknown_customer_value)
        )
        # ==, not isin: isin leaves a value it can't compare (an int on a text column) to fail in the backend
        rows = table.filter(
            (table[period_col] == p1_value) | (table[period_col] == p2_value),
            (table[compare_col] == focus_value) | (table[compare_col] == comparison_value),
            identified,
        )
        # Complements, not equality tests: values that select the same rows (e.g. "2025-01-01" and that date) leave
        # p2 or comparison with no rows, so df raises instead of counting the rows in both
        in_p1 = rows[period_col] == p1_value
        in_p2 = ~in_p1
        is_focus = rows[compare_col] == focus_value
        is_comparison = ~is_focus
        aggregate = getattr(rows[value_col], agg_func)
        keys = [cols.customer_id] if group_col is None else [group_col, cols.customer_id]
        customers = rows.group_by(keys).aggregate(
            focus_p1=aggregate(where=in_p1 & is_focus).fill_null(0),
            comparison_p1=aggregate(where=in_p1 & is_comparison).fill_null(0),
            total_p1=aggregate(where=in_p1).fill_null(0),
            focus_p2=aggregate(where=in_p2 & is_focus).fill_null(0),
            comparison_p2=aggregate(where=in_p2 & is_comparison).fill_null(0),
            total_p2=aggregate(where=in_p2).fill_null(0),
            **{
                _ROW_COUNTS["p1_value"]: rows.count(where=in_p1),
                _ROW_COUNTS["p2_value"]: rows.count(where=in_p2),
                _ROW_COUNTS["focus_value"]: rows.count(where=is_focus),
                _ROW_COUNTS["comparison_value"]: rows.count(where=is_comparison),
            },
        )
        customers = _decompose(customers)
        self.customer_table: ibis.Table = customers.select(*keys, *VALUE_COLS, *COMPONENTS)

        # max, not sum, for the row counts: df only tests them for zero, and SQL Server sums them as INT, which
        # overflows past 2^31 rows
        sums = {col: customers[col].sum() for col in [*VALUE_COLS, *COMPONENTS]} | {
            col: customers[col].max() for col in _ROW_COUNTS.values()
        }
        if group_col is None:
            self._summary = customers.aggregate(**sums)
        else:
            self._summary = customers.group_by(group_col).aggregate(**sums)
        self.table: ibis.Table = self._summary.drop(*_ROW_COUNTS.values())

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """The summary as a pandas DataFrame, ordered by ``group_col`` when grouped.

        Raises:
            ValueError: If ``p1_value``, ``p2_value``, ``focus_value`` or ``comparison_value`` matches no row
                within the other selection (e.g. a focus value with no rows in either period).
        """
        # Sorted here, not in table: SQL Server rejects ORDER BY in the derived table that selecting from table makes
        summary = (self._summary if self.group_col is None else self._summary.order_by(self.group_col)).execute()
        unmatched = [
            f"{param} {value!r} matches no rows of '{col}'"
            for param, (value, col) in self._selection.items()
            if summary[_ROW_COUNTS[param]].sum() == 0
        ]
        if len(unmatched) > 0:
            detail = f"{'; '.join(unmatched)} among the selected periods and groups"
            if len(unmatched) == len(self._selection):
                # Every count is 0 when either pair matches nothing, so the counts can't tell which pair is wrong
                (p1, period_col), (p2, _), (focus, compare_col), (comparison, _) = self._selection.values()
                detail = f"No rows have '{period_col}' {p1!r} or {p2!r} and '{compare_col}' {focus!r} or {comparison!r}"
            msg = (
                f"{detail} with a customer ID. Pass values that exist in the data, matching their type, case and "
                "spacing exactly."
            )
            raise ValueError(msg)
        return summary.drop(columns=list(_ROW_COUNTS.values()))

    def plot(
        self,
        title: str | None = None,
        eyebrow: str | None = None,
        subtitle: str | None = None,
        x_label: str | None = None,
        y_label: str | None = None,
        ax: Axes | None = None,
        source_text: str | None = None,
        move_legend_outside: bool = False,
        **kwargs: Any,  # noqa: ANN401
    ) -> SubplotBase:
        """Plot the summary as a stacked horizontal bar of the six decomposition columns.

        Green segments are ``new`` / ``increased_focus`` / ``switch_from_comparison``; red
        segments are ``lost`` / ``decreased_focus`` / ``switch_to_comparison``. Legend labels
        use the focus and comparison values. ``stacked``, ``color`` and ``x_col`` kwargs
        are silently ignored.

        Args:
            title (str | None): Plot title; defaults to "Gain Loss from {focus} to {comparison}".
            eyebrow (str | None): Small uppercase label rendered above the title.
            subtitle (str | None): Supporting copy rendered below the title.
            x_label (str | None): X-axis label; defaults to the value column name.
            y_label (str | None): Y-axis label; defaults to the focus value, or ``group_col`` when grouped.
            ax (Axes | None): Axes to plot on.
            source_text (str | None): Source annotation.
            move_legend_outside (bool): Whether to move the legend outside the plot.
            **kwargs (Any): Additional keyword arguments forwarded to the bar plot.

        Returns:
            SubplotBase: The plot axes.

        Raises:
            ValueError: If a selected value matches no rows, as for ``df``.
        """
        increase_cols = ["new", "increased_focus", "switch_from_comparison"]
        decrease_cols = ["lost", "decreased_focus", "switch_to_comparison"]
        all_cols = increase_cols + decrease_cols
        segment_colors = [
            COLORS["green"][700],
            COLORS["green"][500],
            COLORS["green"][300],
            COLORS["red"][700],
            COLORS["red"][500],
            COLORS["red"][300],
        ]
        legend_labels = [
            "New",
            f"Increased {self.focus_value}",
            f"Switch From {self.comparison_value}",
            "Lost",
            f"Decreased {self.focus_value}",
            f"Switch To {self.comparison_value}",
        ]

        default_y_label = str(self.focus_value) if self.group_col is None else self.group_col
        plot_data = self.df
        if self.group_col is None:
            # Blank the single bar's tick label, which would otherwise be the row number; the y label names it
            plot_data = plot_data.set_axis([""])

        kwargs.pop("stacked", None)
        kwargs.pop("color", None)
        kwargs.pop("x_col", None)

        resolved_title = title if title is not None else f"Gain Loss from {self.focus_value} to {self.comparison_value}"

        ax = bar.plot(
            df=plot_data,
            value_col=all_cols,
            x_col=self.group_col,
            title=resolved_title,
            eyebrow=eyebrow,
            subtitle=subtitle,
            y_label=y_label if y_label is not None else default_y_label,
            x_label=x_label if x_label is not None else self.value_col,
            orientation="horizontal",
            ax=ax,
            source_text=source_text,
            move_legend_outside=move_legend_outside,
            legend_labels=legend_labels,
            stacked=True,
            color=segment_colors,
            **kwargs,
        )

        ax.axvline(0, color="black", linewidth=0.5)

        return ax


def _decompose(customers: ibis.Table) -> ibis.Table:
    """Add the diffs and the six components to a table of per-customer period values.

    When both groups are zero in p1, all of focus p2 is ``new``; when both are zero in p2, all of
    focus p1 is ``lost``. Otherwise the focus change is split into a part matched by an opposite
    comparison change (switching) and the rest (increased or decreased focus).

    Args:
        customers (ibis.Table): One row per customer with ``focus``, ``comparison`` and ``total`` for
            ``_p1`` and ``_p2``.

    Returns:
        ibis.Table: ``customers`` with the ``_diff`` columns and the six components added.
    """
    # Staged mutates: in one, the SQL would repeat the focus change CASE in all four components
    customers = customers.mutate(
        focus_diff=customers.focus_p2 - customers.focus_p1,
        comparison_diff=customers.comparison_p2 - customers.comparison_p1,
        total_diff=customers.total_p2 - customers.total_p1,
    )
    focus_diff, comparison_diff = customers.focus_diff, customers.comparison_diff
    is_new = (customers.focus_p1 == 0) & (customers.comparison_p1 == 0)
    is_lost = ~is_new & (customers.focus_p2 == 0) & (customers.comparison_p2 == 0)
    customers = customers.mutate(
        new=is_new.ifelse(customers.focus_p2, 0),
        lost=is_lost.ifelse(-customers.focus_p1, 0),
        _focus_change=ibis.cases(
            (is_new | is_lost, 0),
            ((focus_diff > 0) & (comparison_diff > 0), focus_diff),
            (focus_diff > 0, ibis.greatest(0, comparison_diff + focus_diff)),
            (comparison_diff < 0, focus_diff),
            else_=ibis.least(0, comparison_diff + focus_diff),
        ),
    )
    switched = customers.focus_diff - customers.new - customers.lost - customers._focus_change
    return customers.mutate(
        increased_focus=ibis.greatest(0, customers._focus_change),
        decreased_focus=ibis.least(0, customers._focus_change),
        switch_from_comparison=ibis.greatest(0, switched),
        switch_to_comparison=ibis.least(0, switched),
    )
