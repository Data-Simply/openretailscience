"""Decompose period-over-period revenue change into per-factor contributions.

Revenue = customers x spend per customer = customers x transactions per customer x
spend per transaction, extended with units per transaction x price per unit when
``unit_quantity`` is present (those conditional columns define the ``.df`` output).
With the ``data.unknown_customer_value`` option set, revenue = unknown-customer revenue +
identified revenue; the identified part is decomposed as above, and the unknown and total revenue
as transactions x spend per transaction. Provides ``RevenueTree`` (with ``.draw_tree()``) and
``calc_tree_kpis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import numpy as np
import pandas as pd

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_dataframe,
    ensure_ibis_table,
    unknown_customer_flag,
)
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots.styles import graph_utils as gu
from openretailscience.plots.tree_diagram import DetailedTreeNode, TreeGrid

if TYPE_CHECKING:
    from datetime import date

    from matplotlib.axes import Axes

MAX_PERIODS_SHOWN = 10  # period values listed in a "p1_value/p2_value not found" error


def _contribution_ratio(df: pd.DataFrame, ratio: str, numerator: str, denominator: str) -> tuple[pd.Series, pd.Series]:
    """Return a ratio's (p1, p2) values for the contribution split.

    A 0 / 0 period takes the other period's value, and a ratio that is 0 / 0 in both periods is 0 so its factors
    contribute 0; a non-zero amount over a zero count stays NaN.

    Args:
        df (pd.DataFrame): Frame holding the period-suffixed ratio, numerator and denominator columns.
        ratio (str): Unsuffixed ratio column name.
        numerator (str): Unsuffixed numerator column name.
        denominator (str): Unsuffixed denominator column name.

    Returns:
        tuple[pd.Series, pd.Series]: The period 1 and period 2 ratio values.
    """
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
    empty_p1 = (df[f"{numerator}_{p1}"] == 0) & (df[f"{denominator}_{p1}"] == 0)
    empty_p2 = (df[f"{numerator}_{p2}"] == 0) & (df[f"{denominator}_{p2}"] == 0)
    both_empty = empty_p1 & empty_p2
    ratio_p1, ratio_p2 = df[f"{ratio}_{p1}"], df[f"{ratio}_{p2}"]
    return ratio_p1.mask(empty_p1, ratio_p2).mask(both_empty, 0), ratio_p2.mask(empty_p2, ratio_p1).mask(both_empty, 0)


def _variant(name: str, variant: str | None) -> str:
    """Return ``name`` with the unknown or total ``variant`` suffix, or unchanged for the identified metrics (None).

    Args:
        name (str): The identified metric's column name.
        variant (str | None): The ``column.suffix.unknown_customer`` or ``column.suffix.total`` suffix, or None.

    Returns:
        str: The variant's column name.
    """
    return name if variant is None else f"{name}_{variant}"


def _in_both(p1_df: pd.DataFrame, p2_df: pd.DataFrame, col: str) -> bool:
    """Return whether both period frames have ``col``.

    Args:
        p1_df (pd.DataFrame): Period 1 aggregates.
        p2_df (pd.DataFrame): Period 2 aggregates.
        col (str): The optional column.

    Returns:
        bool: True if both frames have it, False if neither does.

    Raises:
        ValueError: If only one frame has it.
    """
    in_p1 = col in p1_df.columns
    if in_p1 != (col in p2_df.columns):
        msg = f"p1_df and p2_df must both have {col!r} or neither."
        raise ValueError(msg)
    return in_p1


def _visit_ratio_parts(variant: str | None, has_quantity: bool) -> dict[str, tuple[str, str]]:
    """Return the spend per transaction branch's ``{ratio: (numerator, denominator)}`` columns for one variant.

    Args:
        variant (str | None): The unknown or total suffix, or None for the identified metrics.
        has_quantity (bool): Whether to include units per transaction and price per unit.

    Returns:
        dict[str, tuple[str, str]]: Spend per transaction first, then the units and price ratios.
    """
    cols = ColumnHelper()
    trans, spend, units = (
        _variant(c, variant) for c in (cols.agg.transaction_id, cols.agg.unit_spend, cols.agg.unit_qty)
    )
    parts = {_variant(cols.calc.spend_per_trans, variant): (spend, trans)}
    if has_quantity:
        parts[_variant(cols.calc.units_per_trans, variant)] = (units, trans)
        parts[_variant(cols.calc.price_per_unit, variant)] = (spend, units)
    return parts


def _add_period_metrics(df: pd.DataFrame, counts: list[str], ratio_parts: dict[str, tuple[str, str]]) -> None:
    """Add each ratio's period columns, then the diff and pct_diff of every count and ratio, to ``df`` in place.

    Args:
        df (pd.DataFrame): Frame holding the period-suffixed count columns.
        counts (list[str]): Unsuffixed count columns.
        ratio_parts (dict[str, tuple[str, str]]): ``{ratio: (numerator, denominator)}`` unsuffixed column names.
    """
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
    for ratio, (numerator, denominator) in ratio_parts.items():
        for period in (p1, p2):
            df[f"{ratio}_{period}"] = df[f"{numerator}_{period}"] / df[f"{denominator}_{period}"].replace(0, np.nan)

    diff, pct_diff = get_option("column.suffix.difference"), get_option("column.suffix.percent_difference")
    for col in [*counts, *ratio_parts]:
        df[f"{col}_{diff}"] = df[f"{col}_{p2}"] - df[f"{col}_{p1}"]
        df[f"{col}_{pct_diff}"] = df[f"{col}_{diff}"] / df[f"{col}_{p1}"].replace(0, np.nan)


def _add_spend_per_trans_contribs(
    df: pd.DataFrame,
    variant: str | None,
    has_quantity: bool,
    scale: pd.Series,
) -> pd.Series:
    """Add the contributions of spend per transaction and, with units, of units per transaction and price per unit.

    Args:
        df (pd.DataFrame): Frame holding the variant's period ratio columns.
        variant (str | None): The unknown or total suffix, or None for the identified metrics.
        has_quantity (bool): Whether the units and price ratios are present.
        scale (pd.Series): Product of the midpoints above this branch, which every contribution in it is
            multiplied by: mean transactions per customer x mean customers for the identified metrics, mean
            transactions for the unknown and total ones.

    Returns:
        pd.Series: Midpoint spend per transaction, which its sibling's change is multiplied by.
    """
    contrib = get_option("column.suffix.contribution")
    parts = _visit_ratio_parts(variant, has_quantity)
    spend_per_trans, *quantity_ratios = parts
    a1, a2 = _contribution_ratio(df, spend_per_trans, *parts[spend_per_trans])
    df[f"{spend_per_trans}_{contrib}"] = (a2 - a1) * scale
    if has_quantity:
        units_per_trans, price_per_unit = quantity_ratios
        u1, u2 = _contribution_ratio(df, units_per_trans, *parts[units_per_trans])
        pr1, pr2 = _contribution_ratio(df, price_per_unit, *parts[price_per_unit])
        df[f"{units_per_trans}_{contrib}"] = (u2 - u1) * (pr1 + pr2) / 2 * scale
        df[f"{price_per_unit}_{contrib}"] = (pr2 - pr1) * (u1 + u2) / 2 * scale
    return (a1 + a2) / 2


def _pair_periods(p1_df: pd.DataFrame, p2_df: pd.DataFrame, keys: list[str], columns: list[str]) -> pd.DataFrame:
    """Join the two periods' ``columns`` side by side on ``keys``, suffixed by period, sorted with a NaN group last.

    A group missing from a period gets 0 for that period. With no ``keys``, each frame holds at most one row.

    Args:
        p1_df (pd.DataFrame): Period 1 aggregates.
        p2_df (pd.DataFrame): Period 2 aggregates.
        keys (list[str]): Group columns to pair rows on; empty when ungrouped.
        columns (list[str]): Metric columns to take from each period.

    Returns:
        pd.DataFrame: The ``keys`` columns, then each of ``columns`` suffixed by period.
    """
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
    if len(keys) == 0:
        df = pd.concat(
            [
                p1_df[columns].add_suffix(f"_{p1}").reset_index(drop=True),
                p2_df[columns].add_suffix(f"_{p2}").reset_index(drop=True),
            ],
            axis=1,
        )
    else:
        # The merge puts a categorical NaN group first; sorting puts it last
        df = (
            p1_df[[*keys, *columns]]
            .merge(p2_df[[*keys, *columns]], on=keys, how="outer", suffixes=(f"_{p1}", f"_{p2}"))
            .sort_values(keys, ignore_index=True)
        )
    period_cols = [f"{col}_{period}" for period in (p1, p2) for col in columns]
    df[period_cols] = df[period_cols].fillna(0)
    return df


def calc_tree_kpis(p1_df: pd.DataFrame, p2_df: pd.DataFrame, group_col: str | list[str] | None = None) -> pd.DataFrame:
    """Calculate the revenue-tree KPIs from per-period aggregates.

    Each frame has one row per group with the option-derived columns ``column.agg.customer_id``,
    ``column.agg.transaction_id``, ``column.agg.unit_spend`` and, optionally,
    ``column.agg.unit_quantity``. Rows are paired across periods on ``group_col``, missing
    values included; a group missing from one frame gets 0 counts for that period. Without
    ``group_col``, each frame holds at most one row.

    To split out unknown customers, both frames also carry the transactions, spend and (with units)
    units of unknown-customer rows and of all rows, as ``{column}_unknown`` and ``{column}_total``
    (``column.suffix.unknown_customer`` and ``column.suffix.total``), with the plain columns covering
    identified customers only, as ``SegTransactionStats`` returns them.

    Output columns are ``{metric}_p1`` / ``_p2`` / ``_diff`` / ``_pct_diff`` (suffixes
    from the ``column.suffix.*`` options) plus ``{metric}_contrib`` contribution
    columns (quantity/price columns present only when units were). ``pct_diff = diff / p1``,
    with p1 = 0 giving NaN. With unknown customers split out, the ``_unknown`` and then the
    ``_total`` transactions, spend, spend per transaction (and units, units per transaction and
    price per unit) follow, each set with its own contributions to its spend.

    Each ``_contrib`` is in spend units: the factor's change times the period midpoints
    of its sibling and of each ancestor's sibling. With C = customers, S = spend per customer,
    T = transactions per customer, A = spend per transaction, U = units per
    transaction, P = price per unit, dX = X_p2 - X_p1 and mX = (X_p1 + X_p2) / 2:

        customers                  dC * mS
        spend_per_customer         dS * mC
        transactions_per_customer  dT * mA * mC
        spend_per_transaction      dA * mT * mC
        units_per_transaction      dU * mP * mT * mC
        price_per_unit             dP * mU * mT * mC

    The unknown and total sets have no customers, so their trees start from transactions, with T
    here the transaction count: ``transactions`` dT * mA, ``spend_per_transaction`` dA * mT, and
    units and price as above without mC.

    Siblings sum exactly to their parent's contribution (the top pair to that set's spend diff)
    and an unchanged factor contributes 0. This is a level-by-level split, not the
    three-factor Shapley value of C x T x A.

    When a count is zero, the ratio inputs (S, T, A, U, P) change:

    - The ``_p1`` / ``_p2`` columns show a ratio over a zero count as NaN.
    - A period with nothing in it (0 / 0, e.g. a group that opened or closed) takes the
      other period's ratio, so that ratio's change is 0 and the group's whole change lands
      on ``customers`` (or ``transactions``). Nothing in either period, e.g. no unknown-customer
      rows, contributes 0.
    - A non-zero amount over a zero count (spend with no units, or, without unknown customers
      split out, spend with no customers because every customer_id is null) can't be split into
      those factors, so the contributions of that ratio, its sibling and every factor below them
      are NaN.

    Args:
        p1_df (pd.DataFrame): Period 1 aggregates, one row per group.
        p2_df (pd.DataFrame): Period 2 aggregates, one row per group.
        group_col (str | list[str] | None): Column(s) identifying the group in both frames.

    Returns:
        pd.DataFrame: One row per group, sorted by group with a missing group last: the
            ``group_col`` columns, then the period, difference, percent-difference, and
            contribution columns described above.

    Raises:
        TypeError: If ``p1_df`` or ``p2_df`` is not a DataFrame.
        ValueError: If a required or ``group_col`` column is missing, a frame repeats a group (or
            has several rows without ``group_col``), or only one frame has ``column.agg.unit_quantity``
            or the unknown-customer spend.

    Examples:
        >>> from openretailscience.analysis import calc_tree_kpis
        >>> p1 = pd.DataFrame({"customers": [120], "transactions": [300], "spend": [4500.0]})
        >>> p2 = pd.DataFrame({"customers": [130], "transactions": [340], "spend": [5200.0]})
        >>> calc_tree_kpis(p1, p2)[["spend_diff", "customers_contrib", "spend_per_customer_contrib"]]
           spend_diff  customers_contrib  spend_per_customer_contrib
        0       700.0              387.5                       312.5
    """
    cols = ColumnHelper()
    required_cols = [cols.agg.customer_id, cols.agg.transaction_id, cols.agg.unit_spend]
    for name, frame in (("p1_df", p1_df), ("p2_df", p2_df)):
        ensure_dataframe(frame, name)
        ensure_data_has_columns(frame, required_cols)
        keys = [] if group_col is None else ensure_columns(frame, group_col, "group_col")
        if len(keys) == 0 and len(frame) > 1:
            msg = f"{name} has {len(frame)} rows; pass group_col to pair rows across periods by group."
            raise ValueError(msg)
        if len(keys) > 0 and frame.duplicated(keys).any():
            msg = f"{name} repeats a group_col value; each group needs at most one row per period."
            raise ValueError(msg)

    has_quantity = _in_both(p1_df, p2_df, cols.agg.unit_qty)
    has_unknown = _in_both(p1_df, p2_df, cols.agg.unit_spend_unknown)
    variants = [get_option("column.suffix.unknown_customer"), get_option("column.suffix.total")] if has_unknown else []
    measures = [cols.agg.transaction_id, cols.agg.unit_spend, *([cols.agg.unit_qty] if has_quantity else [])]
    variant_measures = [_variant(measure, variant) for variant in variants for measure in measures]
    for frame in (p1_df, p2_df):
        ensure_data_has_columns(frame, variant_measures)
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")

    df = _pair_periods(p1_df, p2_df, keys, [cols.agg.customer_id, *measures, *variant_measures])

    customer_parts = {
        cols.calc.spend_per_cust: (cols.agg.unit_spend, cols.agg.customer_id),
        cols.calc.trans_per_cust: (cols.agg.transaction_id, cols.agg.customer_id),
    }
    _add_period_metrics(df, [cols.agg.customer_id, *measures], customer_parts | _visit_ratio_parts(None, has_quantity))
    for variant in variants:
        counts = [_variant(measure, variant) for measure in measures]
        _add_period_metrics(df, counts, _visit_ratio_parts(variant, has_quantity))

    cust_mid = (df[cols.agg.customer_id_p1] + df[cols.agg.customer_id_p2]) / 2
    s1, s2 = _contribution_ratio(df, cols.calc.spend_per_cust, *customer_parts[cols.calc.spend_per_cust])
    t1, t2 = _contribution_ratio(df, cols.calc.trans_per_cust, *customer_parts[cols.calc.trans_per_cust])
    df[cols.agg.customer_id_contrib] = df[cols.agg.customer_id_diff] * (s1 + s2) / 2
    df[cols.calc.spend_per_cust_contrib] = (s2 - s1) * cust_mid
    spend_per_trans_mid = _add_spend_per_trans_contribs(df, None, has_quantity, (t1 + t2) / 2 * cust_mid)
    df[cols.calc.trans_per_cust_contrib] = (t2 - t1) * spend_per_trans_mid * cust_mid

    diff, contrib = get_option("column.suffix.difference"), get_option("column.suffix.contribution")
    for variant in variants:
        trans = _variant(cols.agg.transaction_id, variant)
        trans_mid = (df[f"{trans}_{p1}"] + df[f"{trans}_{p2}"]) / 2
        spend_per_trans_mid = _add_spend_per_trans_contribs(df, variant, has_quantity, trans_mid)
        df[f"{trans}_{contrib}"] = df[f"{trans}_{diff}"] * spend_per_trans_mid

    return df[[*keys, *RevenueTree._get_final_col_order(include_quantity=has_quantity, include_unknown=has_unknown)]]


class RevenueTree:
    """Period-over-period revenue decomposition with per-factor contributions.

    ``df`` (pandas) has one row per group, or a single row when ungrouped. A group
    missing from one period has 0 customers, transactions and spend for it. Columns are the
    ``group_col`` columns (Categorical for a single one), then ``{metric}_p1`` / ``_p2`` /
    ``_diff`` / ``_pct_diff`` for ``customers``, ``transactions``, ``spend``,
    ``spend_per_customer``, ``transactions_per_customer`` and ``spend_per_transaction`` (plus
    ``units``, ``units_per_transaction`` and ``price_per_unit`` with quantity data), and
    ``{metric}_contrib`` for the tree's factors. ``_pct_diff`` is a fraction (0.1 = +10%). See
    `calc_tree_kpis` for the formulas. ``draw_tree()`` renders one row as a diagram.

    With the ``data.unknown_customer_value`` option set, the plain metrics cover identified
    customers only (null IDs count as unknown), and ``_unknown`` and ``_total`` sets of the
    transaction, spend and unit metrics follow.

    Examples:
        >>> from openretailscience.analysis import RevenueTree
        >>> tree = RevenueTree(df=transactions, period_col="period", p1_value="2025Q1", p2_value="2025Q2")
        >>> ax = tree.draw_tree(value_labels=("Q2 2025", "Q1 2025"))
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str | int | date,
        p2_value: str | int | date,
        group_col: str | list[str] | None = None,
    ) -> None:
        """Initialize the Revenue Tree and compute the KPI table immediately.

        If the input contains the option-derived ``column.unit_quantity`` column,
        units and price per unit metrics are added.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with the option-derived
                customer_id, transaction_id, unit_spend, and (optional) unit_quantity columns.
            period_col (str): Column holding a caller-assigned period label per row (e.g.
                ``"2024Q1"``), not a raw transaction date: labels are matched exactly and rows
                of any other period are dropped.
            p1_value (str | int | date): Label of the first (base) period.
            p2_value (str | int | date): Label of the second period.
            group_col (str | list[str] | None): Column(s) to break the analysis down by.

        Raises:
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table, or if the
                data.unknown_customer_value option can't be compared with customer_id.
            ValueError: If the required columns are not present in the data, if
                ``p1_value`` equals ``p2_value``, or if either matches no rows of ``period_col``.
            ibis.common.exceptions.IbisTypeError: If ``p1_value`` or ``p2_value`` can't be compared
                with ``period_col`` (e.g. a str label on an int column).
        """
        cols = ColumnHelper()

        df = ensure_ibis_table(df)
        if p1_value == p2_value:
            msg = f"p1_value and p2_value must be different, got {p1_value!r} for both."
            raise ValueError(msg)
        if group_col is not None:
            group_col = ensure_columns(df, group_col, "group_col")

        required_cols = [cols.customer_id, cols.transaction_id, cols.unit_spend, period_col]
        if cols.unit_qty in df.columns:
            required_cols.append(cols.unit_qty)
        # group_col is already validated above; only the function's hard-coded requirements remain.
        ensure_data_has_columns(df, required_cols)

        self.df = calc_tree_kpis(*self._agg_data(df, period_col, p1_value, p2_value, group_col), group_col=group_col)

    @staticmethod
    def _agg_data(
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str | int | date,
        p2_value: str | int | date,
        group_col: list[str] | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Aggregate data by period and optional grouping columns in the Ibis engine.

        Args:
            df (pd.DataFrame | ibis.Table): Input data.
            period_col (str): Column name for the period.
            p1_value (str | int | date): Value representing period 1.
            p2_value (str | int | date): Value representing period 2.
            group_col (list[str] | None): List of column names to group by.

        Returns:
            tuple[pd.DataFrame, pd.DataFrame]: The period 1 and period 2 aggregates, one row per
                group present in that period, with the ``group_col`` columns first. A single group
                column is Categorical, with the same categories in both periods.
        """
        cols = ColumnHelper()

        table = ensure_ibis_table(df)
        p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
        # Relabel in the backend, whose comparison the filter used: pandas may read the values back as another
        # type (a date column returns Timestamps, which never equal a date)
        df = table.filter(table[period_col].isin([p1_value, p2_value])).mutate(
            **{period_col: (ibis._[period_col] == p1_value).ifelse(p1, p2)},
        )

        unknown_customer_value = get_option("data.unknown_customer_value")
        unknown = None if unknown_customer_value is None else unknown_customer_flag(df, unknown_customer_value)
        identified = None if unknown is None else ~unknown
        measures = {
            cols.agg.transaction_id: df[cols.transaction_id].nunique,
            cols.agg.unit_spend: df[cols.unit_spend].sum,
        }
        if cols.unit_qty in df.columns:
            measures[cols.agg.unit_qty] = df[cols.unit_qty].sum
        # A sum over no rows (e.g. a group with no unknown rows) is NULL, which would turn integer units into floats
        aggs = {cols.agg.customer_id: df[cols.customer_id].nunique(where=identified)} | {
            name: agg(where=identified).coalesce(0) for name, agg in measures.items()
        }
        if unknown is not None:
            suffix = get_option("column.suffix.unknown_customer")
            aggs |= {f"{name}_{suffix}": agg(where=unknown).coalesce(0) for name, agg in measures.items()}
            # Not identified + unknown: a transaction with both kinds of line counts once
            suffix = get_option("column.suffix.total")
            aggs |= {f"{name}_{suffix}": agg().coalesce(0) for name, agg in measures.items()}

        group_by_cols = [*group_col, period_col] if group_col else [period_col]
        df = pd.DataFrame(df.group_by(group_by_cols).aggregate(**aggs).execute())
        p1_df = df[df[period_col] == p1].drop(columns=[period_col])
        p2_df = df[df[period_col] == p2].drop(columns=[period_col])
        for name, value, period_df in (("p1_value", p1_value, p1_df), ("p2_value", p2_value, p2_df)):
            if len(period_df) == 0:
                # Sorted and capped in the backend: a mistaken raw timestamp column has millions of distinct values.
                # Nulls sort last, so one can only take a slot when there are too few values to need it.
                present = (
                    table.select(period_col)
                    .distinct()
                    .order_by(period_col)
                    .limit(MAX_PERIODS_SHOWN + 1)
                    .execute()[period_col]
                    .dropna()
                    .tolist()
                )
                shown = f"{present[:MAX_PERIODS_SHOWN]}" + (" ..." if len(present) > MAX_PERIODS_SHOWN else "")
                msg = f"{name} {value!r} does not appear in period column {period_col!r}, which has {shown}."
                raise ValueError(msg)

        p1_df, p2_df = p1_df.reset_index(drop=True), p2_df.reset_index(drop=True)
        if group_col is not None and len(group_col) == 1:
            # Categories can't be null, so a missing group value stays NaN. union() leaves equal inputs unsorted.
            col = group_col[0]
            categories = pd.Index(p1_df[col]).union(pd.Index(p2_df[col])).dropna().sort_values().rename(None)
            p1_df[col] = pd.Categorical(p1_df[col], categories=categories)
            p2_df[col] = pd.Categorical(p2_df[col], categories=categories)
        return p1_df, p2_df

    @staticmethod
    def _get_final_col_order(include_quantity: bool, include_unknown: bool) -> list[str]:
        """Get the final column order for the RevenueTree DataFrame.

        Args:
            include_quantity (bool): Whether to include quantity-related columns.
            include_unknown (bool): Whether to append the unknown, then the total, transaction trees.

        Returns:
            list[str]: Ordered list of column names for the final DataFrame.
        """
        cols = ColumnHelper()
        # (metric, whether it has a contribution column), top of the tree first
        quantity = [(cols.agg.unit_qty, False), (cols.calc.units_per_trans, True), (cols.calc.price_per_unit, True)]
        quantity = quantity if include_quantity else []
        metrics = [
            (cols.agg.customer_id, True),
            (cols.agg.transaction_id, False),
            (cols.agg.unit_spend, False),
            (cols.calc.spend_per_cust, True),
            (cols.calc.trans_per_cust, True),
            (cols.calc.spend_per_trans, True),
            *quantity,
        ]
        visit_tree = [
            (cols.agg.transaction_id, True),
            (cols.agg.unit_spend, False),
            (cols.calc.spend_per_trans, True),
            *quantity,
        ]
        if include_unknown:
            for variant in (get_option("column.suffix.unknown_customer"), get_option("column.suffix.total")):
                metrics += [(_variant(metric, variant), has_contrib) for metric, has_contrib in visit_tree]

        suffix_names = ("period_1", "period_2", "difference", "percent_difference")
        suffixes = [get_option(f"column.suffix.{name}") for name in suffix_names]
        contrib = get_option("column.suffix.contribution")
        return [
            f"{metric}_{suffix}"
            for metric, has_contrib in metrics
            for suffix in (*suffixes, *([contrib] if has_contrib else []))
        ]

    def draw_tree(
        self,
        row_index: int = 0,
        value_labels: tuple[str, str] | None = None,
        unit_spend_label: str = "Revenue",
        customer_id_label: str = "Customers",
        spend_per_customer_label: str = "Spend / Customer",
        transactions_per_customer_label: str = "Visits / Customer",
        spend_per_transaction_label: str = "Spend / Visit",
        units_per_transaction_label: str = "Units / Visit",
        price_per_unit_label: str = "Price / Unit",
        identified_spend_label: str = "Identified Revenue",
        unknown_spend_label: str = "Unidentified Revenue",
    ) -> Axes:
        """Draw the revenue tree diagram for one row of ``df``.

        The Units / Visit and Price / Unit nodes render only when the input contained
        the unit_quantity column. With the ``data.unknown_customer_value`` option set when the tree
        was built, the Revenue node is total revenue, split into Unidentified Revenue and Identified
        Revenue, which the rest of the tree breaks down.

        Args:
            row_index (int): Row of ``df`` to draw; use it to select a group when grouped.
            value_labels (tuple[str, str] | None): (current, previous) = (p2_value, p1_value)
                labels — easy to reverse. Defaults to ("Current Period", "Previous Period").
            unit_spend_label (str): Header for the Revenue node.
            customer_id_label (str): Header for the Customers node.
            spend_per_customer_label (str): Header for the Spend / Customer node.
            transactions_per_customer_label (str): Header for the Visits / Customer node.
            spend_per_transaction_label (str): Header for the Spend / Visit node.
            units_per_transaction_label (str): Header for the Units / Visit node.
            price_per_unit_label (str): Header for the Price / Unit node.
            identified_spend_label (str): Header for the Identified Revenue node.
            unknown_spend_label (str): Header for the Unidentified Revenue node.

        Returns:
            matplotlib.axes.Axes: The matplotlib axes containing the tree visualization.

        Raises:
            IndexError: If ``row_index`` is out of bounds for ``df``.

        """
        cols = ColumnHelper()
        graph_data = self.df.iloc[row_index].to_dict()
        period_labels = value_labels or ("Current Period", "Previous Period")
        has_quantity = cols.agg.unit_qty_p1 in graph_data
        has_unknown = f"{cols.agg.unit_spend_unknown}_{get_option('column.suffix.period_1')}" in graph_data
        # The split puts total revenue on top, so the identified tree moves down a row and right a column
        shift = 1 if has_unknown else 0

        tree_structure = {
            "spend": _tree_node(
                graph_data,
                cols.agg.unit_spend,
                identified_spend_label if has_unknown else unit_spend_label,
                period_labels,
                position=(1 + shift, shift),
                children=["customers", "spend_per_customer"],
            ),
            "customers": _tree_node(
                graph_data, cols.agg.customer_id, customer_id_label, period_labels, position=(shift, 1 + shift)
            ),
            "spend_per_customer": _tree_node(
                graph_data,
                cols.calc.spend_per_cust,
                spend_per_customer_label,
                period_labels,
                position=(2 + shift, 1 + shift),
                children=["visits_per_customer", "spend_per_visit"],
            ),
            "visits_per_customer": _tree_node(
                graph_data,
                cols.calc.trans_per_cust,
                transactions_per_customer_label,
                period_labels,
                position=(1 + shift, 2 + shift),
            ),
            "spend_per_visit": _tree_node(
                graph_data,
                cols.calc.spend_per_trans,
                spend_per_transaction_label,
                period_labels,
                position=(3 + shift, 2 + shift),
                children=["units_per_visit", "price_per_unit"] if has_quantity else [],
            ),
        }
        if has_quantity:
            tree_structure["units_per_visit"] = _tree_node(
                graph_data,
                cols.calc.units_per_trans,
                units_per_transaction_label,
                period_labels,
                position=(2 + shift, 3 + shift),
            )
            tree_structure["price_per_unit"] = _tree_node(
                graph_data,
                cols.calc.price_per_unit,
                price_per_unit_label,
                period_labels,
                position=(4 + shift, 3 + shift),
            )
        if has_unknown:
            # Their contributions to total revenue equal their own diffs, so neither node repeats it
            tree_structure["total_spend"] = _tree_node(
                graph_data,
                cols.agg.unit_spend_total,
                unit_spend_label,
                period_labels,
                position=(1, 0),
                children=["unknown_spend", "spend"],
            )
            tree_structure["unknown_spend"] = _tree_node(
                graph_data, cols.agg.unit_spend_unknown, unknown_spend_label, period_labels, position=(0, 1)
            )

        grid = TreeGrid(
            tree_structure=tree_structure,
            num_rows=(4 if has_quantity else 3) + shift,
            num_cols=(5 if has_quantity else 4) + shift,
            node_class=DetailedTreeNode,
        )

        return grid.render()


def _tree_node(
    graph_data: dict[str, float],
    metric: str,
    header: str,
    period_labels: tuple[str, str],
    position: tuple[int, int],
    children: list[str] | None = None,
) -> dict[str, str | float | tuple[int, int] | list[str]]:
    """Build one ``draw_tree`` node from a row of ``RevenueTree.df``.

    Args:
        graph_data (dict[str, float]): The row, keyed by column name.
        metric (str): The node's unsuffixed metric column; its contribution is shown when the row has one.
        header (str): The node's header.
        period_labels (tuple[str, str]): The (current, previous) period labels.
        position (tuple[int, int]): The node's (column, row) in the grid, row 0 at the top.
        children (list[str] | None, optional): The ids of the node's children. Defaults to None (a leaf).

    Returns:
        dict[str, str | float | tuple[int, int] | list[str]]: The node's entry in the ``TreeGrid`` structure.
    """
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
    diff, pct_diff = get_option("column.suffix.difference"), get_option("column.suffix.percent_difference")
    contrib = f"{metric}_{get_option('column.suffix.contribution')}"
    node = {
        "header": header,
        "percent": graph_data[f"{metric}_{pct_diff}"] * 100,
        "current_period": gu.format_shorthand(graph_data[f"{metric}_{p2}"], decimals=2),
        "previous_period": gu.format_shorthand(graph_data[f"{metric}_{p1}"], decimals=2),
        "diff": gu.format_shorthand(graph_data[f"{metric}_{diff}"], decimals=2),
        "current_label": period_labels[0],
        "previous_label": period_labels[1],
        "position": position,
        "children": [] if children is None else children,
    }
    if contrib in graph_data:
        node["contribution"] = gu.format_shorthand(graph_data[contrib], decimals=2)
    return node
