"""Decompose period-over-period revenue change into per-factor contributions.

Revenue = customers x spend per customer = customers x transactions per customer x
spend per transaction, extended with units per transaction x price per unit when
``unit_quantity`` is present (those conditional columns define the ``.df`` output).
Provides ``RevenueTree`` (with ``.draw_tree()``) and ``calc_tree_kpis``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import ibis
import numpy as np
import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots.styles import graph_utils as gu
from openretailscience.plots.tree_diagram import DetailedTreeNode, TreeGrid

if TYPE_CHECKING:
    from datetime import date

    from matplotlib.axes import Axes

MAX_PERIODS_SHOWN = 10  # period values listed in a "p1_value/p2_value not found" error


def _contribution_ratio(df: pd.DataFrame, ratio: str, numerator: str, denominator: str) -> tuple[pd.Series, pd.Series]:
    """Return a ratio's (p1, p2) values for the contribution split.

    A 0 / 0 period takes the other period's value; a non-zero amount over a zero count stays NaN.

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
    ratio_p1, ratio_p2 = df[f"{ratio}_{p1}"], df[f"{ratio}_{p2}"]
    return ratio_p1.mask(empty_p1, ratio_p2), ratio_p2.mask(empty_p2, ratio_p1)


def calc_tree_kpis(p1_df: pd.DataFrame, p2_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate the revenue-tree KPIs from per-period aggregates.

    Each frame has one row per group with the option-derived columns ``column.agg.customer_id``,
    ``column.agg.transaction_id``, ``column.agg.unit_spend`` and, optionally,
    ``column.agg.unit_quantity``. Rows are paired across periods by index label; a group
    missing from one frame gets 0 counts for that period. For an ungrouped comparison, pass
    one row per period with the same label (e.g. both indexed ``0``).

    Output columns are ``{metric}_p1`` / ``_p2`` / ``_diff`` / ``_pct_diff`` (suffixes
    from the ``column.suffix.*`` options) plus ``{metric}_contrib`` contribution
    columns (quantity/price columns present only when units were) and the elasticity
    columns (``price_elasticity`` only with units). ``pct_diff = diff / p1``, with p1 = 0
    giving NaN. The elasticity columns use ARC (midpoint) percentage changes: price =
    %delta units / %delta price per unit, frequency = %delta transactions per customer /
    %delta spend per customer.

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

    Siblings sum exactly to their parent's contribution (the top pair to ``spend_diff``)
    and an unchanged factor contributes 0. This is a level-by-level split, not the
    three-factor Shapley value of C x T x A.

    When a count is zero, the ratio inputs (S, T, A, U, P) change:

    - The ``_p1`` / ``_p2`` columns show a ratio over a zero count as NaN.
    - A period with nothing in it (0 / 0, e.g. a group that opened or closed) takes the
      other period's ratio, so that ratio's change is 0 and the group's whole change lands
      on ``customers``.
    - A non-zero amount over a zero count (spend with no units, or with no customers because
      every customer_id is null) can't be split into those factors, so the contributions of
      that ratio, its sibling and every factor below them are NaN.

    Args:
        p1_df (pd.DataFrame): Period 1 aggregates, one row per group.
        p2_df (pd.DataFrame): Period 2 aggregates, one row per group.

    Returns:
        pd.DataFrame: One row per group, sorted by group, with the period, difference,
            percent-difference, and contribution columns described above.

    Raises:
        TypeError: If ``p1_df`` or ``p2_df`` is not a DataFrame.
        ValueError: If a required column is missing, a frame repeats an index label, or only one
            frame has ``column.agg.unit_quantity``.

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
        if not isinstance(frame, pd.DataFrame):
            msg = f"{name} must be a pandas DataFrame, got {type(frame).__name__}."
            raise TypeError(msg)
        ensure_data_has_columns(frame, required_cols)
        if not frame.index.is_unique:
            msg = f"{name} has repeated index labels; each group needs at most one row per period."
            raise ValueError(msg)

    ratio_parts = {
        cols.calc.spend_per_cust: (cols.agg.unit_spend, cols.agg.customer_id),
        cols.calc.trans_per_cust: (cols.agg.transaction_id, cols.agg.customer_id),
        cols.calc.spend_per_trans: (cols.agg.unit_spend, cols.agg.transaction_id),
    }
    metric_cols = [*required_cols, *ratio_parts]
    has_quantity = cols.agg.unit_qty in p1_df.columns
    if has_quantity != (cols.agg.unit_qty in p2_df.columns):
        msg = f"p1_df and p2_df must both have {cols.agg.unit_qty!r} or neither."
        raise ValueError(msg)
    if has_quantity:
        required_cols.append(cols.agg.unit_qty)
        ratio_parts[cols.calc.units_per_trans] = (cols.agg.unit_qty, cols.agg.transaction_id)
        ratio_parts[cols.calc.price_per_unit] = (cols.agg.unit_spend, cols.agg.unit_qty)
        metric_cols += [cols.agg.unit_qty, cols.calc.units_per_trans, cols.calc.price_per_unit]
    p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")

    # A group missing from a period gets 0 counts; sort=True keeps P2-only groups in group order, not appended last
    df = pd.concat(
        [p1_df[required_cols].add_suffix(f"_{p1}"), p2_df[required_cols].add_suffix(f"_{p2}")], axis=1, sort=True
    ).fillna(0)

    for ratio, (numerator, denominator) in ratio_parts.items():
        for period in (p1, p2):
            df[f"{ratio}_{period}"] = df[f"{numerator}_{period}"] / df[f"{denominator}_{period}"].replace(0, np.nan)

    diff, pct_diff = get_option("column.suffix.difference"), get_option("column.suffix.percent_difference")
    for col in metric_cols:
        df[f"{col}_{diff}"] = df[f"{col}_{p2}"] - df[f"{col}_{p1}"]
        df[f"{col}_{pct_diff}"] = df[f"{col}_{diff}"] / df[f"{col}_{p1}"].replace(0, np.nan)

    cust_mid = (df[cols.agg.customer_id_p1] + df[cols.agg.customer_id_p2]) / 2
    trans_per_cust_mid = (df[cols.calc.trans_per_cust_p1] + df[cols.calc.trans_per_cust_p2]) / 2
    spend_per_cust_mid = (df[cols.calc.spend_per_cust_p1] + df[cols.calc.spend_per_cust_p2]) / 2

    freq_pct_change = df[cols.calc.trans_per_cust_diff] / trans_per_cust_mid.replace(0, np.nan)
    spend_pct_change = df[cols.calc.spend_per_cust_diff] / spend_per_cust_mid.replace(0, np.nan)
    df[cols.calc.frequency_elasticity] = freq_pct_change / spend_pct_change.replace(0, np.nan)

    s1, s2 = _contribution_ratio(df, cols.calc.spend_per_cust, *ratio_parts[cols.calc.spend_per_cust])
    t1, t2 = _contribution_ratio(df, cols.calc.trans_per_cust, *ratio_parts[cols.calc.trans_per_cust])
    a1, a2 = _contribution_ratio(df, cols.calc.spend_per_trans, *ratio_parts[cols.calc.spend_per_trans])
    df[cols.agg.customer_id_contrib] = df[cols.agg.customer_id_diff] * (s1 + s2) / 2
    df[cols.calc.spend_per_cust_contrib] = (s2 - s1) * cust_mid
    df[cols.calc.trans_per_cust_contrib] = (t2 - t1) * (a1 + a2) / 2 * cust_mid
    df[cols.calc.spend_per_trans_contrib] = (a2 - a1) * (t1 + t2) / 2 * cust_mid

    if has_quantity:
        qty_mid = (df[cols.agg.unit_qty_p1] + df[cols.agg.unit_qty_p2]) / 2
        price_per_unit_mid = (df[cols.calc.price_per_unit_p1] + df[cols.calc.price_per_unit_p2]) / 2

        qty_pct_change = df[cols.agg.unit_qty_diff] / qty_mid.replace(0, np.nan)
        price_pct_change = df[cols.calc.price_per_unit_diff] / price_per_unit_mid.replace(0, np.nan)
        df[cols.calc.price_elasticity] = qty_pct_change / price_pct_change.replace(0, np.nan)

        u1, u2 = _contribution_ratio(df, cols.calc.units_per_trans, *ratio_parts[cols.calc.units_per_trans])
        pr1, pr2 = _contribution_ratio(df, cols.calc.price_per_unit, *ratio_parts[cols.calc.price_per_unit])
        trans_per_cust_weight = (t1 + t2) / 2 * cust_mid
        df[cols.calc.units_per_trans_contrib] = (u2 - u1) * (pr1 + pr2) / 2 * trans_per_cust_weight
        df[cols.calc.price_per_unit_contrib] = (pr2 - pr1) * (u1 + u2) / 2 * trans_per_cust_weight

    return df[RevenueTree._get_final_col_order(include_quantity=has_quantity)]


class RevenueTree:
    """Period-over-period revenue decomposition with per-factor contributions.

    ``df`` (pandas) has one row per group: a single row labelled 0 when ungrouped, a
    ``CategoricalIndex`` for one ``group_col`` and a ``MultiIndex`` for several. A group
    missing from one period has 0 customers, transactions and spend for it. Columns are
    ``{metric}_p1`` / ``_p2`` / ``_diff`` / ``_pct_diff`` for ``customers``, ``transactions``,
    ``spend``, ``spend_per_customer``, ``transactions_per_customer`` and
    ``spend_per_transaction`` (plus ``units``, ``units_per_transaction`` and
    ``price_per_unit`` with quantity data), ``{metric}_contrib`` for the tree's factors, and
    ``frequency_elasticity`` / ``price_elasticity``. ``_pct_diff`` is a fraction (0.1 = +10%).
    See `calc_tree_kpis` for the formulas. ``draw_tree()`` renders one row as a diagram.

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
        units/price/price-elasticity metrics are added.

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
            ValueError: If the required columns are not present in the data, if
                ``p1_value`` equals ``p2_value``, or if either matches no rows of ``period_col``.
            ibis.common.exceptions.IbisTypeError: If ``p1_value`` or ``p2_value`` can't be compared
                with ``period_col`` (e.g. a str label on an int column).
        """
        cols = ColumnHelper()

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

        self.df = calc_tree_kpis(*self._agg_data(df, period_col, p1_value, p2_value, group_col))

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
                group present in that period, sorted by group. Ungrouped, each is one row labelled
                0; one group column gives a ``CategoricalIndex`` with the same categories in both
                periods, several give a ``MultiIndex``.
        """
        cols = ColumnHelper()

        table = ensure_ibis_table(df)
        p1, p2 = get_option("column.suffix.period_1"), get_option("column.suffix.period_2")
        # Relabel in the backend, whose comparison the filter used: pandas may read the values back as another
        # type (a date column returns Timestamps, which never equal a date)
        df = table.filter(table[period_col].isin([p1_value, p2_value])).mutate(
            **{period_col: (ibis._[period_col] == p1_value).ifelse(p1, p2)},
        )

        aggs = {
            cols.agg.customer_id: df[cols.customer_id].nunique(),
            cols.agg.transaction_id: df[cols.transaction_id].nunique(),
            cols.agg.unit_spend: df[cols.unit_spend].sum(),
        }
        if cols.unit_qty in df.columns:
            aggs[cols.agg.unit_qty] = df[cols.unit_qty].sum()

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

        if group_col is None:
            return p1_df.reset_index(drop=True), p2_df.reset_index(drop=True)

        p1_df, p2_df = p1_df.set_index(group_col).sort_index(), p2_df.set_index(group_col).sort_index()
        if len(group_col) == 1:
            # Categories can't be null; a missing group value stays a NaN label
            categories = p1_df.index.union(p2_df.index).dropna().rename(None)
            p1_df.index = pd.CategoricalIndex(p1_df.index, categories=categories, name=group_col[0])
            p2_df.index = pd.CategoricalIndex(p2_df.index, categories=categories, name=group_col[0])
        return p1_df, p2_df

    @staticmethod
    def _get_final_col_order(include_quantity: bool) -> list[str]:
        """Get the final column order for the RevenueTree DataFrame.

        Args:
            include_quantity: Whether to include quantity-related columns.

        Returns:
            list[str]: Ordered list of column names for the final DataFrame.

        """
        cols = ColumnHelper()
        col_order = [
            # Customers
            cols.agg.customer_id_p1,
            cols.agg.customer_id_p2,
            cols.agg.customer_id_diff,
            cols.agg.customer_id_pct_diff,
            cols.agg.customer_id_contrib,
            # Transactions
            cols.agg.transaction_id_p1,
            cols.agg.transaction_id_p2,
            cols.agg.transaction_id_diff,
            cols.agg.transaction_id_pct_diff,
            # Unit Spend
            cols.agg.unit_spend_p1,
            cols.agg.unit_spend_p2,
            cols.agg.unit_spend_diff,
            cols.agg.unit_spend_pct_diff,
            # Spend / Customer
            cols.calc.spend_per_cust_p1,
            cols.calc.spend_per_cust_p2,
            cols.calc.spend_per_cust_diff,
            cols.calc.spend_per_cust_pct_diff,
            cols.calc.spend_per_cust_contrib,
            # Transactions / Customer
            cols.calc.trans_per_cust_p1,
            cols.calc.trans_per_cust_p2,
            cols.calc.trans_per_cust_diff,
            cols.calc.trans_per_cust_pct_diff,
            cols.calc.trans_per_cust_contrib,
            # Spend / Transaction
            cols.calc.spend_per_trans_p1,
            cols.calc.spend_per_trans_p2,
            cols.calc.spend_per_trans_diff,
            cols.calc.spend_per_trans_pct_diff,
            cols.calc.spend_per_trans_contrib,
            # Elasticity
            cols.calc.frequency_elasticity,
        ]

        if include_quantity:
            col_order.extend(
                [
                    # Unit Quantity
                    cols.agg.unit_qty_p1,
                    cols.agg.unit_qty_p2,
                    cols.agg.unit_qty_diff,
                    cols.agg.unit_qty_pct_diff,
                    # Quantity / Transaction
                    cols.calc.units_per_trans_p1,
                    cols.calc.units_per_trans_p2,
                    cols.calc.units_per_trans_diff,
                    cols.calc.units_per_trans_pct_diff,
                    cols.calc.units_per_trans_contrib,
                    # Price / Unit
                    cols.calc.price_per_unit_p1,
                    cols.calc.price_per_unit_p2,
                    cols.calc.price_per_unit_diff,
                    cols.calc.price_per_unit_pct_diff,
                    cols.calc.price_per_unit_contrib,
                    # Price Elasticity
                    cols.calc.price_elasticity,
                ],
            )

        return col_order

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
    ) -> Axes:
        """Draw the revenue tree diagram for one row of ``df``.

        The Units / Visit and Price / Unit nodes render only when the input contained
        the unit_quantity column.

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

        Returns:
            matplotlib.axes.Axes: The matplotlib axes containing the tree visualization.

        Raises:
            IndexError: If ``row_index`` is out of bounds for ``df``.

        """
        cols = ColumnHelper()
        graph_data = self.df.iloc[row_index].to_dict()

        # Set period labels
        current_label, previous_label = value_labels or ("Current Period", "Previous Period")

        # Build tree structure - always include base 5 nodes
        tree_structure = {
            "revenue": {
                "header": unit_spend_label,
                "percent": graph_data[cols.agg.unit_spend_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.agg.unit_spend_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.agg.unit_spend_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.agg.unit_spend_diff], decimals=2),
                # Contribution omitted for root node (would be same as diff)
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (1, 0),
                "children": ["customers", "spend_per_customer"],
            },
            "customers": {
                "header": customer_id_label,
                "percent": graph_data[cols.agg.customer_id_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.agg.customer_id_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.agg.customer_id_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.agg.customer_id_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.agg.customer_id_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (0, 1),
                "children": [],
            },
            "spend_per_customer": {
                "header": spend_per_customer_label,
                "percent": graph_data[cols.calc.spend_per_cust_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.calc.spend_per_cust_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.calc.spend_per_cust_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.calc.spend_per_cust_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.calc.spend_per_cust_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (2, 1),
                "children": ["visits_per_customer", "spend_per_visit"],
            },
            "visits_per_customer": {
                "header": transactions_per_customer_label,
                "percent": graph_data[cols.calc.trans_per_cust_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.calc.trans_per_cust_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.calc.trans_per_cust_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.calc.trans_per_cust_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.calc.trans_per_cust_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (1, 2),
                "children": [],
            },
            "spend_per_visit": {
                "header": spend_per_transaction_label,
                "percent": graph_data[cols.calc.spend_per_trans_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.calc.spend_per_trans_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.calc.spend_per_trans_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.calc.spend_per_trans_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.calc.spend_per_trans_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (3, 2),
                "children": [],
            },
        }

        grid_rows = 3
        grid_cols = 4

        # Add quantity-related nodes if data is available
        has_quantity = cols.agg.unit_qty_p1 in graph_data
        if has_quantity:
            grid_rows = 4
            grid_cols = 5
            tree_structure["spend_per_visit"]["children"] = ["units_per_visit", "price_per_unit"]
            tree_structure["units_per_visit"] = {
                "header": units_per_transaction_label,
                "percent": graph_data[cols.calc.units_per_trans_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.calc.units_per_trans_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.calc.units_per_trans_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.calc.units_per_trans_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.calc.units_per_trans_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (2, 3),
                "children": [],
            }
            tree_structure["price_per_unit"] = {
                "header": price_per_unit_label,
                "percent": graph_data[cols.calc.price_per_unit_pct_diff] * 100,
                "current_period": gu.format_shorthand(graph_data[cols.calc.price_per_unit_p2], decimals=2),
                "previous_period": gu.format_shorthand(graph_data[cols.calc.price_per_unit_p1], decimals=2),
                "diff": gu.format_shorthand(graph_data[cols.calc.price_per_unit_diff], decimals=2),
                "contribution": gu.format_shorthand(graph_data[cols.calc.price_per_unit_contrib], decimals=2),
                "current_label": current_label,
                "previous_label": previous_label,
                "position": (4, 3),
                "children": [],
            }

        # Create and render the tree grid
        grid = TreeGrid(
            tree_structure=tree_structure,
            num_rows=grid_rows,
            num_cols=grid_cols,
            node_class=DetailedTreeNode,
        )

        return grid.render()
