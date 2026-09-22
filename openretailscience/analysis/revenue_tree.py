"""Decompose period-over-period revenue change into per-factor contributions.

Revenue = customers x spend per customer = customers x transactions per customer x
spend per transaction, extended with units per transaction x price per unit when
``unit_quantity`` is present (those conditional columns define the ``.df`` output).
Provides ``RevenueTree`` (with ``.draw_tree()``) and ``calc_tree_kpis``.
"""

import ibis
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper, get_option
from openretailscience.plots.styles import graph_utils as gu
from openretailscience.plots.tree_diagram import DetailedTreeNode, TreeGrid


def calc_tree_kpis(
    df: pd.DataFrame,
    p1_index: list[bool] | pd.Series,
    p2_index: list[bool] | pd.Series,
) -> pd.DataFrame:
    """Calculate the revenue-tree KPIs from a pre-aggregated frame.

    Input contract: the frame built by ``RevenueTree._agg_data`` — one row per
    (group, period) with the option-derived columns ``column.agg.customer_id``,
    ``column.agg.transaction_id``, ``column.agg.unit_spend``, and optionally
    ``column.agg.unit_quantity`` (only when units were in the input). ``p1_index``
    and ``p2_index`` are boolean row selectors for the two period rows.

    Output columns are ``{metric}_p1`` / ``_p2`` / ``_diff`` / ``_pct_diff`` (suffixes
    from the ``column.suffix.*`` options) plus ``{metric}_contrib`` contribution
    columns (quantity/price columns present only when units were) and the two
    elasticity columns. ``pct_diff = diff / p1``, with p1 = 0 giving NaN. The
    elasticity columns use ARC (midpoint) percentage changes: price = %delta units /
    %delta price per unit, frequency = %delta transactions per customer / %delta
    spend per customer.

    Args:
        df (pd.DataFrame): Pre-aggregated frame as described above.
        p1_index (list[bool] | pd.Series): Boolean row selector for the period 1 rows.
        p2_index (list[bool] | pd.Series): Boolean row selector for the period 2 rows.

    Returns:
        pd.DataFrame: One row per (group) with the period, difference, percent-difference,
            and contribution columns described above.
    """
    cols = ColumnHelper()
    required_cols = [cols.agg.customer_id, cols.agg.transaction_id, cols.agg.unit_spend]

    if cols.agg.unit_qty in df.columns:
        required_cols.append(cols.agg.unit_qty)

    df = df[required_cols].copy()
    df_cols = df.columns

    if cols.agg.unit_qty in df_cols:
        df[cols.calc.units_per_trans] = df[cols.agg.unit_qty] / df[cols.agg.transaction_id]
        df[cols.calc.price_per_unit] = df[cols.agg.unit_spend] / df[cols.agg.unit_qty]

    df[cols.calc.spend_per_cust] = df[cols.agg.unit_spend] / df[cols.agg.customer_id]
    df[cols.calc.spend_per_trans] = df[cols.agg.unit_spend] / df[cols.agg.transaction_id]
    df[cols.calc.trans_per_cust] = df[cols.agg.transaction_id] / df[cols.agg.customer_id]

    p1_df = df[p1_index]
    p1_df.columns = [col + "_" + get_option("column.suffix.period_1") for col in p1_df.columns]
    p2_df = df[p2_index]
    p2_df.columns = [col + "_" + get_option("column.suffix.period_2") for col in p2_df.columns]

    # When df only contains two periods than the indexes should be dropped for proper concatenation
    period_count = 2
    if len(df.index) == period_count:
        p1_df = p1_df.reset_index(drop=True)
        p2_df = p2_df.reset_index(drop=True)

    # fillna with 0 to handle cases when one time period isn't present
    df = pd.concat([p1_df, p2_df], axis=1).fillna(0)

    for col in [
        cols.agg.customer_id,
        cols.agg.transaction_id,
        cols.agg.unit_spend,
        cols.calc.spend_per_trans,
        cols.calc.trans_per_cust,
        cols.calc.spend_per_cust,
    ]:
        # Difference calculations
        df[col + "_" + get_option("column.suffix.difference")] = (
            df[col + "_" + get_option("column.suffix.period_2")] - df[col + "_" + get_option("column.suffix.period_1")]
        )

        # Percentage change calculations
        p1_col = df[col + "_" + get_option("column.suffix.period_1")]
        df[col + "_" + get_option("column.suffix.percent_difference")] = df[
            col + "_" + get_option("column.suffix.difference")
        ] / p1_col.replace(0, np.nan)

    # Calculate price elasticity
    if cols.agg.unit_qty in df_cols:
        qty_avg = ((df[cols.agg.unit_qty_p2] + df[cols.agg.unit_qty_p1]) / 2).replace(0, np.nan)
        qty_pct_change = (df[cols.agg.unit_qty_p2] - df[cols.agg.unit_qty_p1]) / qty_avg

        price_avg = ((df[cols.calc.price_per_unit_p2] + df[cols.calc.price_per_unit_p1]) / 2).replace(0, np.nan)
        price_pct_change = (df[cols.calc.price_per_unit_p2] - df[cols.calc.price_per_unit_p1]) / price_avg

        df[cols.calc.price_elasticity] = qty_pct_change / price_pct_change.replace(0, np.nan)

    # Calculate frequency elasticity
    freq_avg = ((df[cols.calc.trans_per_cust_p2] + df[cols.calc.trans_per_cust_p1]) / 2).replace(0, np.nan)
    freq_pct_change = (df[cols.calc.trans_per_cust_p2] - df[cols.calc.trans_per_cust_p1]) / freq_avg

    spend_avg = ((df[cols.calc.spend_per_cust_p2] + df[cols.calc.spend_per_cust_p1]) / 2).replace(0, np.nan)
    spend_pct_change = (df[cols.calc.spend_per_cust_p2] - df[cols.calc.spend_per_cust_p1]) / spend_avg

    df[cols.calc.frequency_elasticity] = freq_pct_change / spend_pct_change.replace(0, np.nan)

    # Contribution calculations
    df[cols.agg.customer_id_contrib] = (
        df[cols.agg.unit_spend_p2]
        - (df[cols.agg.customer_id_p1] * df[cols.calc.spend_per_cust_p2])
        - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 2)
    )
    df[cols.calc.spend_per_cust_contrib] = (
        df[cols.agg.unit_spend_p2]
        - (df[cols.calc.spend_per_cust_p1] * df[cols.agg.customer_id_p2])
        - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 2)
    )

    df[cols.calc.trans_per_cust_contrib] = (
        (
            df[cols.calc.spend_per_cust_p2]
            - (df[cols.calc.trans_per_cust_p1] * df[cols.calc.spend_per_trans_p2])
            - ((df[cols.calc.trans_per_cust_diff] * df[cols.calc.spend_per_trans_diff]) / 2)
        )
        * df[cols.agg.customer_id_p2]
    ) - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 4)

    df[cols.calc.spend_per_trans_contrib] = (
        (
            df[cols.calc.spend_per_cust_p2]
            - (df[cols.calc.spend_per_trans_p1] * df[cols.calc.trans_per_cust_p2])
            - ((df[cols.calc.trans_per_cust_diff] * df[cols.calc.spend_per_trans_diff]) / 2)
        )
        * df[cols.agg.customer_id_p2]
    ) - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 4)

    if cols.agg.unit_qty in df_cols:
        # Difference calculations
        for col in [
            cols.agg.unit_qty,
            cols.calc.units_per_trans,
            cols.calc.price_per_unit,
        ]:
            df[col + "_" + get_option("column.suffix.difference")] = (
                df[col + "_" + get_option("column.suffix.period_2")]
                - df[col + "_" + get_option("column.suffix.period_1")]
            )

        for col in [
            cols.agg.unit_qty,
            cols.calc.units_per_trans,
            cols.calc.price_per_unit,
        ]:
            p1_col = df[col + "_" + get_option("column.suffix.period_1")]
            df[col + "_" + get_option("column.suffix.percent_difference")] = df[
                col + "_" + get_option("column.suffix.difference")
            ] / p1_col.replace(0, np.nan)

        df[cols.calc.price_per_unit_contrib] = (
            (
                (
                    df[cols.calc.spend_per_trans_p2]
                    - (df[cols.calc.price_per_unit_p1] * df[cols.calc.units_per_trans_p2])
                    - ((df[cols.calc.units_per_trans_diff] * df[cols.calc.price_per_unit_diff]) / 2)
                )
                * df[cols.calc.trans_per_cust_p2]
            )
            - ((df[cols.calc.trans_per_cust_diff] * df[cols.calc.spend_per_trans_diff]) / 4)
        ) * df[cols.agg.customer_id_p2] - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 8)

        df[cols.calc.units_per_trans_contrib] = (
            (
                (
                    df[cols.calc.spend_per_trans_p2]
                    - (df[cols.calc.units_per_trans_p1] * df[cols.calc.price_per_unit_p2])
                    - ((df[cols.calc.units_per_trans_diff] * df[cols.calc.price_per_unit_diff]) / 2)
                )
                * df[cols.calc.trans_per_cust_p2]
            )
            - ((df[cols.calc.trans_per_cust_diff] * df[cols.calc.spend_per_trans_diff]) / 4)
        ) * df[cols.agg.customer_id_p2] - ((df[cols.agg.customer_id_diff] * df[cols.calc.spend_per_cust_diff]) / 8)

    cols = RevenueTree._get_final_col_order(include_quantity=cols.agg.unit_qty in df_cols)

    return df[cols]


class RevenueTree:
    """Period-over-period revenue decomposition with per-factor contributions.

    Accepts pandas or Ibis input; the period aggregation runs in the Ibis engine via
    ``_agg_data``. The result is in ``df`` (pandas); ``draw_tree()`` renders the
    decomposition diagram.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str,
        p2_value: str,
        group_col: str | list[str] | None = None,
    ) -> None:
        """Initialize the Revenue Tree and compute the KPI table immediately.

        If the input contains the option-derived ``column.unit_quantity`` column,
        units/price/price-elasticity metrics are added. ``p1_value`` / ``p2_value``
        are matched by equality against ``period_col``; rows for any other period
        are silently dropped.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with the option-derived
                customer_id, transaction_id, unit_spend, and (optional) unit_quantity columns.
            period_col (str): Column identifying the period.
            p1_value (str): Value representing the first period.
            p2_value (str): Value representing the second period.
            group_col (str | list[str] | None): Column(s) to break the analysis down by.

        Raises:
            ValueError: If the required columns are not present in the data.
        """
        cols = ColumnHelper()

        if group_col is not None:
            group_col = ensure_columns(df, group_col, "group_col")

        required_cols = [cols.customer_id, cols.transaction_id, cols.unit_spend]
        if cols.unit_qty in df.columns:
            required_cols.append(cols.unit_qty)
        # group_col is already validated above; only the function's hard-coded requirements remain.
        ensure_data_has_columns(df, required_cols)

        df, p1_index, p2_index = self._agg_data(df, period_col, p1_value, p2_value, group_col)

        self.df = calc_tree_kpis(
            df=df,
            p1_index=p1_index,
            p2_index=p2_index,
        )

    @staticmethod
    def _agg_data(
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str,
        p2_value: str,
        group_col: list[str] | None = None,
    ) -> tuple[pd.DataFrame, list[bool], list[bool]]:
        """Aggregate data by period and optional grouping columns in the Ibis engine.

        Args:
            df (pd.DataFrame | ibis.Table): Input data.
            period_col (str): Column name for the period.
            p1_value (str): Value representing period 1.
            p2_value (str): Value representing period 2.
            group_col (list[str] | None): List of column names to group by.

        Returns:
            tuple[pd.DataFrame, list[bool], list[bool]]: Materialized frame (one row per
                (group, period), p1 rows first, sorted by ``group_col``) plus boolean masks
                selecting the p1 and p2 rows. The frame index is ``["p1", "p2"]`` when
                ungrouped, else the group index (categorical for a single column,
                MultiIndex for multiple).
        """
        cols = ColumnHelper()

        df = ensure_ibis_table(df)

        aggs = {
            cols.agg.customer_id: df[cols.customer_id].nunique(),
            cols.agg.transaction_id: df[cols.transaction_id].nunique(),
            cols.agg.unit_spend: df[cols.unit_spend].sum(),
        }
        if cols.unit_qty in df.columns:
            aggs[cols.agg.unit_qty] = df[cols.unit_qty].sum()

        group_by_cols = [*group_col, period_col] if group_col else [period_col]
        df = pd.DataFrame(df.group_by(group_by_cols).aggregate(**aggs).execute())
        p1_df = df[df[period_col] == p1_value].drop(columns=[period_col])
        p2_df = df[df[period_col] == p2_value].drop(columns=[period_col])

        if group_col is not None:
            p1_df = p1_df.sort_values(by=group_col)
            p2_df = p2_df.sort_values(by=group_col)

        new_p1_index = [True] * len(p1_df) + [False] * len(p2_df)
        new_p2_index = [not i for i in new_p1_index]

        result_df = pd.concat([p1_df, p2_df], ignore_index=True)

        if group_col is None:
            result_df.index = ["p1", "p2"]
        else:
            result_df = result_df.set_index(group_col)
            if len(group_col) == 1:
                result_df.index = pd.CategoricalIndex(result_df.index)
            # else: MultiIndex created automatically by set_index
        return result_df, new_p1_index, new_p2_index

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
