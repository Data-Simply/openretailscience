"""Purchase Path Analysis for category-progression insights.

## Business Context

Customers rarely buy from every category at once. They discover a retailer through one
category and broaden over time — a customer might start in women's clothing, add kids'
clothing on a later trip, then move into menswear. Purchase Path Analysis surfaces these
journeys by recording, for each customer, the basket in which every product category
*first appears*, then aggregating customers who share the same progression.

## What the Output Means

The result has one row per distinct path. ``basket_1`` is the category (or categories)
a customer first bought, ``basket_2`` the next *new* category they added, and so on. A
position is empty when a trip introduced nothing new (the customer only repeated
categories they had already bought). ``customer_count`` is how many customers followed
the path, and ``pct_customers`` is that count as a share of all analysed customers — so
``0.27`` means 27% of analysed customers took that path.

## Business Applications

- **Cross-sell sequencing**: target the category a customer is statistically likely to add next.
- **Onboarding**: understand which entry category leads to the broadest baskets.
- **Category management**: plan adjacencies and promotions around natural progressions.
"""

from __future__ import annotations

import functools

import ibis
import pandas as pd
from ibis import _

from openretailscience.core.validation import (
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_integer,
    ensure_number,
    ensure_positive,
)
from openretailscience.options import ColumnHelper

# pct_customers is a share in [0, 1]; four places preserves two-decimal percentage precision.
PCT_ROUND_DECIMALS = 4
_EMPTY_COLUMNS = ["customer_count", "pct_customers"]


class PurchasePath:
    """Tracks the order in which customers first buy from each product category.

    For every customer the analysis orders their baskets in time, records the basket in
    which each category first appears, and concatenates categories that first appear in
    the same basket. Customers sharing an identical sequence are grouped into a single
    path with a customer count and a customer share.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        category_col: str = "category",
        min_transactions: int = 3,
        min_basket_size: int = 1,
        min_basket_value: float = 0.0,
        max_depth: int = 10,
        min_customers: int = 1,
        exclude_returns: bool = True,
    ) -> None:
        """Compute purchase paths from transaction line-item data.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction line items containing the customer,
                transaction, date, product, spend and category columns.
            category_col (str): Name of the product category column. Defaults to ``"category"``.
            min_transactions (int): Minimum number of qualifying baskets a customer must have
                to be included. Counted before ``max_depth`` truncation. Defaults to 3.
            min_basket_size (int): Minimum number of distinct products for a basket to qualify.
                Defaults to 1 (no size filtering).
            min_basket_value (float): Minimum total spend for a basket to qualify. Defaults to
                0.0 (no value filtering).
            max_depth (int): Maximum number of baskets per customer to analyse, ordered by date.
                Defaults to 10.
            min_customers (int): Minimum number of customers a path must have to appear in the
                result. Defaults to 1.
            exclude_returns (bool): If True, drop line items with non-positive ``unit_spend``
                (returns/refunds) before building baskets. Defaults to True.

        Raises:
            ValueError: If a required column is missing or a filter parameter is non-positive.
            TypeError: If a filter parameter has the wrong type.
        """
        cols = ColumnHelper()

        for value, name in (
            (min_transactions, "min_transactions"),
            (min_basket_size, "min_basket_size"),
            (max_depth, "max_depth"),
            (min_customers, "min_customers"),
        ):
            ensure_integer(value, name)
            ensure_positive(value, name)
        ensure_number(min_basket_value, "min_basket_value")

        required_cols = [
            cols.customer_id,
            cols.transaction_id,
            cols.transaction_date,
            cols.product_id,
            cols.unit_spend,
            category_col,
        ]
        ensure_data_has_columns(df, required_cols)

        self._first_appearance_df = self._calc_first_appearances(
            df=df,
            cols=cols,
            category_col=category_col,
            min_transactions=min_transactions,
            min_basket_size=min_basket_size,
            min_basket_value=min_basket_value,
            max_depth=max_depth,
            exclude_returns=exclude_returns,
        )
        self._category_col = category_col
        self._min_customers = min_customers

    @staticmethod
    def _calc_first_appearances(
        df: pd.DataFrame | ibis.Table,
        cols: ColumnHelper,
        category_col: str,
        min_transactions: int,
        min_basket_size: int,
        min_basket_value: float,
        max_depth: int,
        exclude_returns: bool,
    ) -> pd.DataFrame:
        """Return, per eligible customer and category, the basket position of first appearance.

        The heavy filtering, sequencing and joining run in Ibis; the executed frame has one
        row per (customer, category) with the 1-based ``first_basket`` it first appeared in.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction line items.
            cols (ColumnHelper): Resolved column names.
            category_col (str): Product category column name.
            min_transactions (int): Minimum qualifying baskets per customer (pre-truncation).
            min_basket_size (int): Minimum distinct products per qualifying basket.
            min_basket_value (float): Minimum spend per qualifying basket.
            max_depth (int): Maximum baskets analysed per customer.
            exclude_returns (bool): Whether to drop non-positive spend line items first.

        Returns:
            pd.DataFrame: Columns ``[customer_id, category_col, first_basket]``.
        """
        table = ensure_ibis_table(df)
        if exclude_returns:
            table = table.filter(table[cols.unit_spend] > 0)

        baskets = (
            table.group_by([cols.customer_id, cols.transaction_id, cols.transaction_date])
            .aggregate(
                item_count=table[cols.product_id].nunique(),
                basket_value=table[cols.unit_spend].sum(),
            )
            .filter((_.item_count >= min_basket_size) & (_.basket_value >= min_basket_value))
            .mutate(
                basket_number=ibis.row_number().over(
                    ibis.window(group_by=cols.customer_id, order_by=[cols.transaction_date, cols.transaction_id]),
                )
                + 1,
            )
        )

        eligible_customers = (
            baskets.group_by(cols.customer_id)
            .aggregate(basket_count=_.basket_number.count())
            .filter(_.basket_count >= min_transactions)
            .select(cols.customer_id)
        )

        analysis_baskets = (
            baskets.filter(_.basket_number <= max_depth)
            .join(eligible_customers, cols.customer_id)
            .select([cols.customer_id, cols.transaction_id, "basket_number"])
        )

        line_items = table.join(analysis_baskets, [cols.customer_id, cols.transaction_id])
        first_appearances = line_items.group_by([cols.customer_id, category_col]).aggregate(
            first_basket=_.basket_number.min(),
        )
        return first_appearances.execute()

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """The aggregated purchase paths.

        Returns:
            pd.DataFrame: One row per distinct path with ``basket_1 .. basket_N`` category
            columns, ``customer_count`` and ``pct_customers``. When no path meets the
            criteria, an empty DataFrame with columns ``["customer_count", "pct_customers"]``.
        """
        first_df = self._first_appearance_df
        if len(first_df) == 0:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)

        cid = ColumnHelper().customer_id
        first_df = first_df.copy()
        first_df[self._category_col] = first_df[self._category_col].astype(str)
        first_df = first_df.sort_values([cid, "first_basket", self._category_col])

        # Concatenate categories that first appear in the same basket (one label per position).
        grouped = (
            first_df.groupby([cid, "first_basket"])[self._category_col].agg(",".join).reset_index(name="categories")
        )

        paths = grouped.pivot(index=cid, columns="first_basket", values="categories")
        total_customers = len(paths)

        # Relabel basket positions to a contiguous basket_1 .. basket_N and fill gaps.
        max_basket = int(paths.columns.max())
        paths = paths.reindex(columns=range(1, max_basket + 1)).fillna("")
        basket_cols = [f"basket_{position}" for position in paths.columns]
        paths.columns = basket_cols

        patterns = paths.groupby(basket_cols).size().reset_index(name="customer_count")
        patterns = patterns[patterns["customer_count"] >= self._min_customers]
        if len(patterns) == 0:
            return pd.DataFrame(columns=_EMPTY_COLUMNS)

        patterns["pct_customers"] = (patterns["customer_count"] / total_customers).round(PCT_ROUND_DECIMALS)
        return patterns.sort_values(
            ["customer_count", *basket_cols],
            ascending=[False, *[True] * len(basket_cols)],
        ).reset_index(drop=True)
