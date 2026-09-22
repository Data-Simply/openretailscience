"""New/Lapsed/Repeating (NLR) customer-lifecycle segmentation across two periods.

A customer counts as active in a period only with a strictly positive aggregated value;
customers inactive in both periods are excluded from the results.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import ibis

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_value_choice,
)
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    import pandas as pd

SEGMENT_NEW = "New"
SEGMENT_REPEATING = "Repeating"
SEGMENT_LAPSED = "Lapsed"

VALID_AGG_FUNCS = ("sum", "mean", "max", "count", "nunique")


class NLRSegmentation:
    """Segments customers into New, Repeating, and Lapsed based on activity across two periods.

    A customer is active in a period only if their aggregated value is strictly positive
    (> 0); zero or negative values do not count:
    - New: active in P2 only
    - Repeating: active in both P1 and P2
    - Lapsed: active in P1 only

    With count-like aggregations ("count", "nunique") the positive-value rule measures
    transaction presence rather than spend.

    Attributes:
        table (ibis.Table): The underlying lazy ibis Table expression; further ibis operations
            can be applied before materializing.
        df (pd.DataFrame): The materialized results, indexed by customer_id (and group_col if
            specified). Cached after first access.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        period_col: str,
        p1_value: str | float | ibis.Scalar,
        p2_value: str | float | ibis.Scalar,
        value_col: str | None = None,
        agg_func: str = "sum",
        group_col: str | list[str] | None = None,
    ) -> None:
        """Segments customers into New, Repeating, and Lapsed based on positive aggregated value.

        A customer is active in a period only if their aggregated value_col is strictly
        positive (> 0):
        - New: positive value in P2 only
        - Repeating: positive value in both P1 and P2
        - Lapsed: positive value in P1 only
        Customers with no positive value in either period are excluded from the output.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data. Must contain customer_id, period_col,
                and value_col columns.
            period_col (str): Column containing period identifiers.
            p1_value (str | float | ibis.Scalar): Value in period_col identifying period 1.
            p2_value (str | float | ibis.Scalar): Value in period_col identifying period 2.
            value_col (str | None, optional): Column to aggregate for determining customer activity.
                Defaults to ColumnHelper().unit_spend.
            agg_func (str, optional): Aggregation function: one of "sum", "mean", "max", "count",
                "nunique". Defaults to "sum".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments.
                When specified, segments are calculated within each group independently.
                Defaults to None.

        Raises:
            ValueError: If required columns are missing, agg_func is unsupported, or p1_value
                equals p2_value.
            TypeError: If agg_func is not a string.
        """
        cols = ColumnHelper()
        value_col = cols.unit_spend if value_col is None else value_col

        df = ensure_ibis_table(df)

        self._group_col: list[str] | None = (
            ensure_columns(df, group_col, "group_col") if group_col is not None else None
        )

        # group_col is already validated above; only the function's hard-coded requirements remain.
        ensure_data_has_columns(df, [cols.customer_id, value_col, period_col])

        p1_expr = p1_value if isinstance(p1_value, ibis.Expr) else ibis.literal(p1_value)
        p2_expr = p2_value if isinstance(p2_value, ibis.Expr) else ibis.literal(p2_value)
        if p1_expr.equals(p2_expr):
            msg = f"p1_value and p2_value must be different, got '{p1_value}' for both"
            raise ValueError(msg)

        agg_func = ensure_value_choice(agg_func, VALID_AGG_FUNCS, "agg_func")

        # Filter to only P1 and P2 rows
        df = df.filter((df[period_col] == p1_value) | (df[period_col] == p2_value))

        # Determine which periods each customer has positive spend in
        group_cols = [cols.customer_id]
        if self._group_col is not None:
            group_cols.extend(self._group_col)

        p1_col = f"{value_col}_p1"
        p2_col = f"{value_col}_p2"

        agg_method = getattr(df[value_col], agg_func)
        p1_agg = agg_method(where=df[period_col] == p1_value)
        p2_agg = agg_method(where=df[period_col] == p2_value)
        customer = df.group_by(*group_cols).aggregate(
            **{
                p1_col: p1_agg.fill_null(0),
                p2_col: p2_agg.fill_null(0),
            },
        )

        # Exclude customers with no positive value in either period
        customer = customer.filter((customer[p1_col] > 0) | (customer[p2_col] > 0))

        # Classify: both periods -> Repeating, P1 only -> Lapsed, P2 only -> New
        # Use of ifelse ensures compatibility with some ibis backends that do not support boolean expressions in cases
        # statements
        in_p1 = (customer[p1_col] > 0).ifelse(1, 0)
        in_p2 = (customer[p2_col] > 0).ifelse(1, 0)
        segment_expr = ibis.cases(
            ((in_p1 == 1) & (in_p2 == 1), SEGMENT_REPEATING),
            (in_p1 == 1, SEGMENT_LAPSED),
            (in_p2 == 1, SEGMENT_NEW),
        )

        self.table: ibis.Table = customer.mutate(segment_name=segment_expr).select(
            *group_cols,
            "segment_name",
            p1_col,
            p2_col,
        )

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the DataFrame with segment names, indexed by customer_id (and group_col if specified).

        Columns: segment_name, {value_col}_p1 and {value_col}_p2.
        """
        cols = ColumnHelper()
        index_cols = [cols.customer_id]
        if self._group_col is not None:
            index_cols.extend(self._group_col)
        return self.table.execute().set_index(index_cols)
