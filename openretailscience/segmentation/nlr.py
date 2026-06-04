"""New-Lapsed-Repeating (NLR) Segmentation for Customer Lifecycle Classification.

## Business Context

Understanding customer lifecycle stages is fundamental to retail strategy. Customers naturally
move between periods of activity and inactivity, and identifying these transitions enables
targeted interventions at each stage.

## The Business Problem

Retailers need to understand which customers are growing the base (new), which are loyal
(repeating), and which have stopped purchasing (lapsed). Without this classification,
marketing budgets are misallocated — spending acquisition dollars on existing customers
or ignoring at-risk customers who are about to churn.

## Segment Definitions

Given two time periods (P1 and P2), customers are classified based on where they have
a positive aggregated value:

- **New**: Positive value in P2 only — these customers were acquired in the later period
- **Repeating**: Positive value in both P1 and P2 — these customers are retained
- **Lapsed**: Positive value in P1 only — these customers have stopped purchasing

A customer must have a positive aggregated value (> 0) in a period to be considered as
having "bought" in that period. Zero or negative values do not count. Customers with
no positive value in either period are excluded from the results entirely, as they do
not meet the criteria for any segment. When using non-spend aggregation functions like
``count`` or ``nunique``, the threshold still applies but measures transaction presence
rather than spend.

## Real-World Applications

### New Customers
- Measure acquisition effectiveness period-over-period
- Design onboarding journeys to convert first-time buyers to repeat customers
- Track new customer quality (spend levels, category breadth)

### Repeating Customers
- Core loyal base driving consistent revenue
- Cross-sell and upsell opportunities
- Loyalty program engagement and reward optimization

### Lapsed Customers
- Win-back campaigns with targeted incentives
- Churn root cause analysis (price sensitivity, assortment gaps)
- Customer lifetime value recalculation and reactivation ROI modeling
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
    """Segments customers into New, Repeating, and Lapsed based on presence across two periods.

    NLRSegmentation compares customer purchasing activity across two defined time periods (P1 and P2) to classify
    each customer's lifecycle stage. A customer is considered active in a period only if their aggregated value is
    strictly positive (> 0). This enables retailers to measure acquisition, retention, and churn rates in a single
    view, and to size the revenue impact of each lifecycle segment.

    The segmentation is commonly used in period-over-period reporting (e.g., year-over-year, quarter-over-quarter)
    to answer questions such as how many customers were retained, how many were lost, and how many are newly acquired.
    When combined with spend data, it reveals whether revenue growth is driven by new customer acquisition or by
    increasing spend from repeating customers.

    Attributes:
        table (ibis.Table): The underlying ibis Table expression containing the segmentation results.
            Can be used for further ibis operations before materializing to a DataFrame.
        df (pd.DataFrame): The materialized segmentation results as a pandas DataFrame, indexed by
            customer_id (and group_col if specified). Cached after first access.

    Example:
        >>> import pandas as pd
        >>> from openretailscience.segmentation.nlr import NLRSegmentation
        >>> df = pd.DataFrame({
        ...     "customer_id": [1, 2, 3, 1, 3, 4],
        ...     "unit_spend": [50.0, 100.0, 80.0, 75.0, 60.0, 150.0],
        ...     "year": [2023, 2023, 2023, 2024, 2024, 2024],
        ... })
        >>> seg = NLRSegmentation(df=df, period_col="year", p1_value=2023, p2_value=2024)
        >>> seg.df[["segment_name"]]
                     segment_name
        customer_id
        1               Repeating
        2                  Lapsed
        3               Repeating
        4                     New
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
        """Segments customers into New, Repeating, and Lapsed based on positive aggregated value across two periods.

        A customer is considered to have "bought" in a period only if their aggregated value_col
        is strictly positive (> 0). Customers are then classified as:
        - New: positive value in P2 only
        - Repeating: positive value in both P1 and P2
        - Lapsed: positive value in P1 only

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data. Must contain customer_id, period_col,
                and value_col columns.
            period_col (str): Column containing period identifiers.
            p1_value (str | float | ibis.Scalar): Value in period_col identifying period 1.
            p2_value (str | float | ibis.Scalar): Value in period_col identifying period 2.
            value_col (str | None, optional): Column to aggregate for determining customer activity.
                Defaults to ColumnHelper().unit_spend.
            agg_func (str, optional): Aggregation function to use when grouping by customer_id.
                Defaults to "sum".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments.
                When specified, segments are calculated within each group independently.
                Defaults to None.

        Raises:
            ValueError: If required columns are missing from the DataFrame, or if agg_func is not
                a supported aggregation function.
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
        """Returns the DataFrame with segment names, indexed by customer_id (and group_col if specified)."""
        cols = ColumnHelper()
        index_cols = [cols.customer_id]
        if self._group_col is not None:
            index_cols.extend(self._group_col)
        return self.table.execute().set_index(index_cols)
