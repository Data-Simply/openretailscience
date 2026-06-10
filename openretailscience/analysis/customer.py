"""Customer Purchase Behavior Analysis for Retention and Value Optimization.

## Business Context

Understanding customer purchase patterns is fundamental to retail success. Some customers
make single purchases and never return, while others become loyal repeat buyers. This
module analyzes the distribution of purchase frequency to identify customer behavior
segments and inform retention strategies.

## The Business Problem

Retailers need to understand the relationship between customer purchase frequency and
business performance:
- What percentage of customers are one-time buyers versus repeat customers?
- How does purchase frequency relate to customer lifetime value?
- Which customer segments offer the greatest growth opportunities?

Without this analysis, businesses may invest equally in all customers or fail to
identify high-potential segments for targeted retention efforts.

## Real-World Applications

### Customer Retention Strategy
- Identify the percentage of one-time buyers for targeted reactivation campaigns
- Segment customers by purchase frequency for differentiated marketing approaches
- Develop loyalty programs based on actual behavior patterns

### Resource Allocation
- Focus retention efforts on customers showing repeat purchase potential
- Allocate customer service resources based on customer value segments
- Optimize marketing spend by targeting high-frequency customer characteristics

### Business Performance Monitoring
- Track changes in purchase frequency distribution over time
- Monitor the health of customer acquisition versus retention balance
- Identify shifts in customer behavior that may indicate market changes

This module computes purchase-frequency statistics that can be visualized with
the plotting helpers in `openretailscience.plots`.
"""

from __future__ import annotations

import datetime
import operator
from typing import TYPE_CHECKING

import ibis

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table, ensure_value_choice
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    import pandas as pd

_COMPARISONS = {
    "less_than": operator.lt,
    "less_than_equal_to": operator.le,
    "equal_to": operator.eq,
    "not_equal_to": operator.ne,
    "greater_than": operator.gt,
    "greater_than_equal_to": operator.ge,
}


def _distinct_customer_days(df: ibis.Table) -> ibis.Table:
    """Project to (customer_id, transaction_day) and dedupe.

    The day-level dedup defines what a "purchase day" means for this module — same-day
    transactions collapse to a single purchase day. Both DaysBetweenPurchases and
    TransactionChurn walk the customer history one row per purchase day.
    """
    cols = ColumnHelper()
    return df.select(
        df[cols.customer_id],
        transaction_day=df[cols.transaction_date].truncate("D"),
    ).distinct()


class PurchasesPerCustomer:
    """Computes the number of distinct purchases per customer.

    Attributes:
        table (ibis.Table): One row per customer with columns ``customer_id``
            and ``purchase_count``.
        df (pd.DataFrame): Materialized view of ``table`` indexed by
            ``customer_id``. Lazily computed on first access.
    """

    def __init__(self, df: pd.DataFrame | ibis.Table) -> None:
        """Initialize the PurchasesPerCustomer class.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data containing the
                ``customer_id`` and ``transaction_id`` columns.

        Raises:
            ValueError: If the required columns are missing.
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table.
        """
        cols = ColumnHelper()
        df = ensure_ibis_table(df)
        ensure_data_has_columns(df, [cols.customer_id, cols.transaction_id])

        self._customer_id_col = cols.customer_id
        self.table = df.group_by(cols.customer_id).aggregate(
            purchase_count=df[cols.transaction_id].nunique(),
        )
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Materialized purchase counts indexed by customer_id."""
        if self._df is None:
            self._df = self.table.execute().set_index(self._customer_id_col).sort_index()
        return self._df

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Return the purchase count at the given percentile across customers.

        Args:
            percentile (float): Percentile in [0, 1]. Defaults to 0.5 (median).

        Returns:
            float: The purchase count at that percentile, using linear interpolation.
        """
        return float(self.table.purchase_count.quantile(percentile).execute())

    def find_purchase_percentile(
        self,
        number_of_purchases: int,
        comparison: str = "less_than_equal_to",
    ) -> float:
        """Return the share of customers whose purchase count matches a comparison.

        Args:
            number_of_purchases (int): Threshold to compare against.
            comparison (str): One of ``less_than``, ``less_than_equal_to``,
                ``equal_to``, ``not_equal_to``, ``greater_than``,
                ``greater_than_equal_to``. Defaults to ``less_than_equal_to``.

        Returns:
            float: Fraction of customers satisfying the comparison.

        Raises:
            ValueError: If ``comparison`` is not a recognized operator name.
        """
        op = _COMPARISONS[ensure_value_choice(comparison, _COMPARISONS, "comparison")]
        agg = self.table.aggregate(
            matched=op(self.table.purchase_count, number_of_purchases).cast("int").sum(),
            total=self.table.count(),
        ).execute()
        total = int(agg["total"].iloc[0])
        if total == 0:
            return float("nan")
        return float(agg["matched"].iloc[0]) / total


class DaysBetweenPurchases:
    """Computes the average number of days between purchases per customer.

    Single-purchase-day customers are excluded.

    Attributes:
        table (ibis.Table): One row per customer with columns ``customer_id``
            and ``avg_days_between_purchases``.
        df (pd.DataFrame): Materialized view of ``table`` indexed by
            ``customer_id``. Lazily computed on first access.
    """

    def __init__(self, df: pd.DataFrame | ibis.Table) -> None:
        """Initialize the DaysBetweenPurchases class.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data containing the
                ``customer_id`` and ``transaction_date`` columns.

        Raises:
            ValueError: If the required columns are missing.
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table.
        """
        cols = ColumnHelper()
        df = ensure_ibis_table(df)
        ensure_data_has_columns(df, [cols.customer_id, cols.transaction_date])
        self._customer_id_col = cols.customer_id
        self.table = self._calculate(df)
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Materialized per-customer mean gaps indexed by customer_id."""
        if self._df is None:
            self._df = self.table.execute().set_index(self._customer_id_col).sort_index()
        return self._df

    @staticmethod
    def _calculate(df: ibis.Table) -> ibis.Table:
        cols = ColumnHelper()
        per_customer_day = _distinct_customer_days(df)
        window = ibis.window(
            group_by=per_customer_day[cols.customer_id],
            order_by=per_customer_day.transaction_day,
        )
        with_prev = per_customer_day.mutate(
            prev_day=per_customer_day.transaction_day.lag(1).over(window),
        )
        with_gap = with_prev.mutate(
            gap_days=with_prev.transaction_day.delta(with_prev.prev_day, unit="day"),
        )
        return (
            with_gap.filter(with_gap.gap_days.notnull())  # noqa: PD004 (ibis API, not pandas)
            .group_by(cols.customer_id)
            .aggregate(avg_days_between_purchases=with_gap.gap_days.mean())
        )

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Return the average inter-purchase gap (in days) at the given percentile.

        Args:
            percentile (float): Percentile in [0, 1]. Defaults to 0.5 (median).

        Returns:
            float: The average gap at that percentile, using linear interpolation.
        """
        return float(self.table.avg_days_between_purchases.quantile(percentile).execute())


class TransactionChurn:
    """Computes the churn rate by transaction number.

    A customer is "churned" at their N-th transaction if it is their final
    transaction and occurred strictly before ``max(transaction_date) -
    churn_period`` days.

    Attributes:
        table (ibis.Table): Per-``transaction_number`` ``retained``,
            ``churned``, and ``churned_pct`` columns.
        df (pd.DataFrame): Materialized view of ``table`` indexed by
            ``transaction_number`` and sorted ascending. Lazily computed on
            first access.
        n_unique_customers (int): Distinct customers in the input data.
    """

    def __init__(self, df: pd.DataFrame | ibis.Table, churn_period: int) -> None:
        """Initialize the TransactionChurn class.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data containing the
                ``customer_id`` and ``transaction_date`` columns.
            churn_period (int): Number of days of inactivity after which a
                customer is considered churned.

        Raises:
            ValueError: If the required columns are missing.
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table.
        """
        cols = ColumnHelper()
        df = ensure_ibis_table(df)
        ensure_data_has_columns(df, [cols.customer_id, cols.transaction_date])

        self.n_unique_customers = int(df[cols.customer_id].nunique().execute())
        self.table = self._calculate(df, churn_period)
        self._df: pd.DataFrame | None = None

    @property
    def df(self) -> pd.DataFrame:
        """Materialized churn table indexed by transaction_number."""
        if self._df is None:
            self._df = self.table.execute().set_index("transaction_number").sort_index()
        return self._df

    @staticmethod
    def _calculate(df: ibis.Table, churn_period: int) -> ibis.Table:
        cols = ColumnHelper()
        per_customer_day = _distinct_customer_days(df)
        cust_window = ibis.window(
            group_by=per_customer_day[cols.customer_id],
            order_by=per_customer_day.transaction_day,
        )

        # Materialize the scalar max once. On non-DuckDB backends, using a deferred
        # reduction in a downstream `<` comparison can plan as a correlated subquery
        # evaluated per row; passing a literal avoids that.
        max_day = per_customer_day.transaction_day.max().execute()
        churn_boundary = max_day - datetime.timedelta(days=churn_period)

        # `is_last_transaction` must be computed BEFORE filtering to the churn window:
        # the lead must look at the customer's full history. Filtering first would
        # mark a customer's last in-window transaction as "last" even when they have
        # later transactions outside the window — the opposite of the intended flag.
        annotated = per_customer_day.mutate(
            transaction_number=ibis.row_number().over(cust_window) + 1,
            is_last_transaction=per_customer_day.transaction_day.lead(1).over(cust_window).isnull(),  # noqa: PD003 (ibis API, not pandas)
        )

        in_window = annotated.filter(annotated.transaction_day < churn_boundary)
        grouped = in_window.group_by(in_window.transaction_number).aggregate(
            retained=(~in_window.is_last_transaction).cast("int").sum(),
            churned=in_window.is_last_transaction.cast("int").sum(),
        )
        return grouped.mutate(
            churned_pct=grouped.churned / (grouped.churned + grouped.retained),
        )
