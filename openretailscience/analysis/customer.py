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
import functools
import operator
from typing import TYPE_CHECKING

import ibis

from openretailscience.core.validation import (
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_integer,
    ensure_number,
    ensure_positive,
    ensure_tznaive_datetime,
    ensure_unit_interval,
    ensure_value_choice,
)
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

# Name of the per-transaction-number index column produced by TransactionChurn. Defined once
# so the aggregation that creates it and the materialized view that indexes on it cannot drift.
_TRANSACTION_NUMBER_COL = "transaction_number"


def _distinct_customer_days(df: ibis.Table, customer_id_col: str, transaction_date_col: str) -> ibis.Table:
    """Project to (customer_id, transaction_day) and dedupe.

    The day-level dedup defines what a "purchase day" means for this module — same-day
    transactions collapse to a single purchase day. Both DaysBetweenPurchases and
    TransactionChurn walk the customer history one row per purchase day.

    The column names are passed in (resolved once by the caller) rather than re-read from
    options here, so this function is pure and cannot drift from the caller's resolution.

    Args:
        df (ibis.Table): Transaction-level data.
        customer_id_col (str): Resolved name of the customer id column.
        transaction_date_col (str): Resolved name of the transaction date column.

    Returns:
        ibis.Table: One row per (customer, purchase day).
    """
    return df.select(
        df[customer_id_col],
        transaction_day=df[transaction_date_col].truncate("D"),
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

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Materialized purchase counts indexed by customer_id."""
        return self.table.execute().set_index(self._customer_id_col).sort_index()

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Return the purchase count at the given percentile across customers.

        Args:
            percentile (float): Percentile in [0, 1]. Defaults to 0.5 (median).

        Returns:
            float: The purchase count at that percentile, using linear interpolation.

        Raises:
            TypeError: If ``percentile`` is not a number.
            ValueError: If ``percentile`` is outside [0, 1].
        """
        ensure_unit_interval(percentile, "percentile")
        return float(self.df["purchase_count"].quantile(percentile))

    def find_purchase_percentile(
        self,
        number_of_purchases: float,
        comparison: str = "less_than_equal_to",
    ) -> float:
        """Return the share of customers whose purchase count matches a comparison.

        Args:
            number_of_purchases (float): Threshold to compare against. Typically an integer count,
                but fractional values are compared directly against the integer purchase counts
                (e.g. ``<= 2.5`` is equivalent to ``<= 2``).
            comparison (str): One of ``less_than``, ``less_than_equal_to``,
                ``equal_to``, ``not_equal_to``, ``greater_than``,
                ``greater_than_equal_to``. Defaults to ``less_than_equal_to``.
                Matching is case-insensitive.

        Returns:
            float: Fraction of customers satisfying the comparison, or NaN if there
                are no customers.

        Raises:
            TypeError: If ``number_of_purchases`` is not a number, or ``comparison`` is not a string.
            ValueError: If ``comparison`` is not a recognized operator name.
        """
        ensure_number(number_of_purchases, "number_of_purchases")
        op = _COMPARISONS[ensure_value_choice(comparison, _COMPARISONS, "comparison")]
        counts = self.df["purchase_count"]
        # mean of the boolean mask is the matching share; on an empty frame it is NaN.
        return float(op(counts, number_of_purchases).mean())


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
            ValueError: If the required columns are missing, or transaction_date is timezone-aware.
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table, or transaction_date
                is not a date/datetime type.
        """
        cols = ColumnHelper()
        df = ensure_ibis_table(df)
        ensure_data_has_columns(df, [cols.customer_id, cols.transaction_date])
        ensure_tznaive_datetime(df, cols.transaction_date)
        self._customer_id_col = cols.customer_id
        self.table = self._calculate(df, cols.customer_id, cols.transaction_date)

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Materialized per-customer mean gaps indexed by customer_id."""
        return self.table.execute().set_index(self._customer_id_col).sort_index()

    @staticmethod
    def _calculate(df: ibis.Table, customer_id_col: str, transaction_date_col: str) -> ibis.Table:
        """Compute each customer's mean gap (in days) between consecutive purchase days.

        Args:
            df (ibis.Table): Transaction-level data.
            customer_id_col (str): Resolved name of the customer id column.
            transaction_date_col (str): Resolved name of the transaction date column.

        Returns:
            ibis.Table: One row per customer with ``avg_days_between_purchases``;
                single-purchase-day customers are excluded.
        """
        per_customer_day = _distinct_customer_days(df, customer_id_col, transaction_date_col)
        window = ibis.window(
            group_by=per_customer_day[customer_id_col],
            order_by=per_customer_day.transaction_day,
        )
        with_gap = per_customer_day.mutate(
            gap_days=per_customer_day.transaction_day.delta(
                per_customer_day.transaction_day.lag(1).over(window),
                unit="day",
            ),
        )
        filtered = with_gap.filter(with_gap.gap_days.notnull())  # noqa: PD004 (ibis API, not pandas)
        return filtered.group_by(customer_id_col).aggregate(
            avg_days_between_purchases=filtered.gap_days.mean(),
        )

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Return the average inter-purchase gap (in days) at the given percentile.

        Args:
            percentile (float): Percentile in [0, 1]. Defaults to 0.5 (median).

        Returns:
            float: The average gap at that percentile, using linear interpolation.

        Raises:
            TypeError: If ``percentile`` is not a number.
            ValueError: If ``percentile`` is outside [0, 1].
        """
        ensure_unit_interval(percentile, "percentile")
        return float(self.df["avg_days_between_purchases"].quantile(percentile))


class TransactionChurn:
    """Computes the churn rate by transaction number.

    A customer is "churned" at their N-th transaction if it is their final
    transaction and occurred strictly before ``max(transaction_date) -
    churn_period`` days.

    Unlike PurchasesPerCustomer and DaysBetweenPurchases — whose queries stay lazy
    until ``.df`` is accessed — construction eagerly runs one aggregate query against
    the backend to read the distinct customer count and the latest purchase day (which
    anchors the churn boundary). Apply any row filtering before constructing this class
    when working with very large remote tables.

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
            churn_period (int): Whole number of days of inactivity after which a
                customer is considered churned. Must be a positive integer; a
                fractional value would place the boundary at a sub-day time.

        Raises:
            ValueError: If the required columns are missing, transaction_date is
                timezone-aware, or ``churn_period`` is not positive.
            TypeError: If ``df`` is not a pandas DataFrame or an Ibis Table,
                transaction_date is not a date/datetime type, or ``churn_period``
                is not an integer.
        """
        cols = ColumnHelper()
        df = ensure_ibis_table(df)
        ensure_data_has_columns(df, [cols.customer_id, cols.transaction_date])
        ensure_tznaive_datetime(df, cols.transaction_date)
        ensure_integer(churn_period, "churn_period")
        ensure_positive(churn_period, "churn_period")

        # One round-trip for both scalars the build needs: the distinct customer count
        # and the latest purchase day (which anchors the churn boundary). max(truncate(x))
        # equals truncate(max(x)) since truncation is monotonic.
        stats = df.aggregate(
            n_unique_customers=df[cols.customer_id].nunique(),
            max_day=df[cols.transaction_date].truncate("D").max(),
        ).execute()
        self.n_unique_customers = int(stats["n_unique_customers"].iloc[0])
        # Empty input: max_day is NaT, so churn_boundary is NaT — harmless, because _calculate
        # then operates on zero rows and returns an empty churn table (see the empty-input test).
        churn_boundary = stats["max_day"].iloc[0] - datetime.timedelta(days=churn_period)
        self.table = self._calculate(df, cols.customer_id, cols.transaction_date, churn_boundary)

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Materialized churn table indexed by transaction_number."""
        return self.table.execute().set_index(_TRANSACTION_NUMBER_COL).sort_index()

    @staticmethod
    def _calculate(
        df: ibis.Table,
        customer_id_col: str,
        transaction_date_col: str,
        churn_boundary: datetime.date | pd.Timestamp,
    ) -> ibis.Table:
        """Compute retained/churned counts and the churn rate per transaction number.

        Args:
            df (ibis.Table): Transaction-level data.
            customer_id_col (str): Resolved name of the customer id column.
            transaction_date_col (str): Resolved name of the transaction date column.
            churn_boundary (datetime.date | pd.Timestamp): Transactions on or after this day
                are outside the churn window. Pre-computed in ``__init__`` from the latest
                purchase day so this method stays a pure function of its arguments.

        Returns:
            ibis.Table: One row per ``transaction_number`` with ``retained``, ``churned``,
                and ``churned_pct``.
        """
        per_customer_day = _distinct_customer_days(df, customer_id_col, transaction_date_col)
        cust_window = ibis.window(
            group_by=per_customer_day[customer_id_col],
            order_by=per_customer_day.transaction_day,
        )

        # `is_last_transaction` must be computed BEFORE filtering to the churn window:
        # the lead must look at the customer's full history. Filtering first would
        # mark a customer's last in-window transaction as "last" even when they have
        # later transactions outside the window — the opposite of the intended flag.
        annotated = per_customer_day.mutate(
            is_last_transaction=per_customer_day.transaction_day.lead(1).over(cust_window).isnull(),  # noqa: PD003 (ibis API, not pandas)
            **{_TRANSACTION_NUMBER_COL: ibis.row_number().over(cust_window) + 1},
        )

        in_window = annotated.filter(annotated.transaction_day < churn_boundary)
        grouped = in_window.group_by(_TRANSACTION_NUMBER_COL).aggregate(
            retained=(~in_window.is_last_transaction).cast("int").sum(),
            churned=in_window.is_last_transaction.cast("int").sum(),
        )
        return grouped.mutate(
            churned_pct=grouped.churned / (grouped.churned + grouped.retained),
        )
