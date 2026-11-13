"""Cohort retention matrix from transaction data.

Column ``min_period_shopped`` = cohort start period; integer columns = periods since start. For churn per
purchase-day ordinal, use `customer.TransactionChurn`; render with `plots.cohort.plot`.
"""

from typing import ClassVar

import ibis
import numpy as np
import pandas as pd

from openretailscience.core.validation import (
    ensure_agg_func,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_period,
    ensure_tznaive_datetime,
)
from openretailscience.options import ColumnHelper


def _periods_between(start: pd.Series, end: pd.Series, period: str) -> pd.Series:
    """Count whole periods elapsed from ``start`` to ``end`` for each row.

    Computed in pandas rather than with Ibis ``.delta(unit=period)`` because the Oracle
    backend supports only day-granularity date deltas. Both inputs are already truncated
    to period boundaries, so the elapsed-period count is exact.

    Args:
        start (pd.Series): The cohort's first period (a truncated date) for each row.
        end (pd.Series): The observed period (a truncated date) for each row.
        period (str): One of "year", "quarter", "month", "week", or "day".

    Returns:
        pd.Series: Whole periods between ``start`` and ``end`` as int64.

    Raises:
        ValueError: If ``period`` is not a supported value.
    """
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    if period == "day":
        result = (end - start).dt.days
    elif period == "week":
        result = (end - start).dt.days // 7
    elif period == "month":
        result = (end.dt.year - start.dt.year) * 12 + (end.dt.month - start.dt.month)
    elif period == "quarter":
        result = (end.dt.year - start.dt.year) * 4 + (end.dt.quarter - start.dt.quarter)
    elif period == "year":
        result = end.dt.year - start.dt.year
    else:
        error_msg = f"Unsupported period: {period!r}"
        raise ValueError(error_msg)
    return result.astype("int64")


class CohortAnalysis:
    """Computes a cohort retention matrix from transaction data.

    Customers are grouped into cohorts by the period of their first purchase. ``.df``
    (pandas) has one row per period from the first cohort to the last: column
    ``min_period_shopped`` (the period's start), then integer ``period_since`` columns from 0
    to the number of periods from the first cohort to the end of the data. Each cell is
    ``agg_func`` of ``aggregation_column`` over that cohort's transactions in that period. A
    period the data covers with no transactions, including in a cohort with no customers, is 0.
    A period the cohort hasn't reached by the end of the data is NaN;
    ``.melt(id_vars="min_period_shopped").dropna()`` gives one row per covered cell. The
    query runs at construction; there is no ``.table``.

    Examples:
        >>> from openretailscience.analysis import CohortAnalysis
        >>> cohorts = CohortAnalysis(
        ...     df=transactions, aggregation_column="customer_id", period="quarter", percentage=True
        ... )
        >>> cohorts.df
        period_since min_period_shopped    0     1     2     3     4     5
        0                    2024-01-01  1.0  0.28  0.38  0.38  0.35  0.28
        1                    2024-04-01  1.0  0.38  0.38  0.33  0.17   NaN
        ...
    """

    VALID_PERIODS: ClassVar[tuple[str, ...]] = ("year", "quarter", "month", "week", "day")

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        aggregation_column: str,
        agg_func: str = "nunique",
        period: str = "month",
        percentage: bool = False,
    ) -> None:
        """Initialize the CohortAnalysis object.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data. ``transaction_date`` must be a
                date or datetime column; strings are rejected.
            aggregation_column (str): The column to aggregate. For customer retention use
                ``customer_id`` with ``agg_func="nunique"``.
            agg_func (str, optional): Ibis column aggregation name, case-sensitive (e.g.
                "nunique", "count", "sum", "mean", "min", "max"). "nunique" on ``unit_spend``
                counts distinct prices, not customers. Defaults to "nunique".
            period (str): Period for cohort analysis: "year", "quarter", "month", "week", or "day"
                (case-insensitive; short forms like "m"/"d" accepted).
            percentage (bool): If True, each row is divided by its period-0 value, giving
                fractions (1.0 = 100%) rounded to 2 decimals. A cohort with 0 in period 0 is 0 in
                every period the data covers.

        Raises:
            TypeError: If `df` is not a pandas DataFrame or an Ibis Table.
            TypeError: If `period` is not a string.
            TypeError: If `transaction_date` is not a date or datetime column.
            ValueError: If `period` is not one of the allowed values.
            ValueError: If `df` is missing required columns
                (`customer_id`, `transaction_date`, or `aggregation_column`).
            ValueError: If `agg_func` is not an aggregation available for `aggregation_column`
                with a numeric result.
            ValueError: If `transaction_date` is timezone-aware.
            ValueError: If `df` has no rows with a non-null `customer_id` (e.g. empty or anonymous-only data).
        """
        cols = ColumnHelper()

        df = ensure_ibis_table(df)
        period = ensure_period(period, self.VALID_PERIODS, "period")

        required_cols = [
            cols.customer_id,
            cols.transaction_date,
            aggregation_column,
        ]
        ensure_data_has_columns(df, required_cols)
        ensure_tznaive_datetime(df, cols.transaction_date)
        ensure_agg_func(df, aggregation_column, agg_func, numeric=True)

        self.df = self._calculate_cohorts(
            df=df,
            agg_func=agg_func,
            period=period,
            aggregation_column=aggregation_column,
            percentage=percentage,
        )

    def _fill_cohort_gaps(
        self,
        cohort_analysis_table: pd.DataFrame,
        period: str,
        last_period: pd.Timestamp,
    ) -> pd.DataFrame:
        """Adds the cohorts and periods missing from the pivot: 0 up to ``last_period``, NaN after it.

        Args:
            cohort_analysis_table (pd.DataFrame): The pivoted cohort matrix, indexed by cohort period.
            period (str): The period of analysis (year, quarter, month, week, or day).
            last_period (pd.Timestamp): The last period in the data.

        Returns:
            pd.DataFrame: The cohort matrix with every cohort and period filled.
        """
        cohort_analysis_table.index = pd.to_datetime(cohort_analysis_table.index)

        min_period = cohort_analysis_table.index.min()
        max_period = cohort_analysis_table.index.max()

        # Week steps by 7 days from the (truncated) min period rather than freq="W", which
        # anchors on Sunday and would miss the Monday week-starts that truncate("week") yields.
        period_lookup = {"year": "YS", "quarter": "QS", "month": "MS", "week": "7D", "day": "D"}
        full_range = pd.date_range(start=min_period, end=max_period, freq=period_lookup[period])
        first_cohort_age = _periods_between(pd.Series([min_period]), pd.Series([last_period]), period).iloc[0]
        all_periods = range(first_cohort_age + 1)
        cohort_analysis_table = cohort_analysis_table.reindex(index=full_range, columns=all_periods).fillna(0)
        # Row i starts i periods after the first cohort, so it reaches the end of the data i periods sooner
        observed = np.add.outer(np.arange(len(full_range)), all_periods) <= first_cohort_age
        return cohort_analysis_table.where(observed)

    def _calculate_cohorts(
        self,
        df: ibis.Table,
        aggregation_column: str,
        agg_func: str = "nunique",
        period: str = "month",
        percentage: bool = False,
    ) -> pd.DataFrame:
        """Computes the cohort matrix described on the class.

        Args:
            df (ibis.Table): The dataset containing transaction data.
            aggregation_column (str): The column to aggregate.
            agg_func (str, optional): Ibis aggregation name. Defaults to "nunique".
            period (str): Period for cohort analysis: "year", "quarter", "month", "week", or "day".
            percentage (bool): If True, divide each row by its period-0 value.

        Returns:
            pd.DataFrame: The cohort matrix.
        """
        cols = ColumnHelper()

        filtered_table = df.mutate(
            period_shopped=df[cols.transaction_date].truncate(period),
            period_value=df[aggregation_column],
        )

        customer_cohort = filtered_table.group_by(cols.customer_id).aggregate(
            min_period_shopped=filtered_table.period_shopped.min(),
        )

        cohort_table = (
            filtered_table.join(customer_cohort, [cols.customer_id])
            .group_by("min_period_shopped", "period_shopped")
            .aggregate(period_value=getattr(filtered_table.period_value, agg_func)())
        )

        cohort_df = cohort_table.execute()
        if len(cohort_df) == 0:
            msg = f"df has no rows with a non-null {cols.customer_id}, so there are no customers to build cohorts from."
            raise ValueError(msg)
        cohort_df["period_since"] = _periods_between(
            cohort_df["min_period_shopped"],
            cohort_df["period_shopped"],
            period,
        )
        cohort_df = cohort_df.drop_duplicates(subset=["min_period_shopped", "period_since"])

        cohort_analysis_table = cohort_df.pivot(
            index="min_period_shopped",
            columns="period_since",
            values="period_value",
        )

        if percentage:
            first_period = cohort_analysis_table[0].replace(0, np.nan)
            cohort_analysis_table = cohort_analysis_table.div(first_period, axis=0).round(2)

        last_period = pd.Timestamp(cohort_df["period_shopped"].max())
        cohort_analysis_table = self._fill_cohort_gaps(cohort_analysis_table, period, last_period)
        cohort_analysis_table.index.name = "min_period_shopped"

        return cohort_analysis_table.reset_index()
