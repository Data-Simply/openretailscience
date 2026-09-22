"""Cohort retention matrix from transaction data.

Index = cohort start period; integer columns = periods since start. For churn per
purchase-day ordinal, use `customer.TransactionChurn`; render with `plots.cohort.plot`.
"""

from typing import ClassVar

import ibis
import numpy as np
import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table, ensure_period
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
    # Count in wall-clock terms: a tz-aware interval spanning a DST change is not a whole
    # number of 24h days, which would round the day/week counts below down by one.
    if start.dt.tz is not None:
        start = start.dt.tz_localize(None)
    if end.dt.tz is not None:
        end = end.dt.tz_localize(None)
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

    Materialized eagerly at construction (``.execute()``); ``.df`` is the pandas result
    (index = cohort start period, integer columns = period index since start, missing
    cells 0). There is no ``.table``; pair with `plots.cohort.plot` for rendering.
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

        Construction runs the aggregate query (eager ``.execute()``); filter large
        remote tables before constructing.

        Args:
            df (pd.DataFrame | ibis.Table): The dataset containing transaction data.
            aggregation_column (str): The column to apply the aggregation function on (e.g., 'unit_spend').
            agg_func (str, optional): Aggregation function (e.g., "nunique", "sum", "mean"). Defaults to "nunique".
            period (str): Period for cohort analysis: "year", "quarter", "month", "week", or "day"
                (case-insensitive; short forms like "m"/"d" accepted).
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Raises:
            TypeError: If `df` is not a pandas DataFrame or an Ibis Table.
            TypeError: If `period` is not a string.
            ValueError: If `period` is not one of the allowed values.
            ValueError: If `df` is missing required columns
                (`customer_id`, `transaction_date`, or `aggregation_column`).
        """
        cols = ColumnHelper()

        period = ensure_period(period, self.VALID_PERIODS, "period")

        required_cols = [
            cols.customer_id,
            cols.transaction_date,
            aggregation_column,
        ]
        ensure_data_has_columns(df, required_cols)

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
    ) -> pd.DataFrame:
        """Fills gaps in the cohort analysis table for missing periods.

        Args:
            cohort_analysis_table (pd.DataFrame): The cohort analysis table to fill gaps in.
            period (str): The period of analysis (year, quarter, month, week, or day).

        Returns:
            pd.DataFrame: Cohort table with missing periods filled.
        """
        cohort_analysis_table.index = pd.to_datetime(cohort_analysis_table.index)

        min_period = cohort_analysis_table.index.min()
        max_period = cohort_analysis_table.index.max()

        # Week steps by 7 days from the (truncated) min period rather than freq="W", which
        # anchors on Sunday and would miss the Monday week-starts that truncate("week") yields.
        period_lookup = {"year": "YS", "quarter": "QS", "month": "MS", "week": "7D", "day": "D"}
        full_range = pd.date_range(start=min_period, end=max_period, freq=period_lookup[period])
        cohort_analysis_table = cohort_analysis_table.reindex(full_range, fill_value=0)
        if cohort_analysis_table.shape[1] > 0:
            max_period_since = cohort_analysis_table.columns.max()
            all_periods = range(max_period_since + 1)
            cohort_analysis_table = cohort_analysis_table.reindex(columns=all_periods, fill_value=0)
        return cohort_analysis_table

    def _calculate_cohorts(
        self,
        df: pd.DataFrame | ibis.Table,
        aggregation_column: str,
        agg_func: str = "nunique",
        period: str = "month",
        percentage: bool = False,
    ) -> pd.DataFrame:
        """Computes a cohort analysis table based on transaction data.

        Args:
            df (pd.DataFrame | ibis.Table): The dataset containing transaction data.
            aggregation_column (str): The column to apply the aggregation function on (e.g., 'unit_spend').
            agg_func (str, optional): Aggregation function (e.g., "nunique", "sum", "mean"). Defaults to "nunique".
            period (str): Period for cohort analysis: "year", "quarter", "month", "week", or "day".
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Returns:
            pd.DataFrame: Pivot with index = cohort start period (`min_period_shopped`) and
                integer columns = period index since start (`period_since`); values are the
                aggregated `aggregation_column`, missing cells 0. With `percentage=True`,
                each row is divided by its first-period value (rounded to 2), NaN filled with 0.
        """
        cols = ColumnHelper()

        ibis_table = ensure_ibis_table(df)

        filtered_table = ibis_table.mutate(
            period_shopped=ibis_table[cols.transaction_date].truncate(period),
            period_value=ibis_table[aggregation_column],
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
        # period_since is computed in pandas (see _periods_between) rather than via Ibis
        # .delta(unit=period); the truncate and group-by above still execute in the database.
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

        cohort_analysis_table = cohort_analysis_table.fillna(0)
        cohort_analysis_table = self._fill_cohort_gaps(cohort_analysis_table, period)
        cohort_analysis_table.index.name = "min_period_shopped"

        return cohort_analysis_table
