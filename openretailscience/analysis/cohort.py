"""Cohort Analysis and User Segmentation.

This module implements functionality for performing cohort analysis, a powerful technique used in customer analytics
and retention strategies.

Cohort analysis helps in understanding customer behavior over time by grouping users based on shared characteristics
or experiences, such as sign-up date, first purchase, or marketing campaign interaction. This method provides
valuable insights into user engagement, retention, and lifetime value, which businesses can leverage in various ways:

1. Customer retention analysis: By tracking how different cohorts behave over time, businesses can identify trends
   in user engagement and develop strategies to improve customer loyalty.

2. Marketing performance evaluation: Understanding how different user groups respond to marketing efforts helps in
   optimizing campaigns for higher conversions and better ROI.

3. Product lifecycle insights: Analyzing user activity across cohorts can reveal product adoption trends and inform
   feature development or enhancements.

4. Revenue forecasting: Cohort-based revenue tracking enables more accurate predictions of future earnings and
   helps in financial planning.

5. Personalization and segmentation: Businesses can tailor their offerings based on cohort behavior to enhance
   customer experience and increase retention rates.

The module employs key metrics such as retention rate, churn rate, and customer lifetime value (CLV) to measure
cohort performance and user engagement over time:

- Retention Rate: The percentage of users who continue to engage with a product or service over a given period.
- Churn Rate: The percentage of users who stop engaging with the product within a specific timeframe.
- Customer Lifetime Value (CLV): The predicted total revenue a customer will generate throughout their relationship
  with the business.

By leveraging cohort analysis, businesses can make data-driven decisions to enhance customer experience, improve
marketing strategies, and drive long-term growth.
"""

from typing import ClassVar

import ibis
import numpy as np
import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns, ensure_value_choice
from openretailscience.options import ColumnHelper


class CohortAnalysis:
    """Class for performing cohort analysis and visualization."""

    VALID_PERIODS: ClassVar[tuple[str, ...]] = ("year", "quarter", "month", "week", "day")

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        aggregation_column: str,
        agg_func: str = "nunique",
        period: str = "month",
        percentage: bool = False,
    ) -> None:
        """Initializes the Cohort Analysis object.

        Args:
            df (pd.DataFrame | ibis.Table): The dataset containing transaction data.
            aggregation_column (str): The column to apply the aggregation function on (e.g., 'unit_spend').
            agg_func (str, optional): Aggregation function (e.g., "nunique", "sum", "mean"). Defaults to "nunique".
            period (str): Period for cohort analysis (must be "year", "quarter", "month", "week", or "day").
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Raises:
            ValueError: If `period` is not one of the allowed values.
            ValueError: If `df` is missing required columns
                (`customer_id`, `transaction_date`, or `aggregation_column`).
        """
        cols = ColumnHelper()

        ensure_value_choice(period, self.VALID_PERIODS, "period")

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

        period_lookup = {"year": "YS", "quarter": "QS", "month": "MS", "week": "W", "day": "D"}
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
            period (str): Period for cohort analysis (must be "year", "quarter", "month", "week", or "day").
            percentage (bool): If True, converts cohort values into retention percentages relative to the first period.

        Returns:
            pd.DataFrame: Cohort analysis table with user retention values.
        """
        cols = ColumnHelper()

        ibis_table = ibis.memtable(df) if isinstance(df, pd.DataFrame) else df

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

        cohort_table = cohort_table.mutate(
            period_since=cohort_table.period_shopped.delta(cohort_table.min_period_shopped, unit=period),
        )

        cohort_df = cohort_table.execute().drop_duplicates(subset=["min_period_shopped", "period_since"])

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
