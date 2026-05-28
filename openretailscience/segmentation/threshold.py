"""Threshold-Based Segmentation for Flexible Business Classification.

## Business Context

Many retail segmentation needs don't fit standard models like RFM. Businesses often need
custom segments based on specific metrics and thresholds - whether segmenting customers
by spend percentiles, stores by performance quintiles, or products by sales velocity.
This module provides flexible threshold-based segmentation for any business dimension.

## The Business Problem

Retailers need custom segmentation rules for different business scenarios:
- Create spend-based customer tiers (Bronze, Silver, Gold, Platinum)
- Classify stores into performance bands (A, B, C, D stores)
- Segment products by velocity (Fast, Medium, Slow movers)
- Define custom categories based on business-specific thresholds

Standard segmentation approaches are too rigid, while manual classification is inconsistent
and doesn't scale across large datasets.

## Real-World Applications

### Customer Classification
- Create VIP tiers based on total spend percentiles
- Segment by transaction frequency for different service levels
- Classify customers by recency for retention campaigns

### Store Performance Tiers
- Classify stores by sales per square foot into performance bands
- Segment locations by customer conversion rates
- Create store tiers for investment prioritization

### Product Categorization
- Segment SKUs by sales velocity for inventory management
- Classify products by margin contribution for pricing strategies
- Create ABC analysis categories for supply chain optimization

## Technical Features

- Flexible percentile-based thresholds for consistent segment sizing
- Custom aggregation functions for different business metrics
- Configurable handling of zero-value entities
- Efficient execution using Ibis for large datasets
"""

from typing import Literal

import ibis
import pandas as pd

from openretailscience.core.validation import ensure_columns, ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper


class ThresholdSegmentation:
    """Segments customers based on user-defined thresholds and segments."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        thresholds: list[float],
        segments: list[str],
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_segment_name: str = "Zero",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
        group_col: str | list[str] | None = None,
    ) -> None:
        """Segments customers based on user-defined thresholds and segments.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data. The dataframe must contain a
                customer_id column.
            thresholds (List[float]): The percentile thresholds for segmentation.
            segments (List[str]): A list of segment names for each threshold.
            value_col (str, optional): The column to use for the segmentation. Defaults to
                ColumnHelper().unit_spend.
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_segment_name (str, optional): The name of the segment for customers with zero spend.
                Defaults to "Zero".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional): How to handle
                customers with zero spend. Defaults to "separate_segment".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments. When
                specified, segments are calculated within each group independently. For example, setting
                group_col="store_id" calculates Heavy/Medium/Light segments within each store. Defaults to None.

        Raises:
            ValueError: If the dataframe is missing the columns option column.customer_id or `value_col`, or these
                columns contain null values.
        """
        self._df: pd.DataFrame | None = None
        self._group_col: list[str] | None = None
        if len(thresholds) != len(set(thresholds)):
            raise ValueError("The thresholds must be unique")

        if len(thresholds) != len(segments):
            raise ValueError("The number of thresholds must match the number of segments")

        # Initialize column helper
        cols = ColumnHelper()

        df = ensure_ibis_table(df)

        self._group_col = ensure_columns(df, group_col, "group_col") if group_col is not None else None

        value_col = cols.unit_spend if value_col is None else value_col

        # group_col is already validated above; only the function's hard-coded requirements remain.
        ensure_data_has_columns(df, [cols.customer_id, value_col])

        # Build group_by columns: customer_id + optional group columns
        group_by_cols = [cols.customer_id]
        if self._group_col is not None:
            group_by_cols.extend(self._group_col)

        df = df.group_by(*group_by_cols).aggregate(
            **{value_col: getattr(df[value_col], agg_func)()},
        )

        # Separate customers with zero spend
        zero_df = None
        if zero_value_customers == "exclude":
            df = df.filter(df[value_col] != 0)
        elif zero_value_customers == "separate_segment":
            zero_df = df.filter(df[value_col] == 0).mutate(segment_name=ibis.literal(zero_segment_name))
            df = df.filter(df[value_col] != 0)

        # Create window function, partitioned by group columns if specified
        # Order by value_col first, then customer_id to ensure deterministic ordering when values are tied
        window = (
            ibis.window(order_by=[ibis.asc(df[value_col]), ibis.asc(df[cols.customer_id])])
            if self._group_col is None
            else ibis.window(
                order_by=[ibis.asc(df[value_col]), ibis.asc(df[cols.customer_id])],
                group_by=self._group_col,
            )
        )
        df = df.mutate(ptile=ibis.percent_rank().over(window))

        case_args = [(df["ptile"] <= quantile, segment) for quantile, segment in zip(thresholds, segments, strict=True)]

        df = df.mutate(segment_name=ibis.cases(*case_args)).drop(["ptile"])

        if zero_value_customers == "separate_segment":
            df = ibis.union(df, zero_df)

        self.table = df

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names."""
        if self._df is None:
            cols = ColumnHelper()
            index_cols = [cols.customer_id]
            if self._group_col is not None:
                index_cols.extend(self._group_col)
            self._df = self.table.execute().set_index(index_cols)
        return self._df
