"""Generic percentile-threshold segmentation with user-defined cut points and segment names.

Base class for `HMLSegmentation`; use this class directly for custom thresholds and segment
names beyond the fixed HML cuts.
"""

import functools
from itertools import pairwise
from typing import Literal

import ibis
import pandas as pd

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_value_choice,
)
from openretailscience.options import ColumnHelper


class ThresholdSegmentation:
    """Segments customers based on user-defined thresholds and segments.

    Examples:
        >>> from openretailscience.segmentation import ThresholdSegmentation
        >>> quartiles = ThresholdSegmentation(
        ...     df=transactions, thresholds=[0.25, 0.5, 0.75, 1.0], segments=["Q1_Bottom", "Q2", "Q3", "Q4_Top"]
        ... )
        >>> quartiles.df
                     unit_spend segment_name
        customer_id
        83                 0.85    Q1_Bottom
        9                  2.10    Q1_Bottom
        ...
    """

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
        """Segments customers by assigning the first segment whose threshold their percentile rank does not exceed.

        Ranks run from 0 (lowest aggregated value_col) to 1 (highest) among the customers in each
        group, so thresholds cut the customer count, not the value range. Equal values still get
        distinct ranks, ordered by customer_id, so customers with the same value can land in
        different segments.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data. The dataframe must contain a
                customer_id column.
            thresholds (list[float]): The percentile thresholds for segmentation, one per segment;
                strictly ascending in (0, 1] and ending at 1.0 so every customer gets a segment.
            segments (list[str]): A list of segment names, one per threshold.
            value_col (str, optional): The column to use for the segmentation. Defaults to
                ColumnHelper().unit_spend.
            agg_func (str, optional): Ibis aggregation name applied per customer, case-sensitive
                (e.g. "sum", "mean", "count", "nunique", "max"). Defaults to "sum".
            zero_segment_name (str, optional): The name of the zero-value segment (used with
                "separate_segment"). Defaults to "Zero".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional):
                What to do with customers whose aggregated value is 0 (case-insensitive):
                "separate_segment" labels them ``zero_segment_name`` and leaves them out of the
                ranking, "exclude" drops them from the result, and "include_with_light" ranks them
                with everyone else; as tied values, enough of them can spill past the first segment.
                Defaults to "separate_segment".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments. When
                specified, percentiles are computed within each group. Defaults to None.

        Raises:
            ValueError: If the dataframe is missing the customer_id or value_col columns, the
                thresholds break the contract above or do not match the number of segments, or
                zero_value_customers is not one of its allowed values.
        """
        self._group_col: list[str] | None = None
        if len(thresholds) != len(segments):
            raise ValueError("The number of thresholds must match the number of segments")

        if any(lower >= upper for lower, upper in pairwise(thresholds)):
            msg = f"thresholds must be strictly ascending. Got {thresholds}."
            raise ValueError(msg)

        if len(thresholds) == 0 or thresholds[0] <= 0 or thresholds[-1] != 1.0:
            msg = f"thresholds must lie in (0, 1] and end at 1.0. Got {thresholds}."
            raise ValueError(msg)

        zero_value_customers = ensure_value_choice(
            zero_value_customers,
            ["separate_segment", "exclude", "include_with_light"],
            "zero_value_customers",
        )

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

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names.

        Indexed by customer_id (and group_col if specified); columns: the aggregated value_col
        and segment_name.
        """
        cols = ColumnHelper()
        index_cols = [cols.customer_id]
        if self._group_col is not None:
            index_cols.extend(self._group_col)
        return self.table.execute().set_index(index_cols)
