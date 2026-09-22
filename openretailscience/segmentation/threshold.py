"""Generic percentile-threshold segmentation with user-defined cut points and segment names.

Base class for `HMLSegmentation`; use this class directly for custom thresholds and segment
names beyond the fixed HML cuts.
"""

import functools
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
        """Segments customers by assigning the first segment whose threshold their percentile rank reaches.

        A customer's aggregated value_col is ranked with percent_rank and assigned the first
        segment whose threshold the rank is <= (percentile cuts, not value cuts). Zero handling
        happens before ranking: "exclude" and "separate_segment" remove zero-value rows from
        the ranked population, while "include_with_light" keeps them in (they typically land
        in the lowest segment; the code does not force a specific segment). Ties break by
        customer_id for deterministic output.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data. The dataframe must contain a
                customer_id column.
            thresholds (list[float]): The percentile thresholds for segmentation; must be unique
                and match the number of segments.
            segments (list[str]): A list of segment names, one per threshold.
            value_col (str, optional): The column to use for the segmentation. Defaults to
                ColumnHelper().unit_spend.
            agg_func (str, optional): The aggregation function to use when grouping by customer_id. Defaults to "sum".
            zero_segment_name (str, optional): The name of the zero-value segment (used with
                "separate_segment"). Defaults to "Zero".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional):
                How to handle zero-value customers. Defaults to "separate_segment".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments. When
                specified, percentiles are computed within each group. Defaults to None.

        Raises:
            ValueError: If the dataframe is missing the customer_id or value_col columns, the
                thresholds are not unique, or the number of thresholds does not match the
                number of segments.
        """
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
