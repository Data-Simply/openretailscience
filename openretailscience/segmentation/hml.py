"""Heavy/Medium/Light/Zero (HML) customer segmentation at fixed 50/80/100 spend percentile cuts.

A `ThresholdSegmentation` subclass: Light up to the 50th percentile, Medium 50-80th, Heavy
above the 80th; zero-spend customers form a separate "Zero" segment by default. Use
`ThresholdSegmentation` for custom cut points and segment names.
"""

from typing import Literal

import ibis
import pandas as pd

from openretailscience.segmentation.threshold import ThresholdSegmentation


class HMLSegmentation(ThresholdSegmentation):
    """Segments customers into Heavy, Medium, Light and Zero spenders based on the total spend."""

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        value_col: str | None = None,
        agg_func: str = "sum",
        zero_value_customers: Literal["separate_segment", "exclude", "include_with_light"] = "separate_segment",
        group_col: str | list[str] | None = None,
    ) -> None:
        """Segments customers into Heavy, Medium, Light and Zero spenders at fixed percentile cuts.

        Light covers spend up to the 50th percentile, Medium the 50-80th, and Heavy above the
        80th; zero-spend customers form a separate "Zero" segment by default.

        Args:
            df (pd.DataFrame | ibis.Table): A dataframe with the transaction data.
                The dataframe must contain a customer_id column.
            value_col (str, optional): The column to aggregate for the segmentation. Any
                spend-like column works (e.g. transaction_id with agg_func="count"). Defaults
                to get_option("column.unit_spend").
            agg_func (str, optional): The aggregation function to use when grouping by customer_id.
                Defaults to "sum".
            zero_value_customers (Literal["separate_segment", "exclude", "include_with_light"], optional):
                How to handle customers with zero spend. Defaults to "separate_segment".
            group_col (str | list[str] | None, optional): Column(s) to group by when calculating segments. When
                specified, percentiles are computed within each group. For example, group_col="store_id"
                computes the segments within each store. Defaults to None.
        """
        thresholds = [0.500, 0.800, 1]
        segments = ["Light", "Medium", "Heavy"]
        super().__init__(
            df=df,
            value_col=value_col,
            agg_func=agg_func,
            thresholds=thresholds,
            segments=segments,
            zero_value_customers=zero_value_customers,
            group_col=group_col,
        )
