"""Customer Segmentation Using RFM Analysis.

This module implements RFM (Recency, Frequency, Monetary) segmentation, a widely used technique in customer analytics
to categorize customers based on their purchasing behavior.

RFM segmentation assigns scores to customers based on:
1. Recency (R): How recently a customer made a purchase.
2. Frequency (F): How often a customer makes purchases.
3. Monetary (M): The total amount spent by a customer.

### Benefits of RFM Segmentation:
- **Customer Value Analysis**: Identifies high-value customers who contribute the most revenue.
- **Personalized Marketing**: Enables targeted campaigns based on customer purchasing behavior.
- **Customer Retention Strategies**: Helps recognize at-risk customers and develop engagement strategies.
- **Sales Forecasting**: Provides insights into future revenue trends based on past spending behavior.

### Scoring Methodology:
- Each metric (R, F, M) is divided into 10 bins (0-9) using the NTILE(10) function.
- A higher score indicates a better customer (e.g., lower recency, higher frequency, and monetary value).
- The final RFM segment is computed as `R*100 + F*10 + M`, providing a unique customer classification.

This module leverages `pandas` and `ibis` for efficient data processing and integrates with retail analytics workflows
to enhance customer insights and business decision-making.
"""

import datetime

import ibis
import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper, get_option


class RFMSegmentation:
    """Segments customers using the RFM (Recency, Frequency, Monetary) methodology.

    Customers are scored on three dimensions:
    - Recency (R): Days since the last transaction (lower is better).
    - Frequency (F): Number of unique transactions (higher is better).
    - Monetary (M): Total amount spent (higher is better).

    Each metric is ranked into bins using either NTILE or custom cut points where,
    - The highest score represents the best score (top percentile of customers).
    - The lowest score represents the lowest score (bottom percentile of customers).
    The RFM segment is a 3-digit number (R*100 + F*10 + M), representing customer value.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        current_date: str | datetime.date | None = None,
        r_segments: int | list[float] = 10,
        f_segments: int | list[float] = 10,
        m_segments: int | list[float] = 10,
        min_monetary: float | None = None,
        max_monetary: float | None = None,
        min_frequency: int | None = None,
        max_frequency: int | None = None,
    ) -> None:
        """Initializes the RFM segmentation process.

        Args:
            df (pd.DataFrame | ibis.Table): A DataFrame or Ibis table containing transaction data.
                Must include the following columns:
                - customer_id
                - transaction_date
                - unit_spend
                - transaction_id
            current_date (Optional[Union[str, datetime.date]]): The reference date for calculating recency.
                Can be a string (format: "YYYY-MM-DD"), a date object, or None (defaults to the current system date).
            r_segments (Union[int, list[float]], optional): Number of bins (1-10) or custom percentile
                cut points (max 9 cut points). Defaults to 10 bins.
            f_segments (Union[int, list[float]], optional): Number of bins (1-10) or custom percentile
                cut points (max 9 cut points). Defaults to 10 bins.
            m_segments (Union[int, list[float]], optional): Number of bins (1-10) or custom percentile
                cut points (max 9 cut points). Defaults to 10 bins.
            min_monetary (Optional[float]): Minimum monetary value to include in segmentation.
                Customers with total spend below this value will be excluded from the analysis.
            max_monetary (Optional[float]): Maximum monetary value to include in segmentation.
                Customers with total spend above this value will be excluded from the analysis.
            min_frequency (Optional[int]): Minimum purchase frequency to include in segmentation.
                Customers with fewer transactions will be excluded from the analysis.
            max_frequency (Optional[int]): Maximum purchase frequency to include in segmentation.
                Customers with more transactions will be excluded from the analysis.

        Raises:
            ValueError: If the dataframe is missing required columns, invalid segment parameters,
                       or invalid filter parameters.
            TypeError: If the input data is not a pandas DataFrame or an Ibis Table.
        """
        self._df: pd.DataFrame | None = None
        cols = ColumnHelper()
        required_cols = [
            cols.customer_id,
            cols.transaction_date,
            cols.unit_spend,
            cols.transaction_id,
        ]
        df = ensure_ibis_table(df)

        ensure_data_has_columns(df, required_cols)

        if isinstance(current_date, str):
            current_date = datetime.date.fromisoformat(current_date)
        elif current_date is None:
            current_date = datetime.datetime.now(datetime.timezone.utc).date()
        elif not isinstance(current_date, datetime.date):
            raise TypeError("current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None")

        self._validate_segments(r_segments, "r_segments")
        self._validate_segments(f_segments, "f_segments")
        self._validate_segments(m_segments, "m_segments")
        self._validate_monetary_filters(min_monetary, max_monetary)
        self._validate_frequency_filters(min_frequency, max_frequency)

        self.r_segments = r_segments
        self.f_segments = f_segments
        self.m_segments = m_segments
        self.min_monetary = min_monetary
        self.max_monetary = max_monetary
        self.min_frequency = min_frequency
        self.max_frequency = max_frequency

        self.table = self._compute_rfm(df, current_date)

    def _validate_segments(self, segments: int | list[float], param_name: str) -> None:
        """Validates segment parameters.

        Args:
            segments: The segment parameter to validate
            param_name: Name of the parameter for error messages

        Raises:
            ValueError: If segment parameters are invalid
        """
        max_segments_int = 10
        max_segments_list = 9
        max_segments = 1

        if isinstance(segments, int):
            if segments < max_segments or segments > max_segments_int:
                msg = f"{param_name} must be between {max_segments} and {max_segments_int} when specified as an integer"
                raise ValueError(msg)
        elif isinstance(segments, list):
            if len(segments) == 0 or len(segments) > max_segments_list:
                msg = (
                    f"{param_name} must contain between {max_segments} and"
                    f" {max_segments_list} cut points when specified as a list"
                )
                raise ValueError(msg)
            if not all(isinstance(x, int | float) for x in segments):  # UP038
                msg = f"All cut points in {param_name} must be numeric"
                raise ValueError(msg)
            if not all(0 <= x <= 1 for x in segments):
                msg = f"All cut points in {param_name} must be between 0 and 1"
                raise ValueError(msg)
            if len(segments) != len(set(segments)):
                msg = f"Cut points in {param_name} must be unique"
                raise ValueError(msg)
            if segments != sorted(segments):
                msg = f"Cut points in {param_name} must be in ascending order"
                raise ValueError(msg)
        else:
            msg = f"{param_name} must be an integer or a list of floats"
            raise TypeError(msg)

    def _validate_monetary_filters(self, min_monetary: float | None, max_monetary: float | None) -> None:
        if min_monetary is not None:
            if not isinstance(min_monetary, int | float):
                raise TypeError("min_monetary must be a numeric value")
            if min_monetary < 0:
                raise ValueError("min_monetary must be non-negative")

        if max_monetary is not None:
            if not isinstance(max_monetary, int | float):
                raise TypeError("max_monetary must be a numeric value")
            if max_monetary < 0:
                raise ValueError("max_monetary must be non-negative")

        if min_monetary is not None and max_monetary is not None and min_monetary >= max_monetary:
            raise ValueError("min_monetary must be less than max_monetary")

    def _validate_frequency_filters(self, min_frequency: float | None, max_frequency: float | None) -> None:
        if min_frequency is not None:
            if not isinstance(min_frequency, int):
                raise TypeError("min_frequency must be an integer")
            if min_frequency < 1:
                raise ValueError("min_frequency must be at least 1")

        if max_frequency is not None:
            if not isinstance(max_frequency, int):
                raise TypeError("max_frequency must be an integer")
            if max_frequency < 1:
                raise ValueError("max_frequency must be at least 1")

        if min_frequency is not None and max_frequency is not None and min_frequency > max_frequency:
            raise ValueError("min_frequency must be less than or equal to max_frequency")

    def _compute_rfm(self, df: ibis.Table, current_date: datetime.date) -> ibis.Table:
        """Computes the RFM metrics and segments customers accordingly.

        Args:
            df (ibis.Table): The transaction data table.
            current_date (datetime.date): The reference date for calculating recency.

        Returns:
            ibis.Table: A table with RFM scores and segment values.
        """
        cols = ColumnHelper()
        current_date_expr = ibis.literal(current_date)

        customer_metrics = df.group_by(cols.customer_id).aggregate(
            recency_days=current_date_expr.delta(df[cols.transaction_date].max().cast("date"), unit="day").cast(
                "int32",
            ),
            frequency=df[cols.transaction_id].nunique(),
            monetary=df[cols.unit_spend].sum(),
        )

        filtered_metrics = self._apply_filters(customer_metrics)

        rfm_scores = filtered_metrics.mutate(
            r_score=self._compute_score(filtered_metrics, "recency_days", self.r_segments, ascending=False),
            f_score=self._compute_score(filtered_metrics, "frequency", self.f_segments, ascending=True),
            m_score=self._compute_score(filtered_metrics, "monetary", self.m_segments, ascending=True),
        )

        return rfm_scores.mutate(
            rfm_segment=(rfm_scores.r_score * 100 + rfm_scores.f_score * 10 + rfm_scores.m_score),
            fm_segment=(rfm_scores.f_score * 10 + rfm_scores.m_score),
        )

    def _apply_filters(self, customer_metrics: ibis.Table) -> ibis.Table:
        """Applies the specified filters to the customer metrics.

        Args:
            customer_metrics: Table with customer metrics (recency_days, frequency, monetary)

        Returns:
            Filtered table containing only customers meeting all filter criteria
        """
        filter_configs = [
            ("monetary", self.min_monetary, self.max_monetary),
            ("frequency", self.min_frequency, self.max_frequency),
        ]

        filtered_table = customer_metrics

        for column_name, min_val, max_val in filter_configs:
            if min_val is not None:
                filtered_table = filtered_table.filter(filtered_table[column_name] >= min_val)

            if max_val is not None:
                filtered_table = filtered_table.filter(filtered_table[column_name] <= max_val)

        return filtered_table

    def _compute_score(
        self,
        table: ibis.Table,
        column: str,
        segments: int | list[float],
        ascending: bool = True,
    ) -> ibis.expr.types.IntegerColumn:
        """Computes score for a given column using either NTILE or custom cut points.

        Args:
            table: The table containing the data
            column: The column name to compute scores for
            segments: Either number of bins or list of cut points
            ascending: Whether lower values should get higher scores (for recency=False, for frequency/monetary=True)

        Returns:
            An Ibis expression representing the computed scores
        """
        order_fn = ibis.asc if ascending else ibis.desc
        window = ibis.window(
            order_by=[order_fn(table[column]), ibis.asc(table[ColumnHelper().customer_id])],
        )

        if isinstance(segments, int):
            return ibis.ntile(segments).over(window)

        percentile = table[column].percent_rank().over(window)

        sorted_segments = sorted(segments)
        case_expr = ibis.literal(0).cast("int32")

        for i, cutpoint in enumerate(sorted_segments):
            condition = percentile > ibis.literal(cutpoint)
            case_expr = condition.ifelse(
                ibis.literal(i + 1).cast("int32"),
                case_expr,
            )

        return case_expr

    @property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names."""
        if self._df is None:
            self._df = self.table.execute().set_index(get_option("column.customer_id"))
        return self._df
