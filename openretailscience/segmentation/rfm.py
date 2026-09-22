"""Customer segmentation via RFM (Recency, Frequency, Monetary) scoring.

Each dimension is binned with NTILE (default 10) or custom percentile cut points (score
0..n for n cut points); a higher score marks a better customer, with recency inverted so
fewer days since the last purchase score higher. The composite segment is R*100 + F*10 + M.
"""

import datetime
import functools

import ibis
import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import ColumnHelper, get_option


class RFMSegmentation:
    """Segments customers using the RFM (Recency, Frequency, Monetary) methodology.

    Customers are scored on three dimensions:
    - Recency (R): days since the last transaction (lower days is better).
    - Frequency (F): number of unique transactions (higher is better).
    - Monetary (M): total amount spent (higher is better).

    Each metric is binned via NTILE or custom cut points; the highest score marks the top
    percentile of customers. The composite `rfm_segment` is R*100 + F*10 + M; with 0-based
    scores (custom cut points, or NTILE on the pandas backend) it is a 3-digit number 0-999.
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
        """Computes RFM scores and the composite segment for each customer.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data. Must contain the customer_id,
                transaction_date, unit_spend and transaction_id columns.
            current_date (str | datetime.date | None): Reference date for recency, as a
                "YYYY-MM-DD" string, a date, or None (today, UTC).
            r_segments (int | list[float]): Number of NTILE bins (1-10) or 1-9 custom percentile
                cut points in (0, 1), unique and ascending.
            f_segments (int | list[float]): As r_segments, for frequency.
            m_segments (int | list[float]): As r_segments, for monetary.
            min_monetary (float | None): Minimum total spend to include a customer.
            max_monetary (float | None): Maximum total spend to include a customer.
            min_frequency (int | None): Minimum number of transactions to include a customer.
            max_frequency (int | None): Maximum number of transactions to include a customer.
                Filters drop customers outside the min/max ranges before scoring.

        Raises:
            ValueError: If required columns are missing, a segment parameter is invalid
                (out of range, non-numeric, duplicate or unsorted cut points), or a filter
                bound is out of range.
            TypeError: If df is not a DataFrame or ibis Table, current_date has an invalid
                type, a segment is neither an int nor a list, or a filter bound has the
                wrong type.
        """
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

        Accepted shapes: an int in 1-10 (NTILE bin count), or a list of 1-9 unique ascending
        floats in (0, 1) (percentile cut points).

        Args:
            segments (int | list[float]): The segment parameter to validate.
            param_name (str): Name of the parameter for error messages.

        Raises:
            ValueError: If segment parameters are invalid.
            TypeError: If segments is neither an int nor a list.
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
            if not all(0 < x < 1 for x in segments):
                msg = f"All cut points in {param_name} must be between 0 and 1 (exclusive)"
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
        """Validates min/max monetary filters: numeric, non-negative, and min < max.

        Raises:
            TypeError: If a bound is not numeric.
            ValueError: If a bound is negative, or min_monetary >= max_monetary.
        """
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
        """Validates min/max frequency filters: integers >= 1, and min <= max.

        Raises:
            TypeError: If a bound is not an integer.
            ValueError: If a bound is below 1, or min_frequency > max_frequency.
        """
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
        """Computes the RFM metrics and scores customers accordingly.

        Args:
            df (ibis.Table): The transaction data table.
            current_date (datetime.date): The reference date for calculating recency.

        Returns:
            ibis.Table: One row per customer with the raw recency_days, frequency and
                monetary metrics, the r/f/m scores, `rfm_segment` (R*100 + F*10 + M) and
                `fm_segment` (F*10 + M).
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

        Bounds are inclusive: monetary >= min_monetary and <= max_monetary; frequency >=
        min_frequency and <= max_frequency.

        Args:
            customer_metrics (ibis.Table): Table with customer metrics (recency_days, frequency, monetary).

        Returns:
            ibis.Table: Filtered table containing only customers meeting all filter criteria.
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

        `ascending` orders the scoring window by value ascending: True for frequency/monetary
        so larger values score higher, False for recency so fewer days score higher (NTILE
        assigns its bins in window order). Ties break deterministically by customer_id. Scores
        are cast to int64 so both branches share a dtype.

        Args:
            table (ibis.Table): The table containing the data.
            column (str): The column name to compute scores for.
            segments (int | list[float]): Either number of bins or list of cut points.
            ascending (bool): Order the scoring window by value ascending.

        Returns:
            ibis.expr.types.IntegerColumn: An ibis expression of the computed scores.
        """
        order_fn = ibis.asc if ascending else ibis.desc
        window = ibis.window(
            order_by=[order_fn(table[column]), ibis.asc(table[ColumnHelper().customer_id])],
        )

        if isinstance(segments, int):
            return ibis.ntile(segments).over(window)

        percentile = ibis.percent_rank().over(window)

        sorted_segments = sorted(segments)
        # Cast to int64 to match the ntile branch so _compute_score returns a consistent dtype.
        case_expr = ibis.literal(0).cast("int64")

        for i, cutpoint in enumerate(sorted_segments):
            condition = percentile > ibis.literal(cutpoint)
            case_expr = condition.ifelse(
                ibis.literal(i + 1).cast("int64"),
                case_expr,
            )

        return case_expr

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the dataframe with the segment names, indexed by customer_id."""
        return self.table.execute().set_index(get_option("column.customer_id"))
