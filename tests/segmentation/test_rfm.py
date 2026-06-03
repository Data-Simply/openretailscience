"""Tests for the RFMSegmentation class."""

import datetime

import ibis
import pandas as pd
import pytest
from freezegun import freeze_time

from openretailscience.options import ColumnHelper, option_context
from openretailscience.segmentation.rfm import RFMSegmentation

cols = ColumnHelper()
MIN_MONETARY = 500.0
MAX_MONETARY = 10000.0
MIN_FREQ = 5
MAX_FREQ = 15
EXPECTED_COUNT_ONE = 3
EXPECTED_COUNT_TWO = 2


class TestRFMSegmentation:
    """Tests for the RFMSegmentation class."""

    @pytest.fixture
    def base_df(self):
        """Return a base DataFrame for testing."""
        return pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [100.0, 200.0, 150.0, 300.0, 250.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-30",
                    "2025-03-10",
                    "2025-02-20",
                ],
            },
        )

    @pytest.fixture
    def expected_df(self):
        """Returns the expected DataFrame for testing segmentation."""
        return pd.DataFrame(
            {
                "customer_id": [1, 2, 3, 4, 5],
                "frequency": [1, 1, 1, 1, 1],
                "monetary": [100.0, 200.0, 150.0, 300.0, 250.0],
                "r_score": [3, 1, 0, 4, 2],
                "f_score": [0, 1, 2, 3, 4],
                "m_score": [0, 2, 1, 4, 3],
                "rfm_segment": [300, 112, 21, 434, 243],
                "fm_segment": [0, 12, 21, 34, 43],
            },
        ).set_index("customer_id")

    @pytest.fixture
    def larger_df(self):
        """Return a larger DataFrame for testing custom segments."""
        return pd.DataFrame(
            {
                cols.customer_id: list(range(1, 21)),
                cols.transaction_id: list(range(101, 121)),
                cols.unit_spend: [100.0 + i * 10 for i in range(20)],
                cols.transaction_date: ["2025-03-01"] * 20,
            },
        )

    @pytest.fixture
    def multi_transaction_df(self):
        """Return a DataFrame with customers having different transaction frequencies."""
        rules = [(1, 100.0), (5, 50.0), (10, 30.0), (20, 25.0)]
        total_rows = sum(rows for rows, _ in rules)
        transaction_ids = iter(range(101, 101 + total_rows))
        data = [
            {
                cols.customer_id: cid,
                cols.transaction_id: next(transaction_ids),
                cols.unit_spend: spend,
                cols.transaction_date: "2025-03-01",
            }
            for cid, (rows, spend) in enumerate(rules, start=1)
            for _ in range(rows)
        ]
        return pd.DataFrame(data)

    @pytest.mark.parametrize("current_date", ["2025-03-17", datetime.date(2025, 3, 17)])
    def test_correct_rfm_segmentation(self, base_df, expected_df, current_date):
        """Test that RFM segmentation calculates scores and segments for string and date inputs.

        Parameterizing current_date covers both the string ("YYYY-MM-DD") and datetime.date
        input paths, which must produce identical results for the same calendar date.
        """
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [16, 30, 46, 7, 25]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    @freeze_time("2025-03-19")
    def test_rfm_segmentation_with_no_date(self, base_df, expected_df):
        """Test that the RFM segmentation works when no current_date is provided."""
        rfm_segmentation = RFMSegmentation(df=base_df)
        result_df = rfm_segmentation.df
        expected_df["recency_days"] = [18, 32, 48, 9, 27]
        expected_df["recency_days"] = expected_df["recency_days"].astype(result_df["recency_days"].dtype)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
        )

    def test_multiple_transactions_per_customer(self):
        """Test that the method correctly handles multiple transactions for the same customer."""
        df_multiple_transactions = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 1, 1],
                cols.transaction_id: [101, 102, 103, 104, 105],
                cols.unit_spend: [120.0, 250.0, 180.0, 300.0, 220.0],
                cols.transaction_date: [
                    "2025-03-01",
                    "2025-02-15",
                    "2025-01-10",
                    "2025-03-10",
                    "2025-02-25",
                ],
            },
        )
        rfm_segmentation = RFMSegmentation(df=df_multiple_transactions, current_date="2025-03-17")
        result_df = rfm_segmentation.df

        # Metrics aggregate across the customer's five transactions: frequency counts unique
        # transactions, monetary sums spend, and recency is measured from the most recent
        # transaction (2025-03-10). A lone customer falls in the bottom bin for every metric.
        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1],
                "frequency": [5],
                "monetary": [1070.0],
                "recency_days": [7],
                "r_score": [0],
                "f_score": [0],
                "m_score": [0],
                "rfm_segment": [0],
                "fm_segment": [0],
            },
        ).set_index(cols.customer_id)

        pd.testing.assert_frame_equal(result_df, expected_df, check_like=True, check_dtype=False)

    def test_ibis_table_input_matches_dataframe_input(self, base_df):
        """Ibis-table input produces the same segmentation as the equivalent DataFrame input."""
        current_date = "2025-03-17"
        dataframe_result = RFMSegmentation(df=base_df, current_date=current_date).df
        ibis_result = RFMSegmentation(df=ibis.memtable(base_df), current_date=current_date).df

        pd.testing.assert_frame_equal(ibis_result.sort_index(), dataframe_result.sort_index())

    def test_df_property_caching(self, base_df):
        """Test that df property caches result and doesn't recompute."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)

        first_df = rfm_segmentation.df
        second_df = rfm_segmentation.df
        assert first_df is second_df

    @pytest.mark.parametrize(
        ("invalid_input", "expected_error", "error_message"),
        [
            ("not_a_dataframe", TypeError, "df must be either a pandas DataFrame or an Ibis Table"),
            (12345, TypeError, "df must be either a pandas DataFrame or an Ibis Table"),
            (None, TypeError, "df must be either a pandas DataFrame or an Ibis Table"),
        ],
    )
    def test_invalid_df_type(self, invalid_input, expected_error, error_message):
        """Test that RFMSegmentation raises appropriate errors for invalid df types."""
        with pytest.raises(expected_error, match=error_message):
            RFMSegmentation(df=invalid_input, current_date="2025-03-17")

    def test_handles_dataframe_with_missing_columns(self):
        """Test error when required columns are missing."""
        incomplete_df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 150.0],
                cols.transaction_id: [101, 102, 103],
            },
        )

        with pytest.raises(ValueError, match="Input data is missing required columns"):
            RFMSegmentation(df=incomplete_df, current_date="2025-03-17")

    @pytest.mark.parametrize(
        ("invalid_date", "expected_error", "error_message"),
        [
            (12345, TypeError, "current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None"),
            ([], TypeError, "current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None"),
            (
                3.14159,
                TypeError,
                "current_date must be a string in 'YYYY-MM-DD' format, a datetime.date object, or None",
            ),
        ],
    )
    def test_invalid_current_date_type(self, base_df, invalid_date, expected_error, error_message):
        """Test that RFMSegmentation raises TypeError for invalid current_date types."""
        with pytest.raises(expected_error, match=error_message):
            RFMSegmentation(base_df, current_date=invalid_date)

    @pytest.mark.parametrize(
        ("r_segments", "f_segments", "m_segments"),
        [
            (5, 3, 4),
            ([0.2, 0.4, 0.6, 0.8], [0.33, 0.66], [0.25, 0.5, 0.75]),
        ],
    )
    def test_custom_segment_counts(self, larger_df, r_segments, f_segments, m_segments):
        """Integer bin counts and equivalent cut points both produce the expected score range.

        Five recency bins, three frequency bins, and four monetary bins yield top scores of
        4, 2, and 3 respectively, with every metric starting at 0.
        """
        max_r, max_f, max_m = 4, 2, 3
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date="2025-03-17",
            r_segments=r_segments,
            f_segments=f_segments,
            m_segments=m_segments,
        )
        result_df = rfm_segmentation.df

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert all(result_df[col].min() == 0 for col in ["r_score", "f_score", "m_score"])

    def test_custom_cut_points_segment_composition(self):
        """Cut-points scoring composes rfm_segment and fm_segment correctly across all metrics.

        Uses customers whose recency, frequency, and monetary rankings differ so that the
        composed segments (r*100 + f*10 + m and f*10 + m) distinguish each coefficient. This
        also pins score direction for all three metrics: higher frequency/monetary and lower
        recency earn the higher score. It is the regression guard for the cut-points branch
        previously ranking recency by ascending order and inverting r_score (most recent
        customers wrongly received the lowest score).
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2, 2, 2, 3, 4, 4, 4],
                cols.transaction_id: list(range(101, 111)),
                cols.unit_spend: [50.0, 50.0, 100.0, 100.0, 100.0, 100.0, 200.0, 100.0, 100.0, 100.0],
                cols.transaction_date: [
                    "2025-03-07",
                    "2025-03-07",
                    "2025-02-25",
                    "2025-02-25",
                    "2025-02-25",
                    "2025-02-25",
                    "2025-02-15",
                    "2025-02-05",
                    "2025-02-05",
                    "2025-02-05",
                ],
            },
        )

        rfm_segmentation = RFMSegmentation(
            df=df,
            current_date="2025-03-17",
            r_segments=[0.5],
            f_segments=[0.5],
            m_segments=[0.5],
        )
        result_df = rfm_segmentation.df

        expected_df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4],
                "frequency": [2, 4, 1, 3],
                "monetary": [100.0, 400.0, 200.0, 300.0],
                "recency_days": [10, 20, 30, 40],
                "r_score": [1, 1, 0, 0],
                "f_score": [0, 1, 0, 1],
                "m_score": [0, 1, 0, 1],
                "rfm_segment": [100, 111, 0, 11],
                "fm_segment": [0, 11, 0, 11],
            },
        ).set_index(cols.customer_id)

        pd.testing.assert_frame_equal(
            result_df.sort_index(),
            expected_df.sort_index(),
            check_like=True,
            check_dtype=False,
        )

    def test_single_bin_segments(self, larger_df):
        """Test using single bins for all segments."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=1,
            f_segments=1,
            m_segments=1,
        )
        result_df = rfm_segmentation.df

        assert (result_df["r_score"] == 0).all()
        assert (result_df["f_score"] == 0).all()
        assert (result_df["m_score"] == 0).all()
        assert (result_df["rfm_segment"] == 0).all()

    @pytest.mark.parametrize(
        ("segment_type", "segment_value", "error_message"),
        [
            ("r_segments", 0, "r_segments must be between 1 and 10 when specified as an integer"),
            ("f_segments", 11, "f_segments must be between 1 and 10 when specified as an integer"),
            ("m_segments", -1, "m_segments must be between 1 and 10 when specified as an integer"),
            ("r_segments", [], "r_segments must contain between 1 and 9 cut points when specified as a list"),
            (
                "f_segments",
                [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
                "f_segments must contain between 1 and 9 cut points when specified as a list",
            ),
            ("r_segments", [0.5, 1.5], "All cut points in r_segments must be between 0 and 1"),
            ("f_segments", [-0.1, 0.5], "All cut points in f_segments must be between 0 and 1"),
            ("m_segments", [0.5, "invalid"], "All cut points in m_segments must be numeric"),
            ("r_segments", [0.3, 0.5, 0.3], "Cut points in r_segments must be unique"),
            ("f_segments", [0.7, 0.3, 0.9], "Cut points in f_segments must be in ascending order"),
            ("m_segments", "invalid", "m_segments must be an integer or a list of floats"),
        ],
    )
    def test_segment_validation(self, base_df, segment_type, segment_value, error_message):
        """Test validation for various segment parameter configurations."""
        kwargs = {segment_type: segment_value}
        with pytest.raises((ValueError, TypeError), match=error_message):
            RFMSegmentation(base_df, **kwargs)

    @pytest.mark.parametrize(
        ("filter_config", "expected_count", "expected_condition"),
        [
            ({"min_monetary": MIN_MONETARY}, 8, lambda df: df["monetary"].min() >= MIN_MONETARY),
            ({"max_monetary": MAX_MONETARY}, 7, lambda df: df["monetary"].max() <= MAX_MONETARY),
            (
                {"min_monetary": MIN_MONETARY, "max_monetary": MAX_MONETARY},
                5,
                lambda df: df["monetary"].min() >= MIN_MONETARY and df["monetary"].max() <= MAX_MONETARY,
            ),
        ],
    )
    def test_monetary_filters(self, filter_config, expected_count, expected_condition):
        """Test monetary filters work correctly."""
        filter_test_df = pd.DataFrame(
            {
                cols.customer_id: list(range(1, 11)),
                cols.transaction_id: list(range(101, 111)),
                cols.unit_spend: [50.0, 100.0, 500.0, 1000.0, 2000.0, 5000.0, 10000.0, 20000.0, 50000.0, 100000.0],
                cols.transaction_date: ["2025-03-01"] * 10,
            },
        )

        rfm_segmentation = RFMSegmentation(df=filter_test_df, current_date="2025-03-17", **filter_config)
        result_df = rfm_segmentation.df

        assert len(result_df) == expected_count
        assert expected_condition(result_df)

    @pytest.mark.parametrize(
        ("filter_config", "expected_count", "expected_condition"),
        [
            ({"min_frequency": MIN_FREQ}, EXPECTED_COUNT_ONE, lambda df: df["frequency"].min() >= MIN_FREQ),
            ({"max_frequency": MAX_FREQ}, EXPECTED_COUNT_ONE, lambda df: df["frequency"].max() <= MAX_FREQ),
            (
                {"min_frequency": MIN_FREQ, "max_frequency": MAX_FREQ},
                EXPECTED_COUNT_TWO,
                lambda df: MIN_FREQ <= df["frequency"].min() <= df["frequency"].max() <= MAX_FREQ,
            ),
        ],
    )
    def test_frequency_filters(self, multi_transaction_df, filter_config, expected_count, expected_condition):
        """Test frequency filters work correctly."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=multi_transaction_df, current_date=current_date, **filter_config)
        result_df = rfm_segmentation.df

        assert len(result_df) == expected_count
        assert expected_condition(result_df)

    @pytest.mark.parametrize(
        ("filter_param", "invalid_value", "expected_error", "error_message"),
        [
            ("min_monetary", "invalid", TypeError, "min_monetary must be a numeric value"),
            ("max_monetary", "invalid", TypeError, "max_monetary must be a numeric value"),
            ("min_frequency", 5.5, TypeError, "min_frequency must be an integer"),
            ("max_frequency", 5.5, TypeError, "max_frequency must be an integer"),
            ("min_monetary", -100.0, ValueError, "min_monetary must be non-negative"),
            ("max_monetary", -100.0, ValueError, "max_monetary must be non-negative"),
            ("min_frequency", 0, ValueError, "min_frequency must be at least 1"),
            ("max_frequency", 0, ValueError, "max_frequency must be at least 1"),
        ],
    )
    def test_filter_validation(self, base_df, filter_param, invalid_value, expected_error, error_message):
        """Test validation for filter parameters."""
        kwargs = {filter_param: invalid_value}
        with pytest.raises(expected_error, match=error_message):
            RFMSegmentation(base_df, **kwargs)

    @pytest.mark.parametrize(
        ("filter_kwargs", "error_message"),
        [
            ({"min_monetary": 500.0, "max_monetary": 500.0}, "min_monetary must be less than max_monetary"),
            ({"min_monetary": 600.0, "max_monetary": 500.0}, "min_monetary must be less than max_monetary"),
            ({"min_frequency": 10, "max_frequency": 5}, "min_frequency must be less than or equal to max_frequency"),
        ],
    )
    def test_filter_relationship_validation(self, base_df, filter_kwargs, error_message):
        """A filter minimum that exceeds its maximum raises a descriptive ValueError."""
        with pytest.raises(ValueError, match=error_message):
            RFMSegmentation(base_df, **filter_kwargs)

    def test_filter_excluding_all_customers_yields_empty_result(self, base_df):
        """A monetary filter above every customer's spend produces an empty segmentation."""
        # base_df spend tops out at 300.0, so a 10000.0 floor excludes all five customers.
        rfm_segmentation = RFMSegmentation(df=base_df, current_date="2025-03-17", min_monetary=10000.0)
        result_df = rfm_segmentation.df

        assert len(result_df) == 0

    def test_filters_applied_before_segmentation(self):
        """Test that filters are applied before calculating segment boundaries.

        This test verifies that segment boundaries (percentiles) are calculated
        based on the filtered dataset, not the original dataset. When filters
        are applied, the same customer should potentially get different scores
        because the percentile calculations are based on a different population.
        """
        filter_test_df = pd.DataFrame(
            {
                cols.customer_id: list(range(1, 21)),
                cols.transaction_id: list(range(101, 121)),
                cols.unit_spend: [
                    100,
                    150,
                    200,
                    250,
                    300,
                    350,
                    400,
                    450,
                    500,
                    550,
                    1000,
                    1500,
                    2000,
                    2500,
                    3000,
                    3500,
                    4000,
                    4500,
                    5000,
                    5500,
                ],
                cols.transaction_date: ["2025-03-01"] * 20,
            },
        )

        current_date = "2025-03-17"

        # Test with 5 segments for clearer boundaries
        rfm_no_filter = RFMSegmentation(
            df=filter_test_df,
            current_date=current_date,
            m_segments=5,
        )

        # Filter to only include high-value customers (>= 1000)
        rfm_with_filter = RFMSegmentation(
            df=filter_test_df,
            current_date=current_date,
            min_monetary=1000.0,
            m_segments=5,
        )

        result_no_filter = rfm_no_filter.df
        result_with_filter = rfm_with_filter.df
        customer_count_no_filter = 20
        customer_count_with_filter = 10

        assert len(result_no_filter) == customer_count_no_filter
        assert len(result_with_filter) == customer_count_with_filter

        # Check that segment boundaries are
        customer_11_no_filter = result_no_filter.loc[11, "m_score"]
        customer_11_with_filter = result_with_filter.loc[11, "m_score"]
        assert customer_11_with_filter < customer_11_no_filter, (
            f"Customer 11 should have lower m_score when filtered. "
            f"No filter: {customer_11_no_filter}, With filter: {customer_11_with_filter}"
        )
        filtered_score_counts = result_with_filter["m_score"].value_counts().sort_index()

        expected_customers_per_segment = len(result_with_filter) // 5
        for score, count in filtered_score_counts.items():
            assert abs(count - expected_customers_per_segment) <= 1, (
                f"Segment {score} has {count} customers, expected ~{expected_customers_per_segment}. "
                f"This suggests boundaries weren't recalculated properly."
            )

    def test_with_custom_column_names(self, base_df):
        """Custom column-name options drive the computation, yielding the default-name result.

        Renaming every input column and pointing the options at the new names must produce the
        same scores and segments as the default-named data (only the index name changes),
        proving the option values are used throughout the pipeline rather than just echoed.
        """
        current_date = "2025-03-17"
        default_result = RFMSegmentation(df=base_df, current_date=current_date).df

        rename_mapping = {
            "customer_id": "custom_cust_id",
            "transaction_id": "custom_txn_id",
            "unit_spend": "custom_spend_amount",
            "transaction_date": "custom_txn_date",
        }
        custom_df = base_df.rename(columns=rename_mapping)

        with option_context(
            "column.customer_id",
            "custom_cust_id",
            "column.transaction_id",
            "custom_txn_id",
            "column.unit_spend",
            "custom_spend_amount",
            "column.transaction_date",
            "custom_txn_date",
        ):
            custom_result = RFMSegmentation(df=custom_df, current_date=current_date).df

        assert custom_result.index.name == "custom_cust_id"
        pd.testing.assert_frame_equal(custom_result.rename_axis(default_result.index.name), default_result)
