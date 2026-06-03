"""Tests for the RFMSegmentation class."""

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

    def test_correct_rfm_segmentation(self, base_df, expected_df):
        """Test that the RFM segmentation correctly calculates the RFM scores and segments."""
        current_date = "2025-03-17"
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

    def test_single_customer(self):
        """Test RFM segmentation for a single customer."""
        df_single_customer = pd.DataFrame(
            {
                cols.customer_id: [1],
                cols.transaction_id: [101],
                cols.unit_spend: [200.0],
                cols.transaction_date: ["2025-03-01"],
            },
        )
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_single_customer, current_date=current_date)
        result_df = rfm_segmentation.df
        assert result_df.loc[1, "rfm_segment"] == 0

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
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=df_multiple_transactions, current_date=current_date)
        result_df = rfm_segmentation.df

        assert result_df.loc[1, "rfm_segment"] == 0

    def test_lazy_execution_with_ibis_table(self, base_df):
        """Test lazy execution of Ibis table on accessing .df."""
        ibis_table = ibis.memtable(base_df)
        rfm = RFMSegmentation(df=ibis_table, current_date="2025-03-17")
        result_df = rfm.df
        assert isinstance(result_df, pd.DataFrame)
        assert not result_df.empty

    def test_df_property_caching(self, base_df):
        """Test that df property caches result and doesn't recompute."""
        current_date = "2025-03-17"
        rfm_segmentation = RFMSegmentation(df=base_df, current_date=current_date)

        first_df = rfm_segmentation.df
        assert first_df is not None
        assert not first_df.empty

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

    def test_custom_bins_with_integers(self, larger_df):
        """Test custom number of bins works correctly."""
        current_date = "2025-03-17"
        max_r, max_f, max_m = 4, 2, 3
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=5,
            f_segments=3,
            m_segments=4,
        )
        result_df = rfm_segmentation.df

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert all(result_df[col].min() == 0 for col in ["r_score", "f_score", "m_score"])

    def test_custom_cut_points(self, larger_df):
        """Test custom cut points work correctly."""
        current_date = "2025-03-17"
        max_r, max_f, max_m = 4, 2, 3
        rfm_segmentation = RFMSegmentation(
            df=larger_df,
            current_date=current_date,
            r_segments=[0.2, 0.4, 0.6, 0.8],
            f_segments=[0.33, 0.66],
            m_segments=[0.25, 0.5, 0.75],
        )
        result_df = rfm_segmentation.df

        assert result_df["r_score"].max() == max_r
        assert result_df["f_score"].max() == max_f
        assert result_df["m_score"].max() == max_m
        assert all(result_df[col].min() == 0 for col in ["r_score", "f_score", "m_score"])

    def test_custom_cut_points_recency_direction(self):
        """Most recent customers must receive the highest r_score under custom cut points.

        Regression test: the cut-points branch previously ranked recency by its natural
        ascending order, ignoring the descending window ordering. This inverted r_score so
        the most recent customers received the lowest score instead of the highest, breaking
        the package's "higher score is better" contract and disagreeing with the integer-bin
        path.
        """
        current_date = "2025-03-17"
        recency_days = [3, 9, 15, 21, 40, 60, 90, 150, 250, 350]
        transaction_dates = [
            (pd.Timestamp(current_date) - pd.Timedelta(days=days)).strftime("%Y-%m-%d") for days in recency_days
        ]
        df = pd.DataFrame(
            {
                cols.customer_id: list(range(1, 11)),
                cols.transaction_id: list(range(101, 111)),
                cols.unit_spend: [100.0] * 10,
                cols.transaction_date: transaction_dates,
            },
        )

        rfm_segmentation = RFMSegmentation(df=df, current_date=current_date, r_segments=[0.5, 0.8])
        result_df = rfm_segmentation.df.sort_values("recency_days")

        # Most recent customer earns the top score; least recent earns the bottom score.
        assert result_df["r_score"].iloc[0] == result_df["r_score"].max()
        assert result_df["r_score"].iloc[-1] == 0
        # r_score is non-increasing as recency grows (more recent => better).
        assert result_df["r_score"].is_monotonic_decreasing

    def test_custom_cut_points_frequency_monetary_direction(self, multi_transaction_df):
        """Higher frequency and monetary values must earn higher scores under custom cut points.

        Companion to the recency-direction regression test: the cut-points branch is shared
        across all three metrics, so frequency and monetary direction are pinned here to guard
        against future ordering regressions in that branch.
        """
        rfm_segmentation = RFMSegmentation(
            df=multi_transaction_df,
            current_date="2025-03-17",
            f_segments=[0.5, 0.8],
            m_segments=[0.5, 0.8],
        )
        result_df = rfm_segmentation.df

        by_frequency = result_df.sort_values("frequency")
        assert by_frequency["f_score"].iloc[0] == 0
        assert by_frequency["f_score"].iloc[-1] == by_frequency["f_score"].max()
        assert by_frequency["f_score"].is_monotonic_increasing

        by_monetary = result_df.sort_values("monetary")
        assert by_monetary["m_score"].iloc[0] == 0
        assert by_monetary["m_score"].iloc[-1] == by_monetary["m_score"].max()
        assert by_monetary["m_score"].is_monotonic_increasing

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
        ("min_val", "max_val", "error_message"),
        [
            (500.0, 500.0, "min_monetary must be less than max_monetary"),
            (600.0, 500.0, "min_monetary must be less than max_monetary"),
        ],
    )
    def test_monetary_filter_relationships(self, base_df, min_val, max_val, error_message):
        """Test validation for monetary filter relationships."""
        with pytest.raises(ValueError, match=error_message):
            RFMSegmentation(base_df, min_monetary=min_val, max_monetary=max_val)

    def test_frequency_filter_relationships(self, base_df):
        """Test validation for frequency filter relationships."""
        with pytest.raises(ValueError, match="min_frequency must be less than or equal to max_frequency"):
            RFMSegmentation(base_df, min_frequency=10, max_frequency=5)

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
        """Test RFMSegmentation with custom column names."""
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
            rfm_segmentation = RFMSegmentation(df=custom_df)
            result_df = rfm_segmentation.df
            assert isinstance(result_df, pd.DataFrame), "Should execute successfully with custom column names"
            assert result_df.index.name == "custom_cust_id", "Should use custom customer_id column name as index"
