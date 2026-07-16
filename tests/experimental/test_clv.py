"""Tests for openretailscience.experimental.clv."""

import datetime

import ibis
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.experimental.clv import CLVStats
from openretailscience.options import ColumnHelper

cols = ColumnHelper()


def _transactions() -> pd.DataFrame:
    """Realistic multi-customer transaction data with a known BTYD summary.

    - Customer 101: three purchase days (two repeats), one basket per day.
    - Customer 102: a single purchase (one-time buyer -> frequency 0).
    - Customer 103: two purchase days, the first day split across two baskets
      (same-day collapse) so the first-day spend is excluded from monetary_value.
    """
    return pd.DataFrame(
        {
            cols.customer_id: [101, 101, 101, 102, 103, 103, 103],
            cols.transaction_date: pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-08",
                    "2023-01-15",
                    "2023-01-01",
                    "2023-01-10",
                    "2023-01-10",
                    "2023-01-24",
                ],
            ),
            cols.unit_spend: [100.0, 50.0, 70.0, 200.0, 30.0, 20.0, 80.0],
        },
    )


class TestCLVStats:
    """Tests for the CLVStats BTYD-summary adapter."""

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    @pytest.mark.parametrize(
        ("period", "observation_period_end", "recency", "expected_t"),
        [
            # Weekly: recency/T are fractional weeks (elapsed days / 7). End given as an ISO string.
            pytest.param("week", "2023-01-29", [14 / 7, 0.0, 14 / 7], [28 / 7, 28 / 7, 19 / 7], id="weekly"),
            # Daily: recency/T are whole days. End given as a datetime.date object.
            pytest.param("day", datetime.date(2023, 1, 29), [14.0, 0.0, 14.0], [28.0, 28.0, 19.0], id="daily"),
            # Default end resolves to the global latest transaction, 2023-01-24 (101/102 age 23, 103 age 14).
            pytest.param("day", None, [14.0, 0.0, 14.0], [23.0, 23.0, 14.0], id="daily-default-obs-end"),
        ],
    )
    def test_summary_matches_hand_computation(self, input_type, period, observation_period_end, recency, expected_t):
        """frequency/recency/T/monetary_value match a hand-computed summary across periods and observation ends."""
        data = self._as_input(_transactions(), input_type)

        result = (
            CLVStats(data, period=period, observation_period_end=observation_period_end)
            .df.sort_values(cols.customer_id)
            .reset_index(drop=True)
        )

        expected = pd.DataFrame(
            {
                cols.customer_id: [101, 102, 103],
                "frequency": np.array([2, 0, 1], dtype="int64"),
                "recency": recency,
                "T": expected_t,
                "monetary_value": [60.0, np.nan, 80.0],
            },
        )
        assert_frame_equal(result, expected)

    def test_missing_column_raises(self):
        """A dataset without unit_spend raises ValueError."""
        data = _transactions().drop(columns=[cols.unit_spend])
        with pytest.raises(ValueError, match="missing required columns"):
            CLVStats(data)

    def test_non_temporal_transaction_date_raises(self):
        """A string transaction_date column is rejected (day math needs a temporal type)."""
        data = _transactions()
        data[cols.transaction_date] = data[cols.transaction_date].astype(str)
        with pytest.raises(TypeError, match="date or datetime"):
            CLVStats(data)

    @pytest.mark.parametrize("bad_period", ["month", "fortnight", "W"])
    def test_invalid_period_raises(self, bad_period):
        """Only the canonical 'day'/'week' period values are accepted."""
        with pytest.raises(ValueError, match="period"):
            CLVStats(_transactions(), period=bad_period)

    def test_observation_period_end_before_last_purchase_raises(self):
        """An observation_period_end earlier than a customer's last purchase is rejected."""
        with pytest.raises(ValueError, match="observation_period_end"):
            CLVStats(_transactions(), observation_period_end="2023-01-10")

    def test_observation_period_end_equal_to_latest_is_allowed(self):
        """An observation_period_end exactly on the latest transaction is accepted (guard is strict <)."""
        # 2023-01-24 is the global latest transaction; an explicit end there must match the default.
        explicit = CLVStats(_transactions(), period="day", observation_period_end="2023-01-24").df
        default = CLVStats(_transactions(), period="day").df
        assert_frame_equal(
            explicit.sort_values(cols.customer_id).reset_index(drop=True),
            default.sort_values(cols.customer_id).reset_index(drop=True),
        )

    def test_observation_period_end_invalid_type_raises(self):
        """A non-date observation_period_end (e.g. an int) is rejected with TypeError."""
        with pytest.raises(TypeError, match="observation_period_end"):
            CLVStats(_transactions(), observation_period_end=20230129)

    @staticmethod
    def _as_input(pdf: pd.DataFrame, input_type: str) -> pd.DataFrame | ibis.Table:
        """Return the frame as a pandas DataFrame or an Ibis memtable."""
        return ibis.memtable(pdf) if input_type == "ibis" else pdf
