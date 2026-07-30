"""Tests for openretailscience.experimental.clv."""

import datetime
import warnings

import ibis
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.experimental.clv import _ONE_HOT_CARDINALITY_WARN, CLVStats
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()

# The base BTYD output columns, in order. All five are pymc-marketing's fixed literals — the id is
# always emitted as "customer_id", not the configured option — so they are not ColumnHelper-resolved.
BASE_BTYD_COLS = ["customer_id", "frequency", "recency", "T", "monetary_value"]

# A "month" period expresses recency/T as elapsed days / this fixed average-Gregorian-month length.
DAYS_PER_MONTH = CLVStats._DAYS_PER_MONTH


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
            # Monthly: recency/T are elapsed days divided by the fixed average-Gregorian-month length.
            pytest.param(
                "month",
                "2023-01-29",
                [14 / DAYS_PER_MONTH, 0.0, 14 / DAYS_PER_MONTH],
                [28 / DAYS_PER_MONTH, 28 / DAYS_PER_MONTH, 19 / DAYS_PER_MONTH],
                id="monthly",
            ),
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

    def test_null_transaction_date_rows_are_dropped(self):
        """Rows with a NULL transaction_date are dropped.

        A partially-undated customer keeps only its real occasions; a fully-undated customer drops out.
        """
        data = pd.DataFrame(
            {
                cols.customer_id: [500, 500, 500, 600, 600],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-08", None, None, None]),
                cols.unit_spend: [10.0, 20.0, 5.0, 30.0, 40.0],
            },
        )
        result = (
            CLVStats(data, period="day", observation_period_end="2023-01-31")
            .df.sort_values(cols.customer_id)
            .reset_index(drop=True)
        )
        expected = pd.DataFrame(
            {
                cols.customer_id: [500],  # 600 (all-undated) drops out entirely
                "frequency": np.array([1], dtype="int64"),  # Jan1 -> Jan8; the undated row is not an extra occasion
                "recency": [7.0],
                "T": [30.0],  # Jan1 -> Jan31
                "monetary_value": [20.0],  # repeat spend is Jan8 only, undiluted by the undated row
            },
        )
        assert_frame_equal(result, expected)

    def test_net_nonpositive_days_are_not_occasions(self):
        """A day with net spend <= 0 is not an occasion; a customer with no positive-spend day drops out."""
        data = pd.DataFrame(
            {
                cols.customer_id: [801, 801, 801, 802, 802, 802, 803, 803],
                cols.transaction_date: pd.to_datetime(
                    [
                        "2023-01-01",  # 801: purchase
                        "2023-01-08",  # 801: net-negative return day -> dropped
                        "2023-01-15",  # 801: purchase
                        "2023-01-01",  # 802: purchase
                        "2023-01-10",  # 802: buy 10 ...
                        "2023-01-10",  # 802: ... and fully return it same day -> net zero, dropped
                        "2023-01-01",  # 803: return only
                        "2023-01-05",  # 803: return only
                    ],
                ),
                cols.unit_spend: [100.0, -20.0, 60.0, 50.0, 10.0, -10.0, -30.0, -10.0],
            },
        )
        result = (
            CLVStats(data, period="day", observation_period_end="2023-01-31")
            .df.sort_values(cols.customer_id)
            .reset_index(drop=True)
        )
        expected = pd.DataFrame(
            {
                cols.customer_id: [801, 802],  # 803 (all returns, no positive day) drops out entirely
                "frequency": np.array([1, 0], dtype="int64"),  # 801's return day is not a repeat occasion
                # 801 Jan1 -> Jan15 (the Jan8 return does not become last_day); 802 is a one-time buyer
                "recency": [14.0, 0.0],
                "T": [30.0, 30.0],
                "monetary_value": [60.0, np.nan],  # 801's repeat spend is Jan15 only, undiluted by the return
            },
        )
        assert_frame_equal(result, expected)

    @pytest.mark.parametrize("bad_period", ["fortnight", "year", "quarter"])
    def test_invalid_period_raises(self, bad_period):
        """Periods outside day/week/month are rejected (year/quarter are real words but unsupported here)."""
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

    def test_repeat_buyers_keeps_only_repeat_buyers(self):
        """repeat_buyers is the GammaGamma-ready subset: the repeat buyers (frequency > 0).

        One-time buyers (frequency 0) are dropped. Customer 104's only repeat day nets negative
        (unit_spend is a margin), so that day is not an occasion and 104 collapses to a one-time
        buyer -- also dropped.
        """
        data = pd.DataFrame(
            {
                cols.customer_id: [101, 101, 102, 103, 103, 104, 104],
                cols.transaction_date: pd.to_datetime(
                    ["2023-01-01", "2023-01-15", "2023-01-01", "2023-01-01", "2023-01-15", "2023-01-01", "2023-01-15"],
                ),
                cols.unit_spend: [50.0, 40.0, 200.0, 30.0, 80.0, 20.0, -5.0],
            },
        )
        result = CLVStats(data, period="day").repeat_buyers.sort_values(cols.customer_id).reset_index(drop=True)

        expected = pd.DataFrame(
            {
                cols.customer_id: [101, 103],  # 102 (frequency 0) and 104 (collapses to frequency 0) dropped
                "frequency": np.array([1, 1], dtype="int64"),
                "recency": [14.0, 14.0],
                "T": [14.0, 14.0],
                "monetary_value": [40.0, 80.0],
            },
        )
        assert_frame_equal(result, expected)

    @staticmethod
    def _as_input(pdf: pd.DataFrame, input_type: str) -> pd.DataFrame | ibis.Table:
        """Return the frame as a pandas DataFrame or an Ibis memtable."""
        return ibis.memtable(pdf) if input_type == "ibis" else pdf


class TestCLVStatsCorrelationWarning:
    """repeat_buyers warns when frequency and monetary_value break GammaGamma's independence assumption."""

    @staticmethod
    def _four_customers(unit_spend: list[float]) -> pd.DataFrame:
        """Four repeat buyers with frequency 1/2/3/4; ``unit_spend`` sets each customer's per-day spend."""
        return pd.DataFrame(
            {
                cols.customer_id: [201, 201, 202, 202, 202, 203, 203, 203, 203, 204, 204, 204, 204, 204],
                cols.transaction_date: pd.to_datetime(
                    [
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-01",
                        "2023-01-02",
                        "2023-01-03",
                        "2023-01-04",
                        "2023-01-05",
                    ],
                ),
                cols.unit_spend: unit_spend,
            },
        )

    @pytest.mark.parametrize(
        ("unit_spend", "expect_warn"),
        [
            # frequency [1,2,3,4] rising lockstep with monetary [10,20,30,40] -> Pearson r = 1.0.
            pytest.param(
                [100.0, 10.0, 100.0, 20.0, 20.0, 100.0, 30.0, 30.0, 30.0, 100.0, 40.0, 40.0, 40.0, 40.0],
                True,
                id="correlated",
            ),
            # frequency [1,2,3,4] against monetary [20,40,10,30] -> Pearson r = 0.0.
            pytest.param(
                [100.0, 20.0, 100.0, 40.0, 40.0, 100.0, 10.0, 10.0, 10.0, 100.0, 30.0, 30.0, 30.0, 30.0],
                False,
                id="uncorrelated",
            ),
        ],
    )
    def test_repeat_buyers_warns_iff_frequency_correlates_with_monetary(self, unit_spend, expect_warn):
        """repeat_buyers warns exactly when |Pearson r(frequency, monetary_value)| exceeds the threshold."""
        data = self._four_customers(unit_spend)
        if expect_warn:
            with pytest.warns(UserWarning, match="frequency and monetary_value are correlated"):
                _ = CLVStats(data, period="day").repeat_buyers
        else:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                result = CLVStats(data, period="day").repeat_buyers
            assert not any("correlated" in str(w.message) for w in caught)
            assert len(result) == data[cols.customer_id].nunique()  # all four are repeat buyers, none dropped


class TestCLVStatsPymcTimeUnit:
    """CLVStats exposes the pymc-marketing time_unit matching its period."""

    @pytest.mark.parametrize(
        ("period", "canonical", "time_unit"),
        [
            ("day", "day", "D"),
            ("week", "week", "W"),
            ("month", "month", "M"),
            ("d", "day", "D"),
            ("M", "month", "M"),
        ],
    )
    def test_period_is_canonical_and_maps_to_pymc_time_unit(self, period, canonical, time_unit):
        """The resolved period is stored canonically and maps to pymc-marketing's D/W/M time_unit."""
        clv = CLVStats(_transactions(), period=period)
        assert clv.period == canonical
        assert clv.pymc_time_unit == time_unit


def _dim_transactions() -> pd.DataFrame:
    """Transactions for four customers (201-204) with distinct repeat-purchase histories."""
    return pd.DataFrame(
        {
            cols.customer_id: [201, 201, 201, 202, 203, 203, 204, 204],
            cols.transaction_date: pd.to_datetime(
                [
                    "2023-01-01",
                    "2023-01-08",
                    "2023-01-15",
                    "2023-01-01",
                    "2023-01-10",
                    "2023-01-24",
                    "2023-01-05",
                    "2023-01-19",
                ],
            ),
            cols.unit_spend: [100.0, 50.0, 70.0, 200.0, 30.0, 80.0, 40.0, 60.0],
        },
    )


def _customer_attributes() -> pd.DataFrame:
    """One row per customer (201-204): distinct stores shopped and signup channel.

    The kind of table a caller builds upstream (e.g. SegTransactionStats grouped on customer_id).
    203's channel is NULL (never signed up through a tracked channel), so it becomes the dropped
    one-hot reference level.
    """
    return pd.DataFrame(
        {
            cols.customer_id: [201, 202, 203, 204],
            "stores_shopped": np.array([2, 1, 1, 2], dtype="int64"),
            "signup_channel": ["email", "referral", None, "search"],
        },
    )


class TestCLVStatsCustomerAttributes:
    """Tests for attaching a caller-supplied per-customer attributes table (join + one-hot)."""

    @staticmethod
    def _summary(**kwargs) -> pd.DataFrame:
        """Build the CLVStats summary from the dim transactions, sorted by customer_id."""
        return (
            CLVStats(_dim_transactions(), period="day", **kwargs)
            .df.sort_values(cols.customer_id)
            .reset_index(drop=True)
        )

    @pytest.mark.parametrize("attr_type", ["pandas", "ibis"])
    def test_customer_attributes_joined_as_is(self, attr_type):
        """A per-customer attribute column joins onto the summary unchanged (pandas or ibis input)."""
        attributes = _customer_attributes()[[cols.customer_id, "stores_shopped"]]
        attributes = ibis.memtable(attributes) if attr_type == "ibis" else attributes
        result = self._summary(customer_attributes=attributes)
        expected = pd.DataFrame(
            {cols.customer_id: [201, 202, 203, 204], "stores_shopped": np.array([2, 1, 1, 2], dtype="int64")},
        )
        assert_frame_equal(result[[cols.customer_id, "stores_shopped"]], expected)
        # Exact schema: base BTYD columns then the attribute, with no stray join-key column.
        assert list(result.columns) == [*BASE_BTYD_COLS, "stores_shopped"]

    def test_customer_attributes_preserves_btyd_columns(self):
        """Attaching attributes does not disturb the frequency/monetary_value computation."""
        base = self._summary()
        # one-hot signup_channel so its NULL (203) is the reference level, not a NULL covariate that raises.
        with_attrs = self._summary(customer_attributes=_customer_attributes(), one_hot_col="signup_channel")
        assert_frame_equal(with_attrs[base.columns.tolist()], base)

    @pytest.mark.parametrize(
        "attributes",
        [
            # A customer present in transactions but absent from customer_attributes: the left join leaves
            # 204's stores_shopped covariate NULL, which breaks ParetoNBD.
            pytest.param(_customer_attributes()[[cols.customer_id, "stores_shopped"]].iloc[:3], id="missing-customer"),
            # Full customer coverage, but 202's loyalty_points is itself NULL; joined as-is it is a NULL covariate.
            pytest.param(
                pd.DataFrame({cols.customer_id: [201, 202, 203, 204], "loyalty_points": [10.0, None, 30.0, 40.0]}),
                id="null-value",
            ),
        ],
    )
    def test_null_covariate_raises(self, attributes):
        """A NULL covariate (a customer missing from attributes, or an explicit NULL attribute value) is rejected."""
        with pytest.raises(ValueError, match="NULL"):
            self._summary(customer_attributes=attributes)

    @pytest.mark.parametrize("one_hot_col", ["signup_channel", ["signup_channel"]])
    def test_one_hot_encodes_and_drops_original(self, one_hot_col):
        """one_hot_col on an attribute column emits dummies and drops the source (str or list form)."""
        result = self._summary(customer_attributes=_customer_attributes(), one_hot_col=one_hot_col)
        expected = pd.DataFrame(
            {
                cols.customer_id: [201, 202, 203, 204],
                "signup_channel_email": np.array([1, 0, 0, 0], dtype="int8"),
                "signup_channel_referral": np.array([0, 1, 0, 0], dtype="int8"),
                "signup_channel_search": np.array([0, 0, 0, 1], dtype="int8"),
            },
        )
        dummy_cols = [c for c in result.columns if c.startswith("signup_channel_")]
        assert_frame_equal(result[[cols.customer_id, *dummy_cols]], expected)
        assert "signup_channel" not in result.columns
        # The column has NULLs (customer 203), so NULL is the dropped reference: every real value
        # emits a dummy and the NULL customer is 0 across all of them, with no dedicated NULL column.
        assert "signup_channel_null" not in result.columns

    def test_one_hot_integer_attribute_column(self):
        """one_hot_col can encode an integer attribute column (drops the sorted-first level)."""
        attributes = _customer_attributes()[[cols.customer_id, "stores_shopped"]]
        result = self._summary(customer_attributes=attributes, one_hot_col="stores_shopped")
        expected = pd.DataFrame(
            {cols.customer_id: [201, 202, 203, 204], "stores_shopped_2": np.array([1, 0, 0, 1], dtype="int8")},
        )
        dummy_cols = [c for c in result.columns if c.startswith("stores_shopped_")]
        assert_frame_equal(result[[cols.customer_id, *dummy_cols]], expected)
        assert "stores_shopped" not in result.columns

    def test_one_hot_multiple_columns(self):
        """A list one_hot_col encodes every named attribute column."""
        result = self._summary(
            customer_attributes=_customer_attributes(),
            one_hot_col=["signup_channel", "stores_shopped"],
        )
        expected = pd.DataFrame(
            {
                cols.customer_id: [201, 202, 203, 204],
                "signup_channel_email": np.array([1, 0, 0, 0], dtype="int8"),
                "signup_channel_referral": np.array([0, 1, 0, 0], dtype="int8"),
                "signup_channel_search": np.array([0, 0, 0, 1], dtype="int8"),
                "stores_shopped_2": np.array([1, 0, 0, 1], dtype="int8"),
            },
        )
        dummy_cols = [c for c in result.columns if c not in BASE_BTYD_COLS]
        assert_frame_equal(result[[cols.customer_id, *dummy_cols]], expected)
        assert "signup_channel" not in result.columns
        assert "stores_shopped" not in result.columns

    @pytest.mark.parametrize(
        ("n_categories", "expect_warn"),
        [
            # Each customer gets a distinct region and there are no NULLs, so one level is dropped as the
            # reference and emitted dummies == n_categories - 1. Pin the strict-`>` threshold exactly:
            # n = WARN + 1 emits WARN dummies (no warning); n = WARN + 2 emits WARN + 1 (warning).
            pytest.param(_ONE_HOT_CARDINALITY_WARN + 1, False, id="at-threshold"),
            pytest.param(_ONE_HOT_CARDINALITY_WARN + 2, True, id="over-threshold"),
        ],
    )
    def test_one_hot_warns_on_high_cardinality(self, n_categories, expect_warn):
        """One-hot encoding warns exactly when emitted dummies exceed _ONE_HOT_CARDINALITY_WARN (strict >)."""
        customer_ids = list(range(1, n_categories + 1))
        transactions = pd.DataFrame(
            {
                cols.customer_id: customer_ids,
                cols.transaction_date: pd.to_datetime(["2023-01-01"] * n_categories),
                cols.unit_spend: [10.0] * n_categories,
            },
        )
        attributes = pd.DataFrame(
            {cols.customer_id: customer_ids, "region": [f"r{i:03d}" for i in range(n_categories)]}
        )
        if expect_warn:
            with pytest.warns(UserWarning, match="dummy columns"):
                CLVStats(transactions, period="day", customer_attributes=attributes, one_hot_col="region")
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("error", UserWarning)  # any one-hot warning here would fail the test
                clv = CLVStats(transactions, period="day", customer_attributes=attributes, one_hot_col="region")
            assert len(clv.covariate_cols) == n_categories - 1  # encode still happens; one reference level dropped

    def test_one_hot_constant_column_warns_and_emits_no_covariate(self):
        """A constant, no-NULL one_hot column emits zero dummies and warns instead of failing silently."""
        transactions = pd.DataFrame(
            {
                cols.customer_id: [201, 202, 203],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
                cols.unit_spend: [10.0, 20.0, 30.0],
            },
        )
        # Every customer shares one region, no NULLs: the sole level is the dropped reference, so no dummy remains.
        attributes = pd.DataFrame({cols.customer_id: [201, 202, 203], "region": ["north", "north", "north"]})
        with pytest.warns(UserWarning, match="zero dummy columns"):
            clv = CLVStats(transactions, period="day", customer_attributes=attributes, one_hot_col="region")
        assert clv.covariate_cols == []
        assert "region" not in clv.df.columns

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            (
                {
                    "customer_attributes": _customer_attributes(),
                    "one_hot_col": ["signup_channel", "stores_shopped"],
                },
                ["signup_channel_email", "signup_channel_referral", "signup_channel_search", "stores_shopped_2"],
            ),
            ({}, []),
        ],
    )
    def test_covariate_cols_lists_only_attached_columns(self, kwargs, expected):
        """covariate_cols returns the attached attribute / one-hot columns, excluding base BTYD columns."""
        summary = CLVStats(_dim_transactions(), period="day", **kwargs)
        assert summary.covariate_cols == expected

    def test_one_hot_integer_column_with_nulls_names_without_decimal(self):
        """A nullable-integer one-hot column names dummies {col}_2, not the float64 {col}_2.0."""
        transactions = pd.DataFrame(
            {
                cols.customer_id: [401, 401, 402, 403],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-01", "2023-01-02"]),
                cols.unit_spend: [10.0, 20.0, 30.0, 40.0],
            },
        )
        attributes = pd.DataFrame(
            {
                cols.customer_id: [401, 402, 403],
                "loyalty_tier": pd.array([2, 3, None], dtype="Int64"),  # nullable int, as a DB source gives
            },
        )
        result = CLVStats(transactions, period="day", customer_attributes=attributes, one_hot_col="loyalty_tier").df
        tier_cols = {c for c in result.columns if c.startswith("loyalty_tier")}
        # 3 categories (2, 3, NULL); NULL is the dropped reference, so both real tiers emit dummies
        # named {col}_2 / {col}_3 (not the float64 {col}_2.0 / {col}_3.0).
        assert tier_cols == {"loyalty_tier_2", "loyalty_tier_3"}
        # The int-cast path must also produce the right 0/1 values, not just the right names: tier-2
        # customer 401 flags loyalty_tier_2, tier-3 customer 402 flags loyalty_tier_3, NULL customer
        # 403 (the reference) flags neither.
        dummies = result.set_index(cols.customer_id)[["loyalty_tier_2", "loyalty_tier_3"]].sort_index()
        expected_dummies = pd.DataFrame(
            {
                "loyalty_tier_2": np.array([1, 0, 0], dtype="int8"),
                "loyalty_tier_3": np.array([0, 1, 0], dtype="int8"),
            },
            index=pd.Index([401, 402, 403], name=cols.customer_id),
        )
        assert_frame_equal(dummies, expected_dummies)

    def test_empty_data_raises_clear_error(self):
        """Empty transaction data raises a clear ValueError, not a misleading date-coercion TypeError."""
        empty = pd.DataFrame(
            {
                cols.customer_id: pd.Series([], dtype="int64"),
                cols.transaction_date: pd.Series([], dtype="datetime64[ns]"),
                cols.unit_spend: pd.Series([], dtype="float64"),
            },
        )
        with pytest.raises(ValueError, match="no transactions"):
            CLVStats(empty)

    def test_duplicate_one_hot_col_is_deduplicated(self):
        """A repeated one_hot_col is de-duplicated, not encoded twice (which would drop-then-not-find it)."""
        transactions = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 3],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-01", "2023-01-02"]),
                cols.unit_spend: [10.0, 20.0, 30.0, 40.0],
            },
        )
        attributes = pd.DataFrame({cols.customer_id: [1, 2, 3], "channel": ["email", "web", "email"]})
        result = CLVStats(
            transactions,
            period="day",
            customer_attributes=attributes,
            one_hot_col=["channel", "channel"],
        ).df
        # Two categories (email, web); email sorts first and is the dropped reference, leaving one dummy.
        assert sorted(c for c in result.columns if c.startswith("channel")) == ["channel_web"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            # customer_attributes missing the customer_id column.
            pytest.param(
                {"customer_attributes": pd.DataFrame({"stores_shopped": [2, 1, 1, 2]})},
                "customer_id",
                id="missing-customer-id",
            ),
            # Duplicate customer_id: a left join on a non-unique key would fan out the summary rows.
            pytest.param(
                {
                    "customer_attributes": pd.DataFrame(
                        {
                            cols.customer_id: [201, 201, 202, 203, 204],
                            "stores_shopped": np.array([2, 2, 1, 1, 2], dtype="int64"),
                        },
                    ),
                },
                "one row per",
                id="duplicate-customer-id",
            ),
            # one_hot_col with no customer_attributes to encode.
            pytest.param({"one_hot_col": "signup_channel"}, "requires customer_attributes", id="one-hot-without-attrs"),
            # one_hot_col naming a column absent from customer_attributes.
            pytest.param(
                {"customer_attributes": _customer_attributes(), "one_hot_col": "loyalty_tier"},
                "loyalty_tier",
                id="one-hot-missing-column",
            ),
            # An attribute column named like a reserved BTYD column would mangle the summary.
            pytest.param(
                {
                    "customer_attributes": pd.DataFrame(
                        {cols.customer_id: [201, 202, 203, 204], "frequency": np.array([1, 2, 3, 4], dtype="int64")},
                    ),
                },
                "reserved BTYD",
                id="reserved-column-collision",
            ),
        ],
    )
    def test_invalid_attributes_or_one_hot_config_raises(self, kwargs, match):
        """A malformed customer_attributes / one_hot_col config is rejected with a clear ValueError."""
        with pytest.raises(ValueError, match=match):
            self._summary(**kwargs)


class TestCLVStatsOneHotNull:
    """NULL handling for one_hot_col: genuine NULLs are the dropped reference level."""

    def test_literal_null_string_coexists_with_genuine_nulls(self):
        """A literal "null" string is an ordinary value dummy; the genuine NULL is the dropped reference."""
        transactions = pd.DataFrame(
            {
                cols.customer_id: [301, 302, 303, 304],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]),
                cols.unit_spend: [10.0, 20.0, 30.0, 40.0],
            },
        )
        attributes = pd.DataFrame(
            {cols.customer_id: [301, 302, 303, 304], "signup_channel": ["null", "web", None, "app"]},
        )
        result = CLVStats(
            transactions,
            period="day",
            customer_attributes=attributes,
            one_hot_col="signup_channel",
        ).df.set_index(cols.customer_id)
        # Every real value emits a dummy (including the literal "null" string); the genuine NULL (303)
        # is the dropped reference, so it has no column of its own and is 0 across all dummies.
        dummy_cols = sorted(c for c in result.columns if c.startswith("signup_channel_"))
        assert dummy_cols == ["signup_channel_app", "signup_channel_null", "signup_channel_web"]
        expected = pd.DataFrame(
            {
                "signup_channel_app": np.array([0, 0, 0, 1], dtype="int8"),
                "signup_channel_null": np.array([1, 0, 0, 0], dtype="int8"),  # literal "null" string, 301
                "signup_channel_web": np.array([0, 1, 0, 0], dtype="int8"),
            },
            index=pd.Index([301, 302, 303, 304], name=cols.customer_id),
        )
        assert_frame_equal(result[dummy_cols].sort_index(), expected)


class TestCLVStatsCustomerIdColumn:
    """The output id column is always the pymc-marketing literal "customer_id", not the option name."""

    @staticmethod
    def _txns(id_col: str) -> pd.DataFrame:
        """Transactions whose customer key lives in ``id_col`` (the configured customer_id column)."""
        return pd.DataFrame(
            {
                id_col: [701, 701, 702, 703],
                cols.transaction_date: pd.to_datetime(["2023-01-01", "2023-01-08", "2023-01-01", "2023-01-02"]),
                cols.unit_spend: [100.0, 50.0, 200.0, 40.0],
            },
        )

    def test_overridden_customer_id_option_still_emits_literal_customer_id(self):
        """Input keyed on a remapped "shopper_id" still yields the literal "customer_id" output id."""
        with option_context("column.customer_id", "shopper_id"):
            result = CLVStats(self._txns("shopper_id"), period="day").df
        assert list(result.columns) == BASE_BTYD_COLS  # id is "customer_id", not the "shopper_id" option
        assert "shopper_id" not in result.columns
        # The renamed column carries the real customer keys, not an emptied/mangled column.
        assert sorted(result["customer_id"].tolist()) == [701, 702, 703]

    def test_overridden_option_attribute_named_customer_id_raises(self):
        """Under a remapped option, a customer_attributes column literally named "customer_id" collides."""
        with option_context("column.customer_id", "shopper_id"):
            attributes = pd.DataFrame(
                {"shopper_id": [701, 702, 703], "customer_id": np.array([1, 2, 3], dtype="int64")},
            )
            with pytest.raises(ValueError, match="reserved BTYD"):
                CLVStats(self._txns("shopper_id"), period="day", customer_attributes=attributes)
