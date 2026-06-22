"""Tests for openretailscience.analysis.customer."""

import functools
import math
from unittest import mock

import ibis
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.analysis.customer import (
    DaysBetweenPurchases,
    PurchasesPerCustomer,
    TransactionChurn,
)
from openretailscience.options import option_context

# Constructor makers normalized to a single ``make(df)`` signature so behaviors shared
# across the three classes can be parametrized. TransactionChurn needs a churn_period.
MAKE_PURCHASES = PurchasesPerCustomer
MAKE_DAYS = DaysBetweenPurchases
MAKE_CHURN = functools.partial(TransactionChurn, churn_period=30)

# Expected derived values for the `transactions_df` fixture below. Per-customer purchase
# counts are [1, 2, 3, 4] and per-customer mean inter-purchase gaps are [14.0, 30.0, 31.0].
EXPECTED_UNIQUE_CUSTOMERS = 4
MEDIAN_PURCHASE_COUNT = 2.5
Q25_PURCHASE_COUNT = 1.75
Q75_PURCHASE_COUNT = 3.25
MEDIAN_DAYS_BETWEEN_PURCHASES = 30.0
Q25_DAYS_BETWEEN_PURCHASES = 22.0
Q75_DAYS_BETWEEN_PURCHASES = 30.5


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """A small retail fixture chosen so every derived metric has a known closed-form value.

    Per-customer purchase patterns:
    - 101: 3 transactions on 2024-01-01 / 01-31 / 03-01 — gaps of 30, 30 days (avg 30.0)
    - 102: 2 transactions on 2024-01-15 / 02-15 — single gap of 31 days (avg 31.0)
    - 103: 4 transactions on 2024-04-01 / 04-15 / 04-29 / 05-13 — gaps of 14, 14, 14 days (avg 14.0)
    - 104: 1 transaction on 2024-04-10 — excluded from any inter-purchase computation

    The fixture also yields predictable churn behaviour with churn_period=30 (boundary 2024-04-13):
    customers 101, 102, 104 fall before the boundary; customer 103 has only its first transaction
    inside the window.
    """
    return pd.DataFrame(
        {
            "transaction_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "customer_id": [101, 101, 101, 102, 102, 103, 103, 103, 103, 104],
            "transaction_date": pd.to_datetime(
                [
                    "2024-01-01",
                    "2024-01-31",
                    "2024-03-01",
                    "2024-01-15",
                    "2024-02-15",
                    "2024-04-01",
                    "2024-04-15",
                    "2024-04-29",
                    "2024-05-13",
                    "2024-04-10",
                ],
            ),
            "unit_spend": [50.0, 60.0, 70.0, 100.0, 110.0, 25.0, 30.0, 35.0, 40.0, 80.0],
        },
    )


@pytest.fixture
def expected_purchase_counts() -> pd.DataFrame:
    """Per-customer transaction counts for the fixture, indexed by customer_id."""
    return pd.DataFrame(
        {"purchase_count": [3, 2, 4, 1]},
        index=pd.Index([101, 102, 103, 104], name="customer_id"),
    )


@pytest.fixture
def expected_days_between_purchases() -> pd.DataFrame:
    """Per-customer mean inter-purchase gap in days (single-transaction customers are excluded)."""
    return pd.DataFrame(
        {"avg_days_between_purchases": [30.0, 31.0, 14.0]},
        index=pd.Index([101, 102, 103], name="customer_id"),
    )


@pytest.fixture
def expected_churn_table() -> pd.DataFrame:
    """Retained / churned counts and churned_pct per transaction_number for the fixture.

    With churn_period=30 the boundary is 2024-04-13; transactions on or after that date are
    excluded. Within the surviving window:
      txn 1: cust 101, 102, 103 retained; cust 104 churned -> retained=3, churned=1, pct=0.25
      txn 2: cust 101 retained;            cust 102 churned -> retained=1, churned=1, pct=0.5
      txn 3: (no retained);                cust 101 churned -> retained=0, churned=1, pct=1.0
    """
    return pd.DataFrame(
        {
            "retained": [3, 1, 0],
            "churned": [1, 1, 1],
            "churned_pct": [0.25, 0.5, 1.0],
        },
        index=pd.Index([1, 2, 3], name="transaction_number"),
    )


class TestSharedConstructorContract:
    """Behaviors structurally identical across the three classes, parametrized over them.

    Each ``make`` is a callable normalizing the constructor to a single ``make(df)``
    signature (TransactionChurn binds churn_period via functools.partial).
    """

    @pytest.mark.parametrize(
        ("make", "expected_columns"),
        [
            (MAKE_PURCHASES, {"customer_id", "purchase_count"}),
            (MAKE_DAYS, {"customer_id", "avg_days_between_purchases"}),
            (MAKE_CHURN, {"transaction_number", "retained", "churned", "churned_pct"}),
        ],
        ids=["purchases", "days_between", "churn"],
    )
    def test_table_is_ibis_table(self, transactions_df, make, expected_columns):
        """The .table attribute is an ibis Table with the expected schema."""
        obj = make(transactions_df)
        assert isinstance(obj.table, ibis.Table)
        assert set(obj.table.columns) == expected_columns

    @pytest.mark.parametrize(
        ("make", "expected_fixture"),
        [
            (MAKE_PURCHASES, "expected_purchase_counts"),
            (MAKE_DAYS, "expected_days_between_purchases"),
            (MAKE_CHURN, "expected_churn_table"),
        ],
        ids=["purchases", "days_between", "churn"],
    )
    def test_accepts_ibis_table_input(self, request, transactions_df, make, expected_fixture):
        """An ibis Table passed in directly produces identical results to a DataFrame."""
        obj = make(ibis.memtable(transactions_df))
        expected = request.getfixturevalue(expected_fixture)
        assert_frame_equal(obj.df, expected)

    @pytest.mark.parametrize(
        ("make", "missing_col"),
        [
            (MAKE_PURCHASES, "transaction_id"),
            (MAKE_DAYS, "transaction_date"),
            (MAKE_CHURN, "transaction_date"),
        ],
        ids=["purchases", "days_between", "churn"],
    )
    def test_missing_required_columns_raises(self, make, missing_col):
        """Dropping a required column raises with the missing column listed."""
        df = pd.DataFrame({"customer_id": [1], "unit_spend": [1.0]})
        with pytest.raises(ValueError, match=missing_col):
            make(df)

    @pytest.mark.parametrize(
        "make", [MAKE_PURCHASES, MAKE_DAYS, MAKE_CHURN], ids=["purchases", "days_between", "churn"]
    )
    def test_rejects_non_table_input(self, make):
        """Inputs other than DataFrame / Ibis Table raise TypeError."""
        with pytest.raises(TypeError, match="pandas DataFrame or an Ibis Table"):
            make([1, 2, 3])

    @pytest.mark.parametrize(
        ("make", "rename_map", "options", "expected_fixture", "axis_name"),
        [
            (
                MAKE_PURCHASES,
                {"transaction_id": "txn_id", "customer_id": "cust_id"},
                ("column.customer_id", "cust_id", "column.transaction_id", "txn_id"),
                "expected_purchase_counts",
                "cust_id",
            ),
            (
                MAKE_DAYS,
                {"customer_id": "cust_id", "transaction_date": "txn_date"},
                ("column.customer_id", "cust_id", "column.transaction_date", "txn_date"),
                "expected_days_between_purchases",
                "cust_id",
            ),
            (
                MAKE_CHURN,
                {"customer_id": "cust_id", "transaction_date": "txn_date"},
                ("column.customer_id", "cust_id", "column.transaction_date", "txn_date"),
                "expected_churn_table",
                None,
            ),
        ],
        ids=["purchases", "days_between", "churn"],
    )
    def test_with_custom_column_names(
        self,
        request,
        transactions_df,
        make,
        rename_map,
        options,
        expected_fixture,
        axis_name,
    ):
        """Custom ColumnHelper names are honoured and produce identical results."""
        renamed = transactions_df.rename(columns=rename_map)
        with option_context(*options):
            actual = make(renamed).df
        expected = request.getfixturevalue(expected_fixture)
        # PurchasesPerCustomer / DaysBetweenPurchases index on the renamed customer_id;
        # TransactionChurn indexes on the derived transaction_number, which is unaffected.
        if axis_name is not None:
            expected = expected.rename_axis(axis_name)
        assert_frame_equal(actual, expected)

    @pytest.mark.parametrize("make", [MAKE_PURCHASES, MAKE_DAYS], ids=["purchases", "days_between"])
    def test_df_access_under_option_context_uses_init_time_column(self, transactions_df, make):
        """.df accessed inside a different option_context still resolves the init-time customer_id column."""
        obj = make(transactions_df)  # built under default "customer_id"
        with option_context("column.customer_id", "cust_id"):
            # Must not raise — the column was resolved at __init__ and baked into the cached frame.
            result = obj.df
        assert result.index.name == "customer_id"

    @pytest.mark.parametrize("make", [MAKE_PURCHASES, MAKE_DAYS], ids=["purchases", "days_between"])
    def test_purchases_percentile_reuses_materialized_df(self, transactions_df, make):
        """Repeated purchases_percentile calls share a single backend execution (the cached .df)."""
        obj = make(transactions_df)
        # Wrap the backend's execute (an external dependency) so the real query still runs but
        # is counted. A per-call re-query would push the count above one.
        table_cls = type(obj.table)
        real_execute = table_cls.execute
        with mock.patch.object(table_cls, "execute", autospec=True, side_effect=real_execute) as execute_spy:
            obj.purchases_percentile(0.25)
            obj.purchases_percentile(0.5)
            obj.purchases_percentile(0.75)
        assert execute_spy.call_count == 1


class TestPurchasesPerCustomer:
    """Behavioral tests for PurchasesPerCustomer."""

    def test_df_holds_unique_transactions_per_customer(self, transactions_df, expected_purchase_counts):
        """The materialized df holds the unique transaction count per customer."""
        ppc = PurchasesPerCustomer(transactions_df)
        assert_frame_equal(ppc.df, expected_purchase_counts)

    @pytest.mark.parametrize(
        ("percentile", "expected"),
        [
            (0.25, Q25_PURCHASE_COUNT),
            (0.5, MEDIAN_PURCHASE_COUNT),
            (0.75, Q75_PURCHASE_COUNT),
            (1.0, 4.0),
        ],
    )
    def test_purchases_percentile_uses_linear_interpolation(self, transactions_df, percentile, expected):
        """purchases_percentile reports the requested quantile using linear interpolation."""
        ppc = PurchasesPerCustomer(transactions_df)
        assert ppc.purchases_percentile(percentile) == pytest.approx(expected)

    @pytest.mark.parametrize(
        ("threshold", "comparison", "expected"),
        [
            (2, "less_than_equal_to", 0.5),
            (3, "less_than", 0.5),
            (3, "equal_to", 0.25),
            (3, "greater_than", 0.25),
            (3, "greater_than_equal_to", 0.5),
            (3, "not_equal_to", 0.75),
            (5, "greater_than_equal_to", 0.0),
        ],
    )
    def test_find_purchase_percentile(self, transactions_df, threshold, comparison, expected):
        """find_purchase_percentile returns the share of customers matching the comparison."""
        ppc = PurchasesPerCustomer(transactions_df)
        assert ppc.find_purchase_percentile(threshold, comparison) == pytest.approx(expected)

    def test_find_purchase_percentile_invalid_comparison_raises(self, transactions_df):
        """Unknown comparison strings are rejected with a clear error."""
        ppc = PurchasesPerCustomer(transactions_df)
        with pytest.raises(ValueError, match="comparison must be one of"):
            ppc.find_purchase_percentile(1, "foo")

    def test_find_purchase_percentile_rejects_non_numeric_threshold(self, transactions_df):
        """A non-numeric number_of_purchases raises a clear TypeError, not a cryptic comparison error."""
        ppc = PurchasesPerCustomer(transactions_df)
        with pytest.raises(TypeError, match="number_of_purchases must be a number"):
            ppc.find_purchase_percentile("3")

    def test_find_purchase_percentile_returns_nan_on_empty_input(self):
        """An empty input yields NaN rather than ZeroDivisionError (matches purchases_percentile)."""
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series([], dtype="int64"),
                "transaction_id": pd.Series([], dtype="int64"),
            },
        )
        ppc = PurchasesPerCustomer(empty)
        assert math.isnan(ppc.find_purchase_percentile(1))

    def test_purchases_percentile_returns_nan_on_empty_input(self):
        """purchases_percentile returns NaN on empty input, consistent with find_purchase_percentile."""
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series([], dtype="int64"),
                "transaction_id": pd.Series([], dtype="int64"),
            },
        )
        ppc = PurchasesPerCustomer(empty)
        assert math.isnan(ppc.purchases_percentile(0.5))

    def test_df_after_option_context_exit_keeps_init_time_column(self, transactions_df):
        """A .df built and materialized inside an option_context retains that column name after the context exits."""
        renamed = transactions_df.rename(columns={"customer_id": "cust_id", "transaction_id": "txn_id"})
        with option_context("column.customer_id", "cust_id", "column.transaction_id", "txn_id"):
            ppc = PurchasesPerCustomer(renamed)
            inside = ppc.df
        outside = ppc.df
        # Same cached object; the index name was fixed at __init__ to "cust_id" and does not silently revert.
        assert inside.index.name == "cust_id"
        assert outside.index.name == "cust_id"


class TestDaysBetweenPurchases:
    """Behavioral tests for DaysBetweenPurchases."""

    def test_df_holds_average_days_between_purchases(self, transactions_df, expected_days_between_purchases):
        """The materialized df holds the per-customer mean gap, excluding single-purchase customers."""
        dbp = DaysBetweenPurchases(transactions_df)
        assert_frame_equal(dbp.df, expected_days_between_purchases)

    def test_empty_input_yields_empty_df(self):
        """An empty input degrades to an empty result with the expected column, not a crash."""
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series([], dtype="int64"),
                "transaction_date": pd.Series([], dtype="datetime64[ns]"),
            },
        )
        dbp = DaysBetweenPurchases(empty)
        assert len(dbp.df) == 0
        assert list(dbp.df.columns) == ["avg_days_between_purchases"]

    def test_same_day_transactions_collapse_to_one_purchase_day(self):
        """Multiple transactions on the same day count as a single purchase day."""
        df = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3, 4],
                "customer_id": [201, 201, 201, 201],
                # Two transactions on 2024-01-01 should dedupe to one purchase day,
                # leaving gaps of 10 and 10 days (avg 10.0). If dedup is broken, a
                # zero-day gap would drag the average down to ~6.67.
                "transaction_date": pd.to_datetime(
                    ["2024-01-01 09:00", "2024-01-01 18:00", "2024-01-11 00:00", "2024-01-21 00:00"],
                ),
            },
        )
        dbp = DaysBetweenPurchases(df)
        assert dbp.df.loc[201, "avg_days_between_purchases"] == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("percentile", "expected"),
        [
            (0.25, Q25_DAYS_BETWEEN_PURCHASES),
            (0.5, MEDIAN_DAYS_BETWEEN_PURCHASES),
            (0.75, Q75_DAYS_BETWEEN_PURCHASES),
            (1.0, 31.0),
        ],
    )
    def test_purchases_percentile_uses_linear_interpolation(self, transactions_df, percentile, expected):
        """purchases_percentile reports the requested quantile of per-customer mean gaps."""
        dbp = DaysBetweenPurchases(transactions_df)
        assert dbp.purchases_percentile(percentile) == pytest.approx(expected)


class TestTransactionChurn:
    """Behavioral tests for TransactionChurn."""

    def test_df_holds_retained_churned_and_rate_per_transaction_number(
        self,
        transactions_df,
        expected_churn_table,
    ):
        """The materialized df reports retained, churned counts and the churned percentage."""
        tc = TransactionChurn(transactions_df, churn_period=30)
        assert_frame_equal(tc.df, expected_churn_table)

    def test_counts_unique_customers_in_source(self, transactions_df):
        """n_unique_customers reflects the distinct customers in the input, not the filtered window."""
        tc = TransactionChurn(transactions_df, churn_period=30)
        assert tc.n_unique_customers == EXPECTED_UNIQUE_CUSTOMERS

    @pytest.mark.parametrize(
        ("churn_period", "expected_in_window_total"),
        [
            # boundary 2024-04-23: cust 101 (3) + 102 (2) + 103 (2: 4/1, 4/15) + 104 (1) = 8 in window
            (20, 8),
            # boundary 2024-04-13: cust 101 (3) + 102 (2) + 103 (1: 4/1) + 104 (1) = 7 in window
            (30, 7),
            # boundary 2024-03-14: cust 101 (3) + 102 (2) = 5 in window
            (60, 5),
            # boundary 2023-05-14: nothing strictly before -> empty table
            (365, 0),
        ],
    )
    def test_churn_period_shifts_window_boundary(self, transactions_df, churn_period, expected_in_window_total):
        """Larger churn_period values exclude more recent transactions from the window.

        Total in-window transactions = sum of (retained + churned) across all transaction_numbers,
        which is the cleanest invariant for "how many transactions fell inside the churn window".
        """
        tc = TransactionChurn(transactions_df, churn_period=churn_period)
        if expected_in_window_total == 0:
            assert len(tc.df) == 0
        else:
            assert (tc.df["retained"] + tc.df["churned"]).sum() == expected_in_window_total

    def test_transactions_on_boundary_are_excluded(self):
        """The churn window uses strict <, so transactions exactly on the boundary are excluded."""
        # max date = 2024-05-01; with churn_period=30 the boundary is 2024-04-01.
        # Customer 301 has one transaction exactly on 2024-04-01 -> must be excluded from the window.
        df = pd.DataFrame(
            {
                "transaction_id": [1, 2, 3],
                "customer_id": [301, 302, 302],
                "transaction_date": pd.to_datetime(["2024-04-01", "2024-03-15", "2024-05-01"]),
            },
        )
        tc = TransactionChurn(df, churn_period=30)
        # Customer 302's transaction on 2024-03-15 is the only row strictly before 2024-04-01.
        # That customer has a later transaction (2024-05-01), so they are retained at txn 1.
        assert tc.df.loc[1, "retained"] == 1
        assert tc.df.loc[1, "churned"] == 0

    @pytest.mark.parametrize("churn_period", [0, -5])
    def test_churn_period_must_be_positive(self, transactions_df, churn_period):
        """A non-positive churn_period is rejected at construction rather than producing nonsense."""
        with pytest.raises(ValueError, match="churn_period must be positive"):
            TransactionChurn(transactions_df, churn_period=churn_period)

    def test_empty_input_yields_zero_customers_and_empty_df(self):
        """An empty input reports zero customers and an empty churn table rather than crashing."""
        empty = pd.DataFrame(
            {
                "customer_id": pd.Series([], dtype="int64"),
                "transaction_date": pd.Series([], dtype="datetime64[ns]"),
            },
        )
        tc = TransactionChurn(empty, churn_period=30)
        assert tc.n_unique_customers == 0
        assert len(tc.df) == 0
        assert list(tc.df.columns) == ["retained", "churned", "churned_pct"]


class TestInputValidation:
    """Validation behaviours shared across the three classes."""

    @pytest.mark.parametrize("make", [MAKE_PURCHASES, MAKE_DAYS], ids=["purchases", "days_between"])
    @pytest.mark.parametrize("bad_percentile", [-0.1, 1.5])
    def test_percentile_out_of_range_raises(self, transactions_df, make, bad_percentile):
        """purchases_percentile rejects values outside [0, 1] with a clear ValueError."""
        obj = make(transactions_df)
        with pytest.raises(ValueError, match="between 0 and 1"):
            obj.purchases_percentile(bad_percentile)

    @pytest.mark.parametrize("make", [MAKE_DAYS, MAKE_CHURN], ids=["days_between", "churn"])
    def test_rejects_tz_aware_transaction_date(self, make):
        """Timezone-aware transaction_date is rejected because the backend re-buckets it to UTC days."""
        df = pd.DataFrame(
            {
                "customer_id": [1, 1],
                "transaction_date": pd.to_datetime(["2024-01-01 23:00", "2024-01-02 02:00"]).tz_localize(
                    "US/Eastern",
                ),
            },
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            make(df)

    @pytest.mark.parametrize("make", [MAKE_DAYS, MAKE_CHURN], ids=["days_between", "churn"])
    def test_rejects_non_datetime_transaction_date(self, make):
        """A string transaction_date raises a clear TypeError instead of a cryptic ibis error."""
        df = pd.DataFrame({"customer_id": [1, 1], "transaction_date": ["2024-01-01", "2024-01-05"]})
        with pytest.raises(TypeError, match="date or datetime"):
            make(df)

    @pytest.mark.parametrize("make", [MAKE_PURCHASES, MAKE_DAYS], ids=["purchases", "days_between"])
    def test_percentile_rejects_bool(self, transactions_df, make):
        """A bool percentile is rejected, not silently treated as 0.0/1.0 (bool is an int subclass)."""
        obj = make(transactions_df)
        with pytest.raises(TypeError, match="must be a number"):
            obj.purchases_percentile(True)

    @pytest.mark.parametrize("make", [MAKE_PURCHASES, MAKE_DAYS], ids=["purchases", "days_between"])
    def test_percentile_rejects_non_numeric(self, transactions_df, make):
        """A non-numeric percentile (e.g. a string) is rejected with the same clear error."""
        obj = make(transactions_df)
        with pytest.raises(TypeError, match="must be a number"):
            obj.purchases_percentile("high")

    def test_churn_period_rejects_bool(self, transactions_df):
        """A bool churn_period is rejected, not silently treated as a 1-day window."""
        with pytest.raises(TypeError, match="must be a number"):
            TransactionChurn(transactions_df, churn_period=True)

    def test_churn_period_rejects_non_numeric(self, transactions_df):
        """A non-numeric churn_period (e.g. a string) is rejected with the same clear error."""
        with pytest.raises(TypeError, match="must be a number"):
            TransactionChurn(transactions_df, churn_period="high")
