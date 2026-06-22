"""Tests for the Purchase Path analysis module."""

import ibis
import pandas as pd
import pandas.testing as pdt
import pytest

from openretailscience.analysis.purchase_path import PurchasePath
from openretailscience.options import ColumnHelper

cols = ColumnHelper()

CATEGORY_COL = "category"


class TestPurchasePath:
    """Tests for the PurchasePath class."""

    @pytest.fixture
    def journeys_df(self) -> pd.DataFrame:
        """Six customers with distinct category-progression journeys.

        Each transaction holds two line items in a single category, so every basket
        is a single category. Customers 4 and 6 only make two trips, so they fall
        below the default ``min_transactions=3`` and drop out.

        - C1: womens -> kids  -> mens
        - C2: womens -> kids  -> kids   (basket 3 introduces no new category)
        - C3: womens -> kids  -> mens
        - C4: womens -> kids            (two trips only)
        - C5: mens   -> womens -> kids
        - C6: mens   -> womens          (two trips only)
        """
        customer_id = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6]
        transaction_id = [
            101,
            101,
            102,
            102,
            103,
            103,
            201,
            201,
            202,
            202,
            203,
            203,
            301,
            301,
            302,
            302,
            303,
            303,
            401,
            401,
            402,
            402,
            501,
            501,
            502,
            502,
            503,
            503,
            601,
            601,
            602,
            602,
        ]
        transaction_date = [
            "2024-01-01",
            "2024-01-01",
            "2024-01-10",
            "2024-01-10",
            "2024-01-20",
            "2024-01-20",
            "2024-01-02",
            "2024-01-02",
            "2024-01-11",
            "2024-01-11",
            "2024-01-21",
            "2024-01-21",
            "2024-01-03",
            "2024-01-03",
            "2024-01-12",
            "2024-01-12",
            "2024-01-22",
            "2024-01-22",
            "2024-01-04",
            "2024-01-04",
            "2024-01-13",
            "2024-01-13",
            "2024-01-05",
            "2024-01-05",
            "2024-01-14",
            "2024-01-14",
            "2024-01-23",
            "2024-01-23",
            "2024-01-06",
            "2024-01-06",
            "2024-01-15",
            "2024-01-15",
        ]
        category = [
            "womens",
            "womens",
            "kids",
            "kids",
            "mens",
            "mens",
            "womens",
            "womens",
            "kids",
            "kids",
            "kids",
            "kids",
            "womens",
            "womens",
            "kids",
            "kids",
            "mens",
            "mens",
            "womens",
            "womens",
            "kids",
            "kids",
            "mens",
            "mens",
            "womens",
            "womens",
            "kids",
            "kids",
            "mens",
            "mens",
            "womens",
            "womens",
        ]
        return pd.DataFrame(
            {
                cols.customer_id: customer_id,
                cols.transaction_id: transaction_id,
                cols.transaction_date: transaction_date,
                cols.product_id: list(range(1, 33)),
                cols.unit_spend: [50.0] * 32,
                CATEGORY_COL: category,
            },
        )

    def test_first_appearance_paths_and_customer_share(self, journeys_df):
        """Builds first-appearance paths and the share of customers on each one.

        Customers 4 and 6 have only two baskets and are excluded by the default
        ``min_transactions=3``. The four eligible customers (1, 2, 3, 5) form three
        distinct paths. ``pct_customers`` is each path's share of those four.
        """
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL).df

        expected = pd.DataFrame(
            {
                "basket_1": ["womens", "mens", "womens"],
                "basket_2": ["kids", "womens", "kids"],
                "basket_3": ["mens", "kids", ""],
                "customer_count": [2, 1, 1],
                "pct_customers": [0.5, 0.25, 0.25],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_basket_three_records_only_newly_introduced_categories(self, journeys_df):
        """Customer 2 repeats kids in basket 3, so its third position is empty, not 'kids'."""
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL).df

        c2_path = result[(result["basket_1"] == "womens") & (result["basket_3"] == "")]
        assert len(c2_path) == 1
        assert c2_path["basket_2"].to_numpy()[0] == "kids"
        assert c2_path["customer_count"].to_numpy()[0] == 1

    def test_min_customers_filters_rare_paths_without_inflating_share(self, journeys_df):
        """Requiring 2+ customers keeps only the womens->kids->mens path.

        Its share stays 0.5 (2 of 4 eligible customers) rather than rescaling to 1.0,
        because the denominator is all eligible customers, not the surviving paths.
        """
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL, min_customers=2).df

        expected = pd.DataFrame(
            {
                "basket_1": ["womens"],
                "basket_2": ["kids"],
                "basket_3": ["mens"],
                "customer_count": [2],
                "pct_customers": [0.5],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_max_depth_truncates_later_baskets(self, journeys_df):
        """With max_depth=2 only the first two trips count, so mens never appears.

        Capping depth at 2 means basket 3 (where mens first appears for C1/C3) is
        dropped, collapsing all four eligible customers onto womens->kids and
        mens->womens. C5's kids (trip 3) is also dropped.
        """
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL, max_depth=2, min_transactions=2).df

        # No basket_3 column at all, and the two surviving 2-step paths cover all 6 customers:
        # womens->kids (C1-C4) and mens->womens (C5, C6).
        expected = pd.DataFrame(
            {
                "basket_1": ["womens", "mens"],
                "basket_2": ["kids", "womens"],
                "customer_count": [4, 2],
                "pct_customers": [round(4 / 6, 4), round(2 / 6, 4)],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_concatenates_categories_first_appearing_in_same_basket(self):
        """Two categories introduced in one basket concatenate alphabetically."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 1, 1, 1],
                cols.transaction_id: [101, 101, 102, 102, 103, 103],
                cols.transaction_date: [
                    "2024-01-01",
                    "2024-01-01",
                    "2024-01-10",
                    "2024-01-10",
                    "2024-01-20",
                    "2024-01-20",
                ],
                cols.product_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [50.0] * 6,
                CATEGORY_COL: ["mens", "womens", "kids", "kids", "shoes", "shoes"],
            },
        )

        result = PurchasePath(df, category_col=CATEGORY_COL, min_transactions=1, min_customers=1).df

        assert result["basket_1"].to_numpy()[0] == "mens,womens"
        assert result["basket_2"].to_numpy()[0] == "kids"
        assert result["basket_3"].to_numpy()[0] == "shoes"

    def test_exclude_returns_drops_negative_spend_line_items(self):
        """Returns (negative unit_spend) are excluded by default so their basket vanishes.

        The kids basket has only a returned line item, so excluding it leaves the
        customer with womens then mens and no kids anywhere in the path.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1],
                cols.transaction_id: [101, 102, 103],
                cols.transaction_date: ["2024-01-01", "2024-01-10", "2024-01-20"],
                cols.product_id: [1, 2, 3],
                cols.unit_spend: [50.0, -25.0, 90.0],
                CATEGORY_COL: ["womens", "kids", "mens"],
            },
        )

        result = PurchasePath(df, category_col=CATEGORY_COL, min_transactions=1, min_customers=1).df

        expected = pd.DataFrame(
            {
                "basket_1": ["womens"],
                "basket_2": ["mens"],
                "customer_count": [1],
                "pct_customers": [1.0],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_keeping_returns_includes_negative_spend_category(self):
        """With exclude_returns=False the returned category is retained in the path.

        The value floor is relaxed below the -25 basket value so that ``min_basket_value``
        does not independently drop the returns-only basket, isolating ``exclude_returns``.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1],
                cols.transaction_id: [101, 102, 103],
                cols.transaction_date: ["2024-01-01", "2024-01-10", "2024-01-20"],
                cols.product_id: [1, 2, 3],
                cols.unit_spend: [50.0, -25.0, 90.0],
                CATEGORY_COL: ["womens", "kids", "mens"],
            },
        )

        result = PurchasePath(
            df,
            category_col=CATEGORY_COL,
            min_transactions=1,
            min_customers=1,
            min_basket_value=-100.0,
            exclude_returns=False,
        ).df

        assert result["basket_1"].to_numpy()[0] == "womens"
        assert result["basket_2"].to_numpy()[0] == "kids"
        assert result["basket_3"].to_numpy()[0] == "mens"

    def test_ibis_table_input_matches_dataframe_input(self, journeys_df):
        """An ibis memtable input yields the same result as the pandas DataFrame."""
        from_pandas = PurchasePath(journeys_df, category_col=CATEGORY_COL).df
        from_ibis = PurchasePath(ibis.memtable(journeys_df), category_col=CATEGORY_COL).df

        pdt.assert_frame_equal(from_ibis, from_pandas)

    def test_no_eligible_customers_returns_empty_result(self, journeys_df):
        """When the filters exclude everyone, an empty two-column frame is returned."""
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL, min_transactions=99).df

        assert len(result) == 0
        assert list(result.columns) == ["customer_count", "pct_customers"]

    def test_all_paths_below_min_customers_returns_empty_result(self, journeys_df):
        """Eligible customers exist but no path is common enough to survive min_customers."""
        # The most common path (womens->kids->mens) has only 2 customers, so requiring 3 drops all.
        result = PurchasePath(journeys_df, category_col=CATEGORY_COL, min_customers=3).df

        assert len(result) == 0
        assert list(result.columns) == ["customer_count", "pct_customers"]

    def test_missing_required_column_raises(self, journeys_df):
        """Dropping a required column raises a ValueError naming the missing column."""
        incomplete = journeys_df.drop(columns=[cols.product_id])

        with pytest.raises(ValueError, match="product_id"):
            PurchasePath(incomplete, category_col=CATEGORY_COL)

    @pytest.mark.parametrize(
        ("param", "value"),
        [
            ("min_transactions", 0),
            ("min_basket_size", 0),
            ("max_depth", 0),
            ("min_customers", 0),
            ("min_transactions", 2.5),
        ],
    )
    def test_invalid_filter_parameters_raise(self, journeys_df, param, value):
        """Non-positive or non-integer filter parameters are rejected."""
        with pytest.raises((ValueError, TypeError)):
            PurchasePath(journeys_df, category_col=CATEGORY_COL, **{param: value})
