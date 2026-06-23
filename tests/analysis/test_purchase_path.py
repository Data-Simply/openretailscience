"""Tests for the Purchase Path analysis module."""

import sys

import ibis
import pandas as pd
import pandas.testing as pdt
import pytest

from openretailscience.analysis.purchase_path import PurchasePath
from openretailscience.options import ColumnHelper

cols = ColumnHelper()

CATEGORY_COL = "category"


def _transactions(rows: list[tuple[int, int, str, str]]) -> pd.DataFrame:
    """Build a transactions frame from (customer_id, transaction_id, date, category) rows.

    Every row is one single-product, single-category basket worth 50 in spend, which keeps
    the fixtures focused on the category sequence rather than basket composition.
    """
    return pd.DataFrame(
        {
            cols.customer_id: [r[0] for r in rows],
            cols.transaction_id: [r[1] for r in rows],
            cols.transaction_date: [r[2] for r in rows],
            cols.product_id: list(range(1, len(rows) + 1)),
            cols.unit_spend: [50.0] * len(rows),
            CATEGORY_COL: [r[3] for r in rows],
        },
    )


class TestPurchasePath:
    """Tests for the PurchasePath transition model."""

    @pytest.fixture
    def transitions_df(self) -> pd.DataFrame:
        """Four customers whose first-category acquisitions diverge after womens.

        - C1: womens -> kids -> mens
        - C2: womens -> kids
        - C3: womens -> mens
        - C4: womens -> kids

        So of the four customers who move on from womens, three go to kids and one to mens.
        """
        return _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "kids"),
                (1, 103, "2024-01-20", "mens"),
                (2, 201, "2024-01-02", "womens"),
                (2, 202, "2024-01-11", "kids"),
                (3, 301, "2024-01-03", "womens"),
                (3, 302, "2024-01-12", "mens"),
                (4, 401, "2024-01-04", "womens"),
                (4, 402, "2024-01-13", "kids"),
            ],
        )

    def test_first_order_transition_probabilities(self, transitions_df):
        """Aggregates consecutive first-acquisitions into from->to transition probabilities.

        Of the four customers who progress past womens, three next acquire kids (0.75) and
        one next acquires mens (0.25). The single customer who progresses past kids goes to
        mens (1.0). mens is terminal for everyone, so it never appears as a source.
        """
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL).df

        expected = pd.DataFrame(
            {
                "from_category": ["kids", "womens", "womens"],
                "to_category": ["mens", "kids", "mens"],
                "customer_count": [1, 3, 1],
                "transition_probability": [1.0, 0.75, 0.25],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_simultaneous_categories_do_not_transition_among_themselves(self):
        """Categories acquired in the same basket flow to the next event, not to each other."""
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 101, "2024-01-01", "kids"),
                (1, 102, "2024-01-10", "mens"),
            ],
        )
        # Two line items share transaction 101, so the customer acquires womens and kids
        # simultaneously, then mens. product_id must stay unique per line item.
        df[cols.product_id] = [1, 2, 3]

        result = PurchasePath(df, category_col=CATEGORY_COL).df

        pairs = set(zip(result["from_category"], result["to_category"], strict=True))
        assert pairs == {("womens", "mens"), ("kids", "mens")}
        # No spurious order is imposed between the two simultaneously-acquired categories.
        assert ("womens", "kids") not in pairs
        assert ("kids", "womens") not in pairs

    def test_min_customers_filters_rare_transitions_without_inflating_probability(self, transitions_df):
        """Requiring 2+ customers drops the womens->mens transition but keeps probabilities honest.

        womens->kids stays at 0.75 (3 of the 4 customers who left womens), not rescaled to 1.0,
        because the denominator is all customers who progressed past womens.
        """
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL, min_customers=2).df

        expected = pd.DataFrame(
            {
                "from_category": ["womens"],
                "to_category": ["kids"],
                "customer_count": [3],
                "transition_probability": [0.75],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_max_depth_truncates_later_acquisitions(self, transitions_df):
        """Capping depth at 2 removes the kids->mens transition from customer 1's third basket."""
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL, max_depth=2).df

        pairs = set(zip(result["from_category"], result["to_category"], strict=True))
        assert ("kids", "mens") not in pairs
        assert ("womens", "kids") in pairs
        assert ("womens", "mens") in pairs

    def test_exclude_returns_drops_negative_spend_basket(self):
        """A returns-only basket is removed, so its category never enters a transition."""
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "kids"),
                (1, 103, "2024-01-20", "mens"),
            ],
        )
        df[cols.unit_spend] = [50.0, -25.0, 90.0]

        result = PurchasePath(df, category_col=CATEGORY_COL).df

        # The kids basket (only a return) is dropped, leaving a single womens -> mens transition.
        expected = pd.DataFrame(
            {
                "from_category": ["womens"],
                "to_category": ["mens"],
                "customer_count": [1],
                "transition_probability": [1.0],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_keeping_returns_retains_negative_spend_category(self):
        """With exclude_returns=False the returned category stays in the transition sequence.

        The value floor is relaxed below the -25 basket so that ``min_basket_value`` does not
        independently drop the returns-only basket, isolating ``exclude_returns``.
        """
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "kids"),
                (1, 103, "2024-01-20", "mens"),
            ],
        )
        df[cols.unit_spend] = [50.0, -25.0, 90.0]

        result = PurchasePath(
            df,
            category_col=CATEGORY_COL,
            min_basket_value=-100.0,
            exclude_returns=False,
        ).df

        pairs = set(zip(result["from_category"], result["to_category"], strict=True))
        assert pairs == {("womens", "kids"), ("kids", "mens")}

    def test_no_eligible_customers_returns_empty(self, transitions_df):
        """When the transaction filter excludes everyone, an empty four-column frame is returned."""
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL, min_transactions=99).df

        assert len(result) == 0
        assert list(result.columns) == ["from_category", "to_category", "customer_count", "transition_probability"]

    def test_ibis_table_input_matches_dataframe_input(self, transitions_df):
        """An ibis memtable input yields the same transitions as the pandas DataFrame."""
        from_pandas = PurchasePath(transitions_df, category_col=CATEGORY_COL).df
        from_ibis = PurchasePath(ibis.memtable(transitions_df), category_col=CATEGORY_COL).df

        pdt.assert_frame_equal(from_ibis, from_pandas)

    def test_customers_with_a_single_acquisition_yield_no_transitions(self):
        """Customers who only ever acquire one category produce an empty transition table."""
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "womens"),
                (2, 201, "2024-01-02", "kids"),
                (2, 202, "2024-01-11", "kids"),
            ],
        )

        result = PurchasePath(df, category_col=CATEGORY_COL).df

        assert len(result) == 0
        assert list(result.columns) == ["from_category", "to_category", "customer_count", "transition_probability"]

    def test_all_transitions_below_min_customers_returns_empty(self, transitions_df):
        """When no transition is common enough, an empty four-column frame is returned."""
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL, min_customers=99).df

        assert len(result) == 0
        assert list(result.columns) == ["from_category", "to_category", "customer_count", "transition_probability"]

    def test_dominant_journeys_traces_most_likely_progression(self, transitions_df):
        """The greedy walk from the womens entry yields womens -> kids -> mens at 0.75.

        From womens, kids is the likeliest next (0.75); from kids, mens is certain (1.0); mens
        is terminal. The journey likelihood is the product, 0.75 * 1.0 = 0.75.
        """
        result = PurchasePath(transitions_df, category_col=CATEGORY_COL).dominant_journeys()

        expected = pd.DataFrame(
            {
                "journey": ["womens -> kids -> mens"],
                "probability": [0.75],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_dominant_journeys_empty_when_no_transitions(self):
        """Single-acquisition customers produce no journeys (empty two-column frame)."""
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "womens"),
            ],
        )

        result = PurchasePath(df, category_col=CATEGORY_COL).dominant_journeys()

        assert len(result) == 0
        assert list(result.columns) == ["journey", "probability"]

    def test_to_networkx_builds_weighted_directed_graph(self, transitions_df):
        """to_networkx exposes the transitions as a DiGraph carrying edge weights."""
        nx = pytest.importorskip("networkx")
        expected_probability = 0.75
        expected_count = 3

        graph = PurchasePath(transitions_df, category_col=CATEGORY_COL).to_networkx()

        assert isinstance(graph, nx.DiGraph)
        assert set(graph.edges) == {("womens", "kids"), ("womens", "mens"), ("kids", "mens")}
        assert graph["womens"]["kids"]["transition_probability"] == expected_probability
        assert graph["womens"]["kids"]["customer_count"] == expected_count

    def test_to_networkx_without_networkx_raises_helpful_error(self, transitions_df, monkeypatch):
        """When networkx is unavailable, to_networkx raises a clear, actionable ImportError."""
        monkeypatch.setitem(sys.modules, "networkx", None)

        with pytest.raises(ImportError, match="to_networkx requires networkx"):
            PurchasePath(transitions_df, category_col=CATEGORY_COL).to_networkx()

    def test_dominant_journeys_only_follows_transitions_meeting_min_customers(self):
        """Journeys ignore entry categories whose onward transitions were filtered out.

        Both customers share the mens -> kids transition (2 customers) but reach mens from
        different, single-customer entries. With min_customers=2 only mens -> kids survives,
        and since neither entry category (womens, shoes) has a surviving onward edge, there is
        no journey to report.
        """
        df = _transactions(
            [
                (1, 101, "2024-01-01", "womens"),
                (1, 102, "2024-01-10", "mens"),
                (1, 103, "2024-01-20", "kids"),
                (2, 201, "2024-01-02", "shoes"),
                (2, 202, "2024-01-11", "mens"),
                (2, 203, "2024-01-21", "kids"),
            ],
        )

        result = PurchasePath(df, category_col=CATEGORY_COL, min_customers=2).dominant_journeys()

        assert len(result) == 0
        assert list(result.columns) == ["journey", "probability"]

    def test_integer_category_codes_keep_their_dtype(self):
        """Integer category codes stay integers in the output rather than being stringified."""
        womens, kids, mens = 100, 200, 300
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 2, 2],
                cols.transaction_id: [101, 102, 103, 201, 202],
                cols.transaction_date: ["2024-01-01", "2024-01-10", "2024-01-20", "2024-01-02", "2024-01-11"],
                cols.product_id: [1, 2, 3, 4, 5],
                cols.unit_spend: [50.0] * 5,
                CATEGORY_COL: [womens, kids, mens, womens, kids],
            },
        )

        pp = PurchasePath(df, category_col=CATEGORY_COL, min_customers=1)
        result = pp.df

        assert pd.api.types.is_integer_dtype(result["from_category"])
        assert pd.api.types.is_integer_dtype(result["to_category"])
        expected = pd.DataFrame(
            {
                "from_category": [womens, kids],
                "to_category": [kids, mens],
                "customer_count": [2, 1],
                "transition_probability": [1.0, 1.0],
            },
        )
        pdt.assert_frame_equal(result, expected)
        # Journey strings still render the integer codes as text.
        assert pp.dominant_journeys()["journey"].to_numpy()[0] == "100 -> 200 -> 300"

    def test_null_categories_are_excluded_from_transitions(self):
        """An uncategorised (null) line item does not create a phantom category node."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 1, 1],
                cols.transaction_id: [101, 102, 102, 103],
                cols.transaction_date: ["2024-01-01", "2024-01-10", "2024-01-10", "2024-01-20"],
                cols.product_id: [1, 2, 3, 4],
                cols.unit_spend: [50.0, 50.0, 50.0, 50.0],
                CATEGORY_COL: ["womens", "kids", None, "mens"],
            },
        )

        result = PurchasePath(df, category_col=CATEGORY_COL, min_transactions=1, min_customers=1).df

        expected = pd.DataFrame(
            {
                "from_category": ["kids", "womens"],
                "to_category": ["mens", "kids"],
                "customer_count": [1, 1],
                "transition_probability": [1.0, 1.0],
            },
        )
        pdt.assert_frame_equal(result, expected)

    def test_empty_result_keeps_typed_metric_columns(self, transitions_df):
        """Empty transition/journey results keep int/float metric dtypes for clean concatenation."""
        empty_transitions = PurchasePath(transitions_df, category_col=CATEGORY_COL, min_customers=99).df
        empty_journeys = PurchasePath(transitions_df, category_col=CATEGORY_COL, min_customers=99).dominant_journeys()

        assert empty_transitions["customer_count"].dtype == "int64"
        assert empty_transitions["transition_probability"].dtype == "float64"
        assert empty_journeys["probability"].dtype == "float64"

    def test_missing_required_column_raises(self, transitions_df):
        """Dropping a required column raises a ValueError naming the missing column."""
        incomplete = transitions_df.drop(columns=[cols.product_id])

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
    def test_invalid_filter_parameters_raise(self, transitions_df, param, value):
        """Non-positive or non-integer filter parameters are rejected."""
        with pytest.raises((ValueError, TypeError)):
            PurchasePath(transitions_df, category_col=CATEGORY_COL, **{param: value})
