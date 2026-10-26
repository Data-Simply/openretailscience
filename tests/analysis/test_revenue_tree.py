"""Tests for the RevenueTree class."""

from __future__ import annotations

import datetime as dt
import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.analysis.revenue_tree import RevenueTree, calc_tree_kpis
from openretailscience.options import ColumnHelper, get_option, option_context

if TYPE_CHECKING:
    from collections.abc import Callable

# (customers, transactions, spend, units) per period, chosen so every factor moves: customers 4->5,
# visits/customer 2->3, spend/visit 50->60, units/visit 5->8, price/unit 10->7.5
ALL_FACTORS_MOVE_TOTALS = ((4, 8, 400.0, 40), (5, 15, 900.0, 120))


def _period_totals(
    p1_totals: tuple[int, int, float, int], p2_totals: tuple[int, int, float, int]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the one-row, ungrouped period frames ``calc_tree_kpis`` takes from per-period totals."""
    cols = ColumnHelper()
    columns = [cols.agg.customer_id, cols.agg.transaction_id, cols.agg.unit_spend, cols.agg.unit_qty]
    return pd.DataFrame([p1_totals], columns=columns), pd.DataFrame([p2_totals], columns=columns)


def _ungrouped_periods(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a two-row ``[p1, p2]`` aggregate into one-row period frames that share the label 0."""
    return df.iloc[[0]].reset_index(drop=True), df.iloc[[1]].reset_index(drop=True)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


class TestRevenueTree:
    """Test the RevenueTree class."""

    @pytest.fixture
    def cols(self):
        """Return a ColumnHelper instance."""
        return ColumnHelper()

    @pytest.fixture
    def contrib_cols(self, cols: ColumnHelper) -> list[str]:
        """Every contribution column, top of the tree first."""
        return [
            cols.agg.customer_id_contrib,
            cols.calc.spend_per_cust_contrib,
            cols.calc.trans_per_cust_contrib,
            cols.calc.spend_per_trans_contrib,
            cols.calc.units_per_trans_contrib,
            cols.calc.price_per_unit_contrib,
        ]

    @pytest.mark.parametrize("dropped_col", ["unit_spend", "period"])
    def test_dataframe_missing_required_columns(self, cols: ColumnHelper, dropped_col: str):
        """Missing spend or period columns are rejected before any aggregation runs."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.transaction_id: [9001, 9002, 9003],
                cols.unit_spend: [20.0, 35.5, 12.0],
                "period": ["P1", "P2", "P1"],
            },
        ).drop(columns=getattr(cols, dropped_col, dropped_col))
        with pytest.raises(ValueError, match="Input data is missing required columns"):
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2")

    def test_dataframe_missing_group_col(self, cols: ColumnHelper):
        """Test that an error naming group_col is raised when a single group_col is missing."""
        data = {
            cols.customer_id: [1, 2, 3],
            cols.unit_spend: [100, 200, 300],
            cols.transaction_date: ["2023-01-01", "2023-06-02", "2023-01-03"],
            "period": ["P1", "P2", "P1"],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col="group_id")
        assert "group_col references columns not present in the data" in str(excinfo.value)
        assert "group_id" in str(excinfo.value)

    def test_dataframe_missing_group_cols_list(self, cols: ColumnHelper):
        """Test that an error is raised when the DataFrame is missing group_col columns from a list."""
        data = {
            cols.customer_id: [1, 2, 3],
            cols.transaction_id: [1, 2, 3],
            cols.unit_spend: [100, 200, 300],
            cols.transaction_date: ["2023-01-01", "2023-06-02", "2023-01-03"],
            "period": ["P1", "P2", "P1"],
            "region": ["North", "South", "North"],
        }
        df = pd.DataFrame(data)

        with pytest.raises(ValueError) as excinfo:
            RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=["region", "store"])
        assert "group_col references columns not present in the data" in str(excinfo.value)
        assert "store" in str(excinfo.value)
        assert "store" in str(excinfo.value)

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_no_group(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with no group_col."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2"],
            },
        )

        expected_p1 = pd.DataFrame(
            {cols.agg.customer_id: [3], cols.agg.transaction_id: [3], cols.agg.unit_spend: [900.0]}
        )
        expected_p2 = pd.DataFrame(
            {cols.agg.customer_id: [3], cols.agg.transaction_id: [3], cols.agg.unit_spend: [1200.0]}
        )

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6]
            expected_p1[cols.agg.unit_qty] = [9]
            expected_p2[cols.agg.unit_qty] = [12]

        p1_df, p2_df = RevenueTree._agg_data(df=df, period_col="period", p1_value="P1", p2_value="P2")

        pd.testing.assert_frame_equal(p1_df, expected_p1)
        pd.testing.assert_frame_equal(p2_df, expected_p2)

    @pytest.mark.parametrize("include_quantity", [True, False])
    @pytest.mark.parametrize(
        ("store_ids", "spends", "units", "expected_p1", "expected_p2"),
        [
            # Both stores trade in both periods
            (
                [101, 101, 101, 102, 102, 102],
                [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                [1, 2, 3, 4, 5, 6],
                {101: (2, 400.0, 4), 102: (1, 500.0, 5)},
                {101: (1, 200.0, 2), 102: (2, 1000.0, 10)},
            ),
            # Store 103 trades only in P1, so it is absent from the P2 frame
            (
                [101, 101, 101, 102, 102, 102, 103],
                [100.0, 200.0, 300.0, 400.0, 500.0, 600.0, 700.0],
                [1, 2, 3, 4, 5, 6, 7],
                {101: (2, 400.0, 4), 102: (1, 500.0, 5), 103: (1, 700.0, 7)},
                {101: (1, 200.0, 2), 102: (2, 1000.0, 10)},
            ),
        ],
    )
    def test_agg_data_with_group(
        self,
        cols: ColumnHelper,
        include_quantity: bool,
        store_ids: list[int],
        spends: list[float],
        units: list[int],
        expected_p1: dict[int, tuple[int, float, int]],
        expected_p2: dict[int, tuple[int, float, int]],
    ) -> None:
        """Each period frame has one row per store present in it, on a CategoricalIndex shared by both periods."""
        df = pd.DataFrame(
            {
                cols.store_id: store_ids,
                cols.customer_id: list(range(1, len(store_ids) + 1)),
                cols.transaction_id: list(range(1, len(store_ids) + 1)),
                cols.unit_spend: spends,
                "period": ["P1", "P2"] * (len(store_ids) // 2) + ["P1"] * (len(store_ids) % 2),
            },
        )
        if include_quantity:
            df[cols.unit_qty] = units
        categories = sorted(set(store_ids))

        p1_df, p2_df = RevenueTree._agg_data(
            df, period_col="period", p1_value="P1", p2_value="P2", group_col=[cols.store_id]
        )

        for result, expected in ((p1_df, expected_p1), (p2_df, expected_p2)):
            expected_df = pd.DataFrame(
                {
                    cols.agg.customer_id: [n for n, _, _ in expected.values()],
                    cols.agg.transaction_id: [n for n, _, _ in expected.values()],
                    cols.agg.unit_spend: [spend for _, spend, _ in expected.values()],
                },
                index=pd.CategoricalIndex(list(expected), categories=categories, name=cols.store_id),
            )
            if include_quantity:
                expected_df[cols.agg.unit_qty] = [qty for _, _, qty in expected.values()]
            pd.testing.assert_frame_equal(result, expected_df)

    @pytest.mark.parametrize("include_quantity", [True, False])
    @pytest.mark.parametrize(
        ("store_ids", "row_label"),
        [
            # Ungrouped: the single row compares all of P1 with all of P2
            (None, 0),
            # Store 103 trades only in P1, so its P2 side is zero-filled
            ([101, 101, 101, 102, 102, 102, 103], 103),
        ],
    )
    def test_contribs_sum_to_parent(
        self,
        cols: ColumnHelper,
        include_quantity: bool,
        store_ids: list[int] | None,
        row_label: int,
    ) -> None:
        """Sibling contributions sum to their parent's, and every ``_diff`` is p2 minus p1."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6, 7],
                cols.transaction_id: [1, 2, 3, 4, 5, 6, 7],
                cols.unit_spend: [150.0, 75.5, 220.0, 310.0, 450.0, 120.0, 700.0],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2", "P1"],
            },
        )
        if include_quantity:
            df[cols.unit_qty] = [2, 1, 3, 4, 5, 2, 7]
        group_col = None
        if store_ids is not None:
            df[cols.store_id] = store_ids
            group_col = cols.store_id

        rt = RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=group_col)
        tree = rt.df.loc[row_label]

        assert math.isclose(
            tree[cols.agg.customer_id_contrib] + tree[cols.calc.spend_per_cust_contrib],
            tree[cols.agg.unit_spend_diff],
        )
        assert math.isclose(
            tree[cols.calc.trans_per_cust_contrib] + tree[cols.calc.spend_per_trans_contrib],
            tree[cols.calc.spend_per_cust_contrib],
        )
        metrics = [
            cols.agg.unit_spend,
            cols.agg.customer_id,
            cols.agg.transaction_id,
            cols.calc.spend_per_cust,
            cols.calc.trans_per_cust,
            cols.calc.spend_per_trans,
        ]
        if include_quantity:
            assert math.isclose(
                tree[cols.calc.units_per_trans_contrib] + tree[cols.calc.price_per_unit_contrib],
                tree[cols.calc.spend_per_trans_contrib],
            )
            metrics += [cols.agg.unit_qty, cols.calc.units_per_trans, cols.calc.price_per_unit]
        else:
            assert cols.calc.units_per_trans_contrib not in rt.df.columns
            assert cols.calc.price_per_unit_contrib not in rt.df.columns

        p1, p2, diff = (get_option(f"column.suffix.{s}") for s in ("period_1", "period_2", "difference"))
        # assert_series_equal treats NaN == NaN, which store 103's undefined P2 ratios need
        pd.testing.assert_series_equal(
            pd.Series({metric: tree[f"{metric}_{diff}"] for metric in metrics}),
            pd.Series({metric: tree[f"{metric}_{p2}"] - tree[f"{metric}_{p1}"] for metric in metrics}),
        )

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_calc_tree_kpis_basic(self, cols: ColumnHelper, include_quantity: bool):
        """Test basic KPI calculations."""
        df = pd.DataFrame(
            {
                cols.agg.customer_id: [3, 3],
                cols.agg.transaction_id: [3, 3],
                cols.agg.unit_spend: [900.0, 1100.0],
            },
            index=["p1", "p2"],
        )

        if include_quantity:
            df[cols.agg.unit_qty] = [10, 12]

        result = calc_tree_kpis(*_ungrouped_periods(df))

        assert isinstance(result, pd.DataFrame)

        expected_columns = [
            "customers_diff",
            "customers_pct_diff",
            "transactions_diff",
            "transactions_pct_diff",
            "spend_diff",
            "spend_pct_diff",
            "spend_per_transaction_diff",
            "spend_per_transaction_pct_diff",
            "transactions_per_customer_diff",
            "transactions_per_customer_pct_diff",
            "spend_per_customer_diff",
            "spend_per_customer_pct_diff",
            "frequency_elasticity",
        ]
        for col in expected_columns:
            assert col in result.columns

        if include_quantity:
            q1, q2 = 10, 12
            p1, p2 = 900.0 / q1, 1100.0 / q2
            expected_elasticity = ((q2 - q1) / ((q2 + q1) / 2)) / ((p2 - p1) / ((p2 + p1) / 2))

            assert round(result["price_elasticity"].iloc[0], 6) == round(expected_elasticity, 6)

    def test_calc_tree_kpis_pairs_periods_by_index_label(self, cols: ColumnHelper) -> None:
        """Rows are paired by label for any index, so a store that closed and one that opened stay separate."""
        totals = {cols.agg.customer_id: [2], cols.agg.transaction_id: [2]}
        p1_df = pd.DataFrame(
            {**totals, cols.agg.unit_spend: [100.0]}, index=pd.Index(["Store_101"], name=cols.store_id)
        )
        p2_df = pd.DataFrame(
            {**totals, cols.agg.unit_spend: [300.0]}, index=pd.Index(["Store_102"], name=cols.store_id)
        )

        result = calc_tree_kpis(p1_df, p2_df)

        expected = pd.DataFrame(
            {cols.agg.unit_spend_p1: [100.0, 0.0], cols.agg.unit_spend_p2: [0.0, 300.0]},
            index=pd.Index(["Store_101", "Store_102"], name=cols.store_id),
        )
        pd.testing.assert_frame_equal(result[[cols.agg.unit_spend_p1, cols.agg.unit_spend_p2]], expected)

    @pytest.mark.parametrize(
        ("make_p2_df", "error", "match"),
        [
            (lambda _cols: [False, True], TypeError, "p2_df must be a pandas DataFrame"),
            (
                lambda cols: pd.DataFrame(
                    {cols.agg.customer_id: [1, 2], cols.agg.transaction_id: [1, 2], cols.agg.unit_spend: [50.0, 80.0]},
                    index=pd.Index(["Store_101", "Store_101"], name=cols.store_id),
                ),
                ValueError,
                "p2_df has repeated index labels",
            ),
            # Units in P2 only: dropping them silently would hide the units and price nodes
            (
                lambda cols: pd.DataFrame(
                    {
                        cols.agg.customer_id: [1],
                        cols.agg.transaction_id: [1],
                        cols.agg.unit_spend: [80.0],
                        cols.agg.unit_qty: [4],
                    },
                    index=pd.Index(["Store_101"], name=cols.store_id),
                ),
                ValueError,
                "both have .* or neither",
            ),
        ],
    )
    def test_calc_tree_kpis_rejects_unpairable_input(
        self,
        cols: ColumnHelper,
        make_p2_df: Callable[[ColumnHelper], list[bool] | pd.DataFrame],
        error: type[Exception],
        match: str,
    ) -> None:
        """Unpairable input raises: a non-DataFrame such as a mask, repeated labels, or units on one side only."""
        p1_df = pd.DataFrame(
            {cols.agg.customer_id: [1], cols.agg.transaction_id: [1], cols.agg.unit_spend: [50.0]},
            index=pd.Index(["Store_101"], name=cols.store_id),
        )

        with pytest.raises(error, match=match):
            calc_tree_kpis(p1_df, make_p2_df(cols))

    @pytest.mark.parametrize(
        ("p1_totals", "p2_totals", "expected_contribs"),
        [
            (*ALL_FACTORS_MOVE_TOTALS, (140.0, 360.0, 247.5, 112.5, 295.3125, -182.8125)),
            # Visits per customer (T=1) and units per basket (U=2) are flat, so both contribute nothing
            ((2, 2, 20.0, 4), (4, 4, 60.0, 8), (25.0, 15.0, 0.0, 15.0, 0.0, 15.0)),
            # P1 spend recorded against zero units can't be split into units x price, so both are NaN
            (
                (*ALL_FACTORS_MOVE_TOTALS[0][:3], 0),
                ALL_FACTORS_MOVE_TOTALS[1],
                (140.0, 360.0, 247.5, 112.5, np.nan, np.nan),
            ),
        ],
    )
    def test_contribs_use_midpoint_weights(
        self,
        contrib_cols: list[str],
        p1_totals: tuple[int, int, float, int],
        p2_totals: tuple[int, int, float, int],
        expected_contribs: tuple[float, ...],
    ) -> None:
        """Each contribution is the factor's change times the midpoints of its sibling and its ancestors' siblings."""
        result = calc_tree_kpis(*_period_totals(p1_totals, p2_totals))

        expected = pd.DataFrame([expected_contribs], columns=contrib_cols)
        pd.testing.assert_frame_equal(result[contrib_cols], expected)

    def test_store_missing_from_a_period_credits_its_change_to_customers(
        self, cols: ColumnHelper, contrib_cols: list[str]
    ) -> None:
        """A store that opened or closed changed only its customer count, so Customers takes its whole spend change."""
        df = pd.DataFrame(
            {
                cols.store_id: ["Store_101", "Store_101", "Store_102", "Store_103"],
                cols.customer_id: [1, 2, 3, 4],
                cols.transaction_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 150.0, 300.0, 80.0],
                cols.unit_qty: [2, 3, 6, 4],
                "period": ["P1", "P2", "P2", "P1"],
            },
        )

        rt = RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=cols.store_id)

        # Store_102 opened in P2 ($300); Store_103 closed after P1 ($80)
        expected = pd.DataFrame(
            [(300.0, 0.0, 0.0, 0.0, 0.0, 0.0), (-80.0, 0.0, 0.0, 0.0, 0.0, 0.0)], columns=contrib_cols
        )
        pd.testing.assert_frame_equal(
            rt.df.loc[["Store_102", "Store_103"], contrib_cols].reset_index(drop=True), expected
        )

    @pytest.mark.parametrize("group_by_store", [False, True])
    @pytest.mark.parametrize(
        ("p1_value", "p2_value", "match"),
        [
            ("2023-Q1", "2023-q2", r"p2_value '2023-q2' .*\['2023-Q1', '2023-Q2'\]"),
            ("2022-Q4", "2023-Q2", r"p1_value '2022-Q4' .*\['2023-Q1', '2023-Q2'\]"),
            ("2023-Q1", "2023-Q1", "must be different"),
        ],
    )
    def test_unusable_period_values_raise(
        self, cols: ColumnHelper, group_by_store: bool, p1_value: str, p2_value: str, match: str
    ) -> None:
        """A period value with no rows (e.g. a typo) or two equal values raise instead of a misleading tree."""
        df = pd.DataFrame(
            {
                cols.store_id: ["Store_101", "Store_101"],
                cols.customer_id: [1, 2],
                cols.transaction_id: [1, 2],
                cols.unit_spend: [100.0, 150.0],
                "period": ["2023-Q1", "2023-Q2"],
            },
        )

        with pytest.raises(ValueError, match=match):
            RevenueTree(
                df=df,
                period_col="period",
                p1_value=p1_value,
                p2_value=p2_value,
                group_col=cols.store_id if group_by_store else None,
            )

    def test_date_period_labels_match_date_values(self, cols: ColumnHelper) -> None:
        """Date period values match a date column, though the backend returns the column as Timestamps."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3],
                cols.transaction_id: [1, 2, 3],
                cols.unit_spend: [100.0, 150.0, 200.0],
                "month": [dt.date(2023, 1, 1), dt.date(2023, 1, 1), dt.date(2023, 2, 1)],
            },
        )

        rt = RevenueTree(df=df, period_col="month", p1_value=dt.date(2023, 1, 1), p2_value=dt.date(2023, 2, 1))

        expected = pd.DataFrame({cols.agg.unit_spend_p1: [250.0], cols.agg.unit_spend_p2: [200.0]})
        pd.testing.assert_frame_equal(rt.df[[cols.agg.unit_spend_p1, cols.agg.unit_spend_p2]], expected)

    def test_elasticities_use_arc_percent_changes(self, cols: ColumnHelper) -> None:
        """Elasticities divide arc (midpoint) percent changes, so swapping P1 and P2 leaves them unchanged."""
        elasticity_cols = [cols.calc.frequency_elasticity, cols.calc.price_elasticity]

        result = calc_tree_kpis(*_period_totals(*ALL_FACTORS_MOVE_TOTALS))

        # Frequency: visits/customer 2->3 (+1/2.5) over spend/customer 100->180 (+80/140) = 0.7
        # Price: units 40->120 (+80/80) over price/unit 10->7.5 (-2.5/8.75) = -3.5
        expected = pd.DataFrame([(0.7, -3.5)], columns=elasticity_cols)
        pd.testing.assert_frame_equal(result[elasticity_cols], expected)

    def test_with_custom_column_names(self, cols: ColumnHelper):
        """Test RevenueTree with custom column names."""
        custom_columns = {
            "column.transaction_id": "txn_id",
            "column.customer_id": "cust_id",
            "column.unit_spend": "spend_amt",
            "column.unit_quantity": "qty",
        }

        data = {
            "cust_id": [1, 2, 3, 4],
            "txn_id": [101, 102, 103, 104],
            "spend_amt": [100.0, 200.0, 150.0, 300.0],
            "qty": [2, 3, 1, 4],
            "period": ["P1", "P1", "P2", "P2"],
        }
        df = pd.DataFrame(data)

        with option_context(*[item for pair in custom_columns.items() for item in pair]):
            rt = RevenueTree(
                df=df,
                period_col="period",
                p1_value="P1",
                p2_value="P2",
            )

            assert not rt.df.empty, "Result should not be empty"
            expected_columns = [
                cols.agg.customer_id_p1,
                cols.agg.customer_id_p2,
                cols.agg.transaction_id_p1,
                cols.agg.transaction_id_p2,
                cols.agg.unit_spend_p1,
                cols.agg.unit_spend_p2,
                cols.agg.unit_qty_p1,
                cols.agg.unit_qty_p2,
            ]

            for col in expected_columns:
                assert col in rt.df.columns, f"Expected column {col} missing from output"

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_agg_data_with_multi_group_cols(self, cols: ColumnHelper, include_quantity: bool):
        """Test the _agg_data method with multiple group columns."""
        df = pd.DataFrame(
            {
                "region": ["North", "North", "North", "South", "South", "South"],
                "store": ["Store_101", "Store_101", "Store_101", "Store_102", "Store_102", "Store_102"],
                cols.customer_id: [1, 2, 3, 4, 5, 6],
                cols.transaction_id: [1, 2, 3, 4, 5, 6],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0],
                cols.transaction_date: [
                    "2023-01-01",
                    "2023-01-05",
                    "2023-01-03",
                    "2023-01-06",
                    "2023-01-02",
                    "2023-01-04",
                ],
                "period": ["P1", "P2", "P1", "P2", "P1", "P2"],
            },
        )

        if include_quantity:
            df[cols.unit_qty] = [1, 2, 3, 4, 5, 6]

        p1_df, p2_df = RevenueTree._agg_data(
            df, period_col="period", p1_value="P1", p2_value="P2", group_col=["region", "store"]
        )

        expected_index = pd.MultiIndex.from_tuples(
            [("North", "Store_101"), ("South", "Store_102")], names=["region", "store"]
        )
        expected_p1 = pd.DataFrame(
            {cols.agg.customer_id: [2, 1], cols.agg.transaction_id: [2, 1], cols.agg.unit_spend: [400.0, 500.0]},
            index=expected_index,
        )
        expected_p2 = pd.DataFrame(
            {cols.agg.customer_id: [1, 2], cols.agg.transaction_id: [1, 2], cols.agg.unit_spend: [200.0, 1000.0]},
            index=expected_index,
        )
        if include_quantity:
            expected_p1[cols.agg.unit_qty] = [4, 5]
            expected_p2[cols.agg.unit_qty] = [2, 10]

        pd.testing.assert_frame_equal(p1_df, expected_p1)
        pd.testing.assert_frame_equal(p2_df, expected_p2)

    def test_revenue_tree_with_multi_group_cols(self, cols: ColumnHelper):
        """Test RevenueTree end-to-end with multiple group columns."""
        df = pd.DataFrame(
            {
                "region": ["North", "North", "South", "South"],
                "store": ["Store_101", "Store_101", "Store_102", "Store_102"],
                cols.customer_id: [1, 2, 3, 4],
                cols.transaction_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 200.0, 300.0, 400.0],
                "period": ["P1", "P2", "P1", "P2"],
            },
        )

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col=["region", "store"],
        )

        # Verify MultiIndex
        assert isinstance(rt.df.index, pd.MultiIndex)
        assert rt.df.index.names == ["region", "store"]
        expected_num_combinations = 2
        assert len(rt.df) == expected_num_combinations  # 2 unique combinations

        # Verify we have all the expected columns
        assert cols.agg.customer_id_p1 in rt.df.columns
        assert cols.agg.customer_id_p2 in rt.df.columns
        assert cols.agg.unit_spend_p1 in rt.df.columns
        assert cols.agg.unit_spend_p2 in rt.df.columns

    @pytest.mark.parametrize("group_by_region", [False, True])
    @pytest.mark.parametrize(
        ("stores", "spends", "expected_stores", "expected_spend_p1", "expected_spend_p2"),
        [
            # Store_101 closed after P1 and Store_102 opened in P2: two rows, not one store compared to the other
            (["Store_101", "Store_102"], [100.0, 300.0], ["Store_101", "Store_102"], [100.0, 0.0], [0.0, 300.0]),
            # Store_101 opened in P2 still sorts ahead of Store_102, which closed after P1
            (["Store_102", "Store_101"], [100.0, 300.0], ["Store_101", "Store_102"], [0.0, 100.0], [300.0, 0.0]),
            # A single store trading in both periods keeps its label
            (["Store_101", "Store_101"], [100.0, 300.0], ["Store_101"], [100.0], [300.0]),
        ],
    )
    def test_two_row_grouped_aggregate_pairs_rows_by_group(
        self,
        cols: ColumnHelper,
        group_by_region: bool,
        stores: list[str],
        spends: list[float],
        expected_stores: list[str],
        expected_spend_p1: list[float],
        expected_spend_p2: list[float],
    ) -> None:
        """Rows are paired across periods by group and sorted by group, even when the aggregate has two rows."""
        df = pd.DataFrame(
            {
                "region": ["North", "North"],
                cols.store_id: stores,
                cols.customer_id: [1, 2],
                cols.transaction_id: [1, 2],
                cols.unit_spend: spends,
                "period": ["P1", "P2"],
            },
        )
        group_col = ["region", cols.store_id] if group_by_region else cols.store_id

        rt = RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=group_col)

        expected_index = (
            pd.MultiIndex.from_arrays([["North"] * len(expected_stores), expected_stores], names=group_col)
            if group_by_region
            else pd.CategoricalIndex(expected_stores, name=cols.store_id)
        )
        expected = pd.DataFrame(
            {cols.agg.unit_spend_p1: expected_spend_p1, cols.agg.unit_spend_p2: expected_spend_p2},
            index=expected_index,
        )
        pd.testing.assert_frame_equal(rt.df[[cols.agg.unit_spend_p1, cols.agg.unit_spend_p2]], expected)

    def test_missing_group_value_gets_its_own_row(self, cols: ColumnHelper) -> None:
        """Transactions with no store recorded form a row of their own, after the named stores."""
        df = pd.DataFrame(
            {
                cols.store_id: ["Store_101", "Store_101", None, None],
                cols.customer_id: [1, 2, 3, 4],
                cols.transaction_id: [1, 2, 3, 4],
                cols.unit_spend: [100.0, 150.0, 40.0, 60.0],
                "period": ["P1", "P2", "P1", "P2"],
            },
        )

        rt = RevenueTree(df=df, period_col="period", p1_value="P1", p2_value="P2", group_col=cols.store_id)

        expected = pd.DataFrame(
            {cols.agg.unit_spend_p1: [100.0, 40.0], cols.agg.unit_spend_p2: [150.0, 60.0]},
            index=pd.CategoricalIndex(["Store_101", None], name=cols.store_id),
        )
        pd.testing.assert_frame_equal(rt.df[[cols.agg.unit_spend_p1, cols.agg.unit_spend_p2]], expected)

    @pytest.mark.parametrize("include_qty", [True, False])
    def test_draw_tree_matplotlib(self, cols: ColumnHelper, include_qty: bool):
        """Test that draw_tree() returns a matplotlib Axes and renders correctly."""
        data = {
            cols.customer_id: [1, 2, 3, 1, 2, 3],
            cols.transaction_id: [1, 2, 3, 4, 5, 6],
            cols.unit_spend: [100.0, 150.0, 200.0, 120.0, 180.0, 240.0],
            "period": ["P1", "P1", "P1", "P2", "P2", "P2"],
        }

        if include_qty:
            data[cols.unit_qty] = [10, 15, 20, 12, 18, 24]

        df = pd.DataFrame(data)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
        )

        # Test with default parameters
        ax = rt.draw_tree()
        assert isinstance(ax, Axes)
        assert ax.get_title() == ""  # TreeGrid doesn't set a title by default

        # Test with custom value labels
        ax = rt.draw_tree(value_labels=("Current", "Previous"))
        rendered = " ".join(t.get_text() for t in ax.texts)
        assert "Current" in rendered
        assert "Previous" in rendered

        # Test with custom node labels
        custom_node_labels = [
            "Sales",
            "Shoppers",
            "Sales / Shopper",
            "Trips / Shopper",
            "Sales / Trip",
        ]
        ax = rt.draw_tree(
            unit_spend_label=custom_node_labels[0],
            customer_id_label=custom_node_labels[1],
            spend_per_customer_label=custom_node_labels[2],
            transactions_per_customer_label=custom_node_labels[3],
            spend_per_transaction_label=custom_node_labels[4],
            units_per_transaction_label="Items / Trip",
            price_per_unit_label="Price / Item",
        )
        rendered_labels = {t.get_text() for t in ax.texts}
        assert set(custom_node_labels).issubset(rendered_labels)

    def test_draw_tree_with_row_index(self, cols: ColumnHelper):
        """Test that draw_tree() can visualize a specific row from a multi-group RevenueTree."""
        data = {
            "region": ["North", "North", "North", "South", "South", "South"],
            cols.customer_id: [1, 2, 1, 3, 4, 3],
            cols.transaction_id: [1, 2, 3, 4, 5, 6],
            cols.unit_spend: [100.0, 150.0, 120.0, 200.0, 250.0, 220.0],
            cols.unit_qty: [10, 15, 12, 20, 25, 22],
            "period": ["P1", "P1", "P2", "P1", "P1", "P2"],
        }

        df = pd.DataFrame(data)

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="region",
        )

        # Should have 2 rows (North and South)
        expected_regions = ["North", "South"]
        assert len(rt.df) == len(expected_regions)
        assert rt.df.index.tolist() == expected_regions

        # Draw first row (North region)
        ax = rt.draw_tree(row_index=0)
        assert isinstance(ax, Axes)
        # Check that the North region's revenue value appears in the plot
        text_strings = [t.get_text() for t in ax.texts]
        # North P2 revenue is 120, P1 revenue is 250 (100 + 150)
        # The formatted values should appear in the text
        assert "120" in text_strings, f"North region P2 revenue '120' should appear in {text_strings}"

        # Draw second row (South region)
        ax = rt.draw_tree(row_index=1)
        assert isinstance(ax, Axes)
        text_strings = [t.get_text() for t in ax.texts]
        # South P2 revenue is 220, P1 revenue is 450 (200 + 250)
        assert "220" in text_strings, f"South region P2 revenue '220' should appear in {text_strings}"

        # Test that out of bounds raises IndexError
        with pytest.raises(IndexError):
            rt.draw_tree(row_index=2)

    @pytest.mark.parametrize("include_quantity", [True, False])
    def test_pct_diff_returns_nan_when_p1_is_zero(self, cols: ColumnHelper, include_quantity: bool):
        """Test that percentage difference produces NaN instead of inf when period 1 values are zero."""
        df = pd.DataFrame(
            {
                cols.agg.customer_id: [0, 5],
                cols.agg.transaction_id: [0, 5],
                cols.agg.unit_spend: [0.0, 500.0],
            },
            index=["p1", "p2"],
        )

        if include_quantity:
            df[cols.agg.unit_qty] = [0, 20]

        result = calc_tree_kpis(*_ungrouped_periods(df))

        # All percentage difference columns should be NaN (not inf) when p1 is zero
        for pct_col in [
            cols.agg.customer_id_pct_diff,
            cols.agg.transaction_id_pct_diff,
            cols.agg.unit_spend_pct_diff,
            cols.calc.spend_per_cust_pct_diff,
            cols.calc.spend_per_trans_pct_diff,
            cols.calc.trans_per_cust_pct_diff,
        ]:
            val = result[pct_col].iloc[0]
            assert math.isnan(val), f"{pct_col} should be NaN when p1 is zero, got {val}"

        if include_quantity:
            for pct_col in [
                cols.agg.unit_qty_pct_diff,
                cols.calc.units_per_trans_pct_diff,
                cols.calc.price_per_unit_pct_diff,
            ]:
                val = result[pct_col].iloc[0]
                assert math.isnan(val), f"{pct_col} should be NaN when p1 is zero, got {val}"

    @pytest.mark.parametrize(
        ("customer_id", "transaction_id", "unit_spend", "unit_qty"),
        [
            pytest.param([3, 3], [3, 3], [900.0, 900.0], [10, 10], id="identical_periods"),
            pytest.param([0, 0], [0, 0], [0.0, 0.0], [0, 0], id="all_zeros"),
        ],
    )
    def test_elasticity_returns_nan_for_zero_divisors(
        self,
        cols: ColumnHelper,
        customer_id,
        transaction_id,
        unit_spend,
        unit_qty,
    ):
        """Test that elasticity produces NaN instead of inf when divisors are zero.

        Covers identical periods (0/0 from zero pct change) and all-zero values (zero midpoint averages).
        """
        df = pd.DataFrame(
            {
                cols.agg.customer_id: customer_id,
                cols.agg.transaction_id: transaction_id,
                cols.agg.unit_spend: unit_spend,
                cols.agg.unit_qty: unit_qty,
            },
            index=["p1", "p2"],
        )

        result = calc_tree_kpis(*_ungrouped_periods(df))

        freq_elasticity = result[cols.calc.frequency_elasticity].iloc[0]
        assert math.isnan(freq_elasticity), f"frequency_elasticity should be NaN, got {freq_elasticity}"

        price_elasticity = result[cols.calc.price_elasticity].iloc[0]
        assert math.isnan(price_elasticity), f"price_elasticity should be NaN, got {price_elasticity}"

    def test_no_inf_values_with_p2_only_group(self, cols: ColumnHelper):
        """Test that a group with data only in P2 (new product) produces NaN, not inf."""
        df = pd.DataFrame(
            {
                "group_id": [1, 1, 2],
                cols.customer_id: [1, 2, 3],
                cols.transaction_id: [1, 2, 3],
                cols.unit_spend: [100.0, 200.0, 300.0],
                cols.unit_qty: [5, 10, 15],
                "period": ["P1", "P2", "P2"],
            },
        )

        rt = RevenueTree(
            df=df,
            period_col="period",
            p1_value="P1",
            p2_value="P2",
            group_col="group_id",
        )

        # Group 2 has no P1 data, so every pct_diff, P1 ratio and elasticity built from them is undefined
        group2_data = rt.df.loc[2]

        expected_nan_cols = [
            cols.calc.frequency_elasticity,
            cols.calc.price_elasticity,
            cols.agg.customer_id_pct_diff,
            cols.agg.transaction_id_pct_diff,
            cols.agg.unit_spend_pct_diff,
            cols.calc.spend_per_cust_pct_diff,
            cols.calc.spend_per_trans_pct_diff,
            cols.calc.trans_per_cust_pct_diff,
            cols.agg.unit_qty_pct_diff,
            cols.calc.units_per_trans_pct_diff,
            cols.calc.price_per_unit_pct_diff,
        ]
        for col_name in expected_nan_cols:
            val = group2_data[col_name]
            assert math.isnan(val), f"{col_name} should be NaN for P2-only group, got {val}"
