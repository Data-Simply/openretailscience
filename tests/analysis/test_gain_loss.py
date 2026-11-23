"""Tests for the GainLoss class in the gain_loss module."""

from __future__ import annotations

import datetime as dt

import ibis
import pandas as pd
import pytest
from ibis.common.exceptions import IbisTypeError

from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()

FOCUS = "Meadow Farm"
COMPARISON = "Greenvale"
OTHER_BRAND = "Hillside"
P1 = "2025Q1"
P2 = "2025Q2"
SELECTION = {
    "period_col": "period",
    "p1_value": P1,
    "p2_value": P2,
    "compare_col": "brand",
    "focus_value": FOCUS,
    "comparison_value": COMPARISON,
}

COMPONENTS = [
    "new",
    "lost",
    "increased_focus",
    "decreased_focus",
    "switch_from_comparison",
    "switch_to_comparison",
]


def _transactions(rows: list[tuple[int, str, str, int, int]]) -> pd.DataFrame:
    """Build transactions from ``(customer_id, period, brand, store_id, unit_spend)`` rows."""
    return pd.DataFrame(rows, columns=[cols.customer_id, "period", "brand", cols.store_id, cols.unit_spend])


def _gain_loss(df: pd.DataFrame | ibis.Table, **kwargs: str | None) -> GainLoss:
    return GainLoss(df=df, **{**SELECTION, **kwargs})


def _customers(gl: GainLoss) -> pd.DataFrame:
    """The per-customer frame, sorted: customer_table has no row order."""
    customers = gl.customer_table.execute()
    keys = [col for col in (cols.store_id, cols.customer_id) if col in customers.columns]
    return customers.sort_values(keys).reset_index(drop=True)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """One customer per decomposition component, plus an other-brand-only customer and a store mover.

    1001 new, 1002 lost (and a 2024Q4 sale in neither period), 1003 increased, 1004 decreased,
    1005 increased and switched from Greenvale, 1006 switched to Greenvale, 1007 buys only Hillside,
    and 1008 buys the same Meadow Farm spend in both periods but moves from store 201 to 202.
    """
    return _transactions(
        [
            (1001, P2, FOCUS, 201, 120),
            (1002, P1, FOCUS, 201, 90),
            (1002, "2024Q4", FOCUS, 201, 75),
            (1003, P1, FOCUS, 201, 100),
            (1003, P1, COMPARISON, 201, 100),
            (1003, P2, FOCUS, 201, 150),
            (1003, P2, COMPARISON, 201, 120),
            (1004, P1, FOCUS, 201, 100),
            (1004, P1, COMPARISON, 201, 100),
            (1004, P2, FOCUS, 201, 50),
            (1004, P2, COMPARISON, 201, 80),
            (1005, P1, FOCUS, 202, 100),
            (1005, P1, COMPARISON, 202, 100),
            (1005, P2, FOCUS, 202, 150),
            (1005, P2, COMPARISON, 202, 80),
            (1006, P1, FOCUS, 202, 100),
            (1006, P1, COMPARISON, 202, 100),
            (1006, P2, FOCUS, 202, 80),
            (1006, P2, COMPARISON, 202, 150),
            (1007, P1, OTHER_BRAND, 202, 40),
            (1007, P2, OTHER_BRAND, 202, 55),
            (1008, P1, FOCUS, 201, 60),
            (1008, P2, FOCUS, 202, 60),
        ],
    )


class TestDecomposition:
    """How one customer's change in focus spend splits into the six components."""

    @pytest.mark.parametrize(
        ("focus_p1", "comparison_p1", "focus_p2", "comparison_p2", "expected"),
        [
            pytest.param(0, 0, 120, 0, (120, 0, 0, 0, 0, 0), id="new_customer"),
            pytest.param(0, 0, 120, 40, (120, 0, 0, 0, 0, 0), id="new_to_both_counts_all_focus_as_new"),
            pytest.param(90, 0, 0, 0, (0, -90, 0, 0, 0, 0), id="lost_customer"),
            pytest.param(90, 60, 0, 0, (0, -90, 0, 0, 0, 0), id="lost_from_both_counts_all_focus_as_lost"),
            pytest.param(100, 100, 150, 120, (0, 0, 50, 0, 0, 0), id="both_grow"),
            pytest.param(100, 100, 130, 100, (0, 0, 30, 0, 0, 0), id="focus_grows_comparison_flat"),
            pytest.param(100, 100, 150, 80, (0, 0, 30, 0, 20, 0), id="focus_gain_exceeds_comparison_loss"),
            pytest.param(100, 100, 110, 50, (0, 0, 0, 0, 10, 0), id="comparison_loss_exceeds_focus_gain"),
            pytest.param(100, 100, 150, 50, (0, 0, 0, 0, 50, 0), id="focus_gain_equals_comparison_loss"),
            pytest.param(100, 100, 50, 80, (0, 0, 0, -50, 0, 0), id="both_shrink"),
            pytest.param(100, 100, 70, 100, (0, 0, 0, -30, 0, 0), id="focus_shrinks_comparison_flat"),
            pytest.param(100, 100, 50, 110, (0, 0, 0, -40, 0, -10), id="focus_loss_exceeds_comparison_gain"),
            pytest.param(100, 100, 80, 150, (0, 0, 0, 0, 0, -20), id="comparison_gain_exceeds_focus_loss"),
            pytest.param(100, 100, 50, 150, (0, 0, 0, 0, 0, -50), id="focus_loss_equals_comparison_gain"),
            pytest.param(100, 100, 100, 150, (0, 0, 0, 0, 0, 0), id="focus_flat_comparison_grows"),
            pytest.param(100, 100, 100, 50, (0, 0, 0, 0, 0, 0), id="focus_flat_comparison_shrinks"),
            pytest.param(0, 80, 80, 0, (0, 0, 0, 0, 80, 0), id="switched_entirely_to_focus"),
            pytest.param(80, 0, 0, 80, (0, 0, 0, 0, 0, -80), id="switched_entirely_to_comparison"),
            pytest.param(0, 50, 0, 70, (0, 0, 0, 0, 0, 0), id="comparison_only"),
        ],
    )
    def test_customer_change_splits_into_components(
        self,
        focus_p1: int,
        comparison_p1: int,
        focus_p2: int,
        comparison_p2: int,
        expected: tuple[int, ...],
    ) -> None:
        """Each branch of the decomposition, for one customer's Meadow Farm and Greenvale spend in each quarter."""
        spends = [
            (P1, FOCUS, focus_p1),
            (P1, COMPARISON, comparison_p1),
            (P2, FOCUS, focus_p2),
            (P2, COMPARISON, comparison_p2),
        ]
        df = _transactions([(1001, period, brand, 201, spend) for period, brand, spend in spends if spend != 0])

        result = _customers(_gain_loss(df))[COMPONENTS]

        pd.testing.assert_frame_equal(result, pd.DataFrame([expected], columns=COMPONENTS))


class TestResults:
    """The per-customer table and the summary."""

    def test_customer_frame_and_table(self, transactions: pd.DataFrame) -> None:
        """One row per customer with focus or comparison sales; other-period and other-brand sales are left out."""
        gl = _gain_loss(transactions)

        expected = pd.DataFrame(
            {
                cols.customer_id: [1001, 1002, 1003, 1004, 1005, 1006, 1008],
                "focus_p1": [0, 90, 100, 100, 100, 100, 60],
                "comparison_p1": [0, 0, 100, 100, 100, 100, 0],
                "total_p1": [0, 90, 200, 200, 200, 200, 60],
                "focus_p2": [120, 0, 150, 50, 150, 80, 60],
                "comparison_p2": [0, 0, 120, 80, 80, 150, 0],
                "total_p2": [120, 0, 270, 130, 230, 230, 60],
                "focus_diff": [120, -90, 50, -50, 50, -20, 0],
                "comparison_diff": [0, 0, 20, -20, -20, 50, 0],
                "total_diff": [120, -90, 70, -70, 30, 30, 0],
                "new": [120, 0, 0, 0, 0, 0, 0],
                "lost": [0, -90, 0, 0, 0, 0, 0],
                "increased_focus": [0, 0, 50, 0, 30, 0, 0],
                "decreased_focus": [0, 0, 0, -50, 0, 0, 0],
                "switch_from_comparison": [0, 0, 0, 0, 20, 0, 0],
                "switch_to_comparison": [0, 0, 0, 0, 0, -20, 0],
            },
        )
        pd.testing.assert_frame_equal(_customers(gl), expected)
        pd.testing.assert_frame_equal(gl.df, expected.drop(columns=cols.customer_id).sum().to_frame().T)

    def test_grouped_customer_frame_and_table(self, transactions: pd.DataFrame) -> None:
        """Grouped, each customer is decomposed per store: 1008's move is lost at store 201 and new at 202."""
        gl = _gain_loss(transactions, group_col=cols.store_id)

        expected = pd.DataFrame(
            {
                cols.store_id: [201, 201, 201, 201, 201, 202, 202, 202],
                cols.customer_id: [1001, 1002, 1003, 1004, 1008, 1005, 1006, 1008],
                "focus_p1": [0, 90, 100, 100, 60, 100, 100, 0],
                "comparison_p1": [0, 0, 100, 100, 0, 100, 100, 0],
                "total_p1": [0, 90, 200, 200, 60, 200, 200, 0],
                "focus_p2": [120, 0, 150, 50, 0, 150, 80, 60],
                "comparison_p2": [0, 0, 120, 80, 0, 80, 150, 0],
                "total_p2": [120, 0, 270, 130, 0, 230, 230, 60],
                "focus_diff": [120, -90, 50, -50, -60, 50, -20, 60],
                "comparison_diff": [0, 0, 20, -20, 0, -20, 50, 0],
                "total_diff": [120, -90, 70, -70, -60, 30, 30, 60],
                "new": [120, 0, 0, 0, 0, 0, 0, 60],
                "lost": [0, -90, 0, 0, -60, 0, 0, 0],
                "increased_focus": [0, 0, 50, 0, 0, 30, 0, 0],
                "decreased_focus": [0, 0, 0, -50, 0, 0, 0, 0],
                "switch_from_comparison": [0, 0, 0, 0, 0, 20, 0, 0],
                "switch_to_comparison": [0, 0, 0, 0, 0, 0, -20, 0],
            },
        )
        pd.testing.assert_frame_equal(_customers(gl), expected)
        pd.testing.assert_frame_equal(
            gl.df,
            expected.drop(columns=cols.customer_id).groupby(cols.store_id, as_index=False).sum(),
        )

    def test_grouped_store_missing_from_a_period_or_brand_gets_integer_zeros(self) -> None:
        """Store 203 has only a Meadow Farm Q1 sale, so its Greenvale and Q2 values are integer 0s, not NaN."""
        df = _transactions(
            [
                (1003, P1, FOCUS, 201, 100),
                (1003, P1, COMPARISON, 201, 100),
                (1003, P2, FOCUS, 201, 150),
                (1003, P2, COMPARISON, 201, 120),
                (1002, P1, FOCUS, 203, 90),
            ],
        )

        result = _gain_loss(df, group_col=cols.store_id).df

        expected = pd.DataFrame(
            {
                cols.store_id: [201, 203],
                "focus_p1": [100, 90],
                "comparison_p1": [100, 0],
                "total_p1": [200, 90],
                "focus_p2": [150, 0],
                "comparison_p2": [120, 0],
                "total_p2": [270, 0],
                "focus_diff": [50, -90],
                "comparison_diff": [20, 0],
                "total_diff": [70, -90],
                "new": [0, 0],
                "lost": [0, -90],
                "increased_focus": [50, 0],
                "decreased_focus": [0, 0],
                "switch_from_comparison": [0, 0],
                "switch_to_comparison": [0, 0],
            },
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_null_group_is_its_own_group(self, transactions: pd.DataFrame) -> None:
        """Sales with no store form their own row, sorted last, so no sales drop out of the grouped table."""
        storeless_customer = 1001
        no_store = transactions[cols.customer_id] == storeless_customer
        df = transactions.assign(**{cols.store_id: transactions[cols.store_id].astype("Int64").mask(no_store)})

        result = _gain_loss(df, group_col=cols.store_id).df

        # The null makes store_id float on the way back from the backend
        expected = pd.DataFrame(
            {
                cols.store_id: [201.0, 202.0, None],
                "focus_p1": [350, 200, 0],
                "comparison_p1": [200, 200, 0],
                "total_p1": [550, 400, 0],
                "focus_p2": [200, 290, 120],
                "comparison_p2": [200, 230, 0],
                "total_p2": [400, 520, 120],
                "focus_diff": [-150, 90, 120],
                "comparison_diff": [0, 30, 0],
                "total_diff": [-150, 120, 120],
                "new": [0, 60, 120],
                "lost": [-150, 0, 0],
                "increased_focus": [50, 30, 0],
                "decreased_focus": [-50, 0, 0],
                "switch_from_comparison": [0, 20, 0],
                "switch_to_comparison": [0, -20, 0],
            },
        )
        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("guest_id", "unknown_customer_value"),
        [pytest.param(pd.NA, None, id="null_id"), pytest.param(-1, -1, id="unknown_customer_value")],
    )
    def test_unidentified_customer_rows_are_left_out(
        self, transactions: pd.DataFrame, guest_id: int | None, unknown_customer_value: int | None
    ) -> None:
        """Guest sales, under a null ID or the data.unknown_customer_value ID, can't be decomposed per customer."""
        guests = _transactions([(0, P1, FOCUS, 201, 500), (0, P2, COMPARISON, 201, 700)])
        df = pd.concat([transactions, guests], ignore_index=True)
        df[cols.customer_id] = df[cols.customer_id].astype("Int64").mask(df[cols.customer_id] == 0, guest_id)

        with option_context("data.unknown_customer_value", unknown_customer_value):
            result = _gain_loss(df).df

        pd.testing.assert_frame_equal(result, _gain_loss(transactions).df)

    def test_grouped_table_has_no_nested_order_by(self, transactions: pd.DataFrame) -> None:
        """SQL Server rejects ORDER BY in a derived table, so table is left unordered and df sorts instead."""
        table = _gain_loss(transactions, group_col=cols.store_id).table

        assert "ORDER BY" not in ibis.to_sql(table, dialect="mssql").upper()

    @pytest.mark.parametrize(
        ("agg_func", "expected"),
        [
            ("count", [2, 1, 3, 1, 1, 2]),
            ("nunique", [1, 1, 1, 1, 1, 2]),
        ],
    )
    def test_agg_func_aggregates_each_period_and_group(self, agg_func: str, expected: list[int]) -> None:
        """agg_func is applied to each period's focus, comparison and combined transactions."""
        df = _transactions(
            [
                (1003, P1, FOCUS, 201, 60),
                (1003, P1, FOCUS, 201, 60),
                (1003, P1, COMPARISON, 201, 60),
                (1003, P2, FOCUS, 201, 150),
                (1003, P2, COMPARISON, 201, 120),
            ],
        )
        period_cols = ["focus_p1", "comparison_p1", "total_p1", "focus_p2", "comparison_p2", "total_p2"]

        result = _customers(_gain_loss(df, agg_func=agg_func))[period_cols]

        pd.testing.assert_frame_equal(result, pd.DataFrame([expected], columns=period_cols))

    def test_value_col_replaces_spend(self, transactions: pd.DataFrame) -> None:
        """value_col decomposes the named column in place of unit_spend."""
        quantities = transactions[cols.unit_spend] // 10
        expected = _customers(_gain_loss(transactions.assign(**{cols.unit_spend: quantities})))

        result = _customers(
            _gain_loss(transactions.assign(**{cols.unit_qty: quantities}), value_col=cols.unit_qty),
        )

        pd.testing.assert_frame_equal(result, expected)

    def test_custom_column_names_from_options(self, transactions: pd.DataFrame) -> None:
        """The customer and spend columns come from the column options."""
        renames = {cols.customer_id: "loyalty_card_id", cols.unit_spend: "net_sales"}
        expected = _customers(_gain_loss(transactions)).rename(columns=renames)

        with option_context("column.customer_id", "loyalty_card_id", "column.unit_spend", "net_sales"):
            result = _gain_loss(transactions.rename(columns=renames)).customer_table.execute()

        pd.testing.assert_frame_equal(result.sort_values("loyalty_card_id").reset_index(drop=True), expected)


class TestValidation:
    """Invalid input raises with a message naming the problem."""

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"p2_value": P1}, "p1_value and p2_value must be different", id="same_period"),
            pytest.param(
                {"comparison_value": FOCUS},
                "focus_value and comparison_value must be different",
                id="same_group",
            ),
            pytest.param({"agg_func": "mean"}, "agg_func must be one of", id="non_additive_agg_func"),
            pytest.param(
                {"group_col": [cols.store_id, "brand"]},
                "group_col must be a single column",
                id="several_group_cols",
            ),
            pytest.param({"group_col": "brand"}, "group_col must differ from", id="group_col_is_compare_col"),
            pytest.param({"group_col": "period"}, "group_col must differ from", id="group_col_is_period_col"),
            pytest.param(
                {"compare_col": "period"},
                "period_col and compare_col must be different",
                id="compare_col_is_period_col",
            ),
            pytest.param({"value_col": "brand"}, "agg_func 'sum' is not an aggregation available", id="sum_of_text"),
        ],
    )
    def test_invalid_argument_raises(self, transactions: pd.DataFrame, kwargs: dict[str, str], match: str) -> None:
        """Arguments that can't give a meaningful decomposition raise before any query runs."""
        with pytest.raises(ValueError, match=match):
            _gain_loss(transactions, **kwargs)

    @pytest.mark.parametrize("param", ["p1_value", "p2_value", "focus_value", "comparison_value"])
    def test_value_of_another_type_raises(self, transactions: pd.DataFrame, param: str) -> None:
        """An int on a text column raises when the queries are built, not from the backend when df runs them."""
        with pytest.raises(IbisTypeError, match="not comparable"):
            _gain_loss(transactions, **{param: 2025})

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"p1_value": "2025-Q1"}, "p1_value '2025-Q1' matches no rows", id="p1_value"),
            pytest.param({"p2_value": "2025-Q2"}, "p2_value '2025-Q2' matches no rows", id="p2_value"),
            pytest.param({"focus_value": "Meadow farm"}, "focus_value 'Meadow farm' matches no rows", id="focus"),
            pytest.param(
                {"comparison_value": "Green vale"},
                "comparison_value 'Green vale' matches no rows",
                id="comparison",
            ),
            # No row is left to count, so the brands, which do exist, aren't reported as matching nothing
            pytest.param(
                {"p1_value": "2025-Q1", "p2_value": "2025-Q2"},
                "No rows have 'period' '2025-Q1' or '2025-Q2' and 'brand' 'Meadow Farm' or 'Greenvale'",
                id="both_periods",
            ),
        ],
    )
    def test_value_matching_no_rows_raises(
        self, transactions: pd.DataFrame, kwargs: dict[str, str], match: str
    ) -> None:
        """A misspelt period or brand raises instead of giving a plausible-looking decomposition."""
        gl = _gain_loss(transactions, **kwargs)

        with pytest.raises(ValueError, match=match):
            _ = gl.df

    def test_same_period_as_text_and_date_raises(self, transactions: pd.DataFrame) -> None:
        """A quarter passed once as text and once as a date selects the same rows, which aren't counted twice."""
        quarter_starts = {P1: dt.date(2025, 1, 1), P2: dt.date(2025, 4, 1), "2024Q4": dt.date(2024, 10, 1)}
        df = transactions.assign(period=transactions["period"].map(quarter_starts))
        gl = _gain_loss(df, p1_value="2025-01-01", p2_value=dt.date(2025, 1, 1))

        with pytest.raises(ValueError, match=r"p2_value datetime\.date\(2025, 1, 1\) matches no rows"):
            _ = gl.df

    @pytest.mark.parametrize(
        ("missing_col", "kwargs", "match"),
        [
            pytest.param(cols.customer_id, {}, "Input data is missing required columns", id="customer_col"),
            pytest.param(cols.unit_spend, {}, "Input data is missing required columns", id="value_col"),
            pytest.param("period", {}, "Input data is missing required columns", id="period_col"),
            pytest.param("brand", {}, "Input data is missing required columns", id="compare_col"),
            pytest.param(
                cols.store_id,
                {"group_col": cols.store_id},
                "group_col references columns not present in the data",
                id="group_col",
            ),
        ],
    )
    def test_missing_column_raises(
        self, transactions: pd.DataFrame, missing_col: str, kwargs: dict[str, str], match: str
    ) -> None:
        """A df without a required column raises naming it, rather than failing inside the query."""
        df = transactions.drop(columns=missing_col)

        with pytest.raises(ValueError, match=f"{match}.*{missing_col}"):
            _gain_loss(df, **kwargs)
