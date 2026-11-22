"""Tests for the GainLoss class in the gain_loss module."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.options import ColumnHelper, option_context
from openretailscience.plots.styles.colors import COLORS

if TYPE_CHECKING:
    from collections.abc import Callable

cols = ColumnHelper()

FOCUS = "Meadow Farm"
COMPARISON = "Greenvale"
OTHER_BRAND = "Hillside"
P1 = "2025Q1"
P2 = "2025Q2"
BRAND_NAMES = {"focus_group_name": FOCUS, "comparison_group_name": COMPARISON}

COMPONENTS = [
    "new",
    "lost",
    "increased_focus",
    "decreased_focus",
    "switch_from_comparison",
    "switch_to_comparison",
]


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


def _transactions(rows: list[tuple[int, str, str, int, int]]) -> pd.DataFrame:
    """Build transactions from ``(customer_id, period, brand, store_id, unit_spend)`` rows."""
    return pd.DataFrame(rows, columns=[cols.customer_id, "period", "brand", cols.store_id, cols.unit_spend])


def _masks(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {
        "p1_index": df["period"] == P1,
        "p2_index": df["period"] == P2,
        "focus_group_index": df["brand"] == FOCUS,
        "comparison_group_index": df["brand"] == COMPARISON,
    }


def _gain_loss(df: pd.DataFrame, **kwargs: str | None) -> GainLoss:
    return GainLoss(df=df, **_masks(df), **BRAND_NAMES, **kwargs)


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

        result = _gain_loss(df).gain_loss_df[COMPONENTS]

        pd.testing.assert_frame_equal(result, pd.DataFrame([expected], columns=COMPONENTS))


class TestResults:
    """The per-customer and summary frames."""

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
        pd.testing.assert_frame_equal(gl.gain_loss_df, expected)
        pd.testing.assert_frame_equal(gl.gain_loss_table_df, expected.drop(columns=cols.customer_id).sum().to_frame().T)

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
        pd.testing.assert_frame_equal(gl.gain_loss_df, expected)
        pd.testing.assert_frame_equal(
            gl.gain_loss_table_df,
            expected.drop(columns=cols.customer_id).groupby(cols.store_id, as_index=False).sum(),
        )

    def test_grouped_store_missing_from_a_period_or_brand_gets_zeros(self) -> None:
        """Store 203 has only a Meadow Farm Q1 sale, so its Greenvale and Q2 values are 0, not NaN."""
        df = _transactions(
            [
                (1003, P1, FOCUS, 201, 100),
                (1003, P1, COMPARISON, 201, 100),
                (1003, P2, FOCUS, 201, 150),
                (1003, P2, COMPARISON, 201, 120),
                (1002, P1, FOCUS, 203, 90),
            ],
        )

        result = _gain_loss(df, group_col=cols.store_id).gain_loss_table_df

        # Pins today's data-dependent dtypes: NaN-filled columns are float, which makes non-zero components float too
        expected = pd.DataFrame(
            {
                cols.store_id: [201, 203],
                "focus_p1": [100, 90],
                "comparison_p1": [100.0, 0.0],
                "total_p1": [200, 90],
                "focus_p2": [150.0, 0.0],
                "comparison_p2": [120.0, 0.0],
                "total_p2": [270.0, 0.0],
                "focus_diff": [50.0, -90.0],
                "comparison_diff": [20.0, 0.0],
                "total_diff": [70.0, -90.0],
                "new": [0, 0],
                "lost": [0.0, -90.0],
                "increased_focus": [50.0, 0.0],
                "decreased_focus": [0, 0],
                "switch_from_comparison": [0, 0],
                "switch_to_comparison": [0, 0],
            },
        )
        pd.testing.assert_frame_equal(result, expected)

    def test_grouped_table_omits_unused_group_categories(self, transactions: pd.DataFrame) -> None:
        """A categorical group_col's unused category (a store with no sales) gets no table row."""
        df = transactions.astype({cols.store_id: pd.CategoricalDtype([201, 202, 203])})

        gl = _gain_loss(df, group_col=cols.store_id)

        assert gl.gain_loss_table_df[cols.store_id].tolist() == [201, 202]

    @pytest.mark.parametrize(
        ("agg_func", "expected"),
        [
            ("count", [2, 1, 3, 1, 1, 2]),
            ("mean", [50.0, 100.0, 200 / 3, 150.0, 120.0, 135.0]),
            ("max", [60, 100, 100, 150, 120, 150]),
        ],
    )
    def test_agg_func_aggregates_each_period_and_group(self, agg_func: str, expected: list[float]) -> None:
        """agg_func is applied to each period's focus, comparison and combined transactions."""
        df = _transactions(
            [
                (1003, P1, FOCUS, 201, 60),
                (1003, P1, FOCUS, 201, 40),
                (1003, P1, COMPARISON, 201, 100),
                (1003, P2, FOCUS, 201, 150),
                (1003, P2, COMPARISON, 201, 120),
            ],
        )
        period_cols = ["focus_p1", "comparison_p1", "total_p1", "focus_p2", "comparison_p2", "total_p2"]

        result = _gain_loss(df, agg_func=agg_func).gain_loss_df[period_cols]

        pd.testing.assert_frame_equal(result, pd.DataFrame([expected], columns=period_cols))

    def test_value_col_replaces_spend(self, transactions: pd.DataFrame) -> None:
        """value_col decomposes the named column in place of unit_spend."""
        quantities = transactions[cols.unit_spend] // 10
        expected = _gain_loss(transactions.assign(**{cols.unit_spend: quantities})).gain_loss_df

        result = _gain_loss(transactions.assign(**{cols.unit_qty: quantities}), value_col=cols.unit_qty).gain_loss_df

        pd.testing.assert_frame_equal(result, expected)

    def test_custom_column_names_from_options(self, transactions: pd.DataFrame) -> None:
        """The customer and spend columns come from the column options."""
        renames = {cols.customer_id: "loyalty_card_id", cols.unit_spend: "net_sales"}
        expected = _gain_loss(transactions).gain_loss_df.rename(columns=renames)

        with option_context("column.customer_id", "loyalty_card_id", "column.unit_spend", "net_sales"):
            result = _gain_loss(transactions.rename(columns=renames)).gain_loss_df

        pd.testing.assert_frame_equal(result, expected)

    def test_groups_with_no_sales_give_an_all_zero_table(self, transactions: pd.DataFrame) -> None:
        """Focus and comparison brands with no sales in either period (e.g. misspelt) give zeros, not a crash."""
        masks = {
            **_masks(transactions),
            "focus_group_index": transactions["brand"] == "Meadow farm",
            "comparison_group_index": transactions["brand"] == "Green vale",
        }
        gl = GainLoss(df=transactions, **masks, **BRAND_NAMES)

        expected = _gain_loss(transactions).gain_loss_table_df * 0
        pd.testing.assert_frame_equal(gl.gain_loss_table_df, expected)
        assert len(gl.gain_loss_df) == 0


class TestMasks:
    """The four positional boolean masks."""

    # astype(object) mimics a nullable flag column read from parquet: after dropna it is object dtype of plain bools
    @pytest.mark.parametrize(
        "to_mask", [list, np.asarray, lambda mask: mask.astype(object)], ids=["list", "ndarray", "object_series"]
    )
    def test_list_array_and_object_masks_match_boolean_series(
        self, transactions: pd.DataFrame, to_mask: Callable[[pd.Series], list[bool] | np.ndarray | pd.Series]
    ) -> None:
        """Lists, numpy arrays and object-dtype Series of bools decompose the same as boolean Series."""
        expected = _gain_loss(transactions).gain_loss_df

        masks = {k: to_mask(v) for k, v in _masks(transactions).items()}
        result = GainLoss(df=transactions, **masks, **BRAND_NAMES).gain_loss_df

        pd.testing.assert_frame_equal(result, expected)

    def test_non_unique_index_matches_default_index(self, transactions: pd.DataFrame) -> None:
        """Masks select rows by position, so a df with repeated index labels (from pd.concat) decomposes the same."""
        expected = _gain_loss(transactions).gain_loss_df
        half = len(transactions) // 2
        concatenated_df = pd.concat([transactions.iloc[:half], transactions.iloc[half:].reset_index(drop=True)])

        result = _gain_loss(concatenated_df).gain_loss_df

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize("brand_dtype", ["string", "string[pyarrow]"])
    def test_nullable_boolean_masks_leave_missing_rows_out(self, transactions: pd.DataFrame, brand_dtype: str) -> None:
        """Comparing a nullable brand column gives <NA> where the brand is missing; those rows join neither group."""
        unbranded_customer = 1005
        unbranded_q1 = (transactions[cols.customer_id] == unbranded_customer) & (transactions["period"] == P1)
        brand = transactions["brand"].astype(brand_dtype).mask(unbranded_q1)
        unbranded_df = transactions.assign(brand=brand)
        masks = _masks(unbranded_df)
        filled_masks = {k: v.fillna(False).astype(bool) for k, v in masks.items()}
        expected = GainLoss(df=unbranded_df, **filled_masks, **BRAND_NAMES).gain_loss_df

        result = GainLoss(df=unbranded_df, **masks, **BRAND_NAMES).gain_loss_df

        pd.testing.assert_frame_equal(result, expected)


class TestValidation:
    """Invalid input raises before any computation."""

    @pytest.mark.parametrize(
        ("mask_name", "make_mask", "error", "match"),
        [
            pytest.param(
                "p2_index",
                lambda df: df["period"].isin([P1, P2]),
                ValueError,
                "p1_index and p2_index should not overlap",
                id="overlapping_periods",
            ),
            pytest.param(
                "comparison_group_index",
                lambda df: df["brand"] != OTHER_BRAND,
                ValueError,
                "focus_group_index and comparison_group_index should not overlap",
                id="overlapping_groups",
            ),
            # Same length, but positions would pick the wrong rows
            pytest.param(
                "focus_group_index",
                lambda df: df.sort_values(cols.unit_spend)["brand"] == FOCUS,
                ValueError,
                r"index must match df\.index",
                id="series_from_reordered_frame",
            ),
            pytest.param(
                "focus_group_index",
                lambda df: list(df["brand"] == FOCUS)[:-1],
                ValueError,
                "one entry per row of df",
                id="list_one_short",
            ),
            pytest.param(
                "focus_group_index",
                lambda df: df.iloc[1:]["brand"] == FOCUS,
                ValueError,
                "one entry per row of df",
                id="series_from_filtered_frame",
            ),
            pytest.param(
                "focus_group_index",
                lambda df: df["brand"],
                TypeError,
                r'must be boolean; .* nullable "boolean" dtype',
                id="brand_names",
            ),
            # NaN coerces to True, so coercing would put every other brand in the focus group
            pytest.param(
                "focus_group_index",
                lambda df: df["brand"].map({FOCUS: True}),
                TypeError,
                r'must be boolean; .* nullable "boolean" dtype',
                id="series_with_nan",
            ),
            pytest.param(
                "focus_group_index",
                lambda df: list(df["brand"].map({FOCUS: True})),
                TypeError,
                r'must be boolean; .* nullable "boolean" dtype',
                id="list_with_nan",
            ),
        ],
    )
    def test_invalid_mask_raises(
        self,
        transactions: pd.DataFrame,
        mask_name: str,
        make_mask: Callable[[pd.DataFrame], pd.Series | list[bool | float]],
        error: type[ValueError | TypeError],
        match: str,
    ) -> None:
        """Overlapping, misaligned or non-boolean masks raise rather than being misapplied or coerced to True/False."""
        masks = {**_masks(transactions), mask_name: make_mask(transactions)}

        with pytest.raises(error, match=match):
            GainLoss(df=transactions, **masks, **BRAND_NAMES)

    @pytest.mark.parametrize(
        ("missing_col", "kwargs", "match"),
        [
            pytest.param(cols.customer_id, {}, "Input data is missing required columns", id="customer_col"),
            pytest.param(cols.unit_spend, {}, "Input data is missing required columns", id="value_col"),
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
        """A df without the customer, value or group column raises naming it, rather than failing inside the groupby."""
        df = transactions.drop(columns=missing_col)

        with pytest.raises(ValueError, match=f"{match}.*{missing_col}"):
            _gain_loss(df, **kwargs)


class TestPlot:
    """GainLoss.plot."""

    @pytest.mark.parametrize(
        ("group_col", "plot_kwargs", "expected_bar_labels", "expected_y_label"),
        [
            pytest.param(None, {}, [""], FOCUS, id="ungrouped"),
            pytest.param(cols.store_id, {}, ["201", "202"], cols.store_id, id="grouped"),
            pytest.param(
                cols.store_id,
                {"x_col": "new", "stacked": False},
                ["201", "202"],
                cols.store_id,
                id="caller_x_col_and_stacked_ignored",
            ),
        ],
    )
    def test_default_labels(
        self,
        transactions: pd.DataFrame,
        group_col: str | None,
        plot_kwargs: dict[str, str | bool],
        expected_bar_labels: list[str],
        expected_y_label: str,
    ) -> None:
        """One stacked bar per table row, unlabelled ungrouped (the y label names the focus group) or by group_col.

        The x label names the value column.
        """
        ax = _gain_loss(transactions, group_col=group_col).plot(**plot_kwargs)
        # Unstacked, each segment gets its own y offset
        bar_positions = {patch.get_y() for bars in ax.containers for patch in bars.patches}

        assert (
            [t.get_text() for t in ax.get_yticklabels()],
            len(bar_positions),
            ax.get_xlabel(),
            ax.get_ylabel(),
        ) == (expected_bar_labels, len(expected_bar_labels), cols.unit_spend, expected_y_label)

    def test_gains_are_green_and_losses_red(self, transactions: pd.DataFrame) -> None:
        """New, increased and switched-from segments are shades of green; lost, decreased and switched-to of red.

        A caller's color is ignored.
        """
        ax = _gain_loss(transactions).plot(color="black")

        expected = [COLORS[hue][shade] for hue in ("green", "red") for shade in (700, 500, 300)]
        assert [to_hex(bars.patches[0].get_facecolor()) for bars in ax.containers] == expected

    def test_legend_and_default_title_use_group_names(self, transactions: pd.DataFrame) -> None:
        """One legend entry per gain/loss segment, and the default title, interpolate the focus and comparison names."""
        ax = _gain_loss(transactions).plot()

        assert f"Gain Loss from {FOCUS} to {COMPARISON}" in [t.get_text().replace("\n", " ") for t in ax.figure.texts]

        assert [t.get_text() for t in ax.get_legend().get_texts()] == [
            "New",
            f"Increased {FOCUS}",
            f"Switch From {COMPARISON}",
            "Lost",
            f"Decreased {FOCUS}",
            f"Switch To {COMPARISON}",
        ]
