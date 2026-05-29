"""Tests for the GainLoss class in the gain_loss module."""

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.options import ColumnHelper, option_context

cols = ColumnHelper()

P1_DATE = datetime.date(2024, 2, 15)
P2_DATE = datetime.date(2024, 5, 15)
PERIOD_BOUNDARY = datetime.date(2024, 4, 1)
FOCUS_BRAND = "Brand A"
COMPARISON_BRAND = "Brand B"


@pytest.fixture
def transactions_df() -> pd.DataFrame:
    """Retail fixture exercising every gain/loss category, partitioned across two stores.

    Per-customer p1 → p2 (focus, comparison) totals and resulting category:

    - 101 (S1): (  0,   0) → (100,   0) → new = 100
    - 102 (S2): ( 80,   0) → (  0,   0) → lost = -80
    - 103 (S1): (100, 100) → (150, 120) → increased_focus = 50
    - 104 (S2): (100, 100) → ( 60,  80) → decreased_focus = -40
    - 105 (S1): (100, 100) → (150,  20) → switch_from_comparison = 50
    - 106 (S2): (100, 100) → ( 50, 180) → switch_to_comparison = -50
    - 107 (S1): (100, 100) → (150,  80) → increased_focus = 30, switch_from_comparison = 20
    - 108 (S2): (100, 100) → ( 50, 110) → decreased_focus = -40, switch_to_comparison = -10
    """
    f, c = FOCUS_BRAND, COMPARISON_BRAND
    # Each row: (customer_id, transaction_date, unit_spend, brand, store_id)
    rows = [
        # --- p1 ---
        (102, P1_DATE,  80, f, "S2"),
        (103, P1_DATE, 100, f, "S1"), (103, P1_DATE, 100, c, "S1"),
        (104, P1_DATE, 100, f, "S2"), (104, P1_DATE, 100, c, "S2"),
        (105, P1_DATE, 100, f, "S1"), (105, P1_DATE, 100, c, "S1"),
        (106, P1_DATE, 100, f, "S2"), (106, P1_DATE, 100, c, "S2"),
        (107, P1_DATE, 100, f, "S1"), (107, P1_DATE, 100, c, "S1"),
        (108, P1_DATE, 100, f, "S2"), (108, P1_DATE, 100, c, "S2"),
        # --- p2 ---
        (101, P2_DATE, 100, f, "S1"),
        (103, P2_DATE, 150, f, "S1"), (103, P2_DATE, 120, c, "S1"),
        (104, P2_DATE,  60, f, "S2"), (104, P2_DATE,  80, c, "S2"),
        (105, P2_DATE, 150, f, "S1"), (105, P2_DATE,  20, c, "S1"),
        (106, P2_DATE,  50, f, "S2"), (106, P2_DATE, 180, c, "S2"),
        (107, P2_DATE, 150, f, "S1"), (107, P2_DATE,  80, c, "S1"),
        (108, P2_DATE,  50, f, "S2"), (108, P2_DATE, 110, c, "S2"),
    ]  # fmt: skip
    return pd.DataFrame(
        rows,
        columns=[cols.customer_id, cols.transaction_date, cols.unit_spend, "brand", cols.store_id],
    )


@pytest.fixture
def p1_index(transactions_df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting transactions in the first analysis period."""
    return transactions_df[cols.transaction_date] < PERIOD_BOUNDARY


@pytest.fixture
def p2_index(transactions_df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting transactions in the second analysis period."""
    return transactions_df[cols.transaction_date] >= PERIOD_BOUNDARY


@pytest.fixture
def focus_group_index(transactions_df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting transactions belonging to the focus brand."""
    return transactions_df["brand"] == FOCUS_BRAND


@pytest.fixture
def comparison_group_index(transactions_df: pd.DataFrame) -> pd.Series:
    """Boolean mask selecting transactions belonging to the comparison brand."""
    return transactions_df["brand"] == COMPARISON_BRAND


@pytest.fixture
def gl(
    transactions_df: pd.DataFrame,
    p1_index: pd.Series,
    p2_index: pd.Series,
    focus_group_index: pd.Series,
    comparison_group_index: pd.Series,
) -> GainLoss:
    """Default GainLoss built from transactions_df (no group_col, default agg_func)."""
    return GainLoss(
        df=transactions_df,
        p1_index=p1_index,
        p2_index=p2_index,
        focus_group_index=focus_group_index,
        focus_group_name=FOCUS_BRAND,
        comparison_group_index=comparison_group_index,
        comparison_group_name=COMPARISON_BRAND,
    )


@pytest.fixture
def expected_customer_level_gain_loss() -> pd.DataFrame:
    """Expected per-customer gain/loss table for transactions_df (no group_col)."""
    return pd.DataFrame(
        {
            "focus_p1": [0, 80, 100, 100, 100, 100, 100, 100],
            "comparison_p1": [0, 0, 100, 100, 100, 100, 100, 100],
            "total_p1": [0, 80, 200, 200, 200, 200, 200, 200],
            "focus_p2": [100, 0, 150, 60, 150, 50, 150, 50],
            "comparison_p2": [0, 0, 120, 80, 20, 180, 80, 110],
            "total_p2": [100, 0, 270, 140, 170, 230, 230, 160],
            "focus_diff": [100, -80, 50, -40, 50, -50, 50, -50],
            "comparison_diff": [0, 0, 20, -20, -80, 80, -20, 10],
            "total_diff": [100, -80, 70, -60, -30, 30, 30, -40],
            "new": [100, 0, 0, 0, 0, 0, 0, 0],
            "lost": [0, -80, 0, 0, 0, 0, 0, 0],
            "increased_focus": [0, 0, 50, 0, 0, 0, 30, 0],
            "decreased_focus": [0, 0, 0, -40, 0, 0, 0, -40],
            "switch_from_comparison": [0, 0, 0, 0, 50, 0, 20, 0],
            "switch_to_comparison": [0, 0, 0, 0, 0, -50, 0, -10],
        },
        index=pd.CategoricalIndex(
            [101, 102, 103, 104, 105, 106, 107, 108],
            categories=[101, 102, 103, 104, 105, 106, 107, 108],
            name=cols.customer_id,
        ),
    )


@pytest.fixture
def expected_gain_loss_table_no_group() -> pd.DataFrame:
    """Expected aggregated gain/loss table for transactions_df (no group_col)."""
    return pd.DataFrame(
        {
            "focus_p1": [680],
            "comparison_p1": [600],
            "total_p1": [1280],
            "focus_p2": [710],
            "comparison_p2": [590],
            "total_p2": [1300],
            "focus_diff": [30],
            "comparison_diff": [-10],
            "total_diff": [20],
            "new": [100],
            "lost": [-80],
            "increased_focus": [80],
            "decreased_focus": [-80],
            "switch_from_comparison": [70],
            "switch_to_comparison": [-60],
        },
        index=[""],
    )


@pytest.fixture
def expected_gain_loss_table_by_store() -> pd.DataFrame:
    """Expected aggregated gain/loss table for transactions_df grouped by store_id."""
    return pd.DataFrame(
        {
            "focus_p1": [300, 380],
            "comparison_p1": [300, 300],
            "total_p1": [600, 680],
            "focus_p2": [550, 160],
            "comparison_p2": [220, 370],
            "total_p2": [770, 530],
            "focus_diff": [250, -220],
            "comparison_diff": [-80, 70],
            "total_diff": [170, -150],
            "new": [100, 0],
            "lost": [0, -80],
            "increased_focus": [80, 0],
            "decreased_focus": [0, -80],
            "switch_from_comparison": [70, 0],
            "switch_to_comparison": [0, -60],
        },
        index=pd.Index(["S1", "S2"], name=cols.store_id),
    )


@pytest.fixture
def expected_customer_level_gain_loss_by_store() -> pd.DataFrame:
    """Expected per-customer gain/loss table for transactions_df grouped by store_id.

    Same values as expected_customer_level_gain_loss but with (store_id, customer_id) MultiIndex,
    sorted by store then customer (S1: 101, 103, 105, 107; S2: 102, 104, 106, 108).
    """
    return pd.DataFrame(
        {
            "focus_p1": [0, 100, 100, 100, 80, 100, 100, 100],
            "comparison_p1": [0, 100, 100, 100, 0, 100, 100, 100],
            "total_p1": [0, 200, 200, 200, 80, 200, 200, 200],
            "focus_p2": [100, 150, 150, 150, 0, 60, 50, 50],
            "comparison_p2": [0, 120, 20, 80, 0, 80, 180, 110],
            "total_p2": [100, 270, 170, 230, 0, 140, 230, 160],
            "focus_diff": [100, 50, 50, 50, -80, -40, -50, -50],
            "comparison_diff": [0, 20, -80, -20, 0, -20, 80, 10],
            "total_diff": [100, 70, -30, 30, -80, -60, 30, -40],
            "new": [100, 0, 0, 0, 0, 0, 0, 0],
            "lost": [0, 0, 0, 0, -80, 0, 0, 0],
            "increased_focus": [0, 50, 0, 30, 0, 0, 0, 0],
            "decreased_focus": [0, 0, 0, 0, 0, -40, 0, -40],
            "switch_from_comparison": [0, 0, 50, 20, 0, 0, 0, 0],
            "switch_to_comparison": [0, 0, 0, 0, 0, 0, -50, -10],
        },
        index=pd.MultiIndex.from_arrays(
            [
                ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"],
                pd.Categorical(
                    [101, 103, 105, 107, 102, 104, 106, 108],
                    categories=[101, 102, 103, 104, 105, 106, 107, 108],
                ),
            ],
            names=[cols.store_id, cols.customer_id],
        ),
    )


@pytest.fixture
def tiny_df() -> pd.DataFrame:
    """Three-row fixture for input-validation tests (three rows lets us craft partial-overlap masks)."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3],
            cols.transaction_date: [P1_DATE, P2_DATE, P2_DATE],
            cols.unit_spend: [100, 200, 300],
            "brand": [FOCUS_BRAND, COMPARISON_BRAND, FOCUS_BRAND],
        },
    )


class TestProcessCustomerGroup:
    """Per-customer categorisation logic — verifies every branch of process_customer_group."""

    @pytest.mark.parametrize(
        ("focus_p1", "comparison_p1", "focus_p2", "comparison_p2", "expected"),
        [
            # NEW branch — focus_p1=0 AND comparison_p1=0 → returns (focus_p2, 0, 0, 0, 0, 0)
            pytest.param(0, 0, 100, 0, (100, 0, 0, 0, 0, 0), id="new__focus_only"),
            pytest.param(0, 0, 100, 80, (100, 0, 0, 0, 0, 0), id="new__comparison_p2_does_not_count"),
            pytest.param(0, 0, 0, 100, (0, 0, 0, 0, 0, 0), id="new__comparison_only_invisible_to_focus"),
            # LOST branch — focus_p2=0 AND comparison_p2=0 → returns (0, -focus_p1, 0, 0, 0, 0)
            pytest.param(100, 0, 0, 0, (0, -100, 0, 0, 0, 0), id="lost__focus_only"),
            pytest.param(100, 80, 0, 0, (0, -100, 0, 0, 0, 0), id="lost__comparison_p1_does_not_count"),
            pytest.param(0, 100, 0, 0, (0, 0, 0, 0, 0, 0), id="lost__comparison_only_invisible_to_focus"),
            # Branch C: focus_diff>0, comparison_diff>0 — both grew, no switching, pure increase
            pytest.param(100, 100, 150, 120, (0, 0, 50, 0, 0, 0), id="branch_C__focus_up_more"),
            pytest.param(100, 100, 120, 150, (0, 0, 20, 0, 0, 0), id="branch_C__focus_up_less"),
            # Branch D: focus_diff>0, comparison_diff<0, FD+CD>0 — split: partial increase + switch_from
            pytest.param(100, 100, 150, 80, (0, 0, 30, 0, 20, 0), id="branch_D__partial_switch_from_small"),
            pytest.param(100, 100, 180, 50, (0, 0, 30, 0, 50, 0), id="branch_D__partial_switch_from_large"),
            # Branch E: focus_diff>0, comparison_diff<0, FD+CD<0 — entire focus gain is switch_from
            pytest.param(100, 100, 110, 50, (0, 0, 0, 0, 10, 0), id="branch_E__all_switch_from_small"),
            pytest.param(100, 100, 150, 20, (0, 0, 0, 0, 50, 0), id="branch_E__all_switch_from_large"),
            # Branch F: focus_diff<0, comparison_diff<0 — both shrank, no switching, pure decrease
            pytest.param(100, 100, 50, 80, (0, 0, 0, -50, 0, 0), id="branch_F__focus_down_more"),
            pytest.param(100, 100, 80, 50, (0, 0, 0, -20, 0, 0), id="branch_F__focus_down_less"),
            # Branch G: focus_diff<0, comparison_diff>0, FD+CD<0 — split: partial decrease + switch_to
            pytest.param(100, 100, 50, 110, (0, 0, 0, -40, 0, -10), id="branch_G__partial_switch_to_small"),
            pytest.param(100, 100, 20, 150, (0, 0, 0, -30, 0, -50), id="branch_G__partial_switch_to_large"),
            # Branch H: focus_diff<0, comparison_diff>0, FD+CD>0 — entire focus loss is switch_to
            pytest.param(100, 100, 80, 150, (0, 0, 0, 0, 0, -20), id="branch_H__all_switch_to_small"),
            pytest.param(100, 100, 50, 180, (0, 0, 0, 0, 0, -50), id="branch_H__all_switch_to_large"),
            # Zero-diff boundaries — comparison_diff or focus_diff exactly zero
            pytest.param(100, 100, 100, 100, (0, 0, 0, 0, 0, 0), id="boundary__both_diffs_zero"),
            pytest.param(100, 100, 100, 150, (0, 0, 0, 0, 0, 0), id="boundary__focus_stable_comparison_up"),
            pytest.param(100, 100, 100, 50, (0, 0, 0, 0, 0, 0), id="boundary__focus_stable_comparison_down"),
            pytest.param(100, 100, 150, 100, (0, 0, 50, 0, 0, 0), id="boundary__focus_up_comparison_stable"),
            pytest.param(100, 100, 50, 100, (0, 0, 0, -50, 0, 0), id="boundary__focus_down_comparison_stable"),
            # Exact cancellation — focus_diff + comparison_diff = 0, all of focus_diff transfers
            pytest.param(100, 100, 150, 50, (0, 0, 0, 0, 50, 0), id="exact_cancellation__focus_gains_all"),
            pytest.param(100, 100, 50, 150, (0, 0, 0, 0, 0, -50), id="exact_cancellation__focus_loses_all"),
            # One-sided p1 customers — exist only in one brand in p1, fall through to categorised branch
            pytest.param(0, 100, 100, 0, (0, 0, 0, 0, 100, 0), id="one_sided__full_switch_comparison_to_focus"),
            pytest.param(100, 0, 0, 100, (0, 0, 0, 0, 0, -100), id="one_sided__full_switch_focus_to_comparison"),
        ],
    )
    def test_returns_expected_category_tuple(
        self,
        focus_p1: float,
        comparison_p1: float,
        focus_p2: float,
        comparison_p2: float,
        expected: tuple[float, float, float, float, float, float],
    ) -> None:
        """Each (p1, p2) combination categorises into the expected (new, lost, inc, dec, sf, st) tuple."""
        result = GainLoss.process_customer_group(
            focus_p1=focus_p1,
            comparison_p1=comparison_p1,
            focus_p2=focus_p2,
            comparison_p2=comparison_p2,
            focus_diff=focus_p2 - focus_p1,
            comparison_diff=comparison_p2 - comparison_p1,
        )
        assert result == expected


class TestGainLossConstruction:
    """Input validation in GainLoss.__init__."""

    @pytest.mark.parametrize(
        ("p1_mask", "p2_mask"),
        [
            # Partial overlap at row 1: p1={0,1}, p2={1,2}, intersection={1}
            pytest.param([True, True, False], [False, True, True], id="partial_overlap_middle"),
            # Identical masks (full equality is the strictest case of overlap)
            pytest.param([True, True, False], [True, True, False], id="identical_masks"),
        ],
    )
    def test_overlapping_p1_p2_indices_raises(
        self,
        tiny_df: pd.DataFrame,
        p1_mask: list[bool],
        p2_mask: list[bool],
    ) -> None:
        """Any shared row between p1 and p2 makes period assignment ambiguous and must raise."""
        with pytest.raises(ValueError, match="p1_index and p2_index should not overlap"):
            GainLoss(
                df=tiny_df,
                p1_index=pd.Series(p1_mask),
                p2_index=pd.Series(p2_mask),
                focus_group_index=pd.Series([True, False, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([False, True, False]),
                comparison_group_name=COMPARISON_BRAND,
            )

    @pytest.mark.parametrize(
        ("focus_mask", "comparison_mask"),
        [
            # Partial overlap at row 1: focus={0,1}, comparison={1,2}, intersection={1}
            pytest.param([True, True, False], [False, True, True], id="partial_overlap_middle"),
            # Identical masks (full equality)
            pytest.param([True, False, False], [True, False, False], id="identical_masks"),
        ],
    )
    def test_overlapping_focus_comparison_indices_raises(
        self,
        tiny_df: pd.DataFrame,
        focus_mask: list[bool],
        comparison_mask: list[bool],
    ) -> None:
        """Any shared row between focus and comparison groups must raise."""
        with pytest.raises(ValueError, match="focus_group_index and comparison_group_index should not overlap"):
            GainLoss(
                df=tiny_df,
                p1_index=pd.Series([True, False, False]),
                p2_index=pd.Series([False, True, True]),
                focus_group_index=pd.Series(focus_mask),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series(comparison_mask),
                comparison_group_name=COMPARISON_BRAND,
            )

    @pytest.mark.parametrize(
        ("df_columns", "group_col", "missing_substring"),
        [
            # customer_id absent → flagged
            pytest.param(["spend", "brand"], None, "customer_id", id="customer_id_missing"),
            # value_col 'unit_spend' absent (df has 'spend' instead) → flagged
            pytest.param(["customer_id", "brand"], None, "unit_spend", id="value_col_missing"),
            # group_col present in __init__ but absent from df → flagged by name
            pytest.param(
                ["customer_id", "unit_spend", "brand"], "not_a_real_column", "not_a_real_column", id="group_col_missing"
            ),
        ],
    )
    def test_missing_required_column_raises(
        self, df_columns: list[str], group_col: str | None, missing_substring: str
    ) -> None:
        """validate_columns surfaces every missing required column by name (customer_id, value_col, or group_col)."""
        df = pd.DataFrame({col: [100, 200, 300] for col in df_columns})
        with pytest.raises(ValueError, match=f"columns are required but missing.*{missing_substring}"):
            GainLoss(
                df=df,
                p1_index=pd.Series([True, False, False]),
                p2_index=pd.Series([False, True, True]),
                focus_group_index=pd.Series([True, False, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([False, True, False]),
                comparison_group_name=COMPARISON_BRAND,
                group_col=group_col,
            )


class TestGainLossPipeline:
    """End-to-end correctness of customer-level and aggregated gain/loss tables."""

    def test_customer_level_table_matches_expected(
        self,
        gl: GainLoss,
        expected_customer_level_gain_loss: pd.DataFrame,
    ) -> None:
        """gain_loss_df contains the expected p1/p2/diff totals and category buckets per customer."""
        assert_frame_equal(gl.gain_loss_df, expected_customer_level_gain_loss)

    def test_aggregated_table_matches_expected(
        self,
        gl: GainLoss,
        expected_gain_loss_table_no_group: pd.DataFrame,
    ) -> None:
        """gain_loss_table_df sums the customer-level table to a single row when no group is given."""
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_no_group)

    def test_grouped_pipeline_aggregates_by_store(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_customer_level_gain_loss_by_store: pd.DataFrame,
        expected_gain_loss_table_by_store: pd.DataFrame,
    ) -> None:
        """With group_col=store_id, both the per-customer MultiIndex df and the aggregated table are correct."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
            group_col=cols.store_id,
        )
        assert_frame_equal(gl.gain_loss_df, expected_customer_level_gain_loss_by_store)
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_by_store)

    @pytest.mark.parametrize(
        ("agg_func", "expected_row"),
        [
            (
                "sum",
                {
                    "focus_p1": 300,
                    "comparison_p1": 200,
                    "total_p1": 500,
                    "focus_p2": 200,
                    "comparison_p2": 100,
                    "total_p2": 300,
                    "focus_diff": -100,
                    "comparison_diff": -100,
                    "total_diff": -200,
                    "new": 0,
                    "lost": 0,
                    "increased_focus": 0,
                    "decreased_focus": -100,
                    "switch_from_comparison": 0,
                    "switch_to_comparison": 0,
                },
            ),
            (
                "max",
                {
                    "focus_p1": 200,
                    "comparison_p1": 150,
                    "total_p1": 200,
                    "focus_p2": 120,
                    "comparison_p2": 60,
                    "total_p2": 120,
                    "focus_diff": -80,
                    "comparison_diff": -90,
                    "total_diff": -80,
                    "new": 0,
                    "lost": 0,
                    "increased_focus": 0,
                    "decreased_focus": -80,
                    "switch_from_comparison": 0,
                    "switch_to_comparison": 0,
                },
            ),
            (
                "mean",
                {
                    "focus_p1": 150,
                    "comparison_p1": 100,
                    "total_p1": 125,
                    "focus_p2": 100,
                    "comparison_p2": 50,
                    "total_p2": 75,
                    "focus_diff": -50,
                    "comparison_diff": -50,
                    "total_diff": -50,
                    "new": 0,
                    "lost": 0,
                    "increased_focus": 0,
                    "decreased_focus": -50,
                    "switch_from_comparison": 0,
                    "switch_to_comparison": 0,
                },
            ),
            (
                "min",
                {
                    "focus_p1": 100,
                    "comparison_p1": 50,
                    "total_p1": 50,
                    "focus_p2": 80,
                    "comparison_p2": 40,
                    "total_p2": 40,
                    "focus_diff": -20,
                    "comparison_diff": -10,
                    "total_diff": -10,
                    "new": 0,
                    "lost": 0,
                    "increased_focus": 0,
                    "decreased_focus": -20,
                    "switch_from_comparison": 0,
                    "switch_to_comparison": 0,
                },
            ),
        ],
    )
    def test_agg_func_applies_to_focus_comparison_and_total_aggregates(
        self,
        agg_func: str,
        expected_row: dict[str, float],
    ) -> None:
        """agg_func is applied to focus, comparison, AND total aggregates, and the categorisation reflects the result.

        Customer 1 has four focus and four comparison transactions split across p1 and p2 chosen so each agg_func
        yields a distinct (focus, comparison, total) triple per period. Without exercising every column, a regression
        that hardcoded 'sum' on any one of the three concat branches would slip through.
        """
        df = pd.DataFrame(
            {
                cols.customer_id: [1] * 8,
                cols.transaction_date: [P1_DATE] * 4 + [P2_DATE] * 4,
                cols.unit_spend: [100, 200, 50, 150, 80, 120, 60, 40],
                "brand": [FOCUS_BRAND, FOCUS_BRAND, COMPARISON_BRAND, COMPARISON_BRAND] * 2,
            },
        )
        gl = GainLoss(
            df=df,
            p1_index=df[cols.transaction_date] < PERIOD_BOUNDARY,
            p2_index=df[cols.transaction_date] >= PERIOD_BOUNDARY,
            focus_group_index=df["brand"] == FOCUS_BRAND,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=df["brand"] == COMPARISON_BRAND,
            comparison_group_name=COMPARISON_BRAND,
            agg_func=agg_func,
        )
        actual = gl.gain_loss_df.loc[1]
        for column, expected in expected_row.items():
            assert actual[column] == expected, (
                f"agg_func={agg_func!r} column={column!r}: got {actual[column]}, expected {expected}"
            )

    def test_uses_configured_column_names(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_gain_loss_table_no_group: pd.DataFrame,
    ) -> None:
        """Configured customer_id and value_col names produce the same numeric output as the defaults."""
        rename_map = {cols.customer_id: "cust_identifier", cols.unit_spend: "total_revenue"}
        renamed_df = transactions_df.rename(columns=rename_map)
        with option_context("column.customer_id", "cust_identifier", "column.unit_spend", "total_revenue"):
            gl = GainLoss(
                df=renamed_df,
                p1_index=p1_index,
                p2_index=p2_index,
                focus_group_index=focus_group_index,
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=comparison_group_index,
                comparison_group_name=COMPARISON_BRAND,
                value_col="total_revenue",
            )
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_no_group)

    def test_value_col_default_is_resolved_at_call_time(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_gain_loss_table_no_group: pd.DataFrame,
    ) -> None:
        """Omitting value_col under option_context picks up the configured column name, not an import-time default."""
        renamed_df = transactions_df.rename(columns={cols.unit_spend: "total_revenue"})
        with option_context("column.unit_spend", "total_revenue"):
            gl = GainLoss(
                df=renamed_df,
                p1_index=p1_index,
                p2_index=p2_index,
                focus_group_index=focus_group_index,
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=comparison_group_index,
                comparison_group_name=COMPARISON_BRAND,
            )
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_no_group)


class TestGainLossPlot:
    """Plot rendering on top of the aggregated gain/loss table."""

    @pytest.fixture(autouse=True)
    def _cleanup_figures(self):
        """Close all matplotlib figures after each plot test."""
        yield
        plt.close("all")

    def test_legend_labels_interpolate_group_names(self, gl: GainLoss) -> None:
        """The legend has one entry per gain/loss segment, with focus/comparison names interpolated."""
        ax = gl.plot()
        legend = ax.get_legend()
        assert legend is not None
        assert [t.get_text() for t in legend.get_texts()] == [
            "New",
            f"Increased {FOCUS_BRAND}",
            f"Switch From {COMPARISON_BRAND}",
            "Lost",
            f"Decreased {FOCUS_BRAND}",
            f"Switch To {COMPARISON_BRAND}",
        ]

    @pytest.mark.parametrize(
        ("title_arg", "expected_title", "absent_title"),
        [
            (None, f"Gain Loss from {FOCUS_BRAND} to {COMPARISON_BRAND}", None),
            ("Q2 brand migration", "Q2 brand migration", f"Gain Loss from {FOCUS_BRAND} to {COMPARISON_BRAND}"),
        ],
    )
    def test_title_is_rendered_in_figure_chrome(
        self,
        gl: GainLoss,
        title_arg: str | None,
        expected_title: str,
        absent_title: str | None,
    ) -> None:
        """Title is rendered as figure-level chrome (not a tick/axis label); custom title overrides default."""
        ax = gl.plot() if title_arg is None else gl.plot(title=title_arg)
        chrome_texts = [t.get_text() for t in ax.figure.texts]
        assert expected_title in chrome_texts
        if absent_title is not None:
            assert absent_title not in chrome_texts
