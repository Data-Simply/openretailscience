"""Tests for the GainLoss class in the gain_loss module."""

import datetime

import matplotlib.pyplot as plt
import matplotlib.text
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


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


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
def tiny_df() -> pd.DataFrame:
    """Minimal two-row fixture for input-validation tests."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2],
            cols.transaction_date: [P1_DATE, P2_DATE],
            cols.unit_spend: [100, 200],
            "brand": [FOCUS_BRAND, COMPARISON_BRAND],
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

    def test_overlapping_p1_p2_indices_raises(self, tiny_df: pd.DataFrame) -> None:
        """A row flagged in both p1 and p2 makes period assignment ambiguous and must raise."""
        with pytest.raises(ValueError, match="p1_index and p2_index should not overlap"):
            GainLoss(
                df=tiny_df,
                p1_index=pd.Series([True, True]),
                p2_index=pd.Series([True, False]),
                focus_group_index=pd.Series([True, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([False, True]),
                comparison_group_name=COMPARISON_BRAND,
            )

    def test_overlapping_focus_comparison_indices_raises(self, tiny_df: pd.DataFrame) -> None:
        """A row flagged in both focus and comparison groups must raise."""
        with pytest.raises(ValueError, match="focus_group_index and comparison_group_index should not overlap"):
            GainLoss(
                df=tiny_df,
                p1_index=pd.Series([True, False]),
                p2_index=pd.Series([False, True]),
                focus_group_index=pd.Series([True, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([True, False]),
                comparison_group_name=COMPARISON_BRAND,
            )

    def test_missing_required_columns_raises(self) -> None:
        """A DataFrame missing customer_id or value_col must raise with a clear error."""
        df = pd.DataFrame({"spend": [100, 200], "brand": [FOCUS_BRAND, COMPARISON_BRAND]})
        with pytest.raises(ValueError, match="columns are required but missing"):
            GainLoss(
                df=df,
                p1_index=pd.Series([True, False]),
                p2_index=pd.Series([False, True]),
                focus_group_index=pd.Series([True, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([False, True]),
                comparison_group_name=COMPARISON_BRAND,
            )

    def test_missing_group_col_raises(self, tiny_df: pd.DataFrame) -> None:
        """If group_col is given but absent from df, the missing-column validation must fire."""
        with pytest.raises(ValueError, match="columns are required but missing"):
            GainLoss(
                df=tiny_df,
                p1_index=pd.Series([True, False]),
                p2_index=pd.Series([False, True]),
                focus_group_index=pd.Series([True, False]),
                focus_group_name=FOCUS_BRAND,
                comparison_group_index=pd.Series([False, True]),
                comparison_group_name=COMPARISON_BRAND,
                group_col="not_a_real_column",
            )


class TestGainLossPipeline:
    """End-to-end correctness of customer-level and aggregated gain/loss tables."""

    def test_customer_level_table_matches_expected(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_customer_level_gain_loss: pd.DataFrame,
    ) -> None:
        """gain_loss_df contains the expected p1/p2/diff totals and category buckets per customer."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
        )
        assert_frame_equal(gl.gain_loss_df, expected_customer_level_gain_loss)

    def test_aggregated_table_matches_expected(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_gain_loss_table_no_group: pd.DataFrame,
    ) -> None:
        """gain_loss_table_df sums the customer-level table to a single row when no group is given."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
        )
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_no_group)

    def test_grouped_pipeline_aggregates_by_store(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
        expected_gain_loss_table_by_store: pd.DataFrame,
    ) -> None:
        """With group_col=store_id, the aggregated table has one row per store with correct sums."""
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
        assert_frame_equal(gl.gain_loss_table_df, expected_gain_loss_table_by_store)

    @pytest.mark.parametrize(
        ("agg_func", "expected_p1_cust1", "expected_p2_cust2"),
        [
            ("sum", 300, 200),
            ("max", 200, 120),
            ("mean", 150, 100),
            ("min", 100, 80),
        ],
    )
    def test_agg_func_changes_per_customer_aggregation(
        self,
        agg_func: str,
        expected_p1_cust1: float,
        expected_p2_cust2: float,
    ) -> None:
        """The agg_func parameter governs how multi-transaction customer totals are aggregated."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 1, 2, 2],
                cols.transaction_date: [P1_DATE, P1_DATE, P2_DATE, P2_DATE],
                cols.unit_spend: [100, 200, 80, 120],
                "brand": [FOCUS_BRAND] * 4,
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
        # Customer 1 only has p1 activity (lost); customer 2 only has p2 activity (new).
        assert gl.gain_loss_df.loc[1, "focus_p1"] == expected_p1_cust1
        assert gl.gain_loss_df.loc[1, "lost"] == -expected_p1_cust1
        assert gl.gain_loss_df.loc[2, "focus_p2"] == expected_p2_cust2
        assert gl.gain_loss_df.loc[2, "new"] == expected_p2_cust2

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


class TestGainLossPlot:
    """Plot rendering on top of the aggregated gain/loss table."""

    def test_legend_labels_interpolate_group_names(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
    ) -> None:
        """The legend has one entry per gain/loss segment, with focus/comparison names interpolated."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
        )
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

    def test_default_title_interpolates_focus_and_comparison_names(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
    ) -> None:
        """With no explicit title, the figure renders 'Gain Loss from {focus} to {comparison}'."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
        )
        ax = gl.plot()
        figure_texts = {
            artist.get_text() for artist in ax.figure.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text))
        }
        assert f"Gain Loss from {FOCUS_BRAND} to {COMPARISON_BRAND}" in figure_texts

    def test_custom_title_overrides_default(
        self,
        transactions_df: pd.DataFrame,
        p1_index: pd.Series,
        p2_index: pd.Series,
        focus_group_index: pd.Series,
        comparison_group_index: pd.Series,
    ) -> None:
        """An explicit title argument replaces the default interpolated title."""
        gl = GainLoss(
            df=transactions_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name=FOCUS_BRAND,
            comparison_group_index=comparison_group_index,
            comparison_group_name=COMPARISON_BRAND,
        )
        ax = gl.plot(title="Q2 brand migration")
        figure_texts = {
            artist.get_text() for artist in ax.figure.findobj(match=lambda obj: isinstance(obj, matplotlib.text.Text))
        }
        assert "Q2 brand migration" in figure_texts
        assert f"Gain Loss from {FOCUS_BRAND} to {COMPARISON_BRAND}" not in figure_texts
