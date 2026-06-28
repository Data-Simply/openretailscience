"""Tests for the GainLoss class in the gain_loss module."""

import datetime

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.options import option_context

# Period totals are sums of non-negative unit spend grouped per customer, so they live in [0, inf).
# Bound the upper end to keep generated examples in a regime where float rounding stays negligible.
spend_total = st.floats(min_value=0.0, max_value=1e6, allow_nan=False, allow_infinity=False)

# Index positions of the three "increase" and three "decrease" buckets in the process_customer_group
# return tuple: (new, lost, increased_focus, decreased_focus, switch_from_comparison, switch_to_comparison).
INCREASE_BUCKET_INDICES = (0, 2, 4)  # new, increased_focus, switch_from_comparison
DECREASE_BUCKET_INDICES = (1, 3, 5)  # lost, decreased_focus, switch_to_comparison


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Automatically close all matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.mark.parametrize(
    ("focus_p1", "comparison_p1", "focus_p2", "comparison_p2", "focus_diff", "comparison_diff", "expected"),
    [
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 100, 0, 100, 0, (100, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (100, 0, 0, 0, -100, 0, (0, -100, 0, 0, 0, 0)),
        # Test when both focus and comparison increase, but focus increase is greater
        (100, 100, 150, 120, 50, 20, (0, 0, 50, 0, 0, 0)),
        # Test when both focus and comparison increase, but comparison increase is greater
        (100, 100, 120, 150, 20, 50, (0, 0, 20, 0, 0, 0)),
        # Test when focus increases and comparison decreases, with focus change greater
        (100, 100, 150, 80, 50, -20, (0, 0, 30, 0, 20, 0)),
        # Test when focus increases and comparison decreases, with comparison change greater
        (100, 100, 110, 50, 10, -50, (0, 0, 0, 0, 10, 0)),
        # Test when both focus and comparison decrease, but focus decrease is greater
        (100, 100, 50, 80, -50, -20, (0, 0, 0, -50, 0, 0)),
        # Test when both focus and comparison decrease, but comparison decrease is greater
        (100, 100, 80, 50, -20, -50, (0, 0, 0, -20, 0, 0)),
        # Test when focus decreases and comparison increases, with focus change greater
        (100, 100, 50, 110, -50, 10, (0, 0, 0, -40, 0, -10)),
        # Test when focus decreases and comparison increases, with comparison change greater
        (100, 100, 80, 150, -20, 50, (0, 0, 0, 0, 0, -20)),
        # Test switch from comparison to focus, with focus change greater
        (100, 100, 180, 50, 80, -50, (0, 0, 30, 0, 50, 0)),
        # Test switch from comparison to focus, with comparison change greater
        (100, 100, 150, 20, 50, -80, (0, 0, 0, 0, 50, 0)),
        # Test switch from focus to comparison, with focus change greater
        (100, 100, 20, 150, -80, 50, (0, 0, 0, -30, 0, -50)),
        # Test switch from focus to comparison, with comparison change greater
        (100, 100, 50, 180, -50, 80, (0, 0, 0, 0, 0, -50)),
        # Test when there's no change in focus or comparison
        (100, 100, 100, 100, 0, 0, (0, 0, 0, 0, 0, 0)),
        # Test when a new customer is added (focus_p1 and comparison_p1 are 0)
        (0, 0, 1, 0, 1, 0, (1, 0, 0, 0, 0, 0)),
        # Test when a customer is lost (focus_p2 and comparison_p2 are 0)
        (1, 0, 0, 0, -1, 0, (0, -1, 0, 0, 0, 0)),
        # Test when both focus and comparison increase
        (0, 0, 1, 1, 1, 1, (1, 0, 0, 0, 0, 0)),
        # Test when focus increases and comparison decreases
        (0, 1, 1, 0, 1, -1, (0, 0, 0, 0, 1, 0)),
        # Test when both focus and comparison decrease
        (1, 1, 0, 0, -1, -1, (0, -1, 0, 0, 0, 0)),
        # Test when focus decreases and comparison increases
        (1, 0, 0, 1, -1, 1, (0, 0, 0, 0, 0, -1)),
        # Test when there's no change in focus or comparison
        (1, 1, 1, 1, 0, 0, (0, 0, 0, 0, 0, 0)),
    ],
)
def test_process_customer_group(
    focus_p1,
    comparison_p1,
    focus_p2,
    comparison_p2,
    focus_diff,
    comparison_diff,
    expected,
):
    """Test the process_customer_group method of the GainLoss class."""
    result = GainLoss.process_customer_group(
        focus_p1=focus_p1,
        comparison_p1=comparison_p1,
        focus_p2=focus_p2,
        comparison_p2=comparison_p2,
        focus_diff=focus_diff,
        comparison_diff=comparison_diff,
    )
    assert result == expected


class TestProcessCustomerGroupProperties:
    """Property-based tests for the gain/loss attribution invariants of process_customer_group.

    process_customer_group splits a single customer's change in focus-group spend into six mutually
    exclusive buckets. The buckets exist purely to *attribute* that change, so two invariants must hold
    for every possible pair of periods, regardless of which branch of the logic runs:

    1. Conservation — the six buckets sum exactly to the focus-group change (focus_p2 - focus_p1). No
       spend is invented or lost in the attribution.
    2. Sign discipline — increase buckets are never negative and decrease buckets are never positive,
       matching the green/red split the plot relies on.
    """

    @given(
        focus_p1=spend_total,
        comparison_p1=spend_total,
        focus_p2=spend_total,
        comparison_p2=spend_total,
    )
    def test_buckets_conserve_focus_change(self, focus_p1, comparison_p1, focus_p2, comparison_p2):
        """The six attribution buckets sum to the total focus-group change for any pair of periods."""
        result = GainLoss.process_customer_group(
            focus_p1=focus_p1,
            comparison_p1=comparison_p1,
            focus_p2=focus_p2,
            comparison_p2=comparison_p2,
            focus_diff=focus_p2 - focus_p1,
            comparison_diff=comparison_p2 - comparison_p1,
        )
        assert sum(result) == pytest.approx(focus_p2 - focus_p1, abs=1e-6)

    @given(
        focus_p1=spend_total,
        comparison_p1=spend_total,
        focus_p2=spend_total,
        comparison_p2=spend_total,
    )
    def test_increase_and_decrease_buckets_keep_their_sign(self, focus_p1, comparison_p1, focus_p2, comparison_p2):
        """Increase buckets are always >= 0 and decrease buckets are always <= 0."""
        result = GainLoss.process_customer_group(
            focus_p1=focus_p1,
            comparison_p1=comparison_p1,
            focus_p2=focus_p2,
            comparison_p2=comparison_p2,
            focus_diff=focus_p2 - focus_p1,
            comparison_diff=comparison_p2 - comparison_p1,
        )
        assert all(result[i] >= 0 for i in INCREASE_BUCKET_INDICES)
        assert all(result[i] <= 0 for i in DECREASE_BUCKET_INDICES)


@pytest.fixture
def sample_df():
    """Sample transaction DataFrame for testing."""
    data = {
        "transaction_id": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        "customer_id": [1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 4],
        "transaction_date": [
            datetime.date(2023, 1, 15),
            datetime.date(2023, 1, 20),
            datetime.date(2023, 2, 5),
            datetime.date(2023, 2, 10),
            datetime.date(2023, 3, 1),
            datetime.date(2023, 3, 15),
            datetime.date(2023, 3, 20),
            datetime.date(2023, 4, 10),
            datetime.date(2023, 4, 25),
            datetime.date(2023, 5, 5),
            datetime.date(2023, 5, 20),
            datetime.date(2023, 6, 10),
        ],
        "unit_spend": [100, 150, 200, 120, 160, 210, 130, 170, 220, 140, 180, 230],
        "brand": [
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
            "Brand A",
            "Brand B",
        ],
    }
    return pd.DataFrame(data)


def test_overlap_in_p1_p2_raises(sample_df):
    """Test that GainLoss raises a ValueError when p1 and p2 indices overlap."""
    p1 = [True] * 6 + [False] * 6
    p2 = [True] + [False] * 11

    with pytest.raises(ValueError, match="p1_index and p2_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=p1,
            p2_index=p2,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[False] * 6 + [True] * 6,
            comparison_group_name="Brand B",
        )


def test_overlap_in_focus_comparison_raises(sample_df):
    """Test that GainLoss raises a ValueError when focus and comparison indices overlap."""
    with pytest.raises(ValueError, match="focus_group_index and comparison_group_index should not overlap"):
        GainLoss(
            df=sample_df,
            p1_index=[True] * 6 + [False] * 6,
            p2_index=[False] * 6 + [True] * 6,
            focus_group_index=[True] * 6 + [False] * 6,
            focus_group_name="Brand A",
            comparison_group_index=[True] * 6 + [False] * 6,
            comparison_group_name="Brand B",
        )


def test_missing_required_column_raises():
    """Test that GainLoss raises a ValueError if required columns are missing in the DataFrame."""
    df = pd.DataFrame(
        {
            "spend": [100, 200],
            "group": ["Brand A", "Brand B"],
        },
    )
    with pytest.raises(ValueError, match="Input data is missing required columns"):
        GainLoss(
            df=df,
            p1_index=[True, False],
            p2_index=[False, True],
            focus_group_index=[True, False],
            focus_group_name="Brand A",
            comparison_group_index=[False, True],
            comparison_group_name="Brand B",
        )


def test_valid_gainloss(sample_df):
    """Test that a GainLoss object is successfully with valid input."""
    gl = GainLoss(
        df=sample_df,
        p1_index=sample_df["transaction_date"] < datetime.date(2023, 5, 1),
        p2_index=sample_df["transaction_date"] >= datetime.date(2023, 5, 1),
        focus_group_index=sample_df["brand"] == "Brand A",
        focus_group_name="Brand A",
        comparison_group_index=sample_df["brand"] == "Brand B",
        comparison_group_name="Brand B",
    )

    assert isinstance(gl, GainLoss)


def test_calc_gains_loss_table_without_group():
    """Test _calc_gains_loss_table with a flat DataFrame (no group index)."""
    df = pd.DataFrame(
        {
            "gain": [10, -5, 20],
            "loss": [5, 2, 1],
        },
    )
    result = GainLoss._calc_gains_loss_table(df)
    expected = pd.DataFrame(
        {
            "gain": [25],
            "loss": [8],
        },
    )
    pd.testing.assert_frame_equal(result.reset_index(drop=True), expected)


def test_calc_gains_loss_table_with_group():
    """Test _calc_gains_loss_table with grouped DataFrame."""
    idx = pd.MultiIndex.from_tuples([("Group A", 1), ("Group A", 2), ("Group B", 1)], names=["group", "id"])
    df = pd.DataFrame(
        {
            "gain": [10, 15, 5],
            "loss": [1, 2, 3],
        },
        index=idx,
    )

    result = GainLoss._calc_gains_loss_table(df, group_col="group")

    expected = pd.DataFrame(
        {
            "gain": [25, 5],
            "loss": [3, 3],
        },
        index=pd.Index(["Group A", "Group B"], name="group"),
    )

    pd.testing.assert_frame_equal(result, expected)


def test_plot_renders_six_gain_loss_segments_with_group_names_in_legend(sample_df):
    """GainLoss.plot renders one labelled legend entry per gain/loss segment, interpolating group names."""
    p1 = sample_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2 = sample_df["transaction_date"] >= datetime.date(2023, 5, 1)

    focus_group_index = sample_df["brand"] == "Brand A"
    comparison_group_index = sample_df["brand"] == "Brand B"

    gl = GainLoss(
        df=sample_df,
        p1_index=p1,
        p2_index=p2,
        focus_group_index=focus_group_index,
        focus_group_name="Brand A",
        comparison_group_index=comparison_group_index,
        comparison_group_name="Brand B",
    )

    ax = gl.plot()
    expected_legend = [
        "New",
        "Increased Brand A",
        "Switch From Brand B",
        "Lost",
        "Decreased Brand A",
        "Switch To Brand B",
    ]
    legend = ax.get_legend()
    assert legend is not None
    assert [t.get_text() for t in legend.get_texts()] == expected_legend


def test_with_custom_column_names(sample_df):
    """Test GainLoss with custom column names."""
    rename_mapping = {
        "customer_id": "cust_identifier",
        "unit_spend": "total_revenue",
    }
    custom_df = sample_df.rename(columns=rename_mapping)
    p1_index = custom_df["transaction_date"] < datetime.date(2023, 5, 1)
    p2_index = custom_df["transaction_date"] >= datetime.date(2023, 5, 1)
    focus_group_index = custom_df["brand"] == "Brand A"
    comparison_group_index = custom_df["brand"] == "Brand B"

    with option_context("column.customer_id", "cust_identifier", "column.unit_spend", "total_revenue"):
        gl = GainLoss(
            df=custom_df,
            p1_index=p1_index,
            p2_index=p2_index,
            focus_group_index=focus_group_index,
            focus_group_name="Brand A",
            comparison_group_index=comparison_group_index,
            comparison_group_name="Brand B",
            value_col="total_revenue",
        )
        gain_loss_df = gl.gain_loss_df
        assert isinstance(gain_loss_df, pd.DataFrame)
        assert not gain_loss_df.empty
