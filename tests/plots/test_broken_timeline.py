"""Tests for the broken timeline plot module."""

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from openretailscience.options import get_option
from openretailscience.plots import broken_timeline


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_dataframe():
    """A sample dataframe for testing with intentional gaps."""
    date_col = get_option("column.transaction_date")

    # Create data with gaps for different stores
    data = {
        date_col: pd.to_datetime(
            [
                "2025-04-01",
                "2025-04-02",
                "2025-04-03",  # Store_North: continuous
                "2025-04-06",
                "2025-04-07",  # Store_North: gap then continues
                "2025-04-01",
                "2025-04-02",  # Store_South: starts same time
                "2025-04-05",
                "2025-04-06",  # Store_South: gap then continues
                "2025-04-03",
                "2025-04-04",
                "2025-04-05",  # Store_East: different pattern
            ],
        ),
        "category": [
            "Store_North",
            "Store_North",
            "Store_North",
            "Store_North",
            "Store_North",
            "Store_South",
            "Store_South",
            "Store_South",
            "Store_South",
            "Store_East",
            "Store_East",
            "Store_East",
        ],
        "value": [100, 150, 200, 120, 180, 300, 250, 400, 350, 80, 90, 110],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_dataframe_single_category():
    """A sample dataframe with a single category for testing."""
    date_col = get_option("column.transaction_date")

    data = {
        date_col: pd.to_datetime(["2025-04-01", "2025-04-02", "2025-04-05", "2025-04-06"]),
        "category": ["Store1", "Store1", "Store1", "Store1"],
        "value": [100, 150, 200, 120],
    }
    return pd.DataFrame(data)


@pytest.fixture
def empty_dataframe():
    """An empty dataframe for testing."""
    date_col = get_option("column.transaction_date")
    return pd.DataFrame(columns=[date_col, "category", "value"])


class TestBrokenTimelinePlot:
    """Test cases for the broken timeline plot function."""

    def test_basic_functionality_and_labels(self, sample_dataframe):
        """Test basic plot creation and custom labels."""
        title = "Data Availability Timeline"
        x_label = "Custom Date Label"
        y_label = "Custom Category Label"

        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            title=title,
            x_label=x_label,
            y_label=y_label,
        )

        assert isinstance(ax, Axes)
        title_texts = [t for t in ax.figure.texts if t.get_text() == title]
        assert len(title_texts) == 1
        assert ax.get_xlabel() == x_label
        assert ax.get_ylabel() == y_label

    def test_single_category(self, sample_dataframe_single_category):
        """Test with a single category."""
        ax = broken_timeline.plot(
            df=sample_dataframe_single_category,
            category_col="category",
            value_col="value",
        )

        assert isinstance(ax, Axes)
        assert len(ax.get_yticklabels()) == 1

    def test_threshold_filtering(self, sample_dataframe):
        """Test threshold value filtering removes low values."""
        threshold = 150
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            threshold_value=threshold,
        )
        ax_no_threshold = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        surviving_categories = sorted(
            sample_dataframe.loc[sample_dataframe["value"] >= threshold, "category"].unique(),
        )
        all_categories = sorted(sample_dataframe["category"].unique())

        assert [t.get_text() for t in ax.get_yticklabels()] == surviving_categories
        assert [t.get_text() for t in ax_no_threshold.get_yticklabels()] == all_categories
        assert len(surviving_categories) < len(all_categories)

    def test_different_periods(self):
        """Test period aggregation works correctly for valid periods."""
        date_col = get_option("column.transaction_date")

        # Create data spanning multiple weeks for comprehensive testing
        data = {
            date_col: pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-08",
                    "2025-01-15",
                    "2025-01-22",
                    "2025-01-29",
                    "2025-02-05",
                    "2025-02-12",
                ],
            ),
            "category": ["Store_North"] * 8,
            "value": [100] * 8,
        }
        df = pd.DataFrame(data)

        segments = {}
        for period in ["D", "W"]:
            ax = broken_timeline.plot(df, "category", "value", period=period)
            segments[period] = sum(len(c.get_paths()) for c in ax.collections)

        # 01-01 and 01-02 form one daily segment; the other six dates are isolated.
        expected_daily_segments = 7
        # All 8 dates fall in 7 contiguous weeks, collapsing to a single weekly segment.
        expected_weekly_segments = 1
        assert segments["D"] == expected_daily_segments
        assert segments["W"] == expected_weekly_segments

    @pytest.mark.parametrize(
        ("canonical", "alias"),
        [
            ("D", "d"),
            ("D", "day"),
            ("D", "DAY"),
            ("W", "w"),
            ("W", "week"),
            ("W", "WEEK"),
        ],
    )
    def test_period_aliases_produce_same_output_as_canonical(self, sample_dataframe, canonical, alias):
        """Test that period aliases (case-insensitive short and long forms) match the canonical short form."""
        ax_canonical = broken_timeline.plot(sample_dataframe, "category", "value", period=canonical)
        ax_alias = broken_timeline.plot(sample_dataframe, "category", "value", period=alias)
        canonical_segments = sum(len(c.get_paths()) for c in ax_canonical.collections)
        alias_segments = sum(len(c.get_paths()) for c in ax_alias.collections)
        assert canonical_segments == alias_segments

    def test_period_aliases_and_gap_days_stay_in_lockstep(self):
        """Every alias must resolve to a key present in PERIOD_GAP_DAYS, and vice versa.

        Guards against adding a new period to one dict and forgetting the other; without
        this, ensure_value_choice succeeds and the next ``PERIOD_GAP_DAYS[period]`` lookup
        raises KeyError at runtime.
        """
        alias_targets = set(broken_timeline.PERIOD_ALIASES.values())
        gap_keys = set(broken_timeline.PERIOD_GAP_DAYS)
        assert alias_targets == gap_keys

    def test_with_source_text(self, sample_dataframe):
        """Test adding source text appears in plot."""
        source_text = "Source: Test Data"
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            source_text=source_text,
        )

        # Check that source text appears in the plot's text elements
        text_elements = [text.get_text() for text in ax.figure.findobj(plt.Text)]
        assert source_text in text_elements

    def test_custom_axes(self, sample_dataframe):
        """Test plotting on a custom axes object."""
        _fig, custom_ax = plt.subplots()

        result_ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            ax=custom_ax,
        )

        assert result_ax is custom_ax

    def test_empty_dataframe_raises_error(self, empty_dataframe):
        """Test that empty dataframe raises ValueError."""
        with pytest.raises(ValueError, match="Cannot plot with empty DataFrame"):
            broken_timeline.plot(
                df=empty_dataframe,
                category_col="category",
                value_col="value",
            )

    def test_missing_column_raises_error(self, sample_dataframe):
        """Test that missing columns raise ValueError naming the missing column."""
        with pytest.raises(ValueError, match=r"\['nonexistent'\]"):
            broken_timeline.plot(
                df=sample_dataframe,
                category_col="nonexistent",
                value_col="value",
            )

    def test_invalid_period_raises_error(self, sample_dataframe):
        """Test that invalid period raises ValueError."""
        with pytest.raises(ValueError, match=r"period must be one of"):
            broken_timeline.plot(
                df=sample_dataframe,
                category_col="category",
                value_col="value",
                period="X",
            )

    def test_kwargs_passed_to_broken_barh(self, sample_dataframe):
        """Extra kwargs reach broken_barh, visible as alpha applied to the rendered collections."""
        alpha = 0.5
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
            alpha=alpha,
        )

        expected_collections = sample_dataframe["category"].nunique()
        assert len(ax.collections) == expected_collections
        assert all(collection.get_alpha() == alpha for collection in ax.collections)

    def test_y_axis_inverted(self, sample_dataframe):
        """Test that y-axis is inverted (categories from top to bottom)."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        # Check that y-axis is inverted
        y_lim = ax.get_ylim()
        assert y_lim[0] > y_lim[1]  # First value should be greater than second for inverted axis

    def test_duplicate_date_category_combinations(self):
        """Duplicate (date, category) rows collapse into a single category row with one contiguous segment."""
        date_col = get_option("column.transaction_date")

        data = {
            date_col: pd.to_datetime(["2025-04-01", "2025-04-01", "2025-04-02"]),
            "category": ["Store_North", "Store_North", "Store_North"],
            "value": [100, 200, 150],
        }
        df_with_duplicates = pd.DataFrame(data)

        ax = broken_timeline.plot(
            df=df_with_duplicates,
            category_col="category",
            value_col="value",
        )

        assert [t.get_text() for t in ax.get_yticklabels()] == ["Store_North"]
        assert sum(len(c.get_paths()) for c in ax.collections) == 1

    def test_categories_sorted_on_y_axis(self, sample_dataframe):
        """Test that categories are sorted on the y-axis."""
        ax = broken_timeline.plot(
            df=sample_dataframe,
            category_col="category",
            value_col="value",
        )

        y_labels = [label.get_text() for label in ax.get_yticklabels()]
        assert y_labels == sorted(y_labels)  # Should be sorted

    def test_no_data_for_category(self):
        """A category whose only row falls below threshold is dropped from the y-axis entirely."""
        date_col = get_option("column.transaction_date")

        data = {
            date_col: pd.to_datetime(["2025-04-01", "2025-04-02"]),
            "category": ["Store_North", "Store_South"],
            "value": [50, 200],
        }
        df = pd.DataFrame(data)

        ax = broken_timeline.plot(
            df=df,
            category_col="category",
            value_col="value",
            threshold_value=100,
        )

        assert [t.get_text() for t in ax.get_yticklabels()] == ["Store_South"]

    @pytest.mark.parametrize(
        ("period", "dates", "num_periods"),
        [
            ("D", ["2025-01-01", "2025-01-02", "2025-01-03"], 3),
            ("W", ["2025-01-01", "2025-01-08"], 2),
        ],
    )
    def test_bar_width_calculation_for_different_periods(self, period, dates, num_periods):
        """Test that bar widths are calculated correctly for different time periods."""
        date_col = get_option("column.transaction_date")
        expected_width = num_periods * broken_timeline.PERIOD_GAP_DAYS[period]

        data = {
            date_col: pd.to_datetime(dates),
            "category": ["Store_North"] * len(dates),
            "value": [100] * len(dates),
        }
        df = pd.DataFrame(data)

        ax = broken_timeline.plot(df, "category", "value", period=period)
        paths = ax.collections[0].get_paths()
        first_path_vertices = paths[0].vertices
        actual_width = first_path_vertices[2][0] - first_path_vertices[0][0]
        assert actual_width == expected_width

    @pytest.mark.parametrize(
        ("period", "dates", "expected_segments"),
        [
            (
                "D",
                ["2025-01-01", "2025-01-02", "2025-01-06", "2025-01-07"],
                2,  # 4-day gap > 1-day threshold creates 2 segments
            ),
            (
                "W",
                ["2025-01-01", "2025-01-08", "2025-01-22", "2025-01-29"],
                2,  # 14-day gap > 7-day threshold creates 2 segments
            ),
        ],
    )
    def test_gap_detection_with_different_periods(self, period, dates, expected_segments):
        """Test that gaps are correctly detected based on period type."""
        date_col = get_option("column.transaction_date")

        data = {
            date_col: pd.to_datetime(dates),
            "category": ["Store_North"] * len(dates),
            "value": [100] * len(dates),
        }
        df = pd.DataFrame(data)

        ax = broken_timeline.plot(df, "category", "value", period=period)
        paths = ax.collections[0].get_paths()
        assert len(paths) == expected_segments
