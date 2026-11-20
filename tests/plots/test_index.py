"""Tests for the index plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex

from openretailscience.analysis.index_analysis import BASELINE_INDEX
from openretailscience.options import ColumnHelper
from openretailscience.plots.index import _filter_by_groups, _filter_by_value_thresholds, _filter_top_bottom_n, plot
from openretailscience.plots.styles.colors import get_named_color

cols = ColumnHelper()

LOYALTY_ASCENDING = ["Snacks", "Meat", "Dairy", "Bakery", "Produce"]
DEPARTMENTS = sorted(LOYALTY_ASCENDING)


@pytest.fixture
def department_spend() -> pd.DataFrame:
    """Loyalty and Regular spend by department.

    Loyalty indexes: Snacks 62.1, Meat 74.5, Dairy 79.0, Bakery 124.1, Produce 130.3.
    """
    return pd.DataFrame(
        {
            "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
            "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
            "store_format": ["Mall", "Outlet"] * 5,
            "spend": [100, 200, 150, 300, 50, 120, 80, 200, 100, 90],
        },
    )


def _plot_loyalty_index(df: pd.DataFrame, **kwargs) -> plt.Axes:
    """Plot each department's spend index for Loyalty customers against all customers."""
    return plot(
        df, value_col="spend", group_col="department", index_col="cust_type", value_to_index="Loyalty", **kwargs
    )


def _y_labels(ax: plt.Axes) -> list[str]:
    """Return the y tick labels, one per bar row, bottom to top."""
    return [t.get_text() for t in ax.get_yticklabels()]


class TestIndexPlot:
    """Tests for index.plot."""

    def teardown_method(self):
        """Close the figures each test draws."""
        plt.close("all")

    def test_chrome_arguments_reach_the_figure(self, department_spend):
        """title, source_text, x_label and y_label are drawn on the figure."""
        ax = _plot_loyalty_index(
            department_spend,
            title="Loyalty spend by department",
            source_text="Source: Q1 transactions",
            x_label="Index",
            y_label="Department",
        )

        figure_texts = [t.get_text() for t in ax.figure.texts]
        assert figure_texts.count("Loyalty spend by department") == 1
        assert figure_texts.count("Source: Q1 transactions") == 1
        assert (ax.get_xlabel(), ax.get_ylabel()) == ("Index", "Department")

    @pytest.mark.parametrize(
        ("highlight_range", "expected_spans"),
        [
            pytest.param("default", [(80, 120)], id="default"),
            pytest.param((90, 110), [(90, 110)], id="custom"),
            pytest.param(None, [], id="none"),
        ],
    )
    def test_highlight_range_shades_the_given_range(self, department_spend, highlight_range, expected_spans):
        """The highlight span covers the given index range; "default" is 80 to 120 and None draws none."""
        ax = _plot_loyalty_index(department_spend, highlight_range=highlight_range)

        # The span is the only patch drawn translucent; bars are opaque
        spans = [(p.get_x(), p.get_x() + p.get_width()) for p in ax.patches if p.get_alpha() == 0.1]  # noqa: PLR2004
        assert spans == expected_spans

    def test_include_only_groups_keeps_groups_the_subset_lacks(self):
        """include_only_groups keeps the listed groups, including one with no Loyalty rows (index 0)."""
        df = pd.DataFrame(
            {
                "department": ["Bakery", "Dairy", "Produce", "Bakery", "Dairy", "Produce"],
                "cust_type": ["Loyalty", "Regular", "Loyalty", "Regular", "Regular", "Regular"],
                "spend": [100, 80, 50, 50, 80, 70],
            },
        )

        ax = _plot_loyalty_index(df, include_only_groups=["Bakery", "Dairy"])

        assert _y_labels(ax) == ["Bakery", "Dairy"]

    @pytest.mark.parametrize("series_col", [None, "region"])
    @pytest.mark.parametrize(
        ("drop_na", "expected_departments"),
        [
            (True, ["Bakery", "Dairy", "Meat"]),
            (False, ["Bakery", "Dairy", "Meat", "Snacks"]),
        ],
    )
    def test_drop_na_drops_groups_with_a_null_index(self, series_col, drop_na, expected_departments):
        """A group with no baseline spend has a null index; only drop_na removes its row, with or without series."""
        df = pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Snacks"] * 2,
                "region": ["North", "South"] * 4,
                "cust_type": ["Loyalty"] * 4 + ["Regular"] * 4,
                "spend": [100, 200, 150, 0, 120, 80, 200, 0],
            },
        )

        ax = _plot_loyalty_index(df, series_col=series_col, drop_na=drop_na)

        assert _y_labels(ax) == expected_departments

    def test_filter_above_keeps_groups_above_the_threshold(self, department_spend):
        """filter_above=100 keeps only the departments Loyalty customers over-index on."""
        ax = _plot_loyalty_index(department_spend, filter_above=BASELINE_INDEX)

        assert set(_y_labels(ax)) == {"Bakery", "Produce"}

    @pytest.mark.parametrize(
        ("sort_by", "sort_order", "top_n", "bottom_n", "expected_y_labels"),
        [
            pytest.param("value", "ascending", None, None, LOYALTY_ASCENDING, id="value_asc"),
            pytest.param("value", "descending", None, None, LOYALTY_ASCENDING[::-1], id="value_desc"),
            pytest.param("value", "ascending", 3, None, ["Dairy", "Bakery", "Produce"], id="value_asc_top3"),
            pytest.param("value", "ascending", None, 2, ["Snacks", "Meat"], id="value_asc_bottom2"),
            pytest.param("group", "descending", None, None, DEPARTMENTS[::-1], id="group_desc"),
            pytest.param(None, "ascending", None, None, DEPARTMENTS, id="no_sort_asc"),
            pytest.param(None, "descending", None, None, DEPARTMENTS, id="no_sort_ignores_order"),
            pytest.param(None, "ascending", 2, 2, ["Produce", "Bakery", "Meat", "Snacks"], id="no_sort_top2_bottom2"),
            pytest.param("VALUE", "Desc", None, None, LOYALTY_ASCENDING[::-1], id="mixed_case"),
        ],
    )
    def test_sort_and_top_bottom_n_set_the_bars(
        self,
        department_spend,
        sort_by,
        sort_order,
        top_n,
        bottom_n,
        expected_y_labels,
    ):
        """sort_by, sort_order, top_n and bottom_n decide which bars appear and their order."""
        ax = _plot_loyalty_index(
            department_spend, sort_by=sort_by, sort_order=sort_order, top_n=top_n, bottom_n=bottom_n
        )

        assert _y_labels(ax) == expected_y_labels

    @pytest.mark.parametrize(
        ("sort_order", "expected_y_labels", "expected_legend"),
        [
            ("ascending", ["Bakery", "Dairy", "Produce"], ["North", "South"]),
            ("descending", ["Produce", "Dairy", "Bakery"], ["South", "North"]),
        ],
    )
    def test_series_col_draws_a_bar_per_series_sorted_by_group(self, sort_order, expected_y_labels, expected_legend):
        """With series_col, groups and series values sort by name, and each series value gets a legend entry."""
        df = pd.DataFrame(
            {
                "department": ["Bakery", "Dairy", "Produce"] * 2,
                "region": ["North"] * 3 + ["South"] * 3,
                "cust_type": ["Loyalty", "Regular", "Loyalty", "Regular", "Loyalty", "Regular"],
                "spend": [100, 200, 150, 120, 180, 160],
            },
        )

        ax = _plot_loyalty_index(df, series_col="region", sort_by="group", sort_order=sort_order)

        assert _y_labels(ax) == expected_y_labels
        assert [t.get_text() for t in ax.get_legend().get_texts()] == expected_legend

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"sort_by": "sideways"}, "sort_by must be one of", id="sort_by"),
            pytest.param({"sort_order": "sideways"}, "sort_order must be one of", id="sort_order"),
            pytest.param(
                {"exclude_groups": ["Dairy"], "include_only_groups": ["Bakery"]},
                "exclude_groups and include_only_groups cannot be used together",
                id="exclude_and_include",
            ),
            pytest.param(
                {"series_col": "store_format", "top_n": 2},
                "top_n, bottom_n, filter_above, and filter_below cannot be used when series_col is provided",
                id="series_with_top_n",
            ),
            pytest.param(
                {"series_col": "store_format", "sort_by": "value"},
                "sort_by cannot be 'value' when series_col is provided",
                id="series_with_value_sort",
            ),
            pytest.param(
                {"color_by_threshold": True, "highlight_range": None},
                "color_by_threshold requires highlight_range to be set",
                id="threshold_colors_without_range",
            ),
            pytest.param(
                {"color_by_threshold": True, "series_col": "store_format"},
                "color_by_threshold cannot be used when series_col is provided",
                id="threshold_colors_with_series",
            ),
            pytest.param({"series_col": "region"}, "^series_col references columns not present .*region", id="missing"),
            pytest.param({"series_col": "department"}, "^series_col must differ from group_col", id="series_is_group"),
            pytest.param({"filter_above": 120, "filter_below": 80}, "must be < filter_below", id="filters_overlap"),
            pytest.param({"filter_above": 100, "filter_below": 100}, "must be < filter_below", id="filters_equal"),
            pytest.param({"top_n": -1}, "must not be negative", id="negative_top_n"),
            pytest.param({"bottom_n": -1}, "must not be negative", id="negative_bottom_n"),
        ],
    )
    def test_rejects_invalid_arguments(self, department_spend, kwargs, match):
        """Invalid or conflicting display options raise before anything is drawn, naming the parameter."""
        with pytest.raises(ValueError, match=match):
            _plot_loyalty_index(department_spend, **kwargs)

    @pytest.mark.parametrize(
        "kwargs",
        [
            pytest.param({"filter_above": 1000}, id="threshold"),
            pytest.param({"include_only_groups": ["Frozen"]}, id="groups"),
            pytest.param({"include_only_groups": ["Frozen"], "series_col": "store_format"}, id="groups_with_series"),
        ],
    )
    def test_filters_that_leave_no_groups_raise(self, department_spend, kwargs):
        """Filters that leave nothing to draw raise the same clear error, with or without series_col."""
        with pytest.raises(ValueError, match="Filtering resulted in an empty dataset"):
            _plot_loyalty_index(department_spend, **kwargs)

    def test_rejects_data_that_is_not_a_table_with_series_col(self, department_spend):
        """Data that is neither a DataFrame nor an Ibis Table raises a TypeError, also when series_col is set."""
        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            _plot_loyalty_index(department_spend.to_dict("list"), series_col="store_format")


class TestColorByThreshold:
    """Tests for color_by_threshold in the index plot."""

    def teardown_method(self):
        """Close the figures each test draws."""
        plt.close("all")

    @pytest.fixture
    def threshold_data(self):
        """Departments indexing Snacks ≈ 60, Meat ≈ 72, Dairy ≈ 99, Bakery ≈ 120 and Produce ≈ 126.

        Covers all three colors for both the default (80, 120) and a custom (90, 110) range.
        """
        return pd.DataFrame(
            {
                "department": ["Dairy", "Bakery", "Meat", "Produce", "Snacks"] * 2,
                "cust_type": ["Loyalty"] * 5 + ["Regular"] * 5,
                "spend": [130, 200, 150, 300, 50, 90, 80, 200, 100, 90],
            },
        )

    @pytest.mark.parametrize(
        ("expected_color_names", "plot_kwargs"),
        [
            pytest.param(
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (neutral), Produce≈126 (pos)
                ["negative", "negative", "neutral", "neutral", "positive"],
                {"sort_by": "value", "sort_order": "ascending"},
                id="default_range_value_sort",
            ),
            pytest.param(
                # Snacks≈60 (neg), Meat≈72 (neg), Dairy≈99 (neutral), Bakery≈120 (pos), Produce≈126 (pos)
                ["negative", "negative", "neutral", "positive", "positive"],
                {"highlight_range": (90, 110), "sort_by": "value", "sort_order": "ascending"},
                id="custom_range_value_sort",
            ),
            pytest.param(
                # Bakery≈120 (neutral), Dairy≈99 (neutral), Meat≈72 (neg), Produce≈126 (pos), Snacks≈60 (neg)
                ["neutral", "neutral", "negative", "positive", "negative"],
                {},
                id="default_range_group_sort",
            ),
        ],
    )
    def test_bars_colored_by_threshold(self, threshold_data, expected_color_names, plot_kwargs):
        """Bars are colored positive, negative or neutral by the highlight range bounds."""
        ax = _plot_loyalty_index(threshold_data, color_by_threshold=True, **plot_kwargs)

        expected_colors = [get_named_color(name) for name in expected_color_names]
        # Bars are opaque; the highlight span is drawn at alpha 0.1
        bar_patches = [p for p in ax.patches if p.get_alpha() is None or p.get_alpha() == 1.0]
        assert [to_hex(p.get_facecolor()) for p in bar_patches] == expected_colors


class TestIndexPlotBars:
    """Tests for where index.plot draws its bars."""

    def teardown_method(self):
        """Close the figures each test draws."""
        plt.close("all")

    def test_bars_run_from_the_baseline_to_each_index(self):
        """Each bar starts at 100 and ends at its group's index, including a band no Meadow Farm customer is in."""
        df = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8],
                "brand": ["Meadow Farm"] * 3 + ["Greenvale"] * 5,
                "age_band": ["18-34", "35-64", "35-64", "18-34", "35-64", "65+", "65+", "65+"],
            },
        )
        expected_index = np.array([100 * (1 / 3) / (2 / 8), 100 * (2 / 3) / (3 / 8), 0.0])

        ax = plot(
            df,
            value_col=cols.customer_id,
            group_col="age_band",
            index_col="brand",
            value_to_index="Meadow Farm",
            agg_func="nunique",
        )

        assert _y_labels(ax) == ["18-34", "35-64", "65+"]
        assert [p.get_x() for p in ax.containers[0]] == [BASELINE_INDEX] * len(expected_index)
        np.testing.assert_allclose([p.get_width() for p in ax.containers[0]], expected_index - BASELINE_INDEX)

    @pytest.mark.parametrize("color_by_threshold", [False, True])
    def test_numeric_groups_are_drawn_in_sort_order(self, color_by_threshold):
        """Numeric store ids get one bar each, ordered by index, not placed at their numeric value."""
        df = pd.DataFrame(
            {
                cols.store_id: [101, 102, 103, 101, 102, 103],
                "brand": ["Meadow Farm"] * 3 + ["Greenvale"] * 3,
                cols.unit_spend: [50, 10, 30, 10, 40, 20],
            },
        )

        ax = plot(
            df,
            value_col=cols.unit_spend,
            group_col=cols.store_id,
            index_col="brand",
            value_to_index="Meadow Farm",
            sort_by="value",
            color_by_threshold=color_by_threshold,
        )

        assert _y_labels(ax) == ["102", "103", "101"]


class TestFilterHelpers:
    """Tests for the plot's display filters, which take the result table with a raw ``index`` column."""

    @pytest.fixture
    def departments_around_baseline(self) -> pd.DataFrame:
        """Three departments indexing just under, at and just over the baseline."""
        return pd.DataFrame({"department": ["Bakery", "Dairy", "Produce"], "index": [99.0, 100.0, 101.0]})

    @pytest.mark.parametrize(
        ("kwargs", "expected_departments"),
        [
            pytest.param({"filter_above": 100}, ["Produce"], id="filter_above_strict_boundary"),
            pytest.param({"filter_below": 100}, ["Bakery"], id="filter_below_strict_boundary"),
            pytest.param({"filter_above": 99, "filter_below": 101}, ["Dairy"], id="open_interval"),
        ],
    )
    def test_value_thresholds_are_strict(self, departments_around_baseline, kwargs, expected_departments):
        """Thresholds compare against the raw index with strict inequality."""
        result = _filter_by_value_thresholds(df=departments_around_baseline, **kwargs)

        assert result["department"].tolist() == expected_departments

    def test_exclude_groups_drops_only_those_groups(self, departments_around_baseline):
        """exclude_groups removes the listed groups and keeps the rest."""
        result = _filter_by_groups(df=departments_around_baseline, group_col="department", exclude_groups=["Dairy"])

        assert result["department"].tolist() == ["Bakery", "Produce"]

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"top_n": 4}, r"top_n \(4\) cannot exceed the number of available groups \(3\)", id="top"),
            pytest.param({"bottom_n": 4}, r"bottom_n \(4\) cannot exceed the number of available groups", id="bottom"),
            pytest.param({"top_n": 2, "bottom_n": 2}, r"The sum of top_n .* and bottom_n .* cannot exceed", id="sum"),
        ],
    )
    def test_top_and_bottom_n_must_fit_the_group_count(self, departments_around_baseline, kwargs, match):
        """A count above the number of groups, or top_n + bottom_n above it, raises."""
        with pytest.raises(ValueError, match=match):
            _filter_top_bottom_n(df=departments_around_baseline, **kwargs)

    def test_bottom_n_ranks_only_groups_with_an_index(self, departments_around_baseline):
        """A group with a null index is not the lowest; bottom_n picks the lowest real index."""
        df = pd.concat(
            [departments_around_baseline, pd.DataFrame({"department": ["Snacks"], "index": [np.nan]})],
            ignore_index=True,
        )

        result = _filter_top_bottom_n(df=df, bottom_n=1)

        assert result["department"].tolist() == ["Bakery"]
