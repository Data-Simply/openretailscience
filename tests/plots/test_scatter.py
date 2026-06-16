"""Tests for the plots.scatter module."""

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.plots import scatter

PERIODS = 6


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_sales_dataframe():
    """A sample dataframe for Sales, Profit, and Expenses data."""
    rng = np.random.default_rng(42)
    data = {
        "date": pd.date_range("2023-01-01", periods=PERIODS, freq="ME"),
        "sales": rng.integers(1000, 5000, size=PERIODS),
        "profit": rng.integers(200, 1000, size=PERIODS),
        "expenses": rng.integers(500, 3000, size=PERIODS),
        "category": ["Electronics", "Clothing", "Furniture", "Electronics", "Clothing", "Furniture"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_product_dataframe():
    """A sample dataframe for product performance with labels."""
    data = {
        "price": [10.5, 15.0, 8.5, 12.0, 20.0],
        "units_sold": [150, 120, 200, 90, 60],
        "product_name": ["Widget A", "Widget B", "Widget C", "Widget D", "Widget E"],
        "category": ["Electronics", "Electronics", "Home", "Home", "Electronics"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_store_dataframe():
    """A sample dataframe for store performance by region with labels."""
    data = {
        "revenue": [50000, 75000, 45000, 90000, 65000, 55000],
        "customer_count": [1200, 1800, 1000, 2200, 1500, 1300],
        "store_id": ["S001", "S002", "S003", "S004", "S005", "S006"],
        "region": ["North", "North", "South", "South", "East", "East"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def bubble_chart_dataframe():
    """A sample dataframe for bubble chart testing with size column."""
    data = {
        "sales": [100, 150, 80, 200, 120, 90, 180, 110, 160, 140],
        "profit_margin": [0.15, 0.22, 0.12, 0.25, 0.18, 0.10, 0.23, 0.16, 0.21, 0.19],
        "store_sqft": [5000, 7500, 4000, 10000, 6000, 4500, 9000, 5500, 8000, 6500],
        "region": ["North", "North", "South", "North", "South", "South", "North", "South", "North", "South"],
        "store_id": [f"Store_{i + 1}" for i in range(10)],
    }
    return pd.DataFrame(data)


def test_plot_with_figsize(sample_sales_dataframe):
    """Test scatter plot sizes the figure when figsize is passed and no ax is provided."""
    width, height = 12, 8
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_col="date",
        figsize=(width, height),
    )

    assert result_ax.figure.get_size_inches()[0] == width
    assert result_ax.figure.get_size_inches()[1] == height


def test_plot_single_column(sample_sales_dataframe):
    """Test scatter plot with a single value column."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Monthly Sales Trend",
        x_col="date",
    )
    assert isinstance(result_ax, Axes)

    # Verify scatter plot collections were created
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) == 1

    # Verify correct number of data points
    offsets = collections[0].get_offsets()
    assert len(offsets) == len(sample_sales_dataframe), (
        f"Expected {len(sample_sales_dataframe)} points, got {len(offsets)}"
    )


def test_plot_with_group_col(sample_sales_dataframe):
    """Test scatter plot with a group column."""
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Sales by Category",
        x_col="date",
        group_col="category",
    )
    assert isinstance(result_ax, Axes)

    # Verify scatter plot collections were created for groups
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    unique_categories = sample_sales_dataframe["category"].nunique()
    assert len(collections) == unique_categories, (
        f"Expected {unique_categories} collections for groups, got {len(collections)}"
    )

    # Verify total data points match original dataframe
    total_points = sum(len(collection.get_offsets()) for collection in collections)
    assert total_points == len(sample_sales_dataframe), (
        f"Total points should match dataframe length, got {total_points}"
    )


def test_plot_multiple_columns(sample_sales_dataframe):
    """Test scatter plot with multiple value columns."""
    value_cols = ["sales", "profit"]
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col=value_cols,
        x_label="Date",
        y_label="Amount",
        title="Sales & Profit Trends",
        x_col="date",
    )
    assert isinstance(result_ax, Axes)

    # Verify scatter plot collections were created for each column
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) == len(value_cols), (
        f"Expected {len(value_cols)} collections for multiple columns, got {len(collections)}"
    )

    # Verify each collection has correct number of points
    for collection in collections:
        offsets = collection.get_offsets()
        assert len(offsets) == len(sample_sales_dataframe), (
            f"Each collection should have {len(sample_sales_dataframe)} points, got {len(offsets)}"
        )


def test_plot_multiple_columns_with_group_col(sample_sales_dataframe):
    """Test scatter plot when using multiple columns along with a group column."""
    with pytest.raises(ValueError, match=r"Cannot use both a list for `value_col` and a `group_col`. Choose one."):
        scatter.plot(
            df=sample_sales_dataframe,
            value_col=["sales", "profit", "expenses"],
            x_label="Date",
            y_label="Amount",
            title="Sales, Profit & Expenses",
            x_col="date",
            group_col="category",
        )


def test_plot_single_column_series(sample_sales_dataframe):
    """Test scatter plot with a single value as a Pandas Series."""
    sales_series = sample_sales_dataframe["sales"]
    result_ax = scatter.plot(
        df=sales_series,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Sales Trend (Series)",
    )
    assert isinstance(result_ax, Axes)

    # Verify scatter plot was created from series
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) == 1

    # Verify correct number of data points from series
    offsets = collections[0].get_offsets()
    assert len(offsets) == len(sales_series), f"Expected {len(sales_series)} points from series, got {len(offsets)}"


def test_plot_with_labels_single_series(sample_product_dataframe):
    """Each row's label_col value is rendered as a Text annotation on the axes."""
    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        title="Product Performance",
        x_label="Price",
        y_label="Units Sold",
    )

    title_texts = [t for t in result_ax.figure.texts if t.get_text() == "Product Performance"]
    assert len(title_texts) == 1
    assert result_ax.get_xlabel() == "Price"
    assert result_ax.get_ylabel() == "Units Sold"

    rendered_labels = [t.get_text() for t in result_ax.texts]
    assert len(rendered_labels) == len(sample_product_dataframe)
    assert sorted(rendered_labels) == sorted(sample_product_dataframe["product_name"])


def test_plot_with_labels_grouped_series(sample_store_dataframe):
    """Grouped scatter plot still renders one Text annotation per row."""
    result_ax = scatter.plot(
        df=sample_store_dataframe,
        value_col="revenue",
        x_col="customer_count",
        group_col="region",
        label_col="store_id",
        title="Store Performance by Region",
        x_label="Customer Count",
        y_label="Revenue",
    )

    title_texts = [t for t in result_ax.figure.texts if t.get_text() == "Store Performance by Region"]
    assert len(title_texts) == 1
    assert result_ax.get_xlabel() == "Customer Count"
    assert result_ax.get_ylabel() == "Revenue"

    # One scatter collection per region (separate styling per group).
    collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
    assert len(collections) == sample_store_dataframe["region"].nunique()

    rendered_labels = [t.get_text() for t in result_ax.texts]
    assert len(rendered_labels) == len(sample_store_dataframe)
    assert sorted(rendered_labels) == sorted(sample_store_dataframe["store_id"])


def test_plot_label_kwargs_are_forwarded_to_textalloc(sample_product_dataframe):
    """label_kwargs values override defaults and reach the rendered Text objects."""
    distinctive_color = "#7f3fbf"

    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        label_kwargs={"textcolor": distinctive_color},
    )

    matching_texts = [t for t in result_ax.texts if mcolors.to_hex(t.get_color()) == distinctive_color]
    assert len(matching_texts) == len(sample_product_dataframe)


@pytest.mark.parametrize(
    ("value_col", "label_col", "expected_error", "expected_msg"),
    [
        (["units_sold", "price"], "product_name", ValueError, "label_col is not supported when value_col is a list"),
        ("units_sold", "nonexistent_col", KeyError, "label_col 'nonexistent_col' not found in DataFrame"),
    ],
)
def test_plot_label_validation_errors(sample_product_dataframe, value_col, label_col, expected_error, expected_msg):
    """Test various label validation errors."""
    with pytest.raises(expected_error, match=expected_msg):
        scatter.plot(
            df=sample_product_dataframe,
            value_col=value_col,
            x_col="price",
            label_col=label_col,
        )


def test_plot_labels_with_nan_values(sample_product_dataframe):
    """Rows with NaN in value_col or label_col are dropped before labels are rendered."""
    df_with_nan = sample_product_dataframe.copy()
    df_with_nan.loc[1, "units_sold"] = np.nan
    df_with_nan.loc[3, "product_name"] = np.nan

    result_ax = scatter.plot(
        df=df_with_nan,
        value_col="units_sold",
        x_col="price",
        label_col="product_name",
        title="Product Performance with NaN values",
    )

    expected_non_nan_count = len(df_with_nan.dropna(subset=["units_sold", "product_name"]))
    assert len(result_ax.texts) == expected_non_nan_count


def test_plot_labels_using_index_as_x(sample_product_dataframe):
    """When x_col is omitted, every row still gets a label rendered against the index."""
    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        label_col="product_name",
        title="Product Performance using Index",
    )

    title_texts = [t for t in result_ax.figure.texts if t.get_text() == "Product Performance using Index"]
    assert len(title_texts) == 1
    rendered_labels = [t.get_text() for t in result_ax.texts]
    assert len(rendered_labels) == len(sample_product_dataframe)
    assert sorted(rendered_labels) == sorted(sample_product_dataframe["product_name"])


def test_plot_without_labels_renders_no_text_annotations(sample_product_dataframe):
    """Without label_col the axes carry no Text annotations."""
    result_ax = scatter.plot(
        df=sample_product_dataframe,
        value_col="units_sold",
        x_col="price",
        title="Product Performance without Labels",
    )

    assert len(result_ax.texts) == 0


def test_plot_adds_source_text(sample_sales_dataframe):
    """Test scatter plot renders source text annotation."""
    source_text = "Source: Test Data"
    result_ax = scatter.plot(
        df=sample_sales_dataframe,
        value_col="sales",
        x_label="Date",
        y_label="Sales",
        title="Test Plot Source Text",
        x_col="date",
        source_text=source_text,
    )
    source_texts = [text for text in result_ax.figure.texts if text.get_text() == source_text]
    assert len(source_texts) == 1


class TestBubbleChartFeature:
    """Tests for the bubble chart functionality (size_col parameter)."""

    @pytest.mark.parametrize("x_col", ["sales", None], ids=["explicit_x_col", "index_x_col"])
    @pytest.mark.parametrize("size_scale", [0.01, 0.5, 1.0, 2.0, 10.0])
    def test_bubble_chart_sizes(self, bubble_chart_dataframe, x_col, size_scale):
        """Test bubble chart sizes are correctly scaled with and without explicit x_col."""
        plot_kwargs = {
            "df": bubble_chart_dataframe,
            "value_col": "profit_margin",
            "size_col": "store_sqft",
            "size_scale": size_scale,
            "title": "Store Performance Bubble Chart",
        }
        if x_col is not None:
            plot_kwargs["x_col"] = x_col

        result_ax = scatter.plot(**plot_kwargs)

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        sizes = collections[0].get_sizes()
        expected_sizes = bubble_chart_dataframe["store_sqft"].to_numpy() * size_scale
        assert np.array_equal(sizes, expected_sizes), f"Sizes should be scaled by {size_scale}"

    @pytest.mark.parametrize("x_col", ["sales", None], ids=["explicit_x_col", "index_x_col"])
    @pytest.mark.parametrize("size_scale", [1.0, 0.5], ids=["default_scale", "half_scale"])
    def test_bubble_chart_with_groups(self, bubble_chart_dataframe, x_col, size_scale):
        """Test bubble chart with group_col parameter, with and without explicit x_col and size_scale."""
        plot_kwargs = {
            "df": bubble_chart_dataframe,
            "value_col": "profit_margin",
            "group_col": "region",
            "size_col": "store_sqft",
            "size_scale": size_scale,
            "title": "Grouped Bubble Chart",
        }
        if x_col is not None:
            plot_kwargs["x_col"] = x_col

        result_ax = scatter.plot(**plot_kwargs)

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        unique_regions = bubble_chart_dataframe["region"].nunique()
        assert len(collections) == unique_regions, f"Expected {unique_regions} collections for groups"

        total_points = sum(len(collection.get_offsets()) for collection in collections)
        assert total_points == len(bubble_chart_dataframe), (
            f"Total points should match dataframe length, got {total_points}"
        )

        for collection in collections:
            sizes = collection.get_sizes()
            offsets = collection.get_offsets()
            assert len(sizes) == len(offsets), "Each point should have a corresponding size"

        all_sizes = sorted(np.concatenate([c.get_sizes() for c in collections]))
        expected_all_sizes = sorted(bubble_chart_dataframe["store_sqft"].to_numpy() * size_scale)
        assert np.array_equal(all_sizes, expected_all_sizes), (
            f"Grouped bubble chart sizes should be scaled by {size_scale}"
        )

    def test_bubble_chart_none_size_col(self, bubble_chart_dataframe):
        """Test that size_col=None behaves like original implementation with uniform sizing."""
        result_ax = scatter.plot(
            df=bubble_chart_dataframe,
            x_col="sales",
            value_col="profit_margin",
            size_col=None,  # Explicitly test None
            title="Regular Scatter Plot",
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        # When size_col is None, all points should have uniform default size
        # matplotlib stores a single broadcast value when no per-point sizes are set
        sizes = collections[0].get_sizes()
        assert len(sizes) == 1, "Expected single broadcast size when size_col=None, got per-point sizes"

    @pytest.mark.parametrize(
        ("size_col", "expected_error", "expected_message"),
        [
            ("nonexistent_col", KeyError, "size_col 'nonexistent_col' not found in DataFrame"),
            ("store_id", ValueError, "size_col 'store_id' must contain numeric values"),
        ],
    )
    def test_bubble_chart_size_col_validation_errors(
        self,
        bubble_chart_dataframe,
        size_col,
        expected_error,
        expected_message,
    ):
        """Test size_col validation error handling."""
        with pytest.raises(expected_error, match=expected_message):
            scatter.plot(
                df=bubble_chart_dataframe,
                x_col="sales",
                value_col="profit_margin",
                size_col=size_col,
            )

    @pytest.mark.parametrize("size_scale", [0, -1.0])
    def test_bubble_chart_invalid_size_scale_raises(self, bubble_chart_dataframe, size_scale):
        """Test that non-positive size_scale raises ValueError."""
        with pytest.raises(ValueError, match="size_scale must be a positive number"):
            scatter.plot(
                df=bubble_chart_dataframe,
                x_col="sales",
                value_col="profit_margin",
                size_col="store_sqft",
                size_scale=size_scale,
            )

    def test_bubble_chart_s_parameter_override(self, bubble_chart_dataframe):
        """Test that user-provided 's' parameter is overridden by size_col with a warning."""
        with pytest.warns(UserWarning, match="'s' keyword argument is ignored when 'size_col' is specified"):
            result_ax = scatter.plot(
                df=bubble_chart_dataframe,
                x_col="sales",
                value_col="profit_margin",
                size_col="store_sqft",
                s=100,  # This should be ignored
                title="Bubble Chart with Override",
            )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        # Verify sizes come from size_col, not the 's' parameter
        sizes = collections[0].get_sizes()
        expected_sizes = bubble_chart_dataframe["store_sqft"].to_numpy()
        assert np.array_equal(sizes, expected_sizes), "Sizes should come from size_col, not 's' parameter"

    def test_s_parameter_passthrough_without_size_col(self, bubble_chart_dataframe):
        """Test that user-provided 's' parameter is passed to matplotlib when size_col is None."""
        uniform_size = 100
        result_ax = scatter.plot(
            df=bubble_chart_dataframe,
            x_col="sales",
            value_col="profit_margin",
            s=uniform_size,
            title="Scatter with Custom Size",
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        # Verify all points use the user-provided 's' value
        sizes = collections[0].get_sizes()
        assert all(s == uniform_size for s in sizes), f"All points should have size {uniform_size}, got {sizes}"

    def test_bubble_chart_multiple_value_columns(self, bubble_chart_dataframe):
        """Test bubble chart with multiple value columns and size_col."""
        value_cols = ["profit_margin", "sales"]
        result_ax = scatter.plot(
            df=bubble_chart_dataframe,
            x_col="store_sqft",
            value_col=value_cols,  # Multiple columns
            size_col="sales",  # Bubble sizes
            title="Multi-Series Bubble Chart",
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        # Verify collections for each value column
        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == len(value_cols), (
            f"Expected {len(value_cols)} collections for multiple value columns, got {len(collections)}"
        )

        # Verify each collection has sizes and correct number of points
        for i, collection in enumerate(collections):
            sizes = collection.get_sizes()
            offsets = collection.get_offsets()

            assert len(sizes) == len(bubble_chart_dataframe), (
                f"Collection {i} should have {len(bubble_chart_dataframe)} sizes, got {len(sizes)}"
            )
            assert len(offsets) == len(bubble_chart_dataframe), (
                f"Collection {i} should have {len(bubble_chart_dataframe)} points, got {len(offsets)}"
            )

            # Verify sizes match the size_col values (sales column)
            expected_sizes = bubble_chart_dataframe["sales"].to_numpy()
            assert np.array_equal(sizes, expected_sizes), f"Collection {i} sizes should match size_col values"

    def test_bubble_chart_with_negative_sizes_raises(self, bubble_chart_dataframe):
        """Test bubble chart raises ValueError when size_col contains negative values."""
        df = bubble_chart_dataframe.copy()
        df.loc[0, "store_sqft"] = -100
        df.loc[1, "store_sqft"] = -50

        with pytest.raises(ValueError, match="contains negative values"):
            scatter.plot(
                df=df,
                x_col="sales",
                value_col="profit_margin",
                size_col="store_sqft",
                title="Bubble Chart with Negative Sizes",
            )

    def test_bubble_chart_with_zero_sizes(self, bubble_chart_dataframe):
        """Test bubble chart renders correctly with zero size values."""
        df = bubble_chart_dataframe.copy()
        df.loc[0, "store_sqft"] = 0
        df.loc[1, "store_sqft"] = 0

        result_ax = scatter.plot(
            df=df,
            x_col="sales",
            value_col="profit_margin",
            size_col="store_sqft",
            title="Bubble Chart with Zero Sizes",
        )

        assert isinstance(result_ax, Axes), "Result should be an Axes object"

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        offsets = collections[0].get_offsets()
        assert len(offsets) == len(df), "All data points should be plotted"

        sizes = collections[0].get_sizes()
        expected_sizes = df["store_sqft"].to_numpy()
        assert np.array_equal(sizes, expected_sizes), "Sizes should match data including zero values"

    def test_bubble_chart_with_labels(self, bubble_chart_dataframe):
        """Bubble chart sizes come from size_col and store_id labels are rendered as text annotations."""
        result_ax = scatter.plot(
            df=bubble_chart_dataframe,
            x_col="sales",
            value_col="profit_margin",
            size_col="store_sqft",
            label_col="store_id",
            title="Labeled Bubble Chart",
        )

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1
        assert np.array_equal(collections[0].get_sizes(), bubble_chart_dataframe["store_sqft"].to_numpy())

        rendered_labels = [t.get_text() for t in result_ax.texts]
        assert len(rendered_labels) == len(bubble_chart_dataframe)
        assert sorted(rendered_labels) == sorted(bubble_chart_dataframe["store_id"].astype(str))

    def test_bubble_chart_x_col_equals_size_col(self, bubble_chart_dataframe):
        """Test bubble chart when x_col and size_col reference the same column."""
        result_ax = scatter.plot(
            df=bubble_chart_dataframe,
            x_col="sales",
            value_col="profit_margin",
            size_col="sales",
            title="Sales as Both X and Size",
        )

        assert isinstance(result_ax, Axes)

        collections = [child for child in result_ax.get_children() if hasattr(child, "get_offsets")]
        assert len(collections) == 1

        sizes = collections[0].get_sizes()
        expected_sizes = bubble_chart_dataframe["sales"].to_numpy()
        assert np.array_equal(sizes, expected_sizes), "Sizes should match sales values when x_col == size_col"
