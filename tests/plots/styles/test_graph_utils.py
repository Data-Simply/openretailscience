"""Tests for the graph_utils module in the style package."""

import datetime
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pytest
from scipy import stats

from openretailscience.options import get_option
from openretailscience.plots.styles import graph_utils as gu
from openretailscience.plots.styles import styling_helpers as sh


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


def _extract_r2_from_plot(ax: plt.Axes) -> float:
    """Extract R² value from plot text annotations.

    Args:
        ax (plt.Axes): The matplotlib axes containing the R² annotation.

    Returns:
        float: The extracted R² value.
    """
    for text in ax.texts:
        content = text.get_text()
        if "R²" in content:
            match = re.search(r"R² = (-?[\d.]+(?:e[+-]?\d+)?)", content)
            if match:
                return float(match.group(1))
    pytest.fail("R² text not found in plot annotations")


@pytest.mark.parametrize(
    ("num", "decimals", "prefix", "expected"),
    [
        # Sub-thousand values get no suffix.
        (0, 0, "", "0"),
        (0.001, 0, "", "0"),
        (500, 0, "", "500"),
        (999, 0, "", "999"),
        # Magnitude suffixes (K/M/B/T) at exact boundaries and rounded values.
        (1000, 0, "", "1K"),
        (1500, 0, "", "2K"),
        (1500000, 0, "", "2M"),
        (1000000, 0, "", "1M"),
        (1000000000, 0, "", "1B"),
        (1000000000000, 0, "", "1T"),
        # Decimals control precision and trailing-zero stripping.
        (1000, 2, "", "1K"),
        (1500, 2, "", "1.5K"),
        (1500000, 1, "", "1.5M"),
        (1500000, 3, "", "1.5M"),
        (1234567, 0, "", "1M"),
        (1234567, 2, "", "1.23M"),
        (1234567, 3, "", "1.235M"),
        (1234567, 4, "", "1.2346M"),
        # Prefixes prepend to the formatted number.
        (500, 0, "¥", "¥500"),
        (1500, 0, "$", "$2K"),
        (1500000, 0, "€", "€2M"),
        # Negative numbers preserve sign through formatting.
        (-1000, 0, "", "-1K"),
        (-1500, 0, "", "-2K"),
        (-1000000, 0, "", "-1M"),
        (-1500000, 1, "", "-1.5M"),
        (-1234567, 3, "", "-1.235M"),
        (-1000000000, 0, "", "-1B"),
        (-1000000000, 2, "", "-1B"),
        # Values that round up to the next magnitude get promoted.
        (999.999, 0, "", "1K"),
        (999.999, 2, "", "1K"),
        (999999.999, 0, "", "1M"),
        (1000.0, 0, "", "1K"),
        # Within the predefined suffix range; "P" is the last predefined suffix.
        (10**15, 0, "", "1P"),
        (10**16, 0, "", "10P"),
        (10**17, 0, "", "100P"),
        # Beyond the predefined suffixes, the number keeps growing under the "P" suffix.
        (10**18, 0, "", "1000P"),
        (10**19, 0, "", "10000P"),
        (-(10**18), 0, "", "-1000P"),
    ],
)
def test_format_shorthand(num, decimals, prefix, expected):
    """format_shorthand renders numbers with K/M/B/T/P suffixes, decimals, prefixes, and sign handling."""
    assert gu.format_shorthand(num, decimals=decimals, prefix=prefix) == expected


@pytest.mark.parametrize(
    ("num_str", "digits", "expected"),
    [
        # No truncation needed: input already fits within the digit budget.
        ("1.5K", 2, "1.5K"),
        ("1.25M", 3, "1.25M"),
        ("1M", 1, "1M"),
        ("10.25M", 4, "10.25M"),
        ("123", 3, "123"),
        ("12.345", 5, "12.345"),
        ("999", 3, "999"),
        ("1.234M", 4, "1.234M"),
        ("1.234B", 4, "1.234B"),
        # Truncation drops trailing decimal digits while preserving suffix.
        ("10.25M", 3, "10.2M"),
        ("10.99M", 3, "10.9M"),
        ("1.234K", 2, "1.2K"),
        ("5.678M", 3, "5.67M"),
        ("9.999B", 2, "9.9B"),
        ("1.234P", 2, "1.2P"),
        ("0.9999", 3, "0.99"),
        ("999.999K", 4, "999.9K"),
        ("100.0001M", 4, "100M"),
        # Trailing zeros are stripped after truncation.
        ("1.500", 3, "1.5"),
        ("1.230K", 4, "1.23K"),
        ("10.000", 2, "10"),
        # Integer parts longer than the digit budget are returned untouched.
        ("500", 2, "500"),
        ("12345", 3, "12345"),
        # Zero values pass through unchanged.
        ("0", 2, "0"),
        ("0K", 2, "0K"),
        # Negative numbers preserve sign.
        ("-1.5K", 2, "-1.5K"),
        ("-1.234M", 3, "-1.23M"),
        # Very small numbers truncate to zero or retain precision when budget allows.
        ("0.001", 2, "0"),
        ("0.000009", 7, "0.000009"),
    ],
)
def test_truncate_to_x_digits(num_str, digits, expected):
    """truncate_to_x_digits keeps the first `digits` significant digits of a formatted number."""
    assert gu.truncate_to_x_digits(num_str, digits) == expected


@pytest.mark.parametrize(
    ("kwargs", "expected_xmax", "expected_decimals", "expected_symbol"),
    [
        ({}, 1, None, "%"),
        ({"xmax": 100, "decimals": 2, "symbol": "pct"}, 100, 2, "pct"),
    ],
    ids=["defaults", "custom_options"],
)
def test_set_axis_percent(kwargs, expected_xmax, expected_decimals, expected_symbol):
    """set_axis_percent applies PercentFormatter with the given options."""
    _, ax = plt.subplots()
    ax.plot([0, 0.25, 0.5, 0.75, 1.0], [0, 0.3, 0.5, 0.7, 1.0])

    gu.set_axis_percent(ax.yaxis, **kwargs)

    formatter = ax.yaxis.get_major_formatter()
    assert isinstance(formatter, mtick.PercentFormatter)
    assert formatter.xmax == expected_xmax
    assert formatter.decimals == expected_decimals
    sample_value = expected_xmax / 2
    assert formatter(sample_value).endswith(expected_symbol)


def test_set_axis_percent_symbol_none_suppresses_symbol():
    """set_axis_percent(symbol=None) renders values without the % suffix."""
    _, ax = plt.subplots()
    ax.plot([0, 0.5, 1.0], [0, 0.5, 1.0])

    gu.set_axis_percent(ax.yaxis, symbol=None)

    assert ax.yaxis.get_major_formatter()(0.5) == "50"


def test_set_axis_shorthand_renders_tick_labels():
    """set_axis_shorthand produces shorthand tick labels (1500 → '2K')."""
    _, ax = plt.subplots()
    ax.plot([0, 1, 2], [0, 1500, 3000])

    gu.set_axis_shorthand(ax.yaxis, decimals=0)

    formatter = ax.yaxis.get_major_formatter()
    assert formatter(1500) == "2K"
    assert formatter(3_700_000) == "4M"


def test_set_axis_shorthand_honours_prefix():
    """set_axis_shorthand(prefix="$") prepends the currency symbol."""
    _, ax = plt.subplots()
    ax.plot([0, 1], [0, 1500])

    gu.set_axis_shorthand(ax.yaxis, decimals=1, prefix="$")

    assert ax.yaxis.get_major_formatter()(1500) == "$1.5K"


@pytest.mark.parametrize("axis_name", ["xaxis", "yaxis"])
def test_set_axis_shorthand_auto_decimals(axis_name):
    """When decimals is None, set_axis_shorthand derives decimals from the matching axis's own ticks."""
    _, ax = plt.subplots()
    # Asymmetric ranges so xaxis and yaxis require different decimal counts.
    # The yaxis range (~1.234M..1.26M) needs several decimals to keep shorthand labels
    # distinct, while the xaxis range (0..3) needs none.
    ax.plot([0, 1, 2, 3], [1_234_567, 1_240_000, 1_250_000, 1_260_000])
    fmt_axis = getattr(ax, axis_name)

    if axis_name == "xaxis":
        expected_decimals = gu.get_decimals(ax.get_xlim(), ax.get_xticks())
    else:
        expected_decimals = gu.get_decimals(ax.get_ylim(), ax.get_yticks())

    gu.set_axis_shorthand(fmt_axis)

    formatter = fmt_axis.get_major_formatter()
    sample = 1_234_567
    assert formatter(sample) == gu.format_shorthand(sample, decimals=expected_decimals)


class TestResolveEndOfLineLabelYs:
    """Tests for the private end-of-line label position resolver."""

    def _make_candidates(self, y_ends: list[float]) -> list[gu._EndOfLineCandidate]:
        return [
            gu._EndOfLineCandidate(
                label=f"Series{i}",
                x_end=9,
                y_end=y,
                color="#000000",
                marker_size=6.0,
                zorder=2.0,
            )
            for i, y in enumerate(y_ends)
        ]

    def test_clusters_near_top_pin_bottom_label_at_data_area(self):
        """Many endpoints clustered near ylim[1] must not push the lowest label below ylim[0].

        Without a symmetric bottom clamp, the top-clamp back-propagation drives the lowest
        label to ``y_top_px - (n-1)*min_gap_px`` which can fall below the data area on a
        short figure with many series. The fix pins the bottom label at ``ylim[0]``;
        any unavoidable overflow (when labels physically can't fit) moves to the top edge
        instead, where it conflicts with title space rather than chart data.
        """
        fig, ax = plt.subplots(figsize=(4, 1.0), dpi=100)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 10)
        fig.canvas.draw()

        candidates = self._make_candidates([99.0 - 0.1 * i for i in range(6)])
        label_ys = gu._resolve_end_of_line_label_ys(ax, candidates, font_pts=12.0)

        ylim_low, _ = ax.get_ylim()
        assert min(label_ys) >= ylim_low - 1e-6, f"min(label_ys)={min(label_ys)} fell below ylim[0]={ylim_low}"

    def test_well_spaced_endpoints_unchanged(self):
        """When endpoints already exceed the min gap, no bumping or clamping happens."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 10)
        fig.canvas.draw()

        y_ends = [10.0, 30.0, 50.0, 70.0, 90.0]
        candidates = self._make_candidates(y_ends)

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            label_ys = gu._resolve_end_of_line_label_ys(ax, candidates, font_pts=8.0)

        np.testing.assert_allclose(label_ys, y_ends, atol=1e-6)

    def test_warns_when_labels_cannot_fit(self):
        """Geometry too tight to fit all labels emits a UserWarning naming a remedy.

        Without the warning, the bottom-clamp+upward-bump silently pushes the top label
        past ylim[1], producing labels outside the data area with no diagnostic for the
        user to act on. The warning surfaces the infeasible geometry and points the
        caller at remedies (switch legend_style, fewer series).
        """
        fig, ax = plt.subplots(figsize=(4, 0.6), dpi=100)
        ax.set_ylim(0, 100)
        ax.set_xlim(0, 10)
        fig.canvas.draw()

        candidates = self._make_candidates([10.0 + 5.0 * i for i in range(8)])

        with pytest.warns(UserWarning, match="cannot fit"):
            gu._resolve_end_of_line_label_ys(ax, candidates, font_pts=14.0)


class TestDrawEndOfLineLabels:
    """Tests for draw_end_of_line_labels.

    End-of-line series labels functionally identify series, so their typography
    must track the legend knobs, not the axis-label knobs.
    """

    def test_end_of_line_label_uses_legend_size(self):
        """The annotation fontsize must equal plot.font.legend_size."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.plot([1, 2, 3], [1, 2, 3], label="Revenue")
        fig.canvas.draw()

        gu.draw_end_of_line_labels(ax)

        labelled = [t for t in ax.texts if t.get_text() == "Revenue"]
        assert len(labelled) == 1
        assert labelled[0].get_fontsize() == get_option("plot.font.legend_size")


class TestRegressionLine:
    """Test class for the add_regression_line function."""

    # Constants to avoid magic numbers
    ORIGINAL_LINE_COUNT = 1
    EXPECTED_LINE_COUNT_AFTER_REGRESSION = 2
    STACKED_PATCH_COUNT = 8  # 4 bars + 4 stacked bars
    GROUPED_PATCH_COUNT = 8  # 4 bars + 4 grouped bars
    REGRESSION_LINE_POINTS = 2  # Regression line has 2 endpoints
    EXPECTED_LINEWIDTH = 2
    EXPECTED_ALPHA = 0.7
    OVERFLOW_X_MIN = 1000
    OVERFLOW_X_MAX = 2000

    def test_line_plot_with_numeric_data(self):
        """Test regression line with a standard line plot and numeric data."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 5, 7, 11])  # Not a perfect line to test regression
        ax.plot(x, y)

        gu.add_regression_line(ax, color="blue", show_equation=True, show_r2=True)

        # Check that a line was added (should now have 2 lines)
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_line_plot_with_datetime_data(self):
        """Regression on datetime x-axis fits in date2num space and slopes with the data."""
        _, ax = plt.subplots()
        dates = [
            datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 4, 1, tzinfo=datetime.timezone.utc),
        ]
        values = [10, 15, 14, 25]
        ax.plot(dates, values)

        gu.add_regression_line(ax, show_equation=True, show_r2=False)

        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

        regression_line = ax.get_lines()[1]
        line_x = regression_line.get_xdata()
        line_y = regression_line.get_ydata()
        date_nums = mdates.date2num(dates)

        # x-values are emitted in matplotlib date-number form, matching the input range.
        assert min(line_x) <= min(date_nums)
        assert max(line_x) >= max(date_nums)

        # Values rise from 10 to 25 over the period, so the fitted line must slope upward.
        assert line_y[-1] > line_y[0]

    def test_scatter_plot(self):
        """Test regression line with a scatter plot."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3.5, 4.5, 7.5, 10])
        ax.scatter(x, y)

        gu.add_regression_line(ax, color="green", linestyle="-.")

        # Check that a line was added to the scatter plot
        assert len(ax.get_lines()) == self.ORIGINAL_LINE_COUNT

    def test_large_numbers(self):
        """Test regression line with very large numbers (billions)."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.5, 3.2, 4.7, 7.1, 8.9]) * 1e9  # Values in billions
        ax.plot(x, y)

        gu.add_regression_line(ax, color="purple", show_equation=True, show_r2=True)

        # Check that a line was added
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_bar_plot(self):
        """Test regression line with a vertical bar chart."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 3, 5, 6])
        ax.bar(x, y)

        gu.add_regression_line(ax, color="orange", show_equation=True, show_r2=True)

        # Check that a regression line was added (bar plots start with 0 lines)
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_barh_plot(self):
        """Test regression line with a horizontal bar chart."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([2, 4, 3, 5, 6])
        ax.barh(y, x)

        gu.add_regression_line(ax, color="green", show_equation=True, show_r2=True)

        # Check that a regression line was added
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_single_data_point(self):
        """Test that regression line raises ValueError with a single data point."""
        _, ax = plt.subplots()
        ax.plot([10], [20])

        # Use pytest.raises to check for the expected exception
        with pytest.raises(ValueError) as excinfo:
            gu.add_regression_line(ax)

        # Check for appropriate error message
        assert "regression" in str(excinfo.value).lower()

    def test_bar_plot_negative_values(self):
        """Test regression line correctly handles negative bar values."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.bar(x, y)

        gu.add_regression_line(ax, color="red")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Get the regression line data
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify the line spans a reasonable range (uses axis limits, not exact data range)
        assert line_x[0] < min(x)  # Line starts before first bar
        assert line_x[1] > max(x)  # Line ends after last bar

        # Verify line handles negative values (should not be all zeros)
        assert not all(val == 0 for val in line_y)

    def test_barh_plot_negative_values(self):
        """Test regression line correctly handles negative horizontal bar values."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.barh(y, x)

        gu.add_regression_line(ax, color="purple")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Get the regression line data
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # For horizontal bars, verify the line spans the value range reasonably
        # Line should encompass the data range (may extend beyond due to axis limits)
        assert min(line_x) <= max(x)  # Line should reach at least the max value
        assert max(line_x) >= min(x)  # Line should reach at least the min value

        # Verify line handles negative values (should not be all zeros)
        assert not all(val == 0 for val in line_y)

    def test_bar_plot_stacked(self):
        """Test regression line with stacked bar chart uses correct data ordering."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4])
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([1, 2, 1, 3])

        # Create stacked bars
        ax.bar(x, y1, label="Series 1")
        ax.bar(x, y2, bottom=y1, label="Series 2")

        gu.add_regression_line(ax, color="blue")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both series (8 total: 4 + 4)
        assert len(ax.patches) == self.STACKED_PATCH_COUNT

        # Get the regression line to ensure it was calculated
        line = ax.get_lines()[0]
        line_x = line.get_xdata()

        # Verify line spans the x range of the bars
        assert min(line_x) <= min(x)
        assert max(line_x) >= max(x)

    def test_bar_plot_grouped(self):
        """Test regression line with grouped bar chart handles data ordering correctly."""
        _, ax = plt.subplots()

        # Create grouped bars with different x positions
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array([1.3, 2.3, 3.3, 4.3])  # Offset for grouping
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([3, 1, 2, 4])

        width = 0.3
        ax.bar(x1, y1, width, label="Group 1")
        ax.bar(x2, y2, width, label="Group 2")

        gu.add_regression_line(ax, color="orange")

        # Verify regression line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both groups (8 total: 4 + 4)
        assert len(ax.patches) == self.GROUPED_PATCH_COUNT

        # Get the regression line
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify line data exists and spans a reasonable range
        assert len(line_x) == self.REGRESSION_LINE_POINTS  # Regression line should have 2 points
        assert len(line_y) == self.REGRESSION_LINE_POINTS

        # Verify line spans across the grouped bars
        all_x_positions = np.concatenate([x1, x2])
        assert min(line_x) <= min(all_x_positions)
        assert max(line_x) >= max(all_x_positions)

    def test_unsupported_regression_type_raises_error(self):
        """Test that unsupported regression types raise ValueError."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise ValueError for unsupported type
        with pytest.raises(ValueError, match="Unsupported regression_type"):
            gu.add_regression_line(ax, regression_type="unsupported")

    # Algorithm Tests - Regression types with known data (parametrized to eliminate duplication)
    @pytest.mark.parametrize(
        ("regression_type", "x_data", "y_data", "description"),
        [
            ("linear", np.array([1, 2, 3, 4, 5]), lambda x: 2 * x + 1, "y = 2x + 1"),
            ("power", np.array([1, 2, 3, 4, 5]), lambda x: 2 * (x**1.5), "y = 2x^1.5"),
            ("logarithmic", np.array([1, 2, 3, 4, 5]), lambda x: 3 * np.log(x) + 1, "y = 3*ln(x) + 1"),
            ("exponential", np.array([0, 1, 2, 3, 4]), lambda x: 2 * np.exp(0.5 * x), "y = 2*e^(0.5x)"),
        ],
    )
    def test_regression_known_data(self, regression_type, x_data, y_data, description):
        """Test regression types with known relationships and validate line correctness."""
        # Generate perfect data based on known relationship
        y = y_data(x_data)

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x_data, y)

        # Apply regression
        result_ax = gu.add_regression_line(ax, regression_type=regression_type)

        # Verify basic functionality
        assert result_ax is ax
        assert len(ax.lines) == 1  # Regression line added

        # Validate that the regression line points are mathematically correct
        regression_line = ax.lines[0]
        line_x = regression_line.get_xdata()
        line_y = regression_line.get_ydata()

        # Check that the regression line produces values close to the expected relationship
        # Sample a few points from the regression line to verify correctness
        sample_indices = [0, len(line_x) // 2, -1]  # Start, middle, end
        for idx in sample_indices:
            x_val = line_x[idx]
            y_val = line_y[idx]
            # Calculate expected y value based on the known relationship
            expected_y = y_data(np.array([x_val]))[0]
            # Perfect data should yield near-exact fit (within floating-point precision)
            tolerance = max(abs(expected_y) * 1e-6, 1e-10)  # Relative 0.0001% with absolute floor
            assert abs(y_val - expected_y) < tolerance, (
                f"Regression line point ({x_val}, {y_val}) deviates too much from "
                f"expected ({x_val}, {expected_y}) for {description}"
            )

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([-1, 0, 1, 2, 3]), np.array([1, 2, 3, 4, 5])),
            (np.array([1, 2, 3]), np.array([1, 2, -3])),
        ],
        ids=["negative/zero x values", "negative y values"],
    )
    def test_power_regression_errors_on_nonpositive_values(self, x_data, y_data):
        """Test that power regression raises error when data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Power regression requires all x and y values to be positive"):
            gu.add_regression_line(ax, regression_type="power")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])),
            (np.array([-1, 0.1, 1]), np.array([1, 2, 3])),
        ],
        ids=["zero x values", "negative x values"],
    )
    def test_logarithmic_regression_errors_on_nonpositive_x_values(self, x_data, y_data):
        """Test that logarithmic regression raises error when x data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Logarithmic regression requires all x values to be positive"):
            gu.add_regression_line(ax, regression_type="logarithmic")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([1, 2, 3, 4, 5]), np.array([-1, 2, 3, 4, 5])),
            (np.array([1, 2, 3]), np.array([1, 0, 2])),
        ],
        ids=["negative y values", "zero y values"],
    )
    def test_exponential_regression_errors_on_nonpositive_y_values(self, x_data, y_data):
        """Test that exponential regression raises error when y data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Exponential regression requires all y values to be positive"):
            gu.add_regression_line(ax, regression_type="exponential")

    def test_exponential_regression_filters_overflow_values(self):
        """Test that exponential regression filters out infinite values from overflow without leaking numpy warnings."""
        # Use data with a steep growth coefficient so that extending x_max triggers overflow
        # With b ≈ 0.5, overflow (exp > 1e308) occurs around x ≈ 1420
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([1.65, 2.72, 4.48, 7.39, 12.18])  # Roughly y = e^(0.5*x)

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Set xlim far beyond overflow threshold to guarantee some points overflow
        ax.set_xlim(-1, self.OVERFLOW_X_MAX)

        # Ensure no numpy RuntimeWarning leaks to users during overflow
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            gu.add_regression_line(ax, regression_type="exponential")

        # Verify line was plotted with only finite values (some points should have been filtered)
        assert len(ax.lines) == 1
        regression_line = ax.lines[0]
        x_line = regression_line.get_xdata()
        y_data = regression_line.get_ydata()
        assert np.all(np.isfinite(y_data))
        # Verify that filtering actually occurred: line should not extend all the way to x_max
        assert max(x_line) < self.OVERFLOW_X_MAX, "Some points should have been filtered due to overflow"

    def test_exponential_regression_warns_and_returns_early_on_total_overflow(self):
        """Test that exponential regression warns and plots nothing when all values overflow."""
        # Exponential data: fit yields b ≈ ln(10) ≈ 2.3, so exp(2.3 * 1000) overflows float64
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Set xlim to an extreme range where all exp() values overflow to infinity
        ax.set_xlim(self.OVERFLOW_X_MIN, self.OVERFLOW_X_MAX)

        # Snapshot axes state before calling add_regression_line
        lines_before = len(ax.lines)
        texts_before = len(ax.texts)

        # Ensure no numpy RuntimeWarning leaks — only the expected UserWarning should be raised
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.warns(UserWarning, match="produced no finite values in the visible range"):
                result = gu.add_regression_line(ax, regression_type="exponential")

        # Verify no regression line or equation text was added
        assert len(ax.lines) == lines_before
        assert len(ax.texts) == texts_before
        # Verify the axes object is still returned for chaining
        assert result is ax

    @pytest.mark.parametrize(
        ("regression_type", "expected_pattern"),
        [
            ("linear", r"y = .+x [+-] .+"),
            ("power", r"y = .+x\^.+"),
            ("logarithmic", r"y = .+ln\(x\) [+-] .+"),
            ("exponential", r"y = .+e\^\(.+x\)"),
        ],
    )
    def test_equation_formatting_different_types(self, regression_type, expected_pattern):
        """Test that equation text uses the correct format for each regression type."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        gu.add_regression_line(ax, regression_type=regression_type, show_equation=True, show_r2=True)

        texts = ax.texts
        assert len(texts) == 1
        equation_text = texts[0].get_text()
        assert re.search(expected_pattern, equation_text), (
            f"Expected equation format matching '{expected_pattern}' for {regression_type}, got '{equation_text}'"
        )

    def test_regression_parameters_forwarding(self):
        """Test that regression parameters are forwarded correctly for new types."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with power regression and custom parameters
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should work with all parameters
        result_ax = gu.add_regression_line(
            ax,
            regression_type="power",
            color="blue",
            linestyle=":",
            text_position=0.8,
            show_equation=False,
            show_r2=True,
            linewidth=self.EXPECTED_LINEWIDTH,
            alpha=self.EXPECTED_ALPHA,
        )

        # Verify basic functionality
        assert result_ax is ax
        assert len(ax.lines) == 1

        # Verify line style parameters were forwarded
        line = ax.lines[0]
        assert line.get_color() == "blue"
        assert line.get_linestyle() == ":"
        assert line.get_linewidth() == self.EXPECTED_LINEWIDTH
        assert line.get_alpha() == self.EXPECTED_ALPHA

        # show_equation=False means no equation text, only R²
        assert len(ax.texts) == 1
        annotation = ax.texts[0].get_text()
        assert "R²" in annotation, f"R² text should be displayed, got '{annotation}'"
        assert "y =" not in annotation, f"Equation should not be displayed when show_equation=False, got '{annotation}'"

    def test_regression_annotation_uses_source_size(self):
        """Regression annotation pairs source_font family with source_size, not label_size."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        ax.scatter(x, y)

        gu.add_regression_line(ax, show_equation=True, show_r2=True)

        assert len(ax.texts) == 1
        annotation = ax.texts[0]
        assert annotation.get_fontsize() == get_option("plot.font.source_size")

    @pytest.mark.parametrize("regression_type", ["linear", "power", "logarithmic", "exponential"])
    def test_insufficient_data_points_all_types(self, regression_type):
        """Test that all regression types handle insufficient data points correctly."""
        # Single data point
        x = np.array([1])
        y = np.array([2])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for insufficient data
        with pytest.raises(ValueError, match=r"At least .* valid data points are required"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["linear", "power", "logarithmic", "exponential"])
    def test_zero_variance_x_values(self, regression_type):
        """Test that all regression types handle zero variance in x values correctly."""
        # All x values are identical
        x = np.array([5, 5, 5, 5])
        y = np.array([1, 2, 3, 4])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for zero variance in x
        with pytest.raises(ValueError, match="all x values are identical"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["power", "exponential"])
    def test_zero_variance_y_values_nonlinear(self, regression_type):
        """Test that regression types with log(y) transform raise on zero variance y."""
        # All y values are identical — log transform makes regression meaningless
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        with pytest.raises(ValueError, match="all y values are identical"):
            gu.add_regression_line(ax, regression_type=regression_type)

    @pytest.mark.parametrize("regression_type", ["linear", "logarithmic"])
    def test_zero_variance_y_values_returns_flat_line(self, regression_type):
        """Test that linear and logarithmic regression handle constant y gracefully."""
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        result = gu.add_regression_line(ax, regression_type=regression_type)
        # Should succeed and return the axes (flat line is valid)
        assert result is ax
        # A regression line should have been added (scatter + regression = 2 lines)
        assert len(ax.get_lines()) == 1

    def test_r_squared_original_space_accuracy(self):
        """Test that R² is calculated in original data space, not transformed space."""
        # Create perfect power law data: y = 2 * x^1.5
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * (x_data**1.5)  # Perfect power law

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power regression
        gu.add_regression_line(ax, regression_type="power", show_r2=True)

        # With perfect power law data, R² should be very close to 1.0
        r2_value = _extract_r2_from_plot(ax)
        high_r_squared_threshold = 0.99
        assert r2_value > high_r_squared_threshold, f"R² should be close to 1.0 for perfect data, got {r2_value}"

    def test_r_squared_comparison_transformed_vs_original(self):
        """Test that R² in original space differs from transformed space for non-linear regression."""
        # Create data with a non-linear relationship that has deliberate outliers
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Quadratic pattern with outliers: y = x^2, but with outliers at the end
        y_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 200.0, 400.0])

        # Calculate R² in transformed space (log-log for power regression)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        _, _, r_value_transformed, _, _ = stats.linregress(log_x, log_y)
        r_squared_transformed = r_value_transformed**2

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power regression - this should calculate R² in original space
        gu.add_regression_line(ax, regression_type="power", show_r2=True)

        # Extract R² from annotation (this is R² in original space)
        r2_value_original = _extract_r2_from_plot(ax)

        # Verify that R² values differ between spaces
        # With outliers, R² in original space should be significantly lower than in transformed space
        # R² in original space can be negative when the model performs worse than the mean
        assert r2_value_original <= 1.0, f"R² should be at most 1.0, got {r2_value_original}"
        assert 0.0 <= r_squared_transformed <= 1.0, (
            f"R² transformed should be between 0 and 1, got {r_squared_transformed}"
        )

        # The key test: original space R² should be lower because outliers
        # have a larger impact in original space than in log-transformed space
        assert r2_value_original < r_squared_transformed, (
            f"R² in original space ({r2_value_original:.4f}) should be lower than "
            f"transformed space ({r_squared_transformed:.4f}) due to outlier sensitivity"
        )

        # Also verify the difference is meaningful, not just floating-point noise
        difference_threshold = 0.01
        assert abs(r2_value_original - r_squared_transformed) > difference_threshold, (
            f"R² should meaningfully differ between original ({r2_value_original:.4f}) and "
            f"transformed ({r_squared_transformed:.4f}) space"
        )


class TestVisualRegression:
    """Visual regression tests to ensure refactored code produces identical output."""

    WHITE_RGBA = (1.0, 1.0, 1.0, 1.0)

    def test_visual_regression_standard_graph_styles_basic(self):
        """Ensure standard_graph_styles produces consistent visual output."""
        _, ax = plt.subplots(figsize=(8, 6))
        rng = np.random.default_rng(42)
        x = np.linspace(0, 10, 50)
        y = np.sin(x) + 0.1 * rng.standard_normal(50)
        ax.plot(x, y, label="Sin Wave")

        sh.standard_graph_styles(
            ax,
            title="Test Graph Title",
            x_label="X Axis Label",
            y_label="Y Axis Label",
            legend_title="Legend Title",
        )

        title_texts = [t for t in ax.figure.texts if t.get_text() == "Test Graph Title"]
        assert len(title_texts) == 1
        assert ax.get_xlabel() == "X Axis Label"
        assert ax.get_ylabel() == "Y Axis Label"
        assert ax.get_facecolor() == self.WHITE_RGBA
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()

    @pytest.mark.parametrize(
        ("plot_type", "data_generator"),
        [
            (
                "line",
                lambda: (
                    np.linspace(0, 10, 20),
                    np.sin(np.linspace(0, 10, 20)) + 0.1 * np.random.default_rng(42).standard_normal(20),
                ),
            ),
            (
                "scatter",
                lambda: (
                    np.linspace(0, 10, 20),
                    np.sin(np.linspace(0, 10, 20)) + 0.1 * np.random.default_rng(42).standard_normal(20),
                ),
            ),
            ("bar", lambda: (["Bakery", "Dairy", "Produce", "Snacks", "Beverages"], [23, 45, 56, 78, 32])),
        ],
    )
    def test_visual_regression_all_plot_types(self, plot_type, data_generator):
        """Test visual consistency across different plot types."""
        _, ax = plt.subplots(figsize=(8, 6))

        x, y = data_generator()

        plot_methods = {
            "line": lambda ax, x, y: ax.plot(x, y),
            "scatter": lambda ax, x, y: ax.scatter(x, y),
            "bar": lambda ax, x, y: ax.bar(x, y),
        }
        plot_methods[plot_type](ax, x, y)

        sh.standard_graph_styles(
            ax,
            title=f"Test {plot_type.title()} Plot",
            x_label="X Values",
            y_label="Y Values",
        )

        assert ax.get_facecolor() == self.WHITE_RGBA
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
