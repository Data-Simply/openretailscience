"""Tests for the trend module in the style package."""

import datetime
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pytest
from scipy import stats

from openretailscience.options import get_option
from openretailscience.plots.styles import trend


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


class TestTrendLine:
    """Test class for the add_trend_line function."""

    # Constants to avoid magic numbers
    ORIGINAL_LINE_COUNT = 1
    EXPECTED_LINE_COUNT_AFTER_REGRESSION = 2
    STACKED_PATCH_COUNT = 8  # 4 bars + 4 stacked bars
    GROUPED_PATCH_COUNT = 8  # 4 bars + 4 grouped bars
    TREND_LINE_POINTS = 2  # Trend line has 2 endpoints
    EXPECTED_LINEWIDTH = 2
    EXPECTED_ALPHA = 0.7
    OVERFLOW_X_MIN = 1000
    OVERFLOW_X_MAX = 2000

    def test_line_plot_with_numeric_data(self):
        """Test trend line with a standard line plot and numeric data."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3, 5, 7, 11])  # Not a perfect line to test trend
        ax.plot(x, y)

        trend.add_trend_line(ax, color="blue", show_equation=True, show_r2=True)

        # Check that a line was added (should now have 2 lines)
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_line_plot_with_datetime_data(self):
        """Trend on datetime x-axis fits in date2num space and slopes with the data."""
        _, ax = plt.subplots()
        dates = [
            datetime.datetime(2023, 1, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 2, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 3, 1, tzinfo=datetime.timezone.utc),
            datetime.datetime(2023, 4, 1, tzinfo=datetime.timezone.utc),
        ]
        values = [10, 15, 14, 25]
        ax.plot(dates, values)

        trend.add_trend_line(ax, show_equation=True, show_r2=False)

        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

        trend_line = ax.get_lines()[1]
        line_x = trend_line.get_xdata()
        line_y = trend_line.get_ydata()
        date_nums = mdates.date2num(dates)

        # x-values are emitted in matplotlib date-number form, matching the input range.
        assert min(line_x) <= min(date_nums)
        assert max(line_x) >= max(date_nums)

        # Values rise from 10 to 25 over the period, so the fitted line must slope upward.
        assert line_y[-1] > line_y[0]

    def test_scatter_plot(self):
        """Test trend line with a scatter plot."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 3.5, 4.5, 7.5, 10])
        ax.scatter(x, y)

        trend.add_trend_line(ax, color="green", linestyle="-.")

        # Check that a line was added to the scatter plot
        assert len(ax.get_lines()) == self.ORIGINAL_LINE_COUNT

    def test_large_numbers(self):
        """Test trend line with very large numbers (billions)."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2.5, 3.2, 4.7, 7.1, 8.9]) * 1e9  # Values in billions
        ax.plot(x, y)

        trend.add_trend_line(ax, color="purple", show_equation=True, show_r2=True)

        # Check that a line was added
        assert len(ax.get_lines()) == self.EXPECTED_LINE_COUNT_AFTER_REGRESSION

    def test_bar_plot(self):
        """Test trend line with a vertical bar chart."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 3, 5, 6])
        ax.bar(x, y)

        trend.add_trend_line(ax, color="orange", show_equation=True, show_r2=True)

        # Check that a trend line was added (bar plots start with 0 lines)
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_barh_plot(self):
        """Test trend line with a horizontal bar chart."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([2, 4, 3, 5, 6])
        ax.barh(y, x)

        trend.add_trend_line(ax, color="green", show_equation=True, show_r2=True)

        # Check that a trend line was added
        assert len(ax.get_lines()) == 1
        # Check that we still have the bar patches
        assert len(ax.patches) == len(x)

    def test_single_data_point(self):
        """Test that trend line raises ValueError with a single data point."""
        _, ax = plt.subplots()
        ax.plot([10], [20])

        # Use pytest.raises to check for the expected exception
        with pytest.raises(ValueError) as excinfo:
            trend.add_trend_line(ax)

        # Check for appropriate error message
        assert "trend" in str(excinfo.value).lower()

    def test_bar_plot_negative_values(self):
        """Test trend line correctly handles negative bar values."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.bar(x, y)

        trend.add_trend_line(ax, color="red")

        # Verify trend line was added
        assert len(ax.get_lines()) == 1

        # Get the trend line data
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify the line spans a reasonable range (uses axis limits, not exact data range)
        assert line_x[0] < min(x)  # Line starts before first bar
        assert line_x[1] > max(x)  # Line ends after last bar

        # Verify line handles negative values (should not be all zeros)
        assert not all(val == 0 for val in line_y)

    def test_barh_plot_negative_values(self):
        """Test trend line correctly handles negative horizontal bar values."""
        _, ax = plt.subplots()
        y = np.array([1, 2, 3, 4, 5])
        x = np.array([-2, 4, -1, 3, -5])  # Mix of positive and negative values
        ax.barh(y, x)

        trend.add_trend_line(ax, color="purple")

        # Verify trend line was added
        assert len(ax.get_lines()) == 1

        # Get the trend line data
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
        """Test trend line with stacked bar chart uses correct data ordering."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4])
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([1, 2, 1, 3])

        # Create stacked bars
        ax.bar(x, y1, label="Series 1")
        ax.bar(x, y2, bottom=y1, label="Series 2")

        trend.add_trend_line(ax, color="blue")

        # Verify trend line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both series (8 total: 4 + 4)
        assert len(ax.patches) == self.STACKED_PATCH_COUNT

        # Get the trend line to ensure it was calculated
        line = ax.get_lines()[0]
        line_x = line.get_xdata()

        # Verify line spans the x range of the bars
        assert min(line_x) <= min(x)
        assert max(line_x) >= max(x)

    def test_bar_plot_grouped(self):
        """Test trend line with grouped bar chart handles data ordering correctly."""
        _, ax = plt.subplots()

        # Create grouped bars with different x positions
        x1 = np.array([1, 2, 3, 4])
        x2 = np.array([1.3, 2.3, 3.3, 4.3])  # Offset for grouping
        y1 = np.array([2, 3, 4, 1])
        y2 = np.array([3, 1, 2, 4])

        width = 0.3
        ax.bar(x1, y1, width, label="Group 1")
        ax.bar(x2, y2, width, label="Group 2")

        trend.add_trend_line(ax, color="orange")

        # Verify trend line was added
        assert len(ax.get_lines()) == 1

        # Verify we have patches for both groups (8 total: 4 + 4)
        assert len(ax.patches) == self.GROUPED_PATCH_COUNT

        # Get the trend line
        line = ax.get_lines()[0]
        line_x = line.get_xdata()
        line_y = line.get_ydata()

        # Verify line data exists and spans a reasonable range
        assert len(line_x) == self.TREND_LINE_POINTS  # Trend line should have 2 points
        assert len(line_y) == self.TREND_LINE_POINTS

        # Verify line spans across the grouped bars
        all_x_positions = np.concatenate([x1, x2])
        assert min(line_x) <= min(all_x_positions)
        assert max(line_x) >= max(all_x_positions)

    def test_unsupported_trend_type_raises_error(self):
        """Test that unsupported trend types raise ValueError."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise ValueError for unsupported type
        with pytest.raises(ValueError, match="Unsupported trend_type"):
            trend.add_trend_line(ax, trend_type="unsupported")

    # Algorithm Tests - Trend types with known data (parametrized to eliminate duplication)
    @pytest.mark.parametrize(
        ("trend_type", "x_data", "y_data", "description"),
        [
            ("linear", np.array([1, 2, 3, 4, 5]), lambda x: 2 * x + 1, "y = 2x + 1"),
            ("power", np.array([1, 2, 3, 4, 5]), lambda x: 2 * (x**1.5), "y = 2x^1.5"),
            ("logarithmic", np.array([1, 2, 3, 4, 5]), lambda x: 3 * np.log(x) + 1, "y = 3*ln(x) + 1"),
            ("exponential", np.array([0, 1, 2, 3, 4]), lambda x: 2 * np.exp(0.5 * x), "y = 2*e^(0.5x)"),
        ],
    )
    def test_trend_known_data(self, trend_type, x_data, y_data, description):
        """Test trend types with known relationships and validate line correctness."""
        # Generate perfect data based on known relationship
        y = y_data(x_data)

        # Create plot
        _, ax = plt.subplots()
        ax.scatter(x_data, y)

        # Apply trend
        result_ax = trend.add_trend_line(ax, trend_type=trend_type)

        # Verify basic functionality
        assert result_ax is ax
        assert len(ax.lines) == 1  # Trend line added

        # Validate that the trend line points are mathematically correct
        trend_line = ax.lines[0]
        line_x = trend_line.get_xdata()
        line_y = trend_line.get_ydata()

        # Check that the trend line produces values close to the expected relationship
        # Sample a few points from the trend line to verify correctness
        sample_indices = [0, len(line_x) // 2, -1]  # Start, middle, end
        for idx in sample_indices:
            x_val = line_x[idx]
            y_val = line_y[idx]
            # Calculate expected y value based on the known relationship
            expected_y = y_data(np.array([x_val]))[0]
            # Perfect data should yield near-exact fit (within floating-point precision)
            tolerance = max(abs(expected_y) * 1e-6, 1e-10)  # Relative 0.0001% with absolute floor
            assert abs(y_val - expected_y) < tolerance, (
                f"Trend line point ({x_val}, {y_val}) deviates too much from "
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
    def test_power_trend_errors_on_nonpositive_values(self, x_data, y_data):
        """Test that power trend raises error when data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Power trend requires all x and y values to be positive"):
            trend.add_trend_line(ax, trend_type="power")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([0, 1, 2, 3, 4]), np.array([1, 2, 3, 4, 5])),
            (np.array([-1, 0.1, 1]), np.array([1, 2, 3])),
        ],
        ids=["zero x values", "negative x values"],
    )
    def test_logarithmic_trend_errors_on_nonpositive_x_values(self, x_data, y_data):
        """Test that logarithmic trend raises error when x data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Logarithmic trend requires all x values to be positive"):
            trend.add_trend_line(ax, trend_type="logarithmic")

    @pytest.mark.parametrize(
        ("x_data", "y_data"),
        [
            (np.array([1, 2, 3, 4, 5]), np.array([-1, 2, 3, 4, 5])),
            (np.array([1, 2, 3]), np.array([1, 0, 2])),
        ],
        ids=["negative y values", "zero y values"],
    )
    def test_exponential_trend_errors_on_nonpositive_y_values(self, x_data, y_data):
        """Test that exponential trend raises error when y data contains non-positive values."""
        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        with pytest.raises(ValueError, match="Exponential trend requires all y values to be positive"):
            trend.add_trend_line(ax, trend_type="exponential")

    def test_exponential_trend_filters_overflow_values(self):
        """Test that exponential trend filters out infinite values from overflow without leaking numpy warnings."""
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
            trend.add_trend_line(ax, trend_type="exponential")

        # Verify line was plotted with only finite values (some points should have been filtered)
        assert len(ax.lines) == 1
        trend_line = ax.lines[0]
        x_line = trend_line.get_xdata()
        y_data = trend_line.get_ydata()
        assert np.all(np.isfinite(y_data))
        # Verify that filtering actually occurred: line should not extend all the way to x_max
        assert max(x_line) < self.OVERFLOW_X_MAX, "Some points should have been filtered due to overflow"

    def test_exponential_trend_warns_and_returns_early_on_total_overflow(self):
        """Test that exponential trend warns and plots nothing when all values overflow."""
        # Exponential data: fit yields b ≈ ln(10) ≈ 2.3, so exp(2.3 * 1000) overflows float64
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([10.0, 100.0, 1000.0, 10000.0, 100000.0])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Set xlim to an extreme range where all exp() values overflow to infinity
        ax.set_xlim(self.OVERFLOW_X_MIN, self.OVERFLOW_X_MAX)

        # Snapshot axes state before calling add_trend_line
        lines_before = len(ax.lines)
        texts_before = len(ax.texts)

        # Ensure no numpy RuntimeWarning leaks — only the expected UserWarning should be raised
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            with pytest.warns(UserWarning, match="produced no finite values in the visible range"):
                result = trend.add_trend_line(ax, trend_type="exponential")

        # Verify no trend line or equation text was added
        assert len(ax.lines) == lines_before
        assert len(ax.texts) == texts_before
        # Verify the axes object is still returned for chaining
        assert result is ax

    @pytest.mark.parametrize(
        ("trend_type", "expected_pattern"),
        [
            ("linear", r"y = .+x [+-] .+"),
            ("power", r"y = .+x\^.+"),
            ("logarithmic", r"y = .+ln\(x\) [+-] .+"),
            ("exponential", r"y = .+e\^\(.+x\)"),
        ],
    )
    def test_equation_formatting_different_types(self, trend_type, expected_pattern):
        """Test that equation text uses the correct format for each trend type."""
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        trend.add_trend_line(ax, trend_type=trend_type, show_equation=True, show_r2=True)

        texts = ax.texts
        assert len(texts) == 1
        equation_text = texts[0].get_text()
        assert re.search(expected_pattern, equation_text), (
            f"Expected equation format matching '{expected_pattern}' for {trend_type}, got '{equation_text}'"
        )

    def test_trend_parameters_forwarding(self):
        """Test that trend parameters are forwarded correctly for new types."""
        # Create test data
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])

        # Test with power trend and custom parameters
        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should work with all parameters
        result_ax = trend.add_trend_line(
            ax,
            trend_type="power",
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

    def test_trend_annotation_uses_source_size(self):
        """Trend annotation pairs source_font family with source_size, not label_size."""
        _, ax = plt.subplots()
        x = np.array([1, 2, 3, 4, 5])
        y = np.array([2, 4, 6, 8, 10])
        ax.scatter(x, y)

        trend.add_trend_line(ax, show_equation=True, show_r2=True)

        assert len(ax.texts) == 1
        annotation = ax.texts[0]
        assert annotation.get_fontsize() == get_option("plot.font.source_size")

    @pytest.mark.parametrize("trend_type", ["linear", "power", "logarithmic", "exponential"])
    def test_insufficient_data_points_all_types(self, trend_type):
        """Test that all trend types handle insufficient data points correctly."""
        # Single data point
        x = np.array([1])
        y = np.array([2])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for insufficient data
        with pytest.raises(ValueError, match=r"At least .* valid data points are required"):
            trend.add_trend_line(ax, trend_type=trend_type)

    @pytest.mark.parametrize("trend_type", ["linear", "power", "logarithmic", "exponential"])
    def test_zero_variance_x_values(self, trend_type):
        """Test that all trend types handle zero variance in x values correctly."""
        # All x values are identical
        x = np.array([5, 5, 5, 5])
        y = np.array([1, 2, 3, 4])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        # Should raise error for zero variance in x
        with pytest.raises(ValueError, match="all x values are identical"):
            trend.add_trend_line(ax, trend_type=trend_type)

    @pytest.mark.parametrize("trend_type", ["power", "exponential"])
    def test_zero_variance_y_values_nonlinear(self, trend_type):
        """Test that trend types with log(y) transform raise on zero variance y."""
        # All y values are identical — log transform makes trend meaningless
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        with pytest.raises(ValueError, match="all y values are identical"):
            trend.add_trend_line(ax, trend_type=trend_type)

    @pytest.mark.parametrize("trend_type", ["linear", "logarithmic"])
    def test_zero_variance_y_values_returns_flat_line(self, trend_type):
        """Test that linear and logarithmic trend handle constant y gracefully."""
        x = np.array([1, 2, 3, 4])
        y = np.array([5, 5, 5, 5])

        _, ax = plt.subplots()
        ax.scatter(x, y)

        result = trend.add_trend_line(ax, trend_type=trend_type)
        # Should succeed and return the axes (flat line is valid)
        assert result is ax
        # A trend line should have been added (scatter + trend = 2 lines)
        assert len(ax.get_lines()) == 1

    def test_r_squared_original_space_accuracy(self):
        """Test that R² is calculated in original data space, not transformed space."""
        # Create perfect power law data: y = 2 * x^1.5
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = 2.0 * (x_data**1.5)  # Perfect power law

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power trend
        trend.add_trend_line(ax, trend_type="power", show_r2=True)

        # With perfect power law data, R² should be very close to 1.0
        r2_value = _extract_r2_from_plot(ax)
        high_r_squared_threshold = 0.99
        assert r2_value > high_r_squared_threshold, f"R² should be close to 1.0 for perfect data, got {r2_value}"

    def test_r_squared_comparison_transformed_vs_original(self):
        """Test that R² in original space differs from transformed space for non-linear trend."""
        # Create data with a non-linear relationship that has deliberate outliers
        x_data = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        # Quadratic pattern with outliers: y = x^2, but with outliers at the end
        y_data = np.array([1.0, 4.0, 9.0, 16.0, 25.0, 36.0, 200.0, 400.0])

        # Calculate R² in transformed space (log-log for power trend)
        log_x = np.log(x_data)
        log_y = np.log(y_data)
        _, _, r_value_transformed, _, _ = stats.linregress(log_x, log_y)
        r_squared_transformed = r_value_transformed**2

        _, ax = plt.subplots()
        ax.scatter(x_data, y_data)

        # Apply power trend - this should calculate R² in original space
        trend.add_trend_line(ax, trend_type="power", show_r2=True)

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
