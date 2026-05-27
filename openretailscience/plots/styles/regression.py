"""Regression line fitting and plotting helpers for matplotlib axes."""

import warnings
from datetime import datetime
from typing import Any, Literal

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from scipy import stats

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties

_MIN_LINE_POINTS = 50
_MAX_LINE_POINTS = 500
_DATA_SIZE_MULTIPLIER = 3
_POSITIVE_X_FLOOR = 1e-6
_VARIANCE_THRESHOLD = 1e-10
_TEXT_X_PADDING = 0.05  # 5% from left

# Map regression_type -> (check_x_positive, check_y_positive)
_POSITIVITY_REQUIREMENTS: dict[str, tuple[bool, bool]] = {
    "power": (True, True),
    "logarithmic": (True, False),
    "exponential": (False, True),
}


def _calculate_r_squared_original_space(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate R² in original data space.

    Args:
        y_actual (np.ndarray): Actual y values.
        y_predicted (np.ndarray): Predicted y values from regression model.

    Returns:
        float: R² value calculated in original data space.
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)  # Total sum of squares

    # Handle edge case where all y values are (nearly) identical
    if ss_tot < _VARIANCE_THRESHOLD:
        return 1.0 if ss_res < _VARIANCE_THRESHOLD else 0.0

    return 1 - (ss_res / ss_tot)


def _perform_regression_calculation(
    regression_type: str,
    x_filtered: np.ndarray,
    y_filtered: np.ndarray,
) -> tuple[float, float, float]:
    """Perform regression calculation and return coefficients and R² in original data space.

    Args:
        regression_type (str): Type of regression to perform.
        x_filtered (np.ndarray): Filtered x data.
        y_filtered (np.ndarray): Filtered y data.

    Returns:
        tuple[float, float, float]: param1, param2, r_squared (calculated in original data space)
    """
    if regression_type == "linear":
        # Linear regression: y = mx + b
        slope, intercept, r_value, _, _ = stats.linregress(x_filtered, y_filtered)
        return slope, intercept, r_value**2

    if regression_type == "power":
        # Power law regression: y = ax^b → log(y) = log(a) + b*log(x)
        log_x = np.log(x_filtered)
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = stats.linregress(log_x, log_y)
        a = np.exp(intercept)  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * (x_filtered**b)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if regression_type == "logarithmic":
        # Logarithmic regression: y = a*ln(x) + b
        log_x = np.log(x_filtered)
        slope, intercept, _, _, _ = stats.linregress(log_x, y_filtered)
        a = slope  # a = slope, b = intercept
        b = intercept

        # Calculate R² in original data space
        y_predicted = a * np.log(x_filtered) + b
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if regression_type == "exponential":
        # Exponential regression: y = ae^(bx) → ln(y) = ln(a) + bx
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = stats.linregress(x_filtered, log_y)
        a = np.exp(intercept)  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * np.exp(b * x_filtered)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    msg = f"Unsupported regression type: {regression_type}"
    raise ValueError(msg)


def _generate_regression_line(
    regression_type: str,
    param1: float,
    param2: float,
    x_min: float,
    x_max: float,
    data_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate regression line points for plotting with adaptive point calculation.

    Args:
        regression_type (str): Type of regression.
        param1 (float): First parameter (slope/a coefficient).
        param2 (float): Second parameter (intercept/b coefficient).
        x_min (float): Minimum x value for line.
        x_max (float): Maximum x value for line.
        data_size (int): Number of original data points for adaptive calculation.

    Returns:
        tuple[np.ndarray, np.ndarray]: x_line, y_line arrays for plotting.
    """
    if regression_type == "linear":
        # Linear: use endpoints for efficiency
        x_line = np.array([x_min, x_max])
        y_line = param1 * x_line + param2
        return x_line, y_line

    # For non-linear types, use adaptive point calculation
    # Base points on data size but ensure smooth curves for complex functions
    adaptive_points = max(data_size * _DATA_SIZE_MULTIPLIER, _MIN_LINE_POINTS)
    num_points = min(adaptive_points, _MAX_LINE_POINTS)

    # Derive from the single source of truth
    requirements = _POSITIVITY_REQUIREMENTS.get(regression_type)
    requires_positive_x = requirements is not None and requirements[0]
    x_start = max(x_min, _POSITIVE_X_FLOOR) if requires_positive_x else x_min
    x_line = np.linspace(x_start, x_max, num_points)

    if regression_type == "power":
        # Suppress overflow warning — large exponents can overflow to inf, which is filtered below
        with np.errstate(over="ignore"):
            y_line = param1 * (x_line**param2)
    elif regression_type == "logarithmic":
        y_line = param1 * np.log(x_line) + param2
    elif regression_type == "exponential":
        # Suppress overflow warning — large x values can overflow exp() to inf, which is filtered below
        with np.errstate(over="ignore"):
            y_line = param1 * np.exp(param2 * x_line)
    else:
        msg = f"Unsupported regression type for line generation: {regression_type}"
        raise ValueError(msg)

    # Filter out infinite/NaN values for types susceptible to overflow
    if regression_type in ("power", "exponential"):
        finite_mask = np.isfinite(y_line)
        if not np.all(finite_mask):
            x_line = x_line[finite_mask]
            y_line = y_line[finite_mask]

    return x_line, y_line


def _add_equation_text(
    ax: Axes,
    param1: float,
    param2: float,
    r_squared: float,
    color: str,
    text_position: float,
    show_equation: bool,
    show_r2: bool,
    regression_type: str = "linear",
) -> None:
    """Add equation and R² text to the plot.

    Args:
        ax (Axes): The matplotlib axes object.
        param1 (float): First regression parameter (slope/a coefficient).
        param2 (float): Second regression parameter (intercept/b coefficient).
        r_squared (float): The R² value of the regression.
        color (str): The color of the text.
        text_position (float): The relative y-position of the text.
        show_equation (bool): Whether to display the equation.
        show_r2 (bool): Whether to display the R² value.
        regression_type (str): The type of regression for equation formatting.
    """
    if not (show_equation or show_r2):
        return

    style = PlotStyleHelper()

    equation_parts = []

    if show_equation:
        if regression_type == "linear":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.4g}x {sign} {abs(param2):.4g}"
        elif regression_type == "power":
            exp_str = f"({param2:.4g})" if param2 < 0 else f"{param2:.4g}"
            equation = f"y = {param1:.4g}x^{exp_str}"
        elif regression_type == "logarithmic":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.4g}ln(x) {sign} {abs(param2):.4g}"
        elif regression_type == "exponential":
            equation = f"y = {param1:.4g}e^({param2:.4g}x)"
        else:
            msg = f"Unsupported regression type for equation formatting: {regression_type}"
            raise ValueError(msg)

        equation_parts.append(equation)

    if show_r2:
        r2_text = f"R² = {r_squared:.4g}"
        equation_parts.append(r2_text)

    text = "\n".join(equation_parts)

    # Calculate text position (relative to axis bounds)
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()
    text_x = x_min + _TEXT_X_PADDING * (x_max - x_min)
    text_y = y_min + text_position * (y_max - y_min)

    ax.text(
        text_x,
        text_y,
        text,
        color=color,
        fontsize=style.source_size,
        fontproperties=get_font_properties(style.source_font),
        bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
    )


def _extract_plot_data(ax: Axes) -> tuple[np.ndarray, np.ndarray]:
    """Extract x and y data from a matplotlib plot (line, scatter, or bar).

    Supports multiple plot types:
    - Line plots: Extracts data from line objects
    - Bar charts: Extracts center positions and heights/widths, with automatic orientation detection
    - Scatter plots: Extracts data from collection offsets

    Args:
        ax (Axes): The matplotlib axes object containing the plot.

    Returns:
        tuple[np.ndarray, np.ndarray]: The x and y data arrays.

    Raises:
        ValueError: If no plot data can be extracted.
    """
    # Try to get data from lines first (line plots)
    lines = [line for line in ax.get_lines() if line.get_visible()]

    if len(lines) > 0:
        x_data = lines[0].get_xdata()
        y_data = lines[0].get_ydata()
    # Check for bar charts (patches)
    elif hasattr(ax, "patches") and len(ax.patches) > 0:
        # Detect bar orientation using BarContainer (stable API). Default covers
        # the case where patches were added without a container via ax.add_patch.
        is_vertical = ax.containers[0].orientation == "vertical" if len(ax.containers) > 0 else True

        if is_vertical:
            # Vertical bars: x is center position, y is height
            bar_data = [(patch.get_x() + patch.get_width() / 2, patch.get_height()) for patch in ax.patches]
        else:
            # Horizontal bars: x is width, y is center position
            bar_data = [(patch.get_width(), patch.get_y() + patch.get_height() / 2) for patch in ax.patches]

        # Sort by x-coordinate to ensure consistency with grouped/stacked bar charts
        bar_data.sort(key=lambda point: point[0])
        x_data, y_data = np.array(bar_data).T
    # If no lines or bars, check for scatter plots (or other collections)
    elif hasattr(ax, "collections") and len(ax.collections) > 0:
        # Extract data from the first collection (e.g., scatter plot)
        collection = ax.collections[0]
        # Get the offsets which contain the x,y coordinates
        if hasattr(collection, "get_offsets") and callable(collection.get_offsets):
            offset_data = collection.get_offsets()
            if len(offset_data) > 0:
                x_data = offset_data[:, 0]
                y_data = offset_data[:, 1]
            else:
                raise ValueError("No data points found in the collection")
        else:
            raise ValueError("Cannot extract data from this type of collection")
    else:
        raise ValueError("No visible lines or collections found in the plot")

    return x_data, y_data


def _prepare_numeric_data(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert plot data to numeric arrays suitable for regression analysis.

    Args:
        x_data (np.ndarray): The raw x-axis data from the plot.
        y_data (np.ndarray): The raw y-axis data from the plot.

    Returns:
        tuple[np.ndarray, np.ndarray]: The numeric x and y data.

    Raises:
        ValueError: If data cannot be converted to numeric format or has insufficient valid points.
    """
    # Simple fallback indices in case we can't process the data
    x_indices = np.arange(len(x_data))

    # Check if x_data contains datetime objects
    is_datetime = False
    try:
        # Try to find a non-null value to check its type
        for val in x_data:
            if val is not None:
                is_datetime = isinstance(val, datetime | pd.Timestamp)
                break
    except (TypeError, IndexError):
        pass

    try:
        # Handle datetime or numeric data appropriately
        x_numeric = date2num(x_data) if is_datetime else np.array(x_data, dtype=float)
    except (TypeError, ValueError):
        # Fallback to simple indices if conversion fails
        x_numeric = x_indices

    try:
        y_numeric = np.array(y_data, dtype=float)
    except (TypeError, ValueError) as err:
        raise ValueError("Cannot convert y-axis values to numeric format for regression") from err

    # Create mask to filter out NaN values
    valid_mask = ~np.isnan(x_numeric) & ~np.isnan(y_numeric)
    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points for regression")

    # Check that we have enough valid data points for regression
    min_points_for_regression = 2
    if np.sum(valid_mask) < min_points_for_regression:
        error_msg = f"At least {min_points_for_regression} valid data points are required for regression analysis"
        raise ValueError(error_msg)

    x_filtered = x_numeric[valid_mask]
    y_filtered = y_numeric[valid_mask]

    # Check for zero variance in x — invalid for all regression types
    if np.var(x_filtered) < _VARIANCE_THRESHOLD:
        raise ValueError("Cannot perform regression: all x values are identical (zero variance)")

    return x_filtered, y_filtered


def _validate_regression_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    regression_type: str,
) -> None:
    """Validate data for specific regression types.

    Args:
        x_data (np.ndarray): The x-axis data.
        y_data (np.ndarray): The y-axis data.
        regression_type (str): The regression type being used.

    Raises:
        ValueError: If data contains values incompatible with the regression type.
    """
    requirements = _POSITIVITY_REQUIREMENTS.get(regression_type)
    if requirements is None:
        # Linear and other types use all data without positivity checks
        return

    check_x, check_y = requirements
    axes_desc = " and ".join(name for name, required in [("x", check_x), ("y", check_y)] if required)
    error_parts = []

    if check_x:
        count = int(np.sum(x_data <= 0))
        if count > 0:
            error_parts.append(f"{count} non-positive x value(s)")

    if check_y:
        count = int(np.sum(y_data <= 0))
        if count > 0:
            error_parts.append(f"{count} non-positive y value(s)")

    if len(error_parts) > 0:
        error_msg = (
            f"{regression_type.capitalize()} regression requires all {axes_desc} values to be positive. "
            f"Found {' and '.join(error_parts)}. "
            f"Please remove or transform these values before applying {regression_type} regression."
        )
        raise ValueError(error_msg)

    # Zero variance in y produces identical log(y) values, making regression meaningless
    if check_y and np.var(y_data) < _VARIANCE_THRESHOLD:
        raise ValueError("Cannot perform regression: all y values are identical (zero variance)")


def add_regression_line(
    ax: Axes,
    regression_type: Literal["linear", "power", "logarithmic", "exponential"] = "linear",
    color: str = "red",
    linestyle: str = "--",
    text_position: float = 0.6,
    show_equation: bool = True,
    show_r2: bool = True,
    **kwargs: Any,  # noqa: ANN401 - forwarded to matplotlib's ax.plot()
) -> Axes:
    """Add a regression line with configurable algorithm to a matplotlib plot.

    This function examines the data in a matplotlib Axes object and adds a
    regression line to it. It supports line plots, scatter plots, and bar charts
    (both vertical and horizontal), and can handle both numeric and datetime x-axis values.

    For bar charts, the function automatically detects orientation using matplotlib's
    BarContainer API and extracts appropriate x,y coordinates from bar positions and heights.

    Note: If an axes contains multiple plot types (e.g., both lines and bars), the function
    processes them in priority order: lines first, then bars, then scatter plots. Only the
    first available plot type will be used for regression analysis.

    Args:
        ax (Axes): The matplotlib axes object containing the plot (line, scatter, or bar).
        regression_type (Literal["linear", "power", "logarithmic", "exponential"], optional):
            Regression algorithm to use. Supported values:
            - "linear": y = mx + b (default, OLS regression)
            - "power": y = ax^b (elasticity analysis, log-log transformation)
            - "logarithmic": y = a*ln(x) + b (diminishing returns analysis)
            - "exponential": y = ae^(bx) (growth/decay patterns)
            Defaults to "linear".
        color (str, optional): Color of the regression line. Defaults to "red".
        linestyle (str, optional): Style of the regression line. Defaults to "--".
        text_position (float, optional): Relative position (0-1) for the equation text. Defaults to 0.6.
        show_equation (bool, optional): Whether to display the equation on the plot. Defaults to True.
        show_r2 (bool, optional): Whether to display the R² value on the plot. Defaults to True.
        kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        Axes: The matplotlib axes with the regression line added.

    Raises:
        ValueError: If the plot contains no visible lines, scatter points, or bar patches, or if
            regression_type is not supported.

    Examples:
        Basic linear regression (backward compatible):
        >>> ax = data.plot.scatter(x='price', y='demand')
        >>> add_regression_line(ax)

        Power law regression for price elasticity:
        >>> add_regression_line(ax, regression_type="power")

        Bar chart with regression line:
        >>> ax = df.plot.bar(x='category', y='sales')
        >>> add_regression_line(ax, regression_type="linear")
    """
    # Validate regression type
    supported_types = ["linear", "power", "logarithmic", "exponential"]
    if regression_type not in supported_types:
        error_msg = f"Unsupported regression_type '{regression_type}'. Supported types: {supported_types}"
        raise ValueError(error_msg)

    # Extract data from the plot
    x_data, y_data = _extract_plot_data(ax)

    # Convert to numeric data and validate
    x_numeric, y_numeric = _prepare_numeric_data(x_data, y_data)

    # Apply algorithm-specific data validation
    _validate_regression_data(x_numeric, y_numeric, regression_type)

    # Perform regression calculation
    param1, param2, r_squared = _perform_regression_calculation(regression_type, x_numeric, y_numeric)

    # Generate regression line points
    x_min, x_max = ax.get_xlim()
    data_size = len(x_numeric)
    x_line, y_line = _generate_regression_line(regression_type, param1, param2, x_min, x_max, data_size)

    if len(x_line) == 0:
        warnings.warn(
            f"Regression line for '{regression_type}' produced no finite values in the visible range. "
            "Consider adjusting axis limits or using a different regression type.",
            UserWarning,
            stacklevel=2,
        )
        return ax

    # Plot the regression line
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, **kwargs)

    # Add equation and R² text if either is requested
    if show_equation or show_r2:
        _add_equation_text(ax, param1, param2, r_squared, color, text_position, show_equation, show_r2, regression_type)

    return ax
