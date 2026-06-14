"""Trend line fitting and plotting helpers for matplotlib axes."""

import warnings
from datetime import datetime
from typing import Any, Literal, cast, get_args

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.container import BarContainer
from matplotlib.dates import date2num
from matplotlib.patches import Rectangle
from scipy import stats

from openretailscience.core.validation import ensure_value_choice
from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties

_MIN_LINE_POINTS = 50
_MAX_LINE_POINTS = 500
_DATA_SIZE_MULTIPLIER = 3
_POSITIVE_X_FLOOR = 1e-6
_VARIANCE_THRESHOLD = 1e-10
_TEXT_X_PADDING = 0.05  # 5% from left

TrendType = Literal["linear", "power", "logarithmic", "exponential"]

_POSITIVITY_REQUIREMENTS: dict[TrendType, tuple[bool, bool]] = {
    "power": (True, True),
    "logarithmic": (True, False),
    "exponential": (False, True),
}

# Types whose closed-form curve can overflow float64 at large inputs and need finite-mask filtering.
_OVERFLOW_SENSITIVE_TYPES: frozenset[TrendType] = frozenset({"power", "exponential"})


def _linregress(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, float]:
    """Run ``scipy.stats.linregress`` with a precise return type.

    SciPy ships no type stub for ``linregress``, so its ``LinregressResult`` is seen as an
    untyped tuple of ``object`` by static checkers. At runtime it is a namedtuple of float64
    scalars; this wrapper restores that contract for callers.

    Args:
        x (np.ndarray): Independent variable samples.
        y (np.ndarray): Dependent variable samples.

    Returns:
        tuple[float, float, float, float, float]: slope, intercept, rvalue, pvalue, stderr.
    """
    return cast("tuple[float, float, float, float, float]", stats.linregress(x, y))


def _calculate_r_squared_original_space(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """Calculate R² in original data space.

    Args:
        y_actual (np.ndarray): Actual y values.
        y_predicted (np.ndarray): Predicted y values from trend model.

    Returns:
        float: R² value calculated in original data space.
    """
    ss_res = np.sum((y_actual - y_predicted) ** 2)  # Sum of squares of residuals
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)  # Total sum of squares

    # Handle edge case where all y values are (nearly) identical
    if ss_tot < _VARIANCE_THRESHOLD:
        return 1.0 if ss_res < _VARIANCE_THRESHOLD else 0.0

    return 1 - (ss_res / ss_tot)


def _perform_trend_calculation(
    trend_type: TrendType,
    x_filtered: np.ndarray,
    y_filtered: np.ndarray,
) -> tuple[float, float, float]:
    """Perform trend calculation and return coefficients and R² in original data space.

    Args:
        trend_type (str): Type of trend to fit.
        x_filtered (np.ndarray): Filtered x data.
        y_filtered (np.ndarray): Filtered y data.

    Returns:
        tuple[float, float, float]: param1, param2, r_squared (calculated in original data space)
    """
    if trend_type == "linear":
        # Linear trend: y = mx + b
        slope, intercept, r_value, _, _ = _linregress(x_filtered, y_filtered)
        return slope, intercept, r_value**2

    if trend_type == "power":
        # Power law trend: y = ax^b → log(y) = log(a) + b*log(x)
        log_x = np.log(x_filtered)
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = _linregress(log_x, log_y)
        a = float(np.exp(intercept))  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * (x_filtered**b)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if trend_type == "logarithmic":
        # Logarithmic trend: y = a*ln(x) + b
        log_x = np.log(x_filtered)
        slope, intercept, _, _, _ = _linregress(log_x, y_filtered)
        a = slope  # a = slope, b = intercept
        b = intercept

        # Calculate R² in original data space
        y_predicted = a * np.log(x_filtered) + b
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    if trend_type == "exponential":
        # Exponential trend: y = ae^(bx) → ln(y) = ln(a) + bx
        log_y = np.log(y_filtered)
        slope, intercept, _, _, _ = _linregress(x_filtered, log_y)
        a = float(np.exp(intercept))  # Convert back: a = exp(intercept), b = slope
        b = slope

        # Calculate R² in original data space
        y_predicted = a * np.exp(b * x_filtered)
        r_squared = _calculate_r_squared_original_space(y_filtered, y_predicted)
        return a, b, r_squared

    msg = f"Unsupported trend type: {trend_type}"
    raise ValueError(msg)


def _generate_trend_line(
    trend_type: TrendType,
    param1: float,
    param2: float,
    x_min: float,
    x_max: float,
    data_size: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate trend line points for plotting with adaptive point calculation.

    Args:
        trend_type (str): Type of trend.
        param1 (float): First parameter (slope/a coefficient).
        param2 (float): Second parameter (intercept/b coefficient).
        x_min (float): Minimum x value for line.
        x_max (float): Maximum x value for line.
        data_size (int): Number of original data points for adaptive calculation.

    Returns:
        tuple[np.ndarray, np.ndarray]: x_line, y_line arrays for plotting.
    """
    if trend_type == "linear":
        # Linear: use endpoints for efficiency
        x_line = np.array([x_min, x_max])
        y_line = param1 * x_line + param2
        return x_line, y_line

    # For non-linear types, use adaptive point calculation
    # Base points on data size but ensure smooth curves for complex functions
    adaptive_points = max(data_size * _DATA_SIZE_MULTIPLIER, _MIN_LINE_POINTS)
    num_points = min(adaptive_points, _MAX_LINE_POINTS)

    # Derive from the single source of truth
    requirements = _POSITIVITY_REQUIREMENTS.get(trend_type)
    requires_positive_x = requirements is not None and requirements[0]
    x_start = max(x_min, _POSITIVE_X_FLOOR) if requires_positive_x else x_min
    x_line = np.linspace(x_start, x_max, num_points)

    if trend_type == "power":
        # Suppress overflow warning — large exponents can overflow to inf, which is filtered below
        with np.errstate(over="ignore"):
            y_line = param1 * (x_line**param2)
    elif trend_type == "logarithmic":
        y_line = param1 * np.log(x_line) + param2
    elif trend_type == "exponential":
        # Suppress overflow warning — large x values can overflow exp() to inf, which is filtered below
        with np.errstate(over="ignore"):
            y_line = param1 * np.exp(param2 * x_line)
    else:
        msg = f"Unsupported trend type for line generation: {trend_type}"
        raise ValueError(msg)

    if trend_type in _OVERFLOW_SENSITIVE_TYPES:
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
    trend_type: TrendType = "linear",
) -> None:
    """Add equation and R² text to the plot.

    Args:
        ax (Axes): The matplotlib axes object.
        param1 (float): First trend parameter (slope/a coefficient).
        param2 (float): Second trend parameter (intercept/b coefficient).
        r_squared (float): The R² value of the trend fit.
        color (str): The color of the text.
        text_position (float): The relative y-position of the text.
        show_equation (bool): Whether to display the equation.
        show_r2 (bool): Whether to display the R² value.
        trend_type (TrendType): The type of trend for equation formatting.
    """
    style = PlotStyleHelper()

    equation_parts = []

    if show_equation:
        if trend_type == "linear":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.4g}x {sign} {abs(param2):.4g}"
        elif trend_type == "power":
            exp_str = f"({param2:.4g})" if param2 < 0 else f"{param2:.4g}"
            equation = f"y = {param1:.4g}x^{exp_str}"
        elif trend_type == "logarithmic":
            sign = "+" if param2 >= 0 else "-"
            equation = f"y = {param1:.4g}ln(x) {sign} {abs(param2):.4g}"
        elif trend_type == "exponential":
            equation = f"y = {param1:.4g}e^({param2:.4g}x)"
        else:
            msg = f"Unsupported trend type for equation formatting: {trend_type}"
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
        x_data = np.asarray(lines[0].get_xdata())
        y_data = np.asarray(lines[0].get_ydata())
    # Check for bar charts (patches)
    elif len(ax.patches) > 0:
        # Detect bar orientation using BarContainer (stable API). Default covers
        # the case where patches were added without a container via ax.add_patch.
        container = ax.containers[0] if len(ax.containers) > 0 else None
        is_vertical = container.orientation == "vertical" if isinstance(container, BarContainer) else True

        # Bars are Rectangle patches; filter to them for the geometry accessors below.
        bars = [patch for patch in ax.patches if isinstance(patch, Rectangle)]
        if len(bars) == 0:
            raise ValueError("No bar (Rectangle) patches found in the plot")
        if is_vertical:
            # Vertical bars: x is center position, y is height
            bar_data = [(bar.get_x() + bar.get_width() / 2, bar.get_height()) for bar in bars]
        else:
            # Horizontal bars: x is width, y is center position
            bar_data = [(bar.get_width(), bar.get_y() + bar.get_height() / 2) for bar in bars]

        # Sort by x-coordinate to ensure consistency with grouped/stacked bar charts
        bar_data.sort(key=lambda point: point[0])
        x_data, y_data = np.array(bar_data).T
    # If no lines or bars, check for scatter plots (or other collections)
    elif len(ax.collections) > 0:
        # Extract data from the first collection (e.g., scatter plot)
        collection = ax.collections[0]
        # Get the offsets which contain the x,y coordinates
        offset_data = np.asarray(collection.get_offsets())
        if len(offset_data) > 0:
            x_data = offset_data[:, 0]
            y_data = offset_data[:, 1]
        else:
            raise ValueError("No data points found in the collection")
    else:
        raise ValueError("No visible lines or collections found in the plot")

    return x_data, y_data


def _prepare_numeric_data(x_data: np.ndarray, y_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert plot data to numeric arrays suitable for trend analysis.

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
        raise ValueError("Cannot convert y-axis values to numeric format for trend fit") from err

    # Create mask to filter out NaN values
    valid_mask = ~np.isnan(x_numeric) & ~np.isnan(y_numeric)
    if not np.any(valid_mask):
        raise ValueError("No valid (non-NaN) data points for trend fit")

    # Check that we have enough valid data points for a trend fit
    min_points_for_trend_fit = 2
    if np.sum(valid_mask) < min_points_for_trend_fit:
        error_msg = f"At least {min_points_for_trend_fit} valid data points are required for trend analysis"
        raise ValueError(error_msg)

    x_filtered = x_numeric[valid_mask]
    y_filtered = y_numeric[valid_mask]

    # Check for zero variance in x — invalid for all trend types
    if np.var(x_filtered) < _VARIANCE_THRESHOLD:
        raise ValueError("Cannot fit trend: all x values are identical (zero variance)")

    return x_filtered, y_filtered


def _validate_trend_data(
    x_data: np.ndarray,
    y_data: np.ndarray,
    trend_type: TrendType,
) -> None:
    """Validate data for specific trend types.

    Args:
        x_data (np.ndarray): The x-axis data.
        y_data (np.ndarray): The y-axis data.
        trend_type (str): The trend type being used.

    Raises:
        ValueError: If data contains values incompatible with the trend type.
    """
    requirements = _POSITIVITY_REQUIREMENTS.get(trend_type)
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
            f"{trend_type.capitalize()} trend requires all {axes_desc} values to be positive. "
            f"Found {' and '.join(error_parts)}. "
            f"Please remove or transform these values before applying a {trend_type} trend."
        )
        raise ValueError(error_msg)

    # Zero variance in y produces identical log(y) values, making the trend fit meaningless
    if check_y and np.var(y_data) < _VARIANCE_THRESHOLD:
        raise ValueError("Cannot fit trend: all y values are identical (zero variance)")


def add_trend_line(
    ax: Axes,
    trend_type: TrendType = "linear",
    color: str = "red",
    linestyle: str = "--",
    text_position: float = 0.6,
    show_equation: bool = True,
    show_r2: bool = True,
    **kwargs: Any,  # noqa: ANN401 - forwarded to matplotlib's ax.plot()
) -> Axes:
    """Add a trend line with configurable algorithm to a matplotlib plot.

    This function examines the data in a matplotlib Axes object and adds a
    trend line to it. It supports line plots, scatter plots, and bar charts
    (both vertical and horizontal), and can handle both numeric and datetime x-axis values.

    For bar charts, the function automatically detects orientation using matplotlib's
    BarContainer API and extracts appropriate x,y coordinates from bar positions and heights.

    Note: If an axes contains multiple plot types (e.g., both lines and bars), the function
    processes them in priority order: lines first, then bars, then scatter plots. Only the
    first available plot type will be used for trend analysis.

    Args:
        ax (Axes): The matplotlib axes object containing the plot (line, scatter, or bar).
        trend_type (TrendType, optional):
            Trend algorithm to use. Supported values:
            - "linear": y = mx + b (default, OLS fit)
            - "power": y = ax^b (elasticity analysis, log-log transformation)
            - "logarithmic": y = a*ln(x) + b (diminishing returns analysis)
            - "exponential": y = ae^(bx) (growth/decay patterns)
            Defaults to "linear".
        color (str, optional): Color of the trend line. Defaults to "red".
        linestyle (str, optional): Style of the trend line. Defaults to "--".
        text_position (float, optional): Relative position (0-1) for the equation text. Defaults to 0.6.
        show_equation (bool, optional): Whether to display the equation on the plot. Defaults to True.
        show_r2 (bool, optional): Whether to display the R² value on the plot. Defaults to True.
        kwargs: Additional keyword arguments to pass to the plot function.

    Returns:
        Axes: The matplotlib axes with the trend line added.

    Raises:
        ValueError: If the plot contains no visible lines, scatter points, or bar patches, or if
            trend_type is not supported.

    Examples:
        Basic linear trend:
        >>> ax = data.plot.scatter(x='price', y='demand')
        >>> add_trend_line(ax)

        Power law trend for price elasticity:
        >>> add_trend_line(ax, trend_type="power")

        Bar chart with trend line:
        >>> ax = df.plot.bar(x='category', y='sales')
        >>> add_trend_line(ax, trend_type="linear")
    """
    trend_type = cast("TrendType", ensure_value_choice(trend_type, get_args(TrendType), "trend_type"))

    # Extract data from the plot
    x_data, y_data = _extract_plot_data(ax)

    # Convert to numeric data and validate
    x_numeric, y_numeric = _prepare_numeric_data(x_data, y_data)

    # Apply algorithm-specific data validation
    _validate_trend_data(x_numeric, y_numeric, trend_type)

    # Perform trend calculation
    param1, param2, r_squared = _perform_trend_calculation(trend_type, x_numeric, y_numeric)

    # Generate trend line points
    x_min, x_max = ax.get_xlim()
    data_size = len(x_numeric)
    x_line, y_line = _generate_trend_line(trend_type, param1, param2, x_min, x_max, data_size)

    if len(x_line) == 0:
        warnings.warn(
            f"Trend line for '{trend_type}' produced no finite values in the visible range. "
            "Consider adjusting axis limits or using a different trend type.",
            UserWarning,
            stacklevel=2,
        )
        return ax

    # Plot the trend line
    ax.plot(x_line, y_line, color=color, linestyle=linestyle, **kwargs)

    # Add equation and R² text if either is requested
    if show_equation or show_r2:
        _add_equation_text(ax, param1, param2, r_squared, color, text_position, show_equation, show_r2, trend_type)

    return ax
