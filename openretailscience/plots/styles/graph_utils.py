"""Helper functions for styling graphs."""

import importlib.resources as pkg_resources
import warnings
from collections.abc import Generator
from datetime import datetime
from itertools import cycle
from typing import Any, Literal, TypedDict

import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.axis import XAxis, YAxis
from matplotlib.dates import date2num
from scipy import stats

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties

ASSETS_PATH = pkg_resources.files("openretailscience").joinpath("assets")
_MAGNITUDE_SUFFIXES = ["", "K", "M", "B", "T", "P"]
_MAGNITUDE_BASE = 1000

_MIN_LINE_POINTS = 50
_MAX_LINE_POINTS = 500
_DATA_SIZE_MULTIPLIER = 3
_POSITIVE_X_FLOOR = 1e-6
_VARIANCE_THRESHOLD = 1e-10
_TEXT_X_PADDING = 0.05  # 5% from left

# End-of-line label layout
_POINTS_PER_INCH = 72.0
_LABEL_GAP_FACTOR = 1.15  # multiple of font line height enforced between adjacent labels
_LABEL_BUMP_THRESHOLD_PX = 1.0  # display-pixel delta above which a leader line is drawn
_LEADER_LINEWIDTH = 0.8
_LEADER_ALPHA = 0.55


class _EndOfLineCandidate(TypedDict):
    """End-of-line label candidate collected from a single visible line.

    ``x_end`` is intentionally typed as object: matplotlib lines may carry
    pandas Period/Timestamp or category strings on their x-axis, none of which
    share a common numeric supertype. ``x_end`` is only fed back to ax.plot /
    ax.annotate, both of which accept any matplotlib-renderable value.
    """

    label: str
    x_end: object
    y_end: float
    color: str
    marker_size: float
    zorder: float


# Map regression_type -> (check_x_positive, check_y_positive)
_POSITIVITY_REQUIREMENTS: dict[str, tuple[bool, bool]] = {
    "power": (True, True),
    "logarithmic": (True, False),
    "exponential": (False, True),
}


def _hatches_gen() -> Generator[str, None, None]:
    """Returns a generator that cycles through predefined hatch patterns.

    Yields:
        str: The next hatch pattern in the sequence.
    """
    _hatches = ["/", "\\", "|", "-", "+", "x", "o", "O", ".", "*"]
    return cycle(_hatches)


def format_shorthand(
    num: float,
    decimals: int = 0,
    prefix: str = "",
) -> str:
    """Format a number the way a person would write it, with K/M/B/T/P magnitude suffixes.

    Examples:
        ``500000 → "500K"``, ``1.4e7 → "14M"``, ``1500 → "2K"`` (zero decimals),
        ``1500 → "1.5K"`` (one decimal). Trailing zeros are dropped.

    Args:
        num (float): The number to format.
        decimals (int, optional): The number of decimals. Defaults to 0.
        prefix (str, optional): The prefix of the returned string, eg '$'. Defaults to "".

    Returns:
        str: The formatted number, with trailing zeros removed.
    """
    magnitude = 0

    while abs(num) >= _MAGNITUDE_BASE:
        magnitude += 1
        num /= _MAGNITUDE_BASE

    # Check if the number rounds to exactly the next magnitude boundary at the current magnitude
    if round(abs(num), decimals) == _MAGNITUDE_BASE:
        num /= _MAGNITUDE_BASE
        magnitude += 1

    # If magnitude exceeds the predefined suffixes, cap at the largest suffix
    # and let the formatted number grow instead. Concatenating a synthesised
    # numeric multiplier (e.g. "1000P") would visually fuse with the formatted
    # number's leading digits and misrepresent the value.
    if magnitude < len(_MAGNITUDE_SUFFIXES):
        suffix = _MAGNITUDE_SUFFIXES[magnitude]
    else:
        overflow = magnitude - (len(_MAGNITUDE_SUFFIXES) - 1)
        num *= _MAGNITUDE_BASE**overflow
        suffix = _MAGNITUDE_SUFFIXES[-1]

    # Format the number and remove trailing zeros
    formatted_num = f"{prefix}%.{decimals}f" % num
    formatted_num = formatted_num.rstrip("0").rstrip(".") if "." in formatted_num else formatted_num

    return f"{formatted_num}{suffix}"


def truncate_to_x_digits(num_str: str, digits: int) -> str:
    """Truncate a shorthand-formatted number to the first `digits` significant digits.

    Args:
        num_str (str): The formatted number (e.g., '999.999K').
        digits (int): The number of digits to keep.

    Returns:
        str: The truncated formatted number (e.g., '999.9K').
    """
    # Split the number part and the suffix (e.g., "999.999K" -> "999.999" and "K")
    suffix = ""
    for s in _MAGNITUDE_SUFFIXES:
        if num_str.endswith(s) and s != "":
            suffix = s
            num_str = num_str[: -len(s)]  # Remove the suffix for now
            break

    # Handle negative numbers
    is_negative = num_str.startswith("-")
    if is_negative:
        num_str = num_str[1:]  # Remove the negative sign for now

    # Handle zero case explicitly
    if float(num_str) == 0:
        return f"0{suffix}"

    # Handle small numbers explicitly to avoid scientific notation
    scientific_notation_threshold = 1e-4
    if abs(float(num_str)) < scientific_notation_threshold:
        return f"{float(num_str):.{digits}f}".rstrip("0").rstrip(".")

    digits_before_decimal = len(num_str.split(".")[0])
    # Calculate how many digits to keep after the decimal
    digits_to_keep_after_decimal = digits - digits_before_decimal

    # Ensure truncation without rounding
    if digits_to_keep_after_decimal > 0:
        factor = 10**digits_to_keep_after_decimal
        truncated_num = str(int(float(num_str) * factor) / factor)
    else:
        factor = 10**digits
        truncated_num = str(int(float(num_str) * factor) / factor)

    # Reapply the negative sign if needed
    if is_negative:
        truncated_num = f"-{truncated_num}"

    # Remove unnecessary trailing zeros and decimal point
    truncated_num = truncated_num.rstrip("0").rstrip(".")

    return f"{truncated_num}{suffix}"


def draw_end_of_line_labels(ax: Axes) -> None:
    """Annotate each visible line with its series label at its right-most point.

    Used by line-style plots when ``legend_style="end_of_line"``. For each
    visible labeled line, places a small filled marker at the line's last
    finite (x, y) and a colored text label to its right. When two or more
    labels would overlap vertically, label y-positions are bumped apart by
    the font line height (markers stay anchored at the true line endpoints)
    and a thin leader connects each displaced label back to its marker.

    Args:
        ax: Matplotlib axes containing the line plots.
    """
    # Force matplotlib to settle the view limits before we read pixel positions.
    # Without this, transData can map endpoints into pixel space using a stale
    # viewLim, which makes the bump algorithm clamp every label to the chart top.
    ax.autoscale_view()

    style = PlotStyleHelper()
    legend_font = get_font_properties(style.legend_font)

    candidates: list[_EndOfLineCandidate] = []
    for line in ax.get_lines():
        label = line.get_label()
        if not label or label.startswith("_"):
            continue
        xdata = np.asarray(line.get_xdata())
        ydata = np.asarray(line.get_ydata())
        if len(xdata) == 0 or len(ydata) == 0:
            continue
        try:
            finite_mask = np.isfinite(ydata.astype(float))
        except (TypeError, ValueError):
            finite_mask = np.array([True] * len(ydata))
        if not finite_mask.any():
            continue
        idx = np.where(finite_mask)[0][-1]
        candidates.append(
            {
                "label": label,
                "x_end": xdata[idx],
                "y_end": ydata[idx],
                "color": line.get_color(),
                "marker_size": max(line.get_linewidth() * 2.0, 6.0),
                "zorder": line.get_zorder(),
            },
        )

    if len(candidates) == 0:
        return

    label_ys = _resolve_end_of_line_label_ys(ax, candidates, style.legend_size)

    for cand, y_lbl in zip(candidates, label_ys, strict=True):
        x_end = cand["x_end"]
        y_end = cand["y_end"]
        color = cand["color"]
        marker_size = cand["marker_size"]
        zorder = cand["zorder"]

        # Leader first so the marker overlays its top end cleanly. Use a dummy x=0
        # in the transform — transData is separable on cartesian axes, and the real
        # x_end may be a non-numeric type (pandas Period, Timestamp, category) that
        # matplotlib's affine transform can't handle directly.
        y_end_px = ax.transData.transform((0, y_end))[1]
        y_lbl_px = ax.transData.transform((0, y_lbl))[1]
        if abs(y_lbl_px - y_end_px) > _LABEL_BUMP_THRESHOLD_PX:
            ax.plot(
                [x_end, x_end],
                [y_end, y_lbl],
                color=color,
                linewidth=_LEADER_LINEWIDTH,
                alpha=_LEADER_ALPHA,
                clip_on=False,
                zorder=zorder + 1,
                scalex=False,
                scaley=False,
            )

        ax.plot(
            [x_end],
            [y_end],
            marker="o",
            markersize=marker_size,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="none",
            clip_on=False,
            zorder=zorder + 2,
            scalex=False,
            scaley=False,
        )
        ax.annotate(
            cand["label"],
            xy=(x_end, y_lbl),
            xytext=(marker_size + 6, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            color=color,
            fontsize=style.legend_size,
            fontproperties=legend_font,
            annotation_clip=False,
        )


def _resolve_end_of_line_label_ys(ax: Axes, candidates: list[_EndOfLineCandidate], font_pts: float) -> list[float]:
    """Greedy bump of overlapping label y-positions, returned in input order.

    Works in display (pixel) space so the minimum gap reflects the rendered
    font height regardless of the data y-scale. Sorts candidates by initial
    pixel y, walks bottom-to-top pushing each label up if it would collide
    with the previous one, and — if the topmost label exceeds the data area
    — clamps it down and back-propagates.

    Args:
        ax: The axes whose transData maps the label points.
        candidates: Input dicts as collected in ``draw_end_of_line_labels``;
            only ``x_end`` and ``y_end`` are read here.
        font_pts: Label font size in points; sets the per-label gap.

    Returns:
        A list of resolved y-positions in data coordinates, aligned with
        ``candidates`` by index.
    """
    n = len(candidates)
    initial_px = np.array([ax.transData.transform((0, c["y_end"]))[1] for c in candidates])
    min_gap_px = font_pts * ax.figure.dpi / _POINTS_PER_INCH * _LABEL_GAP_FACTOR

    order = np.argsort(initial_px)
    bumped = initial_px.copy()
    for k in range(1, n):
        prev = order[k - 1]
        curr = order[k]
        bumped[curr] = max(bumped[curr], bumped[prev] + min_gap_px)

    # If the topmost label was pushed past the data area, clamp it and squash downward.
    y_top_px = ax.transData.transform((0, ax.get_ylim()[1]))[1]
    top_idx = order[-1]
    if bumped[top_idx] > y_top_px:
        bumped[top_idx] = y_top_px
        for k in range(n - 2, -1, -1):
            curr = order[k]
            nxt = order[k + 1]
            bumped[curr] = min(bumped[curr], bumped[nxt] - min_gap_px)

    # Symmetric clamp: if the back-propagation pushed the bottom label below the
    # data area, clamp it up and re-run the upward bump.
    y_bottom_px = ax.transData.transform((0, ax.get_ylim()[0]))[1]
    bottom_idx = order[0]
    if bumped[bottom_idx] < y_bottom_px:
        bumped[bottom_idx] = y_bottom_px
        for k in range(1, n):
            curr = order[k]
            prev = order[k - 1]
            bumped[curr] = max(bumped[curr], bumped[prev] + min_gap_px)
        if bumped[order[-1]] > y_top_px:
            warnings.warn(
                f"{n} end-of-line labels cannot fit between ylim[0] and ylim[1] given the current "
                f"figure height and font size. Some labels render outside the data area; "
                f"consider legend_style='box' or fewer series.",
                UserWarning,
                stacklevel=2,
            )

    inv = ax.transData.inverted()
    return [float(inv.transform((0, bumped[i]))[1]) for i in range(n)]


def apply_hatches(ax: Axes, num_segments: int) -> Axes:
    """Apply hatch patterns to patches in a plot, such as bars, histograms, or area plots.

    This function divides the patches in the given Axes object into the specified
    number of segments and applies a different hatch pattern to each segment.

    Args:
        ax (Axes): The matplotlib Axes object containing the plot with patches (bars, histograms, etc.).
        num_segments (int): The number of segments to divide the patches into, with each
            segment receiving a different hatch pattern.

    Returns:
        Axes: The modified Axes object with hatches applied to the patches.
    """
    available_hatches = _hatches_gen()
    patch_groups = np.array_split(ax.patches, num_segments)
    for patch_group in patch_groups:
        hatch = next(available_hatches)
        for patch in patch_group:
            patch.set_hatch(hatch)

    legend = ax.get_legend()
    if legend:
        existing_hatches = [patch.get_hatch() for patch in ax.patches if patch.get_hatch() is not None]
        unique_hatches = [hatch for idx, hatch in enumerate(existing_hatches) if hatch not in existing_hatches[:idx]]
        for legend_patch, hatch in zip(legend.get_patches(), cycle(unique_hatches)):
            legend_patch.set_hatch(hatch)

    return ax


def get_decimals(axis_limits: tuple[float, float], tick_values: list[float], max_decimals: int = 10) -> int:
    """Pick the smallest decimal count that keeps `format_shorthand` tick labels distinct.

    Used by `set_axis_shorthand` when decimals are auto-derived for either the x-axis or the y-axis.

    Args:
        axis_limits (tuple[float, float]): The axis limits (xlim or ylim).
        tick_values (list[float]): The tick values on the same axis.
        max_decimals (int, optional): The maximum number of decimals to use. Defaults to 10.

    Returns:
        int: The number of decimals to use.
    """
    decimals = 0
    while decimals < max_decimals:
        tick_labels = [
            format_shorthand(t, decimals=decimals) for t in tick_values if t >= axis_limits[0] and t <= axis_limits[1]
        ]
        # Ensure no duplicate labels
        if len(tick_labels) == len(set(tick_labels)):
            break
        decimals += 1
    return decimals


def set_axis_shorthand(
    fmt_axis: YAxis | XAxis,
    decimals: int | None = None,
    prefix: str = "",
) -> None:
    """Apply shorthand (K/M/B/T/P) numeric formatting to a matplotlib axis.

    Args:
        fmt_axis (YAxis | XAxis): The axis to format (e.g. ``ax.xaxis`` or ``ax.yaxis``).
        decimals (int | None, optional): Number of decimal places. ``None`` derives the
            count from the current tick range so labels stay distinct. Defaults to None.
        prefix (str, optional): Prepended to each formatted value (e.g. ``"$"``). Defaults to ``""``.
    """
    if decimals is None:
        parent_ax = fmt_axis.axes
        is_xaxis = fmt_axis is parent_ax.xaxis
        limits = parent_ax.get_xlim() if is_xaxis else parent_ax.get_ylim()
        ticks = parent_ax.get_xticks() if is_xaxis else parent_ax.get_yticks()
        decimals = get_decimals(limits, ticks)
    fmt_axis.set_major_formatter(
        lambda value, _pos=None: format_shorthand(value, decimals=decimals, prefix=prefix),
    )


def set_axis_percent(
    fmt_axis: YAxis | XAxis,
    decimals: int | None = None,
    xmax: float = 1,
    symbol: str | None = "%",
) -> None:
    """Apply percent formatting to a matplotlib axis using matplotlib's ``PercentFormatter``.

    Args:
        fmt_axis (YAxis | XAxis): The axis to format (e.g. ``ax.xaxis`` or ``ax.yaxis``).
        decimals (int | None, optional): Number of decimal places. ``None`` lets matplotlib
            choose based on the displayed values. Defaults to None.
        xmax (float, optional): The value that maps to 100%. Defaults to 1.
        symbol (str | None, optional): Symbol shown after the number; pass ``None`` for
            no symbol. Defaults to ``"%"``.
    """
    fmt_axis.set_major_formatter(mtick.PercentFormatter(xmax=xmax, decimals=decimals, symbol=symbol))


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
        >>> gu.add_regression_line(ax)

        Power law regression for price elasticity:
        >>> gu.add_regression_line(ax, regression_type="power")

        Bar chart with regression line:
        >>> ax = df.plot.bar(x='category', y='sales')
        >>> gu.add_regression_line(ax, regression_type="linear")
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
