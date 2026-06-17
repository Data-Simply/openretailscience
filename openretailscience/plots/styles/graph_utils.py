"""Helper functions for styling graphs."""

from __future__ import annotations

import importlib.resources as pkg_resources
import warnings
from itertools import cycle
from typing import TYPE_CHECKING, TypedDict

import matplotlib.ticker as mtick
import numpy as np

from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.font_utils import get_font_properties

if TYPE_CHECKING:
    from collections.abc import Generator

    from matplotlib.axes import Axes
    from matplotlib.axis import XAxis, YAxis
    from matplotlib.text import Annotation

ASSETS_PATH = pkg_resources.files("openretailscience").joinpath("assets")
_MAGNITUDE_SUFFIXES = ["", "K", "M", "B", "T", "P"]
_MAGNITUDE_BASE = 1000

# End-of-line label layout
_POINTS_PER_INCH = 72.0
_LABEL_GAP_FACTOR = 1.15  # multiple of font line height enforced between adjacent labels
_LABEL_BUMP_THRESHOLD_PX = 1.0  # display-pixel delta above which a leader line is drawn
_LEADER_LINEWIDTH = 0.8
_LEADER_ALPHA = 0.55

# Bar-label headroom
_BAR_LABEL_CLEARANCE_PX = 2.0  # gap kept between an edge label and the axes boundary
_MAX_BAR_LABEL_YLIM_PASSES = 3  # expand/measure passes; the geometric correction settles within a few


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


def expand_ylim_for_bar_labels(ax: Axes, labels: list[Annotation]) -> None:
    """Grow the y-limits so bar-end value labels sit inside the axes data area.

    matplotlib bars carry sticky edges that suppress autoscale margins, so the y-view is pinned
    exactly to the bar extents. ``ax.bar_label(label_type="edge")`` then draws each value at a fixed
    point offset *outside* its bar, leaving the labels on the most extreme bars overflowing past the
    axes — into the x-axis tick-label band below (negative bars) or the chart header above (positive
    bars). This expands ``ylim`` on whichever side overflows until every label clears, measuring in
    pixel space so the reserved room tracks the rendered font height rather than a fixed fraction of
    the data range.

    Must run after the chrome layout has reflowed the axes: converting the labels' pixel overflow to
    data units depends on the final axes height.

    Args:
        ax (Axes): The axes holding the labelled bars.
        labels (list[Annotation]): The annotation artists returned by ``ax.bar_label``.
    """
    visible_labels = [label for label in labels if len(label.get_text()) > 0]
    if len(visible_labels) == 0:
        return

    fig = ax.figure
    for _ in range(_MAX_BAR_LABEL_YLIM_PASSES):
        # Draw first so the labels and axes box report settled pixel positions; with the canvas
        # drawn, get_window_extent resolves the active renderer on its own.
        fig.canvas.draw()
        axes_box = ax.get_window_extent()
        label_boxes = [label.get_window_extent() for label in visible_labels]

        # Pixels by which the outermost labels spill past each axes edge (<= 0 once inside).
        overflow_below = axes_box.y0 - min(box.y0 for box in label_boxes)
        overflow_above = max(box.y1 for box in label_boxes) - axes_box.y1
        if overflow_below <= 0 and overflow_above <= 0:
            return

        # Expanding ylim shifts every bar edge toward the centre, pulling the labels inward; the
        # clearance overshoots so each spilling label lands just inside its edge. Growing the view
        # lowers px-per-data, so a single pass under-corrects — the loop re-measures and tops up.
        y_low, y_high = ax.get_ylim()
        data_per_px = (y_high - y_low) / axes_box.height
        extra_below = overflow_below + _BAR_LABEL_CLEARANCE_PX if overflow_below > 0 else 0.0
        extra_above = overflow_above + _BAR_LABEL_CLEARANCE_PX if overflow_above > 0 else 0.0
        ax.set_ylim(y_low - extra_below * data_per_px, y_high + extra_above * data_per_px)

    # The correction converges within a couple of passes at normal proportions; if it has not after
    # the cap (e.g. a label taller than a very short axes), surface it rather than leaving the labels
    # silently clipped — mirrors the end-of-line label resolver's behaviour for infeasible geometry.
    warnings.warn(
        f"Bar labels could not be brought fully inside the axes within {_MAX_BAR_LABEL_YLIM_PASSES} "
        "passes; consider a taller figure or a smaller plot.font.data_label_size.",
        UserWarning,
        stacklevel=2,
    )


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
