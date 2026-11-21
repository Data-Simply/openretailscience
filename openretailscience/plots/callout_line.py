"""A single line of pre-aggregated data with events called out on it."""

from __future__ import annotations

import textwrap
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cbook
from matplotlib.dates import ConciseDateFormatter
from matplotlib.lines import Line2D
from matplotlib.transforms import ScaledTranslation

from openretailscience.core.validation import ensure_columns, ensure_dataframe
from openretailscience.options import PlotStyleHelper
from openretailscience.plots.styles.colors import get_named_color
from openretailscience.plots.styles.font_utils import get_font_properties
from openretailscience.plots.styles.styling_helpers import apply_chart_chrome, standard_graph_styles

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.text import Text
    from matplotlib.transforms import Bbox, Transform
    from pandas.api.extensions import ExtensionDtype

_LINE_GID = "callout_line:line"
_DEFAULT_FIGSIZE = (10.0, 6.0)
_MIN_ROWS = 2
_WRAP_CHARS = 26
_POINTS_PER_INCH = 72.0
_LINE_WIDTH = 2.0
_LEADER_WIDTH = 0.8
_MARKER_SIZE_PT = 4.0
_LINE_ZORDER, _LEADER_ZORDER, _TEXT_ZORDER = 2, 3, 4  # markers share the leaders'
# Room at each end of the line, so a leader from the first or last point isn't drawn on the axes edge.
_X_MARGIN = 0.03

# Layout distances, in points so they scale with the fonts rather than the figure.
_ANCHOR_GAP_PT = 5.0  # the line at a callout's point to the start of its leader
# A vertical leader's gap is measured from the line this far either side of its point, so a jagged
# line does not close it, but moves the leader's start no further than the reach from the point.
_LEADER_NEAR_PT = 1.5
_ANCHOR_GAP_REACH_PT = 6.0
_TEXT_GAP_PT = 4.0  # end of a leader to the text it points from
_TITLE_TEXT_GAP_PT = 2.0
_LINE_CLEARANCE_PT = 6.0  # text to the data line
_TEXT_CLEARANCE_PT = 8.0  # text to another callout's text
_SIDE_GAP_PT = 6.0  # leader to its own text, when the text sits beside it
# Leader to another callout's text: wider than the side gap, so that text cannot be read as the leader's own.
_LEADER_CLEARANCE_PT = 12.0
_LEADER_SPACING_PT = 6.0  # leader to another callout's leader
# Clear run past a leader's far end, so it cannot be read as pointing at another callout's text.
_LEADER_OVERSHOOT_PT = 20.0
_AXES_INSET_PT = 4.0
# Text keeps this far from other callouts' markers, so it cannot be read as labelling them.
_MARKER_KEEP_OUT_PT = 18.0
_LEADER_MIN_PT = 16.0
_LEADER_STEP_PT = 6.0
_HORIZONTAL_LEADER_MAX_PT = 160.0

# Layout preferences, in points of leader length: a shorter leader wins unless it costs one of these.
_SIDE_TEXT_COST = 14.0  # text beside its leader instead of centred on it
_HORIZONTAL_COST = 30.0
_LONG_LEADER_PT = 120.0  # length at which a leader costs double, so one very long leader loses to two medium ones
_HEADROOM_COST = 60.0  # per multiple of the autoscaled range added to the y-axis
_LEADER_ON_LINE_COST = 12.0  # per point of leader running along a near-vertical stretch of the line
# A leader may run along the line for this long before it counts as hidden by it.
_LEADER_ON_LINE_TOLERANCE_PT = 2.0

# Extra y-range tried in turn, as multiples of the autoscaled range, and the share of it added on top.
_HEADROOM_LEVELS = (0.0, 0.2, 0.4, 0.7, 1.0, 1.5, 2.2, 3.0)
_HEADROOM_TOP_SHARES = (1.0, 0.5, 0.0)
# Levels tried past the first that fits, in case a little more room buys a much tidier layout.
_HEADROOM_LOOKAHEAD = 2
# Search steps allowed before a first layout is found, and to improve on it afterwards.
_SEARCH_NODE_BUDGET = 10000
_REFINE_NODE_BUDGET = 4000
# Search steps per callout when working out which ones to name in the error; it runs once per callout.
_LEFT_OVER_NODE_BUDGET = 1000

_CENTER, _LEFT, _RIGHT = "center", "left", "right"
_COLUMN_CENTER = 0.5  # a pixel column's middle, as an offset from its left edge


def _callout_gid(index: int, part: str) -> str:
    """Return the gid of one artist of callout ``index``: ``title``, ``text``, ``leader`` or ``marker``."""
    return f"callout_line:{index}:{part}"


@dataclass
class _LineEnvelope:
    """The data line's lowest and highest pixel per pixel column of the axes.

    The line is a function of x, so a rectangle touches it exactly when the rectangle's y-range
    overlaps the line's y-range over the rectangle's columns.
    """

    x0: float
    low: np.ndarray
    high: np.ndarray

    def span(self, x_start: float, x_end: float) -> tuple[float, float]:
        """Return the line's (lowest, highest) pixel between two x pixels; (inf, -inf) if it is absent."""
        first = max(int(np.floor(x_start - self.x0)), 0)
        last = min(int(np.floor(x_end - self.x0)), len(self.low) - 1)
        if first > last:
            return np.inf, -np.inf
        return float(self.low[first : last + 1].min()), float(self.high[first : last + 1].max())


@dataclass
class _Candidates:
    """The positions one callout could take, cheapest first.

    Attributes:
        text: ``(n, 4)`` text-block rectangles as ``x0, y0, x1, y1`` display pixels.
        leader: ``(n, 4)`` leader segments as ``x_start, y_start, x_end, y_end``, starting at the anchor end.
        align: Horizontal alignment of the text within its block, per candidate.
        cost: Layout preference per candidate; lower is better.
    """

    text: np.ndarray
    leader: np.ndarray
    align: np.ndarray
    cost: np.ndarray


@dataclass
class _Search:
    """Branch-and-bound state for choosing one candidate per callout."""

    costs: list[np.ndarray]
    conflicts: dict[tuple[int, int], np.ndarray]
    nodes_left: int
    refine_nodes: int
    best_cost: float = np.inf
    best: dict[int, int] | None = None


@dataclass
class _Layout:
    """A solved layout: per callout, its text block and leader as pixel offsets from its anchor."""

    text: np.ndarray
    leader: np.ndarray
    align: list[str]
    cost: float
    ylim: tuple[float, float]


def _line_envelope(line_px: np.ndarray, axes_box: Bbox) -> _LineEnvelope:
    """Build the per-column envelope of the data line across the axes width."""
    num_columns = max(int(np.ceil(axes_box.width)), 1)
    edges = axes_box.x0 + np.arange(num_columns + 1)
    # np.interp needs ascending x, and an inverted x-axis hands the line over right to left.
    line_px = line_px[np.argsort(line_px[:, 0])]
    at_edges = np.interp(edges, line_px[:, 0], line_px[:, 1], left=np.nan, right=np.nan)
    low = np.fmin(at_edges[:-1], at_edges[1:])
    high = np.fmax(at_edges[:-1], at_edges[1:])
    low = np.where(np.isnan(low), np.inf, low)
    high = np.where(np.isnan(high), -np.inf, high)
    columns = np.clip(np.floor(line_px[:, 0] - axes_box.x0).astype(int), 0, num_columns - 1)
    np.minimum.at(low, columns, line_px[:, 1])
    np.maximum.at(high, columns, line_px[:, 1])
    return _LineEnvelope(x0=axes_box.x0, low=low, high=high)


def _length_cost(lengths_pt: np.ndarray | float) -> np.ndarray | float:
    """Return the cost of leaders of the given lengths, growing faster than the length."""
    return lengths_pt * (1.0 + lengths_pt / _LONG_LEADER_PT)


def _vertical_candidates(
    anchor: np.ndarray,
    block: tuple[float, float],
    envelope: _LineEnvelope,
    axes_box: Bbox,
    px: float,
) -> list[tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]]:
    """Candidates with a leader straight up or down, the text centred on its end or beside it.

    The last array of each tuple is the length of leader, in points, that runs along the data line.
    """
    anchor_x, anchor_y = anchor
    width, height = block
    inset = _AXES_INSET_PT * px
    clearance = _LINE_CLEARANCE_PT * px
    lengths = np.arange(_LEADER_MIN_PT * px, axes_box.height, _LEADER_STEP_PT * px)
    near_low, near_high = envelope.span(anchor_x - _SIDE_GAP_PT * px / 2, anchor_x + _SIDE_GAP_PT * px / 2)
    side_gap = _SIDE_GAP_PT * px
    found = []
    beside_low, beside_high = envelope.span(anchor_x - _LEADER_NEAR_PT * px, anchor_x + _LEADER_NEAR_PT * px)
    for direction in (1.0, -1.0):
        line_edge = beside_high if direction > 0 else beside_low
        base_y = anchor_y + direction * np.clip(direction * (line_edge - anchor_y), 0.0, _ANCHOR_GAP_REACH_PT * px)
        leader_start = base_y + direction * _ANCHOR_GAP_PT * px
        for align, text_x0 in (
            (_CENTER, anchor_x - width / 2),
            (_LEFT, anchor_x + side_gap),
            (_RIGHT, anchor_x - side_gap - width),
        ):
            if text_x0 < axes_box.x0 + inset or text_x0 + width > axes_box.x1 - inset:
                continue
            line_low, line_high = envelope.span(text_x0 - clearance, text_x0 + width + clearance)
            if align == _CENTER:
                leader_end = base_y + direction * lengths
                text_near = leader_end + direction * _TEXT_GAP_PT * px
            else:
                text_near = base_y + direction * lengths
                leader_end = text_near + direction * height
            text_far = text_near + direction * height
            text_y0 = np.minimum(text_near, text_far)
            text_y1 = np.maximum(text_near, text_far)
            fits = (
                (text_y0 >= axes_box.y0 + inset)
                & (text_y1 <= axes_box.y1 - inset)
                & ((text_y0 - clearance > line_high) | (text_y1 + clearance < line_low))
            )
            leader_y0 = np.minimum(leader_start, leader_end)
            leader_y1 = np.maximum(leader_start, leader_end)
            on_line = np.clip(np.minimum(leader_y1, near_high) - np.maximum(leader_y0, near_low), 0.0, None)
            cost = (
                _length_cost(lengths / px)
                + on_line / px * _LEADER_ON_LINE_COST
                + (_SIDE_TEXT_COST if align != _CENTER else 0.0)
            )
            count = int(fits.sum())
            text = np.column_stack(
                [np.full(count, text_x0), text_y0[fits], np.full(count, text_x0 + width), text_y1[fits]],
            )
            leader = np.column_stack(
                [np.full(count, anchor_x), np.full(count, leader_start), np.full(count, anchor_x), leader_end[fits]],
            )
            found.append((text, leader, align, cost[fits], on_line[fits] / px))
    return found


def _horizontal_candidates(
    anchor: np.ndarray,
    block: tuple[float, float],
    envelope: _LineEnvelope,
    axes_box: Bbox,
    px: float,
) -> list[tuple[np.ndarray, np.ndarray, str, np.ndarray, np.ndarray]]:
    """Candidates with a leader running left or right to text level with the anchor; none runs along the line."""
    anchor_x, anchor_y = anchor
    width, height = block
    inset = _AXES_INSET_PT * px
    clearance = _LINE_CLEARANCE_PT * px
    leader_clearance = _SIDE_GAP_PT * px / 2
    text_y0 = anchor_y - height / 2
    text_y1 = anchor_y + height / 2
    if text_y0 < axes_box.y0 + inset or text_y1 > axes_box.y1 - inset:
        return []
    found = []
    for direction, align in ((1.0, _LEFT), (-1.0, _RIGHT)):
        leader_start = anchor_x + direction * _ANCHOR_GAP_PT * px
        for length in np.arange(_LEADER_MIN_PT * px, _HORIZONTAL_LEADER_MAX_PT * px, _LEADER_STEP_PT * px):
            leader_end = anchor_x + direction * length
            text_near = leader_end + direction * _TEXT_GAP_PT * px
            text_x0, text_x1 = sorted((text_near, text_near + direction * width))
            if text_x0 < axes_box.x0 + inset or text_x1 > axes_box.x1 - inset:
                break
            line_low, line_high = envelope.span(text_x0 - clearance, text_x1 + clearance)
            if not (text_y0 - clearance > line_high or text_y1 + clearance < line_low):
                continue
            # Checked from the envelope column next to the anchor's, not from the leader's start: a
            # spike in the gap between them would otherwise sit between the marker and its leader.
            anchor_column = np.floor(anchor_x - envelope.x0)
            beside_anchor = envelope.x0 + anchor_column + direction + _COLUMN_CENTER
            leader_low, leader_high = envelope.span(min(beside_anchor, leader_end), max(beside_anchor, leader_end))
            if not (anchor_y - leader_clearance > leader_high or anchor_y + leader_clearance < leader_low):
                continue
            found.append(
                (
                    np.array([[text_x0, text_y0, text_x1, text_y1]]),
                    np.array([[leader_start, anchor_y, leader_end, anchor_y]]),
                    align,
                    np.array([_length_cost(length / px) + _HORIZONTAL_COST]),
                    np.zeros(1),
                ),
            )
    return found


def _candidates(
    anchor: np.ndarray,
    block: tuple[float, float],
    envelope: _LineEnvelope,
    axes_box: Bbox,
    px: float,
    other_anchors: np.ndarray,
) -> _Candidates:
    """Return every position for one callout that stays in the axes and clear of the data line.

    Positions whose text crowds another callout's marker are dropped; positions whose leader runs
    along the line are kept only when the callout has no other.
    """
    found = [
        *_vertical_candidates(anchor, block, envelope, axes_box, px),
        *_horizontal_candidates(anchor, block, envelope, axes_box, px),
    ]
    if len(found) == 0:
        return _Candidates(np.empty((0, 4)), np.empty((0, 4)), np.empty(0, dtype=object), np.empty(0))
    text = np.concatenate([item[0] for item in found])
    leader = np.concatenate([item[1] for item in found])
    align = np.concatenate([np.full(len(item[3]), item[2], dtype=object) for item in found])
    cost = np.concatenate([item[3] for item in found])
    leader_visible = np.concatenate([item[4] for item in found]) <= _LEADER_ON_LINE_TOLERANCE_PT
    keep_out = _grown_rects(text, _MARKER_KEEP_OUT_PT * px)
    crowds_marker = (
        (keep_out[:, None, 0] < other_anchors[None, :, 0])
        & (other_anchors[None, :, 0] < keep_out[:, None, 2])
        & (keep_out[:, None, 1] < other_anchors[None, :, 1])
        & (other_anchors[None, :, 1] < keep_out[:, None, 3])
    ).any(axis=1)
    order = np.argsort(cost, kind="stable")
    order = order[~crowds_marker[order]]
    if leader_visible[order].any():
        order = order[leader_visible[order]]
    return _Candidates(text[order], leader[order], align[order], cost[order])


def _grown_rects(rects: np.ndarray, by: float) -> np.ndarray:
    """Return ``x0, y0, x1, y1`` rectangles (or segments, in either direction) grown by ``by`` on every side."""
    low = np.minimum(rects[:, :2], rects[:, 2:]) - by
    high = np.maximum(rects[:, :2], rects[:, 2:]) + by
    return np.hstack([low, high])


def _overshot(leaders: np.ndarray, by: float) -> np.ndarray:
    """Return the leader segments with their far ends extended ``by`` pixels along their direction."""
    direction = leaders[:, 2:] - leaders[:, :2]
    unit = direction / np.linalg.norm(direction, axis=1, keepdims=True)
    return np.hstack([leaders[:, :2], leaders[:, 2:] + unit * by])


def _overlaps(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the ``(len(first), len(second))`` matrix of which rectangles overlap."""
    return (
        (first[:, None, 0] < second[None, :, 2])
        & (first[:, None, 2] > second[None, :, 0])
        & (first[:, None, 1] < second[None, :, 3])
        & (first[:, None, 3] > second[None, :, 1])
    )


def _conflicts(candidates: list[_Candidates], px: float) -> dict[tuple[int, int], np.ndarray]:
    """Map each ordered pair of callouts to the matrix of their candidates that cannot be used together.

    Pairs too far apart in x for any of their candidates to meet are left out.
    """
    texts = [_grown_rects(c.text, _TEXT_CLEARANCE_PT * px / 2) for c in candidates]
    texts_from_leader = [_grown_rects(c.text, _LEADER_CLEARANCE_PT * px) for c in candidates]
    leaders = [_grown_rects(_overshot(c.leader, _LEADER_OVERSHOOT_PT * px), 0.0) for c in candidates]
    spaced_leaders = [_grown_rects(c.leader, _LEADER_SPACING_PT * px / 2) for c in candidates]
    widest = [texts_from_leader, leaders, spaced_leaders]
    # A callout with no candidates reaches nowhere, so it pairs with nothing.
    reach_low = [min(rects[i][:, 0].min(initial=np.inf) for rects in widest) for i in range(len(candidates))]
    reach_high = [max(rects[i][:, 2].max(initial=-np.inf) for rects in widest) for i in range(len(candidates))]
    conflicts = {}
    for i in range(len(candidates)):
        for j in range(i + 1, len(candidates)):
            if reach_high[i] <= reach_low[j] or reach_high[j] <= reach_low[i]:
                continue
            clash = (
                _overlaps(texts[i], texts[j])
                | _overlaps(texts_from_leader[i], leaders[j])
                | _overlaps(leaders[i], texts_from_leader[j])
                | _overlaps(spaced_leaders[i], spaced_leaders[j])
            )
            conflicts[i, j] = clash
            conflicts[j, i] = clash.T
    return conflicts


def _extend(search: _Search, allowed: dict[int, np.ndarray], chosen: dict[int, int], cost: float) -> None:
    """Place the most constrained remaining callout in each still-allowed position, cheapest first."""
    if len(allowed) == 0:
        search.best_cost = cost
        search.best = chosen
        search.nodes_left = min(search.nodes_left, search.refine_nodes)
        return
    callout = min(allowed, key=lambda index: int(allowed[index].sum()))
    for candidate in np.flatnonzero(allowed[callout]):
        if search.nodes_left <= 0:
            return
        search.nodes_left -= 1
        new_cost = cost + float(search.costs[callout][candidate])
        narrowed = {
            other: mask & ~search.conflicts[callout, other][candidate] if (callout, other) in search.conflicts else mask
            for other, mask in allowed.items()
            if other != callout
        }
        if not all(mask.any() for mask in narrowed.values()):
            continue
        # Candidates are sorted by cost, so the first allowed one is each callout's cheapest.
        bound = new_cost + sum(float(search.costs[other][np.argmax(mask)]) for other, mask in narrowed.items())
        if bound < search.best_cost:
            _extend(search, narrowed, {**chosen, callout: int(candidate)}, new_cost)


def _options(
    ax: Axes,
    line_xy: np.ndarray,
    anchors_xy: np.ndarray,
    blocks: list[tuple[float, float]],
    line_width: float,
) -> tuple[list[_Candidates], dict[tuple[int, int], np.ndarray], np.ndarray]:
    """Return each callout's candidates for the axes' current limits, their conflicts and the anchors in pixels."""
    px = ax.figure.dpi / _POINTS_PER_INCH
    axes_box = ax.get_window_extent()
    line_px = ax.transData.transform(line_xy)
    anchors_px = ax.transData.transform(anchors_xy)
    envelope = _line_envelope(line_px, axes_box)
    # Half the line's stroke sits either side of its path.
    envelope.low -= line_width * px / 2
    envelope.high += line_width * px / 2
    candidates = [
        _candidates(anchor, block, envelope, axes_box, px, np.delete(anchors_px, index, axis=0))
        for index, (anchor, block) in enumerate(zip(anchors_px, blocks, strict=True))
    ]
    return candidates, _conflicts(candidates, px), anchors_px


def _search(
    candidates: list[_Candidates],
    conflicts: dict[tuple[int, int], np.ndarray],
    callouts: list[int],
    nodes: int,
    refine_nodes: int,
) -> _Search:
    """Search for one candidate for each of ``callouts``, ignoring the others."""
    search = _Search([c.cost for c in candidates], conflicts, nodes, refine_nodes)
    if all(len(candidates[index].cost) > 0 for index in callouts):
        _extend(search, {index: np.ones(len(candidates[index].cost), dtype=bool) for index in callouts}, {}, 0.0)
    return search


def _solve(
    ax: Axes,
    line_xy: np.ndarray,
    anchors_xy: np.ndarray,
    blocks: list[tuple[float, float]],
    line_width: float,
) -> _Layout | None:
    """Lay the callouts out for the axes' current limits; return None if they do not all fit."""
    candidates, conflicts, anchors_px = _options(ax, line_xy, anchors_xy, blocks, line_width)
    search = _search(candidates, conflicts, list(range(len(candidates))), _SEARCH_NODE_BUDGET, _REFINE_NODE_BUDGET)
    if search.best is None:
        return None
    picks = [search.best[index] for index in range(len(candidates))]
    anchor_offsets = np.tile(anchors_px, 2)
    return _Layout(
        text=np.array([c.text[pick] for c, pick in zip(candidates, picks, strict=True)]) - anchor_offsets,
        leader=np.array([c.leader[pick] for c, pick in zip(candidates, picks, strict=True)]) - anchor_offsets,
        align=[str(c.align[pick]) for c, pick in zip(candidates, picks, strict=True)],
        cost=search.best_cost,
        ylim=ax.get_ylim(),
    )


def _left_over(
    ax: Axes,
    line_xy: np.ndarray,
    anchors_xy: np.ndarray,
    blocks: list[tuple[float, float]],
    line_width: float,
) -> list[int]:
    """Return the callouts that do not fit beside the earlier ones, for the axes' current limits.

    Callouts are added one at a time in their given order, so one with room of its own is not
    returned. Each is given ``_LEFT_OVER_NODE_BUDGET`` search steps, so in a very crowded chart a
    callout that would fit after a longer search can be returned too.
    """
    candidates, conflicts, _ = _options(ax, line_xy, anchors_xy, blocks, line_width)
    placed: list[int] = []
    for index in range(len(candidates)):
        if _search(candidates, conflicts, [*placed, index], _LEFT_OVER_NODE_BUDGET, refine_nodes=0).best is not None:
            placed.append(index)
    return [index for index in range(len(candidates)) if index not in placed]


def _headroom_ylims(base: tuple[float, float], level: float, floor: float) -> list[tuple[float, float]]:
    """Return the distinct y-limits that add ``level`` times the base range, split between top and bottom.

    The lower limit stops at ``floor``, and what it could not take goes on top instead. All three
    are in the y-axis' scale units, not data units. An inverted ``base`` gives inverted limits.
    """
    low, high = sorted(base)
    extra = (high - low) * level
    ylims = []
    for top_share in _HEADROOM_TOP_SHARES:
        new_low = max(low - extra * (1.0 - top_share), floor)
        ylim = (new_low, high + extra - (low - new_low))
        if base[0] > base[1]:
            ylim = ylim[::-1]
        if ylim not in ylims:
            ylims.append(ylim)
    return ylims


def _find_layout(
    ax: Axes,
    line_xy: np.ndarray,
    anchors_xy: np.ndarray,
    blocks: list[tuple[float, float]],
    line_width: float,
) -> tuple[_Layout | None, list[int]]:
    """Return the best layout over the y-axis headroom levels, or None and the callouts left over.

    The search stops ``_HEADROOM_LOOKAHEAD`` levels past the first one that fits. The callouts left
    over are those that do not fit at the most headroom, all of it above the line.
    """
    given_ylim = ax.get_ylim()
    # Headroom is added in the axis' own scale, so a log axis gains decades and is never asked for a limit at zero.
    y_scale = ax.yaxis.get_transform()
    from_scale = y_scale.inverted()
    base = tuple(y_scale.transform(np.array(given_ylim)))
    # Non-negative data keeps a non-negative axis: revenue should not appear to run below zero.
    floor = float(y_scale.transform(min(*given_ylim, 0.0))) if line_xy[:, 1].min() >= 0 else -np.inf
    layouts: list[_Layout] = []
    levels_left = _HEADROOM_LOOKAHEAD
    for level in _HEADROOM_LEVELS:
        for ylim in _headroom_ylims(base, level, floor):
            ax.set_ylim(tuple(from_scale.transform(np.array(ylim))))
            layout = _solve(ax, line_xy, anchors_xy, blocks, line_width)
            if layout is not None:
                layout.cost += _HEADROOM_COST * level
                layouts.append(layout)
        if len(layouts) > 0:
            if levels_left == 0:
                break
            levels_left -= 1
    if len(layouts) == 0:
        most_headroom = _headroom_ylims(base, _HEADROOM_LEVELS[-1], floor)[0]
        ax.set_ylim(tuple(from_scale.transform(np.array(most_headroom))))
        left_over = _left_over(ax, line_xy, anchors_xy, blocks, line_width)
        ax.set_ylim(given_ylim)
        return None, left_over
    return min(layouts, key=lambda layout: layout.cost), []


def _anchored(ax: Axes, anchor: np.ndarray) -> Transform:
    """Return a transform from inches to display, with its origin at a data point."""
    return ax.figure.dpi_scale_trans + ScaledTranslation(anchor[0], anchor[1], ax.transData)


def _wrap(text: str) -> str:
    """Wrap each line of ``text`` to the callout width."""
    return "\n".join(textwrap.fill(line, _WRAP_CHARS) for line in text.split("\n"))


def _ensure_matching_x_types(
    x: pd.Series,
    callout_x: pd.Series,
    callout_x_col: str,
    given_dtypes: tuple[np.dtype | ExtensionDtype, np.dtype | ExtensionDtype],
) -> None:
    """Raise unless the line's x values are datetime or finite numbers and the callouts' x values are the same type.

    ``given_dtypes`` are the two dtypes as the caller passed them, for the error messages.
    """
    x_dtype, callout_x_dtype = given_dtypes
    x_is_date = pd.api.types.is_datetime64_any_dtype(x)
    if not x_is_date and not pd.api.types.is_numeric_dtype(x):
        msg = f"The line's x values must be datetime or numeric. Got {x_dtype}."
        raise TypeError(msg)
    if not x_is_date and np.isinf(x.to_numpy(dtype=float)).any():
        msg = "The line's x values must not contain infinite values."
        raise ValueError(msg)
    if x_is_date:
        same_type = pd.api.types.is_datetime64_any_dtype(callout_x) and (x.dt.tz is None) == (callout_x.dt.tz is None)
    else:
        same_type = pd.api.types.is_numeric_dtype(callout_x)
    if not same_type:
        msg = (
            f"callouts['{callout_x_col}'] must have the same type as the line's x values "
            f"({x_dtype}). Got {callout_x_dtype}."
        )
        raise TypeError(msg)


def _ensure_no_blank_callouts(callouts: pd.DataFrame, callout_text_cols: list[str]) -> None:
    """Raise if a callout's title and text are both blank, leaving it nothing to draw."""
    if (np.char.strip(callouts[callout_text_cols].to_numpy(dtype=str)) == "").all(axis=1).any():
        msg = "callouts has rows with neither a title nor text; each callout needs one that is not blank."
        raise ValueError(msg)


def _ensure_one_column_name(col: str, param_name: str) -> None:
    """Raise if ``col`` is a list: ensure_columns accepts one, but it would select a frame instead of a column."""
    if not isinstance(col, str):
        msg = f"{param_name} must be one column name. Got {type(col).__name__}."
        raise TypeError(msg)


def _numeric_values(df: pd.DataFrame, value_col: str) -> pd.Series:
    """Return ``value_col`` as numbers, raising if it holds anything else or an infinite value."""
    msg = f"value_col '{value_col}' must be numeric. Got {df[value_col].dtype}."
    # pd.to_numeric would turn dates into nanosecond counts rather than refuse them.
    if pd.api.types.is_datetime64_any_dtype(df[value_col]) or pd.api.types.is_timedelta64_dtype(df[value_col]):
        raise TypeError(msg)
    try:
        values = pd.Series(pd.to_numeric(df[value_col]).to_numpy(), name="y")
    except (ValueError, TypeError) as error:
        raise TypeError(msg) from error
    if np.isinf(values.to_numpy(dtype=float)).any():
        msg = f"value_col '{value_col}' must not contain infinite values."
        raise ValueError(msg)
    return values


def _validate(
    df: pd.DataFrame,
    value_col: str,
    x_col: str | None,
    callouts: pd.DataFrame,
    callout_x_col: str | None,
    callout_title_col: str | None,
    callout_text_col: str | None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Validate the inputs and return the line's x and y sorted by x, and the callouts' x."""
    ensure_dataframe(df, "df")
    ensure_dataframe(callouts, "callouts")
    _ensure_one_column_name(value_col, "value_col")
    ensure_columns(df, value_col, "value_col")
    if x_col is not None:
        _ensure_one_column_name(x_col, "x_col")
        ensure_columns(df, x_col, "x_col")
    callout_x_col = callout_x_col if callout_x_col is not None else x_col
    if callout_x_col is None:
        msg = "callout_x_col is required when x_col is None."
        raise ValueError(msg)
    callout_text_cols = [col for col in (callout_title_col, callout_text_col) if col is not None]
    if len(callout_text_cols) == 0:
        msg = "At least one of callout_title_col or callout_text_col is required."
        raise ValueError(msg)
    ensure_columns(callouts, [callout_x_col, *callout_text_cols], "callout columns")

    if len(df) < _MIN_ROWS:
        msg = f"df must have at least {_MIN_ROWS} rows to draw a line. Got {len(df)}."
        raise ValueError(msg)
    if len(callouts) == 0:
        msg = "callouts must have at least one callout; use line.plot for a line without callouts."
        raise ValueError(msg)

    given_x = df[x_col] if x_col is not None else df.index
    x = pd.Series(given_x.to_numpy(), name="x")
    y = _numeric_values(df, value_col)
    # Through numpy like the line's x, so that an extension dtype (pyarrow timestamps) compares as the same type.
    callout_x = pd.Series(callouts[callout_x_col].to_numpy(), name=callout_x_col)
    for name, values in (
        (f"value_col '{value_col}'", y),
        ("The line's x values", x),
        (f"callouts['{callout_x_col}']", callout_x),
        *((f"callouts['{col}']", callouts[col]) for col in callout_text_cols),
    ):
        if values.isna().any():
            msg = f"{name} must not contain missing values."
            raise ValueError(msg)
    _ensure_matching_x_types(x, callout_x, callout_x_col, (given_x.dtype, callouts[callout_x_col].dtype))
    _ensure_no_blank_callouts(callouts, callout_text_cols)
    if x.duplicated().any():
        msg = "The line's x values must be unique: pass one row per x value."
        raise ValueError(msg)
    if (callout_x < x.min()).any() or (callout_x > x.max()).any():
        msg = f"callouts['{callout_x_col}'] has values outside the range of the line's x values."
        raise ValueError(msg)

    order = x.argsort()
    return x.iloc[order], y.iloc[order], callout_x


def _create_texts(
    ax: Axes,
    callouts: pd.DataFrame,
    callout_title_col: str | None,
    callout_text_col: str | None,
) -> tuple[list[list[Text]], list[tuple[float, float]]]:
    """Create each callout's title and text artists, unplaced; return them with each block's pixel size."""
    style = PlotStyleHelper()
    fig = ax.figure
    renderer = fig._get_renderer()
    gap = _TITLE_TEXT_GAP_PT * fig.dpi / _POINTS_PER_INCH
    parts = [
        part
        for part in (
            ("title", callout_title_col, style.title_font, style.label_size, style.title_color),
            ("text", callout_text_col, style.label_font, style.data_label_size, style.subtitle_color),
        )
        if part[1] is not None
    ]

    texts = []
    blocks = []
    for index in range(len(callouts)):
        artists = []
        for part, col, font, size, color in parts:
            content = _wrap(str(callouts[col].iloc[index]).strip())
            if len(content) == 0:
                continue
            artist = ax.text(
                0,
                0,
                content,
                # Measured in axes coordinates: a log axis has no pixel for a data position of zero.
                transform=ax.transAxes,
                va="top",
                fontproperties=get_font_properties(font),
                fontsize=size,
                color=color,
                gid=_callout_gid(index, part),
                zorder=_TEXT_ZORDER,
                parse_math=False,
            )
            # Unplaced text must not steer the axes box when the layout is re-fitted.
            artist.set_in_layout(False)
            artists.append(artist)
        extents = [artist.get_window_extent(renderer) for artist in artists]
        texts.append(artists)
        blocks.append((max(e.width for e in extents), sum(e.height for e in extents) + gap * (len(extents) - 1)))
    return texts, blocks


def _draw_callouts(
    ax: Axes,
    layout: _Layout,
    anchors_xy: np.ndarray,
    texts: list[list[Text]],
    show_markers: bool,
) -> None:
    """Move each callout's text to its solved position and draw its leader and, if wanted, its marker."""
    style = PlotStyleHelper()
    fig = ax.figure
    renderer = fig._get_renderer()
    dpi = fig.dpi
    gap = _TITLE_TEXT_GAP_PT * dpi / _POINTS_PER_INCH
    for index, (anchor, artists) in enumerate(zip(anchors_xy, texts, strict=True)):
        transform = _anchored(ax, anchor)
        x0, _, x1, top = layout.text[index]
        align = layout.align[index]
        x = {_CENTER: (x0 + x1) / 2, _LEFT: x0, _RIGHT: x1}[align]
        for artist in artists:
            height = artist.get_window_extent(renderer).height
            artist.set_transform(transform)
            artist.set_position((x / dpi, top / dpi))
            artist.set_horizontalalignment(align)
            artist.set_multialignment(align)
            top -= height + gap
        leader = layout.leader[index] / dpi
        ax.add_line(
            Line2D(
                leader[[0, 2]],
                leader[[1, 3]],
                transform=transform,
                color=style.title_color,
                linewidth=_LEADER_WIDTH,
                gid=_callout_gid(index, "leader"),
                zorder=_LEADER_ZORDER,
            ),
        )
        if show_markers:
            ax.plot(
                [anchor[0]],
                [anchor[1]],
                marker="o",
                markersize=_MARKER_SIZE_PT,
                color=style.title_color,
                linestyle="none",
                gid=_callout_gid(index, "marker"),
                zorder=_LEADER_ZORDER,
                scalex=False,
                scaley=False,
            )


def plot(  # noqa: PLR0913, PLR0917
    df: pd.DataFrame,
    value_col: str,
    callouts: pd.DataFrame,
    callout_title_col: str | None = None,
    callout_text_col: str | None = None,
    x_col: str | None = None,
    callout_x_col: str | None = None,
    title: str | None = None,
    eyebrow: str | None = None,
    subtitle: str | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    source_text: str | None = None,
    ax: Axes | None = None,
    figsize: tuple[float, float] | None = None,
    show_markers: bool = True,
    **kwargs: Any,  # noqa: ANN401
) -> Axes:
    """Plot ``value_col`` as one line and call out the events in ``callouts`` on it.

    Each row of ``callouts`` becomes a bold title, wrapped text or both, joined by a leader to
    the line at that row's x value. An x value between two rows of ``df`` is anchored on the
    straight segment joining them. No callout text touches the line, another callout or a leader,
    and the y-axis is given extra headroom when the callouts need the room.

    ``df`` is not aggregated: pass one row per x value. The layout fits the figure's size and the
    axis limits at call time, so pass ``figsize`` (or an ``ax`` on a figure already at its final
    size) instead of resizing the figure afterwards, and do not change the limits afterwards. That
    rules out an ``ax`` whose y-axis is shared with a chart drawn later, and a second call on the
    same ``ax``.

    Args:
        df (pd.DataFrame): The line's data, one row per x value.
        value_col (str): The column to plot.
        callouts (pd.DataFrame): One row per callout.
        callout_title_col (str, optional): The ``callouts`` column holding each callout's bold
            title, such as the event's date already formatted for display.
        callout_text_col (str, optional): The ``callouts`` column holding each callout's text, drawn
            under the title when both are given. At least one of the two is required.
        x_col (str, optional): The column to use as the x-axis, datetime or numeric. If None, the
            index is used.
        callout_x_col (str, optional): The ``callouts`` column holding each callout's x value, of
            the same type as the line's x values. Defaults to ``x_col``; required when ``x_col`` is
            None.
        title (str, optional): The title of the plot.
        eyebrow (str, optional): Small uppercase label rendered above the title.
        subtitle (str, optional): Supporting copy rendered below the title.
        x_label (str, optional): X-axis label; omitted means no label.
        y_label (str, optional): Y-axis label; omitted means no label.
        source_text (str, optional): The source text to add to the plot.
        ax (Axes, optional): Matplotlib axes object to plot on.
        figsize (tuple[float, float], optional): Figure size in inches when ``ax`` is None.
            Defaults to (10, 6).
        show_markers (bool, optional): Draw a dot on the line at each callout's point. Defaults to True.
        **kwargs: Additional keyword arguments for matplotlib's ``Axes.plot``, applied to the data
            line. ``linewidth`` defaults to 2 and ``color`` to the primary color. Leave ``drawstyle``
            at its default: callouts are placed for straight segments, so a stepped line misses them.

    Returns:
        Axes: The matplotlib axes object.

    Raises:
        TypeError: If ``df`` or ``callouts`` is not a pandas DataFrame.
        TypeError: If ``value_col`` or ``x_col`` is not one column name, or the value column is not numeric.
        TypeError: If the line's x values are neither datetime nor numeric, or the callouts' x
            values are not of the same type.
        ValueError: If a named column is missing, or ``x_col`` and ``callout_x_col`` are both None.
        ValueError: If ``callout_title_col`` and ``callout_text_col`` are both None.
        ValueError: If ``df`` has fewer than 2 rows or ``callouts`` is empty.
        ValueError: If the values, x values, callout x values, titles or texts contain missing values.
        ValueError: If the values or the line's x values contain infinite values.
        ValueError: If a callout's title and text are both blank.
        ValueError: If the line's x values are not unique.
        ValueError: If a callout's x value is outside the range of the line's x values.
        ValueError: If the callouts cannot all be placed without overlapping; the message names
            the callouts left over.

    Examples:
        >>> from openretailscience.plots import callout_line
        >>> weekly = pd.DataFrame({"week": range(1, 9), "revenue": [52, 55, 54, 81, 60, 58, 44, 57]})
        >>> events = pd.DataFrame({"week": [4, 7], "event": ["Week 4", "Week 7"], "note": ["Easter", "Tills down"]})
        >>> ax = callout_line.plot(
        ...     df=weekly, value_col="revenue", x_col="week",
        ...     callouts=events, callout_title_col="event", callout_text_col="note",
        ...     eyebrow="Trading", title="Weekly revenue and what moved it", subtitle="Revenue, £K",
        ... )
    """
    x, y, callout_x = _validate(df, value_col, x_col, callouts, callout_x_col, callout_title_col, callout_text_col)

    created_figure = ax is None
    if ax is None:
        _, ax = plt.subplots(figsize=figsize if figsize is not None else _DEFAULT_FIGSIZE)
    y_was_autoscaled = ax.get_autoscaley_on()
    data_lim = ax.dataLim.frozen()
    line_kwargs = {
        "color": get_named_color("primary"),
        "linewidth": _LINE_WIDTH,
        "zorder": _LINE_ZORDER,
        "gid": _LINE_GID,
        **cbook.normalize_kwargs(kwargs, Line2D),
    }
    (line,) = ax.plot(x.to_numpy(), y.to_numpy(), **line_kwargs)
    styles = {
        "ax": ax,
        "title": title,
        "eyebrow": eyebrow,
        "subtitle": subtitle,
        "x_label": x_label,
        "y_label": y_label,
        "source_text": source_text,
        "show_legend": False,
        "grid_axis": "y",
        "x_margin": _X_MARGIN,
    }
    # Before the styles fit the axes box to its tick labels: the callouts are laid out in that box.
    if pd.api.types.is_datetime64_any_dtype(x):
        # The locator ticks in the data's timezone; a formatter left on UTC would label those ticks with UTC's clock.
        ax.xaxis.set_major_formatter(ConciseDateFormatter(ax.xaxis.get_major_locator(), tz=x.dt.tz))
        # The formatter writes the year beside the axis as offset text, which the tick styling does not reach.
        style = PlotStyleHelper()
        ax.xaxis.get_offset_text().set_fontproperties(get_font_properties(style.tick_font))
        ax.xaxis.get_offset_text().set_fontsize(style.tick_size)
    standard_graph_styles(**styles)

    line_xy = np.column_stack([ax.convert_xunits(x.to_numpy()), y.to_numpy()]).astype(float)
    callout_x_num = np.asarray(ax.convert_xunits(callout_x.to_numpy()), dtype=float)
    # Interpolated in the axes' scales: on a supplied log axis the drawn segments are not straight in data units.
    x_scale, y_scale = ax.xaxis.get_transform(), ax.yaxis.get_transform()
    anchors_y = y_scale.inverted().transform(
        np.interp(x_scale.transform(callout_x_num), x_scale.transform(line_xy[:, 0]), y_scale.transform(line_xy[:, 1])),
    )
    anchors_xy = np.column_stack([callout_x_num, anchors_y])

    texts, blocks = _create_texts(ax, callouts, callout_title_col, callout_text_col)
    layout, unplaced = _find_layout(ax, line_xy, anchors_xy, blocks, line.get_linewidth())
    if layout is None:
        for artist in [line, *(artist for artists in texts for artist in artists)]:
            artist.remove()
        # The removed line would otherwise stay in the data limits and scale the caller's next plot.
        # Not relim(): it leaves out collections, so would drop a scatter or fill the caller had drawn.
        ax.dataLim.set(data_lim)
        ax.set_autoscaley_on(y_was_autoscaled)
        ax.autoscale_view()
        if created_figure:
            plt.close(ax.figure)
        else:
            # With no chrome arguments this removes the title and source drawn above.
            apply_chart_chrome(ax)
        left_over = callouts.iloc[unplaced]
        name_col = callout_title_col if callout_title_col is not None else callout_text_col
        names = left_over[name_col].astype(str)
        if callout_title_col is not None and callout_text_col is not None:
            names = names.where(names.str.strip() != "", left_over[callout_text_col].astype(str))
        msg = (
            f"Could not place {len(unplaced)} of {len(callouts)} callouts without overlapping: {names.tolist()}. "
            "Use fewer or shorter callouts, or a larger figure."
        )
        raise ValueError(msg)

    ax.set_ylim(layout.ylim)
    # Applied again: headroom can widen the y tick labels, which moves the axes box and the chrome above it.
    standard_graph_styles(**styles)
    _draw_callouts(ax, layout, anchors_xy, texts, show_markers)
    return ax
