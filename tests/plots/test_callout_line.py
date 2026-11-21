"""Tests for the plots.callout_line module."""

import warnings
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from matplotlib.axes import Axes
from matplotlib.dates import date2num
from matplotlib.lines import Line2D
from matplotlib.path import Path
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from openretailscience.options import PlotStyleHelper
from openretailscience.plots import callout_line
from openretailscience.plots.callout_line import (
    _ANCHOR_GAP_PT,
    _ANCHOR_GAP_REACH_PT,
    _HORIZONTAL_LEADER_MAX_PT,
    _LEADER_MIN_PT,
    _LEADER_NEAR_PT,
    _LEADER_STEP_PT,
    _LEFT,
    _LINE_CLEARANCE_PT,
    _LINE_GID,
    _RIGHT,
    _callout_gid,
    _horizontal_candidates,
    _line_envelope,
    _vertical_candidates,
)
from openretailscience.plots.styles.font_utils import get_font_properties

X_COL = "week_start"
VALUE_COL = "revenue"
TITLE_COL = "event"
TEXT_COL = "what_happened"

# Slack for antialiasing and bbox rounding when deciding that two rendered artists touch.
TOUCH_TOLERANCE_PX = 0.5
# A leader has to visibly start at its anchor and end at its own text.
MAX_LEADER_DETACH_PT = 12.0
# Two leaders closer than this read as one line.
MIN_LEADER_SPACING_PT = 5.0
# A leader that stops this close in front of another callout's text reads as pointing at it.
LEADER_SIGHT_PT = 16.0
# Text this close to another callout's marker reads as labelling it.
MARKER_CROWD_PT = 14.0
# Text this close to another callout's leader reads as that leader's own side text.
LEADER_CROWD_PT = 9.0
POINTS_PER_INCH = 72.0
NUM_FUZZ_SCENARIOS = 36
# Scenarios that have a layout the search only reaches after many dead ends.
HARD_TO_SEARCH_SEEDS = [155]
# Scenarios whose headroom adds a digit to the y tick labels, which shifts the axes sideways.
WIDER_TICK_LABEL_SEEDS = [9, 41, 53]
# Half the data line's stroke plus bbox rounding, by which a measured clearance can fall short of the laid-out one.
CLEARANCE_SLACK_PX = 2.5
# More callouts at one x value than there are directions for a leader to leave it.
PILE_UP_SIZE = 8
PILE_UP_NAMES = [f"Promotion {number}" for number in range(PILE_UP_SIZE)]
HIGH_DPI = 200
# Away from UTC, so a date's local midnight falls at another hour (or on another day) in UTC.
STORE_TIMEZONE = "America/New_York"

EVENT_TEXTS = [
    "Loyalty scheme relaunch",
    "Competitor opens two stores within a mile of the flagship",
    "Easter weekend",
    "Supplier strike empties the chilled aisle for nine days",
    "Price match campaign starts",
    "Heatwave lifts soft drinks and ice cream",
    "Click and collect goes live in every store",
    "Black Friday",
    "Refit closes the Riverside store",
    "Own-label range doubles to 400 lines after the spring range review",
    "Card payments outage",
    "Free delivery over 40 pounds",
]

CALLOUT_PLACEMENTS = ["random", "extrema", "clustered", "edges", "even"]
SMALL_FIGSIZE = (8.0, 5.0)
CROWDED_FIGSIZE = (5.0, 3.0)
FIGSIZES = [(10.0, 6.0), SMALL_FIGSIZE, (14.0, 6.0), (12.0, 8.0)]
MAX_CALLOUTS = 8
# The small figure has no room for more multi-line callouts than this; plot() raises beyond it.
MAX_CALLOUTS_SMALL_FIGURE = 6


def cliff(n: int, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Revenue that drops by more than half in a single week."""
    cliff_at = int(rng.integers(n // 4, 3 * n // 4))
    return np.where(np.arange(n) < cliff_at, 120_000, 45_000) + np.cumsum(noise) * 1_200


def promo_spikes(n: int, noise: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Flat revenue with six one-week promotional spikes."""
    values = 30_000 + noise * 800
    values[rng.choice(n, 6, replace=False)] += rng.uniform(25_000, 70_000, 6)
    return values


# Line shapes that stress the layout, each built from (number of rows, unit noise, rng).
SHAPE_BUILDERS = {
    "random_walk": lambda _n, noise, _rng: 50_000 + np.cumsum(noise) * 2_500,
    "seasonal": lambda n, noise, rng: (
        80_000 + 25_000 * np.sin(np.linspace(0, 4 * np.pi, n) + rng.uniform(0, np.pi)) + noise * 2_000
    ),
    "cliff": cliff,
    "promo_spikes": promo_spikes,
    "growth": lambda n, noise, _rng: 5_000 * np.exp(np.linspace(0, 3, n)) + noise * 300,
    "decline": lambda n, noise, _rng: 150_000 - 110_000 * np.linspace(0, 1, n) + noise * 1_500,
    "flat": lambda n, _noise, _rng: np.full(n, 64_000.0),
    "valley": lambda n, noise, rng: (
        20_000 + 90_000 * np.abs(np.linspace(0, 1, n) - rng.uniform(0.3, 0.7)) + noise * 500
    ),
    "sawtooth": lambda n, noise, _rng: 40_000 + 30_000 * ((np.arange(n) % 8) / 8) + noise * 400,
    "daily_noise": lambda _n, noise, _rng: 12_000 + np.cumsum(noise) * 150 + noise * 1_800,
    "three_points": lambda _n, _noise, _rng: np.array([18_000.0, 42_000.0, 27_000.0]),
    "negative_swing": lambda n, noise, _rng: 30_000 * np.sin(np.linspace(0, 3 * np.pi, n)) + noise * 3_000,
}
SHAPES = list(SHAPE_BUILDERS)
NUM_ROWS = {"daily_noise": 730, "three_points": 3}
DEFAULT_NUM_ROWS = 104


def callout_positions(placement: str, values: np.ndarray, count: int, rng: np.random.Generator) -> np.ndarray:
    """Return ``count`` row positions to call out, chosen the way ``placement`` names."""
    n = len(values)
    count = min(count, n)
    if placement == "random":
        return np.sort(rng.choice(n, count, replace=False))
    if placement == "extrema":
        half = (count + 1) // 2
        ranked = np.argsort(values)
        return np.unique(np.concatenate([ranked[:half], ranked[-(count - half) :] if count > half else ranked[:0]]))
    if placement == "clustered":
        start = int(rng.integers(0, max(n - 2 * count, 1)))
        return np.arange(start, min(start + 2 * count, n), 2)[:count]
    if placement == "edges":
        inner = rng.choice(n, max(count - 2, 0), replace=False)
        return np.unique(np.concatenate([[0, n - 1], inner]))
    if placement == "even":
        return np.unique(np.linspace(0, n - 1, count).astype(int))
    msg = f"unknown placement {placement!r}"
    raise ValueError(msg)


def fuzz_scenario(seed: int) -> tuple[pd.DataFrame, pd.DataFrame, tuple[float, float]]:
    """Build the revenue line, the events to call out and the figure size for one fuzz seed."""
    rng = np.random.default_rng(seed)
    shape = SHAPES[seed % len(SHAPES)]
    placement = CALLOUT_PLACEMENTS[(seed // len(SHAPES) + seed) % len(CALLOUT_PLACEMENTS)]
    n = NUM_ROWS.get(shape, DEFAULT_NUM_ROWS)
    values = SHAPE_BUILDERS[shape](n, rng.normal(0.0, 1.0, n), rng)
    if seed % 3 == 0:
        x = np.arange(n) - n // 2  # weeks relative to a launch
    else:
        x = pd.date_range("2023-01-02", periods=n, freq="D" if shape == "daily_noise" else "W-MON")
    df = pd.DataFrame({X_COL: x, VALUE_COL: values})

    figsize = FIGSIZES[seed % len(FIGSIZES)]
    max_callouts = MAX_CALLOUTS_SMALL_FIGURE if figsize == SMALL_FIGSIZE else MAX_CALLOUTS
    positions = callout_positions(placement, values, min(int(rng.integers(1, MAX_CALLOUTS + 1)), max_callouts), rng)
    callout_x = df[X_COL].iloc[positions].reset_index(drop=True)
    if seed % 3 != 0:
        # Not "%-d": the unpadded-day flag is a glibc extension that fails on Windows.
        titles = callout_x.dt.day.astype(str) + callout_x.dt.strftime(" %b %Y")
    else:
        titles = "Week " + callout_x.astype(str)
    texts = rng.choice(EVENT_TEXTS, len(positions), replace=False)
    callouts = pd.DataFrame({X_COL: callout_x, TITLE_COL: titles, TEXT_COL: texts})
    return df, callouts, figsize


def artist(ax: Axes, gid: str) -> Line2D | Text | None:
    """Return the artist on ``ax`` tagged ``gid``, or None."""
    matches = [child for child in ax.get_children() if child.get_gid() == gid]
    return matches[0] if len(matches) > 0 else None


def x_tick_labels(ax: Axes) -> list[str]:
    """Return the x tick labels as drawn."""
    ax.figure.canvas.draw()
    return [label.get_text() for label in ax.get_xticklabels()]


def display_points(line: Line2D) -> np.ndarray:
    """Return ``line``'s vertices in display pixels."""
    return line.get_transform().transform(line.get_xydata())


def grown(box: Bbox, by: float) -> Bbox:
    """Return ``box`` grown by ``by`` pixels on every side (shrunk when negative)."""
    return Bbox.from_extents(box.x0 - by, box.y0 - by, box.x1 + by, box.y1 + by)


def distance_to_box(point: np.ndarray, box: Bbox) -> float:
    """Return the pixel distance from ``point`` to the nearest edge of ``box`` (0 when inside)."""
    dx = max(box.x0 - point[0], 0.0, point[0] - box.x1)
    dy = max(box.y0 - point[1], 0.0, point[1] - box.y1)
    return float(np.hypot(dx, dy))


def sight_line(leader_px: np.ndarray, length: float) -> np.ndarray:
    """Return the segment continuing ``length`` pixels past the far end of a leader."""
    direction = leader_px[-1] - leader_px[0]
    return np.array([leader_px[-1], leader_px[-1] + direction / np.hypot(*direction) * length])


def leader_gap(first: np.ndarray, second: np.ndarray) -> float:
    """Return the smallest pixel distance between two axis-aligned leaders."""
    gaps = [
        max(max(a.min(), b.min()) - min(a.max(), b.max()), 0.0)
        for a, b in ((first[:, 0], second[:, 0]), (first[:, 1], second[:, 1]))
    ]
    return float(np.hypot(*gaps))


def frame_problems(ax: Axes, line_px: np.ndarray) -> list[str]:
    """List the defects of the chart frame: a data line outside the axes, a tick label cut off by the figure."""
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    axes_box = grown(ax.get_window_extent(), 1.0)
    fig_box = fig.get_window_extent()
    problems = []
    if not axes_box.contains(*line_px.min(axis=0)) or not axes_box.contains(*line_px.max(axis=0)):
        problems.append("data line leaves the axes")
    drawn_labels = [
        tick.label1
        for axis, (low, high) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim())))
        for tick in axis.get_major_ticks()
        if low <= tick.get_loc() <= high and len(tick.label1.get_text()) > 0
    ]
    for label in drawn_labels:
        box = label.get_window_extent(renderer)
        if box.x0 < fig_box.x0 or box.y0 < fig_box.y0 or box.x1 > fig_box.x1 or box.y1 > fig_box.y1:
            problems.append(f"tick label {label.get_text()!r} is cut off by the figure edge")
    return problems


def leader_problems(ax: Axes, i: int, leader_px: np.ndarray, own_text: list[Bbox], line_path: Path) -> list[str]:
    """List the defects of callout ``i``'s leader, and of its marker when it has one."""
    max_detach = MAX_LEADER_DETACH_PT * ax.figure.dpi / POINTS_PER_INCH
    inside_axes = grown(ax.get_window_extent(), -1.0)
    marker = artist(ax, _callout_gid(i, "marker"))
    # Without a marker the leader itself has to start beside the data line.
    anchor_px, reach = (leader_px[0], max_detach) if marker is None else (display_points(marker)[0], 1.0)
    checks = {
        "marker or leader start is off the data line": not line_path.intersects_bbox(
            grown(Bbox([anchor_px, anchor_px]), reach),
            filled=False,
        ),
        "leader does not start at its marker": np.hypot(*(leader_px[0] - anchor_px)) > max_detach,
        "leader does not reach its text": distance_to_box(leader_px[-1], Bbox.union(own_text)) > max_detach,
        "leader crosses the data line": Path(leader_px).intersects_path(line_path, filled=False),
        "leader is on or outside the axes edge": not all(inside_axes.contains(*end) for end in leader_px),
    }
    return [f"callout {i} {problem}" for problem, found in checks.items() if found]


def text_problems(ax: Axes, text_boxes: dict[int, list[Bbox]], leaders: dict[int, np.ndarray], line_path: Path):
    """List the defects of the callout texts: outside the axes, or overlapping the line, a leader or other text."""
    axes_box = ax.get_window_extent()
    px_per_pt = ax.figure.dpi / POINTS_PER_INCH
    problems = []
    for i, boxes in text_boxes.items():
        if len(boxes) > 1 and grown(boxes[0], -TOUCH_TOLERANCE_PX).overlaps(boxes[1]):
            problems.append(f"callout {i} title overlaps its own text")
        for box in boxes:
            tight = grown(box, -TOUCH_TOLERANCE_PX)
            if box.x0 < axes_box.x0 or box.y0 < axes_box.y0 or box.x1 > axes_box.x1 or box.y1 > axes_box.y1:
                problems.append(f"callout {i} text leaves the axes")
            if line_path.intersects_bbox(tight, filled=False):
                problems.append(f"callout {i} text overlaps the data line")
            problems.extend(
                f"callout {i} text overlaps or crowds callout {j} leader"
                for j, leader_px in leaders.items()
                if Path(leader_px).intersects_bbox(
                    grown(box, LEADER_CROWD_PT * px_per_pt) if j != i else tight,
                    filled=False,
                )
            )
            problems.extend(
                f"callout {i} text crowds callout {j} marker"
                for j in text_boxes
                if j != i
                and artist(ax, _callout_gid(j, "marker")) is not None
                and grown(box, MARKER_CROWD_PT * px_per_pt).contains(
                    *display_points(artist(ax, _callout_gid(j, "marker")))[0],
                )
            )
            problems.extend(
                f"callout {j} leader points at callout {i} text"
                for j, leader_px in leaders.items()
                if j != i
                and Path(sight_line(leader_px, LEADER_SIGHT_PT * px_per_pt)).intersects_bbox(tight, filled=False)
            )
    for i, j in combinations(text_boxes, 2):
        if any(grown(a, -TOUCH_TOLERANCE_PX).overlaps(b) for a in text_boxes[i] for b in text_boxes[j]):
            problems.append(f"callout {i} text overlaps callout {j} text")
        if Path(leaders[i]).intersects_path(Path(leaders[j]), filled=False):
            problems.append(f"callout {i} leader crosses callout {j} leader")
        if leader_gap(leaders[i], leaders[j]) < MIN_LEADER_SPACING_PT * px_per_pt:
            problems.append(f"callout {i} leader runs into callout {j} leader")
    return problems


def layout_problems(ax: Axes, num_callouts: int, expect_markers: bool = True) -> list[str]:
    """List every layout defect in the rendered chart, measured from the drawn artists alone.

    Checked for every callout: its artists exist, with a marker only if ``expect_markers``; its
    text is inside the axes and misses the data line, every other callout's text and every leader;
    its leader starts at its marker (or beside the line when it has none), ends at its own text,
    stays off the axes edge and crosses neither the data line nor another leader; its marker is on
    the data line. Checked once: the whole data line is inside the axes and no tick label is cut
    off by the figure edge.
    """
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    line_px = display_points(artist(ax, _LINE_GID))
    line_path = Path(line_px)
    problems = frame_problems(ax, line_px)

    text_boxes: dict[int, list[Bbox]] = {}
    leaders: dict[int, np.ndarray] = {}
    for i in range(num_callouts):
        title = artist(ax, _callout_gid(i, "title"))
        leader = artist(ax, _callout_gid(i, "leader"))
        body = artist(ax, _callout_gid(i, "text"))
        has_marker = artist(ax, _callout_gid(i, "marker")) is not None
        if (title is None and body is None) or leader is None or has_marker != expect_markers:
            problems.append(f"callout {i} has a missing or unexpected artist")
            continue
        text_boxes[i] = [t.get_window_extent(renderer) for t in (title, body) if t is not None]
        leaders[i] = display_points(leader)
        problems.extend(leader_problems(ax, i, leaders[i], text_boxes[i], line_path))
    return problems + text_problems(ax, text_boxes, leaders, line_path)


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def weekly_revenue() -> pd.DataFrame:
    """Two years of weekly revenue with a Christmas peak each December."""
    weeks = pd.date_range("2023-01-02", periods=104, freq="W-MON")
    seasonal = 60_000 + 30_000 * np.exp(-(((np.arange(104) % 52) - 50) ** 2) / 8)
    return pd.DataFrame({X_COL: weeks, VALUE_COL: seasonal + np.arange(104) * 150.0})


@pytest.fixture
def events() -> pd.DataFrame:
    """Three trading events to call out on the weekly revenue line."""
    return pd.DataFrame(
        {
            X_COL: pd.to_datetime(["2023-04-10", "2023-12-18", "2024-07-01"]),
            TITLE_COL: ["10 Apr 2023", "18 Dec 2023", "1 Jul 2024"],
            TEXT_COL: ["Easter weekend", "Christmas week sets a revenue record", "Loyalty scheme relaunch"],
        },
    )


@pytest.fixture
def crowded_events(weekly_revenue) -> pd.DataFrame:
    """Thirty consecutive weeks each with a long callout: more than ``CROWDED_FIGSIZE`` has room for."""
    return pd.DataFrame(
        {
            X_COL: weekly_revenue[X_COL].iloc[40:70].to_numpy(),
            TITLE_COL: [f"Week {week}" for week in range(40, 70)],
            TEXT_COL: "Competitor opens two stores within a mile of the flagship",
        },
    )


class TestPlot:
    """Tests for callout_line.plot."""

    def test_draws_title_and_text_for_every_callout(self, weekly_revenue, events):
        """Each callout row is drawn with its own title and text."""
        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
        )

        drawn = [
            tuple(artist(ax, _callout_gid(i, part)).get_text().replace("\n", " ") for part in ("title", "text"))
            for i in range(len(events))
        ]
        assert drawn == list(zip(events[TITLE_COL], events[TEXT_COL], strict=True))

    @pytest.mark.parametrize(
        ("columns", "drawn_part", "omitted_part"),
        [
            ({"callout_title_col": TITLE_COL}, "title", "text"),
            ({"callout_text_col": TEXT_COL}, "text", "title"),
        ],
    )
    def test_callouts_with_only_a_title_or_only_text(self, weekly_revenue, events, columns, drawn_part, omitted_part):
        """With one of the two columns given, only that part is drawn and the layout is still clean."""
        ax = callout_line.plot(weekly_revenue, value_col=VALUE_COL, x_col=X_COL, callouts=events, **columns)

        drawn = [artist(ax, _callout_gid(i, drawn_part)).get_text().replace("\n", " ") for i in range(len(events))]
        assert drawn == events[next(iter(columns.values()))].tolist()
        assert [artist(ax, _callout_gid(i, omitted_part)) for i in range(len(events))] == [None] * len(events)
        assert layout_problems(ax, len(events)) == []

    def test_show_markers_false_draws_leaders_without_dots(self, weekly_revenue, events):
        """With show_markers=False no marker is drawn and each leader still starts beside the line."""
        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_text_col=TEXT_COL,
            show_markers=False,
        )

        assert layout_problems(ax, len(events), expect_markers=False) == []

    @pytest.mark.parametrize("show_markers", [True, False])
    def test_leader_stops_short_of_its_point(self, weekly_revenue, events, show_markers):
        """A leader starts a small gap away from its point on the line, with or without a dot there."""
        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_text_col=TEXT_COL,
            show_markers=show_markers,
        )

        ax.figure.canvas.draw()
        values = weekly_revenue.set_index(X_COL)[VALUE_COL].loc[events[X_COL]]
        points_px = ax.transData.transform(np.column_stack([date2num(events[X_COL]), values]))
        starts_px = np.array([display_points(artist(ax, _callout_gid(i, "leader")))[0] for i in range(len(events))])
        gaps_pt = np.hypot(*(starts_px - points_px).T) * POINTS_PER_INCH / ax.figure.dpi
        assert (gaps_pt >= _ANCHOR_GAP_PT - TOUCH_TOLERANCE_PX).all()
        assert (gaps_pt <= _ANCHOR_GAP_PT + _ANCHOR_GAP_REACH_PT + TOUCH_TOLERANCE_PX).all()

    def test_blank_lines_around_a_title_or_text_are_not_drawn(self, weekly_revenue, events):
        """A trailing newline, or a title that is only whitespace, adds no blank line to the callout."""
        texts = events[TEXT_COL].tolist()
        events[TITLE_COL] = [" \n ", "18 Dec 2023", "1 Jul 2024"]
        events[TEXT_COL] += "\n"

        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
        )

        assert artist(ax, _callout_gid(0, "title")) is None
        drawn = [artist(ax, _callout_gid(i, "text")).get_text().replace("\n", " ") for i in range(len(events))]
        assert drawn == texts

    def test_dollar_signs_in_callout_text_are_not_read_as_maths(self, weekly_revenue, events):
        """Two dollar signs in one callout are drawn as written, not as a mathtext span."""
        events[TEXT_COL] = "Save $5 on a $40 shop"

        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_text_col=TEXT_COL,
        )

        assert [artist(ax, _callout_gid(i, "text")).get_parse_math() for i in range(len(events))] == [False] * len(
            events,
        )

    def test_long_text_wraps_onto_several_lines(self, weekly_revenue, events):
        """Text longer than the wrap width is broken into lines no longer than it."""
        events[TEXT_COL] = "Own-label range doubles to 400 lines after the spring range review of every category"

        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
        )

        lines = artist(ax, _callout_gid(0, "text")).get_text().split("\n")
        assert " ".join(lines) == events[TEXT_COL].iloc[0]
        assert max(len(line) for line in lines) <= callout_line._WRAP_CHARS

    @pytest.mark.parametrize(
        "date_dtype",
        ["datetime64[ns]", pd.ArrowDtype(pa.timestamp("ns"))],
        ids=["numpy", "pyarrow"],
    )
    def test_marker_sits_on_interpolated_line_between_rows(self, weekly_revenue, events, date_dtype):
        """A callout dated between two rows is anchored on the straight segment joining them."""
        events = events.iloc[[0]].assign(**{X_COL: pd.Timestamp("2023-04-13 12:00")}).astype({X_COL: date_dtype})
        before, after = weekly_revenue[VALUE_COL].iloc[[14, 15]]

        ax = callout_line.plot(
            weekly_revenue.astype({X_COL: date_dtype}),
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
        )

        marker_y = artist(ax, _callout_gid(0, "marker")).get_ydata()[0]
        assert marker_y == pytest.approx((before + after) / 2)

    @pytest.mark.parametrize("nonpositive", ["clip", "mask"])
    def test_marker_sits_on_the_drawn_line_of_a_log_scaled_axes(self, nonpositive):
        """A log y-axis draws a segment straight in log units, so its midpoint is the geometric mean of its ends."""
        launch = pd.DataFrame({X_COL: [0, 10, 20], VALUE_COL: [1_000.0, 100_000.0, 10_000.0]})
        mid_ramp_up = pd.DataFrame({X_COL: [5], TITLE_COL: ["Week 5"]})
        _, ax = plt.subplots()
        ax.set_yscale("log", nonpositive=nonpositive)

        callout_line.plot(
            launch,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=mid_ramp_up,
            callout_title_col=TITLE_COL,
            ax=ax,
        )

        assert artist(ax, _callout_gid(0, "marker")).get_ydata()[0] == pytest.approx(10_000.0)

    def test_headroom_on_a_log_scaled_axes_never_asks_for_a_limit_at_or_below_zero(self):
        """Headroom on a log y-axis is added in decades, so the layout is clean and matplotlib rejects no limit."""
        growth = pd.DataFrame({X_COL: np.arange(40), VALUE_COL: 1_000.0 * np.exp(np.linspace(0, 6, 40))})
        launches = pd.DataFrame(
            {
                X_COL: np.arange(5, 34, 4),
                TITLE_COL: [f"Week {week}" for week in range(5, 34, 4)],
                TEXT_COL: "Competitor opens two stores within a mile of the flagship",
            },
        )
        _, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        ax.set_yscale("log")

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            callout_line.plot(
                growth,
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=launches,
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
                ax=ax,
            )

        assert layout_problems(ax, len(launches)) == []

    def test_headroom_on_an_inverted_y_axis_stays_at_or_above_zero(self):
        """Sales rank drawn with first place at the top gains its headroom past the worst rank, not below zero."""
        weeks = np.arange(1, 41)
        sales_rank = pd.DataFrame(
            {X_COL: weeks, VALUE_COL: np.round(30 - 27 * (1 - np.exp(-weeks / 12)) + 2 * np.sin(weeks))},
        )
        range_reviews = pd.DataFrame(
            {
                X_COL: [6, 14, 22, 30],
                TITLE_COL: ["Week 6", "Week 14", "Week 22", "Week 30"],
                TEXT_COL: "Competitor opens two stores within a mile of the flagship",
            },
        )
        _, ax = plt.subplots(figsize=SMALL_FIGSIZE)
        ax.invert_yaxis()

        callout_line.plot(
            sales_rank,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=range_reviews,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            ax=ax,
        )

        worst_rank_shown, best_rank_shown = ax.get_ylim()
        assert best_rank_shown >= 0
        assert worst_rank_shown > sales_rank[VALUE_COL].max()
        assert layout_problems(ax, len(range_reviews)) == []

    def test_date_offset_label_is_set_in_the_tick_label_font(self, weekly_revenue, events):
        """The year the date formatter writes beside the x-axis matches the tick labels it completes."""
        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
        )

        tick_font = get_font_properties(PlotStyleHelper().tick_font).get_name()
        assert ax.xaxis.get_offset_text().get_fontname() == tick_font

    def test_timezone_aware_dates_are_labelled_in_their_own_timezone(self, weekly_revenue, events):
        """Dates in a timezone away from UTC get the x tick labels the same dates get without a timezone."""
        columns = {"value_col": VALUE_COL, "x_col": X_COL, "callout_title_col": TITLE_COL}
        naive = callout_line.plot(weekly_revenue, callouts=events, **columns)

        local = callout_line.plot(
            weekly_revenue.assign(**{X_COL: weekly_revenue[X_COL].dt.tz_localize(STORE_TIMEZONE)}),
            callouts=events.assign(**{X_COL: events[X_COL].dt.tz_localize(STORE_TIMEZONE)}),
            **columns,
        )

        assert x_tick_labels(local) == x_tick_labels(naive)

    def test_index_is_x_axis_when_x_col_omitted(self, weekly_revenue, events):
        """With x_col=None the frame's index supplies the x values the callouts are matched to."""
        ax = callout_line.plot(
            weekly_revenue.set_index(X_COL),
            value_col=VALUE_COL,
            callouts=events,
            callout_x_col=X_COL,
            callout_title_col=TITLE_COL,
        )

        assert layout_problems(ax, len(events)) == []

    def test_row_order_does_not_change_the_line(self, weekly_revenue, events):
        """Shuffled rows draw the same line as sorted rows."""
        shuffled = weekly_revenue.sample(frac=1.0, random_state=7)

        ax = callout_line.plot(
            shuffled,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
        )

        np.testing.assert_array_equal(artist(ax, _LINE_GID).get_ydata(), weekly_revenue[VALUE_COL].to_numpy())

    @pytest.mark.parametrize("seed", [*range(NUM_FUZZ_SCENARIOS), *HARD_TO_SEARCH_SEEDS])
    def test_fuzzed_layout_has_no_overlaps(self, seed):
        """Across line shapes, callout positions and figure sizes, nothing overlaps or leaves the axes."""
        df, callouts, figsize = fuzz_scenario(seed)

        ax = callout_line.plot(
            df,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=callouts,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            title="Weekly revenue and what moved it",
            source_text="Source: till data",
            figsize=figsize,
        )

        assert layout_problems(ax, len(callouts)) == []

    def test_callouts_fit_the_axes_as_drawn_on_a_week_of_daily_dates(self):
        """A short date range puts the year and month under the x-axis, which resizes the axes the callouts go in."""
        days = pd.date_range("2024-11-25", periods=7, freq="D")
        promotion_and_weekend = np.array([0, 0, 0, 45_000, 0, 15_000, 15_000])
        daily_revenue = pd.DataFrame({X_COL: days, VALUE_COL: 40_000 + 900.0 * np.arange(7) + promotion_and_weekend})
        called_out = days[:6]
        callouts = pd.DataFrame(
            {
                X_COL: called_out,
                TITLE_COL: called_out.day.astype(str) + called_out.strftime(" %b"),
                TEXT_COL: EVENT_TEXTS[: len(called_out)],
            },
        )

        ax = callout_line.plot(
            daily_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=callouts,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            figsize=SMALL_FIGSIZE,
        )

        assert layout_problems(ax, len(callouts)) == []

    @pytest.mark.parametrize("seed", WIDER_TICK_LABEL_SEEDS)
    def test_title_stays_above_the_y_axis_when_headroom_widens_the_tick_labels(self, seed):
        """Headroom that makes the y tick labels wider moves the axes, and the title moves with it."""
        df, callouts, figsize = fuzz_scenario(seed)
        chart_title = "Weekly revenue and what moved it"

        ax = callout_line.plot(
            df,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=callouts,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            title=chart_title,
            figsize=figsize,
        )

        ax.figure.canvas.draw()
        (title,) = [text for text in ax.figure.texts if text.get_text() == chart_title]
        assert title.get_window_extent().x0 == pytest.approx(ax.get_window_extent().x0, abs=1.0)

    @pytest.mark.parametrize("seed", WIDER_TICK_LABEL_SEEDS)
    def test_text_keeps_its_clearance_from_the_line_when_headroom_widens_the_tick_labels(self, seed):
        """The axes box moves when the tick labels widen; the callouts are laid out for the box as drawn."""
        df, callouts, figsize = fuzz_scenario(seed)

        ax = callout_line.plot(
            df,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=callouts,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            title="Weekly revenue and what moved it",
            figsize=figsize,
        )

        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        line_path = Path(display_points(artist(ax, _LINE_GID)))
        clearance_px = _LINE_CLEARANCE_PT * ax.figure.dpi / POINTS_PER_INCH - CLEARANCE_SLACK_PX
        texts = [artist(ax, _callout_gid(i, part)) for i in range(len(callouts)) for part in ("title", "text")]
        too_close = [
            text.get_text()
            for text in texts
            if line_path.intersects_bbox(grown(text.get_window_extent(renderer), clearance_px), filled=False)
        ]
        assert too_close == []

    @pytest.mark.parametrize("seed", [1, 4, 7, 10])
    def test_layout_holds_when_saved_at_higher_dpi(self, seed):
        """The layout is still free of overlaps when the figure is rendered at a higher dpi."""
        df, callouts, figsize = fuzz_scenario(seed)
        ax = callout_line.plot(
            df,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=callouts,
            callout_title_col=TITLE_COL,
            callout_text_col=TEXT_COL,
            figsize=figsize,
        )

        ax.figure.set_dpi(HIGH_DPI)

        assert layout_problems(ax, len(callouts)) == []

    @pytest.mark.parametrize(
        ("pile_up_titles", "pile_up_texts"),
        [(PILE_UP_NAMES, "Competitor opens two stores within a mile of the flagship"), ("", PILE_UP_NAMES)],
        ids=["by title", "by text when the title is blank"],
    )
    def test_error_names_only_the_callouts_that_could_not_be_placed(
        self,
        weekly_revenue,
        events,
        pile_up_titles,
        pile_up_texts,
    ):
        """Callouts with room of their own are not blamed for a pile-up elsewhere on the line."""
        pile_up = pd.DataFrame(
            {
                X_COL: [pd.Timestamp("2024-03-04")] * PILE_UP_SIZE,
                TITLE_COL: pile_up_titles,
                TEXT_COL: pile_up_texts,
            },
        )

        with pytest.raises(ValueError, match="Could not place") as error:
            callout_line.plot(
                weekly_revenue,
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=pd.concat([events, pile_up], ignore_index=True),
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
            )

        assert "Promotion" in str(error.value)
        assert [title for title in events[TITLE_COL] if title in str(error.value)] == []

    def test_failed_layout_takes_its_line_and_headroom_back_off_a_supplied_axes(self, weekly_revenue, crowded_events):
        """When the callouts cannot fit, the caller's axes loses the line, its headroom, data limits and title."""
        _, ax = plt.subplots(figsize=CROWDED_FIGSIZE)

        with pytest.raises(ValueError, match="Could not place"):
            callout_line.plot(
                weekly_revenue,
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=crowded_events,
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
                title="Weekly revenue and what moved it",
                source_text="Source: till data",
                ax=ax,
            )

        assert [text.get_text() for text in ax.figure.texts] == []
        low, high = weekly_revenue[VALUE_COL].min(), weekly_revenue[VALUE_COL].max()
        margin = (high - low) * plt.rcParams["axes.ymargin"]
        assert ax.get_ylim() == pytest.approx((low - margin, high + margin))
        assert len(ax.lines) == 0

        second_year = weekly_revenue.iloc[52:]
        ax.plot(second_year[X_COL], second_year[VALUE_COL])
        first, last = date2num(second_year[X_COL].iloc[[0, -1]])
        x_margin = (last - first) * ax.margins()[0]
        assert ax.get_xlim() == pytest.approx((first - x_margin, last + x_margin))

    def test_failed_layout_leaves_fixed_y_limits_fixed_on_a_supplied_axes(self, weekly_revenue, crowded_events):
        """Y-limits the caller set before a failed call still hold for what the caller plots next."""
        fixed_ylim = (0.0, 150_000.0)
        _, ax = plt.subplots(figsize=CROWDED_FIGSIZE)
        ax.set_ylim(fixed_ylim)

        with pytest.raises(ValueError, match="Could not place"):
            callout_line.plot(
                weekly_revenue,
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=crowded_events,
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
                ax=ax,
            )

        ax.plot(weekly_revenue[X_COL], weekly_revenue[VALUE_COL] * 2)
        assert ax.get_ylim() == fixed_ylim

    def test_failed_layout_leaves_a_supplied_axes_scaled_to_what_the_caller_drew(self, weekly_revenue, crowded_events):
        """A scatter already on the caller's axes keeps its y-limits, and still counts when the caller plots next."""
        _, ax = plt.subplots(figsize=CROWDED_FIGSIZE)
        weekly_target = weekly_revenue[VALUE_COL] * 3
        ax.scatter(weekly_revenue[X_COL], weekly_target)
        target_ylim = ax.get_ylim()

        with pytest.raises(ValueError, match="Could not place"):
            callout_line.plot(
                weekly_revenue,
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=crowded_events,
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
                ax=ax,
            )

        assert ax.get_ylim() == pytest.approx(target_ylim)

        last_year = weekly_revenue[VALUE_COL] / 2
        ax.plot(weekly_revenue[X_COL], last_year)
        margin = (weekly_target.max() - last_year.min()) * plt.rcParams["axes.ymargin"]
        assert ax.get_ylim() == pytest.approx((last_year.min() - margin, weekly_target.max() + margin))

    @pytest.mark.parametrize(
        ("line_kwargs", "getter", "expected"),
        [
            ({"lw": 3.5}, "get_linewidth", 3.5),
            ({"zorder": 5}, "get_zorder", 5),
            ({"gid": "revenue_line"}, "get_gid", "revenue_line"),
        ],
    )
    def test_line_kwargs_replace_the_line_defaults(self, weekly_revenue, events, line_kwargs, getter, expected):
        """A matplotlib alias, a zorder or a gid given for the line replaces the default instead of clashing with it."""
        ax = callout_line.plot(
            weekly_revenue,
            value_col=VALUE_COL,
            x_col=X_COL,
            callouts=events,
            callout_title_col=TITLE_COL,
            **line_kwargs,
        )

        assert getattr(ax.lines[0], getter)() == expected

    def test_line_envelope_is_the_same_for_a_line_given_right_to_left(self):
        """An inverted x-axis hands the line over right to left, which must not change its envelope."""
        axes_box = Bbox.from_extents(100.0, 50.0, 900.0, 550.0)
        line_px = np.array([[100.0, 180.0], [500.0, 420.0], [900.0, 270.0]])

        left_to_right = _line_envelope(line_px, axes_box)
        right_to_left = _line_envelope(line_px[::-1], axes_box)

        np.testing.assert_array_equal(right_to_left.low, left_to_right.low)
        np.testing.assert_array_equal(right_to_left.high, left_to_right.high)

    def test_upward_leader_starts_clear_of_the_line_beside_its_point(self):
        """On a rising line the leader's gap is measured from the line next to the point, not the point itself."""
        axes_box = Bbox.from_extents(100.0, 50.0, 900.0, 550.0)
        point = np.array([500.0, 300.0])
        line_px = np.array([[100.0, 100.0], [400.0, 200.0], point, [600.0, 400.0], [900.0, 420.0]])
        envelope = _line_envelope(line_px, axes_box)

        found = _vertical_candidates(point, (60.0, 20.0), envelope, axes_box, px=1.0)

        upward_starts = np.concatenate(
            [leader[:, 1] for _, leader, _, _, _ in found if (leader[:, 3] > point[1]).all()]
        )
        _, line_beside_point = envelope.span(point[0] - _LEADER_NEAR_PT, point[0] + _LEADER_NEAR_PT)
        assert line_beside_point > point[1]
        assert len(upward_starts) > 0
        np.testing.assert_allclose(upward_starts, line_beside_point + _ANCHOR_GAP_PT)

    def test_level_leaders_are_offered_on_both_sides_of_a_symmetric_spike(self):
        """A one-week promotional spike can take a level leader to its right as readily as to its left."""
        axes_box = Bbox.from_extents(100.3, 50.0, 900.3, 550.0)
        peak = np.array([500.8, 400.0])
        line_px = np.array([[100.3, 100.0], [490.8, 100.0], peak, [510.8, 100.0], [900.3, 100.0]])

        found = _horizontal_candidates(peak, (60.0, 20.0), _line_envelope(line_px, axes_box), axes_box, px=1.0)

        num_lengths = len(np.arange(_LEADER_MIN_PT, _HORIZONTAL_LEADER_MAX_PT, _LEADER_STEP_PT))
        assert Counter(align for _, _, align, _, _ in found) == {_LEFT: num_lengths, _RIGHT: num_lengths}

    @pytest.mark.parametrize(
        ("df_change", "callouts_change", "error", "match"),
        [
            ({VALUE_COL: np.nan}, {}, ValueError, "must not contain missing values"),
            ({VALUE_COL: np.inf}, {}, ValueError, "value_col 'revenue' must not contain infinite values"),
            ({X_COL: [*range(103), np.inf]}, {X_COL: 12}, ValueError, "x values must not contain infinite values"),
            ({VALUE_COL: "sixty thousand"}, {}, TypeError, "must be numeric"),
            ({VALUE_COL: pd.Timestamp("2024-03-04")}, {}, TypeError, "must be numeric"),
            ({X_COL: pd.Timestamp("2023-01-02")}, {}, ValueError, "must be unique"),
            ({}, {X_COL: pd.Timestamp("2026-01-05")}, ValueError, "outside the range"),
            ({}, {X_COL: 12}, TypeError, "same type"),
            ({X_COL: pd.date_range("2023-01-02", periods=104, freq="W-MON", tz="UTC")}, {}, TypeError, "same type"),
            (
                {X_COL: pd.date_range("2023-01-02", periods=104, freq="W-MON").astype(pd.ArrowDtype(pa.date32()))},
                {},
                TypeError,
                r"Got date32\[day\]\[pyarrow\]",
            ),
            ({}, {TITLE_COL: None}, ValueError, "must not contain missing values"),
            ({}, {TITLE_COL: " ", TEXT_COL: ""}, ValueError, "neither a title nor text"),
        ],
    )
    def test_rejects_invalid_values(self, weekly_revenue, events, df_change, callouts_change, error, match):
        """Bad line or callout values raise a specific error before anything is drawn."""
        with pytest.raises(error, match=match):
            callout_line.plot(
                weekly_revenue.assign(**df_change),
                value_col=VALUE_COL,
                x_col=X_COL,
                callouts=events.assign(**callouts_change),
                callout_title_col=TITLE_COL,
                callout_text_col=TEXT_COL,
            )

    @pytest.mark.parametrize(
        ("num_rows", "num_callouts", "columns", "error", "match"),
        [
            (1, 3, {"x_col": X_COL, "callout_title_col": TITLE_COL}, ValueError, "at least 2 rows"),
            (104, 0, {"x_col": X_COL, "callout_title_col": TITLE_COL}, ValueError, "at least one callout"),
            (104, 3, {"x_col": X_COL}, ValueError, "callout_title_col or callout_text_col"),
            (104, 3, {"callout_title_col": TITLE_COL}, ValueError, "callout_x_col is required"),
            (
                104,
                3,
                {"value_col": [VALUE_COL], "x_col": X_COL, "callout_title_col": TITLE_COL},
                TypeError,
                "value_col must be one column name",
            ),
            (
                104,
                3,
                {"x_col": [X_COL], "callout_x_col": X_COL, "callout_title_col": TITLE_COL},
                TypeError,
                "x_col must be one column name",
            ),
        ],
    )
    def test_rejects_arguments_it_cannot_draw(
        self,
        weekly_revenue,
        events,
        num_rows,
        num_callouts,
        columns,
        error,
        match,
    ):
        """A single-row line, no callouts, no title or text column, no callout x column, or a list for one column."""
        with pytest.raises(error, match=match):
            callout_line.plot(
                weekly_revenue.head(num_rows),
                callouts=events.head(num_callouts),
                **{"value_col": VALUE_COL, **columns},
            )
