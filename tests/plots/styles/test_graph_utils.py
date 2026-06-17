"""Tests for the graph_utils module in the style package."""

import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pytest

from openretailscience.options import get_option
from openretailscience.plots.styles import graph_utils as gu
from openretailscience.plots.styles import styling_helpers as sh


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


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


class TestExpandYlimForBarLabels:
    """Tests for expand_ylim_for_bar_labels."""

    def test_already_inside_axes_leaves_ylim_unchanged(self):
        """Labels already inside the data area need no room: the first-pass early return leaves ylim untouched."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.bar(["North", "South"], [120.0, 240.0])
        bar_labels = ax.bar_label(ax.containers[0])
        ax.set_ylim(-100.0, 600.0)
        fig.canvas.draw()
        ylim_before = ax.get_ylim()

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            gu.expand_ylim_for_bar_labels(ax, list(bar_labels))

        assert ax.get_ylim() == ylim_before

    @pytest.mark.parametrize(
        ("heights", "pinned_ylim"),
        [
            # Positive bars pinned at their top extent: edge labels overflow above the axes.
            ([120.0, 240.0], (0.0, 240.0)),
            # Negative bars pinned at their bottom extent: edge labels overflow below the axes.
            ([-120.0, -240.0], (-240.0, 0.0)),
        ],
    )
    def test_grows_ylim_until_overflowing_labels_clear_axes(self, heights, pinned_ylim):
        """An overflowing label pulls ylim out until every label sits inside (above- and below-overflow cases)."""
        fig, ax = plt.subplots(figsize=(6, 4), dpi=100)
        ax.bar(["North", "South"], heights)
        bar_labels = ax.bar_label(ax.containers[0])
        ax.set_ylim(*pinned_ylim)
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

        axes_box = ax.get_window_extent(renderer=renderer)
        spills_out = any(
            label.get_window_extent(renderer=renderer).y0 < axes_box.y0
            or label.get_window_extent(renderer=renderer).y1 > axes_box.y1
            for label in bar_labels
        )
        assert spills_out  # precondition: at least one label starts outside the pinned data area

        gu.expand_ylim_for_bar_labels(ax, list(bar_labels))

        fig.canvas.draw()
        axes_box = ax.get_window_extent(renderer=renderer)
        for label in bar_labels:
            label_box = label.get_window_extent(renderer=renderer)
            assert label_box.y0 >= axes_box.y0
            assert label_box.y1 <= axes_box.y1

    def test_warns_when_labels_cannot_fit(self):
        """A label taller than the axes can never fit, so the pass cap is exhausted and a UserWarning fires."""
        fig, ax = plt.subplots(figsize=(4, 0.6), dpi=100)
        ax.bar(["North", "South"], [120.0, 240.0])
        bar_labels = ax.bar_label(ax.containers[0], fontsize=80)
        fig.canvas.draw()

        with pytest.warns(UserWarning, match="could not be brought fully inside"):
            gu.expand_ylim_for_bar_labels(ax, list(bar_labels))


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
