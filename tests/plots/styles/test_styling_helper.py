"""Tests for the styling_helpers module."""

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter

from openretailscience.options import get_option, option_context
from openretailscience.plots.styles.styling_helpers import (
    _auto_rotate_categorical_x_ticks,
    _ZeroBlankingFormatter,
    apply_base_styling,
    apply_label,
    apply_legend,
    apply_ticks,
)


class TestStylingHelpers:
    """Unit tests for module-level styling helpers."""

    @pytest.fixture
    def fig_ax(self):
        """Fixture to create and yield a matplotlib figure and axis."""
        fig, ax = plt.subplots()
        yield fig, ax
        plt.close(fig)

    def test_apply_base_styling(self, fig_ax):
        """Test base styling application."""
        _, ax = fig_ax
        apply_base_styling(ax)

        # Test default style configuration
        assert ax.get_facecolor() == (1.0, 1.0, 1.0, 1.0)  # white background
        assert not ax.spines["top"].get_visible()
        assert not ax.spines["right"].get_visible()
        assert ax.spines["bottom"].get_visible()
        assert ax.spines["left"].get_visible()
        assert ax.get_axisbelow()

    @pytest.mark.parametrize(
        ("label_text", "axis", "get_label_func"),
        [
            ("X Label", "x", lambda ax: ax.get_xlabel()),
            ("Y Label", "y", lambda ax: ax.get_ylabel()),
        ],
    )
    def test_apply_label_axis(self, fig_ax, label_text, axis, get_label_func):
        """Test axis label styling for both x and y axes."""
        _, ax = fig_ax
        apply_label(ax, label_text, axis, pad=10)

        assert get_label_func(ax) == label_text
        label_obj = ax.xaxis.label if axis == "x" else ax.yaxis.label
        assert label_obj.get_fontsize() == get_option("plot.font.label_size")

    def test_apply_ticks(self, fig_ax):
        """Test tick styling application."""
        _, ax = fig_ax
        ax.plot([1, 2, 3, 4], [1, 4, 2, 3])  # add data to generate ticks
        apply_ticks(ax)

        tick_labels = ax.get_xticklabels() + ax.get_yticklabels()
        assert len(tick_labels) == len(ax.get_xticks()) + len(ax.get_yticks())
        for label in tick_labels:
            assert label.get_fontsize() == get_option("plot.font.tick_size")

    @staticmethod
    def _blank_tick_locs(axis) -> set[float]:
        """Return the major-tick locations whose label renders blank after a draw.

        A tick counts as blank when its label artist is hidden or its rendered
        text is empty, so the check catches a dropped label regardless of the
        mechanism. Requires the figure to have been drawn so the label artists
        carry their live text.
        """
        blank = set()
        for tick, loc in zip(axis.get_major_ticks(), axis.get_majorticklocs(), strict=True):
            if not tick.label1.get_visible() or tick.label1.get_text() == "":
                blank.add(float(loc))
        return blank

    def test_apply_ticks_blanks_only_the_zero_label(self, fig_ax):
        """On a numeric axis, apply_ticks renders the data-0 label blank and keeps every other."""
        fig, ax = fig_ax
        ax.plot([-2, -1, 0, 1, 2], [1, 2, 0, 2, 1])

        apply_ticks(ax)
        fig.canvas.draw()

        for axis in (ax.xaxis, ax.yaxis):
            assert self._blank_tick_locs(axis) == {0.0}

    def test_zero_blank_tracks_value_after_caller_changes_ticks(self, fig_ax):
        """Zero-blanking follows the value, not a tick index, when the caller re-ticks (issue #530).

        The old implementation hid the zero label by flipping a persistent
        visibility flag on the tick artist at zero's index. matplotlib reuses
        those artists by index, so a later ``set_yticks`` left the flag on
        whatever non-zero value moved into that index, silently blanking a real
        label. Deriving the blank from the value keeps it on 0 only.
        """
        fig, ax = fig_ax
        # Symmetric limits put 0 at index 1 of the default numeric ticks
        # ([-80, 0, 80, 160, 240, 320]), reproducing the leak-prone layout.
        ax.plot([0, 1, 2, 3], [-80, 100, 200, 320])
        ax.set_ylim(-80, 320)
        apply_ticks(ax)

        ax.set_ylim(0, 320)
        ax.set_yticks([0, 80, 160, 240])
        fig.canvas.draw()

        assert self._blank_tick_locs(ax.yaxis) == {0.0}

    def test_apply_ticks_is_idempotent_for_zero_blanking(self, fig_ax):
        """Re-applying ticks keeps a single zero-blanking wrapper and still blanks only 0.

        Iterative replotting (Jupyter, parametrized tests) runs the styling path
        repeatedly; the formatter wrapper must not stack on itself each time.
        """
        fig, ax = fig_ax
        ax.plot([-2, -1, 0, 1, 2], [1, 2, 0, 2, 1])

        apply_ticks(ax)
        apply_ticks(ax)
        fig.canvas.draw()

        for axis in (ax.xaxis, ax.yaxis):
            assert isinstance(axis.get_major_formatter(), _ZeroBlankingFormatter)
            assert self._blank_tick_locs(axis) == {0.0}

    def test_apply_ticks_blanks_zero_on_both_label_rows(self, fig_ax):
        """With labeltop enabled, the value-0 label is blank on both the bottom and top rows.

        Both label rows draw from the same major formatter, so blanking by value
        drops the zero label wherever it is rendered while non-zero ticks keep
        their labels on both rows.
        """
        fig, ax = fig_ax
        ax.plot([-2, -1, 0, 1, 2], [1, 2, 0, 2, 1])
        ax.tick_params(labeltop=True, labelbottom=True)

        apply_ticks(ax)
        fig.canvas.draw()

        ticks = ax.xaxis.get_major_ticks()
        locs = ax.xaxis.get_majorticklocs()
        zero_ticks = [tick for tick, loc in zip(ticks, locs, strict=True) if loc == 0]
        nonzero_ticks = [tick for tick, loc in zip(ticks, locs, strict=True) if loc != 0]
        assert len(zero_ticks) == 1
        assert zero_ticks[0].label1.get_text() == ""
        assert zero_ticks[0].label2.get_text() == ""
        assert all(tick.label1.get_text() != "" for tick in nonzero_ticks)
        assert all(tick.label2.get_text() != "" for tick in nonzero_ticks)

    def test_zero_blanking_formatter_delegates_and_blanks_zero(self, fig_ax):
        """_ZeroBlankingFormatter blanks value 0 and delegates formatting and offset to its base.

        The offset delegation matters for large-magnitude axes: ScalarFormatter
        renders a shared ``1e6``-style offset that a naive wrapper would drop.
        """
        fig, ax = fig_ax
        ax.plot([0, 1, 2], [0, 1_500_000, 3_000_000])
        fig.canvas.draw()
        base = ax.yaxis.get_major_formatter()
        base.set_locs(list(ax.yaxis.get_majorticklocs()))

        formatter = _ZeroBlankingFormatter(base)

        assert base.get_offset() != ""  # sanity: the base carries an offset to delegate
        assert formatter.get_offset() == base.get_offset()
        assert formatter(0) == ""
        assert formatter(1_500_000) == base(1_500_000)

    def test_ticklabel_format_still_works_after_apply_ticks(self, fig_ax):
        """A caller's ``ax.ticklabel_format`` succeeds after styling and 0 stays blanked.

        matplotlib's ``ticklabel_format`` calls ``ScalarFormatter``-only setters
        (``set_scientific``/``set_powerlimits``/...) on the major formatter and
        raises "only works with the ScalarFormatter" if they are missing. The
        wrapper must delegate them to the base rather than shadow the real one.
        """
        fig, ax = fig_ax
        ax.plot([0, 1, 2], [0, 1000, 2000])
        apply_ticks(ax)

        ax.ticklabel_format(style="plain", axis="y")  # must not raise
        fig.canvas.draw()

        assert self._blank_tick_locs(ax.yaxis) == {0.0}

    @pytest.mark.parametrize(
        ("outside", "expected_title"),
        [
            (False, "Test Legend"),
            (True, None),
        ],
    )
    def test_apply_legend(self, fig_ax, outside, expected_title):
        """Test legend styling (inside and outside positioning)."""
        _, ax = fig_ax
        ax.plot([1, 2, 3], [1, 2, 3], label="Series 1")
        ax.plot([1, 2, 3], [3, 2, 1], label="Series 2")

        kwargs = {"outside": outside}
        if not outside:
            kwargs["title"] = expected_title

        apply_legend(ax, **kwargs)

        legend = ax.get_legend()

        # Both inside and outside legends should have frame off
        assert not legend.get_frame_on()

        if not outside:
            assert legend.get_title().get_text() == expected_title
            assert legend.get_title().get_fontsize() == get_option("plot.font.legend_size")

        # Verify legend text styling is applied
        legend_texts = legend.get_texts()
        expected_series_count = 2
        assert len(legend_texts) == expected_series_count
        for text in legend_texts:
            assert text.get_fontsize() == get_option("plot.font.legend_size")

    def test_apply_legend_reverse_flips_label_order(self, fig_ax):
        """``reverse=True`` rebuilds the legend with the labelled artists in reversed order.

        Used by stacked area/bar plots where the pandas-default legend ordering (column
        order) doesn't match the visual stack (bottom-up).
        """
        _, ax = fig_ax
        ax.plot([1, 2, 3], [10, 20, 30], label="Footwear")
        ax.plot([1, 2, 3], [20, 30, 40], label="Apparel")
        ax.plot([1, 2, 3], [30, 40, 50], label="Accessories")

        apply_legend(ax, reverse=True)

        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Accessories", "Apparel", "Footwear"]

    def test_apply_legend_custom_labels_replaces_text(self, fig_ax):
        """``custom_labels`` overrides the labels read from the labelled artists."""
        _, ax = fig_ax
        ax.plot([1, 2, 3], [10, 20, 30], label="increased_focus")
        ax.plot([1, 2, 3], [30, 20, 10], label="decreased_focus")

        apply_legend(ax, custom_labels=["Increased Brand A", "Decreased Brand A"])

        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert labels == ["Increased Brand A", "Decreased Brand A"]

    def test_apply_legend_custom_labels_length_mismatch_raises(self, fig_ax):
        """A length mismatch between ``custom_labels`` and labelled artists raises ValueError."""
        _, ax = fig_ax
        ax.plot([1, 2, 3], [10, 20, 30], label="increased_focus")
        ax.plot([1, 2, 3], [30, 20, 10], label="decreased_focus")

        with pytest.raises(ValueError, match="legend_labels length 1 != number of legend handles 2"):
            apply_legend(ax, custom_labels=["Only One"])


class TestAutoRotateCategoricalXTicks:
    """Pin the priority sequence: try 0°, wrap multi-word, then 45°, then 90° as a last resort.

    Pandas' ``df.plot(kind="bar")`` produces a ``FixedLocator`` x-axis with
    rotation=90°, which is exactly what the helper expects. The figsize values
    are chosen empirically to drive each branch of the decision tree
    deterministically given matplotlib's default font metrics.
    """

    @pytest.fixture(autouse=True)
    def _close_all_figures(self):
        """Ensure no figures leak across tests in this class."""
        yield
        plt.close("all")

    @staticmethod
    def _categorical_bar(labels, figsize):
        df = pd.DataFrame({"cat": labels, "val": [1] * len(labels)})
        ax = df.plot(kind="bar", x="cat", y="val", figsize=figsize, legend=False)
        return ax.figure, ax

    def test_zero_rotation_when_short_labels_fit(self):
        """Short labels in a wide figure stay horizontal."""
        _, ax = self._categorical_bar(["Tea", "Jam", "Pie", "Ham"], figsize=(10, 4))
        _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        rendered = [label.get_text() for label in ax.get_xticklabels()]
        assert rotations == {0.0}
        assert all("\n" not in text for text in rendered)

    def test_wraps_multi_word_labels_before_rotating(self):
        """Multi-word labels that overflow at 0° wrap onto two lines and stay horizontal."""
        labels = ["North Region", "South Region", "East Region", "West Region"]
        _, ax = self._categorical_bar(labels, figsize=(4.0, 4))
        _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        rendered = [label.get_text() for label in ax.get_xticklabels()]
        assert rotations == {0.0}
        assert all("\n" in text for text in rendered)
        assert isinstance(ax.xaxis.get_major_formatter(), FixedFormatter)

    def test_rotates_single_word_labels_when_zero_overlaps(self):
        """Single-word labels can't wrap; they rotate to 45° when 0° overlaps."""
        labels = [f"Region{i}" for i in range(4)]
        _, ax = self._categorical_bar(labels, figsize=(3.0, 4))
        _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        rendered = [label.get_text() for label in ax.get_xticklabels()]
        assert rotations == {45.0}
        assert all("\n" not in text for text in rendered)

    def test_falls_back_to_ninety_when_wrap_and_forty_five_overflow(self):
        """When wrapping doesn't fit and 45° still overlaps, rotate to 90° and restore original labels."""
        labels = [
            "North West Region",
            "South East Region",
            "East Midlands Area",
            "West Midlands Area",
            "Greater London Area",
        ]
        _, ax = self._categorical_bar(labels, figsize=(2.0, 4))
        _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        rendered = [label.get_text() for label in ax.get_xticklabels()]
        assert rotations == {90.0}
        # Wrap attempt must be reverted: original (unwrapped) labels are visible.
        assert rendered == labels

    def test_auto_wrap_disabled_skips_wrapping(self):
        """``auto_wrap_x_ticks=False`` forces rotation even for wrappable multi-word labels."""
        labels = ["North Region", "South Region", "East Region", "West Region"]
        _, ax = self._categorical_bar(labels, figsize=(4.0, 4))

        with option_context("plot.style.auto_wrap_x_ticks", False):
            _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        rendered = [label.get_text() for label in ax.get_xticklabels()]
        assert rotations != {0.0}
        assert all("\n" not in text for text in rendered)

    def test_auto_rotate_disabled_leaves_rotation_untouched(self):
        """``auto_rotate_x_ticks=False`` returns early; pandas' default 90° rotation stays."""
        labels = ["Tea", "Jam", "Pie", "Ham"]
        _, ax = self._categorical_bar(labels, figsize=(10, 4))

        with option_context("plot.style.auto_rotate_x_ticks", False):
            _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        assert rotations == {90.0}

    def test_skips_when_locator_is_not_fixed(self):
        """Non-categorical axes (numeric AutoLocator) are left alone."""
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([1, 2, 3, 4], [10, 20, 30, 40])
        fig.canvas.draw()
        before = [label.get_rotation() for label in ax.get_xticklabels()]

        _auto_rotate_categorical_x_ticks(ax)

        after = [label.get_rotation() for label in ax.get_xticklabels()]
        assert after == before

    def test_respects_explicit_non_default_rotation(self):
        """Pre-set rotations outside {0°, 90°} (e.g. heatmap's 45°) are preserved."""
        labels = ["North Region", "South Region", "East Region", "West Region"]
        _, ax = self._categorical_bar(labels, figsize=(4.0, 4))
        for label in ax.get_xticklabels():
            label.set_rotation(45)

        _auto_rotate_categorical_x_ticks(ax)

        rotations = {label.get_rotation() for label in ax.get_xticklabels()}
        assert rotations == {45.0}
