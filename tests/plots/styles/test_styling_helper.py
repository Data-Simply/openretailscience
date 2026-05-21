"""Tests for the styling_helpers module."""

import pandas as pd
import pytest
from matplotlib import pyplot as plt
from matplotlib.ticker import FixedFormatter

from openretailscience.options import get_option, option_context
from openretailscience.plots.styles.styling_helpers import (
    _auto_rotate_categorical_x_ticks,
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

    def test_apply_ticks_is_idempotent_when_zero_label_was_hidden(self, fig_ax):
        """Re-applying ticks after the zero label was hidden must not raise.

        The first call hides the zero tick's label, which makes
        get_majorticklabels() one shorter than get_majorticklocs() (it filters
        invisible labels). A second apply_ticks call would then crash on
        strict-zip if the helper still paired locs against labels.
        """
        _, ax = fig_ax
        ax.plot([-2, -1, 0, 1, 2], [1, 2, 0, 2, 1])

        apply_ticks(ax)
        apply_ticks(ax)

        for axis in (ax.xaxis, ax.yaxis):
            ticks = axis.get_major_ticks()
            locs = axis.get_majorticklocs()
            zero_ticks = [tick for tick, loc in zip(ticks, locs, strict=True) if loc == 0]
            assert len(zero_ticks) == 1
            for tick in zero_ticks:
                assert not tick.label1.get_visible()

    def test_apply_ticks_handles_labeltop_on_numeric_axis(self, fig_ax):
        """tick_params(labeltop=True) doubles get_majorticklabels() length.

        With both label1 and label2 visible on a numeric MaxNLocator/AutoLocator
        axis, get_majorticklabels() returns 2N entries while get_majorticklocs()
        returns N. apply_ticks must still complete without raising, and both
        the bottom and top label artists at zero must be hidden.
        """
        _, ax = fig_ax
        ax.plot([-2, -1, 0, 1, 2], [1, 2, 0, 2, 1])
        ax.tick_params(labeltop=True, labelbottom=True)

        apply_ticks(ax)

        ticks = ax.xaxis.get_major_ticks()
        locs = ax.xaxis.get_majorticklocs()
        zero_ticks = [tick for tick, loc in zip(ticks, locs, strict=True) if loc == 0]
        assert len(zero_ticks) == 1
        for tick in zero_ticks:
            assert not tick.label1.get_visible()
            assert not tick.label2.get_visible()

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
