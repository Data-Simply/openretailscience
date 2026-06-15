"""Tests for the defect catalogue and chrome-text locators."""

import io

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

import matplotlib.pyplot as plt
from visual_regression.defects import DEFECTS, DEFECTS_BY_NAME, _find_subtitle, _find_title, applicable_defects
from visual_regression.retail_plots import BUILDERS, build_grouped_bar, build_vertical_bar

# Title pushed to/over the right edge sits at x >= this fraction of the figure width.
_RIGHT_MARGIN_X = 0.7


@pytest.fixture(autouse=True)
def _cleanup_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


def _first_compatible_builder(requires):
    """Return the first builder whose tags satisfy a defect's requirements."""
    for builder in BUILDERS:
        _fig, _ax, tags = builder.build(np.random.default_rng(0))
        plt.close("all")
        if requires <= tags:
            return builder
    msg = f"no builder satisfies {requires}"
    raise AssertionError(msg)


def _render_bytes(builder, defect):
    """Render ``builder`` to PNG bytes, optionally injecting ``defect`` (mirrors generate._render)."""
    fig, ax, _tags = builder.build(np.random.default_rng(123))
    buffer = io.BytesIO()
    if defect is not None:
        fig.canvas.draw()
        fig.set_layout_engine("none")
        defect.apply(fig, ax, np.random.default_rng(123))
    fig.savefig(buffer, dpi=80)
    plt.close(fig)
    return buffer.getvalue()


class TestDefectInjection:
    """Each defect must visibly change the rendered chart."""

    @pytest.mark.parametrize("defect", DEFECTS, ids=[d.name for d in DEFECTS])
    def test_defect_changes_rendered_image(self, defect):
        """Applying a defect alters the PNG output relative to the clean baseline."""
        builder = _first_compatible_builder(defect.requires)
        clean = _render_bytes(builder, defect=None)
        broken = _render_bytes(builder, defect=defect)
        assert clean != broken

    def test_applicable_defects_filters_by_tags(self):
        """Legend/data-label defects only apply to charts that actually have those features."""
        single_bar_tags = {"chrome", "bars", "vbars", "categorical_x", "numeric_y"}
        names = {d.name for d in applicable_defects(single_bar_tags)}
        assert "yaxis_truncated_baseline" in names  # needs vertical bars
        assert "legend_overlaps_data" not in names  # single series has no legend
        assert "data_labels_overlap" not in names  # no data labels on this chart


class TestChromeLocators:
    """The structural chrome-text locators find the right artists."""

    def test_find_title_returns_largest_header(self):
        """The title is identified as the largest-font header text."""
        fig, _ax, _tags = build_vertical_bar(np.random.default_rng(1))
        fig.canvas.draw()
        title = _find_title(fig)
        other_headers = [t for t in fig.texts if t.get_va() == "top" and t is not title]
        assert all(title.get_fontsize() > h.get_fontsize() for h in other_headers)

    def test_find_subtitle_sits_below_title(self):
        """The subtitle artist is positioned below the title."""
        fig, _ax, _tags = build_vertical_bar(np.random.default_rng(2))
        fig.canvas.draw()
        title = _find_title(fig)
        subtitle = _find_subtitle(fig)
        assert subtitle.get_position()[1] < title.get_position()[1]


class TestSpecificMutations:
    """A couple of defects checked at the artist level, not just 'image changed'."""

    def test_title_clipped_pushes_title_toward_right_edge(self):
        """The title-clipped defect left-aligns the title and moves it into the right margin."""
        fig, ax, _tags = build_vertical_bar(np.random.default_rng(3))
        fig.canvas.draw()
        fig.set_layout_engine("none")
        title = _find_title(fig)
        DEFECTS_BY_NAME["title_clipped"].apply(fig, ax, np.random.default_rng(3))
        assert title.get_ha() == "left"
        assert title.get_position()[0] >= _RIGHT_MARGIN_X

    def test_legend_overlaps_data_recentres_the_legend(self):
        """The legend-overlap defect moves the legend inside the axes box."""
        fig, ax, _tags = build_grouped_bar(np.random.default_rng(4))
        fig.canvas.draw()
        fig.set_layout_engine("none")
        DEFECTS_BY_NAME["legend_overlaps_data"].apply(fig, ax, np.random.default_rng(4))
        legend = ax.get_legend()
        assert legend is not None
        bbox = legend.get_window_extent()
        ax_bbox = ax.get_window_extent()
        assert ax_bbox.x0 < bbox.x0
        assert bbox.x1 < ax_bbox.x1
        assert ax_bbox.y0 < bbox.y0
        assert bbox.y1 < ax_bbox.y1
