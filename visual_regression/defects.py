"""Catalogue of injectable visual regressions.

Each :class:`Defect` mutates an already-rendered ``openretailscience`` figure so that one specific thing
looks wrong: a title that runs off the canvas, tick labels that overlap, a legend dumped on top of the
data, and so on. The mutation happens *after* the chrome layout engine has run and been frozen (see
``visual_regression.generate``), so the breakage survives ``savefig``.

The ``description`` strings are the human-readable ground truth: they are embedded in the dataset
manifest and shown to the detector as the closed set of categories it may report.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.text import Text
    from numpy.random import Generator


# --------------------------------------------------------------------------------------------------
# Chrome-text locators. Title / eyebrow / subtitle / source are figure-level texts added by the
# OpenRetailScience chrome engine. We identify them structurally (no reliance on the exact strings):
# header elements anchor their top edge, the source anchors its bottom edge, and the title is simply
# the largest header by font size. Positions are only meaningful after a draw, which the generator
# guarantees before any mutation runs.
# --------------------------------------------------------------------------------------------------


def _header_texts(fig: Figure) -> list[Text]:
    """Return the top-anchored chrome header texts (eyebrow, title, subtitle)."""
    return [t for t in fig.texts if t.get_va() == "top"]


def _find_title(fig: Figure) -> Text:
    """Return the title artist — the header text rendered in the largest font."""
    headers = _header_texts(fig)
    if len(headers) == 0:
        raise ValueError("No chrome header text found; build the chart with a title before injecting defects.")
    return max(headers, key=lambda t: t.get_fontsize())


def _find_subtitle(fig: Figure) -> Text:
    """Return the subtitle artist — the lowest header text (the title sits above it)."""
    headers = _header_texts(fig)
    if len(headers) < 2:
        raise ValueError("No subtitle text found; build the chart with a subtitle before injecting defects.")
    return min(headers, key=lambda t: t.get_position()[1])


def _find_source(fig: Figure) -> Text:
    """Return the bottom-anchored source/footnote artist."""
    sources = [t for t in fig.texts if t.get_va() == "bottom"]
    if len(sources) == 0:
        raise ValueError("No source text found; build the chart with source_text before injecting defects.")
    return sources[0]


def _bar_extents(ax: Axes) -> list[float]:
    """Return each bar's value extent (height for vertical bars, width for horizontal)."""
    extents = [max(p.get_height(), p.get_width()) for p in ax.patches]
    return [e for e in extents if e > 0]


# --------------------------------------------------------------------------------------------------
# Mutators. Each takes (fig, ax, rng) and breaks exactly one visual property in place.
# --------------------------------------------------------------------------------------------------


def _title_clipped(fig: Figure, ax: Axes, rng: Generator) -> None:
    title = _find_title(fig)
    title.set_ha("left")
    title.set_x(float(rng.uniform(0.74, 0.84)))


def _title_overlaps_plot(fig: Figure, ax: Axes, rng: Generator) -> None:
    title = _find_title(fig)
    title.set_y(float(rng.uniform(0.45, 0.6)))


def _title_misaligned(fig: Figure, ax: Axes, rng: Generator) -> None:
    title = _find_title(fig)
    title.set_ha("center")
    title.set_x(0.5)


def _title_tiny_font(fig: Figure, ax: Axes, rng: Generator) -> None:
    _find_title(fig).set_fontsize(6)


def _title_oversized_font(fig: Figure, ax: Axes, rng: Generator) -> None:
    _find_title(fig).set_fontsize(40)


def _title_rotated(fig: Figure, ax: Axes, rng: Generator) -> None:
    _find_title(fig).set_rotation(float(rng.uniform(12, 22)))


def _title_low_contrast(fig: Figure, ax: Axes, rng: Generator) -> None:
    _find_title(fig).set_color("#ededed")


def _subtitle_overlaps_title(fig: Figure, ax: Axes, rng: Generator) -> None:
    title = _find_title(fig)
    _find_subtitle(fig).set_y(title.get_position()[1])


def _source_clipped(fig: Figure, ax: Axes, rng: Generator) -> None:
    _find_source(fig).set_y(-0.01)


def _xlabel_excessive_gap(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.xaxis.labelpad = 34


def _ylabel_overlaps_ticks(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.yaxis.labelpad = -32


def _xticks_overlap(fig: Figure, ax: Axes, rng: Generator) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(0)
        label.set_fontsize(13)
        label.set_ha("center")


def _xticks_rotated_clipped(fig: Figure, ax: Axes, rng: Generator) -> None:
    for label in ax.get_xticklabels():
        label.set_rotation(55)
        label.set_fontsize(12)
        label.set_ha("right")


def _yticks_clipped_left(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"$ {v:,.2f} per active store unit"))


def _yaxis_truncated_baseline(fig: Figure, ax: Axes, rng: Generator) -> None:
    extents = _bar_extents(ax)
    ax.set_ylim(bottom=min(extents) * 0.96, top=max(extents) * 1.02)


def _gridlines_over_data(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.set_axisbelow(False)
    ax.grid(visible=True, which="major", color="#555555", linewidth=2.0, alpha=0.9)


def _distorted_aspect(fig: Figure, ax: Axes, rng: Generator) -> None:
    fig.set_size_inches(11.5, 2.3)


def _legend_overlaps_data(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.legend(loc="center")


def _legend_offscreen(fig: Figure, ax: Axes, rng: Generator) -> None:
    ax.legend(loc="center left", bbox_to_anchor=(0.98, 0.5))


def _data_labels_overlap(fig: Figure, ax: Axes, rng: Generator) -> None:
    for text in ax.texts:
        text.set_fontsize(26)


def _data_label_clipped_top(fig: Figure, ax: Axes, rng: Generator) -> None:
    extents = _bar_extents(ax)
    ax.set_ylim(top=max(extents) * 0.72)


@dataclass(frozen=True)
class Defect:
    """A single injectable visual regression.

    Attributes:
        name: Stable category id used in the manifest, the detector prompt, and scoring.
        description: Human-readable ground truth shown to the detector.
        severity: Rough impact band — ``"high"``, ``"medium"`` or ``"low"``.
        requires: Structural tags a chart must carry for this defect to apply.
        apply: Mutator that breaks the figure in place.
    """

    name: str
    description: str
    severity: str
    requires: frozenset[str]
    apply: Callable[[Figure, Axes, Generator], None]


DEFECTS: tuple[Defect, ...] = (
    Defect(
        "title_clipped",
        "The chart title runs off the right edge of the figure and is cut off.",
        "high",
        frozenset({"chrome"}),
        _title_clipped,
    ),
    Defect(
        "title_overlaps_plot",
        "The title sits on top of the plotted data instead of above the chart.",
        "high",
        frozenset({"chrome"}),
        _title_overlaps_plot,
    ),
    Defect(
        "title_misaligned",
        "The title is centred over the figure instead of left-aligned with the chart, so it looks misaligned.",
        "medium",
        frozenset({"chrome"}),
        _title_misaligned,
    ),
    Defect(
        "title_tiny_font",
        "The chart title is rendered in a tiny, barely readable font.",
        "high",
        frozenset({"chrome"}),
        _title_tiny_font,
    ),
    Defect(
        "title_oversized_font",
        "The title font is so large the text overflows the width of the figure.",
        "high",
        frozenset({"chrome"}),
        _title_oversized_font,
    ),
    Defect(
        "title_rotated",
        "The chart title is rendered at a slant instead of horizontal.",
        "medium",
        frozenset({"chrome"}),
        _title_rotated,
    ),
    Defect(
        "title_low_contrast",
        "The title text colour is almost the same as the white background, so it is barely visible.",
        "medium",
        frozenset({"chrome"}),
        _title_low_contrast,
    ),
    Defect(
        "subtitle_overlaps_title",
        "The subtitle overlaps the title text.",
        "high",
        frozenset({"chrome"}),
        _subtitle_overlaps_title,
    ),
    Defect(
        "source_text_clipped",
        "The source/footnote text is cut off at the bottom edge of the figure.",
        "medium",
        frozenset({"chrome"}),
        _source_clipped,
    ),
    Defect(
        "xlabel_excessive_gap",
        "There is an unusually large gap between the x-axis tick labels and the x-axis title.",
        "medium",
        frozenset({"chrome"}),
        _xlabel_excessive_gap,
    ),
    Defect(
        "ylabel_overlaps_ticks",
        "The y-axis title overlaps the y-axis tick labels.",
        "medium",
        frozenset({"chrome"}),
        _ylabel_overlaps_ticks,
    ),
    Defect(
        "xticks_overlap",
        "The x-axis tick labels overlap each other and are unreadable.",
        "high",
        frozenset({"categorical_x"}),
        _xticks_overlap,
    ),
    Defect(
        "xticks_rotated_clipped",
        "Rotated x-axis tick labels are cut off at the bottom edge of the figure.",
        "high",
        frozenset({"categorical_x"}),
        _xticks_rotated_clipped,
    ),
    Defect(
        "yticks_clipped_left",
        "The y-axis tick labels are too long and are cut off at the left edge of the figure.",
        "high",
        frozenset({"numeric_y"}),
        _yticks_clipped_left,
    ),
    Defect(
        "yaxis_truncated_baseline",
        "The y-axis does not start at zero, exaggerating the differences between bars.",
        "medium",
        frozenset({"vbars"}),
        _yaxis_truncated_baseline,
    ),
    Defect(
        "gridlines_over_data",
        "Heavy gridlines are drawn on top of the data, obscuring it.",
        "low",
        frozenset(),
        _gridlines_over_data,
    ),
    Defect(
        "distorted_aspect_ratio",
        "The figure has a distorted, squashed aspect ratio.",
        "medium",
        frozenset(),
        _distorted_aspect,
    ),
    Defect(
        "legend_overlaps_data",
        "The legend is placed over the centre of the plot and obscures the data.",
        "high",
        frozenset({"legend"}),
        _legend_overlaps_data,
    ),
    Defect(
        "legend_offscreen",
        "The legend is positioned outside the figure and is cut off.",
        "high",
        frozenset({"legend"}),
        _legend_offscreen,
    ),
    Defect(
        "data_labels_overlap",
        "The data value labels are too large and overlap each other.",
        "medium",
        frozenset({"data_labels"}),
        _data_labels_overlap,
    ),
    Defect(
        "data_label_clipped_top",
        "The tallest bars and their data labels are cut off at the top of the plot area.",
        "medium",
        frozenset({"data_labels"}),
        _data_label_clipped_top,
    ),
)

DEFECTS_BY_NAME: dict[str, Defect] = {d.name: d for d in DEFECTS}


def taxonomy() -> list[dict[str, str]]:
    """Return the defect catalogue as plain dicts (name/description/severity) for the manifest."""
    return [{"name": d.name, "description": d.description, "severity": d.severity} for d in DEFECTS]


def applicable_defects(tags: set[str]) -> list[Defect]:
    """Return the defects whose required structural tags are all present in ``tags``."""
    return [d for d in DEFECTS if d.requires <= tags]
