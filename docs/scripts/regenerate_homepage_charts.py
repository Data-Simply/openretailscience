"""Regenerate the showcase charts on the docs homepage (``docs/overrides/home.html``).

The data is synthetic and sized so each title is literally true for the rendered chart; the chart alt text in
``home.html`` quotes these numbers, so update it alongside any data change.

Run from the repo root::

    uv run python docs/scripts/regenerate_homepage_charts.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

mpl.use("Agg")

from openretailscience.constants import COLORS
from openretailscience.options import option_context
from openretailscience.plots import cohort, venn, waterfall

OUT_DIR = Path(__file__).resolve().parent.parent / "assets" / "images" / "home"
FIGSIZE = (8, 5.4)
# Brand greens plus a neutral grey, so the overlaps never blend to purple. matplotlib_set_diagrams only picks black
# subset labels for set colours with grayscale > 0.5, so each colour stays light enough for black text.
VENN_PALETTE = ["#22c55e", COLORS["neutral"][400], "#bbf7d0"]


def _save(fig: plt.Figure, name: str) -> None:
    path = OUT_DIR / f"chart-{name}.svg"
    # No bbox_inches="tight": it shifts the ORS title chrome.
    fig.savefig(path, format="svg", metadata={"Date": None})
    plt.close(fig)
    print(f"wrote {path}")


def regenerate_waterfall() -> None:
    # 610 + 380 - 290 + 170 - 110 = 760 ($K)
    with option_context("plot.color.difference", COLORS["green"][900]):
        ax = waterfall.plot(
            amounts=[610, 380, -290, 170, -110],
            labels=["New customers", "Basket growth", "Lapsed customers", "Price", "Returns"],
            display_net_bar=True,
            data_label_format="absolute",
            eyebrow="Revenue bridge",
            title="New customers drive the $760K lift",
            subtitle="Year-on-year revenue change by driver, $K",
            y_label="Revenue change ($K)",
            figsize=FIGSIZE,
        )
    _save(ax.figure, "waterfall")


def regenerate_cohort() -> None:
    retention = pd.DataFrame(
        {
            "M1": [0.78, 0.74, 0.71, 0.68, 0.82, 0.75, 0.72, 0.70],
            "M2": [0.62, 0.58, 0.55, 0.52, 0.69, 0.60, 0.57, None],
            "M3": [0.51, 0.47, 0.44, 0.41, 0.57, 0.49, None, None],
            "M4": [0.43, 0.39, 0.36, 0.33, 0.46, None, None, None],
            "M5": [0.37, 0.34, 0.31, 0.28, None, None, None, None],
            "M6": [0.32, 0.30, 0.27, None, None, None, None, None],
            "M7": [0.29, 0.28, None, None, None, None, None, None],
            "M8": [0.27, None, None, None, None, None, None, None],
        },
        index=["Aug 24", "Sep 24", "Oct 24", "Nov 24", "Dec 24", "Jan 25", "Feb 25", "Mar 25"],
    )
    ax = cohort.plot(
        retention,
        y_col=None,
        cbar_label="Retention rate",
        x_label="Months since first purchase",
        y_label="Acquisition cohort",
        eyebrow="Retention",
        title="The December cohort retains best",
        subtitle="Share of each cohort still shopping, by month",
        figsize=FIGSIZE,
    )
    _save(ax.figure, "cohort")


def regenerate_venn() -> None:
    cross_shop = pd.DataFrame(
        {
            "groups": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1), (0, 0, 0)],
            "percent": [18, 14, 10, 12, 8, 9, 7, 22],
        },
    )
    with option_context("plot.color.multi_color_palette", VENN_PALETTE):
        ax = venn.plot(
            cross_shop,
            labels=["Fresh", "Bakery", "Health & Beauty"],
            eyebrow="Cross-shop",
            title="Fresh shoppers most often add Bakery",
            subtitle="Share of customers by category combination",
            subset_label_formatter=lambda value: f"{value}%",
            figsize=(6.4, 6),
        )
    _save(ax.figure, "venn")


def main() -> None:
    regenerate_waterfall()
    regenerate_cohort()
    regenerate_venn()


if __name__ == "__main__":
    main()
