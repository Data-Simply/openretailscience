"""Base chart builders for the visual-regression benchmark.

Every builder renders a *clean*, realistic retail chart using the real ``openretailscience.plots`` API
and returns ``(fig, ax, tags)``. ``tags`` advertise the structural features a chart has (bars, legend,
categorical x-axis, ...) so the defect catalogue can pick only the regressions that make sense for it.

Defects are injected later by ``visual_regression.defects`` by mutating the returned figure, so these
builders deliberately keep every chart on the editorial "chrome" path (eyebrow / title / subtitle /
source / axis labels) to give the mutators something to break.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib as mpl

mpl.use("Agg")  # headless rendering — never opens a window

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from openretailscience.plots import area, bar, histogram, line, scatter

if TYPE_CHECKING:
    from collections.abc import Callable

    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from numpy.random import Generator

# A single figure size keeps the dataset visually consistent so a detector can't cheat off the canvas
# shape. Roughly 900x550 px at the default 110 dpi — comfortably readable for a vision model.
FIGSIZE = (8.2, 5.0)

# Nine grocery departments give the categorical bar charts enough crowded, medium-length labels that the
# "x-tick labels overlap" defect produces a genuinely unreadable axis.
_DEPARTMENTS = (
    "Fresh Produce",
    "Dairy & Eggs",
    "Bakery",
    "Frozen Foods",
    "Beverages",
    "Snacks & Confectionery",
    "Household",
    "Personal Care",
    "Pet Supplies",
)
_MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_STORES = ("North Mall", "Southgate", "East Retail Park")
_SOURCES = (
    "Source: Internal POS data, 2026",
    "Source: Loyalty programme extract, Q1 2026",
    "Source: Company financials, FY2026",
)


@dataclass(frozen=True)
class Builder:
    """A named clean-chart builder.

    Attributes:
        name: Stable identifier recorded as ``chart_type`` in the manifest.
        build: Callable taking a seeded ``Generator`` and returning ``(fig, ax, tags)``.
    """

    name: str
    build: Callable[[Generator], tuple[Figure, Axes, set[str]]]


def _new_axes() -> tuple[Figure, Axes]:
    """Create a fresh figure/axes pair at the benchmark's standard size."""
    return plt.subplots(figsize=FIGSIZE)


def _pick(rng: Generator, choices: tuple[str, ...]) -> str:
    """Return one randomly chosen string from ``choices``."""
    return str(rng.choice(np.asarray(choices)))


def _department_sales(rng: Generator) -> pd.DataFrame:
    """Build a department-level sales frame with randomised but realistic unit totals."""
    units = rng.integers(4_000, 24_000, size=len(_DEPARTMENTS))
    return pd.DataFrame({"department": list(_DEPARTMENTS), "units_sold": units})


def build_vertical_bar(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A single-series vertical bar chart of units sold by department."""
    df = _department_sales(rng)
    fig, ax = _new_axes()
    bar.plot(
        df,
        value_col="units_sold",
        x_col="department",
        ax=ax,
        orientation="v",
        eyebrow="Category performance",
        title="Units Sold by Grocery Department",
        subtitle="Total units moved across the core retail departments this quarter",
        x_label="Department",
        y_label="Units sold",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "bars", "vbars", "categorical_x", "numeric_y"}


def build_horizontal_bar(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A single-series horizontal bar chart (categorical y-axis)."""
    df = _department_sales(rng)
    fig, ax = _new_axes()
    bar.plot(
        df,
        value_col="units_sold",
        x_col="department",
        ax=ax,
        orientation="h",
        eyebrow="Category performance",
        title="Units Sold by Grocery Department",
        subtitle="Departments ranked by total units sold this quarter",
        x_label="Units sold",
        y_label="Department",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "bars"}


def build_grouped_bar(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A two-series grouped bar chart comparing two quarters; carries a legend."""
    df = _department_sales(rng).rename(columns={"units_sold": "q1_units"})
    df["q2_units"] = (df["q1_units"] * rng.uniform(0.85, 1.25, size=len(df))).astype(int)
    fig, ax = _new_axes()
    bar.plot(
        df,
        value_col=["q1_units", "q2_units"],
        x_col="department",
        ax=ax,
        orientation="v",
        legend_labels=["Q1", "Q2"],
        legend_title="Quarter",
        eyebrow="Quarter on quarter",
        title="Quarterly Units Sold by Department",
        subtitle="Comparing Q1 and Q2 unit volumes across departments",
        x_label="Department",
        y_label="Units sold",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "bars", "vbars", "categorical_x", "numeric_y", "legend"}


def build_bar_with_labels(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A vertical bar chart with absolute data labels on every bar."""
    df = _department_sales(rng)
    fig, ax = _new_axes()
    bar.plot(
        df,
        value_col="units_sold",
        x_col="department",
        ax=ax,
        orientation="v",
        data_label_format="absolute",
        eyebrow="Category performance",
        title="Units Sold by Grocery Department",
        subtitle="Labelled unit totals for each core department",
        x_label="Department",
        y_label="Units sold",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "bars", "vbars", "categorical_x", "numeric_y", "data_labels"}


def build_line(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A single-series monthly revenue line chart."""
    revenue = np.cumsum(rng.uniform(4, 12, size=len(_MONTHS))) + rng.uniform(40, 60)
    df = pd.DataFrame({"month": list(_MONTHS), "revenue": revenue})
    fig, ax = _new_axes()
    line.plot(
        df,
        value_col="revenue",
        x_col="month",
        ax=ax,
        eyebrow="Trading trend",
        title="Monthly Net Revenue",
        subtitle="Net revenue trajectory across the trading year",
        x_label="Month",
        y_label="Revenue ($m)",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "numeric_y"}


def build_multi_line(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A multi-series monthly revenue line chart, one line per store; carries a legend."""
    rows = []
    for store in _STORES:
        base = rng.uniform(45, 80)
        for i, month in enumerate(_MONTHS):
            rows.append({"month": month, "revenue": base + i * rng.uniform(1, 4) + rng.normal(0, 3), "store": store})
    df = pd.DataFrame(rows)
    fig, ax = _new_axes()
    line.plot(
        df,
        value_col="revenue",
        x_col="month",
        group_col="store",
        ax=ax,
        legend_title="Store",
        eyebrow="Trading trend",
        title="Monthly Revenue by Store",
        subtitle="Net revenue trajectory for each store across the trading year",
        x_label="Month",
        y_label="Revenue ($m)",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "numeric_y", "legend"}


def build_scatter(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A price-vs-units scatter coloured by store; carries a legend."""
    n = 90
    store = rng.choice(np.asarray(_STORES), size=n)
    price = rng.uniform(2, 40, size=n)
    units = np.clip(60 - price * 0.9 + rng.normal(0, 6, size=n), 1, None)
    df = pd.DataFrame({"price": price, "units": units, "store": store})
    fig, ax = _new_axes()
    scatter.plot(
        df,
        value_col="units",
        x_col="price",
        group_col="store",
        ax=ax,
        legend_title="Store",
        eyebrow="Price elasticity",
        title="Units Sold Against Unit Price",
        subtitle="Each point is a product, coloured by the store that sold it",
        x_label="Unit price ($)",
        y_label="Units sold",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "numeric_y", "legend"}


def build_histogram(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A histogram of basket values."""
    df = pd.DataFrame({"basket_value": rng.gamma(shape=2.0, scale=18.0, size=600)})
    fig, ax = _new_axes()
    histogram.plot(
        df,
        value_col="basket_value",
        ax=ax,
        eyebrow="Basket analysis",
        title="Distribution of Basket Values",
        subtitle="How much customers spend per shopping trip",
        x_label="Basket value ($)",
        y_label="Number of baskets",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "numeric_y"}


def build_area(rng: Generator) -> tuple[Figure, Axes, set[str]]:
    """A stacked area chart of revenue by category over time; carries a legend."""
    df = pd.DataFrame({"month": list(_MONTHS)})
    for category in ("Grocery", "Household", "Apparel"):
        df[category] = np.cumsum(rng.uniform(1, 5, size=len(_MONTHS))) + rng.uniform(10, 30)
    fig, ax = _new_axes()
    area.plot(
        df,
        value_col=["Grocery", "Household", "Apparel"],
        x_col="month",
        ax=ax,
        legend_title="Category",
        eyebrow="Revenue mix",
        title="Revenue by Category Over Time",
        subtitle="Stacked monthly revenue contribution by product category",
        x_label="Month",
        y_label="Revenue ($m)",
        source_text=_pick(rng, _SOURCES),
    )
    return fig, ax, {"chrome", "numeric_y", "legend"}


BUILDERS: tuple[Builder, ...] = (
    Builder("vertical_bar", build_vertical_bar),
    Builder("horizontal_bar", build_horizontal_bar),
    Builder("grouped_bar", build_grouped_bar),
    Builder("bar_with_labels", build_bar_with_labels),
    Builder("line", build_line),
    Builder("multi_line", build_multi_line),
    Builder("scatter", build_scatter),
    Builder("histogram", build_histogram),
    Builder("area", build_area),
)

BUILDERS_BY_NAME: dict[str, Builder] = {b.name: b for b in BUILDERS}
