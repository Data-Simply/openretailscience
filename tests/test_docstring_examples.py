"""Every public entry point carries a usage example in its docstring, and the example runs.

An agent without the bundled skill learns the call from `help()`.
"""

from __future__ import annotations

import doctest
import importlib
import inspect
from typing import TYPE_CHECKING

import ibis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from openretailscience import analysis, plots, segmentation, utils
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

PRODUCT_COL = "product"
CATEGORY_COL = "category"
BRAND_COL = "brand"
PERIOD_COL = "period"
NUM_CUSTOMERS = 120
NUM_TRANSACTIONS = 300
NUM_LINES = 800
STORE_IDS = [101, 102, 103]
FIRST_DATE = pd.Timestamp("2024-01-01")
NUM_DAYS = 547
MAX_UNITS = 4
# Every other module in plots.__all__ is a chart, so one without plot() fails collection instead of going unchecked
NON_CHART_PLOT_MODULES = {"styles", "tree_diagram"}
# Pandas' notebook defaults: outside IPython the repr fits the terminal, so shown frames fail below ~65 columns
PANDAS_DISPLAY_OPTIONS = ("display.max_columns", 20, "display.width", 80)
CATALOGUE = pd.DataFrame(
    {
        PRODUCT_COL: [
            "Whole Milk 2L",
            "Sourdough Loaf",
            "Salted Butter 250g",
            "Free Range Eggs x6",
            "Ground Coffee 227g",
            "Earl Grey Tea x80",
            "Cola 2L",
            "Sparkling Water 1L",
        ],
        CATEGORY_COL: ["Dairy", "Bakery", "Dairy", "Dairy", "Hot Drinks", "Hot Drinks", "Soft Drinks", "Soft Drinks"],
        BRAND_COL: ["Meadow Farm", "Greenvale"] * 4,
        "unit_price": [1.65, 2.80, 2.10, 2.45, 4.50, 3.20, 2.00, 0.85],
    },
)


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Transaction lines from January 2024 to June 2025, labelled with their quarter (e.g. "2025Q1")."""
    cols = ColumnHelper()
    rng = np.random.default_rng(42)
    dates = FIRST_DATE + pd.to_timedelta(np.sort(rng.integers(0, NUM_DAYS, NUM_TRANSACTIONS)), unit="D")
    baskets = pd.DataFrame(
        {
            cols.transaction_id: np.arange(1, NUM_TRANSACTIONS + 1),
            cols.customer_id: rng.integers(1, NUM_CUSTOMERS + 1, NUM_TRANSACTIONS),
            cols.store_id: rng.choice(STORE_IDS, NUM_TRANSACTIONS),
            cols.transaction_date: dates,
            PERIOD_COL: dates.to_period("Q").astype(str),
        },
    )
    lines = pd.concat(
        [
            baskets.iloc[rng.integers(0, NUM_TRANSACTIONS, NUM_LINES)].reset_index(drop=True),
            CATALOGUE.iloc[rng.integers(0, len(CATALOGUE), NUM_LINES)].reset_index(drop=True),
        ],
        axis=1,
    )
    lines[cols.unit_qty] = rng.integers(1, MAX_UNITS + 1, NUM_LINES)
    lines[cols.unit_spend] = (lines["unit_price"] * lines[cols.unit_qty]).round(2)
    return (
        lines.drop_duplicates([cols.transaction_id, PRODUCT_COL])
        .sort_values([cols.transaction_date, cols.transaction_id])
        .reset_index(drop=True)
    )


def _entry_points() -> dict[str, Callable]:
    """Map each public entry point's dotted name to it: subpackage exports, and each chart module's `plot()`."""
    entry_points = {
        f"{package.__name__}.{name}": getattr(package, name)
        for package in (analysis, segmentation, utils)
        for name in package.__all__
    }
    for name in plots.__all__:
        if name in NON_CHART_PLOT_MODULES:
            continue
        module = importlib.import_module(f"{plots.__name__}.{name}")
        entry_points[f"{module.__name__}.plot"] = module.plot
    return entry_points


ENTRY_POINTS = _entry_points()


class TestDocstringExamples:
    """Tests for the usage example in each public entry point's docstring."""

    @pytest.fixture(autouse=True)
    def _close_figures(self) -> Iterator[None]:
        """Close the figures plotting examples draw, so they don't pile up across the run."""
        yield
        plt.close("all")

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_example_runs(self, name: str, transactions: pd.DataFrame) -> None:
        """The docstring has an `Examples:` section whose `>>>` lines run and print what they show.

        For a class this is the class docstring, not `__init__`'s: `Class.__doc__` returns only the former.
        """
        # Not inspect.getdoc: it falls back to a base class's docstring, so a subclass passes on its parent's example
        docstring = inspect.cleandoc(ENTRY_POINTS[name].__doc__ or "")
        _, has_examples, examples = docstring.partition("\nExamples:\n")
        assert has_examples, f"{name} has no Examples: section"

        globs = {"pd": pd, "ibis": ibis, "transactions": transactions}
        test = doctest.DocTestParser().get_doctest(examples, globs, name, filename=None, lineno=0)
        assert len(test.examples) > 0, f"{name} has no >>> example under Examples:"

        report: list[str] = []
        # verbose=False: the default reads "-v" from sys.argv and would bury the failure under every passing example
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.ELLIPSIS | doctest.NORMALIZE_WHITESPACE)
        with pd.option_context(*PANDAS_DISPLAY_OPTIONS):
            runner.run(test, out=report.append)
        assert runner.failures == 0, "".join(report)
