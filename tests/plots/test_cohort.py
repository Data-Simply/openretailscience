"""Tests for the cohort plot module."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from openretailscience.plots import cohort


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_cohort_dataframe():
    """Generates a sample cohort DataFrame."""
    rng = np.random.default_rng(42)
    data = np.round(rng.uniform(0, 1, size=(6, 6)), 2)
    return pd.DataFrame(
        data,
        columns=pd.Index([f"Month {i + 1}" for i in range(6)]),
        index=pd.Index([f"Cohort {i + 1}" for i in range(6)]),
    )


def test_plot_cohort(sample_cohort_dataframe):
    """The cohort plot renders one annotation per cell and honours x_label/y_label/title."""
    title = "Cohort Retention Heatmap"
    x_label = "Months"
    y_label = "Cohorts"

    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        x_label=x_label,
        y_label=y_label,
        title=title,
    )

    assert isinstance(result_ax, Axes)
    assert len(result_ax.texts) == sample_cohort_dataframe.size
    assert result_ax.get_xlabel() == x_label
    assert result_ax.get_ylabel() == y_label
    title_texts = [t for t in result_ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1


def test_plot_cohort_with_source_text(sample_cohort_dataframe):
    """The cohort plot renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"

    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        source_text=source_text,
    )

    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_plot_cohort_with_figsize(sample_cohort_dataframe):
    """Test cohort plot with a specified figsize."""
    width = 14
    height = 10
    result_ax = cohort.plot(
        df=sample_cohort_dataframe,
        cbar_label="Retention Rate",
        figsize=(width, height),
    )

    assert isinstance(result_ax, Axes)
    figure = result_ax.figure
    assert isinstance(figure, Figure)
    assert figure.get_size_inches()[0] == width
    assert figure.get_size_inches()[1] == height


@pytest.mark.parametrize("percentage", [True, False])
def test_plot_cohort_percentage_formatting(percentage):
    """Test cohort plot percentage formatting comprehensively."""
    data = np.array([[0.5, 0.3], [0.8, 0.6]])
    df = pd.DataFrame(data, columns=pd.Index(["Month 1", "Month 2"]), index=pd.Index(["Cohort A", "Cohort B"]))

    result_ax = cohort.plot(df=df, cbar_label="Retention Rate", percentage=percentage)

    assert len(result_ax.texts) == df.size
    texts = result_ax.texts

    text_values = [text.get_text() for text in texts]
    if percentage:
        assert all("%" in val and not val.startswith("0.") for val in text_values), (
            f"Expected every cell in proper percentage format, got: {text_values}"
        )
    else:
        assert all("%" not in val for val in text_values), "Should not have percentage formatting"
