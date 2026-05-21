"""Tests for the plots.venn module."""

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.axes import Axes

from openretailscience.options import get_option, option_context
from openretailscience.plots import venn


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Clean up matplotlib figures after each test."""
    yield
    plt.close("all")


@pytest.fixture
def sample_venn_dataframe():
    """A sample DataFrame for Venn diagram testing."""
    data = {
        "groups": [(1, 0), (0, 1), (1, 1)],
        "percent": [0.4, 0.3, 0.3],
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    ("df", "labels"),
    [
        (
            pd.DataFrame({"groups": [(1, 0), (0, 1), (1, 1)], "percent": [0.4, 0.3, 0.3]}),
            ["Set A", "Set B"],
        ),
        (
            pd.DataFrame(
                {
                    "groups": [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)],
                    "percent": [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1],
                },
            ),
            ["Set A", "Set B", "Set C"],
        ),
    ],
    ids=["two_set", "three_set"],
)
def test_plot_renders_each_set_label(df, labels):
    """Venn rendering produces one set-label text per requested set."""
    result_ax = venn.plot(df=df, labels=labels, title="Test Venn Diagram")
    rendered_set_labels = [t.get_text() for t in result_ax.texts if t.get_text() in set(labels)]
    assert sorted(rendered_set_labels) == sorted(labels)


def test_plot_invalid_sets():
    """Test Venn plot with invalid number of sets (should raise ValueError)."""
    df = pd.DataFrame({"groups": [(1,)], "percent": [1.0]})
    with pytest.raises(ValueError, match="Only 2-set or 3-set Venn diagrams are supported"):
        venn.plot(df=df, labels=["Set A"])


def test_plot_adds_source_text(sample_venn_dataframe):
    """The Venn diagram renders source_text as a figure-level text element."""
    source_text = "Source: Test Data"
    result_ax = venn.plot(
        df=sample_venn_dataframe,
        labels=["Set A", "Set B"],
        title="Test Venn Diagram with Source",
        source_text=source_text,
    )
    rendered = [t.get_text() for t in result_ax.figure.texts]
    assert source_text in rendered


def test_venn_default_ax(sample_venn_dataframe):
    """Test Venn diagram when ax is None to ensure a new figure is created."""
    result_ax = venn.plot(df=sample_venn_dataframe, labels=["A", "B"])
    assert isinstance(result_ax, Axes)


def test_subset_labels_use_data_label_size(sample_venn_dataframe):
    """Venn subset labels (numbers inside circles) are data values; size must track data_label_size."""
    custom_data_label_size = 17.0
    set_labels = ["Set A", "Set B"]
    with option_context("plot.font.data_label_size", custom_data_label_size):
        result_ax = venn.plot(df=sample_venn_dataframe, labels=set_labels)

    subset_label_texts = [t for t in result_ax.texts if t.get_text() and t.get_text() not in set_labels]
    assert len(subset_label_texts) == len(sample_venn_dataframe)
    for text in subset_label_texts:
        assert text.get_fontsize() == custom_data_label_size


def test_set_labels_use_legend_size(sample_venn_dataframe):
    """Venn set labels (Set A / Set B) functionally identify series, so they should track legend_size."""
    set_labels = ["Set A", "Set B"]
    result_ax = venn.plot(df=sample_venn_dataframe, labels=set_labels)

    set_label_texts = [t for t in result_ax.texts if t.get_text() in set_labels]
    assert len(set_label_texts) == len(set_labels)
    expected_size = get_option("plot.font.legend_size")
    for text in set_label_texts:
        assert text.get_fontsize() == expected_size


def test_venn_with_title(sample_venn_dataframe):
    """Test Venn diagram with a title to cover ax.set_title."""
    title = "Test Venn Diagram"
    result_ax = venn.plot(df=sample_venn_dataframe, labels=["A", "B"], title=title)

    assert isinstance(result_ax, Axes)
    title_texts = [t for t in result_ax.figure.texts if t.get_text() == title]
    assert len(title_texts) == 1
