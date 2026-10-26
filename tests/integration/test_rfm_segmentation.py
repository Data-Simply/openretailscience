"""Unified integration tests for RFM Segmentation with multiple database backends."""

import datetime as dt

import pytest

from openretailscience.segmentation.rfm import RFMSegmentation


@pytest.mark.parametrize(
    (
        "current_date",
        "r_segments",
        "f_segments",
        "m_segments",
        "min_monetary",
        "max_monetary",
        "min_frequency",
        "max_frequency",
    ),
    [
        (None, 10, 10, 10, None, None, None, None),
        ("2023-12-31", 5, 5, 5, 0.0, 1000.0, 1, 50),
        (dt.date(2023, 6, 30), [0.2, 0.4, 0.6, 0.8], [0.3, 0.6], [0.1, 0.5, 0.9], 10.0, 5000.0, 2, 100),
    ],
)
def test_rfm_segmentation_integration(
    transactions_table,
    current_date,
    r_segments,
    f_segments,
    m_segments,
    min_monetary,
    max_monetary,
    min_frequency,
    max_frequency,
):
    """Integration test for RFMSegmentation using parameterized database backends.

    This test runs against both BigQuery and PySpark backends automatically
    via pytest parameterization. The same test logic validates functionality
    across different database systems.

    Args:
        transactions_table: Parameterized fixture providing either BigQuery
                          or PySpark transactions table
        current_date: Current date parameter for RFM calculation
        r_segments: Recency segments configuration
        f_segments: Frequency segments configuration
        m_segments: Monetary segments configuration
        min_monetary: Minimum monetary value
        max_monetary: Maximum monetary value
        min_frequency: Minimum frequency value
        max_frequency: Maximum frequency value
    """
    limited_table = transactions_table.limit(1000)

    rfm_segmentation = RFMSegmentation(
        df=limited_table,
        current_date=current_date,
        r_segments=r_segments,
        f_segments=f_segments,
        m_segments=m_segments,
        min_monetary=min_monetary,
        max_monetary=max_monetary,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
    )

    assert rfm_segmentation is not None
    result = rfm_segmentation.df
    assert result is not None
