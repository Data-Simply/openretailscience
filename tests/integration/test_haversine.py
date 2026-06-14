"""Unified integration tests for haversine distance function with multiple database backends."""

from typing import TYPE_CHECKING, cast

import ibis
import pytest

from openretailscience.analysis.haversine import haversine_distance

if TYPE_CHECKING:
    from ibis.expr.types import NumericValue


def test_haversine_integration():
    """Integration test for haversine distance function.

    This test doesn't use the parameterized transactions_table since
    the haversine function works with literal values rather than
    database tables. The test verifies the function works consistently
    across different Ibis backends.
    """
    # ibis.literal is stubbed to return the base Scalar type; cast the float64 literals to
    # NumericValue (the actual runtime type) so they satisfy the numeric column signature.
    lat1 = cast("NumericValue", ibis.literal(37.7749, type="float64"))
    lon1 = cast("NumericValue", ibis.literal(-122.4194, type="float64"))
    lat2 = cast("NumericValue", ibis.literal(40.7128, type="float64"))
    lon2 = cast("NumericValue", ibis.literal(-74.0060, type="float64"))

    distance_expr = haversine_distance(lat1, lon1, lat2, lon2)
    result = distance_expr.execute()

    expected_distance = 4129.086165
    assert pytest.approx(result, rel=1e-3) == expected_distance, "Distance calculation error"
