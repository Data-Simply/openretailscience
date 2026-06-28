"""Tests for the haversine distance module."""

import ibis
import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from openretailscience.analysis.haversine import haversine_distance

EARTH_RADIUS_KM = 6371.0
# Two points on a sphere can be at most half the circumference apart (antipodes).
MAX_GREAT_CIRCLE_KM = np.pi * EARTH_RADIUS_KM

latitudes = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False, allow_infinity=False)
longitudes = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False, allow_infinity=False)
# A store/target coordinate pair.
coordinates = st.tuples(latitudes, longitudes)

# Absolute tolerance (km) for floating-point comparisons against geometric bounds.
DISTANCE_ATOL_KM = 1e-6
# Allowance for tiny negative round-off when asserting non-negativity.
NONNEGATIVE_ATOL_KM = 1e-9


@pytest.fixture
def sample_ibis_table():
    """Fixture to provide a sample Ibis table for testing."""
    data = {
        "lat1": [37.7749, 34.0522],
        "lon1": [-122.4194, -118.2437],
        "lat2": [40.7128, 36.1699],
        "lon2": [-74.0060, -115.1398],
    }
    df = pd.DataFrame(data)
    return ibis.memtable(df)


def test_haversine_distance(sample_ibis_table):
    """Test the haversine_distance function for correct distance calculation."""
    t = sample_ibis_table
    distance_expr = haversine_distance(t["lat1"], t["lon1"], t["lat2"], t["lon2"])

    assert isinstance(distance_expr, ibis.Column), "Output should be an Ibis expression."

    result_df = t.mutate(distance=distance_expr).execute()

    expected_distances = [4129.086165, 367.606322]

    for i, expected in enumerate(expected_distances):
        assert pytest.approx(result_df.iloc[i]["distance"], rel=1e-3) == expected, f"Row {i} distance mismatch."


class TestHaversineDistanceProperties:
    """Property-based tests for the geometric invariants of the haversine distance.

    The haversine formula computes great-circle distance between two points on a sphere. As a distance
    metric on the globe it must satisfy a handful of properties for *every* pair of valid coordinates,
    which is exactly what these tests pin down rather than relying on a couple of hand-computed cities.
    """

    @staticmethod
    def _distances(point_pairs: list[tuple[tuple[float, float], tuple[float, float]]]) -> np.ndarray:
        """Execute haversine_distance over a batch of (source, target) coordinate pairs."""
        df = pd.DataFrame(
            {
                "lat1": [src[0] for src, _ in point_pairs],
                "lon1": [src[1] for src, _ in point_pairs],
                "lat2": [tgt[0] for _, tgt in point_pairs],
                "lon2": [tgt[1] for _, tgt in point_pairs],
            },
        )
        t = ibis.memtable(df)
        distance_expr = haversine_distance(t["lat1"], t["lon1"], t["lat2"], t["lon2"])
        return t.mutate(distance=distance_expr).execute()["distance"].to_numpy()

    @settings(max_examples=50, deadline=None)
    @given(point_pairs=st.lists(st.tuples(coordinates, coordinates), min_size=1, max_size=25))
    def test_distance_is_nonnegative_and_bounded_by_half_circumference(self, point_pairs):
        """Every distance lies in [0, pi * radius] — non-negative and at most antipodal."""
        distances = self._distances(point_pairs)
        assert (distances >= -NONNEGATIVE_ATOL_KM).all()
        assert (distances <= MAX_GREAT_CIRCLE_KM + DISTANCE_ATOL_KM).all()

    @settings(max_examples=50, deadline=None)
    @given(points=st.lists(coordinates, min_size=1, max_size=25))
    def test_distance_from_a_point_to_itself_is_zero(self, points):
        """A store is zero kilometres from itself regardless of where it sits on the globe."""
        distances = self._distances([(p, p) for p in points])
        np.testing.assert_allclose(distances, 0.0, atol=DISTANCE_ATOL_KM)

    @settings(max_examples=50, deadline=None)
    @given(point_pairs=st.lists(st.tuples(coordinates, coordinates), min_size=1, max_size=25))
    def test_distance_is_symmetric(self, point_pairs):
        """Distance from A to B equals distance from B to A."""
        forward = self._distances(point_pairs)
        reverse = self._distances([(tgt, src) for src, tgt in point_pairs])
        np.testing.assert_allclose(forward, reverse, rtol=1e-9, atol=DISTANCE_ATOL_KM)
