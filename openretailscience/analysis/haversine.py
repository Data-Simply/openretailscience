"""Great-circle (haversine) distance between two lat/lon column pairs.

All arguments and the return value are lazy Ibis expressions, so computation is pushed to the
backend, which must support trigonometric functions. Spherical-Earth approximation; the result is
a straight-line distance, not a driving distance, in the units of ``radius`` (km by default).
"""

import ibis


def haversine_distance(
    lat_col: ibis.Column,
    lon_col: ibis.Column,
    target_lat_col: ibis.Column,
    target_lon_col: ibis.Column,
    radius: float = 6371.0,
) -> ibis.Column:
    """Compute the Haversine distance between two sets of latitude and longitude columns.

    Input columns are in degrees; the return value is a lazy Ibis expression in the
    units of ``radius`` — nothing is materialized here.

    Args:
        lat_col (ibis.Column): Source latitudes in degrees.
        lon_col (ibis.Column): Source longitudes in degrees.
        target_lat_col (ibis.Column): Target latitudes in degrees.
        target_lon_col (ibis.Column): Target longitudes in degrees.
        radius (float): Earth's radius in the desired output units; defaults to 6371.0 km.

    Returns:
        ibis.Column: Lazy Ibis expression for the distance between each source/target pair.
    """
    lat1_rad = lat_col.radians()
    lat2_rad = target_lat_col.radians()
    delta_lat = (target_lat_col - lat_col).radians()
    delta_lon = (target_lon_col - lon_col).radians()

    a = (delta_lat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
    c = 2 * a.sqrt().asin()

    return radius * c
