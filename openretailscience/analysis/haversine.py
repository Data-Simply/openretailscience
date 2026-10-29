"""Great-circle (haversine) distance between two lat/lon column pairs.

Spherical-Earth approximation: a straight-line distance, not a driving distance. The backend
must support trigonometric functions.
"""

import ibis


def haversine_distance(
    lat_col: ibis.Column,
    lon_col: ibis.Column,
    target_lat_col: ibis.Column | ibis.Scalar,
    target_lon_col: ibis.Column | ibis.Scalar,
    radius: float = 6371.0,
) -> ibis.Column:
    """Compute the Haversine distance between two sets of latitude and longitude columns.

    Args:
        lat_col (ibis.Column): Source latitudes in degrees.
        lon_col (ibis.Column): Source longitudes in degrees.
        target_lat_col (ibis.Column | ibis.Scalar): Target latitudes in degrees; ``ibis.literal(lat)`` is a fixed point.
        target_lon_col (ibis.Column | ibis.Scalar): Target longitudes in degrees.
        radius (float): Earth's radius in the desired output units; defaults to 6371.0 km.

    Returns:
        ibis.Column: Lazy expression for each source/target pair's distance, in ``radius`` units.

    Examples:
        >>> from openretailscience.analysis import haversine_distance
        >>> stores = ibis.memtable(
        ...     {"store_id": [101, 102], "latitude": [51.5072, 53.4808], "longitude": [-0.1276, -2.2426]}
        ... )
        >>> km = haversine_distance(
        ...     lat_col=stores.latitude, lon_col=stores.longitude,
        ...     target_lat_col=ibis.literal(52.6369), target_lon_col=ibis.literal(-1.1398),
        ... )
        >>> stores.mutate(km_to_warehouse=km).order_by("store_id").to_pandas()
           store_id  latitude  longitude  km_to_warehouse
        0       101   51.5072    -0.1276       143.404197
        1       102   53.4808    -2.2426       119.314868
    """
    lat1_rad = lat_col.radians()
    lat2_rad = target_lat_col.radians()
    delta_lat = (target_lat_col - lat_col).radians()
    delta_lon = (target_lon_col - lon_col).radians()

    a = (delta_lat / 2).sin().pow(2) + lat1_rad.cos() * lat2_rad.cos() * (delta_lon / 2).sin().pow(2)
    c = 2 * a.sqrt().asin()

    return radius * c
