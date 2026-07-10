"""Example script demonstrating Haversine distance calculations with openretailscience."""

import ibis
import numpy as np
import pandas as pd

from openretailscience.analysis.haversine import haversine_distance

# Generate reproducible sample data
rng = np.random.default_rng(42)


# Example 1: Basic distance calculation
# City centers (lat, lon); assign each customer a random city, then add a random offset
# (roughly within 50km).
n = 10
cities = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
}
city_names = np.array(list(cities.keys()))
city_lats = np.array([cities[city][0] for city in city_names])
city_lons = np.array([cities[city][1] for city in city_names])
city_idx = rng.integers(0, len(city_names), size=n)
customers = pd.DataFrame(
    {
        "customer_id": np.arange(1, n + 1),
        "city": city_names[city_idx],
        "latitude": city_lats[city_idx] + rng.uniform(-0.5, 0.5, size=n),
        "longitude": city_lons[city_idx] + rng.uniform(-0.5, 0.5, size=n),
    }
)

stores = pd.DataFrame([
    {"store_id": 1, "name": "NYC Downtown", "latitude": 40.7589, "longitude": -73.9851},
    {"store_id": 2, "name": "NYC Midtown", "latitude": 40.7549, "longitude": -73.9840},
    {"store_id": 3, "name": "LA West", "latitude": 34.0522, "longitude": -118.2437},
    {"store_id": 4, "name": "LA East", "latitude": 34.0407, "longitude": -118.2468},
    {"store_id": 5, "name": "Chicago Loop", "latitude": 41.8781, "longitude": -87.6298},
])

# Convert to Ibis tables
customers_table = ibis.memtable(customers)
stores_table = ibis.memtable(stores)

# Cross join to get all customer-store combinations
customer_stores = customers_table.cross_join(stores_table)

# Calculate distances in kilometers
customer_stores = customer_stores.mutate(
    distance_km=haversine_distance(
        lat_col=customer_stores.latitude,
        lon_col=customer_stores.longitude,
        target_lat_col=customer_stores.latitude_right,
        target_lon_col=customer_stores.longitude_right,
        radius=6371.0,
    )
)
result_df = customer_stores.execute()

# Example 2: Find nearest store for each customer
nearest_stores = (
    customer_stores.group_by("customer_id")
    .mutate(min_distance=ibis._.distance_km.min())
    .filter(ibis._.distance_km == ibis._.min_distance)
    .select("customer_id", "store_id", "name", "distance_km", "city")
)
nearest_df = nearest_stores.execute()

# Example 3: Distance in miles instead of kilometers
customer_stores_miles = customers_table.cross_join(stores_table)
customer_stores_miles = customer_stores_miles.mutate(
    distance_miles=haversine_distance(
        lat_col=customer_stores_miles.latitude,
        lon_col=customer_stores_miles.longitude,
        target_lat_col=customer_stores_miles.latitude_right,
        target_lon_col=customer_stores_miles.longitude_right,
        radius=3959.0,  # Earth's radius in miles
    )
)
miles_df = customer_stores_miles.execute()
