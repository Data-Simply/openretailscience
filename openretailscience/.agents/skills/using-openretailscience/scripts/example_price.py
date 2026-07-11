"""Price architecture bubble charts for competitive price analysis."""

import numpy as np
import pandas as pd
from openretailscience.plots import price

rng = np.random.default_rng(42)

# Retailer A: Low price positioning
retailer_a = pd.DataFrame({
    "product_id": range(1, 51),
    "unit_price": rng.uniform(1, 4, 50),
    "retailer": "Walmart"
})

# Retailer B: Mid price positioning
retailer_b = pd.DataFrame({
    "product_id": range(51, 101),
    "unit_price": rng.uniform(2, 6, 50),
    "retailer": "Target"
})

# Retailer C: High price positioning
retailer_c = pd.DataFrame({
    "product_id": range(101, 151),
    "unit_price": rng.uniform(4, 9, 50),
    "retailer": "Whole Foods"
})

# Retailer D: Mixed pricing
retailer_d = pd.DataFrame({
    "product_id": range(151, 201),
    "unit_price": rng.uniform(1, 10, 50),
    "retailer": "Amazon"
})

df = pd.concat([retailer_a, retailer_b, retailer_c, retailer_d])

# Example 1: Equal-width bins
price.plot(
    df=df,
    value_col="unit_price",
    group_col="retailer",
    bins=10,
    title="Retailer Price Distribution Analysis",
    x_label="Retailer",
    y_label="Price Bands ($)",
    legend_title="Retailer",
    source_text="Source: Competitive Price Intelligence 2024",
    move_legend_outside=True,
)

# Example 2: Custom bin boundaries
price.plot(
    df=df,
    value_col="unit_price",
    group_col="retailer",
    bins=[0, 2, 4, 6, 8, 10],
    title="Retailer Price Architecture by Tier",
    x_label="Retailer",
    y_label="Price Tier ($)",
    legend_title="Retailer",
    move_legend_outside=True,
    alpha=0.7,
    s=1000,
)

# Example 3: Custom styling
price.plot(
    df=df,
    value_col="unit_price",
    group_col="retailer",
    bins=8,
    title="Retailer Price Positioning (Enhanced)",
    x_label="Retailer",
    y_label="Price Bands",
    legend_title="Retailer",
    move_legend_outside=True,
    alpha=0.5,
    s=1200,
    edgecolor="white",
    linewidth=2,
)
