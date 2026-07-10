"""Example script demonstrating histogram plot usage."""

import numpy as np
import pandas as pd
from openretailscience.plots import histogram

# Create sample data - customer purchase revenue
rng = np.random.default_rng(42)
df = pd.DataFrame({
    "first_purchase_revenue": np.concatenate([
        rng.normal(70, 10, 50000),
        rng.normal(90, 15, 50000)
    ]),
    "product": ["Product A"] * 50000 + ["Product B"] * 50000
})

# Example 1: Basic histogram - single distribution
histogram.plot(
    df=df,
    value_col="first_purchase_revenue",
    title="First Purchase Revenue Distribution (£)",
    x_label="Revenue (£)",
    y_label="Number of Customers",
    source_text="Source: OpenRetailScience - 2024",
    bins=30,
)

# Example 2: Grouped histogram with hatch patterns
histogram.plot(
    df=df,
    value_col="first_purchase_revenue",
    group_col="product",
    title="First Purchase Revenue by Product (£)",
    x_label="Revenue (£)",
    y_label="Number of Customers",
    source_text="Source: OpenRetailScience - 2024",
    move_legend_outside=True,
    use_hatch=True,
    bins=30,
)

# Example 3: Range clipping - focus on specific range
histogram.plot(
    df=df,
    value_col="first_purchase_revenue",
    group_col="product",
    clip_range=(50, 120),  # Clamp values into £50-£120, piling outliers at edges
    title="Revenue Distribution £50-£120 (Clipped)",
    x_label="Revenue (£)",
    y_label="Number of Customers",
    move_legend_outside=True,
    bins=25,
)

# Example 4: Excluding outliers by dropping out-of-range values
histogram.plot(
    df=df,
    value_col="first_purchase_revenue",
    group_col="product",
    range=(40, 130),  # matplotlib range kwarg drops out-of-range values
    title="Revenue Distribution (Outliers Excluded)",
    x_label="Revenue (£)",
    y_label="Number of Customers",
    move_legend_outside=True,
    bins=30,
)

# Example 5: Series plotting
# A pre-aggregated/standalone pandas Series can be passed directly with value_col=None.
revenue_series = df["first_purchase_revenue"]
revenue_series.name = "First Purchase Revenue"

histogram.plot(
    df=revenue_series,
    value_col=None,  # Must be None for Series
    title="First Purchase Revenue Distribution",
    x_label="Revenue (£)",
    y_label="Frequency",
    bins=20,
)
