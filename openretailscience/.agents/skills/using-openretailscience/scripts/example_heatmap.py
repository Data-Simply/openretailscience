"""Example script demonstrating the generic heatmap plot."""

import numpy as np
import pandas as pd

from openretailscience.plots import heatmap

rng = np.random.default_rng(42)

# Example 1: segment migration matrix, discrete colormap (the default).
# y_col labels the rows, every other column is drawn as a column of cells.
segments = ["Low", "Medium", "High", "Top"]
migration = pd.DataFrame(rng.integers(20, 400, size=(len(segments), len(segments))), columns=segments)
migration.insert(0, "segment_2023", segments)
heatmap.plot(
    df=migration,
    y_col="segment_2023",
    cbar_label="Customers",
    title="Customer Segment Migration",
    eyebrow="RETENTION",
    subtitle="How 2023 value segments moved in 2024",
)

# Example 2: continuous colormap with a dollar-formatted colorbar. Without y_col, the index labels the rows.
regions = ["North", "South", "East", "West"]
quarters = ["Q1", "Q2", "Q3", "Q4"]
basket = pd.DataFrame(
    rng.uniform(35, 95, size=(len(regions), len(quarters))).round(2),
    index=pd.Index(regions, name="region"),
    columns=pd.Index(quarters, name="quarter"),
)
heatmap.plot(
    df=basket,
    cbar_label="Avg basket ($)",
    title="Average Basket Value by Region and Quarter",
    colormap_style="continuous",
    cbar_format="${x:.0f}",
)
