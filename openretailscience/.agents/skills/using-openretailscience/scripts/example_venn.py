"""Example script demonstrating Venn and Euler diagrams for set overlap visualization."""

import pandas as pd
from openretailscience.plots import venn

# Example 1: 3-set Venn diagram (fixed circles)
df_3set = pd.DataFrame({
    "groups": [
        (1, 0, 0),  # Only set 1
        (0, 1, 0),  # Only set 2
        (0, 0, 1),  # Only set 3
        (1, 1, 0),  # Sets 1 and 2
        (1, 0, 1),  # Sets 1 and 3
        (0, 1, 1),  # Sets 2 and 3
        (1, 1, 1),  # All three sets
    ],
    "percent": [0.119, 0.090, 0.239, 0.209, 0.134, 0.104, 0.105]
})

venn.plot(
    df=df_3set,
    labels=["Electronics", "Clothing", "Home"],
    title="Customer Cross-Shopping Patterns",
    vary_size=False,
    subset_label_formatter=lambda v: f"{v:.1%}",
    source_text="Source: Transaction Data 2024",
)

# Example 2: 3-set Euler diagram (proportional sizing)
venn.plot(
    df=df_3set,
    labels=["Online Buyers", "Store Buyers", "App Users"],
    title="Customer Channel Usage Overlap",
    vary_size=True,  # Euler diagram with proportional circles
    subset_label_formatter=lambda v: f"{v*100:.0f}%",
    source_text="Source: Multi-Channel Analysis",
)

# Example 3: 2-set Venn diagram
df_2set = pd.DataFrame({
    "groups": [
        (1, 0),  # Only set 1
        (0, 1),  # Only set 2
        (1, 1),  # Both sets
    ],
    "percent": [0.35, 0.40, 0.25]
})

venn.plot(
    df=df_2set,
    labels=["Loyal Customers", "High Spenders"],
    title="Customer Segment Overlap",
    vary_size=False,
    subset_label_formatter=lambda v: f"{v:.0%}",
)

# Example 6: Customer segmentation with custom formatter
df_segments = pd.DataFrame({
    "groups": [(1, 0), (0, 1), (1, 1)],
    "percent": [450, 380, 170]  # Actual customer counts
})

venn.plot(
    df=df_segments,
    labels=["Email Subscribers", "SMS Subscribers"],
    title="Communication Channel Opt-Ins",
    vary_size=True,
    subset_label_formatter=lambda v: f"{int(v)} customers",
)
