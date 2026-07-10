"""Example: Advanced Customization

This script demonstrates advanced customization options including:
- Custom styling (linewidth, alpha, linestyle, markers)
- Post-plot modifications
- Custom colors
"""

import pandas as pd
from openretailscience.plots import line

# Sample data
df = pd.DataFrame({
    "months_relative": range(-6, 7),
    "revenue": [50000, 52000, 51000, 53000, 54000, 55000,
                58000, 62000, 65000, 67000, 70000, 72000, 75000]
})

# Custom styling
ax = line.plot(
    df=df,
    value_col="revenue",
    x_col="months_relative",
    x_label="Months Relative to Store Opening",
    y_label="Monthly Revenue (£)",
    title="Revenue Impact with Custom Styling",
    source_text="Source: Store Performance Analysis",
    linewidth=5,        # Thicker line
    alpha=0.7,          # Semi-transparent
    linestyle="--",     # Dashed line
    marker="o",         # Circle markers
    markersize=8,       # Larger markers
    color="#FF6B6B"     # Custom color
)

# Post-plot modification: reference line at the store opening
ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
