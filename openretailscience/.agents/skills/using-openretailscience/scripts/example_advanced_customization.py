"""Advanced line-plot customization: styling (linewidth, alpha, linestyle, markers), custom colors, and post-plot tweaks."""

import pandas as pd
from openretailscience.plots import line

df = pd.DataFrame({
    "months_relative": range(-6, 7),
    "revenue": [50000, 52000, 51000, 53000, 54000, 55000,
                58000, 62000, 65000, 67000, 70000, 72000, 75000]
})

ax = line.plot(
    df=df,
    value_col="revenue",
    x_col="months_relative",
    x_label="Months Relative to Store Opening",
    y_label="Monthly Revenue (£)",
    title="Revenue Impact with Custom Styling",
    source_text="Source: Store Performance Analysis",
    linewidth=5,
    alpha=0.7,
    linestyle="--",
    marker="o",
    markersize=8,
    color="#FF6B6B"
)

# Reference line at store opening (x=0)
ax.axvline(x=0, color="gray", linestyle=":", alpha=0.5)
