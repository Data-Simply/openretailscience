"""Example script demonstrating format_shorthand number formatting."""

import pandas as pd
from openretailscience.plots import bar
from openretailscience.plots.styles.graph_utils import format_shorthand, set_axis_shorthand

# Example 1: Basic number formatting
format_shorthand(1234)  # "1K"
format_shorthand(1234567)  # "1M"
format_shorthand(1234567890)  # "1B"
format_shorthand(1234567890000)  # "1T"

# Example 2: With decimal precision
format_shorthand(1234567, decimals=0)  # "1M"
format_shorthand(1234567, decimals=1)  # "1.2M"
format_shorthand(1234567, decimals=2)  # "1.23M"

# Example 3: Currency formatting
revenue = 2500000
format_shorthand(revenue, decimals=1, prefix="$")  # "$2.5M"
format_shorthand(3400000, decimals=1, prefix="£")
format_shorthand(5600000, decimals=1, prefix="€")

# Example 4: Trailing zero removal
format_shorthand(1000000, decimals=2)  # "1M" not "1.00M"
format_shorthand(1500000, decimals=2)  # "1.5M" not "1.50M"
format_shorthand(1230000, decimals=2)  # "1.23M"

# Example 5: Negative numbers
format_shorthand(-1500000, decimals=1)  # "-1.5M"
format_shorthand(-500, decimals=0)  # "-500"

# Example 6: Edge cases
format_shorthand(0)  # "0"
format_shorthand(50)  # "50"
format_shorthand(999)  # just under 1K, stays "999"
format_shorthand(999.5, decimals=0)  # rounds up to "1K"

# Example 7: Apply shorthand formatting to a plot's axis with set_axis_shorthand
df = pd.DataFrame({
    "month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    "sales": [1200000, 1500000, 1800000, 2100000, 2400000, 2700000],
})
ax = bar.plot(df=df, value_col="sales", x_col="month", title="Monthly Sales (Human-Readable Format)", y_label="Sales")
set_axis_shorthand(ax.yaxis, decimals=1)
