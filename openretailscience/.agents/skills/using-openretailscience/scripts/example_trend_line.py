"""Example script demonstrating fitted trend lines with add_trend_line."""

import numpy as np
import pandas as pd

from openretailscience.plots import scatter
from openretailscience.plots.styles.trend import add_trend_line

rng = np.random.default_rng(42)

# Example 1: linear trend. Draw a chart with any plot function, then pass its
# returned Axes to add_trend_line; it annotates the fitted equation and R-squared.
n_stores = 60
visits = rng.integers(50, 500, size=n_stores)
store_df = pd.DataFrame({"visits": visits, "spend": visits * 12 + rng.normal(0, 250, size=n_stores)})
ax = scatter.plot(df=store_df, value_col="spend", x_col="visits", title="Store Spend vs Footfall")
add_trend_line(ax, trend_type="linear")

# Example 2: power trend for price elasticity (requires positive x and y).
# trend_type also accepts "logarithmic" and "exponential".
n_skus = 60
price = rng.uniform(2, 20, size=n_skus)
demand_df = pd.DataFrame({"price": price, "units_sold": 8000 * price ** (-1.3) * rng.uniform(0.8, 1.2, size=n_skus)})
ax2 = scatter.plot(df=demand_df, value_col="units_sold", x_col="price", title="Price Elasticity of Demand")
add_trend_line(ax2, trend_type="power")
