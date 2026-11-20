"""Example script for IndexAnalysis: how over-represented each age band is among a brand's customers."""

import numpy as np
import pandas as pd

from openretailscience.analysis import IndexAnalysis
from openretailscience.plots import index

MIN_SUBSET_CUSTOMERS = 30
N_CUSTOMERS = 500

rng = np.random.default_rng(42)
n_lines = 2000
age_bands = ["18-34", "35-54", "55-64", "65+"]
customers = pd.DataFrame(
    {
        "customer_id": np.arange(1, N_CUSTOMERS + 1),
        # Some customers never gave their age; IndexAnalysis leaves them out of both totals
        "age_band": rng.choice([*age_bands, None], size=N_CUSTOMERS, p=[0.3, 0.3, 0.15, 0.15, 0.1]),
    },
)
df = pd.DataFrame(
    {
        "customer_id": rng.integers(1, N_CUSTOMERS + 1, n_lines),
        "brand": rng.choice(["Meadow Farm", "Greenvale", "Hillside"], n_lines),
        "unit_spend": rng.uniform(1, 20, n_lines).round(2),
    },
).merge(customers, on="customer_id")

# Customers: share of Meadow Farm's customers in each age band vs the share of all customers
ia = IndexAnalysis(
    df,
    value_col="customer_id",
    group_col="age_band",
    index_col="brand",
    value_to_index="Meadow Farm",
    agg_func="nunique",
)
result = ia.df  # subset_value, baseline_value, subset_share, baseline_share, index per age band

# Hide bands resting on too few customers before reading the index
reliable = result[result["subset_value"] >= MIN_SUBSET_CUSTOMERS]

# Count unknown ages as their own band instead of dropping them
labelled = df.assign(age_band=df["age_band"].fillna("Unknown"))
with_unknown = IndexAnalysis(
    labelled,
    value_col="customer_id",
    group_col="age_band",
    index_col="brand",
    value_to_index="Meadow Farm",
    agg_func="nunique",
).df

ax = index.plot(
    df,
    value_col="customer_id",
    group_col="age_band",
    index_col="brand",
    value_to_index="Meadow Farm",
    agg_func="nunique",
    eyebrow="Meadow Farm",
    title="Who buys Meadow Farm",
    subtitle="Customers by age band, 100 = all brands",
    sort_by="value",  # ascending draws the highest index at the top
)
