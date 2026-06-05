---
title: Metrics
social:
  cards_layout_options:
    title: OpenRetailScience | Retail Metrics
---

!!! warning "Experimental"

    Metrics are experimental. Their APIs and import paths (under `openretailscience.experimental`) may change
    without notice.

## Distribution Metrics

### ACV (All Commodity Volume)

ACV measures the total dollar sales across all products in a set of stores, expressed in millions ($MM). It is commonly
used in retail analytics to quantify the size of a store or group of stores.

$$
\text{ACV} = \frac{\sum \text{unit_spend}}{\text{acv_scale_factor}}
$$

By default, `acv_scale_factor` is 1,000,000 (expressing ACV in $MM).

Example:

```python
import pandas as pd
from openretailscience.experimental.metrics.distribution.acv import Acv

df = pd.DataFrame({
    "store_id": [101, 101, 102, 102, 103],
    "unit_spend": [400_000, 600_000, 300_000, 200_000, 500_000],
})

acv = Acv(df, group_col="store_id")
print(acv.df)
#    store_id  acv
# 0       101  1.0
# 1       102  0.5
# 2       103  0.5
```

### % of Stores (Numeric Distribution)

% of Stores measures the share of total stores in the dataset that sell a given product. Every store counts equally
regardless of its sales volume. It answers the question: "What fraction of stores carry this product?"

$$
\%\text{Stores} = \frac{\text{COUNT(DISTINCT stores selling product)}}{\text{COUNT(DISTINCT all stores)}} \times 100
$$

Example:

```python
import pandas as pd
from openretailscience.experimental.metrics.distribution.pct_of_stores import PctOfStores

df = pd.DataFrame({
    "store_id": [10, 20, 20, 30, 40],
    "product_id": [501, 501, 502, 502, 503],
    "unit_spend": [5.99, 3.49, 4.00, 6.00, 2.50],
})

pct = PctOfStores(df)
print(pct.df)
#    product_id  stores  stores_pct
# 0         501       2        50.0
# 1         502       2        50.0
# 2         503       1        25.0
```

Use `group_col` to add extra grouping dimensions, and `within_group=True` to compute the
percentage relative to stores within each group rather than all stores:

```python
df = pd.DataFrame({
    "store_id": [10, 20, 30, 40, 10],
    "product_id": [501, 501, 502, 502, 502],
    "region": ["North", "North", "South", "South", "North"],
    "unit_spend": [5.99, 3.49, 4.00, 6.00, 2.50],
})

# Percentage relative to all stores (default)
pct = PctOfStores(df, group_col="region")

# Percentage relative to stores within each region
pct_within = PctOfStores(df, group_col="region", within_group=True)
```
