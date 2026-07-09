# openretailscience API overview

Detailed reference for the analysis and segmentation classes. All classes accept
a pandas `DataFrame` or an Ibis `Table` and read column names from the options
system unless an argument overrides them. Results are exposed via a `.df`
(pandas) and/or `.table` (Ibis) property; most classes also provide `.plot()`.

## Options system (`openretailscience.options`)

- `set_option(key, value)` / `get_option(key)` — e.g. `set_option("column.customer_id", "cust_id")`.
- `option_context({...})` — context manager for scoped overrides.
- `ColumnHelper` — resolve/join dotted option keys into concrete column names.
- Config can also be loaded from a TOML file (see `options_template.toml`).

Common keys: `column.customer_id`, `column.transaction_id`, `column.unit_spend`,
`column.unit_quantity`, `column.transaction_date`, plus `column.agg.*` aggregation
names and `column.suffix.*` period suffixes used by period-comparison analyses.

## Segmentation (`openretailscience.segmentation`)

- `rfm.RFMSegmentation(df, current_date=...)` — Recency/Frequency/Monetary scores
  per customer. Read `.df` (indexed by customer id) or `.table`.
- `hml` — Heavy/Medium/Light spend tiers.
- `nlr` — New/Lapsed/Reactivated lifecycle states.
- `threshold` — generic threshold-based labelling used by the above.
- `segstats` — summary statistics (customers, spend, share) per segment.

## Analysis (`openretailscience.analysis`)

- `cross_shop.CrossShop(df, group_1_col=..., group_1_val=..., ...)` — exclusive
  vs. overlapping shoppers across up to three groups; `.plot()` renders a Venn.
- `gain_loss.GainLoss(df, ...)` — decomposes the change in customers/spend between
  two periods into new, lost, and retained (increased/decreased) components.
- `cohort.CohortAnalysis(df, period=...)` — retention matrix by acquisition cohort.
- `product_association.ProductAssociation(df, ...)` — market-basket rules
  (support, confidence, lift) for co-purchased items.
- `revenue_tree.RevenueTree(df, ...)` — multiplicative KPI decomposition of spend;
  `calc_tree_kpis(...)` is the underlying functional entry point.
- `customer_decision_hierarchy.CustomerDecisionHierarchy(df, ...)` — dendrogram of
  substitutable products from co-occurrence.
- `customer.PurchasesPerCustomer`, `customer.DaysBetweenPurchases`,
  `customer.TransactionChurn` — per-customer purchase-count, inter-purchase-gap,
  and churn metrics.
- `composite_rank.CompositeRank(df, columns=...)` — rank items by several weighted
  metrics combined into one score.
- `haversine.haversine_distance(lat1, lon1, lat2, lon2)` — vectorized great-circle
  distance (e.g. customer-to-store).

## Example: gain-loss between two periods

```python
from openretailscience.options import option_context
from openretailscience.analysis.gain_loss import GainLoss

with option_context({"column.customer_id": "cust_id", "column.unit_spend": "net_sales"}):
    gl = GainLoss(transactions, ...)   # see class docstring for period args
    summary = gl.df                    # new / lost / retained decomposition
    gl.plot()
```

For exact constructor signatures and arguments, read each class's docstring
(`help(CrossShop)`), which documents Args, Returns, and Raises in Google style.
