---
name: using-openretailscience
description: >-
  Guidance for writing correct, performant retail analytics with the
  openretailscience Python package. Use when a task involves customer, basket,
  or transaction data and mentions any openretailscience analysis (RFM / HML /
  NLR / threshold segmentation, cross-shop Venn overlap, gain-loss switching,
  cohort retention, product association / market-basket, revenue-tree KPI
  decomposition, customer decision hierarchy, composite rank, haversine), any of
  its plots (bar, line, area, scatter, histogram, waterfall, venn, heatmap,
  cohort, time, period-on-period, broken-timeline, price, index) or trendlines,
  its options/ColumnHelper configuration system, or connecting analyses to a
  database via Ibis. Also use whenever the user imports `openretailscience`.
---

# Using openretailscience

`openretailscience` is a retail data-science toolkit. Every analysis accepts a
pandas `DataFrame` **or** an Ibis `Table` (DuckDB, BigQuery, Snowflake,
Databricks, PySpark, MSSQL, Oracle), so the same code runs on a laptop sample or
a 1B–10B-row warehouse table. Prefer an **Ibis table** for large data so
filtering and aggregation are pushed into the engine.

There are **no top-level re-exports** — always import from the full module path,
e.g. `from openretailscience.analysis.cross_shop import CrossShop`.

## Core workflow

1. **Configure column names once** with the options system, so analyses find your
   schema without per-call column arguments.
2. **Instantiate an analysis class** (or call a plot function) with your data.
3. **Read results** from `.df` (pandas) or `.table` (Ibis), and/or render with
   `.plot()` / `.draw_tree()` or a `plots.*` function.

```python
import ibis
from openretailscience.options import set_option
from openretailscience.segmentation.segstats import SegTransactionStats

con = ibis.duckdb.connect()
transactions = con.read_parquet("data/transactions.parquet")

set_option("column.customer_id", "cust_id")          # map your schema once
q1 = transactions.filter(transactions.transaction_date.between("2023-01-01", "2023-03-31"))
stats = SegTransactionStats(data=q1, segment_col="category_0_name", grouping_sets="total")
print(stats.df)                                      # aggregation runs in the engine
```

## Reference files

Read the relevant reference file for full constructor arguments, result shapes,
and examples — do not guess signatures:

- **`references/analysis.md`** — every analysis and segmentation class, the
  `utils` labelling/date helpers, and the experimental metrics.
- **`references/plotting.md`** — every `plots.*` function, **trendlines**
  (`add_trend_line`), and the styling / color / font helpers.
- **`references/configuration.md`** — the options system, `ColumnHelper`, the
  `openretailscience.toml` file, connecting to each Ibis backend, and the
  `constants.COLORS` palette.

## Runnable examples

Every capability has a complete, copy-pasteable script at
`scripts/example_<name>.py`, such as `scripts/example_cohort_analysis.py`,
`scripts/example_waterfall.py`, or `scripts/example_rfm_segmentation.py`. List the
`scripts/` directory to find the one matching your task, then adapt it to the
user's data. Each script runs end-to-end against the installed package and is
executed in the package's test suite, so it stays correct as the API evolves.
Prefer reading the relevant script over assembling calls from signatures alone.

## Picking an analysis

| Task | Import |
| --- | --- |
| Transaction KPIs by segment/dimension (rollup/cube/total) | `segmentation.segstats.SegTransactionStats` |
| RFM / HML / threshold / NLR customer segments | `segmentation.rfm` / `.hml` / `.threshold` / `.nlr` |
| Category/brand overlap (Venn) | `analysis.cross_shop.CrossShop` |
| Customers/spend gained, lost, switched between periods | `analysis.gain_loss.GainLoss` |
| Retention by acquisition cohort | `analysis.cohort.CohortAnalysis` |
| Market-basket / co-purchase rules | `analysis.product_association.ProductAssociation` |
| KPI decomposition tree (spend = customers × visits × ...) | `analysis.revenue_tree.RevenueTree` |
| Substitutable-product dendrogram | `analysis.customer_decision_hierarchy.CustomerDecisionHierarchy` |
| Per-customer purchase / recency / churn metrics | `analysis.customer` (Purchases/Days/Churn classes) |
| Rank items by several weighted metrics | `analysis.composite_rank.CompositeRank` |
| Great-circle customer-to-store distance | `analysis.haversine.haversine_distance` |

## Performance & correctness rules

- Assume 1B–10B rows. Pass an Ibis table; let the engine aggregate and filter.
  Don't call `.execute()` / `.to_pandas()` until you need the (small) result.
- Avoid redundant work (e.g. no `.nunique()` on already-deduplicated data).
- Timestamps must be timezone-naive; convert before analysis.
- Configure column names through the options system rather than hardcoding — the
  package owns a canonical name for each column (see `references/configuration.md`).
- Use vectorized pandas/numpy on result frames — never `.iterrows()`.

## When NOT to use this skill

Generic pandas wrangling, plotting outside the package, or non-retail data.
