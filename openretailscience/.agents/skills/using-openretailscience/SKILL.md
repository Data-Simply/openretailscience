---
name: using-openretailscience
description: >-
  Guidance for writing correct, performant retail analytics with the
  openretailscience Python package. Use when a task involves customer, basket,
  or transaction data and mentions RFM/HML/NLR segmentation, cross-shop / Venn
  overlap, gain-loss (switching) analysis, cohort/retention, product association
  (market-basket), a revenue tree / KPI decomposition, customer decision
  hierarchy, or the package's column-options system. Also use when the user
  imports `openretailscience` or asks how to configure its column names.
---

# Using openretailscience

`openretailscience` is a retail data-science toolkit. Analyses accept either a
pandas `DataFrame` or an Ibis `Table` (backed by DuckDB, BigQuery, Spark, etc.),
so the same code runs on a laptop sample or a 1B–10B-row warehouse table. Prefer
passing an **Ibis table** for large data so aggregation is pushed to the engine.

## Core workflow

1. **Configure column names once** via the options system, then call analyses
   without repeating column arguments.
2. **Instantiate an analysis class** with your data. Computation is lazy/coalesced
   into the class.
3. **Read results** from the `.df` (pandas) or `.table` (Ibis) property, and/or
   render a chart with `.plot()`.

```python
from openretailscience.options import set_option, option_context
from openretailscience.analysis.cross_shop import CrossShop

# Map your schema to the package's canonical column names.
set_option("column.customer_id", "cust_id")
set_option("column.unit_spend", "net_sales")

cs = CrossShop(df, group_1_col="category", group_1_val="Beer", ...)
result_df = cs.df          # pandas result
cs.plot()                  # Venn diagram
```

Use `option_context({...})` to scope column config to a `with` block instead of
mutating global state.

## Configuration: the options system

Column names, aggregation names, and period suffixes are all options (dotted
keys like `column.customer_id`, `column.unit_spend`, `column.transaction_id`).

- `get_option(key)` / `set_option(key, value)` — read/write one option.
- `option_context({key: value, ...})` — temporary override in a `with` block.
- `ColumnHelper` — resolves/joins option keys into concrete column names; use it
  (rather than hardcoded strings) when building column lists in your own code.

Never hardcode a column name the package owns — read it from the options system
so downstream analyses stay consistent.

## Choosing an analysis

| Task | Import |
| --- | --- |
| RFM / HML / NLR segments, segment stats | `segmentation.rfm.RFMSegmentation`, `segmentation.hml`, `segmentation.nlr`, `segmentation.segstats` |
| Category/brand overlap (Venn) | `analysis.cross_shop.CrossShop` |
| Customers/spend gained vs. lost between periods | `analysis.gain_loss.GainLoss` |
| Retention by acquisition cohort | `analysis.cohort.CohortAnalysis` |
| Market-basket / co-purchase rules | `analysis.product_association.ProductAssociation` |
| KPI decomposition (spend = customers × freq × ...) | `analysis.revenue_tree.RevenueTree` |
| Purchase-behavior tree from co-occurrence | `analysis.customer_decision_hierarchy.CustomerDecisionHierarchy` |
| Per-customer purchase/recency/churn metrics | `analysis.customer` (`PurchasesPerCustomer`, `DaysBetweenPurchases`, `TransactionChurn`) |
| Rank items by several weighted metrics | `analysis.composite_rank.CompositeRank` |

See `references/api-overview.md` for constructor arguments, result shapes, and
worked examples for each class.

## Performance & correctness rules

- Assume 1B–10B rows. Pass an Ibis table and let the engine aggregate; avoid
  `.execute()` / `.to_pandas()` until you need the (already small) result.
- Don't run redundant work — e.g. no `.nunique()` on already-deduplicated data.
- Timestamps must be timezone-naive; convert before analysis.
- Use vectorized pandas/numpy on any result frames — never `.iterrows()`.

## When NOT to use this skill

Generic pandas wrangling, plotting outside the package, or non-retail data. This
skill is specifically about `openretailscience`'s analysis and options APIs.
