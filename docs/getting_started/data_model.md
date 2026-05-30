---
title: Data Model
social:
  cards_layout_options:
    title: OpenRetailScience | Data Model
---

# Data Model

OpenRetailScience works on **transaction-level data**: one row per product line within a transaction. This is the
same grain most point-of-sale and e-commerce systems already produce, so in many cases your data is closer to ready
than you might expect.

This page describes the shape the modules expect, which columns you actually need, and how to reshape a typical
retail star schema into that flat form. Once your data matches this model, see the
[Options & Configuration](options_guide.md) guide to map your column names onto the defaults.

## The grain: one row per transaction line item

A single basket usually contains several products. OpenRetailScience expects one row for **each product within each
transaction**, not one row per basket. A customer who buys three different products in one visit produces three rows
that share the same `transaction_id`:

| transaction_id | transaction_date | customer_id | product_id | unit_quantity | unit_spend | store_id |
| -------------- | ---------------- | ----------- | ---------- | ------------- | ---------- | -------- |
| 16050          | 2023-01-12       | 1           | 15         | 2             | 55.98      | 6        |
| 16050          | 2023-01-12       | 1           | 1317       | 1             | 10.49      | 6        |
| 20090          | 2023-02-05       | 1           | 509        | 3             | 360.00     | 4        |

You can carry extra columns (product names, categories, brands, regions, etc.) alongside these — modules ignore
columns they do not use, and you can pass them as grouping dimensions where a module supports it.

!!! tip "No pre-aggregation needed"
    You do **not** build a per-customer or per-segment summary table first. Tools such as
    [`SegTransactionStats`](../api/segmentation/segstats.md) take transaction-level rows directly and compute
    aggregates — spend per customer, transaction frequency, average basket size — internally. Hand them the raw
    lines and let the library do the rollup.

## Column reference

The table below lists the base columns OpenRetailScience recognizes, along with the option key used to rename each
one and its default name. You only need the columns your chosen analysis uses — see
[Bring only what you need](#bring-only-what-you-need) below.

| Option key                   | Default name          | Meaning                                          | Typical type      |
| ---------------------------- | --------------------- | ------------------------------------------------ | ----------------- |
| `column.customer_id`         | `customer_id`         | Customer identifier                              | integer / string  |
| `column.transaction_id`      | `transaction_id`      | Transaction (basket) identifier                  | integer / string  |
| `column.transaction_date`    | `transaction_date`    | Date of the transaction                          | date / timestamp  |
| `column.transaction_time`    | `transaction_time`    | Time of the transaction                          | time / string     |
| `column.product_id`          | `product_id`          | Product identifier                               | integer / string  |
| `column.unit_quantity`       | `unit_quantity`       | Number of units sold on the line                 | integer           |
| `column.unit_price`          | `unit_price`          | Price of a single unit                           | float             |
| `column.unit_spend`          | `unit_spend`          | Total spend on the line (`unit_price * units`)   | float             |
| `column.unit_cost`           | `unit_cost`           | Total cost of the line (`unit cost * units`)     | float             |
| `column.promo_unit_spend`    | `promo_unit_spend`    | Total promotional spend on the line              | float             |
| `column.promo_unit_quantity` | `promo_unit_quantity` | Number of units sold on promotion                | integer           |
| `column.store_id`            | `store_id`            | Store identifier                                 | integer / string  |

### Input columns vs. generated columns

The columns above are **inputs you provide**. The library also defines names for columns it *produces* — aggregates
like `spend` and `customers` (`column.agg.*`), calculated metrics like `price_per_unit` (`column.calc.*`), and
suffixes like `pct` and `diff` (`column.suffix.*`). You never supply those; they appear in result tables. They are
configurable for the same reason the base columns are — so output matches your naming conventions. See the
[Options & Configuration](options_guide.md) guide for the full list.

## Bring only what you need

No single column is required for everything. Each module validates just the columns it uses and raises a clear error
if one is missing, so you can start with a minimal dataset and add columns as your analysis grows. For example:

- **Cohort analysis** needs `customer_id` and `transaction_date`.
- **Gain/loss analysis** needs `customer_id` and a value column (`unit_spend` by default).
- **ACV** needs `store_id` and `unit_spend`.

If your data is missing a column a module needs, you will see an error naming the missing columns rather than a
silent wrong answer.

## Reshaping a star schema

Most retailers store sales in a **star schema**: a central fact table of sales lines surrounded by dimension tables
for products, stores, customers, and dates. OpenRetailScience expects those joined into a single flat table, so the
transformation is a set of joins from the fact table to whichever dimensions carry the columns you need.

Given a `fact_sales` table keyed to `dim_product`, `dim_store`, and `dim_date`, the join in
[Ibis](https://ibis-project.org/) looks like this:

```python
import ibis

con = ibis.duckdb.connect("retail.ddb")

fact = con.table("fact_sales")
dim_product = con.table("dim_product")
dim_store = con.table("dim_store")
dim_date = con.table("dim_date")

transactions = (
    fact
    .join(dim_product, "product_id")
    .join(dim_store, "store_id")
    .join(dim_date, "date_id")
    .select(
        "transaction_id",
        "customer_id",
        "product_id",
        "store_id",
        transaction_date=dim_date.calendar_date,
        unit_quantity=fact.quantity,
        unit_spend=fact.sales_amount,
        # carry any dimension attributes you want to group by later
        category=dim_product.category_name,
        region=dim_store.region_name,
    )
)
```

The same transformation as a SQL view you materialize in your warehouse:

```sql
CREATE VIEW transactions AS
SELECT
    f.transaction_id,
    f.customer_id,
    f.product_id,
    f.store_id,
    d.calendar_date AS transaction_date,
    f.quantity      AS unit_quantity,
    f.sales_amount  AS unit_spend,
    p.category_name AS category,
    s.region_name   AS region
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_store   s ON f.store_id   = s.store_id
JOIN dim_date    d ON f.date_id    = d.date_id;
```

Two things to keep in mind when flattening:

- **Preserve the line-item grain.** Join dimensions onto the fact table; do not aggregate. Each fact row should
  remain its own output row so the transaction-level grain is intact.
- **Rename in the projection.** Aliasing fact/dimension columns to the names in the
  [column reference](#column-reference) during the join is the simplest mapping. If renaming at the source is not
  practical, keep your own names and point OpenRetailScience at them through the
  [Options & Configuration](options_guide.md) guide instead.

## Next steps

- [Connecting Your Data](connecting_your_data.md) — load this table from a warehouse, Parquet, or pandas.
- [Options & Configuration](options_guide.md) — map your column names onto the defaults above.
- [Analysis Modules](../analysis_modules.md) — run your first analysis on the result.
