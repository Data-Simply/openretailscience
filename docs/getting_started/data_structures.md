# Data Structures

## Overview

OpenRetailScience analyses run on a single, flat table of retail transactions. You bring that table as either a
pandas DataFrame or an [Ibis](https://ibis-project.org/) table, and each analysis reads the columns it needs by their
standard names.

They were shaped by working with dozens of retailers' transactional data and follow the patterns that are standard
across the industry: star-schema warehouses, transaction- and line-item-level sales tables, and single-column keys.
Adopting them is what lets the same OpenRetailScience analysis run unchanged across very different retailers.

This guide explains the shape that table should have: the granularity it can be at, which columns each analysis
requires, how identifiers must be structured, and the data-quality assumptions baked into the package. With those
settled, you can prepare your data correctly on the first attempt.

!!! info "Where this fits"
    This guide covers the *shape* of your data: the granularity and columns each analysis needs. For connecting to
    a backend or mapping your own column names, see [Connecting to your data](connecting_to_data.md) and the
    [Options & configuration guide](options_guide.md).

## Core data format

### One flat table, two granularities

OpenRetailScience expects a single denormalized table where each row is one observation of a sale. That table can be
at either of two granularities, and most functions accept both because they aggregate internally:

- **Line-item level**: one row per product within each transaction. A basket of three products is three rows that
  share a `transaction_id`. This is the richest form and the only one that supports product-level analyses.
- **Transaction level**: one row per transaction (basket). Product detail has already been summed away, so
  `unit_spend` is the basket total and there is no per-product breakdown.

The sample dataset shipped at `data/transactions.parquet` is **line-item level**. Its schema is the canonical
example of an analysis-ready table:

```text
ibis.Schema {
  transaction_id    int64
  transaction_date  date
  transaction_time  time
  customer_id       int64
  product_id        int64
  product_name      string
  category_0_name   string
  category_0_id     int64
  category_1_name   string
  category_1_id     int64
  brand_name        string
  brand_id          int64
  unit_quantity     int64
  unit_cost         float64
  unit_spend        float64
  store_id          int64
}
```

### Which granularity to use

Choose the granularity by the question you are asking:

- Use **line-item level** whenever product, brand, or category matters: product association, customer decision
  hierarchies, and any segmentation or revenue analysis sliced by category or brand all need it.
- **Transaction level** is sufficient for purely basket- or customer-shaped questions (how many baskets per
  customer, spend per customer, churn) and is cheaper to store and scan.

When in doubt, keep line-item level: every transaction-level metric can be derived from it, but not the reverse.
Functions such as `SegTransactionStats` and `RevenueTree` aggregate line items for you (to the segment or period
level their column arguments define), so you do not need to pre-aggregate.

### Star schemas and denormalized views

Most retail data warehouses store sales in a **star schema**: a central fact table of transaction line items
referencing dimension tables for products, stores, customers, and dates. That is good warehouse design, and
OpenRetailScience does not ask you to change it.

What the package works on is a **denormalized analysis view** built by joining the dimensions you need onto the fact
table. You create that view once (in SQL, in Ibis, or in pandas) and pass it to the analysis:

```sql
-- Create an analysis-ready view from a star schema
SELECT
    f.transaction_id,
    f.transaction_date,
    f.customer_id,
    f.unit_quantity,
    f.unit_spend,
    p.category_0_name,
    p.brand_name,
    s.store_region
FROM fact_sales f
JOIN dim_product p ON f.product_id = p.product_id
JOIN dim_store   s ON f.store_id = s.store_id
```

The same join expressed in Ibis, so it runs inside your warehouse:

```python
import ibis

con = ibis.duckdb.connect("warehouse.ddb")
fact = con.table("fact_sales")
product = con.table("dim_product")
store = con.table("dim_store")

analysis_view = (
    fact.join(product, "product_id")
    .join(store, "store_id")
    .select(
        "transaction_id",
        "transaction_date",
        "customer_id",
        "unit_quantity",
        "unit_spend",
        "category_0_name",
        "brand_name",
        "store_region",
    )
)
```

The `.select(...)` keeps the view lightweight: pull only the dimension attributes the analysis needs. A view
dragging every column from every dimension is slower to scan and no more useful.

## Column requirements

### Standard column names

OpenRetailScience reads columns by a fixed set of standard names. The most common are:

| Standard name      | Meaning                                  |
| ------------------ | ---------------------------------------- |
| `customer_id`      | Customer identifier                      |
| `transaction_id`   | Transaction (basket) identifier          |
| `transaction_date` | Date of the transaction                  |
| `transaction_time` | Time of the transaction                  |
| `product_id`       | Product identifier                       |
| `unit_quantity`    | Number of units sold                     |
| `unit_price`       | Price per unit                           |
| `unit_spend`       | Total spend (price × quantity)           |
| `unit_cost`        | Cost per unit                            |
| `store_id`         | Store identifier                         |

Not every dataset contains every column. This is the set of names OpenRetailScience *recognises*, and each
analysis needs only the ones it uses (the sample data, for example, has no `unit_price`).

If your warehouse uses different names, **do not rename every column**. Map your names onto the standard ones once,
through the options system (a `openretailscience.toml` file, `option_context()`, or `set_option()`). See the
[Options & configuration guide](options_guide.md) for the three approaches and when to use each.

### Required vs optional columns

No single set of columns is required for *every* analysis. Each function validates only what it uses and raises a
clear error naming any column it cannot find:

```python
ValueError: Input data is missing required columns: ['customer_id'].
```

Columns fall into three roles:

- **Required**: the function fails without them (for example, `RFMSegmentation` needs `transaction_date` to compute
  recency).
- **Optional, behaviour-enhancing**: used if present, skipped if absent. `SegTransactionStats` adds
  per-customer metrics only when `customer_id` is present, and unit-price and units-per-transaction metrics only
  when `unit_quantity` is present.
- **You-named**: columns you point a function at by name through a constructor parameter, such as `segment_col`,
  `period_col`, or `value_col`. These are always required, but you choose which column fills the role.

## ID column handling

### Single-column identifiers only

!!! warning "No compound keys"
    Every identifier OpenRetailScience reads (`customer_id`, `transaction_id`, `product_id`, `store_id`) must be a
    **single column**. The package has no concept of a multi-column key. If your source system identifies an entity
    by a combination of columns (for example `region_id` + `customer_number`), combine them into one column before
    analysis.

### Creating composite IDs

Build the single identifier upstream, in your warehouse query or in the Ibis/pandas step that prepares the view.
String concatenation with a separator is the robust default because it cannot collide:

```sql
-- Robust: concatenate with a separator that never appears in the parts
SELECT
    region_id || '-' || customer_number AS customer_id,
    ...
FROM customers
```

The same transformation in Ibis:

```python
analysis_view = analysis_view.mutate(
    customer_id=(
        analysis_view.region_id.cast("string") + "-" + analysis_view.customer_number.cast("string")
    ),
)
```

An integer-offset scheme (`region_id * 1_000_000 + customer_number`) yields a compact numeric key, but only when the
offset is provably larger than every `customer_number`; otherwise two different pairs can map to the same id. Prefer
string concatenation unless you have measured a need for integer keys.

### Performance considerations

!!! tip "Integer IDs are cheaper at scale"
    On warehouse-scale tables (the 1B–10B row range OpenRetailScience targets), the `nunique`, `group_by`, and join
    operations these analyses run are faster and lighter on integer keys than on long strings. If you control the
    pipeline, generate stable integer surrogate keys in the warehouse. If you build composite keys by concatenation,
    consider hashing or dense-ranking them to an integer once, then reusing that column.

## Data quality

OpenRetailScience does not silently clean your data. It assumes the analysis view is already correct. A few
expectations are worth checking before you run an analysis.

### Types

- **Dates and times** should be genuine temporal types (`date` / `datetime`), not strings. `DaysBetweenPurchases`
  and `TransactionChurn` validate this and reject a non-temporal column; other date-based analyses such as
  `RFMSegmentation` and `CohortAnalysis` assume a temporal column rather than checking it, so a string date can
  produce wrong results.
- **Timestamps should be timezone-naive.** `DaysBetweenPurchases` and `TransactionChurn` reject a timezone-aware
  timestamp outright, because the backend normalizes it to UTC, which can shift day boundaries. Strip the zone on a
  pandas column with `df["transaction_date"] = df["transaction_date"].dt.tz_localize(None)`.
- **Numeric measures** (`unit_spend`, `unit_quantity`, `unit_cost`) must be numeric types so they sum and average
  correctly. A spend column read as a string will not aggregate.

### Nulls

A null in a grouping or identifier column becomes its own group rather than being dropped, which sometimes is not what
you want. Decide on null handling deliberately:

- Filter or impute nulls in `customer_id`, `transaction_date`, and any `segment_col` before analysis.
- A null in a measure column (`unit_spend`) propagates through sums; clean these at the source.
- A negative `unit_spend` row is a return or refund. It is valid data, but decide deliberately whether to include or
  exclude returns, since most spend metrics assume positive transactions.

## Putting it together

Before running an analysis, confirm your table:

- [ ] Is a single flat table at line-item or transaction granularity.
- [ ] Has one column per identifier (no compound keys).
- [ ] Uses the standard column names, or has them mapped via the options system.
- [ ] Stores dates/times as timezone-naive temporal types, not strings.
- [ ] Stores `unit_spend` and other measures as numeric types.
- [ ] Has nulls in key and grouping columns handled deliberately.
- [ ] Is filtered to the scope of your question (see [Connecting to your data](connecting_to_data.md)).

With the table prepared, a complete analysis against the sample data is just connect, filter, run:

```python
import ibis
from openretailscience.segmentation.segstats import SegTransactionStats

con = ibis.duckdb.connect()
transactions = con.read_parquet("data/transactions.parquet")

q1 = transactions.filter(transactions.transaction_date.between("2023-01-01", "2023-03-31"))

stats = SegTransactionStats(data=q1, segment_col="category_0_name", grouping_sets="total")
print(stats.df)
```

## Related documentation

- [Connecting to your data](connecting_to_data.md): backends, connections, and filtering at the source.
- [Options & configuration](options_guide.md): mapping your column names onto the standard ones.
- [SegTransactionStats API reference](../api/segmentation/segstats.md): the function used in the examples above.
- [Analysis modules](../analysis_modules.md): an overview of every analysis the package provides.
