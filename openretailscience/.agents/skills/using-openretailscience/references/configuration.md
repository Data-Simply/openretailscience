# Configuration & data connection reference

## The options system (`openretailscience.options`)

Different retailers name the same field differently (`customer_id` vs `cid` vs
`token_id`). OpenRetailScience uses canonical internal names that you **override**
through options, so one analysis runs on any schema.

```python
from openretailscience.options import get_option, set_option, option_context, reset_option
```

- `set_option(key, value)` / `get_option(key)` — read/write one dotted option. Only override
  options that actually differ from your schema — setting e.g. `column.customer_id` to
  `"customer_id"` (its own default, see below) is a no-op.
- `option_context(...)` — temporary override in a `with` block; accepts a dict
  **or** alternating name/value pairs, and restores prior values on exit.
- `reset_option(key)`, plus `list_options()` and `describe_option(key)` for discovery.

```python
from openretailscience.analysis.gain_loss import GainLoss

with option_context(
    "column.customer_id", "cust_id",
    "column.unit_spend", "revenue",
    "column.transaction_date", "trans_dt",
):
    gl = GainLoss(transactions, ...)   # analyses now find your columns
```

Prefer `option_context` for scoped/library use; use `set_option` for a whole
script; use an `openretailscience.toml` for a whole project (below). If you mix
methods, the last write wins.

### Key option namespaces (defaults)

- `column.*` — source columns: `customer_id`, `transaction_id`, `transaction_date`,
  `transaction_time`, `product_id`, `store_id`, `unit_quantity`, `unit_price`,
  `unit_spend`, `unit_cost`, `promo_unit_spend`, `promo_unit_quantity`.
- `column.agg.*` — output aggregate names (e.g. `unit_spend`→"spend",
  `customer_id`→"customers", `transaction_id`→"transactions", `unit_quantity`→"units").
- `column.calc.*` — derived metric names: `spend_per_customer`, `spend_per_transaction`,
  `transactions_per_customer`, `units_per_transaction`, `price_per_unit`,
  `price_elasticity`, `frequency_elasticity`.
- `column.suffix.*` — suffixes: `period_1`→"p1", `period_2`→"p2", `difference`→"diff",
  `percent_difference`→"pct_diff", `contribution`→"contrib", `total`→"total".
- `plot.color.*`, `plot.font.*`, `plot.style.*`, `plot.spacing.*` — chart styling.
- `optimization.use_native_sql` — engine optimization toggle.
- `optimization.use_caching` — toggle for the experimental `cache()` helper.

### ColumnHelper — resolve names in your own code

```python
from openretailscience.options import ColumnHelper

cols = ColumnHelper()
cols.customer_id          # configured source customer column
cols.agg.unit_spend       # configured aggregate spend name (e.g. "spend")
cols.agg.unit_spend_p1    # period-1 variant
cols.calc.spend_per_customer
ColumnHelper.join_options("column.agg.unit_spend", "column.suffix.period_1")  # -> "spend_p1"
```

Use `ColumnHelper` (and `cols.agg.*` / `cols.calc.*`) instead of hardcoding
strings, so your code follows the configured names. `PlotStyleHelper()` exposes
the resolved `plot.*` styling values the same way.

### Project config file

Put an `openretailscience.toml` at the project root (a `.git` or this file marks
the root; `openretailscience.options.find_project_root()` locates it). It is
loaded on import; nested tables flatten to dotted keys. See `options_template.toml`
in the repo. Unknown keys raise `ValueError`.

## Connecting analyses to a database (Ibis)

Analyses accept a pandas DataFrame or an Ibis table; pandas is validated and
wrapped internally. Pattern: **connect → get a table reference → filter at the
source → pass to the analysis**, so aggregation runs in the engine.

```python
import ibis

con = ibis.duckdb.connect()                       # core install, no extras
transactions = con.read_parquet("data/transactions.parquet")
# or: con.table("transactions")
```

Backends (install the matching Ibis extra, then `con.table(...)`):

- **DuckDB** — `ibis.duckdb.connect()` / `ibis.duckdb.connect("warehouse.ddb")` (built in).
- **BigQuery** — `ibis.bigquery.connect(project_id=..., dataset_id=...)` —
  `ibis-framework[bigquery]`.
- **Snowflake** — `ibis.snowflake.connect(user=, account=, database="DB/SCHEMA", ...)` —
  `[snowflake]`.
- **Databricks** — from outside: `ibis.databricks.connect(server_hostname=, http_path=, ...)`
  — `[databricks]`; inside a notebook: `ibis.pyspark.connect(spark)`.
- **PySpark** — `ibis.pyspark.connect(session)` — `[pyspark]`.
- **MS SQL Server / Fabric** — `ibis.mssql.connect(host=, database=, user=, password=, ...)`
  — `[mssql]`.
- **Oracle** — `ibis.oracle.connect(user=, password=, host=, port=, service_name=)` —
  `[oracle]`.

Filter with Ibis before handing off, e.g.
`transactions.filter(transactions.transaction_date.between("2023-01-01", "2023-03-31"))`.
The sample dataset `data/transactions.parquet` (127,180 rows, 2023) uses the
default column names, so no configuration is needed to try things out.

## Colors constant

`openretailscience.constants.COLORS[hue][shade]` is the full Tailwind palette
(`COLORS["green"][500]`), hues `slate…rose`, shades `50…950`. Prefer the
`plots.styles.colors` helpers and `plot.color.*` options over reaching into
`COLORS` directly.
