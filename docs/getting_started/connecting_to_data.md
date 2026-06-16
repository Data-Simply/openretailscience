# Connecting to your data

OpenRetailScience analyses accept two kinds of input: a pandas DataFrame or an
[Ibis](https://ibis-project.org/) table. Ibis lets you write one analysis and run it against DuckDB, BigQuery,
Snowflake, Databricks, SQL Server, Microsoft Fabric, Oracle, and other backends without changing your code.

This guide shows how to connect Ibis to each of those sources, how to hand the resulting table to a function such as
`SegTransactionStats`, and how to filter your data before analysis so the database does the heavy lifting.

## Why use Ibis

When you pass an Ibis table to an OpenRetailScience function, the filtering and aggregation run inside your database or
warehouse. Ibis compiles your expression to SQL, the engine executes it, and only the result comes back to Python.

This matters once your data outgrows memory:

- **Query pushdown**: filters and aggregations run in the database, close to the data, not in a Python loop.
- **Less data movement**: only the aggregated result crosses the network, not the raw transaction rows.
- **Lower memory use**: you never materialize the full table in Python.

A pandas DataFrame still works for smaller datasets and quick experiments. The same OpenRetailScience code runs on
either input, so you can prototype on a pandas sample and switch to a warehouse table for production without rewriting
your analysis.

## Install the backend you need

The core install includes DuckDB. Every other backend is an optional Ibis extra. Install the one that matches your
source:

```bash
pip install "ibis-framework[duckdb]"      # local files and the sample data
pip install "ibis-framework[bigquery]"    # Google BigQuery
pip install "ibis-framework[snowflake]"   # Snowflake
pip install "ibis-framework[databricks]"  # Databricks
pip install "ibis-framework[mssql]"       # SQL Server and Microsoft Fabric
pip install "ibis-framework[oracle]"      # Oracle Database
```

## Quick start with the sample data

The repository ships a sample dataset at
[`data/transactions.parquet`](https://github.com/Data-Simply/openretailscience/blob/main/data/transactions.parquet).
It holds 127,180 rows of synthetic retail transactions for the 2023 calendar year. DuckDB reads Parquet files
directly, so no server or credentials are required.

```python
import ibis

con = ibis.duckdb.connect()
transactions = con.read_parquet("data/transactions.parquet")

print(transactions.schema())
```

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

Now filter to the first quarter and run `SegTransactionStats` to compare categories. The `grouping_sets="total"`
argument adds a grand total row:

```python
from openretailscience.segmentation.segstats import SegTransactionStats

q1 = transactions.filter(
    (transactions.transaction_date >= "2023-01-01")
    & (transactions.transaction_date <= "2023-03-31")
)

stats = SegTransactionStats(data=q1, segment_col="category_0_name", grouping_sets="total")
print(stats.df)
```

```text
category_0_name        spend  transactions  customers  spend_per_customer  spend_per_transaction
         Movies    113000.65          2817       1971               57.33                  40.11
          Music  13079701.33          2830       1971             6636.07                4621.80
       Clothing   1168927.06          2809       1987              588.29                 416.14
           Home   1911636.02          2735       1933              988.95                 698.95
           Toys    264865.42          2795       1956              135.41                  94.76
        Grocery     23073.07          2752       1921               12.01                   8.38
         Beauty    217913.04          2924       2016              108.09                  74.53
          Books     69539.48          2862       1975               35.21                  24.30
    Electronics   4863411.56          2865       2003             2428.06                1697.53
         Sports   3980510.10          2745       1948             2043.38                1450.09
          Total  25692577.73          7275       2873             8942.77                3531.63
```

The output above omits the quantity-derived columns for width. The full result also includes `units`,
`price_per_unit`, `transactions_per_customer`, and `units_per_transaction`.

## Connecting to your data source

Every backend follows the same shape: open a connection, then get a table reference with `con.table("table_name")`.
After that, the table behaves the same regardless of where it lives.

!!! warning "Never hardcode credentials"
    Read passwords and tokens from environment variables or a secrets manager. Do not paste them into notebooks or
    commit them to version control. The examples below use `os.environ` for this reason.

### DuckDB (local files)

DuckDB runs in-process and reads Parquet, CSV, and its own database files. Use it for local files, the sample data,
and testing.

```python
import ibis

# In-memory connection over a Parquet file
con = ibis.duckdb.connect()
transactions = con.read_parquet("data/transactions.parquet")

# Or open a persisted DuckDB database and reference a table inside it
con = ibis.duckdb.connect("warehouse.ddb")
transactions = con.table("transactions")
```

### Google BigQuery

```python
import ibis

con = ibis.bigquery.connect(project_id="retail-analytics", dataset_id="sales_data")
transactions = con.table("transactions")
```

For authentication, application default credentials are the simplest option. Run
`gcloud auth application-default login` once, and the connection picks up the credentials automatically. For a service
account, pass a `google.oauth2.service_account.Credentials` object to the `credentials` argument instead.

!!! tip "Control BigQuery cost"
    BigQuery bills by the number of bytes scanned. Filtering on a partitioned or clustered column (often the
    transaction date) reduces both query time and cost, because the engine skips partitions outside your range.

### Snowflake

```python
import os
import ibis

con = ibis.snowflake.connect(
    user="analyst",
    account="myorg-myaccount",
    database="RETAIL_DB/SALES",
    warehouse="ANALYTICS_WH",
    password=os.environ["SNOWFLAKE_PASSWORD"],
)
transactions = con.table("transactions")
```

The `account` value is your Snowflake organization and account identifier joined by a hyphen. The `database` value
combines the database and schema separated by a slash. To use single sign-on or key-pair authentication instead of a
password, pass the `authenticator` argument (see the
[Snowflake backend docs](https://ibis-project.org/backends/snowflake)).

### Databricks

```python
import os
import ibis

con = ibis.databricks.connect(
    server_hostname="adb-1234567890.11.azuredatabricks.net",
    http_path="/sql/1.0/warehouses/abc123def456",
    access_token=os.environ["DATABRICKS_TOKEN"],
    catalog="retail",
    schema="sales",
)
transactions = con.table("transactions")
```

Find the `server_hostname` and `http_path` under the connection details of your SQL warehouse in the Databricks
workspace. The `catalog` and `schema` arguments set the Unity Catalog location that table names resolve against.

### Microsoft SQL Server

```python
import os
import ibis

# SQL Server authentication
con = ibis.mssql.connect(
    host="sql-server.example.com",
    port=1433,
    database="RetailDW",
    user="analyst",
    password=os.environ["MSSQL_PASSWORD"],
    driver="ODBC Driver 18 for SQL Server",
)
transactions = con.table("transactions", database="dbo")
```

For Windows integrated authentication, leave `user` and `password` unset and the driver uses your current Windows
identity:

```python
con = ibis.mssql.connect(
    host="sql-server.example.com",
    database="RetailDW",
    driver="ODBC Driver 18 for SQL Server",
)
```

SQL authentication needs a username and password and works from any platform. Windows authentication avoids storing a
password but only works inside the Active Directory domain. The connection needs an ODBC driver installed: "ODBC Driver
18 for SQL Server" on Windows, or FreeTDS on macOS and Linux.

### Microsoft Fabric

A Fabric warehouse and the SQL analytics endpoint of a lakehouse both speak the SQL Server wire protocol, so the
`mssql` backend connects to them. Point `host` at the Fabric SQL connection string and authenticate with Azure Active
Directory by passing the ODBC `Authentication` keyword through to the driver:

```python
import ibis

con = ibis.mssql.connect(
    host="xxxxxxxx.datawarehouse.fabric.microsoft.com",
    database="SalesLakehouse",
    driver="ODBC Driver 18 for SQL Server",
    Authentication="ActiveDirectoryInteractive",
)
transactions = con.table("transactions")
```

The SQL analytics endpoint is read-only and exposes the tables of a lakehouse, which suits analysis with
OpenRetailScience. A Fabric warehouse supports reads and writes. Both use the
`<workspace>.datawarehouse.fabric.microsoft.com` host pattern. For unattended jobs, use
`Authentication="ActiveDirectoryServicePrincipal"` with a client ID and secret rather than the interactive flow.

### Oracle Database

```python
import os
import ibis

con = ibis.oracle.connect(
    user="analyst",
    password=os.environ["ORACLE_PASSWORD"],
    host="oracle.example.com",
    port=1521,
    service_name="ORCLPDB1",
)
transactions = con.table("transactions")
```

Identify the database by its `service_name`, its `sid`, or a full `dsn`. Supply only one of these. If your tables live
in another user's schema, qualify the table name with `con.table("transactions", database="SALES")`.

## Filter before you analyze

Filtering your data before analysis is the expected workflow, not an optimization you add later. Most business
questions cover a specific time period, and often a specific category, region, or set of stores. Filter to that scope
first, then analyze.

Date filtering applies to nearly every analysis. Decide the period your question covers (this quarter, year to date,
the last 12 months) and filter to it:

```python
q1_data = transactions.filter(
    (transactions.transaction_date >= "2023-01-01")
    & (transactions.transaction_date <= "2023-03-31")
)
```

Combine the date range with category, store, or other dimension filters as the question requires:

```python
electronics_q1 = transactions.filter(
    (transactions.transaction_date >= "2023-01-01")
    & (transactions.transaction_date <= "2023-03-31")
    & (transactions.category_0_name == "Electronics")
)

stats = SegTransactionStats(data=electronics_q1, segment_col="brand_name", grouping_sets="total")
```

## Why filtering at the source matters

Because the filter compiles to a SQL `WHERE` clause, the database applies it before any data reaches Python. The
effect grows with the size of your table:

- **Query pushdown**: the engine evaluates the filter where the data lives, using its indexes and partitions.
- **Smaller transfers**: a quarter of electronics sales is a fraction of several years of every category.
- **Lower memory use**: Python holds the aggregated result, not the raw rows.
- **Faster execution**: warehouses are built to scan and filter large tables; a Python loop is not.

A filter that cuts a billion-row table down to a few million rows before aggregation often runs orders of magnitude
faster than scanning everything.

## End-to-end example

This example connects to the sample data, filters to first-quarter electronics sales, and compares subcategories by
spend. The same three steps (connect, filter, analyze) apply to every backend above. Only the connection line changes.

```python
import ibis
from openretailscience.segmentation.segstats import SegTransactionStats

# 1. Connect to your data source
con = ibis.duckdb.connect()
transactions = con.read_parquet("data/transactions.parquet")

# 2. Filter to the scope of the business question
q1_electronics = transactions.filter(
    (transactions.transaction_date >= "2023-01-01")
    & (transactions.transaction_date <= "2023-03-31")
    & (transactions.category_0_name == "Electronics")
)

# 3. Run the analysis
stats = SegTransactionStats(
    data=q1_electronics,
    segment_col="category_1_name",
    grouping_sets="total",
)
print(stats.df)
```

```text
    category_1_name      spend  transactions  customers  spend_per_customer  spend_per_transaction
            Laptops 1438389.00           536        496             2899.98                2683.56
        Televisions 1133750.00           459        434             2612.33                2470.04
        Smartphones  631928.22           507        481             1313.78                1246.41
Computer Components  603245.73           609        566             1065.81                 990.55
            Tablets  478411.92           505        474             1009.31                 947.35
    Audio Equipment  357203.22           465        440              811.83                 768.18
Wearable Technology  220483.47           512        486              453.67                 430.63
              Total 4863411.56          2865       2003             2428.06                1697.53
```

To run the same analysis against BigQuery, replace step 1 with `con = ibis.bigquery.connect(...)` and
`transactions = con.table("transactions")`. Steps 2 and 3 stay the same.

## Handling credentials securely

- Read passwords, tokens, and connection strings from environment variables or a secrets manager (such as AWS Secrets
  Manager, Azure Key Vault, or Google Secret Manager). The examples above use `os.environ` for this.
- Keep credentials out of notebooks and source control. Add credential files to `.gitignore`.
- Prefer managed identity where the platform offers it: application default credentials on BigQuery, Azure Active
  Directory on SQL Server and Fabric, and OAuth or key-pair authentication on Snowflake.

## Troubleshooting

### Connection or driver errors

A `ModuleNotFoundError` for a backend means the Ibis extra is not installed. Install it with, for example,
`pip install "ibis-framework[snowflake]"`. On SQL Server, Fabric, and Oracle you also need the database client
installed on the machine: an ODBC driver for SQL Server and Fabric, and the Oracle client libraries for Oracle.

### Authentication failures

Confirm the account, host, and database identifiers first, since a typo there often surfaces as an auth error.
Snowflake defers authentication until the first query, so `ibis.snowflake.connect(...)` can succeed while a later call
fails. For Azure Active Directory on Fabric, an interactive sign-in needs a browser; switch to a service principal for
unattended jobs.

### Column not found

OpenRetailScience expects standard column names such as `customer_id`, `transaction_id`, and `unit_spend`. If your
table uses different names, configure them rather than renaming every column. See the
[Options and configuration guide](options_guide.md) for how to map your names to the ones OpenRetailScience uses.

### Slow queries

Check that you filtered before the analysis. A query that scans the whole table will be slow no matter where it runs.
On BigQuery and Snowflake, filtering on the partition or clustering column gives the largest speedup.

## Related documentation

- [Options and configuration guide](options_guide.md): map your column names to the ones OpenRetailScience expects.
- [SegTransactionStats API reference](../api/segmentation/segstats.md): every argument of the function used above.
- [Ibis backend documentation](https://ibis-project.org/backends/): connection details for every supported backend.
