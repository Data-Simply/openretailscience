# Connecting to Your Data

OpenRetailScience is built on [Apache Ibis](https://ibis-project.org/), so every analysis and plot accepts either a
**pandas DataFrame** or an **Ibis table**. Passing an Ibis table means the work is pushed down to your database engine
and only the results come back to Python — which matters when your transaction tables are in the 1B–10B row range.

This guide shows how to connect each supported backend and hand the resulting table to OpenRetailScience.

!!! tip "Column names"
    Whatever backend you use, your columns rarely match the internal defaults. After connecting, configure the column
    mapping as described in the [Options & Configuration guide](options_guide.md) — either by renaming columns on the
    Ibis table or via an `openretailscience.toml` file.

## Pandas (in-memory)

If your data already fits in memory, just pass the DataFrame directly. OpenRetailScience converts it to an in-memory
Ibis table for you.

```python
import pandas as pd
from openretailscience.analysis.gain_loss import GainLoss

df = pd.read_parquet("transactions.parquet")

gl = GainLoss(df, ...)
```

This is ideal for samples, tests, and small extracts. For larger datasets, connect to one of the backends below and
let the database do the heavy lifting.

## DuckDB

DuckDB ships with OpenRetailScience by default, so nothing extra to install. It is a great choice for analysing
local Parquet/CSV files or a `.duckdb` database without standing up a warehouse.

```python
import ibis

con = ibis.duckdb.connect("retail.duckdb")  # or connect() for an in-memory database
table = con.table("transactions")

# Query Parquet/CSV files directly, no import step required
table = con.read_parquet("transactions/*.parquet")
```

## BigQuery

Install the BigQuery backend:

```bash
pip install "ibis-framework[bigquery]"
```

Connect with your GCP project and reference the table by `dataset.table`:

```python
import ibis

con = ibis.bigquery.connect(project_id="my-gcp-project")
table = con.table("retail_analytics.transactions")
```

Authentication uses [Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).
Run `gcloud auth application-default login` locally, or rely on the attached service account when running on GCP.

## Snowflake

Install the Snowflake backend:

```bash
pip install "ibis-framework[snowflake]"
```

```python
import ibis

con = ibis.snowflake.connect(
    account="my_org-my_account",
    user="ANALYST",
    private_key_file="/path/to/rsa_key.p8",  # or password="..."
    database="RETAIL",
    schema="ANALYTICS",
    warehouse="COMPUTE_WH",
)
table = con.table("TRANSACTIONS")
```

!!! warning "Snowflake uppercases identifiers"
    Snowflake returns column names in uppercase. If your analysis expects lowercase names, normalise them after
    connecting:

    ```python
    table = table.rename({col.lower(): col for col in table.columns})
    ```

## Microsoft SQL Server

Install the MS SQL Server backend (which uses [pyodbc](https://github.com/mkleehammer/pyodbc) under the hood, so you
also need a Microsoft ODBC driver installed on your system):

```bash
pip install "ibis-framework[mssql]"
```

```python
import ibis

con = ibis.mssql.connect(
    host="my-sql-server.database.windows.net",
    user="analyst",
    password="...",
    database="retail",
    port=1433,
    driver="ODBC Driver 18 for SQL Server",
)
table = con.table("transactions")
```

Make sure the matching ODBC driver (for example
[ODBC Driver 18 for SQL Server](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server))
is installed and that its name matches the `driver` argument.

## PySpark

Install the PySpark backend:

```bash
pip install "ibis-framework[pyspark]"
```

Connect to an existing Spark session (or let Ibis create a local one) and reference a table or temporary view:

```python
import ibis

con = ibis.pyspark.connect()  # pass an existing SparkSession to reuse your cluster
table = con.table("transactions")
```

## Next steps

Once you have a table, point your column mapping at it and start analysing — see the
[Options & Configuration guide](options_guide.md) and the [Analysis Modules](../analysis_modules.md) overview.
