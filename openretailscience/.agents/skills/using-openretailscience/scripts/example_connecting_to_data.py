"""Connecting openretailscience to data via Ibis: analyses run as SQL in your database, and every module accepts a pandas DataFrame or an Ibis table. DuckDB/pandas examples run locally; credentialed-backend examples are copy-and-call functions."""

import ibis
import numpy as np
import pandas as pd

# Sample data for the runnable examples
rng = np.random.default_rng(42)
n_rows = 1_000
sample_transactions = pd.DataFrame(
    {
        "transaction_id": np.arange(1, n_rows + 1),
        "customer_id": rng.integers(1, 201, size=n_rows),
        "transaction_date": pd.to_datetime("2023-01-01")
        + pd.to_timedelta(rng.integers(0, 365, size=n_rows), unit="D"),
        "category_0_name": rng.choice(["Electronics", "Grocery", "Apparel"], size=n_rows),
        "category_1_name": rng.choice(["Phones", "Laptops", "Snacks", "Shirts"], size=n_rows),
        "unit_spend": np.round(rng.uniform(5, 250, size=n_rows), 2),
    }
)


# Example 1: DuckDB from an in-memory pandas DataFrame (fully runnable)
def example_duckdb_in_memory():
    """Register a pandas DataFrame as an Ibis/DuckDB table and inspect it."""
    con = ibis.duckdb.connect()
    transactions = con.create_table("transactions", sample_transactions)
    transactions.count().execute()  # row count
    transactions.columns  # result columns
    return con, transactions


# Example 2: DuckDB reading a local Parquet/CSV file
def example_duckdb_local_file(parquet_path="data/transactions.parquet"):
    """Connect DuckDB to a local Parquet (or CSV) file without loading it into memory."""
    con = ibis.duckdb.connect()
    transactions = con.read_parquet(parquet_path)
    # For CSV: transactions = con.read_csv("data/transactions.csv")
    return con, transactions


# Example 3: Google BigQuery
def example_bigquery():
    """Connect to a BigQuery dataset and grab a table."""
    con = ibis.bigquery.connect(project_id="retail-analytics", dataset_id="sales_data")
    transactions = con.table("transactions")
    return con, transactions


# Example 4: Snowflake (credentials read from the environment)
def example_snowflake():
    """Connect to Snowflake, reading the password from an environment variable."""
    import os

    con = ibis.snowflake.connect(
        user="analyst",
        account="myorg-myaccount",
        database="RETAIL_DB/SALES",
        warehouse="ANALYTICS_WH",
        password=os.environ["SNOWFLAKE_PASSWORD"],
    )
    transactions = con.table("transactions")
    return con, transactions


# Example 5: Databricks / PySpark
def example_databricks_external():
    """Connect to Databricks SQL warehouse from outside the workspace."""
    import os

    con = ibis.databricks.connect(
        server_hostname=os.environ["DATABRICKS_SERVER_HOSTNAME"],
        http_path=os.environ["DATABRICKS_HTTP_PATH"],
        access_token=os.environ["DATABRICKS_TOKEN"],
        catalog="retail",
        schema="sales",
    )
    return con.table("transactions")


def example_pyspark():
    """Connect through a PySpark session (e.g. inside a Spark cluster)."""
    from pyspark.sql import SparkSession

    session = SparkSession.builder.getOrCreate()
    con = ibis.pyspark.connect(session)
    return con.table("transactions")


# Example 6: Microsoft SQL Server / Microsoft Fabric / Oracle
def example_sql_server():
    """Connect to SQL Server via the ODBC driver."""
    import os

    con = ibis.mssql.connect(
        host="sql-server.example.com",
        port=1433,
        database="RetailDW",
        user="analyst",
        password=os.environ["MSSQL_PASSWORD"],
        driver="ODBC Driver 18 for SQL Server",
    )
    return con.table("transactions", database="dbo")


def example_fabric():
    """Connect to a Microsoft Fabric warehouse with Active Directory auth."""
    con = ibis.mssql.connect(
        host="xxxxxxxx.datawarehouse.fabric.microsoft.com",
        database="SalesLakehouse",
        driver="ODBC Driver 18 for SQL Server",
        Authentication="ActiveDirectoryInteractive",
    )
    return con.table("transactions")


def example_oracle():
    """Connect to an Oracle database."""
    import os

    con = ibis.oracle.connect(
        user="analyst",
        password=os.environ["ORACLE_PASSWORD"],
        host="oracle.example.com",
        port=1521,
        service_name="ORCLPDB1",
    )
    return con.table("transactions")


# Example 7: The connect -> filter -> analyze workflow (fully runnable)
def example_connect_filter_analyze():
    """Connect, filter in the database, then hand the table to an analysis module."""
    con = ibis.duckdb.connect()
    transactions = con.create_table("transactions", sample_transactions)

    # Filter runs in the database (lazy)
    q1_electronics = transactions.filter(
        transactions.transaction_date.between("2023-01-01", "2023-03-31")
        & (transactions.category_0_name == "Electronics")
    )

    from openretailscience.segmentation.segstats import SegTransactionStats

    stats = SegTransactionStats(
        data=q1_electronics,
        segment_col="category_1_name",
        grouping_sets="total",
    )
    return stats


# Example 8: Use a pandas DataFrame directly (small-data path)
def example_pandas_path():
    """Every module also accepts a plain pandas DataFrame for small extracts."""
    from openretailscience.segmentation.rfm import RFMSegmentation

    rfm = RFMSegmentation(df=sample_transactions, current_date="2024-01-01")
    return rfm


if __name__ == "__main__":
    # Runnable, no-credentials examples:
    example_duckdb_in_memory()
    example_connect_filter_analyze()
    example_pandas_path()
    # Remaining functions need live credentials and the matching ibis-framework[<backend>] extra.
