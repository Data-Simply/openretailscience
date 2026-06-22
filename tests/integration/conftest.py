"""Unified integration test fixtures for multiple database backends."""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import ibis
import pandas as pd
import pytest

if TYPE_CHECKING:
    from collections.abc import Callable

    from ibis.backends import BaseBackend
    from ibis.expr.types import Table

_TRANSACTIONS_PARQUET = "data/transactions.parquet"
_TRANSACTIONS_TABLE_NAME = "transactions"
_DEFAULT_MSSQL_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"
_DEFAULT_MSSQL_PORT = 1433
_DEFAULT_ORACLE_PORT = 1521
# Containers accept TCP connections before the engine is ready to authenticate,
# so connection attempts are retried for a short window while the database starts.
_CONNECT_MAX_ATTEMPTS = 30
_CONNECT_RETRY_SECONDS = 2.0


def _read_transactions() -> pd.DataFrame:
    """Read the transactions sample data used to seed container backends.

    Returns:
        pd.DataFrame: The transactions fixture data loaded from parquet.
    """
    return pd.read_parquet(_TRANSACTIONS_PARQUET)


def _connect_with_retry(connect: Callable[[], BaseBackend]) -> BaseBackend:
    """Establish a backend connection, retrying while the container starts up.

    Args:
        connect: Zero-argument callable that opens and returns a backend connection.

    Returns:
        BaseBackend: The established Ibis backend connection.

    Raises:
        RuntimeError: If no connection succeeds within the retry budget.
    """
    last_error: Exception | None = None
    for _ in range(_CONNECT_MAX_ATTEMPTS):
        try:
            return connect()
        except Exception as error:  # noqa: BLE001, PERF203 - retry loop; readiness raises driver-specific errors
            last_error = error
            time.sleep(_CONNECT_RETRY_SECONDS)
    error_msg = f"Could not connect to backend within {_CONNECT_MAX_ATTEMPTS} attempts"
    raise RuntimeError(error_msg) from last_error


def _seed_transactions(connection: BaseBackend) -> Table:
    """Load the transactions sample data into a connected backend and return it.

    Args:
        connection: An Ibis backend connection to seed.

    Returns:
        Table: The seeded transactions table expression.
    """
    df = _read_transactions()
    connection.create_table(_TRANSACTIONS_TABLE_NAME, df, overwrite=True)
    return connection.table(_TRANSACTIONS_TABLE_NAME)


@pytest.fixture(scope="session")
def _mssql_transactions_table() -> Table:
    """Seed transactions into a containerized SQL Server backend once per session.

    The fixture is skipped unless ``MSSQL_HOST`` is set, so the default local and
    cloud-backend test runs are unaffected. It creates the target database if it does
    not already exist, then loads the sample data into it.

    Returns:
        Table: The transactions table on the SQL Server backend.
    """
    host = os.environ.get("MSSQL_HOST")
    if host is None:
        pytest.skip("SQL Server integration backend is not configured (set MSSQL_HOST)")

    port = int(os.environ.get("MSSQL_PORT", str(_DEFAULT_MSSQL_PORT)))
    user = os.environ["MSSQL_USER"]
    password = os.environ["MSSQL_PASSWORD"]
    database = os.environ["MSSQL_DATABASE"]
    driver = os.environ.get("MSSQL_ODBC_DRIVER", _DEFAULT_MSSQL_ODBC_DRIVER)

    admin = _connect_with_retry(
        lambda: ibis.mssql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database="master",
            driver=driver,
            TrustServerCertificate="yes",
            autocommit=True,
        ),
    )
    if database not in admin.list_catalogs():
        admin.create_catalog(database)
    admin.disconnect()

    connection = ibis.mssql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=database,
        driver=driver,
        TrustServerCertificate="yes",
    )
    return _seed_transactions(connection)


@pytest.fixture(scope="session")
def _oracle_transactions_table() -> Table:
    """Seed transactions into a containerized Oracle backend once per session.

    The fixture is skipped unless ``ORACLE_HOST`` is set, so the default local and
    cloud-backend test runs are unaffected. It connects in python-oracledb thin mode,
    which requires no Oracle client libraries.

    Returns:
        Table: The transactions table on the Oracle backend.
    """
    host = os.environ.get("ORACLE_HOST")
    if host is None:
        pytest.skip("Oracle integration backend is not configured (set ORACLE_HOST)")

    connection = _connect_with_retry(
        lambda: ibis.oracle.connect(
            host=host,
            port=int(os.environ.get("ORACLE_PORT", str(_DEFAULT_ORACLE_PORT))),
            user=os.environ["ORACLE_USER"],
            password=os.environ["ORACLE_PASSWORD"],
            service_name=os.environ["ORACLE_SERVICE_NAME"],
        ),
    )
    return _seed_transactions(connection)


@pytest.fixture(
    params=["bigquery", "pyspark", "snowflake", "mssql", "oracle"],
    ids=lambda backend: f"backend={backend}",
)
def transactions_table(request):
    """Parameterized fixture that provides transactions table from different backends."""
    if request.param == "bigquery":
        connection = ibis.bigquery.connect(
            project_id=os.environ["GCP_PROJECT_ID"],
        )
        return connection.table("test_data.transactions")
    if request.param == "pyspark":
        connection = ibis.pyspark.connect()
        # Use pandas to read the parquet file first, then convert to Spark
        # This handles timestamp compatibility issues automatically
        df = pd.read_parquet("data/transactions.parquet")
        # Pyspark has no time column so we have to convert it to a datetime
        df["transaction_time"] = pd.to_datetime(
            df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str),
        )
        spark_df = connection._session.createDataFrame(df)
        # Create a temporary view and read it back as an ibis table
        spark_df.createOrReplaceTempView("transactions")
        return connection.table("transactions")
    if request.param == "snowflake":
        connection = ibis.snowflake.connect(
            account=os.environ["SNOWFLAKE_CI_ACCOUNT"],
            user=os.environ["SNOWFLAKE_CI_USER"],
            private_key_file=os.environ["SNOWFLAKE_CI_PRIVATE_KEY_PATH"],
            database=os.environ["SNOWFLAKE_CI_DATABASE"],
            schema=os.environ["SNOWFLAKE_CI_SCHEMA"],
            warehouse=os.environ["SNOWFLAKE_CI_WAREHOUSE"],
        )
        table = connection.table("TRANSACTIONS")
        # Snowflake returns UPPERCASE column names; lowercase them for compatibility with integration tests
        return table.rename({col.lower(): col for col in table.columns})
    if request.param in ("mssql", "oracle"):
        # Containerized backends are seeded once per session by their own fixtures.
        return request.getfixturevalue(f"_{request.param}_transactions_table")
    error_msg = f"Unknown backend: {request.param}"
    raise ValueError(error_msg)
