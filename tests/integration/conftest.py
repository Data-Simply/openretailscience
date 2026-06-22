"""Unified integration test fixtures for multiple database backends."""

from __future__ import annotations

import os
import socket
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

# Connection details for the local throwaway containers defined in
# tests/integration/docker/. These are fixed, non-secret values that must match the
# corresponding docker-compose files; they are intentionally hardcoded rather than
# configured via environment variables. The only per-version knob is the Oracle PDB
# service name (XEPDB1 for XE, FREEPDB1 for 23ai Free), which the CI matrix overrides.
_MSSQL_HOST = "localhost"
_MSSQL_PORT = 1433
_MSSQL_USER = "sa"
_MSSQL_PASSWORD = "orsTest!Passw0rd"  # noqa: S105 - local throwaway container credential
_MSSQL_DATABASE = "openretailscience"
_MSSQL_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"

_ORACLE_HOST = "localhost"
_ORACLE_PORT = 1521
_ORACLE_USER = "ors"
_ORACLE_PASSWORD = "orsTestApp1"  # noqa: S105 - local throwaway container credential
_DEFAULT_ORACLE_SERVICE_NAME = "FREEPDB1"

# Containers accept TCP connections before the engine is ready to authenticate,
# so connection attempts are retried for a short window while the database starts.
_PORT_PROBE_TIMEOUT_SECONDS = 1.0
_CONNECT_MAX_ATTEMPTS = 30
_CONNECT_RETRY_SECONDS = 2.0


def _read_transactions() -> pd.DataFrame:
    """Read the transactions sample data used to seed container backends.

    Returns:
        pd.DataFrame: The transactions fixture data loaded from parquet.
    """
    return pd.read_parquet(_TRANSACTIONS_PARQUET)


def _require_container_reachable(host: str, port: int, name: str) -> None:
    """Fail loudly if the backend container is not listening on host:port.

    This is a fast pre-flight check so a missing or failed-to-start container surfaces
    as an immediate, clear error rather than skipping (which would hide the problem) or
    burning the whole connection-retry budget. It deliberately does not call
    ``pytest.skip``: these tests are only selected when the backend is meant to run.

    Args:
        host: Host the container is expected to listen on.
        port: Port the container is expected to listen on.
        name: Human-readable backend name used in the error message.

    Raises:
        RuntimeError: If nothing is accepting connections on host:port.
    """
    try:
        with socket.create_connection((host, port), timeout=_PORT_PROBE_TIMEOUT_SECONDS):
            return
    except OSError as error:
        error_msg = (
            f"{name} container not reachable at {host}:{port}; "
            "start it first (see tests/integration/docker/)"
        )
        raise RuntimeError(error_msg) from error


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

    Requires the SQL Server container (see tests/integration/docker/) to be running; an
    unreachable container fails loudly rather than skipping, so a container that failed
    to start is never silently passed over. Creates the target database if it does not
    already exist, then loads the sample data into it.

    Returns:
        Table: The transactions table on the SQL Server backend.
    """
    _require_container_reachable(_MSSQL_HOST, _MSSQL_PORT, "SQL Server")
    admin = _connect_with_retry(
        lambda: ibis.mssql.connect(
            host=_MSSQL_HOST,
            port=_MSSQL_PORT,
            user=_MSSQL_USER,
            password=_MSSQL_PASSWORD,
            database="master",
            driver=_MSSQL_ODBC_DRIVER,
            TrustServerCertificate="yes",
            autocommit=True,
        ),
    )
    if _MSSQL_DATABASE not in admin.list_catalogs():
        admin.create_catalog(_MSSQL_DATABASE)
    admin.disconnect()

    connection = ibis.mssql.connect(
        host=_MSSQL_HOST,
        port=_MSSQL_PORT,
        user=_MSSQL_USER,
        password=_MSSQL_PASSWORD,
        database=_MSSQL_DATABASE,
        driver=_MSSQL_ODBC_DRIVER,
        TrustServerCertificate="yes",
    )
    return _seed_transactions(connection)


@pytest.fixture(scope="session")
def _oracle_transactions_table() -> Table:
    """Seed transactions into a containerized Oracle backend once per session.

    Requires the Oracle container (see tests/integration/docker/) to be running; an
    unreachable container fails loudly rather than skipping, so a container that failed
    to start is never silently passed over. Connects in python-oracledb thin mode,
    which requires no Oracle client libraries. The PDB service name defaults to 23ai
    Free's ``FREEPDB1`` and is overridden to ``XEPDB1`` for the XE matrix entries.

    Returns:
        Table: The transactions table on the Oracle backend.
    """
    _require_container_reachable(_ORACLE_HOST, _ORACLE_PORT, "Oracle")
    connection = _connect_with_retry(
        lambda: ibis.oracle.connect(
            host=_ORACLE_HOST,
            port=_ORACLE_PORT,
            user=_ORACLE_USER,
            password=_ORACLE_PASSWORD,
            service_name=os.environ.get("ORACLE_SERVICE_NAME", _DEFAULT_ORACLE_SERVICE_NAME),
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
