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
    from collections.abc import Callable, Iterator

    from ibis.backends import BaseBackend
    from ibis.expr.types import Table

_TRANSACTIONS_PARQUET = "data/transactions.parquet"
_TRANSACTIONS_TABLE_NAME = "transactions"

# Connection details for the local throwaway containers defined in
# tests/integration/docker/. These are fixed, non-secret values that must match the
# corresponding docker-compose files.
_MSSQL_HOST = "localhost"
_MSSQL_PORT = 1433
_MSSQL_USER = "sa"
_MSSQL_PASSWORD = "orsTest!Passw0rd"  # noqa: S105 - local throwaway container credential
_MSSQL_ODBC_DRIVER = "ODBC Driver 18 for SQL Server"

_ORACLE_HOST = "localhost"
_ORACLE_PORT = 1521
_ORACLE_USER = "ors"
_ORACLE_PASSWORD = "orsTestApp1"  # noqa: S105 - local throwaway container credential
_ORACLE_SERVICE_NAME = "FREEPDB1"  # 23ai Free pluggable database

# Containers accept TCP connections before the engine is ready to authenticate,
# so connection attempts are retried for a short window while the database starts.
_PORT_PROBE_TIMEOUT_SECONDS = 1.0
_CONNECT_MAX_ATTEMPTS = 30
_CONNECT_RETRY_SECONDS = 2.0

# Ibis creates this volume on connect to stage memtables. Named, because the default embeds the
# process id, so every run would leave another volume behind in the CI schema.
_DATABRICKS_MEMTABLE_VOLUME = "ibis_memtables"


def _read_transactions() -> pd.DataFrame:
    """Read the transactions sample data used to seed container backends.

    Returns:
        pd.DataFrame: The transactions fixture data loaded from parquet.
    """
    df = pd.read_parquet(_TRANSACTIONS_PARQUET)
    # transaction_time is a bare time-of-day, which Ibis types as `time`. Oracle has no
    # TIME type (only DATE/TIMESTAMP), so seeding it as `time` fails with ORA-00902. The
    # column is not exercised by the tests, so store it as a string for portable seeding.
    df["transaction_time"] = df["transaction_time"].astype(str)
    return df


def _require_container_reachable(host: str, port: int, name: str) -> None:
    """Fail loudly if the backend container is not listening on host:port.

    Deliberately does not call ``pytest.skip``: these tests are only selected when the
    backend is meant to run, so a container that failed to start must not be passed over.

    Args:
        host (str): Host the container is expected to listen on.
        port (int): Port the container is expected to listen on.
        name (str): Human-readable backend name used in the error message.

    Raises:
        RuntimeError: If nothing is accepting connections on host:port.
    """
    try:
        with socket.create_connection((host, port), timeout=_PORT_PROBE_TIMEOUT_SECONDS):
            return
    except OSError as error:
        error_msg = f"{name} container not reachable at {host}:{port}; start it first (see tests/integration/docker/)"
        raise RuntimeError(error_msg) from error


def _connect_with_retry(connect: Callable[[], BaseBackend]) -> BaseBackend:
    """Establish a backend connection, retrying while the container starts up.

    Args:
        connect (Callable[[], BaseBackend]): Zero-argument callable that opens and returns a backend connection.

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
        connection (BaseBackend): An Ibis backend connection to seed.

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
    to start is never silently passed over. Seeds into the built-in ``master`` database
    so no separate database has to be created in the throwaway container.

    Returns:
        Table: The transactions table on the SQL Server backend.
    """
    _require_container_reachable(_MSSQL_HOST, _MSSQL_PORT, "SQL Server")
    connection = _connect_with_retry(
        lambda: ibis.mssql.connect(
            host=_MSSQL_HOST,
            port=_MSSQL_PORT,
            user=_MSSQL_USER,
            password=_MSSQL_PASSWORD,
            database="master",
            driver=_MSSQL_ODBC_DRIVER,
            TrustServerCertificate="yes",
        ),
    )
    return _seed_transactions(connection)


@pytest.fixture(scope="session")
def _oracle_transactions_table() -> Table:
    """Seed transactions into a containerized Oracle backend once per session.

    Requires the Oracle container (see tests/integration/docker/) to be running; an
    unreachable container fails loudly rather than skipping, so a container that failed
    to start is never silently passed over. Connects in python-oracledb thin mode,
    which requires no Oracle client libraries (23ai Free, service ``FREEPDB1``).

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
            service_name=_ORACLE_SERVICE_NAME,
        ),
    )
    return _seed_transactions(connection)


@pytest.fixture(scope="session")
def _databricks_transactions_table() -> Iterator[Table]:
    """Open one Databricks connection per session and close it on teardown.

    Closing matters: the Thrift transport is otherwise torn down by the garbage collector,
    which logs against pytest's already-closed capture stream.

    Yields:
        Table: The transactions table on the Databricks backend.
    """
    from databricks.sdk.core import Config  # noqa: PLC0415 - keeps the other backends collectable without it

    # Config() takes the OAuth credentials from DATABRICKS_HOST/CLIENT_ID/CLIENT_SECRET, the SDK's
    # own names. credentials_provider needs a callable returning cfg.authenticate, not the method.
    config = Config()
    connection = ibis.databricks.connect(
        server_hostname=config.hostname,
        http_path=os.environ["DATABRICKS_CI_HTTP_PATH"],
        credentials_provider=lambda: config.authenticate,
        catalog=os.environ["DATABRICKS_CI_CATALOG"],
        schema=os.environ["DATABRICKS_CI_SCHEMA"],
        memtable_volume=_DATABRICKS_MEMTABLE_VOLUME,
    )
    try:
        yield connection.table(_TRANSACTIONS_TABLE_NAME)
    finally:
        connection.disconnect()


@pytest.fixture(
    params=["bigquery", "databricks", "pyspark", "snowflake", "mssql", "oracle"],
    ids=lambda backend: f"backend={backend}",
)
def transactions_table(request: pytest.FixtureRequest) -> Table:
    """Provide the transactions table from each backend in turn.

    Args:
        request (pytest.FixtureRequest): Fixture request carrying the backend name.

    Returns:
        Table: The transactions table on the parametrized backend.

    Raises:
        ValueError: If the parametrized backend name is not handled.
    """
    if request.param == "bigquery":
        connection = ibis.bigquery.connect(
            project_id=os.environ["GCP_PROJECT_ID"],
        )
        return connection.table(f"test_data.{_TRANSACTIONS_TABLE_NAME}")
    if request.param == "pyspark":
        connection = ibis.pyspark.connect()
        df = pd.read_parquet(_TRANSACTIONS_PARQUET)
        # Pyspark has no time column so we have to convert it to a datetime
        df["transaction_time"] = pd.to_datetime(
            df["transaction_date"].astype(str) + " " + df["transaction_time"].astype(str),
        )
        spark_df = connection._session.createDataFrame(df)
        spark_df.createOrReplaceTempView(_TRANSACTIONS_TABLE_NAME)
        return connection.table(_TRANSACTIONS_TABLE_NAME)
    if request.param == "snowflake":
        connection = ibis.snowflake.connect(
            account=os.environ["SNOWFLAKE_CI_ACCOUNT"],
            user=os.environ["SNOWFLAKE_CI_USER"],
            private_key_file=os.environ["SNOWFLAKE_CI_PRIVATE_KEY_PATH"],
            database=os.environ["SNOWFLAKE_CI_DATABASE"],
            schema=os.environ["SNOWFLAKE_CI_SCHEMA"],
            warehouse=os.environ["SNOWFLAKE_CI_WAREHOUSE"],
        )
        table = connection.table(_TRANSACTIONS_TABLE_NAME.upper())
        # Snowflake returns UPPERCASE column names; lowercase them for compatibility with integration tests
        return table.rename({col.lower(): col for col in table.columns})
    if request.param in ("databricks", "mssql", "oracle"):
        return request.getfixturevalue(f"_{request.param}_transactions_table")
    error_msg = f"Unknown backend: {request.param}"
    raise ValueError(error_msg)
