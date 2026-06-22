# Dockerized integration backends

The SQL Server and Oracle integration tests run against throwaway database containers,
so the same tests run locally and in CI. The Compose files here define one service per
engine; the image tag selects the version under test. Credentials are fixed,
non-secret values baked into the Compose files and matched by constants in
`../conftest.py` — there is nothing to configure.

The matching CI workflows (`.github/workflows/sqlserver-integration.yml` and
`oracle-integration.yml`) run each test suite across a matrix of supported free editions:

| Engine | Versions | Images |
| --- | --- | --- |
| SQL Server (Developer edition) | 2022, 2025 | `mcr.microsoft.com/mssql/server:<year>-latest` |
| Oracle (Free) | 23ai | `gvenzl/oracle-free:23-slim-faststart` |

The `transactions_table` fixture in `../conftest.py` requires the container to be
running, seeds `data/transactions.parquet` once per session, and retries the initial
connection while the container finishes starting. If the container is unreachable the
tests fail rather than skip, so a container that failed to start is never silently
passed over.

## Prerequisites

- Docker with the Compose plugin.
- For **SQL Server only**: the Microsoft ODBC Driver 18 and unixODBC on the host running
  pytest (Ibis's `mssql` backend uses `pyodbc`). Oracle needs no client libraries —
  `oracledb` connects in thin mode.
    - macOS: `brew install msodbcsql18 unixodbc`
    - Debian/Ubuntu: follow Microsoft's [ODBC driver install guide][odbc]. The
      `Install Microsoft ODBC Driver 18` step in `sqlserver-integration.yml` runs the
      same commands and is a working reference.

[odbc]: https://learn.microsoft.com/sql/connect/odbc/linux-mac/installing-the-microsoft-odbc-driver-for-sql-server

## Running locally

Run from the repository root. `--wait` blocks until the container is healthy.

### SQL Server

```bash
docker compose -f tests/integration/docker/docker-compose.sqlserver.yml up -d --wait
uv run pytest tests/integration -k mssql -v
docker compose -f tests/integration/docker/docker-compose.sqlserver.yml down -v
```

Test another supported version by overriding the image (nothing else changes):

```bash
MSSQL_IMAGE=mcr.microsoft.com/mssql/server:2025-latest \
  docker compose -f tests/integration/docker/docker-compose.sqlserver.yml up -d --wait
```

### Oracle

```bash
docker compose -f tests/integration/docker/docker-compose.oracle.yml up -d --wait
uv run pytest tests/integration -k oracle -v
docker compose -f tests/integration/docker/docker-compose.oracle.yml down -v
```
