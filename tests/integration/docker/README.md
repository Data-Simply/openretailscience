# Dockerized integration backends

The SQL Server and Oracle integration tests run against throwaway database containers,
so the same tests run locally and in CI. The Compose files here define one service per
engine; the image tag selects the version under test.

The matching CI workflows (`.github/workflows/sqlserver-integration.yml` and
`oracle-integration.yml`) run each test suite across a matrix of every free edition
released since 2019:

| Engine | Versions | Images |
| --- | --- | --- |
| SQL Server (Developer edition) | 2019, 2022, 2025 | `mcr.microsoft.com/mssql/server:<year>-latest` |
| Oracle (XE / Free) | 18c, 21c, 23ai | `gvenzl/oracle-xe`, `gvenzl/oracle-free` (`-slim-faststart` tags) |

The `transactions_table` fixture in `../conftest.py` skips each backend unless its
`*_HOST` environment variable is set, seeds `data/transactions.parquet` once per
session, and retries the initial connection while the container finishes starting.

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

The Compose files have defaults baked in, and `.env_sample` lists matching values for
pytest. Copy them into your environment first:

```bash
cp .env_sample .env        # then `set -a; source .env; set +a` to export them
```

### SQL Server

```bash
cd tests/integration/docker
docker compose -f docker-compose.sqlserver.yml up -d --wait      # start (default: 2022)
cd ../../..
uv run pytest tests/integration -k mssql -v
docker compose -f tests/integration/docker/docker-compose.sqlserver.yml down -v
```

Test another version by overriding the image (and nothing else):

```bash
MSSQL_IMAGE=mcr.microsoft.com/mssql/server:2019-latest \
  docker compose -f docker-compose.sqlserver.yml up -d --wait
```

### Oracle

```bash
cd tests/integration/docker
docker compose -f docker-compose.oracle.yml up -d --wait          # start (default: 23ai Free)
cd ../../..
uv run pytest tests/integration -k oracle -v
docker compose -f tests/integration/docker/docker-compose.oracle.yml down -v
```

For an XE image, override the image **and** the service name (XE uses `XEPDB1`):

```bash
ORACLE_IMAGE=gvenzl/oracle-xe:21-slim-faststart ORACLE_SERVICE_NAME=XEPDB1 \
  docker compose -f docker-compose.oracle.yml up -d --wait
ORACLE_SERVICE_NAME=XEPDB1 uv run pytest tests/integration -k oracle -v
```
