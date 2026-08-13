<!-- README.md -->
<div align="center">
  <img
    src="https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/logo.svg"
    alt="OpenRetailScience"
    width="580"
  />
</div>

<div align="center">
  Open source retail analytics that runs in your database, handles billions of rows, and gives you back
  control of your KPIs.
</div>

## Installation

To get the latest release:

```bash
pip install openretailscience
```

Alternatively, if you want the very latest version of the package you can install it from GitHub:

```bash
pip install git+https://github.com/Data-Simply/openretailscience.git
```

## Features

- **Tailored for Retail**: Leverage pre-built functions designed specifically for retail analytics. From customer
  segmentations to gains loss analysis, OpenRetailScience provides over a dozen building blocks you need to tackle
  retail-specific challenges efficiently and effectively.

![New Store Cannibalization Analysis](https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/new_store_cannibalization.png)

- **Reliable Results**: Built with extensive unit testing and best practices, OpenRetailScience ensures the accuracy
  and reliability of your analyses. Confidently present your findings, knowing they're backed by a robust,
  well-tested framework.

- **Professional Charts**: Say goodbye to hours of tweaking chart styles. OpenRetailScience delivers beautifully
  standardized visualizations that are presentation-ready with just a few lines of code. Impress stakeholders and
  save time with our pre-built, customizable chart templates.

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/cross_shop_chart.png)

- **Workflow Automation**: OpenRetailScience streamlines your workflow by automating common retail analytics tasks.
  Easily loop analyses over different dimensions like product categories or countries, and seamlessly use the output
  of one analysis as input for another. Spend less time on data manipulation and more on generating valuable insights.

## Examples

### Gains Loss Analysis

Here is an excerpt from the gain loss analysis example [notebook](https://openretailscience.datasimply.co/examples/gain_loss/)

```python
from openretailscience.analysis.gain_loss import GainLoss

gl = GainLoss(
    df,
    # Flag the rows of period 1
    p1_index=time_period_1,
    # Flag the rows of period 2
    p2_index=time_period_2,
    # Flag which rows are part of the focus group.
    # Namely, which rows are Calvin Klein sales
    focus_group_index=df["brand_name"] == "Calvin Klein",
    focus_group_name="Calvin Klein",
    # Flag which rows are part of the comparison group.
    # Namely, which rows are Diesel sales
    comparison_group_index=df["brand_name"] == "Diesel",
    comparison_group_name="Diesel",
    # Finally we specifiy that we want to calculate
    # the gain/loss in total revenue
    value_col="total_price",
)
# Ok now let's plot the result
gl.plot(
    x_label="Revenue Change",
    source_text="Transactions 2023-01-01 to 2023-12-31",
    move_legend_outside=True,
)
plt.show()
```

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/gain_loss_chart.png)

### Cross Shop Analysis

Here is an excerpt from the cross shop analysis example [notebook](https://openretailscience.datasimply.co/examples/cross_shop/)

```python
from openretailscience.analysis import cross_shop

cs = cross_shop.CrossShop(
    df,
    group_1_col="category_name",
    group_1_val="Jeans",
    group_2_col="category_name",
    group_2_val="Shoes",
    group_3_col="category_name",
    group_3_val="Dresses",
    labels=["Jeans", "Shoes", "Dresses"],
)
cs.plot(
    title="Jeans are a popular cross-shopping category with dresses",
    source_text="Source: Transactions 2023-01-01 to 2023-12-31",
    figsize=(6, 6),
)
plt.show()
# Let's see which customers were in which groups
display(cs.cross_shop_df.head())
# And the totals for all groups
display(cs.cross_shop_table_df)
```

![Cross Shop Analysis Chart](https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/cross_shop_chart.png)

### Customer Retention Analysis

Here is an excerpt from the customer retention analysis example [notebook](https://openretailscience.datasimply.co/examples/retention/)

```python
from openretailscience.plots import histogram
from openretailscience.plots.styles.graph_utils import set_axis_percent

ax = histogram.plot(
    df=dbp.df["avg_days_between_purchases"],
    figsize=(10, 5),
    bins=20,
    cumulative=True,
    density=True,
    source_text="Source: Transactions in 2023",
    title="When Do Customers Make Their Next Purchase?",
)
set_axis_percent(ax.yaxis, decimals=0, xmax=1.0)

# Let's dress up the chart a bit of text and get rid of the legend
churn_period = dbp.purchases_percentile(0.8)
ax.axvline(x=churn_period, color="black", linestyle="--", lw=2)
ax.annotate(
    f"80% of customers made\nanother purchase within\n{round(churn_period)} days",
    xy=(churn_period, 0.81),
    xytext=(dbp.df["avg_days_between_purchases"].min(), 0.8),
    fontsize=15,
    ha="left",
    va="center",
    arrowprops=dict(facecolor="black", arrowstyle="-|>", connectionstyle="arc3,rad=-0.25", mutation_scale=25),
)
ax.legend().set_visible(False)
plt.show()
```

![Cumulative Next Purchase Chart](https://raw.githubusercontent.com/Data-Simply/openretailscience/main/readme_assets/days_until_next_purchase.png)

## Documentation

Please see [this site](https://openretailscience.datasimply.co/) for full documentation, which includes:

- [Analysis Modules](https://openretailscience.datasimply.co/analysis_modules/): Overview of the framework and the
  structure of the docs.
- [Examples](https://openretailscience.datasimply.co/examples/retention/): If you're looking to build something
  specific or are more of a hands-on learner, check out our examples. This is the best place to get started.
- [API Reference](https://openretailscience.datasimply.co/api/gain_loss/): Thorough documentation of every class
  and method.

## Contributing

We welcome contributions from the community to enhance and improve OpenRetailScience. To contribute,
please follow these steps:

1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them with clear messages.
4. Push your changes to your fork.
5. Open a pull request to the main repository's `main` branch.

Please make sure to follow the existing coding style and provide unit tests for new features.

## Contact / Support

This repository is supported by Data simply.

If you are interested in seeing what Data Simply can do for you, then please email
[email us](mailto:murray@datasimply.co). We work with companies at a variety of scales and with varying levels of
data and retail analytics sophistication, to help them build, scale or streamline their analysis capabilities.

## Contributors

<a href="https://github.com/Data-Simply/openretailscience/graphs/contributors">
    <img src="https://contrib.rocks/image?repo=Data-Simply/openretailscience" alt="Contributors" />
</a>

Made with [contrib.rocks](https://contrib.rocks).

## Acknowledgements

Built with expertise doing analytics and data science for scale-ups to multi-nationals, including:

- Loblaws
- Dominos
- Sainbury's
- IKI
- Migros
- Sephora
- Nectar
- Metro
- Coles
- GANNI
- Mindful Chef
- Auchan
- Attraction Tickets Direct
- Roman Originals

## Testing

OpenRetailScience includes comprehensive unit and integration tests to ensure reliability across different backends.

### Unit Tests

Run unit tests using pytest:

```bash
# Install dependencies
uv sync

# Run all unit tests
uv run pytest

# Run specific test file
uv run pytest tests/test_file.py

# Run with coverage
uv run pytest --cov=openretailscience
```

### Multi-Python Version Testing

OpenRetailScience supports Python 3.10 through 3.14. You can test across all supported versions locally using tox:

```bash
# Test all supported Python versions
tox -e py310,py311,py312,py313,py314

# Test specific Python version
tox -e py314

# Run tests in parallel across versions
tox -p auto
```

**Prerequisites:**

- Multiple Python versions installed on your system
- tox installed (`uv sync` installs it automatically)

### Integration Tests

Integration tests verify that all analysis modules work correctly across different backends: distributed computing
engines (PySpark, BigQuery, Snowflake, Databricks) and relational databases (SQL Server, Oracle). These tests ensure
the Ibis-based code paths function properly across different execution environments.

#### PySpark Integration Tests

The PySpark integration tests run locally using the same pytest framework as other tests.

**Prerequisites:**

- Python environment with the backend drivers installed (`uv sync --group integration`)

**Running locally:**

```bash
# Run all PySpark tests
uv run pytest tests/integration -k "pyspark" -v

# Run specific PySpark test
uv run pytest tests/integration/test_cohort_analysis.py -k "pyspark" -v
```

#### BigQuery Integration Tests

The BigQuery integration tests verify compatibility with Google BigQuery as a backend.

**Prerequisites:**

- Access to a Google Cloud Platform account
- A service account with BigQuery permissions
- The service account key JSON file
- The test dataset loaded in BigQuery (dataset: `test_data`, table: `transactions`)

**Running locally:**

```bash
# Set up authentication
export GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
export GCP_PROJECT_ID=your-project-id

# Install dependencies (the backend drivers live in a non-default group)
uv sync --group integration

# Run all BigQuery tests
uv run pytest tests/integration -k "bigquery" -v

# Run specific test module
uv run pytest tests/integration/bigquery/test_cohort_analysis.py -v
```

#### Snowflake Integration Tests

The Snowflake integration tests verify compatibility with Snowflake as a backend.

**Prerequisites:**

- Access to a Snowflake account with a warehouse, database, and schema configured
- A key-pair authentication private key (PEM format) for the Snowflake user
- The test dataset loaded in Snowflake (table: `TRANSACTIONS`)

**Running locally:**

```bash
# Set up Snowflake connection
export SNOWFLAKE_CI_ACCOUNT=your-account-identifier
export SNOWFLAKE_CI_USER=your-username
export SNOWFLAKE_CI_WAREHOUSE=your-warehouse
export SNOWFLAKE_CI_DATABASE=your-database
export SNOWFLAKE_CI_SCHEMA=your-schema
export SNOWFLAKE_CI_PRIVATE_KEY_PATH=/path/to/your/private-key.p8

# Install dependencies (the backend drivers live in a non-default group)
uv sync --group integration

# Run all Snowflake tests
uv run pytest tests/integration -k "snowflake" -v

# Run specific test module
uv run pytest tests/integration/test_cohort_analysis.py -k "snowflake" -v
```

#### Databricks Integration Tests

The Databricks integration tests verify compatibility with Databricks as a backend. They query a pre-loaded
Unity Catalog table over a SQL warehouse, authenticating as an OAuth machine-to-machine service principal.

**Prerequisites:**

- A Databricks workspace with a SQL warehouse and Unity Catalog
- An OAuth service principal (client ID and secret) with `USE_CATALOG`, `USE_SCHEMA` and `SELECT` on the test
  schema, plus `CREATE VOLUME`, `READ VOLUME` and `WRITE VOLUME`: Ibis creates a volume on connect to stage
  memtables, so a principal holding only `SELECT` cannot open the connection
- The test dataset loaded in Unity Catalog (table: `<catalog>.<schema>.transactions`), kept in step with
  `data/transactions.parquet`, which the seeded backends build their copy from
- A `test_data` volume in the same schema, which CI stages the wheel and test module into

**Running locally:**

```bash
# Set up Databricks connection. HOST/CLIENT_ID/CLIENT_SECRET are the standard
# Databricks SDK variables, so a ~/.databrickscfg profile works instead.
export DATABRICKS_HOST=https://your-workspace.cloud.databricks.com
export DATABRICKS_CLIENT_ID=your-service-principal-client-id
export DATABRICKS_CLIENT_SECRET=your-service-principal-secret
export DATABRICKS_CI_HTTP_PATH=/sql/1.0/warehouses/your-warehouse-id
export DATABRICKS_CI_CATALOG=your-catalog
export DATABRICKS_CI_SCHEMA=your-schema

# Install dependencies (the backend drivers live in a non-default group)
uv sync --group integration

# Run all Databricks tests
uv run pytest tests/integration -k "databricks" -v

# Run specific test module
uv run pytest tests/integration/test_cohort_analysis.py -k "databricks" -v
```

##### install_skills() on Databricks

`install_skills` copies the bundled agent skills into `/Workspace/Users/<current_user()>/.assistant/skills`,
a mount that exists only on Databricks compute. Coverage is shared between:

- `tests/test_skills.py::TestDatabricksInstall` runs on every commit, against a temp directory and a
  stub Spark session. It covers the states a real workspace will not produce on demand.
- `tests/integration/test_workspace_skills.py` holds every assertion made against the real mount.
  Plain pytest skips it.
- The `skills-install-tests` job in `.github/workflows/databricks-integration.yml` delivers it, running
  a wheel built from the working tree rather than the last PyPI release.
- The notebook that job generates is where the `DATABRICKS_RUNTIME_VERSION` guard lives.

To run them by hand, attach a Databricks notebook to serverless compute and execute the following.
They delete and reinstall the bundled skills in your own workspace home:

```python
%pip install openretailscience pytest
%restart_python
```

```python
import pytest
pytest.main(["/Workspace/path/to/test_workspace_skills.py", "-v", "-p", "no:cacheprovider"])
```

#### SQL Server and Oracle Integration Tests

SQL Server and Oracle run against throwaway Docker containers, so they execute the same way locally and in CI.
CI covers the supported free editions: SQL Server Developer 2022 and 2025, and Oracle 23ai Free.

These backends have a one-time setup (the SQL Server ODBC driver) and per-version image tags, so their
instructions live next to the Compose files. See
[tests/integration/docker/README.md](tests/integration/docker/README.md) for prerequisites, the version matrix,
and the commands to start a container and run the tests.

## License

This project is licensed under the Elastic License 2.0 - see the [LICENSE](LICENSE) file for details.
