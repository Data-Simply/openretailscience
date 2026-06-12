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
ax = dbp.plot(
    figsize=(10, 5),
    bins=20,
    cumulative=True,
    draw_percentile_line=True,
    percentile_line=0.8,
    source_text="Source: Transactions in 2023",
    title="When Do Customers Make Their Next Purchase?",
)

# Let's dress up the chart a bit of text and get rid of the legend
churn_period = dbp.purchases_percentile(0.8)
ax.annotate(
    f"80% of customers made\nanother purchase within\n{round(churn_period)} days",
    xy=(churn_period, 0.81),
    xytext=(dbp.purchase_dist_s.min(), 0.8),
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

OpenRetailScience supports Python 3.10, 3.11, 3.12, and 3.13. You can test across all supported versions locally using tox:

```bash
# Test all supported Python versions
tox -e py310,py311,py312,py313

# Test specific Python version
tox -e py313

# Run tests in parallel across versions
tox -p auto
```

**Prerequisites:**

- Multiple Python versions installed on your system
- tox installed (`uv sync` installs it automatically)

### Integration Tests

Integration tests verify that all analysis modules work correctly with distributed computing engines (PySpark,
BigQuery, and Snowflake). These tests ensure the Ibis-based code paths function properly across different execution
environments.

#### PySpark Integration Tests

The PySpark integration tests run locally using the same pytest framework as other tests.

**Prerequisites:**

- Python environment with dependencies installed (`uv sync`)

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

# Install dependencies
uv sync

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

# Install dependencies
uv sync

# Run all Snowflake tests
uv run pytest tests/integration -k "snowflake" -v

# Run specific test module
uv run pytest tests/integration/test_cohort_analysis.py -k "snowflake" -v
```

## License

This project is licensed under the Elastic License 2.0 - see the [LICENSE](LICENSE) file for details.
