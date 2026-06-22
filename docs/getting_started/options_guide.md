# Options & Configuration Guide

## The Column Name Problem

Different retailers call the same thing by different names. A customer identifier might be called `customer_id`, `cid`,
`kunden_nr`, `token_id`, or many other variants. Similarly, revenue could be `unit_spend`, `revenue`, `sales`,
`basket_value`, and so on. Additionally, the metrics created could also have different names. Revenue per transaction
may be called `AOV` or `average_order_value` in one retailer and `spend_per_basket` in another.

OpenRetailScience solves this by using **standard internal names that are overridable** through an options system.
This allows the same analysis code to run on many different retailers' data without modification - you simply
configure the options to match your column names.

Options can be set:

- **Globally** for an entire run using a `openretailscience.toml` file
- **Temporarily** using the `option_context()` context manager

## Three Solutions

You have three ways to handle this mismatch:

!!! warning "Avoid Mixing Configuration Methods"
    If you use multiple configuration methods (TOML + `option_context()` + `set_option()`), the last setting
    wins. For example, `set_option()` calls will override TOML values. See
    [Conflicting Configurations](#conflicting-configurations) for details.<br><br>
    **Best practice:** Choose one method per project for consistency.

### Option 1: Rename Columns When Loading Data

!!! info "Best for"
    One-off analyses or when you want explicit control over column names.

Rename columns as part of your data loading process:

```python
import ibis
import pandas as pd

# Using Ibis
con = ibis.duckdb.connect()
table = con.table("transactions")
table = table.rename(
    cust_id="customer_id",
    SKU="product_id",
    revenue="unit_spend",
    trans_dt="transaction_date"
)

# Or with Pandas
df = pd.read_sql(query, connection)
df = df.rename(columns={
    "cust_id": "customer_id",
    "SKU": "product_id",
    "revenue": "unit_spend",
    "trans_dt": "transaction_date"
})
```

### Option 2: Use option_context() for Temporary Configuration

!!! info "Best for"
    Scripts where you want to keep your original column names throughout.

Tell OpenRetailScience what your column names are using `option_context()`. This doesn't rename your columns - it
configures OpenRetailScience's internal settings to look for your column names instead of the defaults. Options can be
supplied as alternating name/value pairs (below) or as a single dictionary such as
`option_context({"column.customer_id": "cust_id"})` - the dict form is handy when overriding a single option, since you
don't have to track its position in the argument list:

```python
from openretailscience.options import option_context
from openretailscience.analysis.gain_loss import GainLoss

# Your data has different column names - no need to rename!
with option_context(
    "column.customer_id", "cust_id",
    "column.product_id", "SKU",
    "column.unit_spend", "revenue",
    "column.transaction_date", "trans_dt"
):
    # OpenRetailScience uses your column names within this block
    gl = GainLoss(
        df,
        p1_index=period_1,
        p2_index=period_2,
        focus_group_index=brand_a,
        focus_group_name="Brand A",
        comparison_group_index=brand_b,
        comparison_group_name="Brand B",
        value_col="revenue"  # Uses your column name
    )
    gl.plot()

# Configuration automatically resets after the block
```

### Option 3: Use a TOML Configuration File (Recommended)

!!! tip "Recommended Approach"
    This is the recommended approach for most projects as it provides automatic, project-wide configuration
    without repetitive code.

!!! success "Best for"
    Projects with consistent column naming, team collaboration, and avoiding repetitive configuration.

Create a `openretailscience.toml` file in your project root to tell OpenRetailScience what your column names are. This
doesn't rename your columns - it configures OpenRetailScience's internal settings to automatically use your column names
throughout the project:

```toml
# openretailscience.toml
[column]
customer_id = "cust_id"
product_id = "SKU"
unit_spend = "revenue"
transaction_date = "trans_dt"
transaction_id = "trans_id"

[column.agg]
customer_id = "customers"
unit_spend = "total_revenue"

[column.suffix]
percent = "pct"
difference = "diff"
```

OpenRetailScience **automatically loads this configuration** when imported:

```python
# No configuration needed - loads from openretailscience.toml automatically!
from openretailscience.analysis.gain_loss import GainLoss

# Works directly with your column names
gl = GainLoss(
    df,
    p1_index=period_1,
    p2_index=period_2,
    focus_group_index=brand_a,
    focus_group_name="Brand A",
    comparison_group_index=brand_b,
    comparison_group_name="Brand B",
    value_col="revenue"  # Uses your column name
)
```

## TOML Configuration

For persistent configuration across your project, use a `openretailscience.toml` file.

### Creating a Configuration File

Create a file named `openretailscience.toml` in your project root (same directory as `.git`):

```toml
[column]
customer_id = "cust_id"
product_id = "SKU"
unit_spend = "revenue"
transaction_date = "trans_date"

[column.agg]
customer_id = "customers"
unit_spend = "total_revenue"
unit_quantity = "total_units"

[column.calc]
spend_per_customer = "revenue_per_customer"
price_per_unit = "avg_price"

[column.suffix]
percent = "pct"
difference = "diff"
count = "cnt"
```

### TOML Structure

The configuration uses nested sections that reflect the structure of the option names:

- `[column]` - Database/raw column names
- `[column.agg]` - Aggregated column names
- `[column.calc]` - Calculated column names
- `[column.suffix]` - Standard suffixes for output columns

### How TOML Loading Works

OpenRetailScience automatically searches for `openretailscience.toml`:

1. Starts in current working directory
2. Walks up the directory tree
3. Stops when it finds `.git` directory or `openretailscience.toml`
4. Loads configuration if found

**No manual loading required** - it happens automatically on import!

### Template File

Use `options_template.toml` in the [OpenRetailScience repository](https://github.com/Data-Simply/openretailscience/blob/main/options_template.toml)
as a starting point. It shows all available options with their default values.

## Available Option Categories

!!! warning
    This is not a complete list of all available options. Use `list_options()` to see all configurable options.

### Database Columns (`column.*`)

These map to your raw data columns:

- `column.customer_id` - Customer identifier
- `column.transaction_id` - Transaction identifier
- `column.transaction_date` - Transaction date
- `column.transaction_time` - Transaction time
- `column.product_id` - Product identifier
- `column.unit_quantity` - Number of units sold
- `column.unit_price` - Price per unit
- `column.unit_spend` - Total spend (price × quantity)
- `column.unit_cost` - Cost per unit
- `column.promo_unit_spend` - Promotional spend
- `column.promo_unit_quantity` - Promotional units
- `column.store_id` - Store identifier

### Aggregation Columns (`column.agg.*`)

Names for aggregated columns in output:

- `column.agg.customer_id` → `"customers"`
- `column.agg.transaction_id` → `"transactions"`
- `column.agg.unit_spend` → `"spend"`
- `column.agg.unit_quantity` → `"units"`
- etc.

### Calculated Columns (`column.calc.*`)

Names for calculated metrics:

- `column.calc.price_per_unit` → `"price_per_unit"`
- `column.calc.spend_per_customer` → `"spend_per_customer"`
- `column.calc.transactions_per_customer` → `"transactions_per_customer"`
- `column.calc.spend_per_transaction` → `"spend_per_transaction"`
- `column.calc.units_per_transaction` → `"units_per_transaction"`
- `column.calc.price_elasticity` → `"price_elasticity"`

### Column Suffixes (`column.suffix.*`)

Standard suffixes for period comparisons and calculations:

- `column.suffix.count` → `"cnt"`
- `column.suffix.percent` → `"pct"`
- `column.suffix.difference` → `"diff"`
- `column.suffix.percent_difference` → `"pct_diff"`
- `column.suffix.contribution` → `"contrib"`
- `column.suffix.period_1` → `"p1"`
- `column.suffix.period_2` → `"p2"`

## Advanced Reference

!!! note "Rarely Used by End Users"
    The functions in this section are primarily for **internal package use** or advanced scenarios. Most users
    won't need them - stick with the three approaches described above (rename columns, `option_context()`, or
    TOML files).

### ColumnHelper

!!! warning "Internal API"
    The `ColumnHelper` class is designed for **internal use within OpenRetailScience modules**. End users
    shouldn't use it directly.

The `ColumnHelper` class is used internally by OpenRetailScience modules to construct consistent column names in a less
verbose way than combining many `get_option` calls:

```python
from openretailscience.options import ColumnHelper

# Internal usage example (you typically won't need this)
cols = ColumnHelper()

# Access base column names
cols.customer_id  # Uses get_option("column.customer_id")
cols.unit_spend   # Uses get_option("column.unit_spend")

# Access aggregation columns via nested structure
cols.agg.unit_spend           # "spend"
cols.agg.customer_id          # "customers"
cols.agg.unit_spend_p1        # "spend_p1" (spend + period 1 suffix)
cols.agg.customer_id_pct_diff # "customers_pct_diff"

# Access calculated columns via nested structure
cols.calc.spend_per_cust        # "spend_per_customer"
cols.calc.price_per_unit        # "price_per_unit"
cols.calc.spend_per_cust_contrib # "spend_per_customer_contrib"
```

#### How ColumnHelper Works Internally

1. Reads option values on initialization
2. Creates nested `AggColumns` and `CalcColumns` objects for organized access
3. Combines base names with suffixes using `join_options()`
4. Provides consistent naming across all modules

The nested structure mirrors the configuration hierarchy:

- Base columns: Direct access (`cols.customer_id`)
- Aggregation columns: Nested access (`cols.agg.customer_id`)
- Calculated columns: Nested access (`cols.calc.spend_per_cust`)

This ensures that all OpenRetailScience functions use the same column naming conventions you've configured.

### get_option()

Retrieve the current value of an option. This is used internally throughout OpenRetailScience.

```python
from openretailscience.options import get_option

# Get current column name mapping
customer_col = get_option("column.customer_id")
print(customer_col)  # Output: customer_id (or your configured value)
```

**When you might use this:** Validating configuration or debugging issues.

### set_option()

Permanently set an option value (until reset or script ends).

```python
from openretailscience.options import set_option

# Change customer ID column globally
set_option("column.customer_id", "cust_id")

# All subsequent OpenRetailScience calls use "cust_id"
```

**When you might use this:** Quick experiments in notebooks. For production code, use `option_context()` or TOML instead.

### reset_option()

Reset an option back to its default value.

```python
from openretailscience.options import reset_option

# Restore default
reset_option("column.customer_id")  # Back to "customer_id"
```

**When you might use this:** Cleaning up after `set_option()` experiments.

### list_options()

List all available option names.

```python
from openretailscience.options import list_options

# See all configurable options
all_options = list_options()
print(all_options)
```

**When you might use this:** Discovering what options are available to configure.

### describe_option()

Get detailed information about a specific option.

```python
from openretailscience.options import describe_option

# Get description and current value
info = describe_option("column.customer_id")
print(info)
# Output: column.customer_id: The name of the column containing customer IDs. (current value: customer_id)
```

**When you might use this:** Understanding what an option controls.

## Troubleshooting

### Option Not Found Error

**Error:**

```python
ValueError: Unknown option: column.custom_field
```

**Solution:** Only predefined options can be used. Check `list_options()` for available options.

### TOML Not Loading

If changes in `openretailscience.toml` are not taking effect:

1. Ensure file is in project root (same directory as `.git`)
2. Restart Python kernel/session (options load on import)

### Column Not Found in Data

**Error:**

```python
ValueError: Input data is missing required columns: ['customer_id']
```

**Solution:** Either configure the option to match your column name, or rename your column to match the expected name:

1. Configure the option: `set_option("column.customer_id", "cust_id")`
2. Or rename your column: `df.rename(columns={"cust_id": "customer_id"})`

### Conflicting Configurations

When TOML and programmatic settings conflict, the last setting wins. `set_option()` overrides TOML values.

**Best Practice:** Choose one method per project for consistency.
