"""Tests for the Options module."""

import re
from pathlib import Path

import pytest
import toml

import openretailscience.options as opt

LIST_HINT = "See openretailscience.options.list_options()."
COLUMN_HINT = (
    "Options rename only the standard input and output columns (openretailscience.options.list_options()); "
    "if {name!r} isn't one, pass it to analyses as an argument instead, e.g. group_col={name!r}."
)


def _enter_option_context(*args):
    with opt.option_context(*args):
        pass


class TestOptions:
    """Test for option handling class."""

    @pytest.mark.parametrize(
        "operation",
        [
            lambda pat: opt.set_option(pat, "brand"),
            opt.get_option,
            opt.reset_option,
            opt.describe_option,
            lambda pat: _enter_option_context(pat, "brand"),
            lambda pat: _enter_option_context({pat: "brand"}),
            opt.ColumnHelper.join_options,
        ],
        ids=[
            "set_option",
            "get_option",
            "reset_option",
            "describe_option",
            "option_context",
            "option_context_dict",
            "join_options",
        ],
    )
    @pytest.mark.parametrize(
        ("pat", "expected"),
        [
            (
                "column.custmer_id",
                "Unknown option 'column.custmer_id'. Did you mean 'column.customer_id'? "
                + COLUMN_HINT.format(name="custmer_id"),
            ),
            (
                "column.cusotmer_id",
                "Unknown option 'column.cusotmer_id'. Did you mean 'column.customer_id'? "
                + COLUMN_HINT.format(name="cusotmer_id"),
            ),
            (
                "column.customer",
                "Unknown option 'column.customer'. Did you mean 'column.customer_id'? "
                + COLUMN_HINT.format(name="customer"),
            ),
            (
                "column.unit_qty",
                "Unknown option 'column.unit_qty'. Did you mean 'column.unit_quantity'? "
                + COLUMN_HINT.format(name="unit_qty"),
            ),
            (
                "column.spend",
                "Unknown option 'column.spend'. Did you mean 'column.unit_spend'? " + COLUMN_HINT.format(name="spend"),
            ),
            (
                "column.spend_per_customer",
                "Unknown option 'column.spend_per_customer'. Did you mean 'column.calc.spend_per_customer'? "
                + COLUMN_HINT.format(name="spend_per_customer"),
            ),
            (
                "column.calc.unit_spend",
                "Unknown option 'column.calc.unit_spend'. Did you mean 'column.agg.unit_spend'? "
                + COLUMN_HINT.format(name="unit_spend"),
            ),
            (
                "column.store_name",
                "Unknown option 'column.store_name'. Did you mean 'column.store_id'? "
                + COLUMN_HINT.format(name="store_name"),
            ),
            (
                "column.date",
                "Unknown option 'column.date'. " + COLUMN_HINT.format(name="date"),
            ),
            (
                "column.brand_name",
                "Unknown option 'column.brand_name'. " + COLUMN_HINT.format(name="brand_name"),
            ),
            (
                "column.loyalty_tier",
                "Unknown option 'column.loyalty_tier'. " + COLUMN_HINT.format(name="loyalty_tier"),
            ),
            (
                "column.source",
                "Unknown option 'column.source'. " + COLUMN_HINT.format(name="source"),
            ),
            (
                "column.customer_segment",
                "Unknown option 'column.customer_segment'. " + COLUMN_HINT.format(name="customer_segment"),
            ),
            (
                "column.agg.gross_margin",
                "Unknown option 'column.agg.gross_margin'. " + COLUMN_HINT.format(name="gross_margin"),
            ),
            (
                "column.calc.margin",
                "Unknown option 'column.calc.margin'. " + COLUMN_HINT.format(name="margin"),
            ),
            (
                "customer_id",
                "Unknown option 'customer_id'. Did you mean 'column.customer_id'? "
                + COLUMN_HINT.format(name="customer_id"),
            ),
            (
                "customer",
                "Unknown option 'customer'. Did you mean 'column.customer_id'? " + COLUMN_HINT.format(name="customer"),
            ),
            (
                "agg.unit_spend",
                "Unknown option 'agg.unit_spend'. Did you mean 'column.agg.unit_spend'? "
                + COLUMN_HINT.format(name="unit_spend"),
            ),
            (
                "calc.spend_per_cust",
                "Unknown option 'calc.spend_per_cust'. Did you mean 'column.calc.spend_per_customer'? "
                + COLUMN_HINT.format(name="spend_per_cust"),
            ),
            (
                "brand_name",
                "Unknown option 'brand_name'. " + COLUMN_HINT.format(name="brand_name"),
            ),
            (
                "column.suffix.percentage",
                "Unknown option 'column.suffix.percentage'. Did you mean 'column.suffix.percent'? " + LIST_HINT,
            ),
            (
                "column.suffixes.count",
                "Unknown option 'column.suffixes.count'. Did you mean 'column.suffix.count'? " + LIST_HINT,
            ),
            (
                "column.difference",
                "Unknown option 'column.difference'. Did you mean 'column.suffix.difference'? " + LIST_HINT,
            ),
            (
                "column.percent",
                "Unknown option 'column.percent'. Did you mean 'column.suffix.percent'? " + LIST_HINT,
            ),
            (
                "plot.colour.primary",
                "Unknown option 'plot.colour.primary'. Did you mean 'plot.color.primary'? " + LIST_HINT,
            ),
            (
                "font.title_size",
                "Unknown option 'font.title_size'. Did you mean 'plot.font.title_size'? " + LIST_HINT,
            ),
            (
                "color.primary",
                "Unknown option 'color.primary'. Did you mean 'plot.color.primary'? " + LIST_HINT,
            ),
            (
                "title_size",
                "Unknown option 'title_size'. Did you mean 'plot.font.title_size'? " + LIST_HINT,
            ),
            (
                "use_native_sql",
                "Unknown option 'use_native_sql'. Did you mean 'optimization.use_native_sql'? " + LIST_HINT,
            ),
            (
                "primary",
                "Unknown option 'primary'. Did you mean 'plot.color.primary'? " + LIST_HINT,
            ),
            (
                "Column.Customer_ID",
                "Unknown option 'Column.Customer_ID'. Did you mean 'column.customer_id'? " + LIST_HINT,
            ),
            ("plot.style.dark_mode", "Unknown option 'plot.style.dark_mode'. " + LIST_HINT),
            ("style.dark_mode", "Unknown option 'style.dark_mode'. " + LIST_HINT),
            ("display.max_rows", "Unknown option 'display.max_rows'. " + LIST_HINT),
            ("", "Unknown option ''. " + LIST_HINT),
            ("column.", "Unknown option 'column.'. " + LIST_HINT),
            (42, "Unknown option '42'. " + LIST_HINT),
            (3.5, "Unknown option '3.5'. " + LIST_HINT),
        ],
        ids=[
            "dropped_letter",
            "swapped_letters",
            "missing_id",
            "abbreviation",
            "lowest_scoring_typo",
            "missing_calc_group",
            "wrong_group",
            "non_standard_near_real_key",
            "short_standard_name",
            "non_standard",
            "non_standard_far_from_any_key",
            "non_standard_named_like_a_plot_option",
            "non_standard_just_below_cutoff",
            "invented_agg_output",
            "invented_calc_output",
            "missing_column_prefix",
            "missing_column_prefix_and_id",
            "missing_column_prefix_on_output",
            "missing_column_prefix_typo",
            "missing_column_prefix_non_standard",
            "suffix_typo",
            "suffix_group_typo",
            "missing_suffix_group",
            "missing_suffix_group_at_cutoff",
            "other_typo",
            "missing_plot_prefix",
            "missing_plot_prefix_color",
            "missing_plot_groups",
            "missing_optimization_group",
            "missing_plot_groups_single_word",
            "mixed_case",
            "other_no_match",
            "missing_plot_prefix_no_match",
            "unknown_group",
            "empty",
            "column_group_without_a_name",
            "non_string",
            "non_string_with_dot",
        ],
    )
    def test_unknown_option_raises_value_error(self, operation, pat, expected):
        """Each entry point names the unknown key, suggests a close real one and says where non-standard columns go."""
        with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
            operation(pat)

    @pytest.mark.parametrize("pat", [["column.customer_id"], {"column.customer_id": "cust_id"}], ids=["list", "dict"])
    def test_unhashable_option_key_raises_value_error(self, pat):
        """A key that can't index the options gets the unknown-option error, not a TypeError."""
        expected = f"Unknown option '{pat}'. " + LIST_HINT
        with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
            opt.get_option(pat)

    def test_option_context_unknown_key_restores_earlier_options(self):
        """An unknown key raises before it is set, and the keys set before it are restored."""
        original_customer = opt.get_option("column.customer_id")
        with (
            pytest.raises(ValueError, match=re.escape("Unknown option 'column.brand_name'.")),
            opt.option_context("column.customer_id", "cust_id", "column.brand_name", "brand"),
        ):
            pass
        assert opt.get_option("column.customer_id") == original_customer
        assert "column.brand_name" not in opt.list_options()

    def test_set_option_updates_value(self):
        """Test setting an option updates the option value correctly."""
        options = opt.Options()
        options.set_option("column.customer_id", "new_customer_id")
        assert options.get_option("column.customer_id") == "new_customer_id"

    def test_reset_option_restores_default_value(self):
        """Test resetting an option restores its default value."""
        options = opt.Options()
        expected_value = options.get_option("column.customer_id")
        options.set_option("column.customer_id", "new_customer_id")
        options.reset_option("column.customer_id")
        assert options.get_option("column.customer_id") == expected_value

    def test_describe_option_returns_well_formed_string_for_all_options(self):
        """Test that describe_option returns a properly formatted string for every option."""
        options = opt.Options()
        for key in options.list_options():
            description = options.describe_option(key)
            assert key in description
            assert "(current value:" in description

    @pytest.mark.parametrize(
        "args",
        [
            ("column.customer_id", "new_customer_id"),
            ({"column.customer_id": "new_customer_id"},),
        ],
    )
    def test_context_manager_overrides_option(self, args):
        """Test that the context manager overrides the option value for both positional and dict call styles."""
        original_value = opt.get_option("column.customer_id")
        with opt.option_context(*args):
            assert opt.get_option("column.customer_id") == "new_customer_id"
        assert opt.get_option("column.customer_id") == original_value

    def test_context_manager_dict_overrides_multiple_options(self):
        """Test that a dict argument overrides multiple options and restores them all on exit."""
        original_customer = opt.get_option("column.customer_id")
        original_store = opt.get_option("column.store_id")
        with opt.option_context({"column.customer_id": "shopper_id", "column.store_id": "outlet_id"}):
            assert opt.get_option("column.customer_id") == "shopper_id"
            assert opt.get_option("column.store_id") == "outlet_id"
        assert opt.get_option("column.customer_id") == original_customer
        assert opt.get_option("column.store_id") == original_store

    def test_context_manager_odd_number_of_arguments_raises_value_error(self):
        """Test that the context manager raises a ValueError when an odd number of arguments is passed."""
        with (
            pytest.raises(ValueError, match=r"even number of arguments.*Got 1: \('column\.customer_id',\)"),
            opt.option_context("column.customer_id"),
        ):
            pass

    def test_set_option_updates_value_global_level(self):
        """Test setting an option updates the option value correctly at the global level."""
        opt.set_option("column.customer_id", "new_customer_id")
        assert opt.get_option("column.customer_id") == "new_customer_id"
        opt.reset_option("column.customer_id")

    def test_reset_option_restores_default_value_global_level(self):
        """Test resetting an option restores its default value at the global level."""
        expected_value = opt.Options().get_option("column.customer_id")
        opt.set_option("column.customer_id", "new_customer_id")
        opt.reset_option("column.customer_id")
        assert opt.get_option("column.customer_id") == expected_value

    def test_describe_option_correct_description_and_value_global_level(self):
        """Test describing an option provides the correct description and current value at the global level."""
        option = "column.customer_id"
        description = opt.describe_option(option)
        # Format contract: "{option}: {description text} (current value: {value})"
        assert description.startswith(f"{option}: ")
        assert "column containing customer IDs" in description
        assert description.endswith(f"(current value: {opt.get_option(option)})")

    def test_load_invalid_format_toml(self):
        """Test loading an invalid TOML file raises a TomlDecodeError."""
        test_file_path = Path("tests/toml_files/corrupt.toml").resolve()
        with pytest.raises(toml.TomlDecodeError):
            opt.Options.load_from_toml(test_file_path)

    def test_load_valid_toml(self):
        """Test loading a valid TOML file updates the options correctly."""
        test_file_path = Path("tests/toml_files/valid.toml").resolve()
        options = opt.Options.load_from_toml(test_file_path)
        assert options.get_option("column.customer_id") == "new_customer_id"
        assert options.get_option("column.product_id") == "new_product_id"
        assert options.get_option("column.agg.customer_id") == "new_customers"
        assert options.get_option("column.calc.price_per_unit") == "new_price_per_unit"
        assert options.get_option("column.suffix.count") == "new_cnt"

    def test_load_invalid_option_toml(self):
        """An unknown key in a TOML file raises the unknown-option error, prefixed with the file's path."""
        test_file_path = Path("tests/toml_files/invalid_option.toml").resolve()
        expected = f"{test_file_path}: Unknown option 'column.brand_name'. " + COLUMN_HINT.format(name="brand_name")
        with pytest.raises(ValueError, match=f"^{re.escape(expected)}$"):
            opt.Options.load_from_toml(test_file_path)

    def test_options_template_matches_defaults(self):
        """Template must contain every default option with its default value, and no extras.

        Why: the template doubles as runnable documentation of the full options surface.
        A missing key hides an option from new users; an extra/stale key trips load_from_toml
        with ValueError; a drifted value silently ships outdated defaults to anyone who
        copies the file.
        """
        template_path = Path("options_template.toml").resolve()
        with template_path.open() as f:
            template_data = toml.load(f)

        flat_template: dict[str, opt.OptionTypes] = {}
        for section, options in template_data.items():
            flat_template.update(opt.Options.flatten_options(section, options))

        defaults = opt.Options()
        expected = {name: defaults.get_option(name) for name in defaults.list_options()}
        assert flat_template == expected

    def test_flatten_options(self):
        """Test flattening the options dictionary."""
        nested_options = {
            "column": {
                "customer_id": "customer_id",
                "agg": {
                    "customer_id": "customer_id",
                    "product_id": "product_id",
                },
            },
        }
        expected_flat_options = {
            "column.customer_id": "customer_id",
            "column.agg.customer_id": "customer_id",
            "column.agg.product_id": "product_id",
        }
        assert expected_flat_options == opt.Options.flatten_options("column", nested_options["column"])

    @pytest.mark.parametrize(
        ("marker_name", "marker_is_dir", "start_subpath"),
        [
            (".git", True, ""),
            ("openretailscience.toml", False, ""),
            (".git", True, "sub/deeper"),
        ],
    )
    def test_find_project_root_locates_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        marker_name: str,
        marker_is_dir: bool,
        start_subpath: str,
    ) -> None:
        """Test that find_project_root returns the directory containing the marker."""
        marker = tmp_path / marker_name
        if marker_is_dir:
            marker.mkdir()
        else:
            marker.write_text("")
        start_dir = tmp_path / start_subpath if len(start_subpath) > 0 else tmp_path
        start_dir.mkdir(parents=True, exist_ok=True)
        monkeypatch.chdir(start_dir)
        assert opt.find_project_root() == tmp_path

    def test_find_project_root_returns_none_when_no_marker(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that find_project_root returns None when no marker is found up to the filesystem root."""
        monkeypatch.chdir(tmp_path)
        assert opt.find_project_root() is None

    def test_find_project_root_reresolves_after_chdir(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Test that find_project_root reflects the current working directory on each call."""
        root_a = tmp_path / "project_a"
        root_b = tmp_path / "project_b"
        (root_a / ".git").mkdir(parents=True)
        (root_b / ".git").mkdir(parents=True)

        monkeypatch.chdir(root_a)
        assert opt.find_project_root() == root_a

        monkeypatch.chdir(root_b)
        assert opt.find_project_root() == root_b

    def test_load_option_toml(self):
        """Test loading the test_options.toml file updates the options correctly."""
        test_file_path = Path("tests/toml_files/test_options.toml").resolve()
        options = opt.Options.load_from_toml(test_file_path)

        assert options.get_option("column.customer_id") == "new_customer_id"
        assert options.get_option("column.transaction_id") == "new_transaction_id"
        assert options.get_option("column.transaction_date") == "new_transaction_date"
        assert options.get_option("column.transaction_time") == "new_transaction_time"
        assert options.get_option("column.product_id") == "new_product_id"
        assert options.get_option("column.unit_quantity") == "new_unit_quantity"
        assert options.get_option("column.unit_price") == "new_unit_price"
        assert options.get_option("column.unit_spend") == "new_unit_spend"
        assert options.get_option("column.unit_cost") == "new_unit_cost"
        assert options.get_option("column.promo_unit_spend") == "new_promo_unit_spend"
        assert options.get_option("column.promo_unit_quantity") == "new_promo_unit_quantity"
        assert options.get_option("column.store_id") == "new_store_id"

        assert options.get_option("column.agg.customer_id") == "new_customers"
        assert options.get_option("column.agg.transaction_id") == "new_transactions"
        assert options.get_option("column.agg.product_id") == "new_products"
        assert options.get_option("column.agg.unit_quantity") == "new_units"
        assert options.get_option("column.agg.unit_price") == "new_prices"
        assert options.get_option("column.agg.unit_spend") == "new_spend"
        assert options.get_option("column.agg.unit_cost") == "new_costs"
        assert options.get_option("column.agg.promo_unit_spend") == "new_promo_spend"
        assert options.get_option("column.agg.promo_unit_quantity") == "new_promo_units"
        assert options.get_option("column.agg.store_id") == "new_stores"

        assert options.get_option("column.calc.price_per_unit") == "new_price_per_unit"
        assert options.get_option("column.calc.units_per_transaction") == "new_units_per_transaction"
        assert options.get_option("column.calc.spend_per_customer") == "new_spend_per_customer"
        assert options.get_option("column.calc.spend_per_transaction") == "new_spend_per_transaction"
        assert options.get_option("column.calc.transactions_per_customer") == "new_transactions_per_customer"
        assert options.get_option("column.calc.price_elasticity") == "new_price_elasticity"

        assert options.get_option("column.suffix.count") == "new_cnt"
        assert options.get_option("column.suffix.percent") == "new_pct"
        assert options.get_option("column.suffix.difference") == "new_diff"
        assert options.get_option("column.suffix.percent_difference") == "new_pct_diff"
        assert options.get_option("column.suffix.contribution") == "new_contrib"
        assert options.get_option("column.suffix.period_1") == "new_p1"
        assert options.get_option("column.suffix.period_2") == "new_p2"
