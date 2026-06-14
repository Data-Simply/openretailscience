"""Tests for the Options module."""

from pathlib import Path
from typing import get_args

import pytest
import toml

import openretailscience.options as opt


class TestOptions:
    """Test for option handling class."""

    def test_unknown_option_raises_value_error(self):
        """Test setting/getting/resetting an unknown option raises a ValueError."""
        options = opt.Options()
        with pytest.raises(ValueError, match=r"Unknown option: unknown.option"):
            options.set_option("unknown.option", "some_value")
        with pytest.raises(ValueError, match=r"Unknown option: unknown.option"):
            options.get_option("unknown_option")
        with pytest.raises(ValueError, match=r"Unknown option: unknown.option"):
            options.reset_option("unknown_option")
        with pytest.raises(ValueError, match=r"Unknown option: unknown.option"):
            options.describe_option("unknown_option")

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

    def test_context_manager_overrides_option(self):
        """Test that the context manager overrides the option value correctly at the global level."""
        original_value = opt.get_option("column.customer_id")
        with opt.option_context("column.customer_id", "new_customer_id"):
            assert opt.get_option("column.customer_id") == "new_customer_id"
        assert opt.get_option("column.customer_id") == original_value

    def test_context_manager_odd_number_of_arguments_raises_value_error(self):
        """Test that the context manager raises a ValueError when an odd number of arguments is passed."""
        with (
            pytest.raises(ValueError, match="The context manager requires an even number of arguments"),
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
        """Test loading an invalid TOML file raises a ValueError."""
        test_file_path = Path("tests/toml_files/invalid_option.toml").resolve()
        with pytest.raises(ValueError, match=r"Unknown option in TOML file: column.agg.unknown_column"):
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

    def test_typed_option_name_groups_match_defaults(self):
        """The typed get_option overloads must cover every default option with the right type.

        Why: ``get_option`` returns precise types (str, float, bool, ...) via overloads keyed on
        Literal groups of option names. If a new option is added to the defaults but not to the
        matching group, callers silently fall back to the wide ``OptionTypes`` union; if a group
        lists a stale name, the overload lies about a key that no longer exists. This pins both
        groups and defaults together so either kind of drift fails loudly.
        """
        group_to_type: dict[tuple[str, ...], type | tuple[type, ...]] = {
            get_args(opt._StrOptionName): str,
            get_args(opt._FloatOptionName): float,
            get_args(opt._IntOptionName): int,
            get_args(opt._BoolOptionName): bool,
            get_args(opt._StrListOptionName): list,
            get_args(opt._FloatListOptionName): list,
        }
        name_to_expected_type = {name: expected for names, expected in group_to_type.items() for name in names}

        defaults = opt.Options()
        default_values = {name: defaults.get_option(name) for name in defaults.list_options()}

        assert set(name_to_expected_type) == set(default_values)
        # bool is a subclass of int, so check bool before int to avoid misclassification.
        for name, expected_type in name_to_expected_type.items():
            value = default_values[name]
            assert isinstance(value, expected_type), f"{name}={value!r} is not {expected_type}"
            if expected_type is int:
                assert not isinstance(value, bool), f"{name} is grouped as int but is a bool"

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
