"""Tests for openretailscience.core.validation."""

import datetime

import ibis
import pandas as pd
import pytest

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_integer,
    ensure_number,
    ensure_positive,
    ensure_tznaive_datetime,
    ensure_unit_interval,
    ensure_value_choice,
)


class TestEnsureColumns:
    """Tests for the ensure_columns helper function."""

    @pytest.mark.parametrize(
        ("columns_in", "expected_out"),
        [
            ("customer_id", ["customer_id"]),
            (["customer_id", "store_id"], ["customer_id", "store_id"]),
        ],
        ids=["string", "list"],
    )
    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_returns_normalized_list(self, input_type, columns_in, expected_out):
        """Test that valid str or list input is normalized to the expected list."""
        pdf = pd.DataFrame({"customer_id": [1, 2], "store_id": [101, 102]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        assert ensure_columns(df, columns_in, "group_col") == expected_out

    @pytest.mark.parametrize(
        ("columns_in", "expected_match"),
        [
            ("unit_spend", r"group_col references columns not present.*unit_spend"),
            (
                ["customer_id", "unit_spend", "store_id"],
                r"group_col references columns not present.*\['store_id', 'unit_spend'\]",
            ),
        ],
        ids=["string", "list"],
    )
    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_raises_when_columns_missing(self, input_type, columns_in, expected_match):
        """Test that ValueError surfaces param name and lists the missing columns."""
        pdf = pd.DataFrame({"customer_id": [1, 2]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        with pytest.raises(ValueError, match=expected_match):
            ensure_columns(df, columns_in, "group_col")

    @pytest.mark.parametrize("bad_input", [42, None, ("customer_id",), {"customer_id"}])
    def test_raises_type_error_for_non_str_non_list_input(self, bad_input):
        """Test that TypeError surfacing the caller's param name is raised for inputs that are neither str nor list."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(TypeError, match="segment_col must be a string or list of strings"):
            ensure_columns(df, bad_input, "segment_col")

    @pytest.mark.parametrize("bad_list", [[1, 2], ["customer_id", 5], [None, "customer_id"]])
    def test_raises_type_error_when_list_contains_non_string(self, bad_list):
        """Test that TypeError surfacing the caller's param name is raised when list contains non-string elements."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(TypeError, match="segment_col must contain only strings"):
            ensure_columns(df, bad_list, "segment_col")

    def test_raises_value_error_for_empty_list(self):
        """Test that ValueError surfacing the caller's param name is raised for an empty list."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(ValueError, match="group_col must not be an empty list"):
            ensure_columns(df, [], "group_col")


class TestEnsureDataHasColumns:
    """Tests for the ensure_data_has_columns helper function."""

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_no_op_when_all_columns_present(self, input_type):
        """Test that no error is raised when every column exists in df."""
        pdf = pd.DataFrame({"customer_id": [1], "unit_spend": [10.0], "store_id": [101]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        assert ensure_data_has_columns(df, ["customer_id", "unit_spend"]) is None

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_error_lists_only_missing_columns_in_sorted_order(self, input_type):
        """Test that the error lists only the missing columns, sorted."""
        pdf = pd.DataFrame({"customer_id": [1]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        with pytest.raises(ValueError, match=r"Input data is missing required columns: \['store_id', 'unit_spend'\]"):
            ensure_data_has_columns(df, ["customer_id", "unit_spend", "store_id"])

    @pytest.mark.parametrize("bad_input", ["unit_spend", ("unit_spend",), {"unit_spend"}, None])
    def test_raises_type_error_when_columns_is_not_a_list(self, bad_input):
        """A bare string (or other non-list) would otherwise iterate as characters; surface a clean TypeError."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(TypeError, match="columns must be a list of column names"):
            ensure_data_has_columns(df, bad_input)


class TestEnsureValueChoice:
    """Tests for the ensure_value_choice helper function."""

    @pytest.mark.parametrize("value", ["ASC", "Asc", "asc"])
    def test_matching_is_case_insensitive(self, value):
        """Test that any-case input matches the canonical lowercase entry from choices."""
        assert ensure_value_choice(value, ["asc", "desc"], "sort_order") == "asc"

    def test_returns_canonical_case_from_choices(self):
        """Test that the returned value is the choice spelling, not the caller's spelling."""
        assert ensure_value_choice("MONTH", ["Year", "Month", "Day"], "period") == "Month"

    def test_error_message_surfaces_param_name_choices_and_received_value(self):
        """Test that the error message includes the param name, the valid choices, and the bad input."""
        with pytest.raises(ValueError) as exc_info:
            ensure_value_choice("sideways", ["asc", "desc"], "sort_order")
        msg = str(exc_info.value)
        assert "sort_order" in msg
        assert "['asc', 'desc']" in msg
        assert "sideways" in msg

    def test_raises_type_error_for_non_string_value(self):
        """Test that TypeError is raised when value is not a string."""
        with pytest.raises(TypeError, match="must be a string"):
            ensure_value_choice(42, ["asc", "desc"], "sort_order")

    @pytest.mark.parametrize(
        "choices_factory",
        [
            lambda: ("asc", "desc"),
            lambda: {"asc": 1, "desc": 2},
            lambda: (c for c in ["asc", "desc"]),
        ],
        ids=["tuple", "dict", "generator"],
    )
    def test_accepts_any_iterable_of_choices(self, choices_factory):
        """Tuple, dict (yields keys), and one-shot generator are all valid ``choices`` inputs."""
        assert ensure_value_choice("ASC", choices_factory(), "sort_order") == "asc"

    def test_generator_choices_still_render_in_error_message(self):
        """Materializing ``choices`` once means a generator's contents appear in the error too."""
        with pytest.raises(ValueError, match=r"\['asc', 'desc'\]"):
            ensure_value_choice("bogus", (c for c in ["asc", "desc"]), "sort_order")


class TestEnsureIbisTable:
    """Tests for the ensure_ibis_table helper after the move to core."""

    @pytest.mark.parametrize(
        "df",
        [
            pd.DataFrame({"customer_id": [1, 2, 3], "unit_spend": [4.50, 5.99, 6.00]}),
            pd.DataFrame({"customer_id": pd.Series([], dtype="int64"), "unit_spend": pd.Series([], dtype="float64")}),
        ],
        ids=["non_empty", "empty"],
    )
    def test_pandas_dataframe_converts_to_ibis_table(self, df):
        """Test that pandas DataFrame is converted to ibis Table with data preserved."""
        result = ensure_ibis_table(df)
        assert isinstance(result, ibis.Table)
        pd.testing.assert_frame_equal(result.execute(), df)

    def test_ibis_table_returns_unchanged(self):
        """Test that ibis Table input is returned unchanged."""
        ibis_table = ibis.memtable(pd.DataFrame({"customer_id": [1, 2]}))
        assert ensure_ibis_table(ibis_table) is ibis_table

    @pytest.mark.parametrize(
        "invalid_input",
        [{"a": [1, 2, 3]}, [1, 2, 3], None, "not a dataframe", 42],
        ids=["dict", "list", "None", "string", "int"],
    )
    def test_invalid_input_raises_type_error(self, invalid_input):
        """Test that invalid input types raise TypeError with the default param name."""
        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            ensure_ibis_table(invalid_input)

    def test_error_surfaces_caller_param_name(self):
        """Test that a caller-supplied param_name appears in the TypeError message."""
        with pytest.raises(TypeError, match="data must be either a pandas DataFrame or an Ibis Table"):
            ensure_ibis_table("not a dataframe", param_name="data")


class TestEnsureNumber:
    """Tests for the ensure_number helper."""

    @pytest.mark.parametrize("value", [0, 5, -3, 2.5, -0.1])
    def test_accepts_int_and_float(self, value):
        """Ints and floats pass without raising."""
        assert ensure_number(value, "value") is None

    @pytest.mark.parametrize("value", [True, False, "5", None, [1]], ids=["true", "false", "str", "none", "list"])
    def test_rejects_non_number(self, value):
        """Bools (an int subclass) and non-numeric types are rejected with a TypeError."""
        with pytest.raises(TypeError, match="value must be a number"):
            ensure_number(value, "value")


class TestEnsureUnitInterval:
    """Tests for the ensure_unit_interval helper."""

    @pytest.mark.parametrize("value", [0.0, 0.5, 1.0, 0, 1])
    def test_accepts_values_in_closed_interval(self, value):
        """Values in [0, 1] pass without raising."""
        assert ensure_unit_interval(value, "percentile") is None

    @pytest.mark.parametrize("value", [-0.1, 1.1, 2, -5])
    def test_rejects_out_of_range(self, value):
        """Values outside [0, 1] raise ValueError."""
        with pytest.raises(ValueError, match="percentile must be between 0 and 1"):
            ensure_unit_interval(value, "percentile")

    def test_rejects_bool(self):
        """Booleans are rejected with a TypeError before the range check."""
        with pytest.raises(TypeError, match="percentile must be a number"):
            ensure_unit_interval(True, "percentile")


class TestEnsurePositive:
    """Tests for the ensure_positive helper."""

    @pytest.mark.parametrize("value", [1, 5, 0.5, 100])
    def test_accepts_positive(self, value):
        """Strictly positive numbers pass without raising."""
        assert ensure_positive(value, "churn_period") is None

    @pytest.mark.parametrize("value", [0, -1, -0.5])
    def test_rejects_non_positive(self, value):
        """Zero and negatives raise ValueError."""
        with pytest.raises(ValueError, match="churn_period must be positive"):
            ensure_positive(value, "churn_period")

    def test_rejects_bool(self):
        """Booleans are rejected with a TypeError before the range check."""
        with pytest.raises(TypeError, match="churn_period must be a number"):
            ensure_positive(True, "churn_period")


class TestEnsureInteger:
    """Tests for the ensure_integer helper."""

    @pytest.mark.parametrize("value", [0, 1, -3, 1000])
    def test_accepts_int(self, value):
        """Integers pass without raising."""
        assert ensure_integer(value, "churn_period") is None

    @pytest.mark.parametrize("value", [True, 1.0, 30.5, "30", None], ids=["bool", "int_float", "float", "str", "none"])
    def test_rejects_non_integer(self, value):
        """Bools, floats, strings, and None raise TypeError."""
        with pytest.raises(TypeError, match="churn_period must be an integer"):
            ensure_integer(value, "churn_period")


class TestEnsureTznaiveDatetime:
    """Tests for the ensure_tznaive_datetime helper."""

    def test_accepts_tznaive_datetime(self):
        """A timezone-naive datetime column passes without raising."""
        table = ibis.memtable(pd.DataFrame({"transaction_date": pd.to_datetime(["2024-01-01", "2024-01-02"])}))
        assert ensure_tznaive_datetime(table, "transaction_date") is None

    def test_accepts_date_column(self):
        """A plain date column passes without raising."""
        table = ibis.memtable(
            pd.DataFrame({"transaction_date": [datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)]}),
        )
        assert ensure_tznaive_datetime(table, "transaction_date") is None

    def test_rejects_tz_aware(self):
        """A timezone-aware column raises ValueError pointing at the tz."""
        table = ibis.memtable(
            pd.DataFrame({"transaction_date": pd.to_datetime(["2024-01-01"]).tz_localize("US/Eastern")}),
        )
        with pytest.raises(ValueError, match="timezone-aware"):
            ensure_tznaive_datetime(table, "transaction_date")

    def test_rejects_non_temporal(self):
        """A non-temporal (string) column raises TypeError."""
        table = ibis.memtable(pd.DataFrame({"transaction_date": ["2024-01-01", "2024-01-02"]}))
        with pytest.raises(TypeError, match="date or datetime"):
            ensure_tznaive_datetime(table, "transaction_date")
