"""Tests for openretailscience.core.validation."""

import ibis
import pandas as pd
import pytest

from openretailscience.core.validation import (
    ensure_columns,
    ensure_ibis_table,
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
