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

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_string_input_returns_single_element_list(self, input_type):
        """Test str input is normalized to a single-element list."""
        pdf = pd.DataFrame({"customer_id": [1, 2], "store_id": [101, 102]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        assert ensure_columns(df, "customer_id") == ["customer_id"]

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_list_input_returns_same_list(self, input_type):
        """Test list input is returned as a list with the same contents."""
        pdf = pd.DataFrame({"customer_id": [1, 2], "store_id": [101, 102]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        assert ensure_columns(df, ["customer_id", "store_id"]) == ["customer_id", "store_id"]

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_raises_when_string_column_missing(self, input_type):
        """Test that ValueError is raised when a string column is missing."""
        pdf = pd.DataFrame({"customer_id": [1, 2]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        with pytest.raises(ValueError, match="unit_spend"):
            ensure_columns(df, "unit_spend")

    @pytest.mark.parametrize("input_type", ["pandas", "ibis"])
    def test_raises_when_any_list_column_missing(self, input_type):
        """Test that ValueError is raised listing missing columns."""
        pdf = pd.DataFrame({"customer_id": [1, 2]})
        df = ibis.memtable(pdf) if input_type == "ibis" else pdf
        with pytest.raises(ValueError, match=r"\['store_id', 'unit_spend'\]"):
            ensure_columns(df, ["customer_id", "unit_spend", "store_id"])

    @pytest.mark.parametrize("bad_input", [42, None, ("customer_id",), {"customer_id"}])
    def test_raises_type_error_for_non_str_non_list_input(self, bad_input):
        """Test that TypeError is raised for inputs that are neither str nor list."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(TypeError, match="columns must be a string or list of strings"):
            ensure_columns(df, bad_input)

    @pytest.mark.parametrize("bad_list", [[1, 2], ["customer_id", 5], [None, "customer_id"]])
    def test_raises_type_error_when_list_contains_non_string(self, bad_list):
        """Test that TypeError is raised when list contains non-string elements."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(TypeError, match="must contain only strings"):
            ensure_columns(df, bad_list)

    def test_raises_value_error_for_empty_list(self):
        """Test that ValueError is raised for an empty list."""
        df = pd.DataFrame({"customer_id": [1]})
        with pytest.raises(ValueError, match="columns must not be an empty list"):
            ensure_columns(df, [])

    def test_returns_new_list_not_same_reference(self):
        """Test that the returned list is a new object so callers may mutate freely."""
        df = pd.DataFrame({"customer_id": [1], "store_id": [10]})
        cols = ["customer_id", "store_id"]
        result = ensure_columns(df, cols)
        assert result is not cols


class TestEnsureValueChoice:
    """Tests for the ensure_value_choice helper function."""

    def test_returns_value_when_in_choices(self):
        """Test that a valid value is returned unchanged."""
        assert ensure_value_choice("asc", ["asc", "desc"], "sort_order") == "asc"

    def test_raises_value_error_when_value_not_in_choices(self):
        """Test that ValueError is raised when value is not in choices."""
        with pytest.raises(ValueError, match="sort_order"):
            ensure_value_choice("sideways", ["asc", "desc"], "sort_order")

    def test_error_message_lists_valid_choices(self):
        """Test that the error message lists the valid choices."""
        with pytest.raises(ValueError, match=r"\['asc', 'desc'\]"):
            ensure_value_choice("sideways", ["asc", "desc"], "sort_order")

    def test_error_message_includes_received_value(self):
        """Test that the error message includes the received value."""
        with pytest.raises(ValueError, match="sideways"):
            ensure_value_choice("sideways", ["asc", "desc"], "sort_order")

    def test_case_insensitive_normalizes_to_lowercase_choice(self):
        """Test that case-insensitive matching returns the canonical (lowercase) choice."""
        assert ensure_value_choice("ASC", ["asc", "desc"], "sort_order", case_insensitive=True) == "asc"

    def test_case_insensitive_mixed_case_matches(self):
        """Test that mixed-case input matches under case-insensitive mode."""
        assert ensure_value_choice("DeSc", ["asc", "desc"], "sort_order", case_insensitive=True) == "desc"

    def test_case_sensitive_by_default_rejects_uppercase(self):
        """Test that ASC is rejected when choices are lowercase and case_insensitive is False."""
        with pytest.raises(ValueError, match="sort_order"):
            ensure_value_choice("ASC", ["asc", "desc"], "sort_order")

    def test_raises_type_error_for_non_string_value(self):
        """Test that TypeError is raised when value is not a string."""
        with pytest.raises(TypeError, match="must be a string"):
            ensure_value_choice(42, ["asc", "desc"], "sort_order")

    def test_case_insensitive_raises_when_no_match(self):
        """Test ValueError raised when input doesn't match any choice under case-insensitive mode."""
        with pytest.raises(ValueError, match="sideways"):
            ensure_value_choice("sideways", ["asc", "desc"], "sort_order", case_insensitive=True)


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
        """Test that invalid input types raise TypeError with correct message."""
        with pytest.raises(TypeError, match="df must be either a pandas DataFrame or an Ibis Table"):
            ensure_ibis_table(invalid_input)
