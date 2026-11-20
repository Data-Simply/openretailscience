"""Tests for IndexAnalysis."""

import re

import ibis
import numpy as np
import pandas as pd
import pytest

from openretailscience.analysis.index_analysis import IndexAnalysis
from openretailscience.options import ColumnHelper

cols = ColumnHelper()

BRAND_COL = "brand"
AGE_BAND_COL = "age_band"
REGION_COL = "region"


@pytest.fixture
def brand_customers() -> pd.DataFrame:
    """Eight customers; Meadow Farm's three are in 18-34 and 35-64, so none of the 65+ band buys it."""
    return pd.DataFrame(
        {
            cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8],
            BRAND_COL: ["Meadow Farm"] * 3 + ["Greenvale"] * 5,
            AGE_BAND_COL: ["18-34", "35-64", "35-64", "18-34", "35-64", "65+", "65+", "65+"],
        },
    )


def _with_unknown_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Add a Meadow Farm and a Greenvale customer with no age band."""
    unknown = pd.DataFrame(
        {cols.customer_id: [9, 10], BRAND_COL: ["Meadow Farm", "Greenvale"], AGE_BAND_COL: [None, None]}
    )
    return pd.concat([df, unknown], ignore_index=True)


def _meadow_farm_index(df: pd.DataFrame | ibis.Table, **kwargs: str) -> IndexAnalysis:
    """Index each age band's customer count among Meadow Farm buyers against all customers."""
    return IndexAnalysis(
        df,
        value_col=cols.customer_id,
        group_col=AGE_BAND_COL,
        index_col=BRAND_COL,
        value_to_index="Meadow Farm",
        agg_func="nunique",
        **kwargs,
    )


class TestIndexAnalysis:
    """Tests for the IndexAnalysis result table."""

    @pytest.mark.parametrize(
        "prepare",
        [
            pytest.param(lambda df: df, id="pandas"),
            pytest.param(ibis.memtable, id="ibis"),
            # Null groups leave both totals, as if the caller had filtered them out first
            pytest.param(_with_unknown_ages, id="null_groups_excluded"),
        ],
    )
    def test_df_has_counts_shares_and_index_for_every_baseline_group(self, brand_customers, prepare):
        """Every baseline group appears, a group the subset has no rows in at an index of 0."""
        expected = pd.DataFrame(
            {
                AGE_BAND_COL: ["18-34", "35-64", "65+"],
                "subset_value": [1, 2, 0],
                "baseline_value": [2, 3, 3],
                "subset_share": [1 / 3, 2 / 3, 0.0],
                "baseline_share": [2 / 8, 3 / 8, 3 / 8],
                "index": [100 * (1 / 3) / (2 / 8), 100 * (2 / 3) / (3 / 8), 0.0],
            },
        )

        result = _meadow_farm_index(prepare(brand_customers)).df

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "extra_rows",
        [
            pytest.param([], id="every_region_known"),
            # Like null groups, null subgroups leave the result rather than forming a subgroup of their own
            pytest.param(
                [{cols.customer_id: 9, REGION_COL: None, BRAND_COL: "Meadow Farm", AGE_BAND_COL: "18-34"}],
                id="null_subgroups_excluded",
            ),
        ],
    )
    def test_subgroup_shares_are_within_each_subgroup(self, extra_rows):
        """With a subgroup, shares are of the subgroup's totals and every baseline pair appears."""
        customers = pd.DataFrame(
            {
                cols.customer_id: [1, 2, 3, 4, 5, 6, 7, 8],
                REGION_COL: ["North"] * 4 + ["South"] * 4,
                BRAND_COL: ["Meadow Farm", "Meadow Farm", "Greenvale", "Greenvale", "Meadow Farm", *["Greenvale"] * 3],
                AGE_BAND_COL: ["18-34", "35-64", "18-34", "35-64", "18-34", "18-34", "35-64", "35-64"],
            },
        )
        df = pd.concat([customers, pd.DataFrame(extra_rows)], ignore_index=True)
        expected = pd.DataFrame(
            {
                REGION_COL: ["North", "North", "South", "South"],
                AGE_BAND_COL: ["18-34", "35-64", "18-34", "35-64"],
                "subset_value": [1, 1, 1, 0],
                "baseline_value": [2, 2, 2, 2],
                "subset_share": [0.5, 0.5, 1.0, 0.0],
                "baseline_share": [0.5, 0.5, 0.5, 0.5],
                "index": [100.0, 100.0, 200.0, 0.0],
            },
        )

        result = _meadow_farm_index(df, subgroup_col=REGION_COL).df

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("agg_func", "expected_dairy_index"),
        [
            ("sum", 500.0),
            ("mean", 700.0),
            ("max", 600.0),
            ("min", 900.0),
            ("nunique", 100 * 1 / (3 / 7)),
        ],
    )
    def test_agg_func_sets_how_value_col_is_aggregated(self, agg_func, expected_dairy_index):
        """Each agg_func gives Dairy its own index; departments outside the Dairy subset index at 0."""
        # Uneven row counts per department, so sum and mean don't coincide
        df = pd.DataFrame(
            {
                "department": ["Dairy", "Dairy", "Dairy", "Bakery", "Bakery", "Meat", "Meat"],
                cols.unit_spend: [100, 200, 150, 300, 400, 500, 600],
            },
        )

        result = IndexAnalysis(
            df,
            value_col=cols.unit_spend,
            group_col="department",
            index_col="department",
            value_to_index="Dairy",
            agg_func=agg_func,
        ).df

        expected = pd.DataFrame({"department": ["Bakery", "Dairy", "Meat"], "index": [0.0, expected_dairy_index, 0.0]})
        pd.testing.assert_frame_equal(result[["department", "index"]], expected)

    def test_null_values_give_a_null_index_and_missing_rows_give_zero(self):
        """Subset rows whose value_col is all null leave the index unknown; no subset rows at all index 0."""
        df = pd.DataFrame(
            {
                "department": ["Bakery", "Dairy", "Bakery", "Dairy", "Meat"],
                BRAND_COL: ["Meadow Farm", "Meadow Farm", "Greenvale", "Greenvale", "Greenvale"],
                cols.unit_spend: [np.nan, 30.0, 20.0, 10.0, 40.0],
            },
        )
        expected = pd.DataFrame(
            {
                "department": ["Bakery", "Dairy", "Meat"],
                "subset_value": [np.nan, 30.0, 0.0],
                "baseline_value": [20.0, 40.0, 40.0],
                "subset_share": [np.nan, 1.0, 0.0],
                "baseline_share": [0.2, 0.4, 0.4],
                "index": [np.nan, 250.0, 0.0],
            },
        )

        result = IndexAnalysis(
            df,
            value_col=cols.unit_spend,
            group_col="department",
            index_col=BRAND_COL,
            value_to_index="Meadow Farm",
        ).df

        pd.testing.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("data", "subgroup_col", "expected_index"),
        [
            pytest.param(
                {
                    REGION_COL: ["North", "South", "East"],
                    "category": ["Electronics"] * 3,
                    cols.unit_spend: [100, 200, 0],
                },
                None,
                [np.nan, 100.0, 100.0],
                id="zero_baseline_group",
            ),
            pytest.param(
                {
                    REGION_COL: ["North", "South", "North", "South"],
                    "category": ["Electronics", "Electronics", "Grocery", "Grocery"],
                    cols.unit_spend: [0, 0, 100, 200],
                },
                None,
                [np.nan, np.nan],
                id="zero_subset_total",
            ),
            pytest.param(
                {
                    "store_format": ["Mall"] * 4 + ["Outlet"] * 4,
                    REGION_COL: ["North", "South"] * 4,
                    "category": ["Electronics", "Electronics", "Grocery", "Grocery"] * 2,
                    cols.unit_spend: [10, 20, 30, 40, 0, 0, 0, 0],
                },
                "store_format",
                [100 * (10 / 30) / (40 / 100), 100 * (20 / 30) / (60 / 100), np.nan, np.nan],
                id="zero_subgroup_total",
            ),
        ],
    )
    def test_index_is_nan_when_a_share_denominator_is_zero(self, data, subgroup_col, expected_index):
        """A zero total or zero baseline share gives a NaN index rather than raising."""
        result = IndexAnalysis(
            pd.DataFrame(data),
            value_col=cols.unit_spend,
            group_col=REGION_COL,
            index_col="category",
            value_to_index="Electronics",
            subgroup_col=subgroup_col,
        ).df

        pd.testing.assert_series_equal(result["index"], pd.Series(expected_index, name="index"))

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            pytest.param({"agg_func": "median"}, "agg_func must be one of", id="invalid_agg_func"),
            pytest.param(
                {"group_col": "loyalty_tier"},
                "^group_col references columns not present .*loyalty_tier",
                id="missing_group_col",
            ),
            pytest.param(
                {"subgroup_col": "store_format"},
                r"^subgroup_col references columns not present .*store_format",
                id="missing_subgroup_col",
            ),
            pytest.param(
                {"value_col": BRAND_COL},
                "^agg_func 'sum' is not an aggregation available for column 'brand'",
                id="sum_of_text_column",
            ),
            pytest.param(
                {"subgroup_col": AGE_BAND_COL},
                "^subgroup_col must differ from group_col",
                id="subgroup_is_group",
            ),
        ],
    )
    def test_rejects_invalid_arguments(self, brand_customers, kwargs, match):
        """Invalid aggregations and missing columns raise before any query runs."""
        args = {
            "value_col": cols.customer_id,
            "group_col": AGE_BAND_COL,
            "index_col": BRAND_COL,
            "value_to_index": "Meadow Farm",
        }
        with pytest.raises(ValueError, match=match):
            IndexAnalysis(brand_customers, **(args | kwargs))

    @pytest.mark.parametrize(
        ("rows", "value_to_index"),
        [
            pytest.param([], "Meadow farm", id="value_not_in_index_col"),
            pytest.param(
                [{cols.customer_id: 9, BRAND_COL: "Hillside", AGE_BAND_COL: None}],
                "Hillside",
                id="every_subset_row_has_a_null_group",
            ),
        ],
    )
    def test_df_raises_when_the_subset_has_no_rows(self, brand_customers, rows, value_to_index):
        """Reading df says the value must exist exactly and names the null-group cause and fix."""
        df = pd.concat([brand_customers, pd.DataFrame(rows)], ignore_index=True)
        # Building the analysis runs no query, so it cannot know the subset is empty yet
        ia = IndexAnalysis(
            df,
            value_col=cols.customer_id,
            group_col=AGE_BAND_COL,
            index_col=BRAND_COL,
            value_to_index=value_to_index,
            agg_func="nunique",
        )
        expected_message = (
            f"value_to_index={value_to_index!r} matches no rows of index_col 'brand' with a non-null 'age_band', so "
            "there is no subset to index. Pass a value that exists in 'brand', matching its case and spacing exactly. "
            "If its rows have a null 'age_band', fill those first, e.g. df['age_band'].fillna('Unknown') "
            "in pandas or table['age_band'].fill_null('Unknown') in Ibis."
        )

        with pytest.raises(ValueError, match=f"^{re.escape(expected_message)}$"):
            _ = ia.df
