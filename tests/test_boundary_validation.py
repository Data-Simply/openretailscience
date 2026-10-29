"""Bad input is rejected at each public constructor, with an error naming the parameter at fault."""

import pandas as pd
import pytest

from openretailscience.analysis.cohort import CohortAnalysis
from openretailscience.analysis.cross_shop import CrossShop
from openretailscience.analysis.customer import DaysBetweenPurchases, TransactionChurn
from openretailscience.analysis.customer_decision_hierarchy import CustomerDecisionHierarchy
from openretailscience.analysis.gain_loss import GainLoss
from openretailscience.analysis.product_association import ProductAssociation
from openretailscience.analysis.revenue_tree import RevenueTree
from openretailscience.experimental.clv import CLVStats
from openretailscience.options import ColumnHelper
from openretailscience.segmentation.hml import HMLSegmentation
from openretailscience.segmentation.nlr import NLRSegmentation
from openretailscience.segmentation.rfm import RFMSegmentation
from openretailscience.segmentation.threshold import ThresholdSegmentation

cols = ColumnHelper()


@pytest.fixture
def transactions() -> pd.DataFrame:
    """Two customers buying Coke and Pepsi across two quarters."""
    return pd.DataFrame(
        {
            cols.customer_id: [101, 101, 102, 102],
            cols.transaction_id: [9001, 9002, 9003, 9004],
            cols.unit_spend: [24.99, 61.50, 12.75, 149.00],
            cols.transaction_date: pd.to_datetime(["2024-02-03", "2024-08-14", "2024-03-21", "2024-09-02"]),
            "brand": ["Coke", "Pepsi", "Pepsi", "Coke"],
            "quarter": ["2024Q1", "2024Q3", "2024Q1", "2024Q3"],
        },
    )


def _gain_loss(df: pd.DataFrame, **kwargs: str) -> GainLoss:
    return GainLoss(
        df=df,
        p1_index=df["quarter"] == "2024Q1",
        p2_index=df["quarter"] == "2024Q3",
        focus_group_index=df["brand"] == "Coke",
        focus_group_name="Coke",
        comparison_group_index=df["brand"] == "Pepsi",
        comparison_group_name="Pepsi",
        **kwargs,
    )


DATE_CONSTRUCTORS = [
    pytest.param(lambda df: CohortAnalysis(df, aggregation_column=cols.customer_id), id="CohortAnalysis"),
    pytest.param(lambda df: RFMSegmentation(df, current_date="2024-12-31"), id="RFMSegmentation"),
    pytest.param(DaysBetweenPurchases, id="DaysBetweenPurchases"),
    pytest.param(lambda df: TransactionChurn(df, churn_period=90), id="TransactionChurn"),
    pytest.param(CLVStats, id="CLVStats"),
]


class TestDateColumn:
    """transaction_date must be a real date or datetime column in every class that reads it."""

    @pytest.mark.parametrize(
        "dates",
        [
            pytest.param(["2024-02-03", "2024-08-14", "2024-03-21", "2024-09-02"], id="iso_string"),
            pytest.param(["02/03/2024", "08/14/2024", "03/21/2024", "09/02/2024"], id="non_iso_string"),
            pytest.param([20240203, 20240814, 20240321, 20240902], id="integer_yyyymmdd"),
        ],
    )
    @pytest.mark.parametrize("construct", DATE_CONSTRUCTORS)
    def test_rejects_non_date_transaction_date(self, transactions, construct, dates):
        """String and integer dates raise before any query runs, naming the column and its type."""
        with pytest.raises(TypeError, match=rf"'{cols.transaction_date}' must be a date or datetime type\. Got \w+\.$"):
            construct(transactions.assign(**{cols.transaction_date: dates}))

    @pytest.mark.parametrize("construct", DATE_CONSTRUCTORS)
    def test_rejects_timezone_aware_transaction_date(self, transactions, construct):
        """A tz-aware timestamp, which the backend would shift to UTC day boundaries, raises naming the column."""
        aware = transactions[cols.transaction_date].dt.tz_localize("Europe/London")
        with pytest.raises(ValueError, match=rf"'{cols.transaction_date}' is timezone-aware \(Europe/London\)"):
            construct(transactions.assign(**{cols.transaction_date: aware}))


class TestAggFunc:
    """An agg_func that isn't an aggregation is rejected naming the parameter and the value passed."""

    @pytest.mark.parametrize(
        "construct",
        [
            pytest.param(
                lambda df, agg_func: CohortAnalysis(df, aggregation_column=cols.unit_spend, agg_func=agg_func),
                id="CohortAnalysis",
            ),
            pytest.param(
                lambda df, agg_func: ThresholdSegmentation(
                    df, thresholds=[0.5, 1.0], segments=["Light", "Heavy"], agg_func=agg_func
                ),
                id="ThresholdSegmentation",
            ),
            pytest.param(lambda df, agg_func: HMLSegmentation(df, agg_func=agg_func), id="HMLSegmentation"),
            pytest.param(
                lambda df, agg_func: CrossShop(
                    df, group_1_col="brand", group_1_val="Coke", group_2_val="Pepsi", agg_func=agg_func
                ),
                id="CrossShop",
            ),
            pytest.param(lambda df, agg_func: _gain_loss(df, agg_func=agg_func), id="GainLoss"),
        ],
    )
    # cumsum is a real method on both Ibis columns and pandas groupbys, but returns one row per transaction.
    # prod reduces, but GainLoss's empty customer groups would get prod's empty value 1 rather than 0.
    # group_concat reduces, but to text that can't be summed, ranked or compared with 0.
    @pytest.mark.parametrize(
        "agg_func",
        ["summ", "cumsum", "prod", "group_concat"],
        ids=["typo", "running_total", "product_of_empty_group_is_one", "text_result"],
    )
    def test_rejects_non_aggregation_agg_func(self, transactions, construct, agg_func):
        """A misspelt name, a non-reducing method, prod or a non-numeric result raises ValueError before any query."""
        with pytest.raises(ValueError, match=f"agg_func '{agg_func}'"):
            construct(transactions, agg_func)


class TestGroupColExcludesCustomer:
    """group_col can't be the customer column in analyses that already group by customer."""

    @pytest.mark.parametrize(
        "construct",
        [
            pytest.param(
                lambda df: ThresholdSegmentation(
                    df,
                    thresholds=[0.5, 1.0],
                    segments=["Light", "Heavy"],
                    group_col=cols.customer_id,
                ),
                id="ThresholdSegmentation",
            ),
            pytest.param(lambda df: HMLSegmentation(df, group_col=[cols.customer_id]), id="HMLSegmentation"),
            pytest.param(
                lambda df: NLRSegmentation(
                    df,
                    period_col="quarter",
                    p1_value="2024Q1",
                    p2_value="2024Q3",
                    group_col=cols.customer_id,
                ),
                id="NLRSegmentation",
            ),
            pytest.param(lambda df: _gain_loss(df, group_col=cols.customer_id), id="GainLoss"),
        ],
    )
    def test_rejects_customer_column(self, transactions, construct):
        """Passing the customer column as group_col raises a ValueError naming group_col."""
        with pytest.raises(ValueError, match=f"group_col must not include the customer column '{cols.customer_id}'"):
            construct(transactions)


class TestNonDataFrameInput:
    """A list in place of the data is reported as a type error, not an attribute error on the list."""

    @pytest.mark.parametrize(
        "construct",
        [
            pytest.param(lambda data: CohortAnalysis(data, aggregation_column=cols.customer_id), id="CohortAnalysis"),
            pytest.param(
                lambda data: CrossShop(data, group_1_col="brand", group_1_val="Coke", group_2_val="Pepsi"),
                id="CrossShop",
            ),
            pytest.param(lambda data: ProductAssociation(data, value_col="brand"), id="ProductAssociation"),
            pytest.param(
                lambda data: RevenueTree(data, period_col="quarter", p1_value="2024Q1", p2_value="2024Q3"),
                id="RevenueTree",
            ),
            pytest.param(
                lambda data: GainLoss(
                    data,
                    p1_index=[True, False],
                    p2_index=[False, True],
                    focus_group_index=[True, False],
                    focus_group_name="Coke",
                    comparison_group_index=[False, True],
                    comparison_group_name="Pepsi",
                ),
                id="GainLoss",
            ),
            pytest.param(
                lambda data: CustomerDecisionHierarchy(data, product_col="brand"), id="CustomerDecisionHierarchy"
            ),
        ],
    )
    def test_rejects_list(self, construct):
        """A plain list of spend values raises TypeError naming the DataFrame requirement."""
        with pytest.raises(TypeError, match=r"df must be .*DataFrame"):
            construct([24.99, 61.50])
