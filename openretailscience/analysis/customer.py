"""Customer Purchase Behavior Analysis for Retention and Value Optimization.

## Business Context

Understanding customer purchase patterns is fundamental to retail success. Some customers
make single purchases and never return, while others become loyal repeat buyers. This
module analyzes the distribution of purchase frequency to identify customer behavior
segments and inform retention strategies.

## The Business Problem

Retailers need to understand the relationship between customer purchase frequency and
business performance:
- What percentage of customers are one-time buyers versus repeat customers?
- How does purchase frequency relate to customer lifetime value?
- Which customer segments offer the greatest growth opportunities?

Without this analysis, businesses may invest equally in all customers or fail to
identify high-potential segments for targeted retention efforts.

## Real-World Applications

### Customer Retention Strategy
- Identify the percentage of one-time buyers for targeted reactivation campaigns
- Segment customers by purchase frequency for differentiated marketing approaches
- Develop loyalty programs based on actual behavior patterns

### Resource Allocation
- Focus retention efforts on customers showing repeat purchase potential
- Allocate customer service resources based on customer value segments
- Optimize marketing spend by targeting high-frequency customer characteristics

### Business Performance Monitoring
- Track changes in purchase frequency distribution over time
- Monitor the health of customer acquisition versus retention balance
- Identify shifts in customer behavior that may indicate market changes

This module computes purchase-frequency statistics that can be visualized with
the plotting helpers in `openretailscience.plots`.
"""

import operator

import pandas as pd

from openretailscience.core.validation import ensure_data_has_columns
from openretailscience.options import ColumnHelper


class PurchasesPerCustomer:
    """Computes the number of purchases per customer.

    Attributes:
        cust_purchases_s (pd.Series): The number of purchases per customer.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the PurchasesPerCustomer class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must comply with the
                contain customer_id and transaction_id columns, which must be non-null.

        Raises:
            ValueError: If the dataframe doesn't contain the columns customer_id and transaction_id, or if the columns
                are null.

        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_id]
        ensure_data_has_columns(df, required_cols)

        self.cust_purchases_s = df.groupby(cols.customer_id)[cols.transaction_id].nunique()

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Get the number of purchases at a given percentile.

        Args:
            percentile (float): The percentile to get the number of purchases at.

        Returns:
            float: The number of purchases at the given percentile.
        """
        return self.cust_purchases_s.quantile(percentile)

    def find_purchase_percentile(self, number_of_purchases: int, comparison: str = "less_than_equal_to") -> float:
        """Find the percentile of the number of purchases.

        Args:
            number_of_purchases (int): The number of purchases to find the percentile of.
            comparison (str, optional): The comparison to use. Defaults to "less_than_equal_to". Must be one of
                less_than, less_than_equal_to, equal_to, not_equal_to, greater_than, or greater_than_equal_to.

        Returns:
            float: The percentile of the number of purchases.
        """
        ops = {
            "less_than": operator.lt,
            "less_than_equal_to": operator.le,
            "equal_to": operator.eq,
            "not_equal_to": operator.ne,
            "greater_than": operator.gt,
            "greater_than_equal_to": operator.ge,
        }

        if comparison not in ops:
            msg = f"Comparison must be one of {', '.join(repr(k) for k in ops)}"
            raise ValueError(msg)

        return len(self.cust_purchases_s[ops[comparison](self.cust_purchases_s, number_of_purchases)]) / len(
            self.cust_purchases_s,
        )


class DaysBetweenPurchases:
    """Computes the average number of days between purchases per customer.

    Attributes:
        purchase_dist_s (pd.Series): The average number of days between purchases per customer.
    """

    def __init__(self, df: pd.DataFrame) -> None:
        """Initialize the DaysBetweenPurchases class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date, which must be non-null.

        Raises:
            ValueError: If the dataframe doesn't contain the columns customer_id and transaction_date, or if the
                columns are null.

        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        ensure_data_has_columns(df, required_cols)

        self.purchase_dist_s = self._calculate_days_between_purchases(df)

    @staticmethod
    def _calculate_days_between_purchases(df: pd.DataFrame) -> pd.Series:
        """Calculate the average number of days between purchases per customer.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date, which must be non-null.

        Returns:
            pd.Series: The average number of days between purchases per customer.
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        ensure_data_has_columns(df, required_cols)

        purchase_dist_df = df[[cols.customer_id, cols.transaction_date]].copy()
        purchase_dist_df[cols.transaction_date] = df[cols.transaction_date].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates().sort_values([cols.customer_id, cols.transaction_date])
        purchase_dist_df["diff"] = purchase_dist_df[cols.transaction_date].diff()
        new_cust_mask = purchase_dist_df[cols.customer_id] != purchase_dist_df[cols.customer_id].shift(1)
        purchase_dist_df = purchase_dist_df[~new_cust_mask]
        purchase_dist_df["diff"] = purchase_dist_df["diff"].dt.days
        return purchase_dist_df.groupby(cols.customer_id)["diff"].mean()

    def purchases_percentile(self, percentile: float = 0.5) -> float:
        """Get the average number of days between purchases at a given percentile.

        Args:
            percentile (float): The percentile to get the average number of days between purchases at.

        Returns:
            float: The average number of days between purchases at the given percentile.
        """
        return self.purchase_dist_s.quantile(percentile)


class TransactionChurn:
    """Computes the churn rate by number of purchases.

    Attributes:
        purchase_dist_df (pd.DataFrame): The churn rate by number of purchases.
        n_unique_customers (int): The number of unique customers in the dataframe.
    """

    def __init__(self, df: pd.DataFrame, churn_period: float) -> None:
        """Initialize the TransactionChurn class.

        Args:
            df (pd.DataFrame): A dataframe with the transaction data. The dataframe must have the columns customer_id
                and transaction_date.
            churn_period (float): The number of days to consider a customer churned.

        Raises:
            ValueError: If the dataframe doesn't contain the columns customer_id and transaction_date.
        """
        cols = ColumnHelper()
        required_cols = [cols.customer_id, cols.transaction_date]
        ensure_data_has_columns(df, required_cols)

        purchase_dist_df = df[[cols.customer_id, cols.transaction_date]].copy()
        # Truncate the transaction_date to the day
        purchase_dist_df[cols.transaction_date] = df[cols.transaction_date].dt.floor("D")
        purchase_dist_df = purchase_dist_df.drop_duplicates()
        purchase_dist_df = purchase_dist_df.sort_values([cols.customer_id, cols.transaction_date])
        purchase_dist_df["transaction_number"] = purchase_dist_df.groupby(cols.customer_id).cumcount() + 1

        purchase_dist_df["last_transaction"] = (
            purchase_dist_df.groupby(cols.customer_id)[cols.transaction_date].shift(-1).isna()
        )
        purchase_dist_df["transaction_before_churn_window"] = purchase_dist_df[cols.transaction_date] < (
            purchase_dist_df[cols.transaction_date].max() - pd.Timedelta(days=churn_period)
        )
        purchase_dist_df["churned"] = (
            purchase_dist_df["last_transaction"] & purchase_dist_df["transaction_before_churn_window"]
        )

        purchase_dist_df = (
            purchase_dist_df[purchase_dist_df["transaction_before_churn_window"]]
            .groupby(["transaction_number"])["churned"]
            .value_counts()
            .unstack()
        )
        purchase_dist_df.columns = ["retained", "churned"]
        purchase_dist_df["churned_pct"] = purchase_dist_df["churned"].div(purchase_dist_df.sum(axis=1))
        self.purchase_dist_df = purchase_dist_df

        self.n_unique_customers = df[cols.customer_id].nunique()
