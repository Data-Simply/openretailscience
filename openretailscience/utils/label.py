"""Label Utilities - Labeling by Condition.

This module provides utilities to label groups in an Ibis table based on whether
any items in the group meet a specified condition. It supports both binary labeling
(contains/not_contains) and extended labeling (contains/mixed/not_contains).

Example use cases:
- Tag transactions as containing a product, product category, or promotion
- Tag customers as containing a product, product category, or promotion, or store_id
- Tag a transaction as containing promo items, no promo items, or both on/off promo items
"""

from typing import Literal

import ibis
from ibis import _

from openretailscience.options import get_option


def label_by_condition(
    table: ibis.Table,
    condition: ibis.expr.types.BooleanColumn,
    label_col: str | None = None,
    return_col: str = "label_name",
    labeling_strategy: Literal["binary", "extended"] = "binary",
    contains_label: str | ibis.Value = "contains",
    not_contains_label: str | ibis.Value = "not_contains",
    mixed_label: str | ibis.Value = "mixed",
) -> ibis.Table:
    """Labels groups in a table based on whether items in the group meet a condition.

    This function groups a table by the specified label column and determines whether
    any items in each group meet the given condition. It supports two labeling strategies:
    - "binary": Labels groups as either "contains" or "not_contains" based on whether
      any item in the group meets the condition
    - "extended": Labels groups as "contains" (all items meet condition), "mixed"
      (some items meet condition), or "not_contains" (no items meet condition)

    Args:
        table (ibis.Table): An ibis table to process.
        condition (ibis.expr.types.BooleanColumn): Boolean expression representing
            the condition to evaluate.
        label_col (str | None, optional): Column name to group by for labeling. If None, defaults to the setting
            column.customer_id.
        return_col (str, optional): Name of the column to add with the labels. Defaults to "label_name".
        labeling_strategy (Literal["binary", "extended"], optional): Strategy for labeling groups.
            Defaults to "binary".
        contains_label (str | ibis.Value, optional): Label for groups that contain
            items meeting the condition. Defaults to "contains".
        not_contains_label (str | ibis.Value, optional): Label for groups that do not
            contain items meeting the condition. Defaults to "not_contains".
        mixed_label (str | ibis.Value, optional): Label for groups with mixed results
            (only used with "extended" strategy). Defaults to "mixed".

    Returns:
        ibis.Table: An ibis table grouped by label_col with an added label column.
    """
    if label_col is None:
        label_col = get_option("column.customer_id")

    grouped = table.mutate(label_condition=condition.ifelse(1, 0)).group_by(label_col)

    if labeling_strategy == "binary":
        return grouped.aggregate(
            (_.label_condition.max() == 1).ifelse(contains_label, not_contains_label).name(return_col),
        )

    return grouped.aggregate(
        ibis.cases(
            ((_.label_condition.max() == 1) & (_.label_condition.min() == 1), contains_label),
            ((_.label_condition.max() == 1) & (_.label_condition.min() == 0), mixed_label),
            else_=not_contains_label,
        ).name(return_col),
    )
