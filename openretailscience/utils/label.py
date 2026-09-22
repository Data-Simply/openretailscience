"""Label ibis groups by whether items meet a condition (binary or extended strategies)."""

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
    """Label groups in a table based on whether their items meet a condition.

    binary: a group is labeled ``contains`` if ANY item meets the condition, else
    ``not_contains``. extended: ``contains`` (all items meet it), ``mixed`` (some do),
    ``not_contains`` (none do).

    Args:
        table (ibis.Table): The ibis table to label.
        condition (ibis.expr.types.BooleanColumn): Boolean expression evaluated per row.
        label_col (str | None): Column to group by. If None, uses the ``column.customer_id``
            option.
        return_col (str): Name of the added label column.
        labeling_strategy (Literal["binary", "extended"]): Labeling strategy.
        contains_label (str | ibis.Value): Label for contains.
        not_contains_label (str | ibis.Value): Label for not_contains.
        mixed_label (str | ibis.Value): Label for mixed (extended strategy only).

    Returns:
        ibis.Table: One row per group, with the added label column.
    """
    if label_col is None:
        label_col = get_option("column.customer_id")

    table = table.mutate(label_condition=condition.ifelse(1, 0)).group_by(label_col)

    if labeling_strategy == "binary":
        return table.aggregate(
            {
                return_col: (_.label_condition.max() == 1).ifelse(contains_label, not_contains_label),
            },
        )

    return table.aggregate(
        {
            return_col: ibis.cases(
                ((_.label_condition.max() == 1) & (_.label_condition.min() == 1), contains_label),
                ((_.label_condition.max() == 1) & (_.label_condition.min() == 0), mixed_label),
                else_=not_contains_label,
            ),
        },
    )
