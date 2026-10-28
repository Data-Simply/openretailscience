"""Filter ibis table rows by arbitrary boolean conditions and tag the matches."""

import ibis


def filter_and_label_by_condition(
    table: ibis.Table,
    conditions: dict[str, ibis.expr.types.BooleanColumn],
    label_col: str = "label",
) -> ibis.Table:
    """Filter rows matching any condition and tag each kept row with its matching label.

    A row is kept if it matches at least one condition; the label is the first matching
    key in the dict order of ``conditions``.

    Args:
        table (ibis.Table): The ibis table to filter.
        conditions (dict[str, ibis.expr.types.BooleanColumn]): Labels mapped to boolean
            row conditions.
        label_col (str): Name of the label column to add. Defaults to ``"label"``.

    Returns:
        ibis.Table: Filtered table with the label column added.

    Examples:
        >>> from openretailscience.utils import filter_and_label_by_condition
        >>> table = ibis.memtable(transactions)
        >>> drinks = filter_and_label_by_condition(
        ...     table,
        ...     conditions={"Hot": table.category == "Hot Drinks", "Cold": table.category == "Soft Drinks"},
        ...     label_col="drink_type",
        ... )
        >>> drinks.group_by("drink_type").aggregate(spend=drinks.unit_spend.sum()).order_by("drink_type").to_pandas()
          drink_type   spend
        0       Cold   568.1
        1        Hot  1561.2
    """
    branches = [(condition, ibis.literal(label)) for label, condition in conditions.items()]
    combined_condition = ibis.or_(*[condition for condition, _ in branches])

    return table.filter(combined_condition).mutate(**{label_col: ibis.cases(*branches)})
