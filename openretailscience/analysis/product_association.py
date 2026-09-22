"""Market-basket (product association) rules from transaction data.

Rules are computed lazily in the Ibis engine: ``table`` is the lazy Ibis table and
``df`` materializes it.
"""

import functools

import ibis
import pandas as pd
from ibis import _

from openretailscience.core.validation import ensure_data_has_columns, ensure_ibis_table
from openretailscience.options import get_option


class ProductAssociation:
    """Association rules between products from transaction data.

    The unit of analysis is the distinct (group, product) pair. Output columns are
    ``{value_col}_1`` / ``{value_col}_2`` (the product pair), ``occurrences_1`` /
    ``occurrences_2`` (count of distinct ``group_col`` values containing that product),
    ``cooccurrences`` (distinct ``group_col`` values containing both), plus:

    - ``support = cooccurrences / total``, where ``total`` is the count of distinct
      ``group_col`` values over the ENTIRE input — not just groups containing either product.
    - ``confidence = cooccurrences / occurrences_1`` = P(product_2 | product_1) — asymmetric.
    - ``uplift = support / (prob_1 * prob_2)`` — may legitimately exceed 1.

    With ``target_item=None`` both directions (a, b) and (b, a) are emitted, each with its
    own confidence; with a given ``target_item`` only rows with the target as product_1 are
    emitted. ``min_confidence`` filters both directions after the union.

    ``table`` is the lazy Ibis result; ``df`` materializes it.
    """

    def __init__(
        self,
        df: pd.DataFrame | ibis.Table,
        value_col: str,
        group_col: str | None = None,
        target_item: str | None = None,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
    ) -> None:
        """Initialize the ProductAssociation and compute the rules lazily.

        Args:
            df (pd.DataFrame | ibis.Table): Transaction data with one row per (group, product) occurrence.
            value_col (str): Column containing the product identifiers.
            group_col (str | None): Column identifying unique transactions or customers; defaults to
                option ``column.customer_id``.
            target_item (str | float | list[str | float] | None): Product(s) to focus the analysis on;
                if None, associations for all products are calculated.
            min_occurrences (int): Minimum distinct-group count per product; must be >= 1. Defaults to 1.
            min_cooccurrences (int): Minimum distinct-group count per pair; must be >= 1. Defaults to 1.
            min_support (float): Minimum support; must be in [0, 1]. Defaults to 0.0.
            min_confidence (float): Minimum confidence; must be in [0, 1]. Defaults to 0.0.
            min_uplift (float): Minimum uplift; must be >= 0. Defaults to 0.0.

        Raises:
            ValueError: If any ``min_*`` value is out of range.
            ValueError: If ``target_item`` is an empty list.
            TypeError: If ``target_item`` contains values that are not str or float.
            ValueError: If ``group_col`` or ``value_col`` columns are missing from the data.
        """
        group_col = group_col or get_option("column.customer_id")
        required_cols = [group_col, value_col]
        ensure_data_has_columns(df, required_cols)

        self.table = self._calc_association(
            df=df,
            value_col=value_col,
            group_col=group_col,
            target_item=target_item,
            min_occurrences=min_occurrences,
            min_cooccurrences=min_cooccurrences,
            min_support=min_support,
            min_confidence=min_confidence,
            min_uplift=min_uplift,
        )

    @staticmethod
    def _validate_minimum_values(
        min_occurrences: int,
        min_cooccurrences: int,
        min_support: float,
        min_confidence: float,
        min_uplift: float,
    ) -> None:
        """Validate the minimum value parameters.

        Args:
            min_occurrences (int): Minimum occurrences per product; must be >= 1.
            min_cooccurrences (int): Minimum co-occurrences per pair; must be >= 1.
            min_support (float): Minimum support; must be in [0, 1].
            min_confidence (float): Minimum confidence; must be in [0, 1].
            min_uplift (float): Minimum uplift; must be >= 0. Uplift is a ratio and may
                legitimately exceed 1.

        Raises:
            ValueError: If any parameter is outside its valid range.
        """
        if min_occurrences < 1:
            raise ValueError("Minimum occurrences must be at least 1")
        if min_cooccurrences < 1:
            raise ValueError("Minimum cooccurrences must be at least 1")
        if min_support < 0.0 or min_support > 1.0:
            raise ValueError("Minimum support must be between 0 and 1")
        if min_confidence < 0.0 or min_confidence > 1.0:
            raise ValueError("Minimum confidence must be between 0 and 1")
        if min_uplift < 0.0:
            raise ValueError("Minimum uplift must be greater or equal to 0")

    @staticmethod
    def _calc_association(
        df: pd.DataFrame | ibis.Table,
        value_col: str,
        group_col: str = get_option("column.customer_id"),
        target_item: str | float | list[str | float] | None = None,
        min_occurrences: int = 1,
        min_cooccurrences: int = 1,
        min_support: float = 0.0,
        min_confidence: float = 0.0,
        min_uplift: float = 0.0,
    ) -> pd.DataFrame:
        """Build the association rules as a lazy Ibis expression tree.

        Returns the result table without executing it. The unit of analysis is the
        distinct (group, product) pair; ``min_confidence`` is applied after the
        both-directions union. Argument semantics as in ``ProductAssociation.__init__``.

        Returns:
            ibis.Table: Lazy result table with columns ``{value_col}_1``, ``{value_col}_2``,
                ``occurrences_1``, ``occurrences_2``, ``cooccurrences``, ``support``,
                ``confidence``, ``uplift``.

        Raises:
            ValueError: Out-of-range ``min_*`` values (see ``_validate_minimum_values``) or
                an empty ``target_item`` list.
            TypeError: If ``target_item`` items are not str or float.
        """
        ProductAssociation._validate_minimum_values(
            min_occurrences=min_occurrences,
            min_cooccurrences=min_cooccurrences,
            min_support=min_support,
            min_confidence=min_confidence,
            min_uplift=min_uplift,
        )

        # Normalize target_item to a list for consistent processing
        if target_item is not None:
            if not isinstance(target_item, list):
                target_item = [target_item]

            # Validate that all items in target_item are of supported types
            for item in target_item:
                if not isinstance(item, str | float):
                    msg = f"target_item must contain only str or float values. Got {type(item)}"
                    raise TypeError(msg)

            # Ensure target_item is not empty
            if len(target_item) == 0:
                raise ValueError("target_item cannot be an empty list")

        df = ensure_ibis_table(df)

        unique_transactions = df.select(_[group_col], _[value_col]).distinct()
        total_transactions = unique_transactions.alias("t")[group_col].nunique().name("total_count")

        product_occurrences = (
            unique_transactions.group_by(value_col)
            .aggregate(occurrences=_[group_col].nunique())
            .mutate(occurrence_probability=_.occurrences / total_transactions)
            .filter(_.occurrences >= min_occurrences)
        )

        left_table = unique_transactions.rename({"item_1": value_col})
        right_table = unique_transactions.rename({"item_2": value_col})

        join_logic = [left_table[group_col] == right_table[group_col]]
        if target_item is None:
            join_logic.append(left_table.item_1 < right_table.item_2)
        else:
            join_logic.extend(
                [
                    left_table.item_1 != right_table.item_2,
                    left_table.item_1.isin(target_item),
                ],
            )
        merged_df = left_table.join(right_table, predicates=join_logic)

        cooccurrences = (
            merged_df.group_by(["item_1", "item_2"])
            .aggregate(cooccurrences=merged_df[group_col].nunique())
            .mutate(support=_.cooccurrences / total_transactions)
            .filter((_.cooccurrences >= min_cooccurrences) & (_.support >= min_support))
        )

        product_occurrences_1_rename = product_occurrences.rename(
            {"item_1": value_col, "occurrences_1": "occurrences", "prob_1": "occurrence_probability"},
        )
        product_occurrences_2_rename = product_occurrences.rename(
            {"item_2": value_col, "occurrences_2": "occurrences", "prob_2": "occurrence_probability"},
        )

        result = (
            cooccurrences.join(product_occurrences_1_rename, "item_1")
            .join(product_occurrences_2_rename, "item_2")
            .mutate(
                confidence=_.cooccurrences / _.occurrences_1,
                uplift=_.support / (_.prob_1 * _.prob_2),
            )
            .filter(_.uplift >= min_uplift)
        )

        if target_item is None:
            col_order = [
                "item_1",
                "item_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
            inverse_pairs = result.mutate(
                item_1=result.item_2,
                item_2=result.item_1,
                occurrences_1=result.occurrences_2,
                occurrences_2=result.occurrences_1,
                prob_1=result.prob_2,
                prob_2=result.prob_1,
                confidence=result.cooccurrences / result.occurrences_2,
            )
            result = result[col_order].union(inverse_pairs[col_order])

        result = (
            result.filter(result.confidence >= min_confidence)
            .order_by(["item_1", "item_2"])
            .rename({f"{value_col}_1": "item_1", f"{value_col}_2": "item_2"})
        )
        return result[
            [
                f"{value_col}_1",
                f"{value_col}_2",
                "occurrences_1",
                "occurrences_2",
                "cooccurrences",
                "support",
                "confidence",
                "uplift",
            ]
        ]

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Materialized association rules DataFrame (cached after first execution)."""
        return self.table.execute().reset_index(drop=True)
