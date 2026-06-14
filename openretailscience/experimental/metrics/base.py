"""Shared ibis expression helpers for metric calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import ibis

if TYPE_CHECKING:
    import ibis.expr.types as ir

PERCENTAGE_SCALE = 100


def ratio_metric(
    numerator: ir.NumericValue,
    denominator: ir.NumericValue,
    scale: float = PERCENTAGE_SCALE,
) -> ir.FloatingValue:
    """Computes a scaled ratio, returning NULL on zero denominator.

    Args:
        numerator (ir.NumericValue): The numerator ibis expression.
        denominator (ir.NumericValue): The denominator ibis expression.
        scale (float, optional): Multiplicative scale factor. Defaults to 100 for percentages.

    Returns:
        ir.FloatingValue: The scaled ratio expression. Evaluates to NULL
            when the denominator is zero (materializes as NaN in pandas).
    """
    # Division/multiplication on ibis NumericValue is statically typed as NumericValue,
    # but the runtime result of a float ratio is always a FloatingValue. Cast to the
    # documented narrower return type.
    return cast("ir.FloatingValue", (numerator / denominator.nullif(ibis.literal(0))) * scale)
