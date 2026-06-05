"""Shared ibis expression helpers for metric calculations."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
    return (numerator / denominator.nullif(0)) * scale
