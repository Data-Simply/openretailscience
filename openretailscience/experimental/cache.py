"""Generic Ibis-expression caching for openretailscience, with a Spark Connect (Databricks) workaround.

The public entry point is :func:`cache`. It mirrors Ibis's native ``Table.cache()``: the handle it
returns is usable directly as a table, doubles as a context manager, and is released explicitly with
``.release()`` or automatically when garbage-collected.

Why this exists: Ibis's native ``Table.cache()`` raises on the PySpark backend under Spark Connect
(Databricks Runtime 13 and later) because its ``DataFrame.is_cached`` assertion is not satisfied
synchronously. :func:`cache` transparently substitutes a temporary-view-based implementation there and
delegates to the native ``.cache()`` on every other backend (including classic PySpark, where the native
path works). Caching can be turned off globally via the ``optimization.use_caching`` option or per call via
``enabled=False``, in which case :func:`cache` returns the expression unchanged behind the same interface,
so call sites never have to branch.

.. warning::
    This module is experimental and its API may change without notice. Unlike Ibis's native cache, the
    Spark Connect path does not de-duplicate: caching the same expression twice creates two independent
    temporary views over a single shared Spark cache entry, so releasing one uncaches the other.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING

import ibis
import ibis.expr.types as ir
from ibis import util

from openretailscience.options import get_option

if TYPE_CHECKING:
    from ibis.backends import BaseBackend

__all__ = ["DatabricksCachedTable", "cache"]

# Base for auto-generated Spark temp-view names (``util.gen_name`` yields e.g. ``ibis_ors_cache_<hash>``),
# so cached relations are recognizable in the catalog and unique per call.
_CACHE_NAME_PREFIX = "ors_cache"

# Ibis backend name for the PySpark backend.
_PYSPARK_BACKEND_NAME = "pyspark"

# Spark Connect session classes live under ``pyspark.sql.connect.*``; classic sessions do not. This is how
# the native ``.cache()`` path is distinguished from the one that trips Ibis's ``is_cached`` assertion.
_SPARK_CONNECT_MODULE_PREFIX = "pyspark.sql.connect"


def _release_view(con: BaseBackend, name: str) -> None:
    """Uncache a Spark relation and drop its temporary view, if it still exists.

    Idempotent: a no-op when the view is already gone, so it is safe to call from both an explicit
    :meth:`DatabricksCachedTable.release` and the garbage-collection finalizer. Presence is checked
    up front instead of catching a broad exception, so a real failure (e.g. a live session refusing
    the drop) surfaces rather than being silently swallowed.

    Args:
        con (BaseBackend): The Ibis PySpark backend connection holding the view.
        name (str): The temporary view name to uncache and drop.
    """
    catalog = con._session.catalog
    if not catalog.tableExists(name):
        return
    # Drop the view even if the uncache fails, so a transient uncache error cannot leak the view; the
    # uncache error still surfaces once the drop has run.
    try:
        catalog.uncacheTable(name)
    finally:
        catalog.dropTempView(name)


class DatabricksCachedTable(ir.CachedTable):
    """An Ibis table backed by a cached Spark temporary view, with Ibis-native cache lifecycle.

    Behaves like any Ibis table (build further expressions on it directly), doubles as a context manager,
    and is released with :meth:`release` or automatically when garbage-collected. Instances are created by
    :func:`cache`; they are not meant to be constructed directly.
    """

    def release(self) -> None:
        """Uncache the Spark relation and drop its temporary view."""
        con = self._find_backend(use_default=True)
        _release_view(con, self.op().name)


class _PassthroughCachedTable(ir.CachedTable):
    """A no-op cache handle returned when caching is disabled: the original expression, unchanged."""

    def release(self) -> None:
        """Nothing was materialized, so there is nothing to release."""


def _is_spark_connect(con: BaseBackend) -> bool:
    """Whether ``con`` is an Ibis PySpark backend running against Spark Connect.

    Native ``Table.cache()`` works on classic PySpark; only Spark Connect (the mode Databricks Runtime 13+
    uses) trips Ibis's ``is_cached`` assertion and needs the temporary-view workaround.

    Args:
        con (BaseBackend): The Ibis backend connection to inspect.

    Returns:
        bool: ``True`` if ``con`` is a PySpark backend backed by a Spark Connect session.
    """
    if getattr(con, "name", None) != _PYSPARK_BACKEND_NAME:
        return False
    session = getattr(con, "_session", None)
    return session is not None and type(session).__module__.startswith(_SPARK_CONNECT_MODULE_PREFIX)


def _cache_on_spark_connect(con: BaseBackend, expr: ir.Table) -> DatabricksCachedTable:
    """Cache ``expr`` as a Spark temporary view and return a handle that releases it (also on GC).

    Args:
        con (BaseBackend): The Ibis PySpark backend connection ``expr`` is bound to.
        expr (ir.Table): The Ibis table expression to cache.

    Returns:
        DatabricksCachedTable: A cache handle bound to the temporary view.
    """
    name = util.gen_name(_CACHE_NAME_PREFIX)
    spark_df = con._session.sql(con.compile(expr))
    spark_df.cache()
    spark_df.createOrReplaceTempView(name)
    cached = DatabricksCachedTable(con.table(name).op())
    # Drop the view when the cached relation is garbage-collected, so a forgotten cache does not leak.
    # The finalizer is attached to the op, not the ``cached`` expression wrapper: Ibis wrappers are
    # ephemeral, but every expression derived from the cache references the op and keeps it alive, so the
    # view is released only once nothing uses it -- mirroring how Ibis's native cache finalizes on the
    # cached op. (Attaching to the wrapper would drop the view the moment ``cached`` is reassigned or
    # goes out of scope, even while derived expressions still reference it.) The name is unique per call,
    # so this can only ever drop its own relation. ``atexit=False`` skips release during interpreter
    # shutdown -- the Spark session is torn down with its temp views then, and weakref.finalize runs its
    # atexit pass with ``sys.is_finalizing()`` still False, so an in-callback guard would not fire.
    finalizer = weakref.finalize(cached.op(), _release_view, con, name)
    finalizer.atexit = False
    return cached


def cache(expr: ir.Table, *, enabled: bool | None = None) -> ir.CachedTable:
    """Cache an Ibis expression, choosing the right strategy for its backend.

    Mirrors Ibis's native ``Table.cache()``: the returned handle is usable directly as a table, as a
    context manager, or released explicitly with ``.release()``. On Spark Connect (Databricks) it uses a
    temporary-view workaround for Ibis's broken native cache; on every other backend it delegates to the
    native ``.cache()``. When caching is disabled -- globally via the ``optimization.use_caching`` option or
    per call via ``enabled=False`` -- the expression is returned unchanged behind the same interface.

    Args:
        expr (ir.Table): The Ibis table expression to cache. Must be bound to a backend.
        enabled (bool | None, optional): Per-call override of the ``optimization.use_caching`` option.
            Defaults to ``None``, which reads the option.

    Returns:
        ir.CachedTable: A cache handle. Call ``.release()`` or use it as a context manager to free it.

    Raises:
        TypeError: If ``expr`` is not an Ibis table, or if the resolved ``enabled`` flag is not a bool.
    """
    if not isinstance(expr, ibis.Table):
        msg = f"expr must be an Ibis table. Got {type(expr).__name__}."
        raise TypeError(msg)
    resolved_enabled = get_option("optimization.use_caching") if enabled is None else enabled
    if not isinstance(resolved_enabled, bool):
        msg = f"enabled must be a bool or None. Got {type(resolved_enabled).__name__}."
        raise TypeError(msg)
    if not resolved_enabled:
        return _PassthroughCachedTable(expr.op())
    con = expr.get_backend()
    if _is_spark_connect(con):
        return _cache_on_spark_connect(con, expr)
    return expr.cache()
