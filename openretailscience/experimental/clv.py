"""Customer-lifetime-value (BTYD) model input preparation.

This module adapts OpenRetailScience transaction data into the per-customer summary
frame consumed by the "buy-till-you-die" models in
`pymc-marketing <https://www.pymc-marketing.io/>`_ — ``ParetoNBDModel`` (churn and
purchase frequency) and ``GammaGammaModel`` (per-transaction spend).

The summary follows the standard BTYD convention, where each customer is "born" at their
first purchase and observed until ``observation_period_end``:

- ``frequency`` — number of *repeat* purchase occasions (total occasions minus one). An
  occasion is a distinct calendar day with positive net spend (returns-only days excluded).
- ``recency`` — time from the first purchase to the last purchase.
- ``T`` — the customer's "age": time from the first purchase to ``observation_period_end``.
- ``monetary_value`` — mean spend across the *repeat* purchases (excludes the first, per the
  Gamma-Gamma assumption). ``NaN`` for one-time buyers, who fall back to the population mean.

``recency`` and ``T`` are expressed in ``period`` units (``"day"``, ``"week"``, or ``"month"``;
a month is a fixed 365.25/12-day unit) and are fractional (e.g. 2.5 weeks). The elapsed time is
measured in whole days and then divided by the days-per-period, because the Oracle backend supports
only day-granularity date deltas (see ``openretailscience.analysis.cohort`` for the same constraint).

The first purchase is always the earliest transaction *in the supplied data*, exactly as
`pymc_marketing.clv.rfm_summary` treats it. A customer who was already active before the
data window is therefore treated as born at the window's start — a known limitation of a
truncated observation period, not something this adapter corrects.
"""

from __future__ import annotations

import datetime
import functools
import warnings
from typing import TYPE_CHECKING, ClassVar

import ibis
import pandas as pd

from openretailscience.core.validation import (
    ensure_columns,
    ensure_data_has_columns,
    ensure_ibis_table,
    ensure_integer,
    ensure_period,
    ensure_positive,
    ensure_tznaive_datetime,
    ensure_unit_interval,
)
from openretailscience.options import ColumnHelper

if TYPE_CHECKING:
    from typing import Self


def _coerce_to_date(value: str | datetime.date) -> datetime.date:
    """Normalize a date-like value to a ``datetime.date``.

    Args:
        value (str | datetime.date): An ISO-8601 date string, a ``datetime.date``, or a
            ``datetime.datetime`` (its date part is used).

    Returns:
        datetime.date: The normalized date.

    Raises:
        TypeError: If ``value`` is not a string or ``datetime.date``.
    """
    if isinstance(value, str):
        return datetime.date.fromisoformat(value)
    # ``datetime.datetime`` is a ``datetime.date`` subclass; take its date part explicitly.
    if isinstance(value, datetime.datetime):
        return value.date()
    if isinstance(value, datetime.date):
        return value
    msg = "observation_period_end must be an ISO-8601 date string, a datetime.date, or None."
    raise TypeError(msg)


#: Past this many dummies, a one_hot_col is almost certainly a high-cardinality column (product_id, a
#: per-transaction field) encoded by mistake.
_ONE_HOT_CARDINALITY_WARN = 32


def _one_hot_encode(table: ibis.Table, cols_to_encode: list[str]) -> ibis.Table:
    """One-hot encode customer-constant columns into 0/1 dummy columns.

    Each distinct value ``v`` of a column becomes ``{col}_{v}`` (raw value verbatim in the name). Per
    column, one level is dropped as the reference (dummies would otherwise be collinear with a model
    intercept): NULL when the column has NULLs, else the first value in sorted order. Dropping NULL rather
    than a NULL-indicator column is a pure reparameterization (identical fit) that needs no NULL name.

    Categories are read from ``table`` itself in a single ``distinct`` query over all ``cols_to_encode``,
    so N columns cost one execute, not N.

    Args:
        table (ibis.Table): The customer-level table containing every column in ``cols_to_encode`` (one
            row per customer); distinct values, dtypes, and dummies are all computed from it.
        cols_to_encode (list[str]): The columns to one-hot encode.

    Returns:
        ibis.Table: ``table`` with each encoded column replaced by its 0/1 dummy columns.
    """
    combos = table.select(cols_to_encode).distinct().execute()
    dummies = {}
    for col in cols_to_encode:
        # combos holds distinct *combinations*, so each column repeats a value once per pairing; dedupe
        # per column or the reference-level drop (values[1:]) would leave the reference value in place.
        distinct = combos[col]
        has_null = bool(distinct.isna().any())
        non_null = distinct.dropna().drop_duplicates()
        if table[col].type().is_integer():
            # A nullable integer column executes to float64 in pandas; restore integer values so dummy
            # columns are named `{col}_2` (not `{col}_2.0`) and the equality below is an exact int match.
            non_null = non_null.astype("int64")
        values = sorted(non_null.tolist())
        emitted = values if has_null else values[1:]
        if len(emitted) == 0:
            warnings.warn(
                f"one_hot_col '{col}' has a single distinct level (one category, or all NULL), so dropping the "
                "reference level leaves zero dummy columns and no covariate. Drop it from one_hot_col, or check "
                "the column has the categories you expect.",
                stacklevel=4,  # user -> __init__ -> _attach_customer_attributes -> _one_hot_encode -> warn
            )
        elif len(emitted) > _ONE_HOT_CARDINALITY_WARN:
            warnings.warn(
                f"one_hot_col '{col}' expands to {len(emitted)} dummy columns; high-cardinality one-hot "
                "makes a large, sparse ParetoNBD covariate matrix. Group rare levels or use another encoding.",
                stacklevel=4,  # user -> __init__ -> _attach_customer_attributes -> _one_hot_encode -> warn
            )
        # ifelse -> portable CASE WHEN (`col == value` as a value is invalid SQL on Oracle/SQL Server);
        # NULL or non-match -> 0. int8 (TINYINT) not int64: a 0/1 flag is one byte.
        dummies.update({f"{col}_{value}": ibis.ifelse(table[col] == value, 1, 0).cast("int8") for value in emitted})
    return table.mutate(**dummies).drop(*cols_to_encode)


class CLVStats:
    """Prepares the per-customer BTYD summary for pymc-marketing CLV models.

    Aggregates transaction data into the ``frequency`` / ``recency`` / ``T`` /
    ``monetary_value`` summary that ``ParetoNBDModel`` and ``GammaGammaModel`` consume.
    Results are accessible via the ``table`` attribute (ibis Table) or the ``df`` property
    (materialized pandas DataFrame). See the module docstring for column definitions.

    Constructing ``CLVStats`` runs one aggregate query against the backend to resolve and
    validate the observation window (plus, when ``customer_attributes`` is given, one aggregate to
    check its customer_id is unique and a single distinct-value query enumerating all ``one_hot_col``
    categories); the full per-customer summary stays lazy and is materialized only on first
    ``df`` access.

    Args:
        data (pd.DataFrame | ibis.Table): Transaction data with the ``customer_id``,
            ``transaction_date`` (a date/datetime type), and ``unit_spend`` columns. Undated rows and
            returns-only days (net spend <= 0) are excluded; a customer left with none drops out.
        period (str, optional): The time unit for ``recency`` and ``T``. One of ``"day"``,
            ``"week"``, or ``"month"`` (case-insensitive; short forms like ``"d"``/``"m"`` accepted).
            A month is a fixed 365.25/12-day unit. Defaults to ``"week"``.
        observation_period_end (str | datetime.date | None, optional): The end of the
            observation window, from which each customer's age ``T`` is measured. An
            ISO-8601 string or a ``datetime.date``. Defaults to the latest transaction date
            in ``data``.
        customer_attributes (pd.DataFrame | ibis.Table | None, optional): A per-customer table (one
            row per ``customer_id``) of extra columns to attach to the summary via a left join —
            covariates such as signup channel, region, or a pre-computed ``stores_shopped`` count.
            Build it however you like (e.g. ``SegTransactionStats`` grouped on ``customer_id``). Must
            cover every customer with a non-NULL value for each (non-one-hot) covariate; a missing
            customer or NULL value is rejected on :attr:`df` access. Defaults to ``None``.
        one_hot_col (str | list[str] | None, optional): Column(s) of ``customer_attributes`` to
            one-hot encode into 0/1 dummy columns suitable for ``ParetoNBDModel`` covariates. For
            each column one level is dropped as the reference level (dummies would otherwise be
            collinear with the model intercept) — NULL when the column contains NULLs, otherwise the
            first value in sorted order — and the original column is removed. Warns (``UserWarning``)
            if a column yields more than 32 dummies (a likely high-cardinality mistake) or zero dummies
            (a single-category column that contributes no covariate). Defaults to ``None``.

    Raises:
        TypeError: If ``data`` or ``customer_attributes`` is not a pandas DataFrame or an Ibis Table,
            if ``transaction_date`` is not a date/datetime type, or if ``observation_period_end`` is
            not a valid date-like value.
        ValueError: If required columns are missing, if ``period`` is not ``"day"``, ``"week"``, or
            ``"month"``, if ``observation_period_end`` is before the latest transaction, if
            ``customer_attributes`` lacks the ``customer_id`` column or has a duplicate ``customer_id``,
            if ``one_hot_col`` is given without ``customer_attributes`` or names a column absent from
            it, if an attached column name collides with a reserved summary column, or if ``data``
            contains no transactions.
    """

    #: A "month" is a fixed 365.25/12-day unit (the average Gregorian month), not a calendar month, so
    #: fractional age stays well defined and 12 months is exactly one year.
    _DAYS_PER_MONTH: ClassVar[float] = 365.25 / 12
    VALID_PERIODS: ClassVar[tuple[str, ...]] = ("day", "week", "month")
    _DAYS_PER_PERIOD: ClassVar[dict[str, float]] = {"day": 1, "week": 7, "month": _DAYS_PER_MONTH}
    #: Maps each period to pymc-marketing's ``time_unit`` code; exposed via ``pymc_time_unit``.
    _PYMC_TIME_UNIT: ClassVar[dict[str, str]] = {"day": "D", "week": "W", "month": "M"}
    #: |Pearson r| between frequency and monetary_value above which ``repeat_buyers`` warns: GammaGamma
    #: assumes the two are independent, and a stronger correlation biases its spend estimates (0.10-0.15
    #: is the practical "weak enough" band in BTYD guidance).
    _MONETARY_FREQUENCY_CORR_WARN: ClassVar[float] = 0.15
    #: Buckets the per-customer sampling hash is folded into, setting the resolution of ``frac``
    #: (1e-6). Large enough that rounding never distorts a realistic sample share.
    _SAMPLE_BUCKETS: ClassVar[int] = 1_000_000
    #: The base BTYD output columns (literal pymc-marketing names); anything else on the summary is a
    #: covariate attached via customer_attributes / one_hot_col. See ``covariate_cols``.
    _BASE_SUMMARY_COLUMNS: ClassVar[tuple[str, ...]] = ("customer_id", "frequency", "recency", "T", "monetary_value")

    def __init__(
        self,
        data: pd.DataFrame | ibis.Table,
        *,
        period: str = "week",
        observation_period_end: str | datetime.date | None = None,
        customer_attributes: pd.DataFrame | ibis.Table | None = None,
        one_hot_col: str | list[str] | None = None,
    ) -> None:
        """Initializes and computes the BTYD summary."""
        cols = ColumnHelper()
        data = ensure_ibis_table(data, "data")
        ensure_data_has_columns(data, [cols.customer_id, cols.transaction_date, cols.unit_spend])
        ensure_tznaive_datetime(data, cols.transaction_date)
        period = ensure_period(period, self.VALID_PERIODS, "period")
        attributes = self._prepare_customer_attributes(customer_attributes)
        one_hot_cols = self._resolve_one_hot_cols(attributes, one_hot_col)

        # Drop undated rows: a NULL date forms its own occasion and skews recency/T.
        data = data.filter(data[cols.transaction_date].notnull())  # noqa: PD004 (ibis API, not pandas)
        # Cast the single max scalar to a date (not every row) — same idiom as segmentation/rfm.py.
        latest_raw = data[cols.transaction_date].max().cast("date").execute()
        if pd.isna(latest_raw):  # None or NaT: the table has no transactions to summarize.
            msg = "data contains no transactions; cannot build a CLV summary."
            raise ValueError(msg)
        latest_transaction = _coerce_to_date(latest_raw)
        if observation_period_end is None:
            observation_period_end = latest_transaction
        else:
            observation_period_end = _coerce_to_date(observation_period_end)
            if observation_period_end < latest_transaction:
                msg = (
                    f"observation_period_end ({observation_period_end}) must be on or after the latest "
                    f"transaction date ({latest_transaction}); an earlier end would produce a negative age."
                )
                raise ValueError(msg)

        summary = self._compute_summary(data, period, observation_period_end)
        table = self._attach_customer_attributes(summary, attributes, one_hot_cols)
        # pymc-marketing requires the id column named literally "customer_id" (no remap arg), like the
        # other four. Rename so an overridden customer_id option still yields a model-ready summary.
        if cols.customer_id != "customer_id":
            table = table.rename({"customer_id": cols.customer_id})
        self.period = period
        self.table = table

    @staticmethod
    def _prepare_customer_attributes(customer_attributes: pd.DataFrame | ibis.Table | None) -> ibis.Table | None:
        """Validate and normalize the customer_attributes table to an Ibis Table.

        Args:
            customer_attributes (pd.DataFrame | ibis.Table | None): The caller-supplied per-customer
                attribute table (one row per ``customer_id``), or ``None``.

        Returns:
            ibis.Table | None: The attributes as an Ibis Table, or ``None`` if none were supplied.

        Raises:
            TypeError: If ``customer_attributes`` is neither a pandas DataFrame nor an Ibis Table.
            ValueError: If it lacks the ``customer_id`` column, or has a duplicate ``customer_id``
                (a left join on a non-unique key would fan out, i.e. duplicate, the summary rows).
        """
        if customer_attributes is None:
            return None
        cols = ColumnHelper()
        attributes = ensure_ibis_table(customer_attributes, "customer_attributes")
        if cols.customer_id not in attributes.columns:
            msg = f"customer_attributes must contain the customer_id column '{cols.customer_id}'."
            raise ValueError(msg)
        counts = attributes.aggregate(
            rows=attributes.count(),
            customers=attributes[cols.customer_id].nunique(),
        ).execute()
        if counts["rows"].iloc[0] != counts["customers"].iloc[0]:
            msg = "customer_attributes must have one row per customer_id; found duplicate customer ids."
            raise ValueError(msg)
        return attributes

    @staticmethod
    def _resolve_one_hot_cols(
        attributes: ibis.Table | None,
        one_hot_col: str | list[str] | None,
    ) -> list[str]:
        """Normalize ``one_hot_col`` to a de-duplicated list of columns present in ``attributes``.

        Args:
            attributes (ibis.Table | None): The prepared customer_attributes table, or ``None``.
            one_hot_col (str | list[str] | None): The user-supplied one-hot column(s).

        Returns:
            list[str]: The normalized, de-duplicated one-hot column names (empty if ``one_hot_col`` is None).

        Raises:
            ValueError: If ``one_hot_col`` is given without ``customer_attributes``, or names a column
                absent from ``customer_attributes``.
        """
        if one_hot_col is None:
            return []
        if attributes is None:
            msg = "one_hot_col requires customer_attributes; there are no columns to encode."
            raise ValueError(msg)
        one_hot_cols = ensure_columns(attributes, one_hot_col, "one_hot_col")
        # De-duplicate while preserving order: a repeated column would be encoded twice, and the first
        # pass drops it, so the second would fail to find it.
        return list(dict.fromkeys(one_hot_cols))

    @staticmethod
    def _attach_customer_attributes(
        summary: ibis.Table,
        attributes: ibis.Table | None,
        one_hot_cols: list[str],
    ) -> ibis.Table:
        """Left-join the per-customer attributes onto the summary, one-hot encoding the requested columns.

        Args:
            summary (ibis.Table): The base BTYD summary (one row per customer).
            attributes (ibis.Table | None): The prepared customer_attributes table, or ``None``.
            one_hot_cols (list[str]): Normalized one-hot column names (a subset of the attribute columns).

        Returns:
            ibis.Table: The summary with the attribute and one-hot columns attached.

        Raises:
            ValueError: If an attached column name collides with a reserved BTYD summary column
                (``customer_id``, ``frequency``, ``recency``, ``T``, ``monetary_value``).
        """
        if attributes is None:
            return summary
        cols = ColumnHelper()
        if len(one_hot_cols) > 0:
            attributes = _one_hot_encode(attributes, one_hot_cols)

        attached = [col for col in attributes.columns if col != cols.customer_id]
        # Reserve the literal "customer_id" too (the output id name __init__ renames to); else under an
        # overridden customer_id option an attribute named "customer_id" silently collides.
        collisions = sorted(set(attached) & (set(summary.columns) | {"customer_id"}))
        if len(collisions) > 0:
            msg = f"customer_attributes / one_hot_col columns collide with reserved BTYD summary columns: {collisions}"
            raise ValueError(msg)

        # Reselect the base columns plus the attached ones, dropping the duplicated join key ibis
        # appends (e.g. customer_id_right); selecting by name avoids hardcoding the join suffix.
        return summary.left_join(attributes, cols.customer_id)[[*summary.columns, *attached]]

    @classmethod
    def _compute_summary(
        cls,
        data: ibis.Table,
        period: str,
        observation_period_end: datetime.date,
    ) -> ibis.Table:
        """Computes the BTYD summary table.

        Args:
            data (ibis.Table): The validated transaction data.
            period (str): The validated period unit (``"day"``, ``"week"``, or ``"month"``).
            observation_period_end (datetime.date): The resolved observation window end.

        Returns:
            ibis.Table: One row per customer with the BTYD summary columns.
        """
        cols = ColumnHelper()
        # Collapse to one purchase occasion per customer per calendar day, summing spend
        # within the day (same-day baskets are a single occasion under the BTYD convention).
        day = data[cols.transaction_date].cast("date").name("_day")
        daily = data.group_by([cols.customer_id, day]).aggregate(_day_spend=data[cols.unit_spend].sum())

        # A day is an occasion only if its net spend is positive; returns-only days are not purchases.
        daily = daily.filter(daily._day_spend > 0)

        # A repeat occasion is any purchase day after the customer's first purchase day. The flag is
        # an int rather than a bool because SQL Server has no boolean type to project into a SELECT.
        first_day = daily["_day"].min().over(ibis.window(group_by=daily[cols.customer_id]))
        daily = daily.mutate(_is_repeat=ibis.ifelse(daily["_day"] > first_day, 1, 0).cast("int8"))

        summary = daily.group_by(cols.customer_id).aggregate(
            _first_day=daily["_day"].min(),
            _last_day=daily["_day"].max(),
            _occasions=daily.count(),
            _repeat_spend=daily._day_spend.sum(where=daily._is_repeat == 1),
        )

        days_per_period = cls._DAYS_PER_PERIOD[period]
        observation_end = ibis.literal(observation_period_end)
        frequency = summary._occasions - 1

        # frequency / recency / T / monetary_value are pymc-marketing's required literal names
        # (deliberately not options.py names). The id keeps its configured name here for the join
        # keys; __init__ renames the final column to the literal "customer_id" the models also need.
        return summary.mutate(
            frequency=frequency.cast("int64"),
            # Day-granularity delta (Oracle-safe) divided by days-per-period for a fractional age.
            recency=summary._last_day.delta(summary._first_day, unit="day") / days_per_period,
            T=observation_end.delta(summary._first_day, unit="day") / days_per_period,
            monetary_value=summary._repeat_spend / frequency.nullif(0),
        ).select(
            cols.customer_id,
            "frequency",
            "recency",
            "T",
            "monetary_value",
        )

    def sample(self, *, n: int | None = None, frac: float | None = None, random_state: int = 42) -> Self:
        """Draws a random sample of customers, as a new :class:`CLVStats` over the same summary.

        For fitting a model on a tractable subset and scoring the full population: fit on
        ``clv.sample(n=50_000)``, predict on ``clv.df``. Selection is a deterministic hash of
        ``customer_id`` salted with ``random_state``, so the same arguments draw the same customers on
        every execution (unlike a ``random()`` predicate, which re-rolls per execution and is unseeded
        on several backends), while different ``random_state`` values draw independent samples.

        Sampling the summary is equivalent to sampling customers before aggregating -- a customer's
        summary row depends only on their own transactions -- so the one aggregate serves both the fit
        and the scoring. Row order in the result is not meaningful.

        Args:
            n (int | None, optional): Number of customers to draw. Yields every customer if it exceeds
                the population. Mutually exclusive with ``frac``. Defaults to ``None``.
            frac (float | None, optional): Share of customers to draw, in (0, 1]. A per-customer
                Bernoulli draw, so the count lands near ``frac`` * population rather than on it, and it
                costs a single predicate instead of ``n``'s sort. Mutually exclusive with ``n``.
                Defaults to ``None``.
            random_state (int, optional): Seed for reproducible sampling. Defaults to 42.

        Returns:
            Self: A ``CLVStats`` over the sampled customers, with the same ``period`` and covariates.

        Raises:
            TypeError: If ``n`` or ``random_state`` is not an integer, or ``frac`` is not a number.
            ValueError: If both or neither of ``n`` and ``frac`` are given, if ``n`` is not positive,
                or if ``frac`` is outside (0, 1].
        """
        if (n is None) == (frac is None):
            msg = "sample requires exactly one of n or frac."
            raise ValueError(msg)
        ensure_integer(random_state, "random_state")
        # The id column is the literal "customer_id" here: __init__ renames it. Cast to string so the
        # salt concatenates whatever the configured id type is.
        key = (self.table["customer_id"].cast("string") + ibis.literal(f"::{random_state}")).hash()
        if n is not None:
            ensure_integer(n, "n")
            ensure_positive(n, "n")
            # Ordering by the hash is a pseudo-random permutation of customers, so the first n are a
            # uniform sample; the backend runs it as a top-N. Order on the raw hash, not the bucketed
            # value below, whose ties would make the cut ambiguous past _SAMPLE_BUCKETS customers.
            sampled = self.table.order_by(key).limit(n)
        else:
            ensure_positive(frac, "frac")
            ensure_unit_interval(frac, "frac")
            # abs() after the modulo, not before: abs(-2**63) overflows int64.
            bucket = (key % self._SAMPLE_BUCKETS).abs()
            sampled = self.table.filter(bucket < int(frac * self._SAMPLE_BUCKETS))
        # Bypass __init__: the summary is already built and validated, and the sample shares its state
        # (period, table) wholesale. Constructing through __init__ would re-run the observation-window
        # and attribute queries against a table that no longer holds transactions.
        sample = object.__new__(type(self))
        sample.period = self.period
        sample.table = sampled
        return sample

    @functools.cached_property
    def df(self) -> pd.DataFrame:
        """Returns the materialized BTYD summary as a pandas DataFrame.

        The ``customer_id`` is a column (not the index) so the frame can be passed straight
        to ``ParetoNBDModel`` / ``GammaGammaModel``, which expect it as a column.

        Returns:
            pd.DataFrame: One row per customer with columns ``customer_id``, ``frequency``,
                ``recency``, ``T``, and ``monetary_value``, followed by any ``customer_attributes``
                and one-hot columns requested at construction.

        Raises:
            ValueError: If a covariate column is NULL for any customer (a customer missing from
                ``customer_attributes``, or a NULL attribute value). NULL covariates silently break
                ``ParetoNBDModel``'s fit, which does not validate for them.
        """
        result = self.table.execute()
        # Checking one-hot dummies too is safe: a NULL *source* value became the reference level (0), so a
        # dummy reads NULL only for a customer missing from the join -- exactly the case worth rejecting.
        null_covariates = [col for col in self.covariate_cols if result[col].isna().any()]
        if len(null_covariates) > 0:
            msg = (
                f"customer_attributes leaves NULL in covariate column(s) {null_covariates}: a customer is "
                "missing from customer_attributes, or has a NULL attribute value. NULL covariates make "
                "ParetoNBDModel's fit fail. Provide non-NULL covariates for every customer, or drop those "
                "customers first."
            )
            raise ValueError(msg)
        return result

    @property
    def repeat_buyers(self) -> pd.DataFrame:
        """The GammaGamma-ready subset of :attr:`df` — the repeat buyers (``frequency > 0``, index reset).

        One-time buyers (``NaN`` monetary_value) are excluded; ``GammaGammaModel`` cannot fit them. No
        spend filter is needed because ``monetary_value`` is always positive here. Warns (``UserWarning``)
        if ``|Pearson r|`` between ``frequency`` and ``monetary_value`` exceeds
        ``_MONETARY_FREQUENCY_CORR_WARN``, breaking GammaGamma's independence assumption.

        Returns:
            pd.DataFrame: The ``frequency > 0`` rows of :attr:`df`, index reset.
        """
        summary = self.df
        gg_ready = summary[summary["frequency"] > 0].reset_index(drop=True)
        if len(gg_ready) == 0:  # no repeat buyers: nothing to correlate
            return gg_ready
        # Correlation is only defined when both columns vary; a constant column would divide by a zero
        # standard deviation (NaN, plus a numpy warning), so skip the check in that degenerate case.
        frequency, monetary_value = gg_ready["frequency"], gg_ready["monetary_value"]
        both_vary = frequency.min() != frequency.max() and monetary_value.min() != monetary_value.max()
        if both_vary:
            corr = frequency.corr(monetary_value)
            if abs(corr) > self._MONETARY_FREQUENCY_CORR_WARN:
                warnings.warn(
                    f"frequency and monetary_value are correlated (Pearson r={corr:.2f}); GammaGamma assumes "
                    "they are independent, so its spend estimates may be biased.",
                    stacklevel=2,
                )
        return gg_ready

    @property
    def covariate_cols(self) -> list[str]:
        """The covariate columns attached to the summary (customer_attributes and one-hot dummies).

        Every column beyond the base BTYD summary, ready to pass as ``purchase_covariate_cols`` /
        ``dropout_covariate_cols`` to ``ParetoNBDModel``.

        Returns:
            list[str]: The covariate column names in summary order (empty if none were requested).
        """
        return [col for col in self.table.columns if col not in self._BASE_SUMMARY_COLUMNS]

    @property
    def pymc_time_unit(self) -> str:
        """The pymc-marketing ``time_unit`` matching this summary's ``period`` (``"D"``/``"W"``/``"M"``).

        Pass as the ``time_unit`` of ``GammaGammaModel.expected_customer_lifetime_value``, whose ``future_t``
        is in months and defaults to ``"D"``, silently wrong for a weekly summary (horizon off ~7x) or a
        monthly one (~30x).

        Returns:
            str: ``"D"`` for ``period="day"``, ``"W"`` for ``"week"``, ``"M"`` for ``"month"``.
        """
        return self._PYMC_TIME_UNIT[self.period]
