# Test Writing Guidelines

Every test imports and calls OpenRetailScience code, and its assertions check what its name claims and can fail.

## Write

- Group related tests in a class per module or class under test. Imports go at the top of the module, never inside a
  test.
- Compare frames with `assert_frame_equal` against an expected frame rather than asserting cell by cell. Don't pass
  `check_dtype=False`; build the expected frame with the right dtypes so dtype regressions fail.
- Parametrize instead of adding a test whose setup, assertion or expected exception matches an existing one in the
  class with different inputs, even when it targets a different parameter.
- Cover threshold values and limits.
- Use realistic retail data (customer, store and transaction values), not "A"/"B"/"test".
- Name columns with `ColumnHelper` (`cols.store_id`), not string literals, so tests survive changed option defaults.
- Compare against package constants (colors, formats), not hardcoded copies.
- Share setup through pytest fixtures.
- Mock only external dependencies (databases, APIs, file and network I/O), never package logic.

## Don't

- Test pandas, the standard library or Python built-ins without exercising package code.
- Assert only `is not None`, `isinstance`, `hasattr` or `callable`, or write an assertion that can't fail.
- Assert only that something changed (`filtered.shape != unfiltered.shape`) or exists (`len(x) > 0`) when the result
  is knowable. Assert `len(filtered) == 5` and `(filtered["price"] > 100).all()`, and pin counts to their source
  (`len(bar_labels) == len(bars)`) so a regression that drops all but one item fails.
- Assert on internal state (`obj._cache == {...}`). Private method outputs are fine (`obj._calculate(5) == 10`).
- Test that a constant, class name or config default equals its definition. Test that package code uses the
  configured value.
