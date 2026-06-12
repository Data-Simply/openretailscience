# OpenRetailScience Development Guide

## Architecture & Design Principles

### Simplicity Criterion

All else being equal, simpler is better. Weigh every change as **improvement magnitude vs. complexity cost**: a
marginal gain that drags in a new abstraction, an extra parameter, or another layer of indirection is rarely worth
it. **Removing code while preserving or improving behavior is one of the best outcomes available** — a net-negative
diff is a feature, not a sign of cut corners.

Apply this lens to every change:

- **Prefer deletion over addition** when both achieve the goal. If a requirement can be met by removing a code
  path rather than guarding it, do that.
- **Reject sunk-cost momentum.** If a refactor is getting uglier as you go — more special cases, more wrappers,
  more comments needed to justify each step — stop and reconsider. The right answer is often a smaller change,
  or a different shape of change entirely.
- **No premature abstraction.** A helper for one caller has to be read every time too. Three similar lines beat
  a wrapper introduced "just in case" a second use site appears.
- **Default to the boring shape.** Flat functions beat dispatch tables for two cases; direct checks beat
  extension points; explicit code beats clever code.

When in doubt about whether a piece of complexity is earning its keep, leave it out. Adding it back when a real
second use case demands it is cheap; removing an abstraction once code depends on its shape is not.

## Build & Test Commands

- Install dependencies: `uv sync`
- Install pre-commit hooks: `uv run pre-commit install` — **run this before your first commit.** The hooks
  (ruff, conventional-commit, pytest, etc.) gate every commit and mirror CI; setting them up first catches
  failures locally instead of in the pipeline.
- Run all tests: `uv run pytest`
- Run specific test: `uv run pytest tests/test_file.py::test_function_name`
- Run tests with pattern: `uv run pytest -k "pattern"`
- Run tests with coverage: `uv run pytest --cov=openretailscience`
- Lint code: `uv run ruff check .`
- Format code: `uv run ruff format .`
- Run Python scripts: `uv run python script.py`
- Run notebook: `uv run jupyter nbconvert --to notebook --execute my_notebook.ipynb`
- When running a specific test, use `pytest -k "test_name"` if you don't already know the full node path
  (class name, parameterization, etc.). Don't guess full paths.

**Note**: Always use `uv run` to execute Python commands to ensure the correct virtual environment and
dependencies are used.

## Code Style Guidelines

- Follow Google Python Style Guide
- Use Google-style docstrings with Args, Returns, and Raises sections
- Import order: standard library → third-party → internal packages
- Use explicit type annotations for all functions and parameters
- Class names use CamelCase; functions, methods, variables use snake_case
- Private methods/functions should be prefixed with underscore (_)
- Max line length: 120 characters
- Error handling: validate inputs at public API boundaries with specific
  exceptions and descriptive messages. Private functions should not
  re-validate inputs, but must include an explicit `raise` for unhandled
  cases in branching logic (e.g., if/elif chains dispatching on a type
  parameter) to prevent implicit `None` returns. Every code path must
  either return a value or raise an exception.
- File naming: Module names should be lowercase with underscores
- Using double quotes when quoting strings
- When checking for None values use `if my_var is not None` rather than `if my_var:`
- When checking for empty lists values use `if len(my_list) > 0` rather than `if my_list:`
- Always create "conventional commit" commit messages
- Do not stage files or commit unless asked to
- Docstrings should specify argument and return types
- Type annotations should use Python 3.10-compatible formats (e.g., use `from __future__ import annotations` and
  import newer typing constructs like `Self` inside `if TYPE_CHECKING:` blocks)
- Remove unnecessary trailing whitespace
- Assume datasets processed via Ibis could be in the 1B–10B row range. Avoid redundant operations (e.g., a `.nunique()`
  on already-deduplicated data) and prefer the cheapest correct operation for a given context.
- Use vectorized operations for numpy arrays and pandas DataFrames/Series wherever possible instead of loops or
  .apply() to keep code clean and performant. Avoid converting Series to lists for iteration or using .iterrows() -
  use vectorized operations directly on the DataFrame/Series
- Extract magic numbers as named constants with descriptive names (e.g., `GREEN_THRESHOLD = 1.0` instead of hardcoded
  `1.0` in conditionals)
- Add input validation for public functions and class methods: validate types, ranges, and constraints early with clear
  error messages
- Use ternary operators for simple conditional assignments (e.g., `status = "active" if count > 0 else "inactive"`
  instead of multi-line if/else blocks)
- Avoid using `Any` or `object` types in annotations; determine and use the correct specific type whenever possible
- Use `TYPE_CHECKING` blocks for type-only imports to avoid runtime import overhead. Do not use `noqa: TC003` to
  suppress these warnings; instead, properly move type-only imports into `if TYPE_CHECKING:` blocks
- In LaTeX math blocks (`$$ ... $$`), do not escape underscores inside `\text{}` commands. Use `\text{unit_spend}`,
  not `\text{unit\_spend}` — the backslash-escaped form renders a visible backslash in the output.
- Shared validation helpers belong in `openretailscience/core/validation.py`. Do not place validation helpers in
  metric-specific modules.
- **No nested function definitions.** A `def` (or `async def`) must not appear inside another function's body.
  Lift inline helpers to module scope. If the helper genuinely needs to bind outer-scope state, take that state as
  an explicit parameter rather than relying on a closure. Lambdas are exempt; the rule is specifically about
  `def` / `async def` inside another `def`. (Ruff has no built-in rule for this; enforce by review.)

## Test Writing Guidelines

### Red/Green TDD

**Strongly prefer red/green TDD wherever possible.** When implementing new functionality or fixing bugs:

1. **Red** — Write a failing test first that captures the expected behavior
2. **Green** — Write the minimum code to make the test pass
3. **Refactor** — Clean up while keeping tests green

This applies to new features, bug fixes, and significant refactoring. Skip TDD only for trivial changes (typo fixes,
comment updates, etc.) where writing a test first would be purely mechanical.

**Confirming red/green status requires running the tests.** Pyright diagnostics (type errors, missing parameters, etc.)
are not a substitute for actually executing `uv run pytest` on the relevant test file. A test is only "red" when it
fails at runtime, and only "green" when it passes at runtime.

### Core Principles

- **Test package functionality, not libraries**: Every test must import and exercise OpenRetailScience code. Do not test
  pandas, Python built-ins, or third-party libraries in isolation.
- **Verify behavior, not execution**: Tests must have meaningful assertions that validate the actual behavior claimed by
  the test name. Checking return types or "not None" is insufficient.
- **Test outputs, not implementation**: Focus on public API behavior and results. Avoid testing internal state, private
  attributes, or "how" code works internally. Testing private method outputs/behavior is acceptable (e.g.,
  `obj._calculate(5) == 10`), but testing internal state is not (e.g., `obj._cache == {...}`).
- **Use realistic retail data**: Test data should reflect real retail scenarios (customer_id, store_id, transaction
  amounts) rather than generic placeholders ("A", "B", "test", "data").

### Required Practices

- Prefer test classes for organizing related tests (group tests by module or class being tested)
- Always put imports at the top of the module (never import inside test methods)
- Where possible prefer pandas `assert_frame_equal` to asserting individual values of the dataframe
    - This may require creating an expected dataframe to compare against
    - Avoid `check_dtype=False`; match the expected dtype instead (it lazily hides real dtype regressions)
- Use `pytest.mark.parametrize` when testing multiple input variations of the same behavior
- Test files should follow pattern: `test_*.py` with test functions `test_*`
- Test names must accurately describe what is being tested
- Each test should have at least one meaningful assertion that validates the claimed behavior
- Include boundary/edge case tests for threshold values, limits, and special cases
- When testing against expected values (colors, formats, etc.), reference package constants rather than hardcoding
  values in tests
- Use `ColumnHelper` for column names in test DataFrames (e.g., `cols.store_id`, `cols.customer_id`) instead of
  hardcoding string literals like `"store_id"`. This keeps tests decoupled from the current option defaults.
- Use pytest fixtures for shared test data setup to improve readability and reduce duplication

### Anti-Patterns to Avoid

- **Don't test without package imports**: If no OpenRetailScience code is imported, the test is invalid
- **Don't test Python/library basics**: Avoid tests that verify fundamental Python or standard library behavior (set
  operations, dict.get(), list methods, string operations) without exercising OpenRetailScience functionality. These tests
  belong in Python's own test suite.
- **Don't write assertion-free tests**: Every test must have assertions that can actually fail. Avoid:
    - Tests with no assertions at all
    - Only `assert obj is not None` after object creation (always true)
    - Only `assert hasattr()` or `callable()` checks without behavior verification
    - Only `assert isinstance()` type checks without validating actual behavior
    - Assertions that can never fail (e.g., `assert x >= 0 or x < 0`)
    - Vague comparison assertions that only check "something changed" (e.g., `assert filtered_df.shape !=
      unfiltered_df.shape`). Better: verify the actual filtering behavior (e.g., `assert len(filtered_df) == 5` and
      `assert (filtered_df['price'] > 100).all()`). This includes count-existence checks like `assert len(x) > 0`
      or `>= 1` when the expected count is knowable (e.g., one label per bar, one legend entry per category). Pin
      the count to its source: `assert len(bar_labels) == len(bars)`. A regression that drops all but one item
      should fail the test.
- **Don't duplicate tests**: If you are about to write a new test method whose body follows the same pattern as an
  existing method in the class (same assertion, same exception, same setup) but with different input values, stop and
  add a `pytest.mark.parametrize` entry to the existing method instead. This applies even when the methods test
  "different parameters" — if the structure is identical, they belong in one parametrized test.
- **Don't over-mock**: Mock external dependencies only (databases, APIs, file I/O, network calls). Never mock internal
  OpenRetailScience logic or calculations. Test real package behavior.
- **Don't write trivial tests**: Avoid testing constants equal their definitions, class names, or tautological
  assertions (True == True)
- **Don't test implementation details**: Avoid asserting internal state values or data structures. Testing private
  method behavior/outputs is OK; testing how data is stored internally is not.
- **Don't test configuration defaults**: Avoid tests that only verify default values match their definitions (e.g.,
  `assert get_option("foo") == "bar"` when you just set `"foo": "bar"` in the defaults dictionary). These are
  tautological and can only fail from typos noticed immediately. Instead, test that configured values are actually
  *used* by package functionality through integration tests.
- **Don't use generic test data**: Use domain-specific examples (retail customer data, not "A", "B", "C")
- **Don't mismatch test names and assertions**: If test claims to verify sorting, actually assert sort order. If it
  claims validation, verify validation logic works.

### Before Committing Tests

Before committing test code, verify each test meets these criteria:

1. **Imports OpenRetailScience code**: At least one import from `openretailscience.*`
2. **Calls package functionality**: Test actually invokes package functions/classes/methods
3. **Has meaningful assertions**: Assertions verify actual behavior, not just types or existence
4. **Test name matches behavior**: Assertions validate what the test name claims to test
5. **No substantial duplication**: If two or more test methods share the same structure (same assertion pattern, same
   exception, same setup) and differ only in one or two input values, they must be combined with
   `pytest.mark.parametrize`. Look for this across the entire test class, not just adjacent methods.
6. **Uses realistic data**: Test data reflects retail domain (customer IDs, prices, stores, etc.)
7. **Minimal mocking**: Only external dependencies are mocked, not internal package logic
