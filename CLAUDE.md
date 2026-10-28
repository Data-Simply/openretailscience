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

## Agent Skill Maintenance

The package ships an AI-agent-facing skill at `openretailscience/.agents/skills/using-openretailscience/`
(`SKILL.md`, `references/*.md`, `scripts/example_*.py`). These files are read directly by AI agents as API
documentation, not by humans running a tutorial. Whenever a change touches public API surface (a renamed/added/
removed parameter, a changed default, a new recommended utility function, a deprecated pattern), check whether
`SKILL.md`, the `references/` docs, or the affected `example_*.py` scripts need updating to match. A stale example
silently teaches an agent a broken or outdated pattern. `tests/test_skills.py::TestExampleScripts` only catches
outright breakage (a script that raises); it does not catch an example that still runs but no longer reflects
current best practice.

The `Examples:` section of each class and function exported by `analysis`, `segmentation` and `utils`, and of each
plot module's `plot()`, is the other place agents read the API: an agent without the skill loaded learns the call
from `help()` or `__doc__`. When public API changes, update the affected docstring example along with the skill,
using the same call pattern as the matching `example_*.py` script. Keep it to about five lines: the call and what a
user does next with the result, with no display-only steps (`.iloc`, column listings, rounding to fit). Show at most
a few rows of a printed table and end it with a `...` line (the test runs with ELLIPSIS); `help()` on `analysis`,
`segmentation` or `utils` renders every one. `tests/test_docstring_examples.py` fails if one of these has no example,
and runs each one with `pd`, `ibis` and a sample `transactions` frame in scope.

## Build & Test Commands

- Install pre-commit hooks: `uv run pre-commit install` — **run this before your first commit.** The hooks
  (ruff, conventional-commit, pytest, etc.) gate every commit and mirror CI; setting them up first catches
  failures locally instead of in the pipeline.
- Run Python scripts: `uv run python script.py`
- When running a specific test, use `pytest -k "test_name"` if you don't already know the full node path
  (class name, parameterization, etc.). Don't guess full paths.

**Note**: Always use `uv run` to execute Python commands to ensure the correct virtual environment and
dependencies are used.

## Code Style Guidelines

- Follow Google Python Style Guide
- Use Google-style docstrings with Args, Returns, and Raises sections
- Comments, docstrings, and docs must be concise and load-bearing: say only what the reader cannot get from the
  code. Cut restatements of the code, change history (it belongs in the commit message), and self-justification.
  A public API's `Examples:` section is load-bearing; never remove it because the skill has the same example.
- Private methods/functions should be prefixed with underscore (_)
- Error handling: validate inputs at public API boundaries with specific
  exceptions and descriptive messages. Private functions should not
  re-validate inputs, but must include an explicit `raise` for unhandled
  cases in branching logic (e.g., if/elif chains dispatching on a type
  parameter) to prevent implicit `None` returns. Every code path must
  either return a value or raise an exception.
- When checking for None values use `if my_var is not None` rather than `if my_var:`
- When checking for empty lists values use `if len(my_list) > 0` rather than `if my_list:`
- PR titles must also be conventional commits, and every PR needs exactly one of the `major-change` (a release-notes
  headline) or `minor-change` labels
- Do not stage files or commit unless asked to
- Docstrings should specify argument and return types
- Type annotations should use Python 3.11-compatible formats (e.g., use `from __future__ import annotations` and
  import newer typing constructs like `Self` inside `if TYPE_CHECKING:` blocks)
- Assume datasets processed via Ibis could be in the 1B–10B row range. Avoid redundant operations (e.g., a `.nunique()`
  on already-deduplicated data) and prefer the cheapest correct operation for a given context.
- Use vectorized operations for numpy arrays and pandas DataFrames/Series wherever possible instead of loops or
  .apply() to keep code clean and performant. Avoid converting Series to lists for iteration or using .iterrows() -
  use vectorized operations directly on the DataFrame/Series
- Avoid using `Any` or `object` types in annotations; determine and use the correct specific type whenever possible
- Do not use `noqa: TC003`; move type-only imports into `if TYPE_CHECKING:` blocks instead.
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

The rest of the test-writing rules live in tests/CLAUDE.md, which loads when you work on test files.
