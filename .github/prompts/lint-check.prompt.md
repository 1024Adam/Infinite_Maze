---
description: "Run all code quality checks: black formatting, flake8 linting, and mypy strict type checking across the infinite_maze source."
---
Run the following three code quality checks in order against the `infinite_maze/` package. Report results for each step before moving to the next, and stop if any step produces errors.

## Step 1 — Formatting (black)

```
black --check infinite_maze/
```

If black reports files that would be reformatted, run:

```
black infinite_maze/
```

and list the files that were changed.

## Step 2 — Linting (flake8)

```
flake8 infinite_maze/
```

List every violation grouped by file. The project config ignores E203 and W503; do not suppress any other codes.

## Step 3 — Type checking (mypy)

```
mypy infinite_maze/
```

The project uses strict settings (`disallow_untyped_defs`, `disallow_incomplete_defs`, `warn_return_any`, etc. — all defined in `pyproject.toml`). List every error with file, line, and error code.

## Summary

After all three steps, produce a table:

| Check | Status | Issues |
|---|---|---|
| black | ✅ / ❌ | n files reformatted / already formatted |
| flake8 | ✅ / ❌ | n violations |
| mypy | ✅ / ❌ | n errors |

If there are any failures, ask whether to fix them before finishing.
