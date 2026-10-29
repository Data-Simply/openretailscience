# Docs

- In LaTeX math blocks (`$$ ... $$`), do not escape underscores inside `\text{}` commands. Use `\text{unit_spend}`,
  not `\text{unit\_spend}` — the backslash-escaped form renders a visible backslash in the output.
