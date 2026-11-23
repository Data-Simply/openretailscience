# Examples

## Audience and shape

- Write for an analyst, category manager or marketer who has a question and may never have used ORS. Each page stands
  alone: open with their problem as a concrete situation and go straight to solving it. Don't introduce or describe
  ORS; show `pip install` and the input data.
- One page answers one retail question. Put it, in the retail terms people search with ("cross-shopping", "which
  customers buy both"), in the front-matter `title` and `description`, the H1 and the opening brief. Describe the
  searcher's problem, not what the function does.
- Linear path: load data, one ORS call, read named output columns, state the recommendation in a
  `!!! success "Recommendation"` box. No data mutation, no redundant arguments, no display-only steps.
- Read the answer from the class's own output by column name. If it needs a lot of pandas, the question doesn't suit
  the class. Rename opaque output columns once (e.g. `group_1` to `machines`) so the code reads in business terms.
- Explain each output in terms of the question ("404 customers bought only a machine") before any caveat. Put each
  gotcha beside the line it affects, phrased as what goes wrong.
- State every number the conclusion rests on in prose or a printed table, and give charts alt text that states the
  finding. Call a difference within noise a tie.
- Consulting-report style. The opening reads as an analyst brief: the business situation, the decision it informs,
  the questions to answer as a list, and the method. Findings sections get action titles that state the finding, each
  paragraph leads with its takeaway, and the recommendation box answers the brief. Quantify claims.
- Professional register: no contractions, no "say you..." set-ups, no casual phrasing. A relatable scene is a
  recognisable business situation, not a chatty voice.
- No colons except to introduce a list: not in titles, headings, prose, alt text or the sentence before a code block.
- Show how little code the answer takes; don't claim it. No "easy", "powerful" or "seamless".
- End with "Adapting the analysis" (other columns, basket vs customer grain) and links to related pages.
- No retail theory or who-does-this sections; this is ORS documentation.

## Build mechanics

- Pages are Markdown with `markdown-exec` blocks, executed on every build; a failing block fails `mkdocs build
  --strict`. Prose numbers are typed by hand, so re-check them after changing the data or the code.
- Each page starts with a hidden setup block defining `show_table` and `show_figure`. Visible code ends with the
  expression a notebook would display; a line marked `# markdown-exec: hide` prints it, through those helpers for
  tables and charts. Round only in the hidden line, never in the visible code.
- `show_figure` prints a PNG data URI; `docs/hooks/notebook_assets.py` moves it into a linked file.
- Blocks run with `workdir="docs/examples/data"` so the visible code reads data by bare filename. Never write files
  from a block: they land in that directory.
- Data comes from the seeded `docs/scripts/generate_example_data.py`, written small (sorted, delta-encoded IDs). Synthetic
  data is fine; say so on the page.
- Match the call pattern of the skill's `example_*.py` scripts and the class docstring.
