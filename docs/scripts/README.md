# docs/scripts

Maintenance scripts for regenerating committed documentation assets. These are
intentionally **not** referenced from the public docs — they are tools for
maintainers, not library users.

## regenerate_analysis_modules_svgs.py

Regenerates every SVG used on `docs/analysis_modules.md`. Run after touching
plot styling, chart copy, or the synthetic-data shapes the curated charts depend
on:

```sh
uv run python docs/scripts/regenerate_analysis_modules_svgs.py
```

Outputs land in `docs/assets/images/analysis_modules/`. Commit the resulting SVGs
alongside whatever change motivated the regeneration.

## regenerate_homepage_charts.py

Regenerates the showcase charts on the homepage (`docs/overrides/home.html`):

```sh
uv run python docs/scripts/regenerate_homepage_charts.py
```

Outputs land in `docs/assets/images/home/chart-*.svg`. The chart alt text in `home.html` quotes the chart data, so
update it if the data changes.
