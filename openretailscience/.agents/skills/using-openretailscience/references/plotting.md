# Plotting reference

Each chart is a module-level `plot(...)` function under `openretailscience.plots`
that returns a matplotlib `Axes`. Import the module and call `.plot`:

```python
from openretailscience.plots import bar, line, area, scatter, histogram
from openretailscience.plots import waterfall, venn, heatmap, cohort
from openretailscience.plots import time, period_on_period, broken_timeline, price, index
```

Most `plot()` functions share these chrome keywords: `title`, `eyebrow`,
`subtitle`, `source_text`, `x_label`, `y_label`, `ax`, `legend_title`,
`move_legend_outside`. Pass `ax=` to draw onto an existing axes. Only the
distinctive arguments are listed below — see each function's docstring for the rest.

## Charts

- `bar.plot(df, value_col=None, x_col=None, orientation="vertical", sort_order=None, data_label_format=None, ...)` —
  bar / grouped bar. `data_label_format` ∈ absolute / percentage_by_bar_group /
  percentage_by_series.
- `line.plot(df, value_col=None, x_col=None, group_col=None, legend_style=None, highlight=None, ...)` —
  line chart; `legend_style` ∈ box / end_of_line.
- `area.plot(df, value_col, x_col=None, group_col=None, legend_style=None, ...)` — stacked area.
- `scatter.plot(df, value_col, x_col=None, group_col=None, size_col=None, label_col=None, ...)` —
  scatter / bubble.
- `histogram.plot(df, value_col=None, group_col=None, clip_range=None, ...)` — histogram.
- `waterfall.plot(amounts, labels, data_label_format=None, display_net_bar=False, ...)` —
  waterfall; also exposes `waterfall.format_data_labels(...)`.
- `venn.plot(df, labels, vary_size=False, subset_label_formatter=None, ...)` — 2/3-set
  Venn (df needs `groups` + `percent`). Usually reached via `CrossShop.plot`.
- `heatmap.plot(df, cbar_label, colormap_style="discrete", ...)` — generic heatmap.
- `cohort.plot(df, cbar_label, percentage=True, ...)` — cohort heatmap (wraps
  `heatmap.plot`); feed it `CohortAnalysis(...).df`.
- `time.plot(df, value_col, period="D", agg_func="sum", group_col=None, ...)` — resampled
  time series.
- `period_on_period.plot(df, x_col, value_col, periods, ...)` — overlay periods aligned to
  a reference start (`periods` from `utils.date.find_overlapping_periods`).
- `broken_timeline.plot(df, category_col, value_col, period="D", threshold_value=None, ...)` —
  data-availability / gaps timeline.
- `price.plot(df, value_col, group_col, bins, ...)` — price-band bubble distribution.
- `index.plot(df, value_col, group_col, index_col, value_to_index, top_n=None, bottom_n=None, ...)` —
  index plot (baseline 100). Same-module helpers: `index.get_indexes(...)`,
  `index.filter_by_groups(...)`, `index.filter_by_value_thresholds(...)`,
  `index.filter_top_bottom_n(...)`.

## Trendlines

Add a fitted trend line to any line / scatter / bar axes:

```python
from openretailscience.plots.styles.trend import add_trend_line

ax = scatter.plot(df, value_col="spend", x_col="visits")
add_trend_line(ax, trend_type="linear", show_equation=True, show_r2=True)
```

- `add_trend_line(ax, trend_type="linear", color="red", show_equation=True, show_r2=True, ...)` —
  `trend_type` ∈ linear / power / logarithmic / exponential; annotates the equation and R².

## Styling, colors & fonts (`openretailscience.plots.styles`)

```python
from openretailscience.plots.styles.colors import get_plot_colors, get_named_color, get_base_cmap, get_sequential_cmap
from openretailscience.plots.styles.graph_utils import format_shorthand, set_axis_shorthand, set_axis_percent
from openretailscience.plots.styles.styling_helpers import standard_graph_styles, apply_chart_chrome
from openretailscience.plots.styles.font_utils import get_font_properties
```

- Colors: `get_plot_colors(num_series)` (palette for N series),
  `get_named_color("positive"|"negative"|"neutral"|"difference"|"context"|"primary")`,
  `get_base_cmap()`, `get_sequential_cmap()`, plus `get_color_list`, `get_listed_cmap`,
  `get_linear_cmap`.
- Axis/number: `format_shorthand(...)` (1.2K/3.4M), `set_axis_shorthand(...)`,
  `set_axis_percent(...)`, `apply_hatches(ax, n)`, `draw_end_of_line_labels(ax)`.
- Chrome: `apply_chart_chrome(...)`, `apply_base_styling(ax, ...)`, `apply_legend(...)`,
  `standard_graph_styles(...)`.
- Fonts: `get_font_properties(font_name)` returns matplotlib `FontProperties` for the
  bundled Poppins fonts.

All colors and fonts are driven by `plot.color.*` / `plot.font.*` / `plot.style.*`
options — override them (see `references/configuration.md`) rather than editing
matplotlib directly, so every chart stays visually consistent. The raw Tailwind
palette is available as `constants.COLORS[hue][shade]`.
