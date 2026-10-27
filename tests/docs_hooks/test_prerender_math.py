"""Tests for the prerender_math docs build hook."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from prerender_math import on_page_content

METRICS_PAGE = SimpleNamespace(file=SimpleNamespace(src_uri="experimental/metrics.md"))
# arithmatex's generic output for $$ ... $$ on the metrics page, HTML-escaped as it is in the page.
ACV_BLOCK = (
    '<div class="arithmatex">\\[\n\\text{ACV} = \\frac{\\sum \\text{unit_spend}}{\\text{acv_scale_factor}}\n\\]</div>'
)


class TestPrerenderMath:
    """Tests for rendering arithmatex LaTeX to MathML at build time."""

    def test_block_math_becomes_mathml_inside_the_scroll_wrapper(self):
        """Block math keeps arithmatex's div, whose CSS scrolls equations wider than a phone screen."""
        output = on_page_content(ACV_BLOCK, METRICS_PAGE, None, None)

        assert output.startswith('<div class="arithmatex"><math ')
        assert output.endswith("</math></div>")
        assert 'display="block"' in output
        assert "<mtext>unit_spend</mtext>" in output
        assert "<mfrac>" in output
        assert "\\frac" not in output

    def test_inline_math_becomes_inline_mathml_with_html_unescaped(self):
        """Inline math loses its span, and an escaped `<` in the LaTeX becomes a MathML less-than operator."""
        html = '<p>Stores where <span class="arithmatex">\\(\\text{spend} &lt; 1000\\)</span> are small.</p>'

        output = on_page_content(html, METRICS_PAGE, None, None)

        assert output.startswith("<p>Stores where <math ")
        assert output.endswith("</math> are small.</p>")
        assert 'display="inline"' in output
        assert "<mo>&#x0003C;</mo>" in output
        assert "arithmatex" not in output

    def test_less_than_inside_text_stays_escaped(self):
        """latex2mathml copies text-mode LaTeX verbatim, so an unescaped `<` there would open a stray HTML tag."""
        html = '<span class="arithmatex">\\(\\text{spend&lt;budget}\\)</span>'

        output = on_page_content(html, METRICS_PAGE, None, None)

        assert "<mtext>spend&lt;budget</mtext>" in output

    def test_math_is_excluded_from_search(self):
        """MathML text indexes as run-together tokens, so every equation opts out of Material's search."""
        output = on_page_content(ACV_BLOCK, METRICS_PAGE, None, None)

        assert output.count("<math data-search-exclude ") == 1

    @pytest.mark.parametrize(
        ("latex", "should_warn"),
        [
            ("\\text{ACV} = \\frac{\\sum \\text{unit_spend}}{\\text{acv_scale_factor}}", False),
            ("\\text{units} = \\begin{cases} 0 & \\text{returned} \\\\ 1 & \\text{sold} \\end{cases}", False),
            ("\\text{margin} = \\unknownmacro{spend}", True),
            ("\\begin{aligned} \\text{revenue} &= \\text{units} \\times \\text{price} \\end{aligned}", True),
            ("\\text{spend} \\\\ \\text{visits}", True),
            ("\\%\\text{Stores} = \\text{50\\% of stores}", True),
        ],
    )
    def test_warns_only_when_latex2mathml_would_render_math_wrongly(self, caplog, latex, should_warn):
        """Unsupported commands, aligned columns, line breaks and escapes inside text render wrongly, so they warn."""
        on_page_content(f'<div class="arithmatex">\\[{latex}\\]</div>', METRICS_PAGE, None, None)

        warnings = [record.getMessage() for record in caplog.records if record.levelname == "WARNING"]
        assert len(warnings) == (1 if should_warn else 0)
        assert all("experimental/metrics.md" in warning for warning in warnings)

    def test_unparseable_math_raises_naming_the_page(self):
        """A LaTeX syntax error stops the build with the page and equation instead of a bare traceback."""
        with pytest.raises(ValueError, match=r"experimental/metrics\.md: .*\\\\frac\{spend\}\{"):
            on_page_content('<div class="arithmatex">\\[\\frac{spend}{\\]</div>', METRICS_PAGE, None, None)

    def test_warns_when_arithmatex_output_isnt_recognised(self, caplog):
        """A wrapper with an extra attribute escapes the patterns; without MathJax it would show raw LaTeX."""
        on_page_content('<div id="acv" class="arithmatex">\\[x\\]</div>', METRICS_PAGE, None, None)

        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "unconverted" in caplog.records[0].getMessage()
