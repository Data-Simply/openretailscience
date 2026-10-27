"""Render arithmatex's LaTeX to MathML at build time, so math pages need no MathJax script or web fonts."""

from __future__ import annotations

import html
import logging
import re
from typing import TYPE_CHECKING

from latex2mathml.converter import convert

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

# pymdownx.arithmatex with `generic: true` wraps block math in \[...\] and inline math in \(...\), HTML-escaped.
_BLOCK_MATH = re.compile(r'<div class="arithmatex">\\\[(.*?)\\\]</div>', re.DOTALL)
_INLINE_MATH = re.compile(r'<span class="arithmatex">\\\((.*?)\\\)</span>', re.DOTALL)
# latex2mathml emits these instead of failing: an unsupported command or an `aligned` column `&` as a literal
# identifier, an escape inside \text{} (`\%`, `\_`) copied with its backslash, and a top-level `\\` as a line break
# that browsers' MathML doesn't render.
_UNRENDERED = re.compile(r'<mi>\\|<mi>&</mi>|<mtext[^>]*>[^<]*\\|linebreak="newline"')
# latex2mathml copies \text{} contents verbatim, so a `<` the page had escaped would open a stray HTML tag.
_MTEXT = re.compile(r"(<mtext[^>]*>)(.*?)(?=</mtext>)", re.DOTALL)

log = logging.getLogger("mkdocs.hooks.prerender_math")


def on_page_content(html_content: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Replace every arithmatex block and inline span with MathML.

    Args:
        html_content (str): The page body HTML.
        page (Page): The page being built.
        config (MkDocsConfig): The MkDocs config.
        files (Files): All site files.

    Returns:
        str: The page body with math as MathML.

    Raises:
        ValueError: If latex2mathml can't parse an equation.
    """
    # Keep the wrapper: its Material CSS scrolls equations wider than a phone, which overflow-x: hidden would clip.
    html_content = _BLOCK_MATH.sub(
        lambda match: f'<div class="arithmatex">{_to_mathml(match.group(1), "block", page)}</div>', html_content
    )
    html_content = _INLINE_MATH.sub(lambda match: _to_mathml(match.group(1), "inline", page), html_content)
    # With no MathJax fallback, math the patterns miss (say, a wrapper that gained attributes) shows as raw LaTeX.
    if 'class="arithmatex">\\' in html_content:
        log.warning("%s has arithmatex math left unconverted; has its HTML output changed?", page.file.src_uri)
    return html_content


def _to_mathml(escaped_latex: str, display: str, page: Page) -> str:
    latex = html.unescape(escaped_latex)
    try:
        mathml = convert(latex, display=display)
    except Exception as err:
        msg = f"{page.file.src_uri}: latex2mathml could not parse {latex.strip()!r}"
        raise ValueError(msg) from err
    mathml = _MTEXT.sub(lambda match: match.group(1) + match.group(2).replace("<", "&lt;"), mathml)
    if _UNRENDERED.search(mathml) is not None:
        log.warning("%s: latex2mathml can't faithfully render %r", page.file.src_uri, latex.strip())
    # MathML text indexes as run-together tokens ("ACV=∑unit_spend..."), which only adds noise to search results.
    return mathml.replace("<math ", "<math data-search-exclude ", 1)
