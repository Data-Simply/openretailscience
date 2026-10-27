"""Serve mkdocs-jupyter's CSS as shared, cacheable stylesheets instead of inlining ~575KB into every notebook."""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.utils import get_relative_url

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# The tag mkdocs-jupyter's template emits. A pandas Styler table uses it too and is extracted the same way, harmlessly.
_JUPYTER_STYLE = re.compile(r'<style type="text/css">(.*?)</style>', re.DOTALL)
_HASH_LENGTH = 12

log = logging.getLogger("mkdocs.hooks.notebook_css")


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Replace each inline mkdocs-jupyter style block in a notebook page with a link to a content-hashed file.

    Args:
        output (str): The rendered page HTML.
        page (Page): The page being built.
        config (MkDocsConfig): The MkDocs config.

    Returns:
        str: The page HTML with the notebook CSS linked rather than inlined.
    """
    if not page.file.src_uri.endswith(".ipynb"):
        return output
    output, replaced = _JUPYTER_STYLE.subn(lambda match: _link_stylesheet(match.group(1), page, config), output)
    if replaced == 0:
        log.warning(
            "%s has no inline mkdocs-jupyter CSS to extract; has the template's <style> tag changed?", page.file.src_uri
        )
    return output


def _link_stylesheet(css: str, page: Page, config: MkDocsConfig) -> str:
    # The content hash in the name means a mkdocs-jupyter upgrade can't be masked by a cached old copy.
    path = f"assets/notebook/{hashlib.sha256(css.encode()).hexdigest()[:_HASH_LENGTH]}.css"
    target = Path(config.site_dir) / path
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(css, encoding="utf-8")
    return f'<link rel="stylesheet" href="{get_relative_url(path, page.url)}">'
