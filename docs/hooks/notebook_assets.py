"""Move mkdocs-jupyter's inline CSS and base64 plots out of notebook pages into shared, content-hashed files.

Works on the rendered HTML, so re-executed notebooks need no extra step.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import logging
import re
import struct
from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.utils import get_relative_url, write_file

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.pages import Page

# mkdocs-jupyter's template tag. pandas Styler output uses it too and is extracted the same way, harmlessly.
_JUPYTER_STYLE = re.compile(r'<style type="text/css">(.*?)</style>', re.DOTALL)
# Whitespace is allowed because nbconvert copies the notebook's base64 verbatim; b64decode discards it.
_PNG_IMAGE = re.compile(r'<img ([^>]*?)src="data:image/png;base64,([A-Za-z0-9+/=\s]+)"([^>]*?)/?>')
_HASH_LENGTH = 12
# A PNG's width and height are big-endian uint32s at bytes 16-24, inside the mandatory first (IHDR) chunk.
_PNG_SIZE = slice(16, 24)
# nbconvert sets each from the output's metadata independently, so either alone means the notebook chose a size.
_SIZE_ATTR = re.compile(r"(?:^|\s)(?:width|height)=")

log = logging.getLogger("mkdocs.hooks.notebook_assets")


def on_post_page(output: str, page: Page, config: MkDocsConfig) -> str:
    """Replace a notebook page's inline style blocks and PNG data URIs with links to content-hashed files.

    Args:
        output (str): The rendered page HTML.
        page (Page): The page being built.
        config (MkDocsConfig): The MkDocs config.

    Returns:
        str: The page HTML with notebook CSS and images linked rather than inlined.
    """
    if not page.file.src_uri.endswith(".ipynb"):
        return output
    output, replaced = _JUPYTER_STYLE.subn(lambda match: _link_stylesheet(match.group(1), page, config), output)
    if replaced == 0:
        log.warning(
            "%s has no inline mkdocs-jupyter CSS to extract; has the template's <style> tag changed?", page.file.src_uri
        )
    # The first plot may be the largest contentful paint, which lazy loading would delay.
    image_index = itertools.count()
    return _PNG_IMAGE.sub(lambda match: _link_image(match, page, config, lazy=next(image_index) > 0), output)


def _link_stylesheet(css: str, page: Page, config: MkDocsConfig) -> str:
    path = _write_asset(css.encode(), "css", config)
    return f'<link rel="stylesheet" href="{get_relative_url(path, page.url)}">'


def _link_image(match: re.Match[str], page: Page, config: MkDocsConfig, lazy: bool) -> str:
    before, data, after = match.groups()
    png = base64.b64decode(data)
    attrs = f'{before}src="{get_relative_url(_write_asset(png, "png", config), page.url)}"{after}'.rstrip()
    if _SIZE_ATTR.search(attrs) is None:
        # Lets the browser reserve the plot's space before it loads, so text below doesn't jump.
        width, height = struct.unpack(">II", png[_PNG_SIZE])
        attrs += f' width="{width}" height="{height}"'
    if lazy:
        attrs += ' loading="lazy" decoding="async"'
    return f"<img {attrs}>"


def _write_asset(content: bytes, suffix: str, config: MkDocsConfig) -> str:
    # The content hash in the name means changed output can't be masked by a cached old copy.
    path = f"assets/notebook/{hashlib.sha256(content).hexdigest()[:_HASH_LENGTH]}.{suffix}"
    target = Path(config.site_dir) / path
    if not target.exists():
        write_file(content, str(target))
    return path
