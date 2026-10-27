"""Give every page a meta description, which search engines show as the result snippet.

Markdown pages set ``description`` in front matter. mkdocs-jupyter pages have no front matter (their source is the
notebook JSON), so they set ``description`` in the notebook's top-level ``metadata`` instead.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure.files import Files
    from mkdocs.structure.pages import Page

log = logging.getLogger("mkdocs.hooks.page_descriptions")


def on_page_markdown(markdown: str, page: Page, config: MkDocsConfig, files: Files) -> str:
    """Copy a notebook's ``metadata.description`` into the page meta and warn on pages without one.

    Args:
        markdown (str): The page source; for a notebook, its raw JSON.
        page (Page): The page being built.
        config (MkDocsConfig): The MkDocs config.
        files (Files): All site files.

    Returns:
        str: The unchanged page source.
    """
    if page.file.src_uri.endswith(".ipynb"):
        description = json.loads(markdown).get("metadata", {}).get("description")
        if description is not None:
            page.meta["description"] = description
    if "description" not in page.meta:
        log.warning("%s has no meta description", page.file.src_uri)
    return markdown
