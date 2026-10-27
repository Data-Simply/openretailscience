"""Generate llms.txt (https://llmstxt.org) from the nav and page meta descriptions, so its page list can't drift."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from mkdocs.structure.nav import Section
from mkdocs.structure.pages import Page

if TYPE_CHECKING:
    from collections.abc import Iterator

    from mkdocs.config.defaults import MkDocsConfig
    from mkdocs.structure import StructureItem
    from mkdocs.structure.files import Files
    from mkdocs.structure.nav import Navigation

_INTRO = """# OpenRetailScience

> OpenRetailScience (formerly PyRetailScience) is an open source Python package for retail analytics: RFM, HML and \
threshold segmentation, gain/loss, cross-shop, cohort retention, revenue trees and market basket analysis. It runs on \
pandas or directly in DuckDB, BigQuery, Snowflake, Databricks, PySpark, SQL Server or Oracle through Ibis.

Install with `pip install openretailscience`. Analyses take a transaction table with standard columns (customer_id, \
transaction_id, transaction_date, unit_spend, ...) whose names can be remapped through options.

AI agents writing code with the package should start with its [agent skill](https://github.com/Data-Simply/\
openretailscience/blob/main/openretailscience/.agents/skills/using-openretailscience/SKILL.md), an API guide with \
references and runnable examples.
"""

_OPTIONAL = """## Optional

- [Source code](https://github.com/Data-Simply/openretailscience)
"""

# on_post_build doesn't receive the nav, and page descriptions are only read after on_nav, so keep it until then.
_build: dict[str, Navigation] = {}


def on_nav(nav: Navigation, config: MkDocsConfig, files: Files) -> Navigation:
    """Keep the nav for on_post_build.

    Args:
        nav (Navigation): The site navigation.
        config (MkDocsConfig): The MkDocs config.
        files (Files): All site files.

    Returns:
        Navigation: The unchanged nav.
    """
    _build["nav"] = nav
    return nav


def on_post_build(config: MkDocsConfig) -> None:
    """Write llms.txt to the site root with one section per top-level nav item.

    Args:
        config (MkDocsConfig): The MkDocs config.
    """
    # The homepage is skipped because _INTRO already describes the package.
    sections = [
        _section(item.title, item.children if isinstance(item, Section) else [item])
        for item in _build["nav"].items
        if isinstance(item, Section) or (isinstance(item, Page) and not item.is_homepage)
    ]
    Path(config.site_dir, "llms.txt").write_text("\n".join([_INTRO, *sections, _OPTIONAL]), encoding="utf-8")


def _section(title: str, items: list[StructureItem]) -> str:
    lines = [f"## {title}", ""]
    for page, trail in _pages(items, []):
        description = page.meta.get("description")
        suffix = f": {description}" if description is not None else ""
        lines.append(f"- [{': '.join([*trail, page.title])}]({page.canonical_url}){suffix}")
    return "\n".join(lines) + "\n"


def _pages(items: list[StructureItem], trail: list[str]) -> Iterator[tuple[Page, list[str]]]:
    for item in items:
        if isinstance(item, Section):
            yield from _pages(item.children, [*trail, item.title])
        elif isinstance(item, Page):
            yield item, trail
