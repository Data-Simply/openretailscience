"""Tests for the llms_txt docs build hook, run through a real MkDocs build of a small site."""

from __future__ import annotations

from pathlib import Path

import pytest
from mkdocs.commands.build import build
from mkdocs.config import load_config

HOOK = Path(__file__).parents[2] / "docs" / "hooks" / "llms_txt.py"
SITE_URL = "https://docs.example-retailer.com/"
PAGES = {
    "index.md": "Retail analytics in Python.",
    "getting_started/installation.md": "Install the retail analytics package with pip.",
    "api/segmentation/rfm.md": "Score customers on recency, frequency and monetary value.",
    "api/options.md": None,
    "changelog.md": "What changed in each release.",
}
MKDOCS_YML = f"""
site_name: Retail Analytics
site_url: {SITE_URL}
hooks:
  - {HOOK}
nav:
  - Home: index.md
  - Getting Started:
      - Installation: getting_started/installation.md
  - Reference:
      - Options: api/options.md
      - Segmentation:
          - RFM Segmentation: api/segmentation/rfm.md
  - Changelog: changelog.md
"""


@pytest.fixture(scope="module")
def llms_txt(tmp_path_factory: pytest.TempPathFactory) -> str:
    """Build the small site once and return its generated llms.txt."""
    root = tmp_path_factory.mktemp("site")
    for path, description in PAGES.items():
        front_matter = f"---\ndescription: {description}\n---\n" if description is not None else ""
        Path(root, "docs", path).parent.mkdir(parents=True, exist_ok=True)
        Path(root, "docs", path).write_text(f"{front_matter}# {Path(path).stem}\n")
    Path(root, "mkdocs.yml").write_text(MKDOCS_YML)
    build(load_config(config_file=str(root / "mkdocs.yml"), site_dir=str(root / "site")))
    return Path(root, "site", "llms.txt").read_text()


class TestLlmsTxt:
    """Tests for generating llms.txt from the nav and page descriptions."""

    def test_sections_follow_the_nav(self, llms_txt):
        """Each top-level nav item becomes a section; nested sections prefix page names, and Home is skipped."""
        sections = llms_txt[llms_txt.index("## Getting Started") : llms_txt.index("## Optional")]

        assert sections == (
            "## Getting Started\n\n"
            f"- [Installation]({SITE_URL}getting_started/installation/): "
            "Install the retail analytics package with pip.\n\n"
            "## Reference\n\n"
            f"- [Options]({SITE_URL}api/options/)\n"
            f"- [Segmentation: RFM Segmentation]({SITE_URL}api/segmentation/rfm/): "
            "Score customers on recency, frequency and monetary value.\n\n"
            "## Changelog\n\n"
            f"- [Changelog]({SITE_URL}changelog/): What changed in each release.\n\n"
        )

    def test_opens_with_the_package_summary(self, llms_txt):
        """The llmstxt.org format needs the H1 and summary blockquote before any section."""
        assert llms_txt.startswith("# OpenRetailScience\n\n> ")
