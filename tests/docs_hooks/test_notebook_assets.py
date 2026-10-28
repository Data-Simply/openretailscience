"""Tests for the notebook_assets docs build hook."""

from __future__ import annotations

import base64
import re
import struct
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from notebook_assets import on_post_page

BAR_GALLERY_PAGE = SimpleNamespace(file=SimpleNamespace(src_uri="gallery/plots/bar.ipynb"), url="gallery/plots/bar/")
NOTEBOOK_CSS = ".jp-OutputPrompt{color:#616161}"
LINK_HREF = re.compile(r'<link rel="stylesheet" href="([^"]+)">')
IMG_TAG = re.compile(r"<img [^>]*>")


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))


def _png(width: int, height: int) -> bytes:
    """Build a valid white RGB PNG of the given size, like a matplotlib figure saved by a notebook."""
    rows = b"".join(b"\x00" + b"\xff" * (3 * width) for _ in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows))
        + _png_chunk(b"IEND", b"")
    )


def _notebook_img(png: bytes, base64_suffix: str = "") -> str:
    """Render a plot output the way mkdocs-jupyter does."""
    data = base64.b64encode(png).decode() + base64_suffix
    return f'<img alt="No description has been provided for this image" class="" src="data:image/png;base64,{data}"/>'


def _page_html(body: str) -> str:
    return f'<style type="text/css">{NOTEBOOK_CSS}</style><div class="jupyter-wrapper">{body}</div>'


@pytest.fixture
def site_config(tmp_path: Path) -> SimpleNamespace:
    """A stand-in MkDocs config whose site_dir is a fresh temporary directory."""
    return SimpleNamespace(site_dir=str(tmp_path))


class TestNotebookAssets:
    """Tests for moving notebook CSS and plots out of the page HTML."""

    @pytest.mark.parametrize(
        ("page_url", "prefix"),
        [("gallery/plots/bar/", "../../../"), ("examples/segmentation/", "../../")],
    )
    def test_inline_css_moves_to_a_linked_file(self, site_config, page_url, prefix):
        """The style block becomes a link, relative to the page, to a file holding the same CSS."""
        page = SimpleNamespace(file=SimpleNamespace(src_uri="notebook.ipynb"), url=page_url)

        output = on_post_page(_page_html(""), page, site_config)

        hrefs = LINK_HREF.findall(output)
        assert len(hrefs) == 1
        assert hrefs[0].startswith(f"{prefix}assets/notebook/")
        assert "<style" not in output
        assert Path(site_config.site_dir, hrefs[0].removeprefix(prefix)).read_text() == NOTEBOOK_CSS

    # nbconvert copies the notebook's base64 verbatim, and older ipykernels saved it with a trailing newline.
    @pytest.mark.parametrize("base64_suffix", ["", "\n"])
    def test_plot_moves_to_a_file_with_its_intrinsic_size(self, site_config, base64_suffix):
        """A base64 plot is written out unchanged and the tag gains the PNG's width and height."""
        png = _png(640, 480)

        output = on_post_page(_page_html(_notebook_img(png, base64_suffix)), BAR_GALLERY_PAGE, site_config)

        [tag] = IMG_TAG.findall(output)
        src = re.search(r'src="\.\./\.\./\.\./([^"]+)"', tag).group(1)
        assert Path(site_config.site_dir, src).read_bytes() == png
        assert 'width="640" height="480"' in tag
        assert "data:image" not in output

    def test_every_plot_but_the_first_is_lazy_loaded(self, site_config):
        """The first plot may be the largest paint, so only the plots after it wait until scrolled to."""
        plots = "".join(_notebook_img(_png(width, 300)) for width in (600, 610, 620))

        output = on_post_page(_page_html(plots), BAR_GALLERY_PAGE, site_config)

        assert ['loading="lazy"' in tag for tag in IMG_TAG.findall(output)] == [False, True, True]

    # nbconvert writes each of width and height only when the output's metadata has it, unquoted in older templates.
    @pytest.mark.parametrize("size_attr", ['width="600"', "width=600", 'height="400"'])
    def test_existing_size_is_kept(self, site_config, size_attr):
        """A plot whose tag already sets a width or height doesn't get a second, conflicting size."""
        data = base64.b64encode(_png(1200, 800)).decode()
        html = _page_html(f'<img src="data:image/png;base64,{data}" {size_attr}/>')

        [tag] = IMG_TAG.findall(on_post_page(html, BAR_GALLERY_PAGE, site_config))

        assert re.findall(r"(?:width|height)=[^\s>]+", tag) == [size_attr]

    def test_identical_outputs_share_one_file(self, site_config):
        """Two pages with the same CSS and plot link the same files, so browsers download them once."""
        segmentation_page = SimpleNamespace(
            file=SimpleNamespace(src_uri="examples/segmentation.ipynb"), url="examples/segmentation/"
        )
        html = _page_html(_notebook_img(_png(640, 480)))

        bar_output = on_post_page(html, BAR_GALLERY_PAGE, site_config)
        segmentation_output = on_post_page(html, segmentation_page, site_config)

        bar_files = [ref.split("/")[-1] for ref in re.findall(r'(?:href|src)="([^"]+)"', bar_output)]
        segmentation_files = [ref.split("/")[-1] for ref in re.findall(r'(?:href|src)="([^"]+)"', segmentation_output)]
        assert bar_files == segmentation_files
        assert len(list(Path(site_config.site_dir, "assets/notebook").iterdir())) == len(bar_files)

    def test_markdown_pages_are_left_alone(self, site_config):
        """Only notebook pages are rewritten; a Markdown page's own style block stays inline."""
        page = SimpleNamespace(file=SimpleNamespace(src_uri="getting_started/installation.md"), url="installation/")
        html = _page_html(_notebook_img(_png(640, 480)))

        assert on_post_page(html, page, site_config) == html

    def test_warns_when_a_notebook_has_no_inline_css(self, site_config, caplog):
        """If mkdocs-jupyter stops emitting the expected style tag, the build says so instead of silently inlining."""
        on_post_page('<div class="jupyter-wrapper"></div>', BAR_GALLERY_PAGE, site_config)

        assert [record.levelname for record in caplog.records] == ["WARNING"]
        assert "gallery/plots/bar.ipynb" in caplog.records[0].getMessage()
