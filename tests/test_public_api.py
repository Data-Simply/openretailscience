"""Tests for the package-level public API: short imports, lazy loading and `help()` pages."""

import functools
import importlib
import inspect
import pkgutil
import pydoc
import re
import subprocess
import sys
from types import FunctionType, ModuleType
from typing import NoReturn

import pytest

import openretailscience
from openretailscience import analysis, plots, segmentation, utils

HELP_PAGE_PACKAGES = [openretailscience, analysis, segmentation, utils, plots]


def _public_definitions(package: ModuleType) -> dict[str, type | FunctionType]:
    """Map each public class/function defined in the package's modules to that object."""
    definitions = {}
    for module_info in pkgutil.iter_modules(package.__path__):
        module = importlib.import_module(f"{package.__name__}.{module_info.name}")
        names = (
            module.__all__
            if hasattr(module, "__all__")
            else [
                name
                for name, obj in vars(module).items()
                if not name.startswith("_")
                and (inspect.isclass(obj) or inspect.isfunction(obj))
                and obj.__module__ == module.__name__
            ]
        )
        definitions.update({name: getattr(module, name) for name in names})
    return definitions


def _help_page(package: ModuleType) -> str:
    """Render the page `help(package)` prints."""
    return pydoc.plain(pydoc.render_doc(package))


def _raise_attribute_error(name: str) -> NoReturn:
    """Stand in for a dependency whose API changed, so a module fails to import with an AttributeError."""
    msg = f"importing {name} hit a removed dependency attribute"
    raise AttributeError(msg)


class TestSubpackageExports:
    """Each subpackage exports the public classes and functions of its modules."""

    @pytest.mark.parametrize("package", [analysis, segmentation, utils])
    def test_exports_match_module_definitions(self, package: ModuleType) -> None:
        """A class or function added to a module without being exported fails here."""
        exported = {name: getattr(package, name) for name in dir(package)}
        assert exported == _public_definitions(package)

    def test_plots_exports_every_chart_module(self) -> None:
        """A chart module missing from `dir(plots)` is unreachable as `plots.<name>`."""
        assert dir(plots) == sorted(module.name for module in pkgutil.iter_modules(plots.__path__))

    def test_from_import_surfaces_module_import_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`from analysis import name` must not hide why the defining module failed as "cannot import name"."""
        monkeypatch.setattr(importlib, "import_module", _raise_attribute_error)
        with pytest.raises(ImportError) as excinfo:
            __import__(analysis.__name__, fromlist=["CrossShop"])
        assert isinstance(excinfo.value.__cause__, AttributeError)

    @pytest.mark.parametrize(
        ("package", "name"),
        [
            (openretailscience, "analytics"),
            (analysis, "GainLos"),
            (plots, "pie"),
        ],
    )
    def test_unknown_name_raises_attribute_error(self, package: ModuleType, name: str) -> None:
        """Lazy lookup must not swallow typos, or hasattr() and `from x import y` stop failing."""
        with pytest.raises(AttributeError, match=name):
            getattr(package, name)


class TestLazyLoading:
    """A package's heavy submodules load on first use of one of its names, not on package import."""

    @pytest.mark.parametrize(
        ("package_name", "name", "lazy_module"),
        [
            ("openretailscience", "analysis", "openretailscience.analysis"),
            ("openretailscience.analysis", "CustomerDecisionHierarchy", "scipy"),
            ("openretailscience.plots", "bar", "matplotlib"),
        ],
    )
    def test_module_loads_on_first_name_access_not_package_import(
        self, package_name: str, name: str, lazy_module: str
    ) -> None:
        """Runs in a fresh interpreter, since this session has already imported every submodule."""
        code = (
            f"import sys, {package_name} as package; "
            f"assert {lazy_module!r} not in sys.modules; "
            f"package.{name}; "
            f"assert {lazy_module!r} in sys.modules"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)  # noqa: S603
        assert result.returncode == 0, result.stderr


class TestHelpPages:
    """`help()` on a package is a map that names the call one level down, not every class's full docs."""

    @pytest.mark.parametrize("package", HELP_PAGE_PACKAGES)
    def test_page_has_no_class_or_function_docs(self, package: ModuleType) -> None:
        """help() renders every name in `__all__` in full, which put whole classes on the package page."""
        assert re.findall(r"^(CLASSES|FUNCTIONS)$", _help_page(package), flags=re.MULTILINE) == []

    @pytest.mark.parametrize("package", HELP_PAGE_PACKAGES)
    def test_page_names_a_working_next_call(self, package: ModuleType) -> None:
        """The one `help(...)` call on the page must resolve to a name the package lists, so it can't go stale."""
        calls = re.findall(r"help\(([\w.]+)\)", _help_page(package))
        assert len(calls) == 1
        child, *rest = calls[0].removeprefix(f"{package.__name__}.").split(".")
        assert child in dir(package)
        assert pydoc.locate(calls[0]) is functools.reduce(getattr, rest, getattr(package, child))

    @pytest.mark.parametrize("package", HELP_PAGE_PACKAGES)
    def test_map_names_every_listed_name(self, package: ModuleType) -> None:
        """A name `dir()` lists but the map omits is invisible to a reader of `help()`."""
        page = _help_page(package)
        assert [name for name in dir(package) if f"`{name}`" not in page] == []
