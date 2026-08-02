"""Tests for ``install_skills`` running on Databricks compute.

The Databricks branch of ``install_skills`` copies into ``/Workspace/Users/<current_user()>``,
a mount that exists only on the platform, so ``tests/test_skills.py::TestDatabricksInstall`` can
only pin our own wiring against a temp directory and a mocked Spark session.

Plain pytest skips these; ``.github/workflows/databricks-integration.yml`` runs them. README.md,
under "install_skills() on Databricks", maps the pieces.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

import openretailscience
from openretailscience.skills import (
    AGENTS_DIR_NAME,
    BYTECODE_CACHE_DIR,
    DATABRICKS_ASSISTANT_DIR,
    SKILL_MARKER_FILENAME,
    SKILLS_SUBDIR,
    install_skills,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

_DATABRICKS_RUNTIME_ENV = "DATABRICKS_RUNTIME_VERSION"

# /Workspace and Users are spelled out, not imported: they are the location these tests exist to
# pin, and an expectation derived from the code under test would agree with a wrong answer.
_WORKSPACE_ROOT = Path("/Workspace")
_WORKSPACE_USERS_ROOT = _WORKSPACE_ROOT / "Users"
# The admin-only workspace-wide directory a default install must never write.
_WORKSPACE_SKILLS_DIR = _WORKSPACE_ROOT / DATABRICKS_ASSISTANT_DIR / SKILLS_SUBDIR

_SCRIPTS_SUBDIR = "scripts"

pytestmark = pytest.mark.skipif(
    _DATABRICKS_RUNTIME_ENV not in os.environ,
    reason="install_skills' Databricks branch needs the real /Workspace mount",
)


def _bundled_skill_names(bundled_skills_dir: Path) -> list[str]:
    """List the skill folders shipped in the installed package.

    Args:
        bundled_skills_dir (Path): The package's bundled skills directory.

    Returns:
        list[str]: Sorted skill folder names.
    """
    return sorted(
        entry.name
        for entry in bundled_skills_dir.iterdir()
        if entry.is_dir() and (entry / SKILL_MARKER_FILENAME).is_file()
    )


def _installed_skill_names(skills_dir: Path, bundled_skills_dir: Path) -> list[str]:
    """List the bundled skills that are present under ``skills_dir``.

    Args:
        skills_dir (Path): A skills directory an install may have written to.
        bundled_skills_dir (Path): The package's bundled skills directory.

    Returns:
        list[str]: Names present under ``skills_dir``, empty when nothing was installed there.
    """
    return [name for name in _bundled_skill_names(bundled_skills_dir) if (skills_dir / name).exists()]


def _remove_installed_copies(workspace_skills_dir: Path, bundled_skills_dir: Path) -> None:
    """Delete only the copies this package owns, leaving any other skills in place.

    Args:
        workspace_skills_dir (Path): The Genie skills directory in the caller's workspace home.
        bundled_skills_dir (Path): The package's bundled skills directory.
    """
    for name in _bundled_skill_names(bundled_skills_dir):
        shutil.rmtree(workspace_skills_dir / name, ignore_errors=True)


@pytest.fixture(scope="module")
def bundled_skills_dir() -> Path:
    """The skills directory shipped inside the installed package."""
    return Path(openretailscience.__file__).parent / AGENTS_DIR_NAME / SKILLS_SUBDIR


@pytest.fixture(scope="module")
def workspace_skills_dir() -> Path:
    """The Genie skills directory in the workspace home of the identity running the job."""
    # Imported here, not at module scope: pytest imports the module before evaluating any mark,
    # so a module-scope import would raise instead of letting the skipif above take effect.
    from pyspark.sql import SparkSession  # noqa: PLC0415 - only Databricks compute provides pyspark

    spark = SparkSession.getActiveSession()
    assert spark is not None, "Databricks compute did not provide an active Spark session"
    user = str(spark.sql("SELECT current_user()").collect()[0][0])
    return _WORKSPACE_USERS_ROOT / user / DATABRICKS_ASSISTANT_DIR / SKILLS_SUBDIR


@pytest.fixture(autouse=True)
def _clean_workspace_target(workspace_skills_dir: Path, bundled_skills_dir: Path) -> Iterator[None]:
    """Clear previously installed copies so each test starts from a known workspace state.

    Args:
        workspace_skills_dir (Path): The Genie skills directory in the caller's workspace home.
        bundled_skills_dir (Path): The package's bundled skills directory.

    Yields:
        None: Control to the test.
    """
    _remove_installed_copies(workspace_skills_dir, bundled_skills_dir)
    yield
    _remove_installed_copies(workspace_skills_dir, bundled_skills_dir)


class TestWorkspaceInstall:
    """Tests for install_skills against a real Databricks workspace filesystem."""

    def test_installs_a_readable_copy_into_the_callers_workspace_home(
        self, workspace_skills_dir: Path, bundled_skills_dir: Path
    ) -> None:
        """Every bundled skill lands in the caller's own workspace home and reads back intact."""
        result = install_skills()

        names = _bundled_skill_names(bundled_skills_dir)
        assert len(result.installed) == len(names)
        assert len(result.skipped) == 0
        for name in names:
            target = workspace_skills_dir / name
            # A symlink would point back into the package that a cluster restart wipes.
            assert not target.is_symlink()
            assert (target / SKILL_MARKER_FILENAME).read_bytes() == (
                bundled_skills_dir / name / SKILL_MARKER_FILENAME
            ).read_bytes()

    def test_workspace_wide_directory_is_never_written(
        self, workspace_skills_dir: Path, bundled_skills_dir: Path
    ) -> None:
        """The default install stays inside the user's home, not the admin-only workspace root.

        Writing the workspace-wide directory is what failed with PERMISSION_DENIED for every
        non-admin in #542.
        """
        install_skills()

        assert workspace_skills_dir.is_dir()
        assert _installed_skill_names(_WORKSPACE_SKILLS_DIR, bundled_skills_dir) == []

    def test_pip_byte_compiles_the_skill_and_the_copy_leaves_it_behind(
        self, workspace_skills_dir: Path, bundled_skills_dir: Path
    ) -> None:
        """The installed package carries bytecode caches, and none reach the workspace.

        Copying one raises ``OSError 95`` on the ``/Workspace`` mount. The precondition is
        asserted first: if ``%pip`` stopped byte-compiling, the rest would pass vacuously.
        """
        names = _bundled_skill_names(bundled_skills_dir)
        python_dirs = {path.parent for path in bundled_skills_dir.rglob("*.py")}
        # Set equality alone is satisfied by two empty sets, so the source is pinned first:
        # every bundled skill ships its example scripts under scripts/.
        assert python_dirs == {bundled_skills_dir / name / _SCRIPTS_SUBDIR for name in names}
        assert {path.parent for path in bundled_skills_dir.rglob(BYTECODE_CACHE_DIR)} == python_dirs

        install_skills()

        for name in names:
            assert list((workspace_skills_dir / name).rglob(BYTECODE_CACHE_DIR)) == []

    def test_rerun_reports_up_to_date(self, bundled_skills_dir: Path) -> None:
        """Re-running against the workspace filesystem re-copies nothing."""
        install_skills()

        result = install_skills()

        assert len(result.installed) == 0
        assert len(result.up_to_date) == len(_bundled_skill_names(bundled_skills_dir))

    def test_rerun_refreshes_a_stale_copy(self, workspace_skills_dir: Path, bundled_skills_dir: Path) -> None:
        """A workspace copy that no longer matches the package is rewritten."""
        install_skills()
        name = _bundled_skill_names(bundled_skills_dir)[0]
        stale_marker = workspace_skills_dir / name / SKILL_MARKER_FILENAME
        stale_marker.write_text("stale copy from an older release", encoding="utf-8")

        result = install_skills()

        assert stale_marker.read_bytes() == (bundled_skills_dir / name / SKILL_MARKER_FILENAME).read_bytes()
        assert len(result.installed) == 1

    def test_global_mode_is_refused(self, workspace_skills_dir: Path, bundled_skills_dir: Path) -> None:
        """Workspace-wide installs are refused rather than attempted without admin rights."""
        with pytest.raises(NotImplementedError, match="global_mode"):
            install_skills(global_mode=True)

        assert _installed_skill_names(_WORKSPACE_SKILLS_DIR, bundled_skills_dir) == []
        assert _installed_skill_names(workspace_skills_dir, bundled_skills_dir) == []
