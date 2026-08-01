"""Tests for openretailscience.skills."""

from __future__ import annotations

import os
import re
import runpy
import shutil
import sys
from pathlib import PurePath
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

from openretailscience import skills
from openretailscience.options import get_option, list_options, set_option
from openretailscience.skills import (
    _discover_skills,
    _find_project_root,
    _get_source_skills_dir,
    _is_owned_target,
    _relative_symlink_target,
    _skill_copy_matches,
    install_skills,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator
    from pathlib import Path
    from typing import NoReturn

# Two realistic bundled-skill names plus a junk dir with no SKILL.md.
SKILL_NAMES = ("retail-metrics", "using-openretailscience")
DATABRICKS_USER = "analyst@retail.com"
# Spelled out, not imported: renaming the package constant must fail these tests,
# because Databricks reads this exact path.
DATABRICKS_ASSISTANT_DIR = ".assistant"
# Patched by name so pyspark, a dev-only transitive dependency, stays out of imports.
GET_ACTIVE_SESSION = "pyspark.sql.SparkSession.getActiveSession"
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)
SHIPPED_SKILL_NAME = "using-openretailscience"
# Fenced code blocks, and openretailscience import statements inside them.
CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
IMPORT_RE = re.compile(r"^[ \t]*(?:from|import)\s+openretailscience[\w. ,]*(?:import[\w. ,]+)?$", re.MULTILINE)
# Markdown references to sibling skill files, e.g. `references/plotting.md`.
REFERENCE_RE = re.compile(r"references/[\w./-]+\.md")
# Lower bounds, not targets: the skill ships far more of each. Without them a glob
# that matches nothing would let the guards below pass vacuously.
MIN_IMPORT_EXAMPLES = 20
MIN_REFERENCE_FILES = 3
MIN_EXAMPLE_SCRIPTS = 30
MIN_DESCRIPTION_LENGTH = 50


def _import_error(statement: str) -> str | None:
    """Return an error message if an import statement fails to resolve, else None."""
    try:
        exec(statement, {})  # noqa: S102 - trusted first-party skill content
    except ImportError as exc:
        return f"{statement!r} ({exc})"
    return None


def _shipped_skill_markdown() -> list[Path]:
    """Return SKILL.md and every reference markdown file of the shipped skill."""
    skill_root = _get_source_skills_dir() / SHIPPED_SKILL_NAME
    return [skill_root / "SKILL.md", *sorted(skill_root.glob("references/*.md"))]


def _skill_import_statements() -> list[str]:
    """Extract every openretailscience import line from the shipped skill's code fences."""
    statements: list[str] = []
    for md_file in _shipped_skill_markdown():
        for block in CODE_FENCE_RE.findall(md_file.read_text(encoding="utf-8")):
            statements.extend(match.strip() for match in IMPORT_RE.findall(block))
    return statements


def _raise_oserror(*_args: object, **_kwargs: object) -> NoReturn:
    """Raise OSError; a patched-call stand-in for a failing operation (unsupported symlink, cross-drive path)."""
    msg = "symlinks not supported"
    raise OSError(msg)


def _raise_value_error(*_args: object, **_kwargs: object) -> NoReturn:
    """Stand-in for os.path.relpath that reports incompatible paths."""
    msg = "paths are on different drives"
    raise ValueError(msg)


def _raise_not_implemented(*_args: object, **_kwargs: object) -> NoReturn:
    """Raise NotImplementedError; a patched-call stand-in for an operation unsupported on the platform."""
    raise NotImplementedError


def _make_bare_skill(root: Path, name: str, body: bytes = b"guidance") -> Path:
    """Create a skill dir with a SKILL.md of the given bytes; return the dir."""
    skill_dir = root / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_bytes(body)
    return skill_dir


def _write_skill(skills_dir: Path, name: str) -> None:
    """Create a minimal valid skill folder under ``skills_dir``."""
    skill_dir = skills_dir / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: Test skill {name}.\n---\n\n# {name}\n\nGuidance.\n",
        encoding="utf-8",
    )


@pytest.fixture
def source_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake bundled-skills source dir with two skills and one non-skill dir."""
    src = tmp_path / "site-packages" / "openretailscience" / ".agents" / "skills"
    src.mkdir(parents=True)
    for name in SKILL_NAMES:
        _write_skill(src, name)
    (src / "not-a-skill").mkdir()  # no SKILL.md -> must be ignored
    monkeypatch.setattr(skills, "_get_source_skills_dir", lambda: src)
    return src


@pytest.fixture
def fake_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect ``Path.home()`` to a temp dir and clear Databricks detection.

    ``USERPROFILE`` too: ``ntpath.expanduser`` ignores ``HOME``, so on Windows a
    global-mode test would install into the developer's real home directory.
    """
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    return home


@pytest.fixture
def project_dir(tmp_path: Path, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A project working directory (cwd) with no agent markers yet.

    The ``.git`` marker anchors ``_find_project_root`` here deterministically, so
    the upward root search cannot escape the tmp sandbox into a real repository.
    """
    project = tmp_path / "retail-project"
    project.mkdir()
    (project / ".git").mkdir()
    monkeypatch.chdir(project)
    return project


class TestDiscoverSkills:
    """Tests for _discover_skills."""

    def test_returns_only_dirs_with_skill_marker(self, source_dir: Path) -> None:
        """Only subdirectories containing SKILL.md are treated as skills."""
        assert _discover_skills(source_dir) == list(SKILL_NAMES)

    def test_returns_empty_for_missing_dir(self, tmp_path: Path) -> None:
        """A missing source directory yields no skills rather than raising."""
        assert _discover_skills(tmp_path / "does-not-exist") == []


class TestSymlinkInstall:
    """Tests for symlink installation, into a project (default) or the home directory."""

    @pytest.mark.parametrize("global_mode", [False, True])
    def test_creates_symlinks_resolving_to_source(
        self, source_dir: Path, project_dir: Path, fake_home: Path, global_mode: bool
    ) -> None:
        """Each skill is symlinked into the .agents/skills of the project or the home dir."""
        result = install_skills(global_mode=global_mode)

        target_dir = (fake_home if global_mode else project_dir) / ".agents" / "skills"
        for name in SKILL_NAMES:
            link = target_dir / name
            assert link.is_symlink()
            assert link.resolve() == (source_dir / name).resolve()
        assert len(result.installed) == len(SKILL_NAMES)

    @pytest.mark.parametrize("global_mode", [False, True])
    @pytest.mark.parametrize("claude_present", [True, False])
    def test_claude_dir_targeted_only_when_claude_home_exists(
        self, source_dir: Path, project_dir: Path, fake_home: Path, claude_present: bool, global_mode: bool
    ) -> None:
        """The .claude/skills target is used only when ~/.claude exists."""
        if claude_present:
            (fake_home / ".claude").mkdir()

        install_skills(global_mode=global_mode)

        base = fake_home if global_mode else project_dir
        assert (base / ".claude" / "skills" / SKILL_NAMES[0]).is_symlink() is claude_present

    @pytest.mark.parametrize("symlinks_supported", [True, False])
    def test_rerun_is_idempotent(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch, symlinks_supported: bool
    ) -> None:
        """A second run reports every skill up to date, whether it linked or fell back to copying.

        The fallback case is the interesting one: the first run leaves a real
        directory in a symlink-mode target, and the second must recognize that copy
        as its own (byte-identical to the source) rather than skip it as a conflict.
        """
        if not symlinks_supported:
            monkeypatch.setattr("pathlib.Path.symlink_to", _raise_oserror)

        install_skills()
        result = install_skills()

        assert len(result.installed) == 0
        assert len(result.skipped) == 0
        assert len(result.up_to_date) == len(SKILL_NAMES)

    @pytest.mark.parametrize("conflict_kind", ["unrelated_file", "user_authored_skill"])
    def test_conflicting_target_is_skipped_not_clobbered(
        self, source_dir: Path, project_dir: Path, conflict_kind: str
    ) -> None:
        """A conflicting target is left byte-for-byte intact and reported skipped.

        Two separate refusals: an unrelated file is not the installer's to touch at
        all, while a real skill directory is owned by name but still left alone in
        symlink mode, because it may be the user's own rather than a prior install.
        """
        target_dir = project_dir / ".agents" / "skills"
        conflict = target_dir / SKILL_NAMES[0]
        if conflict_kind == "unrelated_file":
            target_dir.mkdir(parents=True)
            conflict.write_text("user data", encoding="utf-8")
            preserved = conflict
        else:
            conflict.mkdir(parents=True)
            (conflict / "SKILL.md").write_text("my own skill", encoding="utf-8")
            preserved = conflict / "SKILL.md"
        expected = preserved.read_text(encoding="utf-8")

        result = install_skills()

        assert preserved.read_text(encoding="utf-8") == expected
        assert not conflict.is_symlink()
        assert result.skipped == [str(conflict.relative_to(project_dir))]
        # The non-conflicting skill still installs.
        assert (target_dir / SKILL_NAMES[1]).is_symlink()

    def test_stale_symlink_is_repointed_to_source(self, source_dir: Path, project_dir: Path) -> None:
        """An owned symlink pointing at the wrong source is unlinked and reinstalled."""
        target_dir = project_dir / ".agents" / "skills"
        target_dir.mkdir(parents=True)
        stale_source = project_dir / "old-source"
        stale_source.mkdir()
        link = target_dir / SKILL_NAMES[0]
        link.symlink_to(stale_source, target_is_directory=True)

        result = install_skills()

        assert link.is_symlink()
        assert link.resolve() == (source_dir / SKILL_NAMES[0]).resolve()
        assert str(link.relative_to(project_dir)) in result.installed

    @pytest.mark.parametrize("raiser", [_raise_oserror, _raise_not_implemented])
    def test_copies_directory_when_symlink_unsupported(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch, raiser: Callable[..., NoReturn]
    ) -> None:
        """When creating the symlink raises OSError or NotImplementedError, the skill is copied instead."""
        # Patch Path.symlink_to (what _try_symlink calls), not os.symlink: on Python 3.10 pathlib binds
        # os.symlink at import time, so patching os.symlink would not reach the symlink call.
        monkeypatch.setattr("pathlib.Path.symlink_to", raiser)

        install_skills()

        for name in SKILL_NAMES:
            target = project_dir / ".agents" / "skills" / name
            assert target.is_dir()
            assert not target.is_symlink()
            assert (target / "SKILL.md").read_bytes() == (source_dir / name / "SKILL.md").read_bytes()


class TestDatabricksInstall:
    """Tests for the Databricks copy-to-Workspace branch."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Simulate a Databricks runtime with a redirected persistent workspace root."""
        root = tmp_path / "Workspace"
        (root / "Users" / DATABRICKS_USER).mkdir(parents=True)
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "17.3")
        monkeypatch.setattr(skills, "_DATABRICKS_WORKSPACE_ROOT", root)
        return root

    @pytest.fixture
    def user_skills_dir(self, workspace_root: Path) -> Path:
        """The per-user Genie skills directory install_skills must write to."""
        return workspace_root / "Users" / DATABRICKS_USER / DATABRICKS_ASSISTANT_DIR / "skills"

    @pytest.fixture(autouse=True)
    def spark_session(self, monkeypatch: pytest.MonkeyPatch) -> MagicMock:
        """Stand in for the Databricks-provided active Spark session, answering current_user()."""
        session = MagicMock()
        session.sql.return_value.collect.return_value = [[DATABRICKS_USER]]
        monkeypatch.setattr(GET_ACTIVE_SESSION, lambda: session)
        return session

    def test_copies_to_per_user_assistant_dir(
        self, source_dir: Path, workspace_root: Path, user_skills_dir: Path, spark_session: MagicMock
    ) -> None:
        """The default install copies (not links) into the workspace home of the current_user()."""
        install_skills()

        assert "current_user()" in spark_session.sql.call_args.args[0]
        for name in SKILL_NAMES:
            target = user_skills_dir / name
            assert target.is_dir()
            assert not target.is_symlink()
            assert (target / "SKILL.md").is_file()
        # The workspace-wide directory needs admin rights.
        assert not (workspace_root / DATABRICKS_ASSISTANT_DIR).exists()

    def test_global_mode_is_refused(
        self, source_dir: Path, workspace_root: Path, user_skills_dir: Path, spark_session: MagicMock
    ) -> None:
        """Workspace-wide installs are refused rather than attempted without admin rights."""
        with pytest.raises(NotImplementedError, match="global_mode"):
            install_skills(global_mode=True)

        # Refusing before the session lookup keeps a sessionless caller from being
        # told to fix the wrong thing.
        spark_session.sql.assert_not_called()
        assert not (workspace_root / DATABRICKS_ASSISTANT_DIR).exists()
        assert not user_skills_dir.exists()

    def test_raises_without_a_spark_session(
        self, source_dir: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With no session to name the user, the install raises instead of guessing a target."""
        monkeypatch.setattr(GET_ACTIVE_SESSION, lambda: None)

        with pytest.raises(RuntimeError, match="Spark session"):
            install_skills()

        assert list(workspace_root.rglob(DATABRICKS_ASSISTANT_DIR)) == []

    def test_raises_when_workspace_home_does_not_exist(
        self, source_dir: Path, workspace_root: Path, spark_session: MagicMock
    ) -> None:
        """A username with no workspace home raises rather than creating one the caller may not own."""
        spark_session.sql.return_value.collect.return_value = [["service-principal-1234"]]

        with pytest.raises(RuntimeError, match="service-principal-1234"):
            install_skills()

        assert not (workspace_root / "Users" / "service-principal-1234").exists()
        assert list(workspace_root.rglob(DATABRICKS_ASSISTANT_DIR)) == []

    @pytest.mark.parametrize(
        "rows",
        [
            pytest.param([], id="no-rows"),
            pytest.param([[""]], id="blank"),
            pytest.param([[".."]], id="parent-traversal"),
            # Exists, so only the guard stops the install, and is harmless if it regresses.
            pytest.param([["/tmp"]], id="absolute"),  # noqa: S108
            pytest.param([["team/analyst@retail.com"]], id="nested"),
        ],
    )
    def test_raises_when_current_user_is_not_a_workspace_user(
        self, source_dir: Path, workspace_root: Path, spark_session: MagicMock, rows: list[list[str]]
    ) -> None:
        """A blank or path-like current_user() is refused before it is joined into a path.

        Each of these joins to a directory that exists, so the workspace-home check
        would wave it through: pathlib drops an empty segment, restarts from an
        absolute one, and keeps ``..``.
        """
        spark_session.sql.return_value.collect.return_value = rows

        with pytest.raises(RuntimeError, match="current_user"):
            install_skills()

        assert list(workspace_root.rglob(DATABRICKS_ASSISTANT_DIR)) == []

    def test_rerun_reports_up_to_date_when_copy_matches(self, source_dir: Path, workspace_root: Path) -> None:
        """Re-running on Databricks with unchanged skills reports them up to date."""
        install_skills()
        result = install_skills()

        assert len(result.installed) == 0
        assert len(result.up_to_date) == len(SKILL_NAMES)

    def test_rerun_refreshes_copy_when_source_changed(self, source_dir: Path, user_skills_dir: Path) -> None:
        """A changed bundled skill is re-copied over the stale Databricks copy."""
        install_skills()
        (source_dir / SKILL_NAMES[0] / "SKILL.md").write_text(
            f"---\nname: {SKILL_NAMES[0]}\ndescription: Updated.\n---\n\n# updated\n",
            encoding="utf-8",
        )

        result = install_skills()

        target = user_skills_dir / SKILL_NAMES[0] / "SKILL.md"
        assert "Updated." in target.read_text(encoding="utf-8")
        assert len(result.installed) == 1

    @pytest.mark.parametrize("existing_kind", ["stale_copy", "symlink"])
    def test_existing_target_is_replaced_by_a_current_copy(
        self, source_dir: Path, user_skills_dir: Path, existing_kind: str
    ) -> None:
        """A stale copy or a leftover symlink at the target is replaced by a real, current copy.

        It must end up a real directory: a symlink would point back into the package
        that a cluster restart wipes.
        """
        target = user_skills_dir / SKILL_NAMES[0]
        target.parent.mkdir(parents=True)
        if existing_kind == "stale_copy":
            target.mkdir()
            (target / "SKILL.md").write_text("stale copy", encoding="utf-8")
        else:
            target.symlink_to(source_dir / SKILL_NAMES[0])

        install_skills()

        assert target.is_dir()
        assert not target.is_symlink()
        assert (target / "SKILL.md").read_bytes() == (source_dir / SKILL_NAMES[0] / "SKILL.md").read_bytes()

    def test_copy_is_independent_of_source(self, source_dir: Path, user_skills_dir: Path) -> None:
        """The copied skill survives the ephemeral package being wiped on restart."""
        install_skills()
        expected = (source_dir / SKILL_NAMES[0] / "SKILL.md").read_bytes()

        # Simulate the ephemeral package being wiped on cluster restart.
        shutil.rmtree(source_dir)

        assert (user_skills_dir / SKILL_NAMES[0] / "SKILL.md").read_bytes() == expected


class TestBundledSkill:
    """Tests validating the real skill shipped inside the package."""

    def test_shipped_skill_is_discoverable(self) -> None:
        """The real bundled skill is discovered from the installed package."""
        assert SHIPPED_SKILL_NAME in _discover_skills(_get_source_skills_dir())

    def test_shipped_skill_frontmatter_is_valid(self) -> None:
        """The shipped SKILL.md has YAML frontmatter with matching name and a description."""
        skill_md = _get_source_skills_dir() / SHIPPED_SKILL_NAME / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")

        match = FRONTMATTER_RE.match(text)
        assert match is not None, "SKILL.md must start with YAML frontmatter"
        block = match.group(1)

        name_match = re.search(r"^name:\s*(\S+)", block, re.MULTILINE)
        assert name_match is not None
        assert name_match.group(1) == SHIPPED_SKILL_NAME

        assert "description:" in block
        # The description is a YAML folded scalar (``>-``) with its body on the
        # following wrapped lines; assert that body is substantive, not just the
        # ``>-`` indicator.
        description_body = block.split("description:", 1)[1].replace(">-", " ").strip()
        assert len(description_body) >= MIN_DESCRIPTION_LENGTH

    def test_every_reference_file_is_linked_and_every_link_resolves(self) -> None:
        """The reference files the skill links to are exactly the ones on disk.

        Equality catches drift in both directions: a link to a file that was renamed
        or deleted, and a reference file no agent will ever be pointed at.
        """
        skill_root = _get_source_skills_dir() / SHIPPED_SKILL_NAME
        referenced: set[str] = set()
        for md_file in _shipped_skill_markdown():
            referenced.update(REFERENCE_RE.findall(md_file.read_text(encoding="utf-8")))

        # as_posix(): the links are written with forward slashes on every platform.
        on_disk = {path.relative_to(skill_root).as_posix() for path in (skill_root / "references").rglob("*.md")}
        assert len(on_disk) >= MIN_REFERENCE_FILES
        assert referenced == on_disk

    def test_import_examples_resolve_against_the_package(self) -> None:
        """Every openretailscience import the skill teaches still resolves.

        This is the content-drift guard: renaming or removing a module or public
        symbol the skill documents breaks its import example and fails here,
        forcing the skill to be updated alongside the API change.
        """
        statements = _skill_import_statements()
        assert len(statements) >= MIN_IMPORT_EXAMPLES, "skill should teach many concrete imports"
        failures = [msg for statement in statements if (msg := _import_error(statement)) is not None]
        assert len(failures) == 0, "skill imports no longer resolve:\n" + "\n".join(failures)


class TestFindProjectRoot:
    """Unit tests for _find_project_root."""

    @pytest.mark.parametrize(
        ("markers", "root_is_ancestor"),
        [
            pytest.param([".git"], True, id="git-ancestor"),
            pytest.param([".agents"], True, id="agents-ancestor"),
            pytest.param([".claude"], True, id="claude-ancestor"),
            pytest.param([], False, id="no-markers"),
        ],
    )
    def test_marker_on_an_ancestor_selects_the_root(
        self, tmp_path: Path, fake_home: Path, markers: list[str], root_is_ancestor: bool
    ) -> None:
        """Any of .git, .agents or .claude on an ancestor makes it the root; without one, the start dir is."""
        repo = tmp_path / "repo"
        sub = repo / "pkg"
        sub.mkdir(parents=True)
        for marker in markers:
            (repo / marker).mkdir()

        assert _find_project_root(sub) == (repo if root_is_ancestor else sub)

    def test_agents_ancestor_wins_over_a_nearer_git_root(self, tmp_path: Path, fake_home: Path) -> None:
        """A .agents marker outranks a .git directory found closer to the start dir."""
        root = tmp_path / "proj"
        (root / ".agents").mkdir(parents=True)
        nested = root / "nested"
        (nested / ".git").mkdir(parents=True)
        start = nested / "pkg"
        start.mkdir()

        assert _find_project_root(start) == root

    def test_never_crosses_into_home(self, fake_home: Path) -> None:
        """A marker at the home directory is ignored; the search stops at that boundary."""
        (fake_home / ".git").mkdir()  # marker at home must not be selected
        sub = fake_home / "project"
        sub.mkdir()

        assert _find_project_root(sub) == sub


class TestSkillCopyMatches:
    """Unit tests for _skill_copy_matches."""

    @pytest.mark.parametrize("nested_reference", [False, True], ids=["flat", "with-references-dir"])
    def test_true_for_identical_trees(self, tmp_path: Path, nested_reference: bool) -> None:
        """Identical trees match, including every file of a nested references directory."""
        source = _make_bare_skill(tmp_path / "src", SKILL_NAMES[0])
        target = _make_bare_skill(tmp_path / "dst", SKILL_NAMES[0])
        if nested_reference:
            for root in (source, target):
                (root / "references").mkdir()
                (root / "references" / "plotting.md").write_text("# Plotting\n", encoding="utf-8")

        assert _skill_copy_matches(source, target) is True

    def test_false_when_target_is_not_a_directory(self, tmp_path: Path) -> None:
        """A non-directory target never matches."""
        source = _make_bare_skill(tmp_path / "src", SKILL_NAMES[0])
        target = tmp_path / SKILL_NAMES[0]
        target.write_text("# Retail metrics\n", encoding="utf-8")
        assert _skill_copy_matches(source, target) is False

    def test_false_when_file_sets_differ(self, tmp_path: Path) -> None:
        """A file present only in the source means no match."""
        source = _make_bare_skill(tmp_path / "src", SKILL_NAMES[0])
        (source / "references").mkdir()
        (source / "references" / "plotting.md").write_text("# Plotting\n", encoding="utf-8")
        target = _make_bare_skill(tmp_path / "dst", SKILL_NAMES[0])
        assert _skill_copy_matches(source, target) is False

    def test_false_when_bytes_differ(self, tmp_path: Path) -> None:
        """Same file names but different content means no match."""
        source = _make_bare_skill(tmp_path / "src", SKILL_NAMES[0], body=b"# Retail metrics\n")
        target = _make_bare_skill(tmp_path / "dst", SKILL_NAMES[0], body=b"# Retail metrics, revised\n")
        assert _skill_copy_matches(source, target) is False

    def test_false_on_file_versus_directory_mismatch(self, tmp_path: Path) -> None:
        """A path that is a file in the source but a directory in the target returns False, not raises."""
        source = _make_bare_skill(tmp_path / "src", SKILL_NAMES[0])
        (source / "references").write_text("not a directory", encoding="utf-8")
        target = _make_bare_skill(tmp_path / "dst", SKILL_NAMES[0])
        (target / "references").mkdir()
        assert _skill_copy_matches(source, target) is False


class TestIsOwnedTarget:
    """Unit tests for _is_owned_target."""

    def test_false_for_unbundled_name(self, tmp_path: Path) -> None:
        """A directory whose name is not a bundled skill is never owned."""
        other = _make_bare_skill(tmp_path, "my-own-skill")
        assert _is_owned_target(other, {SHIPPED_SKILL_NAME}) is False

    def test_true_for_symlink_with_bundled_name(self, tmp_path: Path) -> None:
        """A symlink named after a bundled skill is owned."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / SHIPPED_SKILL_NAME
        link.symlink_to(real, target_is_directory=True)
        assert _is_owned_target(link, {SHIPPED_SKILL_NAME}) is True

    def test_false_for_real_dir_without_marker(self, tmp_path: Path) -> None:
        """A real directory with a bundled name but no SKILL.md is user content, not ours."""
        directory = tmp_path / SHIPPED_SKILL_NAME
        directory.mkdir()
        assert _is_owned_target(directory, {SHIPPED_SKILL_NAME}) is False

    def test_true_for_real_dir_with_marker(self, tmp_path: Path) -> None:
        """A real directory with a bundled name and a SKILL.md is a prior copy install.

        This is the branch that authorizes deleting an existing tree.
        """
        prior_copy = _make_bare_skill(tmp_path, SHIPPED_SKILL_NAME)
        assert _is_owned_target(prior_copy, {SHIPPED_SKILL_NAME}) is True


class TestRelativeSymlinkTarget:
    """Unit tests for _relative_symlink_target."""

    def test_returns_a_relative_path_that_rejoins_to_the_source(self, tmp_path: Path) -> None:
        """The link target is relative, which is what lets the tree survive being relocated."""
        source = tmp_path / "site-packages" / "openretailscience" / SHIPPED_SKILL_NAME
        source.mkdir(parents=True)
        target = tmp_path / "project" / ".agents" / "skills" / SHIPPED_SKILL_NAME
        target.parent.mkdir(parents=True)

        result = _relative_symlink_target(source, target)

        assert not PurePath(result).is_absolute()
        assert os.path.realpath(target.parent / result) == os.path.realpath(source)

    @pytest.mark.parametrize("raiser", [_raise_value_error, _raise_oserror])
    def test_falls_back_to_absolute_source_on_relpath_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raiser: Callable[..., NoReturn]
    ) -> None:
        """A ValueError (cross-drive) or OSError while computing the relative path falls back."""
        source = tmp_path / "src"
        source.mkdir()
        (tmp_path / "dst").mkdir()
        target = tmp_path / "dst" / "link"
        monkeypatch.setattr(skills.os.path, "relpath", raiser)

        assert _relative_symlink_target(source, target) == os.path.realpath(source)


class TestInstallSkillsErrors:
    """Error paths for install_skills when the bundled source is missing or empty."""

    @pytest.mark.parametrize(
        ("source_exists", "expected_error", "match"),
        [
            pytest.param(False, FileNotFoundError, "Bundled skills directory", id="missing-source-dir"),
            pytest.param(True, RuntimeError, "No installable skills", id="empty-source-dir"),
        ],
    )
    def test_unusable_source_dir_raises(
        self,
        tmp_path: Path,
        fake_home: Path,
        monkeypatch: pytest.MonkeyPatch,
        source_exists: bool,
        expected_error: type[Exception],
        match: str,
    ) -> None:
        """A bundled skills directory that is missing, or present but empty, fails loudly."""
        source = tmp_path / "src" / ".agents" / "skills"
        if source_exists:
            source.mkdir(parents=True)
        monkeypatch.setattr(skills, "_get_source_skills_dir", lambda: source)

        with pytest.raises(expected_error, match=match):
            install_skills()


def _example_scripts_dir() -> Path:
    """Return the directory holding the shipped skill's example scripts."""
    return _get_source_skills_dir() / SHIPPED_SKILL_NAME / "scripts"


def _example_scripts() -> list[Path]:
    """Return every runnable example script bundled with the shipped skill."""
    return sorted(_example_scripts_dir().glob("example_*.py"))


class TestExampleScripts:
    """Every bundled example script must run against the installed package.

    This is the drift guard: a script demonstrating a renamed or removed API fails
    here instead of silently teaching an agent a broken pattern. It has to execute,
    because what the scripts hide (missing DataFrame columns, changed signatures)
    only surfaces at runtime.

    Scripts run in-process with runpy so the heavy openretailscience / matplotlib
    imports load once instead of per script.
    """

    @pytest.fixture(autouse=True)
    def _script_sandbox(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
        """Sandbox each in-process script run.

        All scripts share one interpreter, so no script may leak options into, or
        observe stray figures from, another. Options are restored to their prior
        values rather than library defaults, so a project or session configuration
        survives. Plots render headless into a temp directory, not the repo.
        """
        saved_options = {option: get_option(option) for option in list_options()}
        monkeypatch.setenv("MPLBACKEND", "Agg")
        monkeypatch.chdir(tmp_path)
        matplotlib = sys.modules.get("matplotlib")
        if matplotlib is not None:
            matplotlib.use("Agg", force=True)
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None:
            pyplot.close("all")
        yield
        pyplot = sys.modules.get("matplotlib.pyplot")
        if pyplot is not None:
            pyplot.close("all")
        for option, value in saved_options.items():
            set_option(option, value)

    def test_every_bundled_script_is_collected(self) -> None:
        """Every .py file shipped in scripts/ is picked up by the drift guard.

        The guard is parametrized over a glob, so a script renamed off the
        ``example_*`` pattern would drop out of the suite without failing anything.
        """
        collected = _example_scripts()

        assert len(collected) >= MIN_EXAMPLE_SCRIPTS
        assert {path.name for path in collected} == {path.name for path in _example_scripts_dir().glob("*.py")}

    @pytest.mark.parametrize("script", _example_scripts(), ids=lambda p: p.name)
    def test_example_script_runs(self, script: Path) -> None:
        """The example script runs end-to-end against the installed package."""
        namespace = runpy.run_path(str(script), run_name="__main__")

        # Catches a script reduced to its docstring: it runs clean and binds nothing.
        assert len([name for name in namespace if not name.startswith("__")]) > 0
