"""Tests for openretailscience.skills."""

from __future__ import annotations

import os
import re
import shutil
from typing import TYPE_CHECKING

import pytest

from openretailscience import skills
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
    from pathlib import Path

# Two realistic bundled-skill names plus a junk dir with no SKILL.md.
SKILL_NAMES = ("retail-metrics", "using-openretailscience")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)
SHIPPED_SKILL_NAME = "using-openretailscience"
# Fenced code blocks, and openretailscience import statements inside them.
CODE_FENCE_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.DOTALL)
IMPORT_RE = re.compile(r"^[ \t]*(?:from|import)\s+openretailscience[\w. ,]*(?:import[\w. ,]+)?$", re.MULTILINE)
# Markdown references to sibling skill files, e.g. `references/plotting.md`.
REFERENCE_RE = re.compile(r"references/[\w./-]+\.md")
MIN_REFERENCE_FILES = 3
MIN_IMPORT_EXAMPLES = 20
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


def _raise_oserror(*_args: object, **_kwargs: object) -> None:
    """Stand-in for os.symlink that reports symlinks are unsupported."""
    msg = "symlinks not supported"
    raise OSError(msg)


def _raise_eof(*_args: object, **_kwargs: object) -> str:
    """Stand-in for input() in a non-interactive session."""
    raise EOFError


def _fail_if_called(*_args: object, **_kwargs: object) -> str:
    """Stand-in for input() that fails if the prompt is ever shown."""
    msg = "input() must not be called when yes=True"
    raise AssertionError(msg)


def _raise_value_error(*_args: object, **_kwargs: object) -> str:
    """Stand-in for os.path.relpath that reports incompatible paths."""
    msg = "paths are on different drives"
    raise ValueError(msg)


def _raise_not_implemented(*_args: object, **_kwargs: object) -> None:
    """Stand-in for os.symlink on a platform without symlink support."""
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
    """Redirect ``Path.home()`` to a temp dir and clear Databricks detection."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
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


class TestProjectInstall:
    """Tests for project-mode (symlink) installation."""

    def test_creates_symlinks_resolving_to_source(self, source_dir: Path, project_dir: Path) -> None:
        """Project install symlinks each skill into .agents/skills back to the source."""
        result = install_skills(yes=True)

        target_dir = project_dir / ".agents" / "skills"
        for name in SKILL_NAMES:
            link = target_dir / name
            assert link.is_symlink()
            assert link.resolve() == (source_dir / name).resolve()
        assert len(result.installed) == len(SKILL_NAMES)

    @pytest.mark.parametrize("claude_present", [True, False])
    def test_claude_dir_targeted_only_when_claude_home_exists(
        self, source_dir: Path, project_dir: Path, fake_home: Path, claude_present: bool
    ) -> None:
        """The .claude/skills target is used only when ~/.claude exists."""
        if claude_present:
            (fake_home / ".claude").mkdir()

        install_skills(yes=True)

        claude_skill = project_dir / ".claude" / "skills" / SKILL_NAMES[0]
        assert claude_skill.is_symlink() is claude_present

    def test_rerun_is_idempotent(self, source_dir: Path, project_dir: Path) -> None:
        """Re-running reports all skills up to date and installs nothing new."""
        install_skills(yes=True)
        result = install_skills(yes=True)

        assert len(result.installed) == 0
        assert len(result.up_to_date) == len(SKILL_NAMES)
        for name in SKILL_NAMES:
            assert (project_dir / ".agents" / "skills" / name).is_symlink()

    def test_unrelated_existing_file_is_skipped_not_clobbered(self, source_dir: Path, project_dir: Path) -> None:
        """A pre-existing unrelated file at a target path is left untouched."""
        target_dir = project_dir / ".agents" / "skills"
        target_dir.mkdir(parents=True)
        conflict = target_dir / SKILL_NAMES[0]
        conflict.write_text("user data", encoding="utf-8")

        result = install_skills(yes=True)

        assert conflict.read_text(encoding="utf-8") == "user data"
        assert not conflict.is_symlink()
        assert len(result.skipped) == 1
        # The non-conflicting skill still installs.
        assert (target_dir / SKILL_NAMES[1]).is_symlink()

    def test_existing_real_skill_dir_is_skipped_in_symlink_mode(self, source_dir: Path, project_dir: Path) -> None:
        """A user's own real skill dir (same name, with SKILL.md) is skipped, not clobbered."""
        target_dir = project_dir / ".agents" / "skills"
        user_skill = target_dir / SKILL_NAMES[0]
        user_skill.mkdir(parents=True)
        (user_skill / "SKILL.md").write_text("my own skill", encoding="utf-8")

        result = install_skills(yes=True)

        assert user_skill.is_dir()
        assert not user_skill.is_symlink()
        assert (user_skill / "SKILL.md").read_text(encoding="utf-8") == "my own skill"
        assert str(user_skill.relative_to(project_dir)) in result.skipped
        # The non-conflicting skill still installs as a symlink.
        assert (target_dir / SKILL_NAMES[1]).is_symlink()

    def test_stale_symlink_is_repointed_to_source(self, source_dir: Path, project_dir: Path) -> None:
        """An owned symlink pointing at the wrong source is unlinked and reinstalled."""
        target_dir = project_dir / ".agents" / "skills"
        target_dir.mkdir(parents=True)
        stale_source = project_dir / "old-source"
        stale_source.mkdir()
        link = target_dir / SKILL_NAMES[0]
        link.symlink_to(stale_source, target_is_directory=True)

        result = install_skills(yes=True)

        assert link.is_symlink()
        assert link.resolve() == (source_dir / SKILL_NAMES[0]).resolve()
        assert str(link.relative_to(project_dir)) in result.installed

    def test_skipped_skill_leaves_no_empty_directory(self, source_dir: Path, project_dir: Path) -> None:
        """A skill skipped for a conflict does not create its own empty target dir."""
        target_dir = project_dir / ".agents" / "skills"
        target_dir.mkdir(parents=True)
        (target_dir / SKILL_NAMES[0]).write_text("user data", encoding="utf-8")

        install_skills(yes=True)

        # The conflicting target stays a file; no empty directory replaces it.
        assert (target_dir / SKILL_NAMES[0]).is_file()


class TestCopyFallback:
    """Tests for the copy fallback when symlinks are unsupported."""

    @pytest.mark.parametrize("raiser", [_raise_oserror, _raise_not_implemented])
    def test_copies_directory_when_symlink_unsupported(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch, raiser: object
    ) -> None:
        """When os.symlink raises OSError or NotImplementedError, the skill is copied instead."""
        monkeypatch.setattr(os, "symlink", raiser)

        install_skills(yes=True)

        for name in SKILL_NAMES:
            target = project_dir / ".agents" / "skills" / name
            assert target.is_dir()
            assert not target.is_symlink()
            assert (target / "SKILL.md").read_bytes() == (source_dir / name / "SKILL.md").read_bytes()


class TestGlobalInstall:
    """Tests for global-mode installation into home directories."""

    def test_targets_home_agents_dir(self, source_dir: Path, fake_home: Path) -> None:
        """Global install symlinks skills into ~/.agents/skills."""
        result = install_skills(global_mode=True, yes=True)

        for name in SKILL_NAMES:
            link = fake_home / ".agents" / "skills" / name
            assert link.is_symlink()
            assert link.resolve() == (source_dir / name).resolve()
        assert len(result.installed) == len(SKILL_NAMES)

    def test_targets_home_claude_dir_when_present(self, source_dir: Path, fake_home: Path) -> None:
        """Global install also targets ~/.claude/skills when Claude Code is present."""
        (fake_home / ".claude").mkdir()

        install_skills(global_mode=True, yes=True)

        assert (fake_home / ".claude" / "skills" / SKILL_NAMES[0]).is_symlink()


class TestDatabricksInstall:
    """Tests for the Databricks copy-to-Workspace branch."""

    @pytest.fixture
    def workspace_root(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        """Simulate a Databricks runtime with a redirected persistent workspace root."""
        root = tmp_path / "Workspace"
        root.mkdir()
        monkeypatch.setenv("DATABRICKS_RUNTIME_VERSION", "15.4")
        monkeypatch.delenv("DATABRICKS_USER", raising=False)
        monkeypatch.setattr(skills, "_DATABRICKS_WORKSPACE_ROOT", root)
        return root

    def test_project_mode_copies_to_shared_assistant_dir(self, source_dir: Path, workspace_root: Path) -> None:
        """On Databricks, project install copies (not links) into .assistant/skills."""
        install_skills(yes=True)

        for name in SKILL_NAMES:
            target = workspace_root / ".assistant" / "skills" / name
            assert target.is_dir()
            assert not target.is_symlink()
            assert (target / "SKILL.md").is_file()

    def test_rerun_reports_up_to_date_when_copy_matches(self, source_dir: Path, workspace_root: Path) -> None:
        """Re-running on Databricks with unchanged skills reports them up to date."""
        install_skills(yes=True)
        result = install_skills(yes=True)

        assert len(result.installed) == 0
        assert len(result.up_to_date) == len(SKILL_NAMES)

    def test_rerun_refreshes_copy_when_source_changed(self, source_dir: Path, workspace_root: Path) -> None:
        """A changed bundled skill is re-copied over the stale Databricks copy."""
        install_skills(yes=True)
        (source_dir / SKILL_NAMES[0] / "SKILL.md").write_text(
            f"---\nname: {SKILL_NAMES[0]}\ndescription: Updated.\n---\n\n# updated\n",
            encoding="utf-8",
        )

        result = install_skills(yes=True)

        target = workspace_root / ".assistant" / "skills" / SKILL_NAMES[0] / "SKILL.md"
        assert "Updated." in target.read_text(encoding="utf-8")
        assert len(result.installed) == 1

    def test_existing_workspace_skill_dir_is_replaced(self, source_dir: Path, workspace_root: Path) -> None:
        """In copy mode the managed Genie skills dir is refreshed, replacing an owned same-name dir.

        This is the documented asymmetry with symlink mode: the workspace
        ``.assistant/skills`` directory is installer-managed, so a same-named
        directory there is treated as a prior copy and overwritten on re-run.
        """
        existing = workspace_root / ".assistant" / "skills" / SKILL_NAMES[0]
        existing.mkdir(parents=True)
        (existing / "SKILL.md").write_text("stale copy", encoding="utf-8")

        install_skills(yes=True)

        assert existing.is_dir()
        assert (existing / "SKILL.md").read_bytes() == (source_dir / SKILL_NAMES[0] / "SKILL.md").read_bytes()

    def test_copy_is_independent_of_source(self, source_dir: Path, workspace_root: Path) -> None:
        """The copied skill survives the ephemeral package being wiped on restart."""
        install_skills(yes=True)
        expected = (source_dir / SKILL_NAMES[0] / "SKILL.md").read_bytes()

        # Simulate the ephemeral package being wiped on cluster restart.
        shutil.rmtree(source_dir)

        target = workspace_root / ".assistant" / "skills" / SKILL_NAMES[0] / "SKILL.md"
        assert target.read_bytes() == expected

    def test_global_mode_uses_per_user_dir_when_user_known(
        self, source_dir: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Global Databricks install targets the per-user workspace dir when known."""
        monkeypatch.setenv("DATABRICKS_USER", "analyst@retail.com")

        install_skills(global_mode=True, yes=True)

        target = workspace_root / "Users" / "analyst@retail.com" / ".assistant" / "skills" / SKILL_NAMES[0]
        assert target.is_dir()

    @pytest.mark.parametrize("user_env", [None, ""])
    def test_global_mode_falls_back_to_shared_when_user_missing_or_blank(
        self, source_dir: Path, workspace_root: Path, monkeypatch: pytest.MonkeyPatch, user_env: str | None
    ) -> None:
        """Global Databricks install falls back to the shared dir when the user is unset or blank."""
        if user_env is not None:
            monkeypatch.setenv("DATABRICKS_USER", user_env)

        install_skills(global_mode=True, yes=True)

        assert (workspace_root / ".assistant" / "skills" / SKILL_NAMES[0]).is_dir()
        # A blank user must not produce a malformed ``/Workspace/Users//...`` path.
        assert not (workspace_root / "Users").exists()


class TestConfirmation:
    """Tests for the confirmation prompt behavior."""

    def test_yes_true_does_not_prompt(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """yes=True installs without ever calling input()."""
        monkeypatch.setattr("builtins.input", _fail_if_called)

        result = install_skills(yes=True)

        assert len(result.installed) == len(SKILL_NAMES)

    def test_non_interactive_without_yes_raises(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-interactive session without yes=True raises a clear error."""
        monkeypatch.setattr("builtins.input", _raise_eof)

        with pytest.raises(RuntimeError, match="yes=True"):
            install_skills(yes=False)

    def test_declining_prompt_installs_nothing(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Answering no at the prompt installs nothing."""
        monkeypatch.setattr("builtins.input", lambda *_a, **_k: "n")

        result = install_skills(yes=False)

        assert len(result.installed) == 0
        assert not (project_dir / ".agents" / "skills" / SKILL_NAMES[0]).exists()

    @pytest.mark.parametrize("answer", ["", "y", "yes", "Y", "YES"])
    def test_accepting_prompt_installs_skills(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch, answer: str
    ) -> None:
        """Accepting the prompt (blank, y, or yes, case-insensitive) proceeds with the install."""
        monkeypatch.setattr("builtins.input", lambda *_a, **_k: answer)

        result = install_skills(yes=False)

        assert len(result.installed) == len(SKILL_NAMES)
        assert (project_dir / ".agents" / "skills" / SKILL_NAMES[0]).is_symlink()


class TestBundledSkill:
    """Tests validating the real skill shipped inside the package."""

    def test_shipped_skill_is_discoverable(self) -> None:
        """The real bundled skill is discovered from the installed package."""
        assert "using-openretailscience" in _discover_skills(_get_source_skills_dir())

    def test_shipped_skill_frontmatter_is_valid(self) -> None:
        """The shipped SKILL.md has YAML frontmatter with matching name and a description."""
        skill_md = _get_source_skills_dir() / "using-openretailscience" / "SKILL.md"
        text = skill_md.read_text(encoding="utf-8")

        match = FRONTMATTER_RE.match(text)
        assert match is not None, "SKILL.md must start with YAML frontmatter"
        block = match.group(1)

        name_match = re.search(r"^name:\s*(\S+)", block, re.MULTILINE)
        assert name_match is not None
        assert name_match.group(1) == "using-openretailscience"

        assert "description:" in block
        # The description is a YAML folded scalar (``>-``) with its body on the
        # following wrapped lines; assert that body is substantive, not just the
        # ``>-`` indicator.
        description_body = block.split("description:", 1)[1].replace(">-", " ").strip()
        assert len(description_body) >= MIN_DESCRIPTION_LENGTH

    def test_referenced_files_exist(self) -> None:
        """Every references/*.md path the shipped skill points at exists on disk."""
        skill_root = _get_source_skills_dir() / SHIPPED_SKILL_NAME
        referenced: set[str] = set()
        for md_file in _shipped_skill_markdown():
            referenced.update(REFERENCE_RE.findall(md_file.read_text(encoding="utf-8")))

        assert len(referenced) >= MIN_REFERENCE_FILES, "SKILL.md should link to its reference files"
        for rel in sorted(referenced):
            assert (skill_root / rel).is_file(), f"skill references a missing file: {rel}"

    def test_import_examples_resolve_against_the_package(self) -> None:
        """Every openretailscience import the skill teaches still resolves.

        This is the content-drift guard: renaming or removing a module or public
        symbol the skill documents breaks its import example and fails here,
        forcing the skill to be updated alongside the API change.
        """
        statements = _skill_import_statements()
        assert len(statements) >= MIN_IMPORT_EXAMPLES, "skill should teach many concrete imports"
        failures = [msg for statement in statements if (msg := _import_error(statement)) is not None]
        assert not failures, "skill imports no longer resolve:\n" + "\n".join(failures)


class TestFindProjectRoot:
    """Unit tests for _find_project_root."""

    def test_returns_git_root_from_subdirectory(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A .git ancestor is used as the project root."""
        (tmp_path / "home").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        repo = tmp_path / "repo"
        (repo / ".git").mkdir(parents=True)
        sub = repo / "pkg"
        sub.mkdir()

        assert _find_project_root(sub) == repo

    def test_agents_ancestor_wins_over_git_walk(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """An existing .agents dir on an ancestor is preferred as the root."""
        (tmp_path / "home").mkdir()
        monkeypatch.setenv("HOME", str(tmp_path / "home"))
        root = tmp_path / "proj"
        (root / ".agents").mkdir(parents=True)
        sub = root / "nested"
        sub.mkdir()

        assert _find_project_root(sub) == root

    def test_falls_back_to_start_dir_without_markers(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no .agents/.claude/.git found, the start dir itself is returned."""
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setenv("HOME", str(home))
        start = tmp_path / "loose"
        start.mkdir()

        assert _find_project_root(start) == start

    def test_never_crosses_into_home(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A marker at the home directory is ignored; the search stops at that boundary."""
        home = tmp_path / "home"
        (home / ".git").mkdir(parents=True)  # marker at home must not be selected
        monkeypatch.setenv("HOME", str(home))
        sub = home / "project"
        sub.mkdir()

        assert _find_project_root(sub) == sub


class TestSkillCopyMatches:
    """Unit tests for _skill_copy_matches."""

    def test_true_for_identical_trees(self, tmp_path: Path) -> None:
        """Two directories with the same files and bytes match."""
        source = _make_bare_skill(tmp_path / "src", "s")
        target = _make_bare_skill(tmp_path / "dst", "s")
        assert _skill_copy_matches(source, target) is True

    def test_true_for_identical_multi_file_trees(self, tmp_path: Path) -> None:
        """A match iterates every file (including a nested references dir) and returns True."""
        source = _make_bare_skill(tmp_path / "src", "s")
        target = _make_bare_skill(tmp_path / "dst", "s")
        for root in (source, target):
            (root / "references").mkdir()
            (root / "references" / "guide.md").write_text("shared body", encoding="utf-8")
        assert _skill_copy_matches(source, target) is True

    def test_false_when_target_is_not_a_directory(self, tmp_path: Path) -> None:
        """A non-directory target never matches."""
        source = _make_bare_skill(tmp_path / "src", "s")
        target = tmp_path / "a-file"
        target.write_text("x", encoding="utf-8")
        assert _skill_copy_matches(source, target) is False

    def test_false_when_file_sets_differ(self, tmp_path: Path) -> None:
        """Different relative file sets do not match."""
        source = _make_bare_skill(tmp_path / "src", "s")
        (source / "extra.md").write_text("more", encoding="utf-8")
        target = _make_bare_skill(tmp_path / "dst", "s")
        assert _skill_copy_matches(source, target) is False

    def test_false_when_bytes_differ(self, tmp_path: Path) -> None:
        """Same file names but different bytes do not match."""
        source = _make_bare_skill(tmp_path / "src", "s", body=b"one")
        target = _make_bare_skill(tmp_path / "dst", "s", body=b"two")
        assert _skill_copy_matches(source, target) is False

    def test_false_on_file_versus_directory_mismatch(self, tmp_path: Path) -> None:
        """A path that is a file in source but a directory in target returns False, not raises."""
        source = _make_bare_skill(tmp_path / "src", "s")
        (source / "refs").write_text("a file", encoding="utf-8")
        target = _make_bare_skill(tmp_path / "dst", "s")
        (target / "refs").mkdir()
        assert _skill_copy_matches(source, target) is False


class TestIsOwnedTarget:
    """Unit tests for _is_owned_target."""

    def test_false_for_unbundled_name(self, tmp_path: Path) -> None:
        """A directory whose name is not a bundled skill is never owned."""
        other = _make_bare_skill(tmp_path, "other-skill")
        assert _is_owned_target(other, {"using-openretailscience"}) is False

    def test_true_for_symlink_with_bundled_name(self, tmp_path: Path) -> None:
        """A symlink named after a bundled skill is owned."""
        real = tmp_path / "real"
        real.mkdir()
        link = tmp_path / "using-openretailscience"
        link.symlink_to(real, target_is_directory=True)
        assert _is_owned_target(link, {"using-openretailscience"}) is True

    def test_false_for_real_dir_without_marker(self, tmp_path: Path) -> None:
        """A real directory with a bundled name but no SKILL.md is not owned."""
        directory = tmp_path / "using-openretailscience"
        directory.mkdir()
        assert _is_owned_target(directory, {"using-openretailscience"}) is False


class TestRelativeSymlinkTarget:
    """Unit tests for _relative_symlink_target."""

    @pytest.mark.parametrize("raiser", [_raise_value_error, _raise_oserror])
    def test_falls_back_to_absolute_source_on_relpath_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raiser: object
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

    def test_missing_source_dir_raises(self, tmp_path: Path, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A missing bundled skills directory raises FileNotFoundError."""
        missing = tmp_path / "nope" / ".agents" / "skills"
        monkeypatch.setattr(skills, "_get_source_skills_dir", lambda: missing)

        with pytest.raises(FileNotFoundError, match="Bundled skills directory"):
            install_skills(yes=True)

    def test_empty_source_dir_raises(self, tmp_path: Path, fake_home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """A bundled skills directory with no valid skills raises FileNotFoundError."""
        empty = tmp_path / "src" / ".agents" / "skills"
        empty.mkdir(parents=True)
        monkeypatch.setattr(skills, "_get_source_skills_dir", lambda: empty)

        with pytest.raises(FileNotFoundError, match="No installable skills"):
            install_skills(yes=True)
