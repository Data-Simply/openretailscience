"""Tests for openretailscience.skills."""

from __future__ import annotations

import os
import re
import shutil
from typing import TYPE_CHECKING

import pytest

from openretailscience import skills
from openretailscience.skills import (
    SkillInstallResult,
    _discover_skills,
    _get_source_skills_dir,
    install_skills,
)

if TYPE_CHECKING:
    from pathlib import Path

# Two realistic bundled-skill names plus a junk dir with no SKILL.md.
SKILL_NAMES = ("retail-metrics", "using-openretailscience")
FRONTMATTER_RE = re.compile(r"\A---\n(.*?)\n---", re.DOTALL)


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
    """A project working directory (cwd) with no agent markers yet."""
    project = tmp_path / "retail-project"
    project.mkdir()
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


class TestCopyFallback:
    """Tests for the copy fallback when symlinks are unsupported."""

    def test_copies_directory_when_symlink_raises(
        self, source_dir: Path, project_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When os.symlink fails, the skill is copied as a real directory."""
        monkeypatch.setattr(os, "symlink", _raise_oserror)

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

    def test_global_mode_falls_back_to_shared_when_user_unknown(self, source_dir: Path, workspace_root: Path) -> None:
        """Global Databricks install falls back to the shared dir when user unknown."""
        install_skills(global_mode=True, yes=True)

        assert (workspace_root / ".assistant" / "skills" / SKILL_NAMES[0]).is_dir()


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

        description_match = re.search(r"^description:\s*(\S.*)", block, re.MULTILINE)
        assert description_match is not None
        assert len(description_match.group(1).strip()) > 0

    def test_result_is_skill_install_result(self, source_dir: Path, project_dir: Path) -> None:
        """install_skills returns a SkillInstallResult."""
        result = install_skills(yes=True)
        assert isinstance(result, SkillInstallResult)
