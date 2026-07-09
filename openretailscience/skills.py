"""Install bundled agent skills into well-known agent skill directories.

This module ships the package's agent skills (folders containing a ``SKILL.md``
with YAML frontmatter) under ``openretailscience/.agents/skills/`` and links them
into the directories that AI coding agents read from, so upgrading the package
propagates updated skills automatically.

The public entry point is :func:`install_skills`. There is no CLI:

    from openretailscience.skills import install_skills
    install_skills()

Linking behaviour, modeled on Streamlit's ``streamlit skills`` command:

* On POSIX (and Windows with symlink support) skills are **symlinked** into the
  target directories, so a ``pip install -U`` of the package updates them in
  place.
* When symlinks are unavailable (e.g. Windows without Developer Mode) the skill
  folder is **copied** instead.
* On Databricks the package lives on ephemeral compute that is wiped on cluster
  restart, and Genie reads skills from a persistent ``/Workspace`` location that
  cannot symlink into site-packages. There we always **copy** skills into the
  Genie skills directory; re-run :func:`install_skills` after upgrading the
  package to refresh the copy.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SKILL_MARKER_FILENAME = "SKILL.md"
AGENTS_DIR_NAME = ".agents"
CLAUDE_DIR_NAME = ".claude"
SKILLS_SUBDIR = "skills"

# Databricks Genie / Assistant reads skills from ``<root>/.assistant/skills``.
DATABRICKS_ASSISTANT_DIR = ".assistant"
DATABRICKS_USERS_DIR = "Users"
_DATABRICKS_RUNTIME_ENV = "DATABRICKS_RUNTIME_VERSION"
_DATABRICKS_USER_ENV = "DATABRICKS_USER"
# Persistent workspace root on Databricks. Kept as a module constant so tests
# can redirect it away from the real ``/Workspace`` mount.
_DATABRICKS_WORKSPACE_ROOT = Path("/Workspace")


@dataclass
class SkillInstallResult:
    """Outcome of an :func:`install_skills` call.

    Attributes:
        installed (list[str]): Target paths that were newly linked or copied.
        up_to_date (list[str]): Target paths that already pointed at the source.
        skipped (list[str]): Target paths left untouched due to a conflicting
            file or directory the installer does not own.
    """

    installed: list[str] = field(default_factory=list)
    up_to_date: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)


def _get_source_skills_dir() -> Path:
    """Return the path to the skills bundled inside the installed package.

    Returns:
        Path: The ``openretailscience/.agents/skills`` directory.
    """
    return Path(__file__).parent / AGENTS_DIR_NAME / SKILLS_SUBDIR


def _discover_skills(source_dir: Path) -> list[str]:
    """List installable skill names in ``source_dir``.

    A valid skill is a subdirectory containing a ``SKILL.md`` file.

    Args:
        source_dir (Path): Directory holding skill folders.

    Returns:
        list[str]: Sorted skill folder names.
    """
    if not source_dir.is_dir():
        return []
    return [
        entry.name
        for entry in sorted(source_dir.iterdir())
        if entry.is_dir() and (entry / SKILL_MARKER_FILENAME).is_file()
    ]


def _find_project_root(start: Path | None = None) -> Path:
    """Resolve the project root to install project-mode skills into.

    Walks up from ``start`` looking first for an existing ``.agents``/``.claude``
    directory, then for a ``.git`` directory, never crossing into the home
    directory. Falls back to ``start`` itself.

    Args:
        start (Path | None): Directory to search from. Defaults to the current
            working directory.

    Returns:
        Path: The resolved project root.
    """
    start_dir = (start or Path.cwd()).resolve()
    home = Path.home().resolve()

    for parent in [start_dir, *start_dir.parents]:
        if parent == home:
            break
        if (parent / AGENTS_DIR_NAME).is_dir() or (parent / CLAUDE_DIR_NAME).is_dir():
            return parent

    for parent in [start_dir, *start_dir.parents]:
        if parent == home:
            break
        if (parent / ".git").exists():
            return parent

    return start_dir


def _skills_dir(base: Path, harness_dir: str) -> Path:
    """Return the ``<base>/<harness_dir>/skills`` path.

    Args:
        base (Path): Project root or home directory.
        harness_dir (str): Agent config directory name (e.g. ``.agents``).

    Returns:
        Path: The skills directory for that harness.
    """
    return base / harness_dir / SKILLS_SUBDIR


def _get_project_target_dirs(project_root: Path) -> list[Path]:
    """Return project-mode target directories.

    Always targets ``<root>/.agents/skills``; adds ``<root>/.claude/skills`` when
    Claude Code is detected (a ``~/.claude`` directory exists).

    Args:
        project_root (Path): The resolved project root.

    Returns:
        list[Path]: Target skill directories.
    """
    targets = [_skills_dir(project_root, AGENTS_DIR_NAME)]
    if (Path.home() / CLAUDE_DIR_NAME).is_dir():
        targets.append(_skills_dir(project_root, CLAUDE_DIR_NAME))
    return targets


def _get_global_target_dirs() -> list[Path]:
    """Return global-mode target directories under the home directory.

    Always targets ``~/.agents/skills``; adds ``~/.claude/skills`` when Claude
    Code is detected.

    Returns:
        list[Path]: Target skill directories.
    """
    home = Path.home()
    targets = [_skills_dir(home, AGENTS_DIR_NAME)]
    if (home / CLAUDE_DIR_NAME).is_dir():
        targets.append(_skills_dir(home, CLAUDE_DIR_NAME))
    return targets


def _is_databricks() -> bool:
    """Return whether the current process runs on Databricks compute.

    Returns:
        bool: True when the Databricks runtime environment variable is set.
    """
    return _DATABRICKS_RUNTIME_ENV in os.environ


def _databricks_user() -> str | None:
    """Best-effort lookup of the current Databricks workspace user.

    Returns:
        str | None: The user identifier, or None when it cannot be determined.
    """
    value = os.environ.get(_DATABRICKS_USER_ENV)
    return value if value is not None and len(value) > 0 else None


def _get_databricks_target_dirs(*, global_mode: bool) -> list[Path]:
    """Return the persistent Databricks Genie skills directory.

    Project mode installs to the shared workspace skills directory; global mode
    installs to the per-user directory when the user is known, otherwise falls
    back to the shared directory with a note.

    Args:
        global_mode (bool): Whether to install per-user rather than shared.

    Returns:
        list[Path]: The single Genie skills directory to copy into.
    """
    if global_mode:
        user = _databricks_user()
        if user is not None:
            base = _DATABRICKS_WORKSPACE_ROOT / DATABRICKS_USERS_DIR / user
            return [_skills_dir(base, DATABRICKS_ASSISTANT_DIR)]
        print(  # noqa: T201 - user-facing installer output
            "Could not determine the Databricks user; installing to the shared workspace skills directory instead."
        )
    return [_skills_dir(_DATABRICKS_WORKSPACE_ROOT, DATABRICKS_ASSISTANT_DIR)]


def _relative_symlink_target(source_path: Path, target_path: Path) -> str:
    """Compute the symlink target for ``source_path`` from ``target_path``.

    Uses ``realpath``-resolved endpoints so the ``..`` count matches the physical
    layout even when an ancestor is itself a symlink (macOS ``/var``, bind
    mounts). Falls back to the absolute real path when a relative path cannot be
    computed (e.g. different Windows drives).

    Args:
        source_path (Path): The bundled skill directory the link points to.
        target_path (Path): The link path being created.

    Returns:
        str: The (relative or absolute) symlink target.
    """
    try:
        return os.path.relpath(os.path.realpath(source_path), os.path.realpath(target_path.parent))
    except (ValueError, OSError):
        return os.path.realpath(source_path)


def _try_symlink(source_path: Path, target_path: Path) -> bool:
    """Attempt to create a directory symlink at ``target_path``.

    Args:
        source_path (Path): The bundled skill directory.
        target_path (Path): The link to create (must not already exist).

    Returns:
        bool: True on success; False when symlinks are unsupported.
    """
    rel_source = _relative_symlink_target(source_path, target_path)
    try:
        target_path.symlink_to(rel_source, target_is_directory=True)
    except (OSError, NotImplementedError):
        return False
    return True


def _skill_copy_matches(source_path: Path, target_path: Path) -> bool:
    """Return whether a copied skill directory matches the source byte-for-byte.

    Args:
        source_path (Path): The bundled skill directory.
        target_path (Path): The existing target directory.

    Returns:
        bool: True when both hold the same relative files with identical bytes.
    """
    if not target_path.is_dir():
        return False
    source_files = sorted(p.relative_to(source_path) for p in source_path.rglob("*"))
    if source_files != sorted(p.relative_to(target_path) for p in target_path.rglob("*")):
        return False
    for rel in source_files:
        source_file = source_path / rel
        if source_file.is_dir():
            continue
        if (target_path / rel).read_bytes() != source_file.read_bytes():
            return False
    return True


def _is_owned_target(target_path: Path, bundled_names: set[str]) -> bool:
    """Return whether the installer may safely replace ``target_path``.

    A target is owned when it is a symlink whose name matches a bundled skill, or
    a real directory named after a bundled skill that itself contains a
    ``SKILL.md`` (a prior copy install). Anything else is user content.

    Args:
        target_path (Path): The candidate target.
        bundled_names (set[str]): Names of the skills being installed.

    Returns:
        bool: True when the target may be cleared.
    """
    if target_path.name not in bundled_names:
        return False
    if target_path.is_symlink():
        return True
    return target_path.is_dir() and (target_path / SKILL_MARKER_FILENAME).is_file()


def _prepare_target(source_path: Path, target_path: Path, bundled_names: set[str], *, use_copy: bool) -> str:
    """Clear or evaluate an existing target before installing.

    Args:
        source_path (Path): The bundled skill directory.
        target_path (Path): Where the skill will be installed.
        bundled_names (set[str]): Names of the skills being installed.
        use_copy (bool): Whether the install method is copy (vs. symlink).

    Returns:
        str: One of ``"install"``, ``"up_to_date"``, or ``"skip"``.
    """
    if not target_path.exists() and not target_path.is_symlink():
        return "install"

    if not _is_owned_target(target_path, bundled_names):
        return "skip"

    if target_path.is_symlink():
        if not use_copy:
            try:
                if target_path.resolve() == source_path.resolve():
                    return "up_to_date"
            except OSError:
                pass
        target_path.unlink()  # os.unlink semantics: remove the link, not its target
        return "install"

    # Owned real directory (a previous copy install).
    if use_copy and _skill_copy_matches(source_path, target_path):
        return "up_to_date"
    shutil.rmtree(target_path)
    return "install"


def _display_label(target_path: Path) -> str:
    """Return a concise label for ``target_path`` relative to the cwd.

    Args:
        target_path (Path): The installed target path.

    Returns:
        str: A cwd-relative path when possible, else the absolute path.
    """
    try:
        return str(target_path.relative_to(Path.cwd()))
    except ValueError:
        return str(target_path)


def _install_one(
    skill_name: str,
    source_dir: Path,
    target_dir: Path,
    result: SkillInstallResult,
    bundled_names: set[str],
    *,
    use_copy: bool,
) -> None:
    """Install a single skill into ``target_dir`` and record the outcome.

    Args:
        skill_name (str): Name of the skill folder.
        source_dir (Path): Directory holding the bundled skills.
        target_dir (Path): Destination skills directory.
        result (SkillInstallResult): Accumulator updated in place.
        bundled_names (set[str]): Names of the skills being installed.
        use_copy (bool): Force copying instead of symlinking (Databricks).
    """
    source_path = source_dir / skill_name
    target_path = target_dir / skill_name
    label = _display_label(target_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    disposition = _prepare_target(source_path, target_path, bundled_names, use_copy=use_copy)
    if disposition == "up_to_date":
        result.up_to_date.append(label)
        return
    if disposition == "skip":
        result.skipped.append(label)
        return

    if use_copy or not _try_symlink(source_path, target_path):
        shutil.copytree(source_path, target_path)
    result.installed.append(label)


def _print_plan(source_dir: Path, skills: list[str], target_dirs: list[Path], *, use_copy: bool) -> None:
    """Print the installation plan for interactive confirmation.

    Args:
        source_dir (Path): Bundled skills source directory.
        skills (list[str]): Skills to install.
        target_dirs (list[Path]): Destination directories.
        use_copy (bool): Whether skills will be copied rather than symlinked.
    """
    method = "Copying" if use_copy else "Linking"
    print(f"{method} {len(skills)} skill(s) from {source_dir}:")  # noqa: T201
    for skill in skills:
        print(f"  - {skill}")  # noqa: T201
    print("Into:")  # noqa: T201
    for target_dir in target_dirs:
        print(f"  - {target_dir}")  # noqa: T201


def _confirm(source_dir: Path, skills: list[str], target_dirs: list[Path], *, use_copy: bool) -> bool:
    """Show the plan and prompt for confirmation.

    Args:
        source_dir (Path): Bundled skills source directory.
        skills (list[str]): Skills to install.
        target_dirs (list[Path]): Destination directories.
        use_copy (bool): Whether skills will be copied rather than symlinked.

    Returns:
        bool: True when the user accepts installation.

    Raises:
        RuntimeError: When run non-interactively (stdin is closed).
    """
    _print_plan(source_dir, skills, target_dirs, use_copy=use_copy)
    try:
        answer = input("Proceed with installation? [Y/n] ").strip().lower()
    except EOFError:
        msg = "Non-interactive session detected. Pass yes=True to install_skills() to skip confirmation."
        raise RuntimeError(msg) from None
    return answer in {"", "y", "yes"}


def _print_result(result: SkillInstallResult) -> None:
    """Print a summary of the installation outcome.

    Args:
        result (SkillInstallResult): The completed install result.
    """
    for label, paths in (
        ("Installed", result.installed),
        ("Already up to date", result.up_to_date),
        ("Skipped (conflicting file)", result.skipped),
    ):
        for path in paths:
            print(f"{label}: {path}")  # noqa: T201


def install_skills(global_mode: bool = False, yes: bool = False) -> SkillInstallResult:
    """Install the package's bundled agent skills.

    In project mode (default) skills are linked into the current project's
    ``.agents/skills/`` (and ``.claude/skills/`` when Claude Code is detected). In
    global mode they are linked into the equivalent home directories. On
    Databricks, skills are copied into the persistent Genie workspace skills
    directory instead of linked. The operation is idempotent.

    Args:
        global_mode (bool): Install to user-level home directories instead of the
            current project. Defaults to False.
        yes (bool): Skip the interactive confirmation prompt. Defaults to False.

    Returns:
        SkillInstallResult: The skills that were installed, already up to date,
        or skipped. Empty when the user declines the confirmation.

    Raises:
        FileNotFoundError: When the bundled skills directory is missing or empty.
        RuntimeError: When confirmation is required but stdin is non-interactive.
    """
    source_dir = _get_source_skills_dir()
    if not source_dir.is_dir():
        msg = f"Bundled skills directory not found: {source_dir}"
        raise FileNotFoundError(msg)

    skills = _discover_skills(source_dir)
    if len(skills) == 0:
        msg = "No installable skills found in the openretailscience package."
        raise FileNotFoundError(msg)

    if _is_databricks():
        target_dirs = _get_databricks_target_dirs(global_mode=global_mode)
        use_copy = True
    elif global_mode:
        target_dirs = _get_global_target_dirs()
        use_copy = False
    else:
        target_dirs = _get_project_target_dirs(_find_project_root())
        use_copy = False

    if not yes and not _confirm(source_dir, skills, target_dirs, use_copy=use_copy):
        print("Installation cancelled.")  # noqa: T201
        return SkillInstallResult()

    result = SkillInstallResult()
    bundled_names = set(skills)
    for skill_name in skills:
        for target_dir in target_dirs:
            _install_one(skill_name, source_dir, target_dir, result, bundled_names, use_copy=use_copy)

    _print_result(result)
    return result
