# Agent Skills

OpenRetailScience ships an **agent skill**: a folder of guidance that AI coding
agents (Claude Code, Databricks Genie, and other harnesses that read the
`.agents`/`.claude` skill directories) load to write correct, idiomatic
OpenRetailScience code. Installing the skill points your agent at guidance that
is versioned with the package, so it stays accurate as you upgrade.

Install the skill by calling a Python function:

```python
from openretailscience.skills import install_skills

install_skills()
```

## What gets installed

The package bundles its skills under its own install tree. `install_skills()`
creates links to them in the directories your agent reads from, so the skill and
the installed package never drift apart:

- **Project mode** (default) links into the current project's
  `.agents/skills/using-openretailscience/`, and into
  `.claude/skills/using-openretailscience/` when Claude Code is detected (a
  `~/.claude` directory exists).
- **Global mode** links into the equivalent directories under your home folder.

Each skill installs as a subfolder named after itself, so the bundled skill
lands at `using-openretailscience/` inside whichever skills directory applies.

Because the entries are symlinks back into the installed package, a
`pip install -U openretailscience` updates the skill in place, so there is nothing
to reinstall.

```python
install_skills()                       # project install
install_skills(global_mode=True)       # install for all your projects
```

The operation is **idempotent**: re-running it re-uses existing links and never
overwrites unrelated files or real directories you already have in those
directories. It does repoint a **symlink** that shares the bundled skill's name
(`using-openretailscience`), treating such a link as its own so it can refresh
one left by an earlier install.

!!! note "Symlinks are environment-specific"
    Project-mode links point into the Python environment that installed the
    package, so they generally should not be committed to source control. Add the
    installed skill paths (for example `.agents/skills/using-openretailscience/`)
    to your `.gitignore`.

## Databricks

Databricks works differently, and `install_skills()` adapts
automatically. A pip-installed package lives on the cluster's ephemeral storage,
which is wiped on restart, and Databricks Genie reads skills from a persistent
workspace location that cannot symlink back into that storage. So on Databricks
the installer **copies** the skill into the Genie skills directory instead of
linking it:

- Project mode → `/Workspace/.assistant/skills/`
- Global mode → `/Workspace/Users/<you>/.assistant/skills/` (falling back to the
  shared workspace directory when the user cannot be determined)

Run it from a notebook, or from a cluster init script so it re-applies on every
start:

```python
from openretailscience.skills import install_skills

install_skills()
```

Because this is a copy rather than a link, it does **not** update itself when you
upgrade the package. Re-run `install_skills()` after upgrading OpenRetailScience
(a cluster init script is the easiest way to keep it current).

!!! warning "Managed directory"
    The workspace `.assistant/skills/` directory is treated as installer-managed:
    a folder there whose name matches a bundled skill is refreshed on every run.
    If you hand-author your own skill under that directory, give it a different
    name so `install_skills()` does not overwrite it.

## Keeping the skill current

| Environment | Install method | Updates on `pip install -U`? |
| --- | --- | --- |
| Local / project | symlink | Yes, automatically |
| Global (home) | symlink | Yes, automatically |
| Databricks | copy | No, re-run `install_skills()` |

The skill's content is maintained alongside the codebase and validated in CI, so
each release ships guidance that matches that version's public API.
