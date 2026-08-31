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
  Not available on Databricks (see below).

Each skill installs as a subfolder named after itself, so the bundled skill
lands at `using-openretailscience/` inside whichever skills directory applies.

Outside Databricks the entries are symlinks back into the installed package, so a
`pip install -U openretailscience` updates the skill in place and there is nothing
to reinstall.

```python
install_skills()                       # project install
install_skills(global_mode=True)       # install for all your projects (not on Databricks)
```

The operation is **idempotent**: re-running it re-uses existing links and never
overwrites unrelated files or real directories you already have in those
directories. It does repoint a **symlink** that shares the bundled skill's name
(`using-openretailscience`), treating such a link as its own so it can refresh
one left by an earlier install.

### Committing the links

A symlink points at the installed package by the shortest relative path that
reaches it, so whether it belongs in source control depends on where your
environment lives.

With a **repo-local virtualenv** — `uv`'s default, and what you get from
`python -m venv .venv` or Poetry's `virtualenvs.in-project = true` — the target
stays inside the repo:

<!-- markdownlint-disable MD013 -->

```text
.agents/skills/using-openretailscience -> ../../.venv/lib/python3.12/site-packages/openretailscience/.agents/skills/using-openretailscience
```

<!-- markdownlint-enable MD013 -->

Commit that, and a fresh clone has the skill without anyone remembering to run the
installer; the link is broken until dependencies are installed, then resolves. It
keeps resolving only where a checkout reproduces that exact path, which needs both:

- **A pinned Python.** The target names the interpreter version (`lib/python3.12/`),
  so without a `.python-version` file — or a CI setup that matches it — any checkout
  resolving a different interpreter dangles.
- **One platform.** POSIX virtualenvs use `lib/pythonX.Y/site-packages`; Windows uses
  `Lib/site-packages`, with no version segment at all. A committed link resolves on
  one or the other, never both, so a mixed-OS team should not commit it even with
  Python pinned.

A dangling skill link fails silently: the agent simply does not load the skill.

With an environment **outside the project tree** — a global install, conda,
`uv tool`, or Poetry's default cache location — the target escapes the repo as a
long `../../..` chain, or as an absolute path when no relative path exists at all
(different Windows drives). Those resolve only on the machine that installed them,
so add the installed skill paths (for example
`.agents/skills/using-openretailscience/`) to your `.gitignore`.

Where symlinks are unavailable the installer copies the skill instead. A copy has
none of this to weigh, but committing one vendors a snapshot that no longer tracks
the installed package.

## Installing with the library-skills CLI

The bundled skill also follows the
[Library Skills](https://library-skills.io){target="_blank" rel="noopener"}
convention: libraries publish their skills under `<package>/.agents/skills/`, and
one CLI wires them into any project that depends on them. OpenRetailScience ships
that layout already, so the tool needs no extra configuration:

```bash
uvx library-skills
```

It reads your project's dependencies, finds `using-openretailscience` inside the
installed package, and offers to symlink it into `.agents/skills/`. The `--claude`
flag adds `.claude/skills/`, which Claude Code needs because it does not read
`.agents`. Run the command again after your dependencies change and it repairs
stale links; `--copy` writes real files where symlinks are unavailable.

The command prompts for what to install, so a scripted setup names the skill and
skips the prompts: `uvx library-skills -s using-openretailscience -y`.

## Databricks

Databricks works differently, and the default `install_skills()` adapts
automatically. A pip-installed package lives on the cluster's ephemeral storage,
which is wiped on restart, and Databricks Genie reads skills from a persistent
workspace location that cannot symlink back into that storage. So on Databricks
the installer **copies** the skill into the Genie skills directory instead of
linking it.

Skills go to `/Workspace/Users/<you>/.assistant/skills/`, your own workspace home.
`global_mode=True` raises `NotImplementedError` there: the workspace-wide
`/Workspace/.assistant/skills/` needs admin rights, and a single shared copy
cannot match the package version each user has installed.

Run it from a notebook, as yourself. Your workspace home comes from the Spark
session's `current_user()`, so cases like these raise rather than install: a
context with no session (a cluster init script, a bare Python job), and an identity
with no workspace home of its own, such as a job running as a service principal.

```python
from openretailscience.skills import install_skills

install_skills()
```

Because this is a copy rather than a link, it does **not** update itself when you
upgrade the package: re-run `install_skills()` after upgrading OpenRetailScience.

!!! warning "Managed directory"
    Your workspace `.assistant/skills/` directory is treated as installer-managed:
    a folder there whose name matches a bundled skill is refreshed on every run.
    If you hand-author your own skill under that directory, give it a different
    name so `install_skills()` does not overwrite it.

## Keeping the skill current

| Environment | Install method | Updates on `pip install -U`? |
| --- | --- | --- |
| Local / project | symlink | Yes, automatically |
| Global (home) | symlink | Yes, automatically |
| Databricks (no global mode) | copy | No, re-run `install_skills()` |

The skill's content is maintained alongside the codebase and validated in CI, so
each release ships guidance that matches that version's public API.
