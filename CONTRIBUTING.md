# Welcome Contributors

Hi there! Thank you for even being interested in contributing to OpenRetailScience.
As an open-source project in a rapidly developing field, we are extremely open to contributions—
whether they involve new features, improved infrastructure, better documentation, or bug fixes.

## Guidelines

### Ways to contribute

There are many ways to contribute to OpenRetailScience. Here are some common ways people contribute:

* Documentation: Help improve our docs, including this one!
* Code: Help us write code, fix bugs, or improve our infrastructure.
* Integrations: Help us integrate with your favorite vendors and tools.
* Discussions: Help answer usage questions and discuss issues with users.

### GitHub Issues

Our issues page is kept up to date with bugs, improvements, and feature requests.

If you start working on an issue, please assign it to yourself.

If you are adding an issue, please try to keep it focused on a single, modular bug, improvement, or feature.
If two issues are related or blocking, please link them rather than combining them.

We will try to keep these issues as up-to-date as possible.
With the rapid rate of development in this field, some may get out of date.
If you notice this happening, please let us know.

### GitHub Discussions

We have a [discussions](https://github.com/Data-Simply/openretailscience/discussions) page
where users can ask usage questions, discuss design decisions, and propose new features.

If you are able to help answer questions, please do so!
This will allow the maintainers to spend more time focused on development and bug fixing.

### Keeping the agent skill current

OpenRetailScience ships an AI-agent skill at
`openretailscience/.agents/skills/using-openretailscience/` (installed by
`openretailscience.skills.install_skills`). Its `SKILL.md` and `references/*.md`
describe the public API — analyses, plots, options, and how to connect to data.

If you add, rename, or remove a public class, function, plot, or option, update
the skill in the same PR. `tests/test_skills.py` guards this: it re-executes
every import example in the skill and fails if one no longer resolves, and it
checks that every referenced file exists. Run `uv run pytest tests/test_skills.py`
before pushing.

### Getting Help

Our goal is to have the simplest developer setup possible.
Should you experience any difficulty getting setup, please contact a maintainer!
Not only do we want to help get you unblocked, but we also want to make sure that the process is smooth for future contributors.

In a similar vein, we do enforce certain linting, formatting, and documentation standards in the codebase.
If you are finding these difficult (or even just annoying) to work with, feel free to contact a maintainer for help.
We do not want these to get in the way of getting good code into the codebase.

### Recognition

If your contribution has been included in a release, we will add you to the README.md and note you as a contributor.
