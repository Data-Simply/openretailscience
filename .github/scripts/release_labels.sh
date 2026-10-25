#!/usr/bin/env bash
# Prints the release-note type labels (see .github/release.yml) for a conventional-commit PR title, or exits 1 if the
# title isn't one.
set -euo pipefail

title_re='^(build|chore|ci|docs|feat|fix|perf|refactor|revert|style|test)(\([^)]*\))?(!)?: '
if [[ ! "$1" =~ $title_re ]]; then
  echo "PR title must be a conventional commit, e.g. 'feat(clv): add CLVStats.sample'. Got: $1" >&2
  exit 1
fi

if [[ -n "${BASH_REMATCH[3]}" ]]; then
  echo "type:breaking"
fi
case "${BASH_REMATCH[1]}" in
  feat) echo "type:feature" ;;
  fix) echo "type:bug" ;;
  docs) echo "type:docs" ;;
  *) echo "type:maintenance" ;;
esac
