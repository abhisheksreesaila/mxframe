#!/usr/bin/env bash
# Merge a reviewed branch into main and stop on any conflict, before anything can commit it
# (a scripted `git commit -am` after a failed merge twice recorded conflict markers; lessons 2026-10-16/17).
# Usage: scripts/merge_branch.sh <branch> "<merge message>"
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
branch="$1"; msg="$2"
if ! git -C "$root" merge --no-ff "$branch" -m "$msg"; then
  echo "merge of $branch stopped on conflicts:" >&2
  git -C "$root" diff --name-only --diff-filter=U >&2
  exit 1
fi
if git -C "$root" grep -qlI -E '^(<<<<<<< |>>>>>>> )' -- . ':!tests/test_no_conflict_markers.py'; then
  echo "conflict markers left after merging $branch" >&2
  exit 1
fi
echo "merged $branch"
