#!/usr/bin/env bash
# Run pytest on this checkout from any directory name (git worktrees included).
# The repo root is the `mxframe` package, and pytest names it after its directory,
# so tests run through a symlink called `mxframe`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
shim="${TMPDIR:-/tmp}/mxframe-pytest-$(printf '%s' "$root" | sha1sum | cut -c1-10)"
mkdir -p "$shim" && ln -sfn "$root" "$shim/mxframe"
# Test paths (`tests/test_x.py`, `tests/test_x.py::name`) run alone, through the shim;
# with none, the whole tests directory runs. Other arguments pass through.
args=() paths=()
for a in "$@"; do
  case "$a" in
    -*) args+=("$a") ;;
    tests/*|./tests/*|"$root"/tests/*) paths+=("$shim/mxframe/${a#"$root"/}") ;;
    *) args+=("$a") ;;
  esac
done
[ ${#paths[@]} -eq 0 ] && paths=("$shim/mxframe/tests")
cd "$shim/mxframe" && exec pytest "${paths[@]}" "${args[@]}"
