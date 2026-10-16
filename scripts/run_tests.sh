#!/usr/bin/env bash
# Run pytest on this checkout from any directory name (git worktrees included).
# The repo root is the `mxframe` package, and pytest names it after its directory,
# so tests run through a symlink called `mxframe`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
# Test temp files (generated TPC-H data, pytest tmp_path) go to disk, not the RAM /tmp, which
# several agents' suites filled until tests failed with "Disk quota exceeded".
export TMPDIR="${MXFRAME_TEST_TMPDIR:-$HOME/.cache/mxframe-test-tmp}"
mkdir -p "$TMPDIR"
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
# a test stuck for 15 minutes dumps every thread's stack (a full-suite run hung once under load, F-070)
cd "$shim/mxframe" && exec pytest -o faulthandler_timeout=900 "${paths[@]}" "${args[@]}"
