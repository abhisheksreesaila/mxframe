#!/usr/bin/env bash
# Run pytest on this checkout from any directory name (git worktrees included).
# The repo root is the `mxframe` package, and pytest names it after its directory,
# so tests run through a symlink called `mxframe`.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
shim="${TMPDIR:-/tmp}/mxframe-pytest-$(printf '%s' "$root" | sha1sum | cut -c1-10)"
mkdir -p "$shim" && ln -sfn "$root" "$shim/mxframe"
cd "$shim/mxframe" && exec pytest "$shim/mxframe/tests" "$@"
