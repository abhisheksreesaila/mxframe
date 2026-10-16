#!/usr/bin/env bash
# F-036: give the AOT libraries a relocatable RUNPATH.
#
# `mojo build` stamps each .so with the absolute path of the build machine's pixi env, so the
# library finds libKGENCompilerRTShared.so (the Mojo runtime) only while that directory exists.
# This replaces it with $ORIGIN-relative entries, searched in order, for every install layout:
#   $ORIGIN/../.pixi/envs/default/lib   a source checkout (or worktree) with its pixi env;
#                                  first, so a checkout never prefers a directory outside itself
#   $ORIGIN/../../modular/lib      pip wheel: site-packages/modular/lib next to site-packages/mxframe
#   $ORIGIN/../../../..            venv or conda env: <env>/lib, with the libraries in
#                                  <env>/lib/pythonX.Y/site-packages/mxframe/kernels_aot
# In a checkout without a pixi env the last two resolve outside the repo (../modular/lib,
# four levels up): do not keep a checkout under a directory others can write to.
# Linux only (ELF RUNPATH); macOS dylibs are out of scope.
#
# Run it after every `mojo build` of a kernels_aot library (the build scripts do).
# Usage:  bash scripts/set_runpath.sh [lib.so ...]      (default: every kernels_aot/*.so)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUNPATH='$ORIGIN/../.pixi/envs/default/lib:$ORIGIN/../../modular/lib:$ORIGIN/../../../..'

if [[ $# -eq 0 ]]; then
    set -- "$ROOT"/kernels_aot/*.so
fi
for lib in "$@"; do
    pixi run patchelf --set-rpath "$RUNPATH" "$lib"
done
echo "runpath set on: $*"
