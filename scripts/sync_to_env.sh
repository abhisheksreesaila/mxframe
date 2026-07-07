#!/usr/bin/env bash
# sync_to_env.sh — Copy workspace source files to the active pixi environment.
#
# Run this after editing any .py file in the workspace root so that
# scripts (which use the mojo-gpu-tutorials pixi env) pick up the changes.
#
# Usage:
#   bash scripts/sync_to_env.sh
#   bash scripts/sync_to_env.sh --gpu-so   # also sync the GPU .so
#
set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
SPKG=/home/ablearn/mojo-gpu-tutorials/.pixi/envs/default/lib/python3.12/site-packages/mxframe

echo "Syncing Python source files: $REPO → $SPKG"
cp "$REPO/custom_ops.py"     "$SPKG/"
cp "$REPO/aot_kernels.py"    "$SPKG/"
cp "$REPO/compiler.py"       "$SPKG/"
cp "$REPO/lazy_frame.py"     "$SPKG/"
cp "$REPO/lazy_expr.py"      "$SPKG/"
cp "$REPO/optimizer.py"      "$SPKG/"
cp "$REPO/plan_validation.py" "$SPKG/"
cp "$REPO/sql_frontend.py"   "$SPKG/"
cp "$REPO/__init__.py"       "$SPKG/"

if [[ "${1:-}" == "--gpu-so" ]]; then
    echo "Syncing libmxkernels_aot_gpu.so"
    # Copy into the mxframe package's kernels_aot/ subdirectory where aot_kernels.py looks
    mkdir -p "$SPKG/kernels_aot"
    cp "$REPO/kernels_aot/libmxkernels_aot_gpu.so" "$SPKG/kernels_aot/"
fi

echo "Done. Run: python scripts/audit_gpu_paths.py --device cpu --rows 100000"
