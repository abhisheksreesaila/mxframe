#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

SYNC_MODE="1"

echo "[0/7] Environment + GPU diagnostics"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/_check_gpu.py

echo "[1/7] Build kernels"
pixi run bash scripts/build_kernels.sh

echo "[2/7] Debug write-one custom op"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/_test_gpu_debug_write.py

echo "[3/7] Minimal grouped custom op"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/_test_gpu_custom_op.py

echo "[4/7] CPU vs GPU parity harness"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/_test_gpu_noop.py

echo "[5/7] End-to-end grouped regression"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/_test_gpu_regression.py

echo "[6/7] Fused reduction shape contract"
MODULAR_DEVICE_CONTEXT_SYNC_MODE="$SYNC_MODE" pixi run python3 scripts/test_fused_reduce.py

echo "All GPU playbook checks passed ✅"