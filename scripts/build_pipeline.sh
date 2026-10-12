#!/usr/bin/env bash
# Build the fused pipeline kernels (F-008): libmxpipeline_cpu.so and libmxpipeline_gpu.so.
# Same commands as the AOT builds in scripts/release.sh; ahead-of-time, nothing compiles at run time.
#
# Usage:  bash scripts/build_pipeline.sh [cpu|gpu]      (default: both)
#
# Both libraries are built for x86-64-v3 (AVX2 + FMA + BMI2, Haswell / Zen1 and newer), not the
# build host's native CPU, so a wheel does not SIGILL on an older machine. Measured on SF10 Q1
# (60M rows, 24 threads): 51 ms vs 56 ms native, i.e. no cost.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WHICH="${1:-all}"

if [[ "$WHICH" == "all" || "$WHICH" == "cpu" ]]; then
    pixi run mojo build kernels_aot/pipeline_cpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxpipeline_cpu.so
fi
if [[ "$WHICH" == "all" || "$WHICH" == "cpu" || "$WHICH" == "exec" ]]; then
    # F-030 native executor: one call runs the whole plan (imports pipeline_cpu.mojo)
    pixi run mojo build kernels_aot/executor_cpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxexec_cpu.so
fi
if [[ "$WHICH" == "all" || "$WHICH" == "gpu" ]]; then
    pixi run mojo build kernels_aot/pipeline_gpu.mojo -I kernels_aot --target-cpu x86-64-v3 \
        --emit shared-lib -o kernels_aot/libmxpipeline_gpu.so
fi
echo "built: $(ls kernels_aot/libmxpipeline_*.so kernels_aot/libmxexec_cpu.so | tr '\n' ' ')"
