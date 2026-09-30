#!/usr/bin/env bash
# Build the fused pipeline kernels (F-008): libmxpipeline_cpu.so and libmxpipeline_gpu.so.
# Same commands as the AOT builds in scripts/release.sh; ahead-of-time, nothing compiles at run time.
#
# Usage:  bash scripts/build_pipeline.sh [cpu|gpu]      (default: both)
#
# The CPU library targets the host CPU's SIMD width (native); the GPU library needs a
# CUDA-capable build host only for testing, the kernels themselves are embedded.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WHICH="${1:-all}"

if [[ "$WHICH" == "all" || "$WHICH" == "cpu" ]]; then
    pixi run mojo build kernels_aot/pipeline_cpu.mojo -I kernels_aot \
        --emit shared-lib -o kernels_aot/libmxpipeline_cpu.so
fi
if [[ "$WHICH" == "all" || "$WHICH" == "gpu" ]]; then
    pixi run mojo build kernels_aot/pipeline_gpu.mojo -I kernels_aot \
        --emit shared-lib -o kernels_aot/libmxpipeline_gpu.so
fi
echo "built: $(ls kernels_aot/libmxpipeline_*.so | tr '\n' ' ')"
