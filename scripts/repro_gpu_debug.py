"""Standalone custom-op repro runner (CPU + GPU) for quick debugging.

Usage:
  pixi run python3 scripts/repro_gpu_debug.py

What it does:
1) Builds a local mojopkg from scripts/repro_kernels/
2) Runs the custom op on CPU and checks output
3) Runs the custom op on GPU and reports exact failure/success
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from max import driver, engine
from max.graph import Graph, TensorType, ops
from max.graph.type import DType, DeviceRef


REPO_ROOT = Path(__file__).resolve().parents[1]
KERNELS_DIR = REPO_ROOT / "scripts" / "repro_kernels"
KERNELS_PKG = REPO_ROOT / "scripts" / "repro_kernels.mojopkg"
KERNEL_NAME = "repro_debug_add_one"


def build_kernels() -> None:
    print(f"Building kernels: {KERNELS_DIR} -> {KERNELS_PKG}")
    subprocess.run(
        ["mojo", "package", str(KERNELS_DIR), "-o", str(KERNELS_PKG)],
        check=True,
    )
    print("Build OK")


def run_once(device: str) -> np.ndarray:
    if device == "gpu":
        dev = driver.Accelerator()
        dref = DeviceRef.GPU(0)
    else:
        dev = driver.CPU()
        dref = DeviceRef.CPU()

    sess = engine.InferenceSession(devices=[dev])

    data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)

    graph = Graph(
        name=f"repro_{device}",
        input_types=[TensorType(DType.float32, [len(data)], dref)],
        custom_extensions=[KERNELS_PKG],
    )
    with graph:
        inp = graph.inputs[0]
        out_type = TensorType(DType.float32, [len(data)], dref)
        out = ops.custom(
            name=KERNEL_NAME,
            values=[inp],
            out_types=[out_type],
            device=dref,
        )[0]
        graph.output(out)

    model = sess.load(graph)

    if device == "gpu":
        inp_tensor = driver.Tensor(data, device=dev)
        outputs = model.execute(inp_tensor)
    else:
        outputs = model.execute(data)

    return outputs[0].to_numpy()


def main() -> None:
    if os.getenv("MXF_SKIP_REPRO_KERNEL_BUILD", "0") != "1":
        build_kernels()
    else:
        print("Skipping kernel build because MXF_SKIP_REPRO_KERNEL_BUILD=1")

    expected = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    print("\n[CPU] Running...")
    cpu_out = run_once("cpu")
    print(f"CPU result: {cpu_out}")
    assert np.allclose(cpu_out, expected), f"CPU mismatch: {cpu_out} != {expected}"
    print("CPU PASS")

    print("\n[GPU] Running...")
    try:
        gpu_out = run_once("gpu")
        print(f"GPU result: {gpu_out}")
        if np.allclose(gpu_out, expected):
            print("GPU PASS")
        else:
            print(f"GPU WRONG VALUES: {gpu_out} != {expected}")
    except Exception as exc:
        print("GPU FAILED with exception:")
        print(repr(exc))
        raise


if __name__ == "__main__":
    main()
