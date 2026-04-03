"""Minimal GPU custom op test through the MAX graph path.
Tests whether ANY GPU kernel works at all through ops.custom().
"""
import numpy as np
import pyarrow as pa
from pathlib import Path
import os
from max import engine, driver
from max.graph import Graph, TensorType, ops
from max.graph.type import DType, DeviceRef

KERNELS_PATH = str(Path("/home/ablearn/mxdf/mxframe/kernels_v261"))

print("=== _test_gpu_custom_op diagnostics ===")
print("MODULAR_DEVICE_CONTEXT_SYNC_MODE:", os.getenv("MODULAR_DEVICE_CONTEXT_SYNC_MODE", "<unset>"))
print("kernels pkg:", KERNELS_PATH)

acc = driver.Accelerator()
session = engine.InferenceSession(devices=[acc])
dev_ref = DeviceRef.GPU(0)

N = 8
n_groups = 2
WARP_SIZE = 32
num_warps = (N + WARP_SIZE - 1) // WARP_SIZE  # = 1

values_np = np.array([1.,2.,3.,4.,5.,6.,7.,8.], dtype=np.float32)
group_ids_np = np.array([0,1,0,1,0,1,0,1], dtype=np.int32)

# Build graph
graph = Graph(
    name="test_gpu_sum",
    input_types=[
        TensorType(DType.float32, [N], dev_ref),   # values
        TensorType(DType.int32,   [N], dev_ref),   # group_ids
    ],
    custom_extensions=[Path(KERNELS_PATH)],
)

with graph:
    val_node = graph.inputs[0]
    gid_node = graph.inputs[1]

    out_type = TensorType(DType.float32, [num_warps * n_groups], dev_ref)
    partial = ops.custom(
        name="group_sum",
        values=[val_node, gid_node],
        out_types=[out_type],
        device=dev_ref,
    )[0]
    # Reshape [1, 2] then sum axis=0 → [2]
    reshaped = ops.reshape(partial, [num_warps, n_groups])
    result = ops.sum(reshaped, axis=0)
    graph.output(result)

print("Graph built OK. Loading model...")
try:
    model = session.load(graph)
except Exception as exc:
    print("FAILURE PHASE: session.load(graph)")
    raise RuntimeError("session.load failed in _test_gpu_custom_op") from exc

print("Model loaded OK. Executing...")

val_gpu = driver.Buffer.from_numpy(values_np).to(acc)
gid_gpu = driver.Buffer.from_numpy(group_ids_np).to(acc)

try:
    outputs = model.execute(val_gpu, gid_gpu)
except Exception as exc:
    print("FAILURE PHASE: model.execute(inputs)")
    raise RuntimeError("model.execute failed in _test_gpu_custom_op") from exc

result_np = outputs[0].to_numpy()
result_np = np.asarray(result_np).reshape(-1)
print(f"Result: {result_np}")
print(f"Expected: [16.0, 20.0]")

assert abs(result_np[0] - 16.0) < 1e-4, f"Wrong: {result_np[0]}"
assert abs(result_np[1] - 20.0) < 1e-4, f"Wrong: {result_np[1]}"
print("PASSED")
