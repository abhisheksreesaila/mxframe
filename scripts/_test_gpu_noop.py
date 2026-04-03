"""Minimal debug kernel that does nothing on GPU — isolates framework vs kernel."""
# Usage: pixi run python3 scripts/_test_gpu_noop.py
import numpy as np
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops
from max.graph.type import DType, DeviceRef

# We'll test with the existing group_sum kernel but in CPU mode
# to confirm: group_sum CPU path is fine
KERNELS_PATH = str(Path("/home/ablearn/mxdf/mxframe/kernels_v261"))

N = 8
n_groups = 2
WARP_SIZE = 32
num_warps = (N + WARP_SIZE - 1) // WARP_SIZE

values_np    = np.array([1.,2.,3.,4.,5.,6.,7.,8.], dtype=np.float32)
group_ids_np = np.array([0,1,0,1,0,1,0,1], dtype=np.int32)

def run_test(device_str, expect_gpu):
    acc    = driver.Accelerator() if expect_gpu else None
    dev_d  = acc if expect_gpu else driver.CPU()
    session = engine.InferenceSession(devices=[dev_d])
    dref    = DeviceRef.GPU(0) if expect_gpu else DeviceRef.CPU()

    graph = Graph(
        name="debug_sum",
        input_types=[
            TensorType(DType.float32, [N], dref),
            TensorType(DType.int32,   [N], dref),
        ],
        custom_extensions=[Path(KERNELS_PATH)],
    )
    nw = num_warps if expect_gpu else 1
    ng = n_groups
    with graph:
        v = graph.inputs[0]
        g = graph.inputs[1]
        out_type = TensorType(DType.float32, [nw * ng], dref)
        partial = ops.custom(
            name="group_sum",
            values=[v, g],
            out_types=[out_type],
            device=dref,
        )[0]
        if expect_gpu:
            reshaped = ops.reshape(partial, [nw, ng])
            result = ops.sum(reshaped, axis=0)
        else:
            result = partial
        graph.output(result)

    model = session.load(graph)

    if expect_gpu:
        inputs = [
            driver.Buffer.from_numpy(values_np).to(acc),
            driver.Buffer.from_numpy(group_ids_np).to(acc),
        ]
    else:
        inputs = [values_np, group_ids_np]

    outputs = model.execute(*inputs)
    r = np.asarray(outputs[0].to_numpy()).reshape(-1)
    print(f"  [{device_str}] result = {r}")
    return r

print("=== CPU test ===")
r_cpu = run_test("cpu", expect_gpu=False)
assert abs(r_cpu[0] - 16.0) < 1e-4 and abs(r_cpu[1] - 20.0) < 1e-4, f"CPU wrong: {r_cpu}"
print("  CPU PASSED")

print()
print("=== GPU test ===")
try:
    r_gpu = run_test("gpu", expect_gpu=True)
    assert abs(r_gpu[0] - 16.0) < 1e-4 and abs(r_gpu[1] - 20.0) < 1e-4, f"GPU wrong: {r_gpu}"
    print("  GPU PASSED")
except Exception as e:
    print(f"  GPU FAILED: {e}")
