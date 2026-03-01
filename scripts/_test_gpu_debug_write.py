"""Test: add_one-style debug custom op on GPU (output = input + 1)."""
import numpy as np
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops
from max.graph.type import DType, DeviceRef

KERNELS_PATH = str(Path("/home/ablearn/mxdf/mxframe/kernels_v261"))

acc = driver.Accelerator()
session = engine.InferenceSession(devices=[acc])
dref = DeviceRef.GPU(0)

graph = Graph(
    name="debug_noop",
    input_types=[TensorType(DType.float32, [2], dref)],
    custom_extensions=[Path(KERNELS_PATH)],
)
with graph:
    inp = graph.inputs[0]
    out_type = TensorType(DType.float32, [2], dref)
    result = ops.custom(
        name="debug_write_one",
        values=[inp],
        out_types=[out_type],
        device=dref,
    )[0]
    graph.output(result)

print("Graph built. Loading model...")
model = session.load(graph)
print("Model loaded. Executing...")
inp_data = driver.Buffer.from_numpy(np.array([1.0, 2.0], dtype=np.float32)).to(acc)
outputs = model.execute(inp_data)
result_np = outputs[0].to_numpy()

print(f"Result: {result_np}  (expected [2.0, 3.0])")
if abs(result_np[0] - 2.0) < 1e-4 and abs(result_np[1] - 3.0) < 1e-4:
    print("PASSED — add_one-style GPU custom op works")
else:
    print(f"WRONG VALUES or FAILED")
