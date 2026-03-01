import sys, numpy as np
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops
from max.graph.type import DType, DeviceRef

KERNELS = Path("mxframe/kernels.mojopkg")
acc = driver.Accelerator()
session = engine.InferenceSession(devices=[acc])
dref = DeviceRef.GPU(0)
N = 4

graph = Graph(
    name="dbg_test",
    input_types=[TensorType(DType.float32, [N], dref)],
    custom_extensions=[KERNELS],
)
with graph:
    inp = graph.inputs[0]
    out_type = TensorType(DType.float32, [N], dref)
    result = ops.custom(
        name="debug_write_one",
        values=[inp],
        out_types=[out_type],
        device=dref,
    )[0]
    graph.output(result)

model = session.load(graph)
inp_data = driver.Tensor(np.array([10., 20., 30., 40.], dtype=np.float32), device=acc)
print("Executing...")
result = model.execute(inp_data)
r = result[0].to_numpy()
expected = np.array([11., 21., 31., 41.], dtype=np.float32)
print("Result:  ", r)
print("Expected:", expected)
print("Match:", np.allclose(r, expected))
