"""Verify ops.chunk(shaped, 6, 1) works for column extraction."""
from max.graph import ops, Graph, TensorType, DeviceRef
from max.dtype import DType
from max import driver, engine
import numpy as np

sess = engine.InferenceSession(devices=[driver.CPU()])
arr  = np.arange(12, dtype=np.float32)  # [0..11]
t    = driver.Tensor.from_numpy(arr)

# [3,4] → chunk into 4 cols of [3,1] → sum each col
# col0 = 0+4+8=12, col1=1+5+9=15, col2=2+6+10=18, col3=3+7+11=21
with Graph("test_chunk", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
    flat   = g.inputs[0]
    shaped = ops.reshape(flat, [3, 4])
    chunks = ops.chunk(shaped, 4, 1)       # axis=1, positional
    sums   = [ops.sum(c) for c in chunks]
    g.output(*sums)

model  = sess.load(g)
out    = model.execute(t)
result = [float(o.to_numpy().flat[0]) for o in out]
print(f"chunk(axis=1): {result}")
expected = [12., 15., 18., 21.]
assert result == expected, f"Expected {expected}, got {result}"
print("PASS: ops.chunk(x, n, axis) works for column extraction")
