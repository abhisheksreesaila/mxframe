"""Test: reshape [3,4] → sum axis=0 → [4] → split into 4 scalars."""
from max.graph import ops, Graph, TensorType, DeviceRef
from max.dtype import DType
from max import driver, engine
import numpy as np

sess = engine.InferenceSession(devices=[driver.CPU()])
arr  = np.arange(12, dtype=np.float32)
t    = driver.Tensor.from_numpy(arr)

# Expected: col0=0+4+8=12, col1=1+5+9=15, col2=2+6+10=18, col3=3+7+11=21
with Graph("test_sum_axis", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
    flat     = g.inputs[0]
    shaped   = ops.reshape(flat, [3, 4])         # [3, 4]
    col_sums = ops.sum(shaped, 0)                # [4]  sum along axis 0
    cols     = ops.split(col_sums, [1,1,1,1], 0) # 4 × [1]
    results  = [ops.sum(c) for c in cols]         # 4 scalars (sum of [1] = same val)
    g.output(*results)

model  = sess.load(g)
out    = model.execute(t)
result = [float(o.to_numpy().flat[0]) for o in out]
print(f"sum(axis=0)+split: {result}")
expected = [12., 15., 18., 21.]
assert result == expected, f"Expected {expected}, got {result}"
print("PASS")
