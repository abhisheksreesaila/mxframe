"""Full end-to-end test of the fused agg reduction pattern."""
from max.graph import ops, Graph, TensorType, DeviceRef
from max.dtype import DType
from max import driver, engine
import numpy as np

sess = engine.InferenceSession(devices=[driver.CPU()])

# Simulate kernel output: [num_warps * 6] interleaved
# Layout: [w0_agg0, w0_agg1, w0_agg2, w0_agg3, w0_agg4, w0_agg5,
#          w1_agg0, w1_agg1, ...]
num_warps = 3
N = num_warps * 6
# warp0: agg0=1, agg1=2, ..., agg5=6
# warp1: agg0=7, agg1=8, ..., agg5=12
# warp2: agg0=13, agg1=14, ..., agg5=18
arr = np.array([i+1 for w in range(num_warps) for a in range(6) for i in [w*6+a]], dtype=np.float32)
print(f"input shape: {arr.shape}, values: {arr.tolist()}")
# expected sums: agg0 = 1+7+13=21, agg1=2+8+14=24, ..., agg5=6+12+18=36
expected = [1+7+13, 2+8+14, 3+9+15, 4+10+16, 5+11+17, 6+12+18]
print(f"expected: {expected}")

t = driver.Buffer.from_numpy(arr).to(driver.CPU())

with Graph("test_fused_reduce", input_types=[TensorType(DType.float32, (N,), DeviceRef.CPU())]) as g:
    flat     = g.inputs[0]
    shaped   = ops.reshape(flat, [num_warps, 6])   # [3, 6]
    transpos = ops.transpose(shaped, 0, 1)          # [6, 3]
    col_sums = ops.sum(transpos)                    # [6, 1]
    flat6    = ops.reshape(col_sums, [6])           # [6]
    cols     = ops.split(flat6, [1,1,1,1,1,1], 0)  # 6 × [1]
    g.output(*cols)

model  = sess.load(g)
out    = model.execute(t)
result = [float(o.to_numpy().flat[0]) for o in out]
print(f"got: {result}")
assert result == expected, f"Expected {expected}, got {result}"
print("PASS: transpose+sum+reshape+split works!")
