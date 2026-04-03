"""Test various ops to find column extraction approach for fused_group_agg."""
from max.graph import ops, Graph, TensorType, DeviceRef
from max.dtype import DType
from max import driver, engine
import numpy as np
import inspect

sess_dev = driver.CPU()
sess     = engine.InferenceSession(devices=[sess_dev])
# arr = [0,1,2,3, 4,5,6,7, 8,9,10,11] → reshape [3,4]
# col 0 sum = 0+4+8=12, col1=1+5+9=15, col2=2+6+10=18, col3=3+7+11=21
arr = np.arange(12, dtype=np.float32)
t   = driver.Tensor.from_numpy(arr)

# --- Test ops.split signature ---
try:
    print(f"ops.split sig: {inspect.signature(ops.split)}")
except: pass

# --- Test ops.transpose + ops.chunk ---
try:
    with Graph("test_transpose_chunk", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
        flat    = g.inputs[0]
        shaped  = ops.reshape(flat, [3, 4])         # [3, 4]  rows=warps, cols=aggs
        transp  = ops.transpose(shaped, 1, 0)        # [4, 3]  rows=aggs, cols=warps
        chunks  = ops.chunk(transp, 4, dim=0)        # 4 tensors of [1, 3]
        sums    = [ops.sum(c) for c in chunks]
        g.output(*sums)
    model = sess.load(g)
    out = model.execute(t)
    results = [float(o.to_numpy().flat[0]) for o in out]
    print(f"transpose+chunk: {results}")
    assert abs(results[0]-12)<0.1 and abs(results[1]-15)<0.1, f"Got: {results}"
    print("transpose+chunk PASS")
except Exception as e:
    print(f"transpose+chunk FAIL: {e}")

# --- Test ops.split(tensor, num_splits, dim)  (positional) ---
try:
    with Graph("test_split2", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
        flat   = g.inputs[0]
        shaped = ops.reshape(flat, [3, 4])
        cols   = ops.split(shaped, 4, 1)    # positional: (tensor, num, dim)
        sums   = [ops.sum(c) for c in cols]
        g.output(*sums)
    model  = sess.load(g)
    out    = model.execute(t)
    results= [float(o.to_numpy().flat[0]) for o in out]
    print(f"split positional: {results}")
    assert abs(results[0]-12)<0.1, f"Got: {results}"
    print("split positional PASS")
except Exception as e:
    print(f"split positional FAIL: {e}")

# --- Test ops.chunk(tensor, num_chunks, dim) ---
try:
    with Graph("test_chunk", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
        flat   = g.inputs[0]
        shaped = ops.reshape(flat, [3, 4])
        chunks = ops.chunk(shaped, 4, dim=1)   # 4 tensors of [3,1]
        sums   = [ops.sum(c) for c in chunks]
        g.output(*sums)
    model  = sess.load(g)
    out    = model.execute(t)
    results= [float(o.to_numpy().flat[0]) for o in out]
    print(f"chunk(dim=1): {results}")
    assert abs(results[0]-12)<0.1, f"Got: {results}"
    print("chunk PASS")
except Exception as e:
    print(f"chunk FAIL: {e}")
