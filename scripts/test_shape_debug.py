"""Debug: what shape does ops.sum(x, axis=0) produce?"""
from max.graph import ops, Graph, TensorType, DeviceRef
from max.dtype import DType
from max import driver, engine
import numpy as np

sess = engine.InferenceSession(devices=[driver.CPU()])
arr  = np.arange(12, dtype=np.float32)
t    = driver.Tensor.from_numpy(arr)

# Test what reshape does to the shape information
with Graph("test_shapes", input_types=[TensorType(DType.float32, (12,), DeviceRef.CPU())]) as g:
    flat     = g.inputs[0]
    shaped   = ops.reshape(flat, [3, 4])    # [3, 4]
    transpos = ops.transpose(shaped, 0, 1)  # [4, 3]
    col_sum  = ops.sum(transpos)            # axis=-1 (default) → [4]
    g.output(col_sum)

model  = sess.load(g)
out    = model.execute(t)
result = out[0].to_numpy()
print(f"transpose+sum(axis=-1): shape={result.shape}, values={result.tolist()}")
# For transpos[i][j] = shaped[j][i] = arr[4j+i]
# row 0 of transpos: [0,4,8] → sum=12
# row 1 of transpos: [1,5,9] → sum=15
# etc.
