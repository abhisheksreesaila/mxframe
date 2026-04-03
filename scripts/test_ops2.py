"""Find correct signatures for split, chunk, transpose, slice_tensor."""
from max.graph import ops
import inspect

for name in ["split", "chunk", "transpose", "slice_tensor", "gather", "permute"]:
    try:
        fn  = getattr(ops, name)
        sig = str(inspect.signature(fn))
        print(f"{name}: {sig}")
    except Exception as e:
        print(f"{name}: ERROR {e}")
