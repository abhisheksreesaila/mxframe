"""Quick smoke test to verify grouped kernel dispatch works."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pyarrow as pa, numpy as np
from mxframe import LazyFrame, col, lit
from mxframe.custom_ops import clear_cache, KERNELS_PATH

print("KERNELS_PATH:", KERNELS_PATH)
clear_cache()

tbl = pa.table({"a": np.arange(10, dtype=np.float32),
                "g": np.array([0]*5+[1]*5, dtype=np.int32)})

try:
    r = LazyFrame(tbl).groupby("g").agg(col("a").sum().alias("s")).compute(device="cpu")
    print("grouped sum OK:", r)
except Exception as e:
    print("grouped sum FAIL:", e)

try:
    r2 = LazyFrame(tbl).groupby().agg(col("a").sum().alias("s")).compute(device="cpu")
    print("global sum OK:", r2)
except Exception as e:
    print("global sum FAIL:", e)
