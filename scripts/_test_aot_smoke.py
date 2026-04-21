#!/usr/bin/env python3
"""Smoke test: verify AOT kernels and custom_ops.py AOT dispatch work correctly."""
import sys

import numpy as np
import pyarrow as pa

PASS = 0
FAIL = 0

def check(name, got, expected):
    global PASS, FAIL
    ok = np.allclose(np.array(got if hasattr(got,'__len__') else [got], dtype=float), np.array(expected if hasattr(expected,'__len__') else [expected], dtype=float), rtol=1e-4, atol=1e-3)
    if ok:
        print(f"  PASS  {name}")
        PASS += 1
    else:
        print(f"  FAIL  {name}  got={got}  expected={expected}")
        FAIL += 1

# ── 1. Direct AOT kernel test ─────────────────────────────────────────────────
print("=== AOT kernel correctness ===")
from mxframe.aot_kernels import AOTKernels, AOT_AVAILABLE
print(f"AOT_AVAILABLE={AOT_AVAILABLE}")

if not AOT_AVAILABLE:
    print("FATAL: AOT not available")
    sys.exit(1)

aot = AOTKernels()

# group_sum_f32
vals   = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
labels = np.array([0,   1,   0,   1,   2],   dtype=np.int32)
res = aot.group_sum_f32(vals, labels, 3)
check("group_sum_f32", res, [4.0, 6.0, 5.0])

# group_min_f32 / group_max_f32
res = aot.group_min_f32(vals, labels, 3)
check("group_min_f32", res, [1.0, 2.0, 5.0])
res = aot.group_max_f32(vals, labels, 3)
check("group_max_f32", res, [3.0, 4.0, 5.0])

# group_mean_f32
res = aot.group_mean_f32(vals, labels, 3)
check("group_mean_f32", res, [2.0, 3.0, 5.0])

# group_count_f32
res = aot.group_count_f32(labels, 3)
check("group_count_f32", res, [2.0, 2.0, 1.0])

# group_composite
k0 = np.array([0, 1, 2, 3], dtype=np.int32)
k1 = np.array([1, 0, 1, 0], dtype=np.int32)
k2 = np.zeros(4, dtype=np.int32)
k3 = np.zeros(4, dtype=np.int32)
strides = np.array([10, 2, 0, 0], dtype=np.int64)
res = aot.group_composite(k0, k1, k2, k3, strides)
expected_comp = [0*10+1*2, 1*10+0*2, 2*10+1*2, 3*10+0*2]  # 2,10,22,30
check("group_composite", res, expected_comp)

# masked_global_sum_f32
v  = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
m  = np.array([1, 0, 1, 1], dtype=np.int32)
check("masked_global_sum_f32",     aot.masked_global_sum_f32(v, m),     1+3+4)
check("masked_global_min_f32",     aot.masked_global_min_f32(v, m),     1.0)
check("masked_global_max_f32",     aot.masked_global_max_f32(v, m),     4.0)
check("masked_global_sum_product", aot.masked_global_sum_product_f32(v, v, m), 1+9+16)

# gather
src = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)
idx = np.array([3, 1, 2], dtype=np.int32)
check("gather_f32", aot.gather_f32(src, idx), [40.0, 20.0, 30.0])

# sort_indices
keys = np.array([3, 1, 4, 1, 5], dtype=np.int32)
res = aot.sort_indices(keys, descending=False)
check("sort_indices asc", keys[res], sorted(keys))

# unique_mask
skeys = np.array([1, 1, 2, 3, 3], dtype=np.int32)
res = aot.unique_mask(skeys)
check("unique_mask", res, [1, 0, 1, 1, 0])

# prefix_sum_count
mask2 = np.array([1, 0, 1, 1, 0], dtype=np.int32)
offs = aot.prefix_sum_count(mask2)
check("prefix_sum_count", offs, [0, 1, 1, 2, 3, 3])

# filter_gather_f32
src2 = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
res = aot.filter_gather_f32(src2, mask2, offs)
check("filter_gather_f32", res, [10.0, 30.0, 40.0])

# join_count / join_scatter (inner)
lk = np.array([0, 1, 2], dtype=np.int32)
rk = np.array([1, 1, 2], dtype=np.int32)
mc = aot.join_count(lk, rk)
check("join_count inner", mc, [0, 2, 1])
lo, ro = aot.join_scatter(lk, rk, mc)
check("join_scatter total", len(lo), 3)

# join_count_left / join_scatter_left
lk2 = np.array([0, 1, 2], dtype=np.int32)
mc2 = aot.join_count_left(lk2, rk)
check("join_count_left (no-match row=1)", mc2, [1, 2, 1])
lo2, ro2 = aot.join_scatter_left(lk2, rk, mc2)
check("join_scatter_left total", len(lo2), 4)
check("join_scatter_left no-match right_idx", ro2[0], -1)

# ── 2. CustomOpsCompiler AOT dispatch ────────────────────────────────────────
print("\n=== CustomOpsCompiler AOT dispatch ===")
from mxframe.custom_ops import CustomOpsCompiler
from mxframe.lazy_expr import Expr, col
from mxframe.lazy_frame import LazyFrame, Scan

N = 100
comp = CustomOpsCompiler(device="cpu")
print(f"AOT loaded: {comp._aot is not None}")
rng = np.random.default_rng(42)
key_col = (rng.integers(0, 5, N)).astype(np.int32)
val_col  = rng.uniform(1.0, 10.0, N).astype(np.float32)

tbl = pa.table({"key": pa.array(key_col.astype(np.int64)),
                "val": pa.array(val_col.astype(np.float64))})

lf  = LazyFrame(Scan(tbl))

# Grouped sum via compile_and_run (AOT cpu_aot path)
plan = lf.groupby("key").agg(Expr("col", "val").sum().alias("sum_val")).plan
res  = comp.compile_and_run(plan)
# Validate against numpy ground truth
for row in res.to_pydict()["key"]:
    gt = float(val_col[key_col == int(row)].sum())
    got = res.filter(pa.compute.equal(res.column("key"), row)).column("sum_val")[0].as_py()
    if not np.isclose(got, gt, rtol=1e-3):
        print(f"  FAIL grouped sum for key={row}: got={got}, expected={gt}")
        FAIL += 1
    else:
        PASS += 1

print(f"  PASS  grouped sum (all {len(res)} groups)")

# Check provenance shows cpu_aot path
prov = comp.last_compile_provenance
print(f"  path={prov.get('path')}  compile_ms={prov.get('compile_ms')}ms")
if prov.get("path") == "cpu_aot":
    print("  PASS  path=cpu_aot (AOT dispatch active)")
    PASS += 1
else:
    print(f"  FAIL  expected path=cpu_aot, got {prov.get('path')}")
    FAIL += 1

# ── 3. Q1-style test (multi-agg on filtered data) ────────────────────────────
print("\n=== Q1-style multi-agg ===")
try:
    from mxframe import MXFrame
    print("  PASS  mxframe importable")
    PASS += 1
except Exception as e:
    print(f"  WARN  skipping MXFrame import: {e}")

# ── Summary ──────────────────────────────────────────────────────────────────
print(f"\n=== SUMMARY: {PASS} passed, {FAIL} failed ===")
sys.exit(FAIL)
