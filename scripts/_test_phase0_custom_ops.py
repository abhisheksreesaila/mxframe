"""Phase 0 verification: CustomOpsCompiler tests."""
import pyarrow as pa
from pathlib import Path
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Scan
from mxframe.custom_ops import CustomOpsCompiler

compiler = CustomOpsCompiler()

# ── Test 1: projection ───────────────────────────────────────────────────────
table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
result = compiler.compile_and_run(LazyFrame(Scan(table)).select(col("a") + lit(10)).plan)
assert result.column(0).to_pylist() == [11, 12, 13]
print("✅ Test 1: projection")

# ── Test 2: global sum ───────────────────────────────────────────────────────
lf2 = LazyFrame(Scan(pa.table({"x": [1.0, 2.0, 3.0, 4.0]})))
result2 = compiler.compile_and_run(
    lf2.groupby().agg(col("x").sum().alias("total")).plan
)
assert abs(result2.column("total").to_pylist()[0] - 10.0) < 1e-6
print("✅ Test 2: global sum")

# ── Test 3: grouped sum + key columns in result ──────────────────────────────
table3 = pa.table({
    "group": ["a", "b", "a", "b", "a"],
    "val":   [1.0, 2.0, 3.0, 4.0, 5.0],
})
lf3 = LazyFrame(Scan(table3))
result3 = compiler.compile_and_run(
    lf3.groupby("group").agg(col("val").sum().alias("total")).plan
)
assert "group" in result3.column_names, f"Missing group key: {result3.column_names}"
assert result3.column_names[0] == "group", f"Key should be first: {result3.column_names}"
groups = result3.column("group").to_pylist()
totals = result3.column("total").to_pylist()
d = dict(zip(groups, totals))
assert abs(d["a"] - 9.0) < 1e-6, f"Sum for a should be 9.0: {d}"
assert abs(d["b"] - 6.0) < 1e-6, f"Sum for b should be 6.0: {d}"
print("✅ Test 3: grouped sum with key columns in result (a=9, b=6)")

# ── Test 4: filter before grouped agg ────────────────────────────────────────
table4 = pa.table({
    "group": ["a", "b", "a", "b", "a"],
    "val":   [1.0, 2.0, 3.0, 4.0, 5.0],
    "flag":  [1,   1,   0,   1,   1  ],
})
lf4 = LazyFrame(Scan(table4))
result4 = compiler.compile_and_run(
    lf4.filter(col("flag") > lit(0)).groupby("group").agg(col("val").sum().alias("total")).plan
)
groups4 = result4.column("group").to_pylist()
totals4 = result4.column("total").to_pylist()
d4 = dict(zip(groups4, totals4))
assert abs(d4["a"] - 6.0) < 1e-6, f"Filtered sum for a should be 6.0: {d4}"
assert abs(d4["b"] - 6.0) < 1e-6, f"Filtered sum for b should be 6.0: {d4}"
print("✅ Test 4: filter removes rows before grouped aggregation")

# ── Test 5: grouped min/max/mean now work (kernels exist) ────────────────────
table5 = pa.table({"g": ["x", "x", "y"], "v": pa.array([1.0, 2.0, 3.0], type=pa.float32())})
lf5 = LazyFrame(Scan(table5))

# min
r_min = compiler.compile_and_run(lf5.groupby("g").agg(col("v").min().alias("mn")).plan)
d_min = dict(zip(r_min.column("g").to_pylist(), r_min.column("mn").to_pylist()))
assert abs(d_min["x"] - 1.0) < 1e-5, f"min(x) should be 1.0: {d_min}"
assert abs(d_min["y"] - 3.0) < 1e-5, f"min(y) should be 3.0: {d_min}"

# max
r_max = compiler.compile_and_run(lf5.groupby("g").agg(col("v").max().alias("mx")).plan)
d_max = dict(zip(r_max.column("g").to_pylist(), r_max.column("mx").to_pylist()))
assert abs(d_max["x"] - 2.0) < 1e-5, f"max(x) should be 2.0: {d_max}"
assert abs(d_max["y"] - 3.0) < 1e-5, f"max(y) should be 3.0: {d_max}"

# mean
r_mean = compiler.compile_and_run(lf5.groupby("g").agg(col("v").mean().alias("av")).plan)
d_mean = dict(zip(r_mean.column("g").to_pylist(), r_mean.column("av").to_pylist()))
assert abs(d_mean["x"] - 1.5) < 1e-5, f"mean(x) should be 1.5: {d_mean}"
assert abs(d_mean["y"] - 3.0) < 1e-5, f"mean(y) should be 3.0: {d_mean}"

print("✅ Test 5: grouped min/max/mean produce correct results")

print("\nAll CustomOpsCompiler (Phase 0) tests passed! 🎉")
