"""Phase 1 verification: all grouped aggregations."""
import pyarrow as pa
from pathlib import Path
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Scan
from mxframe.custom_ops import CustomOpsCompiler

compiler = CustomOpsCompiler()

# Helper to turn result into a {key: value} dict
def as_dict(result, key_col, val_col):
    return dict(zip(
        result.column(key_col).to_pylist(),
        result.column(val_col).to_pylist(),
    ))

# Data for all Phase 1 tests:
#   group x: [1.0, 2.0]  → sum=3  min=1  max=2  mean=1.5  count=2
#   group y: [3.0]        → sum=3  min=3  max=3  mean=3.0  count=1
table = pa.table({"g": ["x", "x", "y"], "v": [1.0, 2.0, 3.0]})
lf = LazyFrame(Scan(table))

# ── Test 1: grouped sum (verifying Phase 0 still works) ──
r = compiler.compile_and_run(lf.groupby("g").agg(col("v").sum().alias("s")).plan)
d = as_dict(r, "g", "s")
assert abs(d["x"] - 3.0) < 1e-5 and abs(d["y"] - 3.0) < 1e-5, f"sum: {d}"
assert r.column_names[0] == "g", "key column must be first"
print("✅ grouped sum  — x=3.0, y=3.0")

# ── Test 2: grouped min ──
r = compiler.compile_and_run(lf.groupby("g").agg(col("v").min().alias("mn")).plan)
d = as_dict(r, "g", "mn")
assert abs(d["x"] - 1.0) < 1e-5, f"min x: {d}"
assert abs(d["y"] - 3.0) < 1e-5, f"min y: {d}"
print("✅ grouped min  — x=1.0, y=3.0")

# ── Test 3: grouped max ──
r = compiler.compile_and_run(lf.groupby("g").agg(col("v").max().alias("mx")).plan)
d = as_dict(r, "g", "mx")
assert abs(d["x"] - 2.0) < 1e-5, f"max x: {d}"
assert abs(d["y"] - 3.0) < 1e-5, f"max y: {d}"
print("✅ grouped max  — x=2.0, y=3.0")

# ── Test 4: grouped mean ──
r = compiler.compile_and_run(lf.groupby("g").agg(col("v").mean().alias("avg")).plan)
d = as_dict(r, "g", "avg")
assert abs(d["x"] - 1.5) < 1e-5, f"mean x: {d}"
assert abs(d["y"] - 3.0) < 1e-5, f"mean y: {d}"
print("✅ grouped mean — x=1.5, y=3.0")

# ── Test 5: grouped count ──
r = compiler.compile_and_run(lf.groupby("g").agg(col("v").count().alias("cnt")).plan)
d = as_dict(r, "g", "cnt")
assert abs(d["x"] - 2.0) < 1e-5, f"count x: {d}"
assert abs(d["y"] - 1.0) < 1e-5, f"count y: {d}"
print("✅ grouped count — x=2, y=1")

# ── Test 6: all 5 aggs in a single groupby call ──
r = compiler.compile_and_run(lf.groupby("g").agg(
    col("v").sum().alias("s"),
    col("v").min().alias("mn"),
    col("v").max().alias("mx"),
    col("v").mean().alias("avg"),
    col("v").count().alias("cnt"),
).plan)
assert set(r.column_names) == {"g", "s", "mn", "mx", "avg", "cnt"}
ds = as_dict(r, "g", "s");   assert abs(ds["x"]  - 3.0) < 1e-5
dm = as_dict(r, "g", "mn");  assert abs(dm["x"]  - 1.0) < 1e-5
dmx= as_dict(r, "g", "mx");  assert abs(dmx["x"] - 2.0) < 1e-5
da = as_dict(r, "g", "avg"); assert abs(da["x"]  - 1.5) < 1e-5
dc = as_dict(r, "g", "cnt"); assert abs(dc["x"]  - 2.0) < 1e-5
print("✅ all 5 aggs in one groupby — correct for both groups")

# ── Test 7: filter + grouped min/max/mean/count ──
table7 = pa.table({
    "g":    ["a", "b", "a", "b", "a"],
    "v":    [10.0, 20.0, 30.0, 40.0, 50.0],
    "keep": [1,    1,    0,    1,    1   ],  # row 2 (v=30, g=a) filtered out
})
lf7 = LazyFrame(Scan(table7))
r7 = compiler.compile_and_run(
    lf7.filter(col("keep") > lit(0)).groupby("g").agg(
        col("v").min().alias("mn"),
        col("v").max().alias("mx"),
        col("v").mean().alias("avg"),
        col("v").count().alias("cnt"),
    ).plan
)
da7 = {
    "mn":  as_dict(r7, "g", "mn"),
    "mx":  as_dict(r7, "g", "mx"),
    "avg": as_dict(r7, "g", "avg"),
    "cnt": as_dict(r7, "g", "cnt"),
}
# group a (after filter): [10.0, 50.0]  → min=10 max=50 mean=30 count=2
# group b                : [20.0, 40.0]  → min=20 max=40 mean=30 count=2
assert abs(da7["mn"]["a"]  - 10.0) < 1e-5, f"filter+min a: {da7}"
assert abs(da7["mx"]["a"]  - 50.0) < 1e-5, f"filter+max a: {da7}"
assert abs(da7["avg"]["a"] - 30.0) < 1e-5, f"filter+mean a: {da7}"
assert abs(da7["cnt"]["a"] - 2.0)  < 1e-5, f"filter+count a: {da7}"
assert abs(da7["mn"]["b"]  - 20.0) < 1e-5, f"filter+min b: {da7}"
assert abs(da7["mx"]["b"]  - 40.0) < 1e-5, f"filter+max b: {da7}"
print("✅ filter + grouped min/max/mean/count — all correct")

# ── Test 8: unknown grouped op raises NotImplementedError ──
# Build a fake Expr with an unsupported op and verify the guard fires.
from mxframe.lazy_expr import Expr
from mxframe.lazy_frame import Aggregate
bad_expr = Expr("stddev", col("v"))
bad_plan = LazyFrame(Scan(table)).groupby("g").agg(bad_expr).plan
try:
    compiler.compile_and_run(bad_plan)
    assert False, "Should have raised NotImplementedError for 'stddev'"
except NotImplementedError as e:
    assert "stddev" in str(e), f"Error should mention op name: {e}"
print("✅ unknown grouped op raises NotImplementedError")

print("\nPhase 1 complete — all grouped aggregations pass! 🎉")
