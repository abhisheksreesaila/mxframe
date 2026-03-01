"""Phase 0 verification: GraphCompiler tests."""
import pyarrow as pa
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Scan
from mxframe.compiler import GraphCompiler

# ── lazy_expr tests ──────────────────────────────────────────────────────────
from mxframe.lazy_expr import Expr

e = col("a")
assert e.sum().op == "sum"
assert e.min().op == "min"
assert e.max().op == "max"
assert e.mean().op == "mean"
assert e.count().op == "count"
and_expr = (col("a") > lit(0)) & (col("b") < lit(10))
assert and_expr.op == "and"
print("✅ lazy_expr: count() and & operator OK")

# ── GraphCompiler: Test 1 — projection ──────────────────────────────────────
table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
lf = LazyFrame(Scan(table))
result = GraphCompiler().compile_and_run(lf.select(col("a") + lit(10)).plan)
assert result.column(0).to_pylist() == [11, 12, 13], f"Projection failed: {result}"
print("✅ Test 1: projection")

# ── GraphCompiler: Test 2 — global sum ──────────────────────────────────────
lf2 = LazyFrame(Scan(pa.table({"x": [1.0, 2.0, 3.0, 4.0]})))
result2 = GraphCompiler().compile_and_run(
    lf2.groupby().agg(col("x").sum().alias("total")).plan
)
assert abs(result2.column("total").to_pylist()[0] - 10.0) < 1e-6
print("✅ Test 2: global sum")

# ── GraphCompiler: Test 3 — filter REMOVES rows ──────────────────────────────
table3 = pa.table({"a": [1, 2, 3, 4, 5], "b": [10.0, 20.0, 30.0, 40.0, 50.0]})
lf3 = LazyFrame(Scan(table3))
result3 = GraphCompiler().compile_and_run(lf3.filter(col("a") > lit(2)).plan)
assert result3.num_rows == 3, f"Filter should produce 3 rows, got {result3.num_rows}"
assert result3.column("a").to_pylist() == [3, 4, 5]
print("✅ Test 3: filter removes rows (not zeros)")

# ── GraphCompiler: Test 4 — mean correct after filter ───────────────────────
result4 = GraphCompiler().compile_and_run(
    lf3.filter(col("a") > lit(2)).groupby().agg(col("b").mean().alias("avg_b")).plan
)
avg_b = result4.column("avg_b").to_pylist()[0]
assert abs(avg_b - 40.0) < 1e-6, f"Mean should be 40.0, got {avg_b} (old bug was 24.0)"
print("✅ Test 4: mean() correct after filter (old bug was 24.0)")

# ── GraphCompiler: Test 5 — count global ────────────────────────────────────
table5 = pa.table({"v": [10, 20, 30, 40, 50]})
lf5 = LazyFrame(Scan(table5))
result5 = GraphCompiler().compile_and_run(
    lf5.groupby().agg(col("v").count().alias("n")).plan
)
assert result5.column("n").to_pylist()[0] == 5
print("✅ Test 5: count() global")

# ── GraphCompiler: Test 6 — count after filter ──────────────────────────────
result6 = GraphCompiler().compile_and_run(
    lf5.filter(col("v") > lit(25)).groupby().agg(col("v").count().alias("n")).plan
)
assert result6.column("n").to_pylist()[0] == 3
print("✅ Test 6: count() after filter")

print("\nAll GraphCompiler (Phase 0) tests passed! ✅")
