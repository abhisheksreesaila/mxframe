"""Phase 5 verification: Join operations (CPU path)."""
import pyarrow as pa
import numpy as np
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Join

# ═══════════════════════════════════════════════════════════════════════
# Test data
# ═══════════════════════════════════════════════════════════════════════
orders = pa.table({
    "o_key": [1, 2, 3, 4, 5],
    "c_key": [10, 20, 10, 30, 20],
    "o_amount": [100.0, 200.0, 150.0, 300.0, 250.0],
})
customers = pa.table({
    "c_key": [10, 20, 30, 40],
    "c_name": ["Alice", "Bob", "Carol", "Dave"],
})
items = pa.table({
    "o_key": [1, 1, 2, 3, 5, 5],
    "i_price": [10.0, 20.0, 50.0, 30.0, 40.0, 60.0],
})

# ═══════════════════════════════════════════════════════════════════════
# Test 1: simple two-table join (on= shorthand)
# ═══════════════════════════════════════════════════════════════════════
lf_orders = LazyFrame(orders)
lf_customers = LazyFrame(customers)

result = lf_orders.join(lf_customers, on="c_key").compute(device="cpu")
assert result.num_rows == 5, f"Expected 5 rows, got {result.num_rows}"
assert "c_name" in result.column_names, f"Missing c_name: {result.column_names}"
assert "o_amount" in result.column_names, f"Missing o_amount: {result.column_names}"
# Check join correctness: order 1 (c_key=10) should match Alice
r_dict = {
    int(result.column("o_key")[i].as_py()): result.column("c_name")[i].as_py()
    for i in range(result.num_rows)
}
assert r_dict[1] == "Alice", f"order 1 should be Alice, got {r_dict[1]}"
assert r_dict[2] == "Bob", f"order 2 should be Bob, got {r_dict[2]}"
assert r_dict[4] == "Carol", f"order 4 should be Carol, got {r_dict[4]}"
print("✅ Test 1: simple two-table join — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 2: join with left_on / right_on
# ═══════════════════════════════════════════════════════════════════════
left = pa.table({"id_a": [1, 2, 3], "val": [10, 20, 30]})
right = pa.table({"id_b": [2, 3, 4], "val2": [200, 300, 400]})
result2 = LazyFrame(left).join(LazyFrame(right), left_on="id_a", right_on="id_b").compute(device="cpu")
assert result2.num_rows == 2, f"Expected 2 rows, got {result2.num_rows}"
assert "val" in result2.column_names and "val2" in result2.column_names
print("✅ Test 2: left_on/right_on join — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 3: join + filter
# ═══════════════════════════════════════════════════════════════════════
result3 = (
    lf_orders
    .join(lf_customers, on="c_key")
    .filter(col("o_amount") > lit(150.0))
    .compute(device="cpu")
)
assert result3.num_rows == 3, f"Expected 3 rows (200, 300, 250), got {result3.num_rows}"
amounts = sorted(result3.column("o_amount").to_pylist())
assert amounts == [200.0, 250.0, 300.0], f"Unexpected amounts: {amounts}"
print("✅ Test 3: join + filter — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 4: join + groupby + agg
# ═══════════════════════════════════════════════════════════════════════
result4 = (
    lf_orders
    .join(lf_customers, on="c_key")
    .groupby("c_name")
    .agg(col("o_amount").sum().alias("total"))
    .compute(device="cpu")
)
totals = dict(zip(
    result4.column("c_name").to_pylist(),
    result4.column("total").to_pylist(),
))
assert abs(totals["Alice"] - 250.0) < 1e-5, f"Alice total: {totals['Alice']}"
assert abs(totals["Bob"] - 450.0) < 1e-5, f"Bob total: {totals['Bob']}"
assert abs(totals["Carol"] - 300.0) < 1e-5, f"Carol total: {totals['Carol']}"
print("✅ Test 4: join + groupby + agg — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 5: three-way chained join (A.join(B).join(C))
# ═══════════════════════════════════════════════════════════════════════
lf_items = LazyFrame(items)
result5 = (
    lf_orders
    .join(lf_items, on="o_key")
    .join(lf_customers, on="c_key")
    .compute(device="cpu")
)
assert result5.num_rows == 6, f"Expected 6 rows, got {result5.num_rows}"
assert "c_name" in result5.column_names
assert "i_price" in result5.column_names
print("✅ Test 5: three-way chained join — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 6: join + sort + limit
# ═══════════════════════════════════════════════════════════════════════
result6 = (
    lf_orders
    .join(lf_customers, on="c_key")
    .sort("o_amount", descending=True)
    .limit(3)
    .compute(device="cpu")
)
assert result6.num_rows == 3, f"Expected 3 rows, got {result6.num_rows}"
top_amounts = result6.column("o_amount").to_pylist()
assert top_amounts == [300.0, 250.0, 200.0], f"Unexpected top amounts: {top_amounts}"
print("✅ Test 6: join + sort + limit — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 7: non-matching rows excluded (inner join)
# ═══════════════════════════════════════════════════════════════════════
left_nm = pa.table({"k": [1, 2, 3], "v": [10, 20, 30]})
right_nm = pa.table({"k": [4, 5, 6], "w": [40, 50, 60]})
result7 = LazyFrame(left_nm).join(LazyFrame(right_nm), on="k").compute(device="cpu")
assert result7.num_rows == 0, f"Expected 0 rows (no matches), got {result7.num_rows}"
print("✅ Test 7: no matching rows — passed")

# ═══════════════════════════════════════════════════════════════════════
# Test 8: duplicate key join (many-to-many)
# ═══════════════════════════════════════════════════════════════════════
left_dup = pa.table({"k": [1, 1, 2], "a": [10, 20, 30]})
right_dup = pa.table({"k": [1, 1, 2], "b": [100, 200, 300]})
result8 = LazyFrame(left_dup).join(LazyFrame(right_dup), on="k").compute(device="cpu")
# 1×1=2 left rows with k=1 × 2 right rows with k=1 = 4 + 1×1=1 row with k=2 = 5
assert result8.num_rows == 5, f"Expected 5 rows (m2m), got {result8.num_rows}"
print("✅ Test 8: many-to-many join — passed")

print("\n🎉 All Phase 5 tests passed!")
