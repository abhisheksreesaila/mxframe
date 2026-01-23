"""Test TPC-H Q1 Transposed Aggregator"""
import numpy as np
import time
from max_main import (
    Q1Accumulator, 
    encode_returnflag, 
    encode_linestatus, 
    compute_group_id
)

print("=== TPC-H Q1 Transposed Aggregator Test ===\n")

# Test 1: Encoding functions
print("Test 1: Encoding functions")
rf = np.array(['A', 'N', 'N', 'R', 'A'])
ls = np.array(['F', 'F', 'O', 'F', 'F'])

rf_enc = encode_returnflag(rf)
ls_enc = encode_linestatus(ls)
group_ids = compute_group_id(rf_enc, ls_enc)

print(f"returnflag: {rf} -> encoded: {rf_enc}")
print(f"linestatus: {ls} -> encoded: {ls_enc}")
print(f"group_ids: {group_ids}")
expected_groups = [0, 1, 2, 3, 0]  # (A,F), (N,F), (N,O), (R,F), (A,F)
assert list(group_ids) == expected_groups, f"Got {list(group_ids)}"
print("✓ Test 1 PASSED\n")

# Test 2: Small aggregation
print("Test 2: Small aggregation (5 rows)")
l_quantity = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
l_extendedprice = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
l_discount = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
l_tax = np.array([0.08, 0.08, 0.08, 0.08, 0.08])

acc = Q1Accumulator()
results = acc.aggregate(group_ids, l_quantity, l_extendedprice, l_discount, l_tax)

print("Results:")
for key, val in sorted(results.items()):
    print(f"  {key}: sum_qty={val['sum_qty']:.1f}, count={val['count_order']}")

# Verify group (A, F): rows 0 and 4
# sum_qty = 10 + 50 = 60
# sum_base_price = 100 + 500 = 600
# sum_disc_price = 100*0.95 + 500*0.75 = 95 + 375 = 470
# count = 2
assert results[('A', 'F')]['sum_qty'] == 60.0, f"Got {results[('A', 'F')]['sum_qty']}"
assert results[('A', 'F')]['count_order'] == 2
print("✓ Test 2 PASSED\n")

# Test 3: Larger benchmark
print("Test 3: Performance benchmark (100K rows)")
np.random.seed(42)
n_rows = 100_000

# Generate random TPC-H-like data
rf_chars = np.random.choice(['A', 'N', 'R'], n_rows)
ls_chars = np.where(rf_chars == 'N', 
                    np.random.choice(['F', 'O'], n_rows),
                    'F')  # A and R only pair with F

rf_enc = encode_returnflag(rf_chars)
ls_enc = encode_linestatus(ls_chars)
group_ids = compute_group_id(rf_enc, ls_enc)

l_quantity = np.random.uniform(1, 50, n_rows)
l_extendedprice = np.random.uniform(100, 10000, n_rows)
l_discount = np.random.uniform(0, 0.10, n_rows)
l_tax = np.random.uniform(0, 0.08, n_rows)

# Warmup
acc = Q1Accumulator()
_ = acc.aggregate(group_ids, l_quantity, l_extendedprice, l_discount, l_tax)

# Benchmark
start = time.perf_counter()
results = acc.aggregate(group_ids, l_quantity, l_extendedprice, l_discount, l_tax)
elapsed = time.perf_counter() - start

print(f"100K rows aggregated in {elapsed*1000:.2f}ms")
print("\nResults:")
for row in acc.to_dataframe():
    print(f"  ({row['l_returnflag']}, {row['l_linestatus']}): "
          f"sum_qty={row['sum_qty']:.0f}, "
          f"avg_price={row['avg_price']:.2f}, "
          f"count={row['count_order']}")

print("✓ Test 3 PASSED\n")

print("=== All tests passed! Q1 Aggregator ready ===")
