"""Test apply_mask() - Multi-column filtering for TPC-H Q1"""
import pyarrow as pa
import numpy as np
from datetime import date
from max import driver
from max_main import MXFrame, apply_mask

print("=== Testing apply_mask() - Multi-Column Filtering ===\n")

# Simulate TPC-H lineitem columns (5 rows)
print("Test 1: Apply mask to multiple columns")

# Dates - some pass, some fail the filter
l_shipdate = pa.array([
    date(1998, 9, 1),   # pass
    date(1998, 9, 2),   # pass
    date(1998, 9, 3),   # fail
    date(1998, 12, 1),  # fail
    date(1992, 1, 1),   # pass
], type=pa.date32())

# Other columns to filter
l_quantity = np.array([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float32)
l_extendedprice = np.array([100.0, 200.0, 300.0, 400.0, 500.0], dtype=np.float32)
l_discount = np.array([0.05, 0.10, 0.15, 0.20, 0.25], dtype=np.float32)
l_tax = np.array([0.08, 0.08, 0.08, 0.08, 0.08], dtype=np.float32)
l_returnflag = np.array(['A', 'N', 'N', 'R', 'A'])
l_linestatus = np.array(['F', 'F', 'O', 'F', 'F'])

# Get mask from date filter
shipdate_mxf = MXFrame(l_shipdate)
_, mask, count = shipdate_mxf.where("l_shipdate <= 10471")

print(f"Filter: l_shipdate <= 10471 (1998-09-02)")
print(f"Rows passing: {count} out of 5")

# Apply mask to all other columns
filtered = apply_mask({
    'l_quantity': l_quantity,
    'l_extendedprice': l_extendedprice,
    'l_discount': l_discount,
    'l_tax': l_tax,
    'l_returnflag': l_returnflag,
    'l_linestatus': l_linestatus,
}, mask)

print(f"\nFiltered results:")
for name, arr in filtered.items():
    print(f"  {name}: {arr}")

# Verify: rows 0, 1, 4 should pass
assert list(filtered['l_quantity']) == [10.0, 20.0, 50.0], f"Got {filtered['l_quantity']}"
assert list(filtered['l_returnflag']) == ['A', 'N', 'A'], f"Got {filtered['l_returnflag']}"
assert count == 3
print("\n✓ Test 1 PASSED\n")

# Test 2: Larger scale
print("Test 2: Performance benchmark (100K rows)")
import time

np.random.seed(42)
n_rows = 100_000

# Generate random data
start_epoch = 8035   # 1992-01-01
end_epoch = 10561    # 1998-12-01
random_epochs = np.random.randint(start_epoch, end_epoch + 1, size=n_rows).astype(np.int32)
l_shipdate_large = pa.array(random_epochs, type=pa.date32())

l_qty_large = np.random.uniform(1, 50, n_rows).astype(np.float32)
l_price_large = np.random.uniform(100, 10000, n_rows).astype(np.float32)
l_disc_large = np.random.uniform(0, 0.10, n_rows).astype(np.float32)
l_tax_large = np.random.uniform(0, 0.08, n_rows).astype(np.float32)

# Filter
shipdate_mxf = MXFrame(l_shipdate_large)
_, mask, count = shipdate_mxf.where("l_shipdate <= 10471")

# Warmup
_ = apply_mask({'qty': l_qty_large[:1000]}, mask.copy(device=driver.CPU()).to_numpy()[:1000])

# Benchmark
start = time.perf_counter()
filtered = apply_mask({
    'l_quantity': l_qty_large,
    'l_extendedprice': l_price_large,
    'l_discount': l_disc_large,
    'l_tax': l_tax_large,
}, mask)
elapsed = time.perf_counter() - start

print(f"100K rows, 4 columns filtered in {elapsed*1000:.2f}ms")
print(f"Rows passing: {count:,} ({100*count/n_rows:.1f}%)")
print(f"Output shapes: {[arr.shape for arr in filtered.values()]}")
print("✓ Test 2 PASSED\n")

print("=== All tests passed! apply_mask() ready for TPC-H Q1 ===")
