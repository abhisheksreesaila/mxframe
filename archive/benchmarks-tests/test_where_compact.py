"""Test MXFrame.where() - Compaction filtering for TPC-H Q1"""
import pyarrow as pa
import numpy as np
from datetime import date
from max import driver
from max_main import MXFrame

print("=== Testing MXFrame.where() - Compact Filtering ===\n")

# Test data: 5 dates, 3 should pass filter
test_dates = [
    date(1998, 9, 1),   # 10470 - pass
    date(1998, 9, 2),   # 10471 - pass (equal to cutoff)
    date(1998, 9, 3),   # 10472 - fail
    date(1998, 12, 1),  # 10561 - fail
    date(1992, 1, 1),   # 8035 - pass
]

arrow_dates = pa.array(test_dates, type=pa.date32())
mxf = MXFrame(arrow_dates)

print("Test 1: where() returns compact result")
print(f"Input shape: {mxf.tensor.shape}")

# TPC-H Q1 filter: l_shipdate <= 1998-09-02 (epoch day 10471)
compact_mxf, mask, count = mxf.where("l_shipdate <= 10471")

print(f"Output shape: {compact_mxf.tensor.shape}")
print(f"Match count: {count}")

# Verify compaction
result = compact_mxf.to_numpy()
expected = [10470, 10471, 8035]  # Only passing dates, in order
print(f"Compact result: {list(result)}")
print(f"Expected:       {expected}")

assert list(result) == expected, f"Mismatch! Got {list(result)}"
assert count == 3, f"Count mismatch! Got {count}"
print("✓ Test 1 PASSED\n")

# Test 2: Mask can be reused for other columns
print("Test 2: Mask tensor for multi-column filtering")
if 'cuda' in str(mask):
    mask_np = mask.copy(device=driver.CPU()).to_numpy()
else:
    mask_np = mask.to_numpy()
print(f"Mask: {mask_np}")
expected_mask = [True, True, False, False, True]
assert list(mask_np) == expected_mask, f"Mask mismatch! Got {list(mask_np)}"
print("✓ Test 2 PASSED\n")

# Test 3: Compare old sql() vs new where()
print("Test 3: sql() vs where() comparison")
sql_result = mxf.sql("SELECT * FROM data WHERE l_shipdate <= 10471")
sql_values = sql_result.to_numpy()
print(f"sql() result (zeros in place): {list(sql_values)}")
print(f"where() result (compact):      {list(result)}")
print(f"sql() shape: {sql_result.tensor.shape} (same as input)")
print(f"where() shape: {compact_mxf.tensor.shape} (compacted)")
assert sql_result.tensor.shape[0] == 5, "sql() should keep same size"
assert compact_mxf.tensor.shape[0] == 3, "where() should compact"
print("✓ Test 3 PASSED\n")

# Test 4: Large scale benchmark
print("Test 4: Performance benchmark (100K rows)")
import time

np.random.seed(42)
start_epoch = 8035   # 1992-01-01
end_epoch = 10561    # 1998-12-01
random_epochs = np.random.randint(start_epoch, end_epoch + 1, size=100_000).astype(np.int32)
arrow_large = pa.array(random_epochs, type=pa.date32())
mxf_large = MXFrame(arrow_large)

# Warmup
_ = mxf_large.where("l_shipdate <= 10471")

# Benchmark
start = time.perf_counter()
compact, mask, count = mxf_large.where("l_shipdate <= 10471")
elapsed = time.perf_counter() - start

filter_ratio = count / 100_000 * 100
print(f"100K rows filtered in {elapsed*1000:.2f}ms")
print(f"Matching rows: {count:,} ({filter_ratio:.1f}%)")
print(f"Output shape: {compact.tensor.shape}")
print("✓ Test 4 PASSED\n")

print("=== All tests passed! Compact filtering ready for TPC-H Q1 ===")
