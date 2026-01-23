"""Test TPC-H Q1 style date filtering: l_shipdate <= '1998-09-02' (epoch day 10471)"""
import pyarrow as pa
import numpy as np
from datetime import date, timedelta
from max import driver
from max_main import MXFrame

print("=== TPC-H Q1 Date Filter Test ===\n")

# Simulate TPC-H lineitem l_shipdate column
# TPC-H Q1 filter: l_shipdate <= date('1998-12-01') - interval '90' day
# = l_shipdate <= '1998-09-02' = epoch day 10471

print("Test 1: Create MXFrame from Date32 array")
test_dates = [
    date(1998, 9, 1),   # Should pass (before cutoff)
    date(1998, 9, 2),   # Should pass (equal to cutoff)
    date(1998, 9, 3),   # Should fail (after cutoff)
    date(1998, 12, 1),  # Should fail (90 days after cutoff)
    date(1992, 1, 1),   # Should pass (way before)
]

arrow_dates = pa.array(test_dates, type=pa.date32())
mxf = MXFrame(arrow_dates)

print(f"MXFrame dtype: {mxf.dtype}")
print(f"MXFrame tensor: {mxf.tensor}")

# Get tensor values for verification
if 'cuda' in str(mxf.tensor):
    tensor_values = mxf.tensor.copy(device=driver.CPU()).to_numpy()
else:
    tensor_values = mxf.tensor.to_numpy()
print(f"Tensor values (epoch days): {tensor_values}")
print("✓ Test 1 PASSED\n")

# Test 2: Apply TPC-H Q1 filter
print("Test 2: Apply filter l_shipdate <= 10471 (1998-09-02)")
# TPC-H Q1 cutoff: 1998-09-02 = epoch day 10471
result_mxf = mxf.sql("SELECT * FROM data WHERE l_shipdate <= 10471")

if 'cuda' in str(result_mxf.tensor):
    result_values = result_mxf.tensor.copy(device=driver.CPU()).to_numpy()
else:
    result_values = result_mxf.tensor.to_numpy()

print(f"Result values: {result_values}")

# Expected: dates <= 1998-09-02 keep their epoch day, others become 0
# 1998-09-01 = 10470, 1998-09-02 = 10471, 1992-01-01 = 8035
expected = [10470, 10471, 0, 0, 8035]  # First 2 and last pass, middle 2 fail
print(f"Expected:      {expected}")

assert list(result_values) == expected, f"Mismatch! Got {list(result_values)}"
print("✓ Test 2 PASSED\n")

# Test 3: Large scale benchmark (simulating TPC-H scale)
print("Test 3: Large scale benchmark (1M dates)")
import time

# Generate 1M random dates in TPC-H range (1992-01-01 to 1998-12-01)
np.random.seed(42)
start_epoch = 8035   # 1992-01-01
end_epoch = 10561    # 1998-12-01
random_epochs = np.random.randint(start_epoch, end_epoch + 1, size=1_000_000).astype(np.int32)

# Create Arrow Date32 from epoch days
arrow_large = pa.array(random_epochs, type=pa.date32())
mxf_large = MXFrame(arrow_large)

# Warm up
_ = mxf_large.sql("SELECT * FROM data WHERE l_shipdate <= 10471")

# Benchmark
start = time.perf_counter()
result = mxf_large.sql("SELECT * FROM data WHERE l_shipdate <= 10471")
elapsed = time.perf_counter() - start

# Count filtered rows
if 'cuda' in str(result.tensor):
    res_np = result.tensor.copy(device=driver.CPU()).to_numpy()
else:
    res_np = result.tensor.to_numpy()
    
passing_rows = np.count_nonzero(res_np)
print(f"1M rows filtered in {elapsed*1000:.2f}ms")
print(f"Rows passing filter: {passing_rows:,} ({100*passing_rows/1_000_000:.1f}%)")
print("✓ Test 3 PASSED\n")

print("=== All tests passed! Ready for TPC-H Q1 ===")
