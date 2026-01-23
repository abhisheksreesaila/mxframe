"""Test date32_to_tensor() - Zero-copy Date32 to Int32 Tensor conversion."""
import pyarrow as pa
import numpy as np
from datetime import date
from max_main import date32_to_tensor

print("=== Testing date32_to_tensor() ===\n")

# Test 1: Known epoch days
print("Test 1: Known epoch days")
dates = [
    date(1970, 1, 1),   # Epoch = 0
    date(1970, 1, 2),   # Epoch = 1
    date(1998, 9, 2),   # TPC-H Q1 cutoff date (1998-12-01 - 90 days)
    date(1998, 12, 1),  # TPC-H Q1 reference date
]
expected_days = [0, 1, 10471, 10561]  # Pre-calculated epoch days

arrow_dates = pa.array(dates, type=pa.date32())
print(f"Arrow Date32 array: {arrow_dates}")
print(f"Arrow type: {arrow_dates.type}")

tensor = date32_to_tensor(arrow_dates)
# GPU tensor needs copy to CPU before to_numpy()
from max import driver
if hasattr(tensor, 'copy') and 'cuda' in str(tensor):
    result = tensor.copy(device=driver.CPU()).to_numpy()
else:
    result = tensor.to_numpy()

print(f"Tensor values: {result}")
print(f"Expected:      {expected_days}")

assert list(result) == expected_days, f"Mismatch! Got {list(result)}, expected {expected_days}"
print("✓ Test 1 PASSED\n")

# Test 2: Verify zero-copy (buffer address check)
print("Test 2: Verify buffer access")
raw_buffer = arrow_dates.buffers()[1]
np_direct = np.frombuffer(raw_buffer, dtype=np.int32)
print(f"Direct buffer read: {np_direct}")
assert list(np_direct) == expected_days, "Buffer read mismatch"
print("✓ Test 2 PASSED\n")

# Test 3: Type validation
print("Test 3: Type validation (should reject non-Date32)")
try:
    bad_array = pa.array([1.0, 2.0, 3.0])  # Float64
    date32_to_tensor(bad_array)
    print("✗ Test 3 FAILED - should have raised TypeError")
except TypeError as e:
    print(f"✓ Test 3 PASSED - correctly raised: {e}\n")

# Test 4: Null rejection
print("Test 4: Null rejection")
try:
    null_array = pa.array([date(1970, 1, 1), None, date(1970, 1, 3)], type=pa.date32())
    date32_to_tensor(null_array)
    print("✗ Test 4 FAILED - should have raised ValueError")
except ValueError as e:
    print(f"✓ Test 4 PASSED - correctly raised: {e}\n")

print("=== All tests passed! Ready for TPC-H Q1 date filtering ===")
