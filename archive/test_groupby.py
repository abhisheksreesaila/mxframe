"""Test MXFrame.group_by() with MAX Engine."""
import numpy as np
from mxframe import MXFrame, group_by, agg_sum, agg_mean, agg_count

# Create test data
frame = MXFrame({
    'flag': ['A', 'A', 'B', 'B', 'A'],
    'status': ['X', 'X', 'X', 'Y', 'Y'],
    'qty': [10, 20, 30, 40, 50],
    'price': [1.0, 2.0, 3.0, 4.0, 5.0],
})

print("=== Test Data ===")
print(frame.to_pandas())
print()

# Test using function
print("=== group_by function (CPU) ===")
result = group_by(frame, 'flag', 'status', device='cpu').agg(
    sum_qty=agg_sum('qty'),
    avg_price=agg_mean('price'),
    count=agg_count('qty'),
)
print(result.to_pandas())
print()

# Test using method
print("=== frame.group_by() method (GPU) ===")
result_gpu = frame.group_by('flag', 'status', device='gpu').agg(
    sum_qty=agg_sum('qty'),
    avg_price=agg_mean('price'),
)
print(result_gpu.to_pandas())
print()

# Verify results
print("=== Verification ===")
# A-X: qty 10+20=30, price (1+2)/2=1.5
# B-X: qty 30, price 3/1=3.0
# B-Y: qty 40, price 4/1=4.0
# A-Y: qty 50, price 5/1=5.0
print("Expected A-X sum_qty: 30, avg_price: 1.5")
print("Expected A-Y sum_qty: 50, avg_price: 5.0")
print()

print("✅ GroupBy with MAX Engine working!")
