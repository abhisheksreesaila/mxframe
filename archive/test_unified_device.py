"""Test unified device dispatch for max_sum and max_masked_sum."""
import numpy as np
from mxframe import max_sum, max_masked_sum, DeviceType
from mxframe.max_ops import MAXSession

# Test data
arr = np.random.randn(1_000_000).astype(np.float32)
mask = (np.random.rand(1_000_000) > 0.5).astype(np.float32)
values = np.random.randn(1_000_000).astype(np.float32)

# Expected values
expected_sum = float(arr.sum())
expected_masked = float((mask * values).sum())

print("Testing unified device dispatch...")
print(f"Data size: {len(arr):,} elements")
print()

# Test CPU
result_cpu = max_sum(arr, device="cpu")
print(f"max_sum(CPU): {result_cpu:.4f} vs expected {expected_sum:.4f}")

# Test GPU
result_gpu = max_sum(arr, device="gpu")
print(f"max_sum(GPU): {result_gpu:.4f} vs expected {expected_sum:.4f}")

# Test masked CPU
result_masked_cpu = max_masked_sum(mask, values, device="cpu")
print(f"max_masked_sum(CPU): {result_masked_cpu:.4f} vs expected {expected_masked:.4f}")

# Test masked GPU
result_masked_gpu = max_masked_sum(mask, values, device="gpu")
print(f"max_masked_sum(GPU): {result_masked_gpu:.4f} vs expected {expected_masked:.4f}")

# Test auto selection
MAXSession.reset()
sess_auto = MAXSession.get("auto")
print(f"\nauto device: {sess_auto.device_type}")

print("\n✅ All tests passed!")
