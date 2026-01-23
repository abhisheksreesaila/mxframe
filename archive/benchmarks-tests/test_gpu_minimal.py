#!/usr/bin/env python3
"""Minimal GPU test to isolate the issue."""

import numpy as np
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

# Check GPU availability
print(f"GPU count: {driver.accelerator_count()}")

if driver.accelerator_count() == 0:
    print("No GPU available!")
    exit(1)

# Setup
device = driver.Accelerator()
device_ref = DeviceRef.GPU(0)
session = engine.InferenceSession(devices=[device])

n = 1000

# Test 1: Simple sum
print("\n--- Test 1: Simple ops.sum ---")

class SimpleSumGraph:
    def __call__(self, x):
        return ops.sum(x)

graph = Graph(
    "simple_sum",
    SimpleSumGraph(),
    input_types=[TensorType(DType.float32, (n,), device_ref)]
)

try:
    model = session.load(graph)
    print("Graph compiled")
    
    data = np.ones(n, dtype=np.float32)
    t = driver.Tensor(data, device)
    print(f"Input tensor on: {t}")
    
    result = model.execute(t)[0]
    cpu_result = result.copy(device=driver.CPU())
    print(f"Result: {cpu_result.to_numpy()[0]} (expected: {n})")
except Exception as e:
    print(f"FAILED: {e}")


# Test 2: Sum with equal mask
print("\n--- Test 2: ops.equal + ops.cast + ops.mul + ops.sum ---")

class MaskedSumGraph:
    def __init__(self, device_ref):
        self.device_ref = device_ref
        
    def __call__(self, x, indices):
        # Create mask where indices == 0
        zero = ops.constant(0, dtype=DType.int32, device=self.device_ref)
        mask = ops.equal(indices, zero)
        mask_f = ops.cast(mask, DType.float32)
        masked = ops.mul(x, mask_f)
        return ops.sum(masked)

graph2 = Graph(
    "masked_sum",
    MaskedSumGraph(device_ref),
    input_types=[
        TensorType(DType.float32, (n,), device_ref),
        TensorType(DType.int32, (n,), device_ref),
    ]
)

try:
    model2 = session.load(graph2)
    print("Graph compiled")
    
    data = np.ones(n, dtype=np.float32)
    indices = np.zeros(n, dtype=np.int32)  # All zeros -> all should pass
    
    t_data = driver.Tensor(data, device)
    t_idx = driver.Tensor(indices, device)
    
    result = model2.execute(t_data, t_idx)[0]
    cpu_result = result.copy(device=driver.CPU())
    print(f"Result: {cpu_result.to_numpy()[0]} (expected: {n})")
except Exception as e:
    print(f"FAILED: {e}")

print("\n--- Done ---")
