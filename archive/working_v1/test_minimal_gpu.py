#!/usr/bin/env python3
"""Minimal GPU kernel test - 10 elements only."""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

# Very small test data - 10 elements
SIZE = 10
mask = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1], dtype=np.float32)
values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.float32)

# Expected: 1*1 + 0*2 + 1*3 + 1*4 + 0*5 + 1*6 + 0*7 + 0*8 + 1*9 + 1*10
#         = 1 + 3 + 4 + 6 + 9 + 10 = 33
expected = np.sum(mask * values)
print(f"Test data: {SIZE} elements")
print(f"mask:   {mask}")
print(f"values: {values}")
print(f"Expected sum: {expected}")
print()

# Setup GPU
device = driver.Accelerator()
device_ref = DeviceRef.GPU()
session = engine.InferenceSession(devices=[device])

# Path to kernels folder
mojo_kernels = Path(os.getcwd()) / "kernels"

print("Building graph...")
with Graph(
    "test_minimal",
    input_types=[
        TensorType(DType.float32, (SIZE,), device_ref),
        TensorType(DType.float32, (SIZE,), device_ref),
    ],
    custom_extensions=[mojo_kernels],
) as graph:
    # Get inputs
    m, v = graph.inputs
    
    # Call custom kernel
    result = ops.custom(
        name="masked_sum_gpu_only",
        device=device_ref,
        values=[m, v],
        out_types=[TensorType(DType.float32, (1,), device_ref)],
    )[0]
    
    # Set output
    graph.output(result)

print("Compiling...")
model = session.load(graph)
print("Compiled!")

print("Executing...")
outputs = model.execute(
    driver.Tensor(mask, device),
    driver.Tensor(values, device),
)
result = float(outputs[0].to_numpy()[0])

print(f"\nResult: {result}")
print(f"Expected: {expected}")
print(f"Match: {'✅ YES' if abs(result - expected) < 0.01 else '❌ NO'}")
