#!/usr/bin/env python3
"""Test basic GPU tensor operations."""

from max import driver
import numpy as np

print(f"GPU count: {driver.accelerator_count()}")

if driver.accelerator_count() > 0:
    gpu = driver.Accelerator()
    cpu = driver.CPU()
    
    # Create tensor on CPU, copy to GPU, copy back
    data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    t_cpu = driver.Tensor(data, cpu)
    print(f"CPU tensor: {t_cpu.to_numpy()}")
    
    t_gpu = t_cpu.copy(device=gpu)
    print(f"GPU tensor created: {t_gpu}")
    
    t_back = t_gpu.copy(device=cpu)
    print(f"Copied back: {t_back.to_numpy()}")
    print("Basic GPU transfers work!")
else:
    print("No GPU available")
