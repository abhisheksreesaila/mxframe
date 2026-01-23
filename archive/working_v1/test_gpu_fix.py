#!/usr/bin/env python3
"""Test GPU with Modular team's recommended tensor transfer pattern."""

import numpy as np
import os

# Set sync mode BEFORE importing max
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

from max import engine, driver
from max.graph import Graph, ops, TensorType, DeviceRef
from max.dtype import DType

def test_sum_gpu():
    print("🔧 Setting up GPU...")
    device = driver.Accelerator(id=0)
    device_ref = DeviceRef.GPU(0)
    session = engine.InferenceSession(devices=[device])
    
    SIZE = 1024
    
    def simple_sum(x):
        return ops.sum(x)
    
    print("⚙️  Building graph...")
    graph = Graph("test_sum", simple_sum, 
                  input_types=[TensorType(DType.float32, (SIZE,), device_ref)])
    
    print("📦 Compiling...")
    model = session.load(graph)
    print("✅ Compiled!")
    
    # Generate test data
    np.random.seed(42)
    data = np.random.randn(SIZE).astype(np.float32)
    expected = np.sum(data)
    
    print("🚀 Executing with NEW tensor transfer pattern...")
    
    # FIX: Use from_numpy().to(device) pattern (Modular team recommendation)
    gpu_tensor = driver.Tensor.from_numpy(data).to(device)
    
    output = model.execute(gpu_tensor)
    
    # FIX: Copy output to CPU before reading
    cpu_output = output[0].to(driver.CPU()).to_numpy()
    result = float(cpu_output.item() if cpu_output.ndim == 0 else cpu_output[0])
    
    print(f"\n📊 Results:")
    print(f"   Expected: {expected:.4f}")
    print(f"   Actual:   {result:.4f}")
    print(f"   Match:    {'✅ PASSED' if abs(result - expected) < 0.01 else '❌ FAILED'}")

if __name__ == "__main__":
    test_sum_gpu()
