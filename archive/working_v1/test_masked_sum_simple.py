"""Test simple masked_sum GPU kernel with atomics."""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType


def test_masked_sum_simple(use_gpu: bool = False, size: int = 1024, seed: int = 42):
    """Test the simple masked_sum kernel."""
    np.random.seed(seed)
    
    device_name = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*50}")
    print(f"🧪 Testing masked_sum_simple on {device_name} (size={size:,})")
    print(f"{'='*50}")
    
    # Setup device
    if use_gpu:
        device = driver.Accelerator()
        device_ref = DeviceRef.GPU()
    else:
        device = driver.CPU()
        device_ref = DeviceRef.CPU()
    
    session = engine.InferenceSession(devices=[device])
    
    # For GPU, output is array of warp partial sums
    # For CPU, output is single scalar
    BLOCK_SIZE = 256
    WARP_SIZE = 32
    num_warps = (size + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
    out_size = num_warps if use_gpu else 1
    
    # Build the graph
    def masked_sum_graph(mask, values):
        return ops.custom(
            name="masked_sum_simple",
            values=[mask, values],
            out_types=[TensorType(DType.float32, (out_size,), device_ref)],
            device=device_ref,
        )[0]
    
    print("⚙️  Compiling graph...")
    graph = Graph(
        "test_masked_sum_simple",
        masked_sum_graph,
        input_types=[
            TensorType(DType.float32, (size,), device_ref),
            TensorType(DType.float32, (size,), device_ref),
        ],
        custom_extensions=[Path("kernels")],  # Use package directory
    )
    model = session.load(graph)
    print("✅ Compiled!")
    
    # Generate test data
    mask_data = np.random.choice([0.0, 1.0], size, p=[0.3, 0.7]).astype(np.float32)
    values_data = np.random.uniform(1, 100, size).astype(np.float32)
    expected = float(np.sum(mask_data * values_data))
    
    # Create tensors with proper transfer pattern
    if use_gpu:
        mask_tensor = driver.Tensor.from_numpy(mask_data).to(device)
        values_tensor = driver.Tensor.from_numpy(values_data).to(device)
    else:
        mask_tensor = driver.Tensor(mask_data, device)
        values_tensor = driver.Tensor(values_data, device)
    
    # Execute
    print("🚀 Executing...")
    outputs = model.execute(mask_tensor, values_tensor)
    
    # Read result with proper GPU→CPU transfer
    if use_gpu:
        out_arr = outputs[0].to(driver.CPU()).to_numpy()
        actual = float(np.sum(out_arr))  # Sum partial warp results
    else:
        actual = float(outputs[0].to_numpy().flat[0])
    
    # Compare
    abs_err = abs(actual - expected)
    rel_err = abs_err / abs(expected) if expected != 0 else abs_err
    
    print(f"\n📊 Results:")
    print(f"   Expected: {expected:.4f}")
    print(f"   Actual:   {actual:.4f}")
    print(f"   Abs Err:  {abs_err:.6f}")
    print(f"   Rel Err:  {rel_err:.2e}")
    
    # Pass threshold (float32 atomics can have some precision loss)
    passed = rel_err < 1e-4
    print(f"\n{'✅ PASSED!' if passed else '❌ FAILED!'}")
    
    return passed, {'expected': expected, 'actual': actual, 'rel_err': rel_err}


def run_benchmark(size: int = 1_000_000, n_runs: int = 10):
    """Benchmark the kernel on GPU vs CPU."""
    import time
    
    print(f"\n{'='*60}")
    print(f"⏱️  Benchmarking masked_sum_simple (size={size:,})")
    print(f"{'='*60}")
    
    np.random.seed(42)
    mask_data = np.random.choice([0.0, 1.0], size, p=[0.3, 0.7]).astype(np.float32)
    values_data = np.random.uniform(1, 100, size).astype(np.float32)
    
    for use_gpu in [False, True]:
        device_name = "GPU" if use_gpu else "CPU"
        device = driver.Accelerator() if use_gpu else driver.CPU()
        device_ref = DeviceRef.GPU() if use_gpu else DeviceRef.CPU()
        
        session = engine.InferenceSession(devices=[device])
        
        def masked_sum_graph(mask, values):
            return ops.custom(
                name="masked_sum_simple",
                values=[mask, values],
                out_types=[TensorType(DType.float32, (1,), device_ref)],
                device=device_ref,
            )[0]
        
        graph = Graph(
            f"bench_masked_sum_{device_name.lower()}",
            masked_sum_graph,
            input_types=[
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
            ],
            custom_extensions=[Path("kernels")],  # Use package directory
        )
        model = session.load(graph)
        
        if use_gpu:
            mask_tensor = driver.Tensor.from_numpy(mask_data).to(device)
            values_tensor = driver.Tensor.from_numpy(values_data).to(device)
        else:
            mask_tensor = driver.Tensor(mask_data, device)
            values_tensor = driver.Tensor(values_data, device)
        
        # Warmup
        for _ in range(3):
            model.execute(mask_tensor, values_tensor)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.execute(mask_tensor, values_tensor)
            times.append(time.perf_counter() - t0)
        
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        print(f"\n{device_name}: avg={avg:.3f}ms, best={best:.3f}ms")


if __name__ == "__main__":
    import sys
    
    # Test on CPU first
    cpu_passed, cpu_results = test_masked_sum_simple(use_gpu=False, size=1024)
    
    # Test on GPU
    try:
        gpu_passed, gpu_results = test_masked_sum_simple(use_gpu=True, size=1024)
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        gpu_passed = False
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Summary")
    print(f"{'='*50}")
    print(f"CPU: {'✅ PASSED' if cpu_passed else '❌ FAILED'}")
    print(f"GPU: {'✅ PASSED' if gpu_passed else '❌ FAILED'}")
    
    # If both pass, run benchmark
    if cpu_passed and gpu_passed:
        run_benchmark(size=1_000_000, n_runs=10)
    
    sys.exit(0 if (cpu_passed and gpu_passed) else 1)
