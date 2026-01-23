"""Test Level 1 fused kernel: All 6 aggregations for one group."""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
import time
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

NUM_AGGS = 6
WARP_SIZE = 32


def test_fused_group_agg(use_gpu: bool = False, size: int = 100_000, seed: int = 42):
    """Test the fused_group_agg kernel."""
    np.random.seed(seed)
    
    device_name = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*60}")
    print(f"🧪 Testing fused_group_agg on {device_name} (size={size:,})")
    print(f"{'='*60}")
    
    # Setup device
    if use_gpu:
        device = driver.Accelerator()
        device_ref = DeviceRef.GPU()
    else:
        device = driver.CPU()
        device_ref = DeviceRef.CPU()
    
    session = engine.InferenceSession(devices=[device])
    
    # Output size: GPU returns warp partial sums, CPU returns 6 values
    num_warps = (size + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
    out_size = num_warps * NUM_AGGS if use_gpu else NUM_AGGS
    
    # Build the graph
    def fused_agg_graph(mask, qty, price, disc, disc_price, charge):
        return ops.custom(
            name="fused_group_agg",
            values=[mask, qty, price, disc, disc_price, charge],
            out_types=[TensorType(DType.float32, (out_size,), device_ref)],
            device=device_ref,
        )[0]
    
    print("⚙️  Compiling graph...")
    graph = Graph(
        "test_fused_group_agg",
        fused_agg_graph,
        input_types=[
            TensorType(DType.float32, (size,), device_ref),  # mask
            TensorType(DType.float32, (size,), device_ref),  # qty
            TensorType(DType.float32, (size,), device_ref),  # price
            TensorType(DType.float32, (size,), device_ref),  # disc
            TensorType(DType.float32, (size,), device_ref),  # disc_price
            TensorType(DType.float32, (size,), device_ref),  # charge
        ],
        custom_extensions=[Path("kernels")],
    )
    model = session.load(graph)
    print("✅ Compiled!")
    
    # Generate test data (simulating one group's mask)
    mask_data = np.random.choice([0.0, 1.0], size, p=[0.7, 0.3]).astype(np.float32)
    qty_data = np.random.uniform(1, 50, size).astype(np.float32)
    price_data = np.random.uniform(100, 10000, size).astype(np.float32)
    disc_data = np.random.uniform(0, 0.10, size).astype(np.float32)
    disc_price_data = (price_data * (1 - disc_data)).astype(np.float32)
    charge_data = (disc_price_data * (1 + np.random.uniform(0, 0.08, size))).astype(np.float32)
    
    # Expected results
    expected = {
        'sum_qty': float(np.sum(mask_data * qty_data)),
        'sum_price': float(np.sum(mask_data * price_data)),
        'sum_disc': float(np.sum(mask_data * disc_data)),
        'sum_disc_price': float(np.sum(mask_data * disc_price_data)),
        'sum_charge': float(np.sum(mask_data * charge_data)),
        'count': float(np.sum(mask_data)),
    }
    
    # Create tensors
    if use_gpu:
        tensors = [
            driver.Tensor.from_numpy(mask_data).to(device),
            driver.Tensor.from_numpy(qty_data).to(device),
            driver.Tensor.from_numpy(price_data).to(device),
            driver.Tensor.from_numpy(disc_data).to(device),
            driver.Tensor.from_numpy(disc_price_data).to(device),
            driver.Tensor.from_numpy(charge_data).to(device),
        ]
    else:
        tensors = [
            driver.Tensor(mask_data, device),
            driver.Tensor(qty_data, device),
            driver.Tensor(price_data, device),
            driver.Tensor(disc_data, device),
            driver.Tensor(disc_price_data, device),
            driver.Tensor(charge_data, device),
        ]
    
    # Execute
    print("🚀 Executing...")
    outputs = model.execute(*tensors)
    
    # Read and aggregate results
    if use_gpu:
        out_arr = outputs[0].to(driver.CPU()).to_numpy()
        # Sum across warps for each aggregation
        actual = {}
        for i, name in enumerate(['sum_qty', 'sum_price', 'sum_disc', 'sum_disc_price', 'sum_charge', 'count']):
            actual[name] = float(np.sum(out_arr[i::NUM_AGGS]))
    else:
        out_arr = outputs[0].to_numpy()
        actual = {
            'sum_qty': float(out_arr[0]),
            'sum_price': float(out_arr[1]),
            'sum_disc': float(out_arr[2]),
            'sum_disc_price': float(out_arr[3]),
            'sum_charge': float(out_arr[4]),
            'count': float(out_arr[5]),
        }
    
    # Compare
    print(f"\n📊 Results ({device_name}):")
    print(f"{'Metric':<15} {'Expected':>15} {'Actual':>15} {'Rel Err':>12} {'Status':<8}")
    print("-" * 70)
    
    all_passed = True
    for name in ['sum_qty', 'sum_price', 'sum_disc', 'sum_disc_price', 'sum_charge', 'count']:
        exp = expected[name]
        act = actual[name]
        rel_err = abs(act - exp) / abs(exp) if exp != 0 else abs(act)
        passed = rel_err < 1e-4
        all_passed = all_passed and passed
        status = "✅" if passed else "❌"
        print(f"{name:<15} {exp:>15.2f} {act:>15.2f} {rel_err:>12.2e} {status:<8}")
    
    print(f"\n{'✅ ALL PASSED!' if all_passed else '❌ SOME FAILED!'}")
    return all_passed, actual


def run_benchmark(size: int = 1_000_000, n_runs: int = 10):
    """Benchmark fused kernel vs separate ops."""
    print(f"\n{'='*70}")
    print(f"⏱️  Benchmarking fused_group_agg (size={size:,})")
    print(f"{'='*70}")
    
    np.random.seed(42)
    mask_data = np.random.choice([0.0, 1.0], size, p=[0.7, 0.3]).astype(np.float32)
    qty_data = np.random.uniform(1, 50, size).astype(np.float32)
    price_data = np.random.uniform(100, 10000, size).astype(np.float32)
    disc_data = np.random.uniform(0, 0.10, size).astype(np.float32)
    disc_price_data = (price_data * (1 - disc_data)).astype(np.float32)
    charge_data = (disc_price_data * 1.04).astype(np.float32)
    
    results = {}
    
    for use_gpu in [False, True]:
        device_name = "GPU" if use_gpu else "CPU"
        device = driver.Accelerator() if use_gpu else driver.CPU()
        device_ref = DeviceRef.GPU() if use_gpu else DeviceRef.CPU()
        
        num_warps = (size + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
        out_size = num_warps * NUM_AGGS if use_gpu else NUM_AGGS
        
        session = engine.InferenceSession(devices=[device])
        
        def fused_agg_graph(mask, qty, price, disc, disc_price, charge):
            return ops.custom(
                name="fused_group_agg",
                values=[mask, qty, price, disc, disc_price, charge],
                out_types=[TensorType(DType.float32, (out_size,), device_ref)],
                device=device_ref,
            )[0]
        
        graph = Graph(
            f"bench_fused_{device_name.lower()}",
            fused_agg_graph,
            input_types=[
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
                TensorType(DType.float32, (size,), device_ref),
            ],
            custom_extensions=[Path("kernels")],
        )
        model = session.load(graph)
        
        if use_gpu:
            tensors = [
                driver.Tensor.from_numpy(mask_data).to(device),
                driver.Tensor.from_numpy(qty_data).to(device),
                driver.Tensor.from_numpy(price_data).to(device),
                driver.Tensor.from_numpy(disc_data).to(device),
                driver.Tensor.from_numpy(disc_price_data).to(device),
                driver.Tensor.from_numpy(charge_data).to(device),
            ]
        else:
            tensors = [
                driver.Tensor(mask_data, device),
                driver.Tensor(qty_data, device),
                driver.Tensor(price_data, device),
                driver.Tensor(disc_data, device),
                driver.Tensor(disc_price_data, device),
                driver.Tensor(charge_data, device),
            ]
        
        # Warmup
        for _ in range(3):
            model.execute(*tensors)
        
        # Benchmark
        times = []
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model.execute(*tensors)
            times.append(time.perf_counter() - t0)
        
        avg = sum(times) / len(times) * 1000
        best = min(times) * 1000
        results[device_name] = {'avg': avg, 'best': best}
        print(f"\n{device_name}: avg={avg:.3f}ms, best={best:.3f}ms")
    
    speedup = results['CPU']['avg'] / results['GPU']['avg']
    print(f"\n🏆 GPU is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than CPU")
    
    return results


if __name__ == "__main__":
    import sys
    
    # Test on CPU first
    cpu_passed, _ = test_fused_group_agg(use_gpu=False, size=100_000)
    
    # Test on GPU
    try:
        gpu_passed, _ = test_fused_group_agg(use_gpu=True, size=100_000)
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        gpu_passed = False
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary")
    print(f"{'='*60}")
    print(f"CPU: {'✅ PASSED' if cpu_passed else '❌ FAILED'}")
    print(f"GPU: {'✅ PASSED' if gpu_passed else '❌ FAILED'}")
    
    # If both pass, run benchmark
    if cpu_passed and gpu_passed:
        run_benchmark(size=1_000_000, n_runs=10)
    
    sys.exit(0 if (cpu_passed and gpu_passed) else 1)
