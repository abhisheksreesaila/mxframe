"""Test Level 2 fused kernel: Full TPC-H Q1 in one kernel launch."""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
import time
from pathlib import Path
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

NUM_GROUPS = 4
NUM_AGGS = 6
TOTAL_OUTPUTS = NUM_GROUPS * NUM_AGGS  # 24
WARP_SIZE = 32
DATE_CUTOFF = 10471  # 1998-09-02 as epoch days

GROUP_LABELS = [('A', 'F'), ('N', 'F'), ('N', 'O'), ('R', 'F')]
AGG_NAMES = ['sum_qty', 'sum_price', 'sum_disc', 'sum_disc_price', 'sum_charge', 'count']


def generate_tpch_data(n_rows: int, seed: int = 42):
    """Generate synthetic TPC-H lineitem data."""
    np.random.seed(seed)
    
    # Date range: 1992-01-01 to 1998-12-01 (as epoch days)
    l_shipdate = np.random.randint(8035, 10562, size=n_rows).astype(np.int32)
    
    # Return flag: A=0, N=1, R=2
    l_returnflag_enc = np.random.choice([0, 1, 2], n_rows, p=[0.25, 0.50, 0.25]).astype(np.int32)
    
    # Line status: F=0 if shipped before 1995-06-17, else O=1
    l_linestatus_enc = (l_shipdate >= 9299).astype(np.int32)
    
    # Numeric columns
    l_quantity = np.random.uniform(1, 50, n_rows).astype(np.float32)
    part_price = np.random.uniform(2, 200, n_rows).astype(np.float32)
    l_extendedprice = (l_quantity * part_price).astype(np.float32)
    l_discount = np.random.uniform(0, 0.10, n_rows).astype(np.float32)
    l_tax = np.random.uniform(0, 0.08, n_rows).astype(np.float32)
    
    # Derived columns
    disc_price = (l_extendedprice * (1 - l_discount)).astype(np.float32)
    charge = (disc_price * (1 + l_tax)).astype(np.float32)
    
    return {
        'l_shipdate': l_shipdate,
        'l_returnflag_enc': l_returnflag_enc,
        'l_linestatus_enc': l_linestatus_enc,
        'l_quantity': l_quantity,
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'disc_price': disc_price,
        'charge': charge,
    }


def compute_expected(data: dict):
    """Compute expected Q1 results using NumPy."""
    mask = data['l_shipdate'] <= DATE_CUTOFF
    rf = data['l_returnflag_enc']
    ls = data['l_linestatus_enc']
    
    results = {}
    for g in range(NUM_GROUPS):
        if g == 0:  # A (rf=0)
            group_mask = mask & (rf == 0)
        elif g == 1:  # N+F (rf=1, ls=0)
            group_mask = mask & (rf == 1) & (ls == 0)
        elif g == 2:  # N+O (rf=1, ls=1)
            group_mask = mask & (rf == 1) & (ls == 1)
        else:  # R (rf=2)
            group_mask = mask & (rf == 2)
        
        results[g] = {
            'sum_qty': float(np.sum(data['l_quantity'][group_mask])),
            'sum_price': float(np.sum(data['l_extendedprice'][group_mask])),
            'sum_disc': float(np.sum(data['l_discount'][group_mask])),
            'sum_disc_price': float(np.sum(data['disc_price'][group_mask])),
            'sum_charge': float(np.sum(data['charge'][group_mask])),
            'count': float(np.sum(group_mask)),
        }
    
    return results


def test_fused_q1_full(use_gpu: bool = False, size: int = 100_000, seed: int = 42):
    """Test the full fused Q1 kernel."""
    device_name = "GPU" if use_gpu else "CPU"
    print(f"\n{'='*70}")
    print(f"🧪 Testing fused_q1_full on {device_name} (size={size:,})")
    print(f"{'='*70}")
    
    # Generate data
    data = generate_tpch_data(size, seed)
    expected = compute_expected(data)
    
    # Setup device
    if use_gpu:
        device = driver.Accelerator()
        device_ref = DeviceRef.GPU()
    else:
        device = driver.CPU()
        device_ref = DeviceRef.CPU()
    
    session = engine.InferenceSession(devices=[device])
    
    # Output size
    num_warps = (size + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
    out_size = num_warps * TOTAL_OUTPUTS if use_gpu else TOTAL_OUTPUTS
    
    # Build graph
    def fused_q1_graph(shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge):
        return ops.custom(
            name="fused_q1_full",
            values=[shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge],
            out_types=[TensorType(DType.float32, (out_size,), device_ref)],
            device=device_ref,
        )[0]
    
    print("⚙️  Compiling graph...")
    graph = Graph(
        "test_fused_q1_full",
        fused_q1_graph,
        input_types=[
            TensorType(DType.int32, (size,), device_ref),    # shipdate
            TensorType(DType.int32, (size,), device_ref),    # rf_enc
            TensorType(DType.int32, (size,), device_ref),    # ls_enc
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
    
    # Create tensors
    if use_gpu:
        tensors = [
            driver.Tensor.from_numpy(data['l_shipdate']).to(device),
            driver.Tensor.from_numpy(data['l_returnflag_enc']).to(device),
            driver.Tensor.from_numpy(data['l_linestatus_enc']).to(device),
            driver.Tensor.from_numpy(data['l_quantity']).to(device),
            driver.Tensor.from_numpy(data['l_extendedprice']).to(device),
            driver.Tensor.from_numpy(data['l_discount']).to(device),
            driver.Tensor.from_numpy(data['disc_price']).to(device),
            driver.Tensor.from_numpy(data['charge']).to(device),
        ]
    else:
        tensors = [
            driver.Tensor(data['l_shipdate'], device),
            driver.Tensor(data['l_returnflag_enc'], device),
            driver.Tensor(data['l_linestatus_enc'], device),
            driver.Tensor(data['l_quantity'], device),
            driver.Tensor(data['l_extendedprice'], device),
            driver.Tensor(data['l_discount'], device),
            driver.Tensor(data['disc_price'], device),
            driver.Tensor(data['charge'], device),
        ]
    
    # Execute
    print("🚀 Executing...")
    outputs = model.execute(*tensors)
    
    # Read and aggregate results
    if use_gpu:
        out_arr = outputs[0].to(driver.CPU()).to_numpy()
        # Sum across warps for each of 24 outputs
        actual = {}
        for g in range(NUM_GROUPS):
            actual[g] = {}
            for a, agg_name in enumerate(AGG_NAMES):
                idx = g * NUM_AGGS + a
                actual[g][agg_name] = float(np.sum(out_arr[idx::TOTAL_OUTPUTS]))
    else:
        out_arr = outputs[0].to_numpy()
        actual = {}
        for g in range(NUM_GROUPS):
            actual[g] = {}
            for a, agg_name in enumerate(AGG_NAMES):
                idx = g * NUM_AGGS + a
                actual[g][agg_name] = float(out_arr[idx])
    
    # Compare
    print(f"\n📊 Results ({device_name}):")
    all_passed = True
    
    for g in range(NUM_GROUPS):
        rf, ls = GROUP_LABELS[g]
        print(f"\n  Group {g} ({rf}, {ls}):")
        print(f"  {'Metric':<15} {'Expected':>15} {'Actual':>15} {'Rel Err':>12} {'Status'}")
        print(f"  {'-'*65}")
        
        for agg_name in AGG_NAMES:
            exp = expected[g][agg_name]
            act = actual[g][agg_name]
            if exp != 0:
                rel_err = abs(act - exp) / abs(exp)
            else:
                rel_err = abs(act)
            passed = rel_err < 1e-4 or (exp == 0 and act == 0)
            all_passed = all_passed and passed
            status = "✅" if passed else "❌"
            print(f"  {agg_name:<15} {exp:>15.2f} {act:>15.2f} {rel_err:>12.2e} {status}")
    
    print(f"\n{'✅ ALL PASSED!' if all_passed else '❌ SOME FAILED!'}")
    return all_passed, actual


def run_benchmark(size: int = 1_000_000, n_runs: int = 10):
    """Benchmark full Q1 kernel vs ops-based approach."""
    print(f"\n{'='*70}")
    print(f"⏱️  Benchmarking fused_q1_full (size={size:,})")
    print(f"{'='*70}")
    
    data = generate_tpch_data(size, seed=42)
    results = {}
    
    for use_gpu in [False, True]:
        device_name = "GPU" if use_gpu else "CPU"
        device = driver.Accelerator() if use_gpu else driver.CPU()
        device_ref = DeviceRef.GPU() if use_gpu else DeviceRef.CPU()
        
        num_warps = (size + WARP_SIZE - 1) // WARP_SIZE if use_gpu else 1
        out_size = num_warps * TOTAL_OUTPUTS if use_gpu else TOTAL_OUTPUTS
        
        session = engine.InferenceSession(devices=[device])
        
        def fused_q1_graph(shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge):
            return ops.custom(
                name="fused_q1_full",
                values=[shipdate, rf_enc, ls_enc, qty, price, disc, disc_price, charge],
                out_types=[TensorType(DType.float32, (out_size,), device_ref)],
                device=device_ref,
            )[0]
        
        graph = Graph(
            f"bench_fused_q1_{device_name.lower()}",
            fused_q1_graph,
            input_types=[
                TensorType(DType.int32, (size,), device_ref),
                TensorType(DType.int32, (size,), device_ref),
                TensorType(DType.int32, (size,), device_ref),
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
                driver.Tensor.from_numpy(data['l_shipdate']).to(device),
                driver.Tensor.from_numpy(data['l_returnflag_enc']).to(device),
                driver.Tensor.from_numpy(data['l_linestatus_enc']).to(device),
                driver.Tensor.from_numpy(data['l_quantity']).to(device),
                driver.Tensor.from_numpy(data['l_extendedprice']).to(device),
                driver.Tensor.from_numpy(data['l_discount']).to(device),
                driver.Tensor.from_numpy(data['disc_price']).to(device),
                driver.Tensor.from_numpy(data['charge']).to(device),
            ]
        else:
            tensors = [
                driver.Tensor(data['l_shipdate'], device),
                driver.Tensor(data['l_returnflag_enc'], device),
                driver.Tensor(data['l_linestatus_enc'], device),
                driver.Tensor(data['l_quantity'], device),
                driver.Tensor(data['l_extendedprice'], device),
                driver.Tensor(data['l_discount'], device),
                driver.Tensor(data['disc_price'], device),
                driver.Tensor(data['charge'], device),
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
    cpu_passed, _ = test_fused_q1_full(use_gpu=False, size=100_000)
    
    # Test on GPU
    try:
        gpu_passed, _ = test_fused_q1_full(use_gpu=True, size=100_000)
    except Exception as e:
        print(f"\n❌ GPU test failed: {e}")
        import traceback
        traceback.print_exc()
        gpu_passed = False
    
    # Summary
    print(f"\n{'='*70}")
    print("📊 Summary")
    print(f"{'='*70}")
    print(f"CPU: {'✅ PASSED' if cpu_passed else '❌ FAILED'}")
    print(f"GPU: {'✅ PASSED' if gpu_passed else '❌ FAILED'}")
    
    # If both pass, run benchmark
    if cpu_passed and gpu_passed:
        for size in [1_000_000, 6_000_000]:
            run_benchmark(size=size, n_runs=5)
    
    sys.exit(0 if (cpu_passed and gpu_passed) else 1)
