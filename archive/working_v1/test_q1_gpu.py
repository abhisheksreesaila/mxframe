#!/usr/bin/env python3
"""Test full TPC-H Q1 on GPU with tensor transfer fix."""

import os
os.environ["MODULAR_DEVICE_CONTEXT_SYNC_MODE"] = "true"

import numpy as np
import time
from max import engine, driver
from max.graph import Graph, TensorType, ops, DeviceRef
from max.dtype import DType

def generate_tpch_lineitem(n_rows, seed=42):
    """Generate synthetic TPC-H lineitem data."""
    np.random.seed(seed)
    l_shipdate = np.random.randint(8035, 10562, size=n_rows).astype(np.int32)
    l_returnflag_enc = np.random.choice([0, 1, 2], n_rows, p=[0.25, 0.50, 0.25]).astype(np.int32)
    l_linestatus_enc = (l_shipdate >= 9299).astype(np.int32)
    l_quantity = np.random.uniform(1, 50, n_rows).astype(np.float32)
    part_price = np.random.uniform(2, 200, n_rows).astype(np.float32)
    l_extendedprice = (l_quantity * part_price).astype(np.float32)
    l_discount = np.random.uniform(0, 0.10, n_rows).astype(np.float32)
    l_tax = np.random.uniform(0, 0.08, n_rows).astype(np.float32)
    
    return {
        'l_shipdate': l_shipdate,
        'l_returnflag_enc': l_returnflag_enc,
        'l_linestatus_enc': l_linestatus_enc,
        'l_quantity': l_quantity,
        'l_extendedprice': l_extendedprice,
        'l_discount': l_discount,
        'l_tax': l_tax,
    }

class FusedQ1Graph:
    def __init__(self, device_ref, date_cutoff=10471):
        self.device_ref = device_ref
        self.date_cutoff = date_cutoff
    
    def __call__(self, shipdate, rf_enc, ls_enc, qty, price, disc, tax):
        zero_f = ops.constant(0.0, dtype=DType.float32, device=self.device_ref)
        one_f = ops.constant(1.0, dtype=DType.float32, device=self.device_ref)
        cutoff = ops.constant(self.date_cutoff, dtype=DType.int32, device=self.device_ref)
        
        date_cond = ops.greater_equal(cutoff, shipdate)
        date_mask = ops.cast(date_cond, DType.float32)
        
        one_minus_disc = ops.sub(one_f, disc)
        one_plus_tax = ops.add(one_f, tax)
        disc_price = ops.mul(price, one_minus_disc)
        charge = ops.mul(disc_price, one_plus_tax)
        
        two = ops.constant(2, dtype=DType.int32, device=self.device_ref)
        raw_group = ops.add(ops.mul(rf_enc, two), ls_enc)
        
        results = []
        for g in range(4):
            if g == 0:    raw_vals = [0, 1]
            elif g == 1:  raw_vals = [2]
            elif g == 2:  raw_vals = [3]
            else:         raw_vals = [4, 5]
            
            group_mask = None
            for raw_v in raw_vals:
                val = ops.constant(raw_v, dtype=DType.int32, device=self.device_ref)
                eq_float = ops.cast(ops.equal(raw_group, val), DType.float32)
                group_mask = eq_float if group_mask is None else ops.add(group_mask, eq_float)
            
            combined_mask = ops.mul(date_mask, group_mask)
            
            results.extend([
                ops.sum(ops.mul(combined_mask, qty)),
                ops.sum(ops.mul(combined_mask, price)),
                ops.sum(ops.mul(combined_mask, disc)),
                ops.sum(ops.mul(combined_mask, disc_price)),
                ops.sum(ops.mul(combined_mask, charge)),
                ops.sum(combined_mask),
            ])
        
        return tuple(results)

class MaxNativeQ1:
    GROUP_LABELS = [('A', 'F'), ('N', 'F'), ('N', 'O'), ('R', 'F')]
    
    def __init__(self, n_rows, use_gpu=False):
        self.n_rows = n_rows
        self.use_gpu = use_gpu
        
        if use_gpu:
            self.device = driver.Accelerator()
            self.device_ref = DeviceRef.GPU()
        else:
            self.device = driver.CPU()
            self.device_ref = DeviceRef.CPU()

        self.session = engine.InferenceSession(devices=[self.device])
        self._compile()
    
    def _compile(self):
        n = self.n_rows
        graph = Graph(
            "max_native_q1",
            FusedQ1Graph(self.device_ref),
            input_types=[
                TensorType(DType.int32, (n,), self.device_ref),
                TensorType(DType.int32, (n,), self.device_ref),
                TensorType(DType.int32, (n,), self.device_ref),
                TensorType(DType.float32, (n,), self.device_ref),
                TensorType(DType.float32, (n,), self.device_ref),
                TensorType(DType.float32, (n,), self.device_ref),
                TensorType(DType.float32, (n,), self.device_ref),
            ]
        )
        self.model = self.session.load(graph)
    
    def execute(self, data):
        # FIX: Use from_numpy().to(device) pattern for GPU
        if self.use_gpu:
            inputs = [
                driver.Tensor.from_numpy(data['l_shipdate']).to(self.device),
                driver.Tensor.from_numpy(data['l_returnflag_enc']).to(self.device),
                driver.Tensor.from_numpy(data['l_linestatus_enc']).to(self.device),
                driver.Tensor.from_numpy(data['l_quantity']).to(self.device),
                driver.Tensor.from_numpy(data['l_extendedprice']).to(self.device),
                driver.Tensor.from_numpy(data['l_discount']).to(self.device),
                driver.Tensor.from_numpy(data['l_tax']).to(self.device),
            ]
        else:
            inputs = [
                driver.Tensor(data['l_shipdate'], self.device),
                driver.Tensor(data['l_returnflag_enc'], self.device),
                driver.Tensor(data['l_linestatus_enc'], self.device),
                driver.Tensor(data['l_quantity'], self.device),
                driver.Tensor(data['l_extendedprice'], self.device),
                driver.Tensor(data['l_discount'], self.device),
                driver.Tensor(data['l_tax'], self.device),
            ]
        
        outputs = self.model.execute(*inputs)
        
        # FIX: Copy to CPU before reading
        if self.use_gpu:
            values = []
            for out in outputs:
                cpu_out = out.to(driver.CPU()).to_numpy()
                val = float(cpu_out.item() if cpu_out.ndim == 0 else cpu_out[0])
                values.append(val)
        else:
            values = [float(out.to_numpy()[0]) for out in outputs]
        
        results = []
        for g in range(4):
            base = g * 6
            count = int(values[base + 5])
            if count > 0:
                rf, ls = self.GROUP_LABELS[g]
                results.append({
                    'l_returnflag': rf, 'l_linestatus': ls,
                    'sum_qty': values[base], 'count_order': count,
                })
        
        return sorted(results, key=lambda x: (x['l_returnflag'], x['l_linestatus']))


if __name__ == "__main__":
    N_ROWS = 100_000
    
    print("=" * 60)
    print(f"🧪 Testing TPC-H Q1 on GPU with {N_ROWS:,} rows")
    print("=" * 60)
    
    # Generate data
    print("\n📊 Generating test data...")
    data = generate_tpch_lineitem(N_ROWS)
    
    # Test GPU
    print("\n⚙️  Compiling for GPU...")
    t0 = time.perf_counter()
    q1_gpu = MaxNativeQ1(N_ROWS, use_gpu=True)
    print(f"✅ Compiled in {time.perf_counter() - t0:.2f}s")
    
    print("\n🚀 Executing on GPU...")
    t0 = time.perf_counter()
    results_gpu = q1_gpu.execute(data)
    gpu_time = time.perf_counter() - t0
    print(f"✅ Executed in {gpu_time*1000:.2f}ms")
    
    # Test CPU for comparison
    print("\n⚙️  Compiling for CPU...")
    t0 = time.perf_counter()
    q1_cpu = MaxNativeQ1(N_ROWS, use_gpu=False)
    print(f"✅ Compiled in {time.perf_counter() - t0:.2f}s")
    
    print("\n🚀 Executing on CPU...")
    t0 = time.perf_counter()
    results_cpu = q1_cpu.execute(data)
    cpu_time = time.perf_counter() - t0
    print(f"✅ Executed in {cpu_time*1000:.2f}ms")
    
    # Compare results
    print("\n" + "=" * 60)
    print("📊 Results Comparison")
    print("=" * 60)
    print(f"{'Group':<10} {'GPU sum_qty':>15} {'CPU sum_qty':>15} {'Match':>10}")
    print("-" * 50)
    
    all_match = True
    for gpu_row, cpu_row in zip(results_gpu, results_cpu):
        match = abs(gpu_row['sum_qty'] - cpu_row['sum_qty']) < 1.0
        all_match = all_match and match
        group = f"{gpu_row['l_returnflag']}/{gpu_row['l_linestatus']}"
        print(f"{group:<10} {gpu_row['sum_qty']:>15,.0f} {cpu_row['sum_qty']:>15,.0f} {'✅' if match else '❌':>10}")
    
    print("\n" + "=" * 60)
    if all_match:
        print("✅ ALL TESTS PASSED! GPU produces correct results.")
    else:
        print("❌ MISMATCH DETECTED!")
    
    print(f"\n⚡ Performance: GPU={gpu_time*1000:.2f}ms, CPU={cpu_time*1000:.2f}ms")
    if gpu_time < cpu_time:
        print(f"   GPU is {cpu_time/gpu_time:.1f}x faster!")
    else:
        print(f"   CPU is {cpu_time/gpu_time:.1f}x faster (GPU overhead for small data)")
