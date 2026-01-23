#!/usr/bin/env python3
"""
Test the MaxNativeQ1 class - Fully MAX-native TPC-H Q1.

This implementation uses MAX Engine for ALL operations:
- Date filtering via mask
- Group encoding via arithmetic
- Precompute disc_price and charge
- 4-group masked aggregation

No numpy fallbacks in the hot path!
"""

from max_main import generate_tpch_lineitem, MaxNativeQ1, CompiledQ1
import time


def test_max_native_q1(n_rows=100_000, n_executions=5):
    """Test and benchmark MaxNativeQ1."""
    
    print("=" * 60)
    print(f"MaxNativeQ1 Test: {n_rows:,} rows")
    print("=" * 60)
    
    # Generate test data
    print('\nGenerating test data...')
    t0 = time.perf_counter()
    data = generate_tpch_lineitem(n_rows)
    gen_time = time.perf_counter() - t0
    print(f"Data generation: {gen_time*1000:.1f}ms")

    # Compile MAX-native version
    print('\n--- MaxNativeQ1 (Fully MAX) ---')
    t0 = time.perf_counter()
    q1_native = MaxNativeQ1(n_rows, verbose=True)
    compile_time_native = time.perf_counter() - t0
    
    # Execute multiple times
    print(f'\nExecuting {n_executions} times:')
    native_times = []
    for i in range(n_executions):
        t0 = time.perf_counter()
        results_native, timings = q1_native.execute(data)
        exec_time = time.perf_counter() - t0
        native_times.append(exec_time)
        print(f"  Execution {i+1}: {exec_time*1000:.2f}ms "
              f"(prep={timings['prep']*1000:.1f}, exec={timings['execute']*1000:.1f}, extract={timings['extract']*1000:.1f})")
    
    print(f"\nResults:")
    for row in results_native:
        print(f"  ({row['l_returnflag']}, {row['l_linestatus']}): "
              f"sum_qty={row['sum_qty']:.0f}, count={row['count_order']}")
    
    # Compare with CompiledQ1 (numpy fallback version)
    print('\n--- CompiledQ1 (Numpy fallbacks) ---')
    t0 = time.perf_counter()
    q1_compiled = CompiledQ1(n_rows, verbose=True)
    compile_time_compiled = time.perf_counter() - t0
    
    compiled_times = []
    for i in range(n_executions):
        t0 = time.perf_counter()
        results_compiled, _ = q1_compiled.execute(data)
        exec_time = time.perf_counter() - t0
        compiled_times.append(exec_time)
        print(f"  Execution {i+1}: {exec_time*1000:.2f}ms")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    
    avg_native = sum(native_times) / len(native_times)
    min_native = min(native_times)
    avg_compiled = sum(compiled_times) / len(compiled_times)
    min_compiled = min(compiled_times)
    
    print(f"\n{'Implementation':<25} {'Compile':<12} {'Avg Exec':<12} {'Best Exec':<12}")
    print("-" * 60)
    print(f"{'MaxNativeQ1 (all MAX)':<25} {compile_time_native*1000:<12.1f} {avg_native*1000:<12.2f} {min_native*1000:<12.2f}")
    print(f"{'CompiledQ1 (numpy)':<25} {compile_time_compiled*1000:<12.1f} {avg_compiled*1000:<12.2f} {min_compiled*1000:<12.2f}")
    
    speedup = min_compiled / min_native if min_native > 0 else 0
    print(f"\nMaxNativeQ1 is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than CompiledQ1")
    
    # Verify results match
    print("\nVerifying results match...")
    for r1, r2 in zip(results_native, results_compiled):
        assert r1['l_returnflag'] == r2['l_returnflag']
        assert r1['l_linestatus'] == r2['l_linestatus']
        assert abs(r1['sum_qty'] - r2['sum_qty']) < 1.0, f"sum_qty mismatch: {r1['sum_qty']} vs {r2['sum_qty']}"
        assert abs(r1['count_order'] - r2['count_order']) < 1, f"count mismatch"
    print("✓ All results match!")
    
    return results_native


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--executions', type=int, default=5)
    args = parser.parse_args()
    
    test_max_native_q1(args.rows, args.executions)
