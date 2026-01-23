#!/usr/bin/env python3
"""
Test the CompiledQ1 class - Pre-compiled TPC-H Q1.

This demonstrates the key optimization: compile all MAX graphs once,
then reuse them for fast subsequent executions.
"""

from max_main import generate_tpch_lineitem, CompiledQ1
import time

def test_compiled_q1(n_rows=100_000, n_executions=5):
    """Test CompiledQ1 with timing breakdown."""
    
    print(f"=" * 60)
    print(f"CompiledQ1 Test: {n_rows:,} rows, {n_executions} executions")
    print(f"=" * 60)
    
    # Generate test data
    print('\nGenerating test data...')
    t0 = time.perf_counter()
    data = generate_tpch_lineitem(n_rows)
    gen_time = time.perf_counter() - t0
    print(f"Data generation: {gen_time*1000:.1f}ms")

    # Compile (one-time cost)
    print('\nCompiling Q1 graphs (one-time cost)...')
    t0 = time.perf_counter()
    q1 = CompiledQ1(n_rows, verbose=True)
    compile_time = time.perf_counter() - t0

    # Execute multiple times
    print(f'\nExecuting {n_executions} times:')
    exec_times = []
    for i in range(n_executions):
        t0 = time.perf_counter()
        results, timings = q1.execute(data)
        exec_time = time.perf_counter() - t0
        exec_times.append(exec_time)
        print(f"  Execution {i+1}: {exec_time*1000:.1f}ms")
    
    avg_exec = sum(exec_times) / len(exec_times)
    min_exec = min(exec_times)
    
    print(f'\nResults:')
    for row in results:
        print(f"  ({row['l_returnflag']}, {row['l_linestatus']}): "
              f"sum_qty={row['sum_qty']:.0f}, "
              f"avg_price={row['avg_price']:.2f}, "
              f"count={row['count_order']}")
    
    print(f'\nPerformance Summary:')
    print(f"  Compilation (one-time): {compile_time*1000:.1f}ms")
    print(f"  Average execution:      {avg_exec*1000:.1f}ms")
    print(f"  Best execution:         {min_exec*1000:.1f}ms")
    print(f"  Total for first query:  {(compile_time + exec_times[0])*1000:.1f}ms")
    
    # Show benefit of amortization
    if n_executions > 1:
        amortized = (compile_time + sum(exec_times)) / n_executions
        print(f"\n  After {n_executions} queries, amortized cost: {amortized*1000:.1f}ms per query")
    
    print()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rows', type=int, default=100_000)
    parser.add_argument('--executions', type=int, default=5)
    args = parser.parse_args()
    
    test_compiled_q1(args.rows, args.executions)
