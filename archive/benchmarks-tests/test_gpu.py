#!/usr/bin/env python3
"""Test GPU execution with debug mode."""

from max_main import generate_tpch_lineitem, MaxNativeQ1

# Small test
data = generate_tpch_lineitem(10000)
print("Creating MaxNativeQ1...")
q1 = MaxNativeQ1(10000, verbose=True)
print("Executing...")
results, timings = q1.execute(data)
print(f"Results: {len(results)} groups")
for r in results:
    print(f"  ({r['l_returnflag']}, {r['l_linestatus']}): count={r['count_order']}")
