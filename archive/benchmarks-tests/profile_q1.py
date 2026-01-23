#!/usr/bin/env python3
"""Profile CompiledQ1 execution to find bottlenecks."""

from max_main import generate_tpch_lineitem, CompiledQ1

# Generate test data
n_rows = 1_000_000
print(f'Generating {n_rows:,} rows...')
data = generate_tpch_lineitem(n_rows)

# Compile once
print('Compiling...')
q1 = CompiledQ1(n_rows, verbose=True)

# Run multiple times and show detailed timings
print('\nExecuting 3 times:')
for i in range(3):
    results, timings = q1.execute(data)
    print(f'\nIteration {i+1}:')
    for k, v in sorted(timings.items(), key=lambda x: -x[1] if x[0] != 'total' else 0):
        pct = (v / timings['total']) * 100 if timings['total'] > 0 else 0
        print(f'  {k:15s}: {v*1000:8.2f}ms  ({pct:5.1f}%)')
