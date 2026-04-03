"""Phase 4 integration tests: Sort, Limit, Distinct via Mojo kernels."""
import pyarrow as pa
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Scan, Sort, Limit, Distinct
from mxframe.custom_ops import CustomOpsCompiler, clear_cache

clear_cache()

# Test 1: Sort via Mojo kernel
table = pa.table({
    'group': ['b', 'a', 'c', 'a', 'b'],
    'val':   [2.0, 1.0, 3.0, 4.0, 5.0],
})
lf = LazyFrame(Scan(table))
result = lf.groupby('group').agg(col('val').sum().alias('total')).sort('group').compute()
groups = result.column('group').to_pylist()
print(f'Sort test: groups={groups}')
assert groups == ['a', 'b', 'c'], f'Expected sorted groups, got {groups}'
print('✅ Test 1: Sort works!')

# Test 2: Sort + Limit
clear_cache()
result2 = lf.groupby('group').agg(col('val').sum().alias('total')).sort('group').limit(2).compute()
assert result2.num_rows == 2, f'Expected 2 rows, got {result2.num_rows}'
assert result2.column('group').to_pylist() == ['a', 'b']
print('✅ Test 2: Sort + Limit works!')

# Test 3: Descending sort
clear_cache()
result3 = lf.groupby('group').agg(col('val').sum().alias('total')).sort('group', descending=True).compute()
groups3 = result3.column('group').to_pylist()
assert groups3 == ['c', 'b', 'a'], f'Expected descending, got {groups3}'
print('✅ Test 3: Descending sort works!')

# Test 4: Distinct on aggregated result
clear_cache()
table4 = pa.table({
    'x': ['a', 'b', 'a', 'c', 'b'],
    'y': [1.0, 2.0, 1.0, 3.0, 2.0],
})
lf4 = LazyFrame(Scan(table4))
result4 = lf4.groupby('x').agg(col('y').sum().alias('total')).distinct('x').compute()
xs = sorted(result4.column('x').to_pylist())
assert xs == ['a', 'b', 'c'], f'Expected distinct groups, got {xs}'
print('✅ Test 4: Distinct works!')

# Test 5: Multi-key sort (l_returnflag, l_linestatus style)
clear_cache()
table5 = pa.table({
    'flag':   ['R', 'A', 'N', 'R', 'A', 'N'],
    'status': ['F', 'F', 'O', 'F', 'F', 'O'],
    'val':    [1.0,  2.0, 3.0, 4.0, 5.0, 6.0],
})
lf5 = LazyFrame(Scan(table5))
result5 = (lf5.groupby('flag', 'status')
              .agg(col('val').sum().alias('total'))
              .sort('flag', 'status')
              .compute())
flags = result5.column('flag').to_pylist()
statuses = result5.column('status').to_pylist()
pairs = list(zip(flags, statuses))
# Should be sorted: (A,F), (N,O), (R,F)
assert pairs == [('A', 'F'), ('N', 'O'), ('R', 'F')], f'Multi-key sort failed: {pairs}'
print('✅ Test 5: Multi-key sort works!')

# Test 6: Limit without sort (just take first N)
clear_cache()
table6 = pa.table({'a': [10, 20, 30, 40, 50], 'b': [1.0, 2.0, 3.0, 4.0, 5.0]})
lf6 = LazyFrame(Scan(table6))
result6 = lf6.select(col('b')).limit(3).compute()
assert result6.num_rows == 3, f'Expected 3 rows, got {result6.num_rows}'
print('✅ Test 6: Limit on projection works!')

print('\nAll Phase 4 integration tests passed! 🎉')
