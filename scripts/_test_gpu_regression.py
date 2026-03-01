"""GPU end-to-end regression test for all 5 grouped aggregations."""
import numpy as np
import pyarrow as pa
from mxframe.lazy_frame import LazyFrame, Scan
from mxframe.lazy_expr import col

# N=8 rows, 2 groups — matches the standalone Mojo test data
# group 0: values [1,3,5,7]  -> sum=16, min=1, max=7, mean=4, count=4
# group 1: values [2,4,6,8]  -> sum=20, min=2, max=8, mean=5, count=4
table = pa.table({
    'g': pa.array([0, 1, 0, 1, 0, 1, 0, 1], type=pa.int32()),
    'v': pa.array([1., 2., 3., 4., 5., 6., 7., 8.], type=pa.float32()),
})
lf = LazyFrame(Scan(table))

print("Testing device='gpu' ...")
r = lf.groupby('g').agg(
    col('v').sum(),
    col('v').min(),
    col('v').max(),
    col('v').mean(),
    col('v').count(),
).compute('gpu')
print(r)
print()

r_sorted = r.sort_by('g')
g  = r_sorted.column('g').to_pylist()
s  = r_sorted.column('agg_0').to_pylist()
mn = r_sorted.column('agg_1').to_pylist()
mx = r_sorted.column('agg_2').to_pylist()
me = r_sorted.column('agg_3').to_pylist()
ct = r_sorted.column('agg_4').to_pylist()

print(f"group 0:  sum={s[0]} (exp 16)  min={mn[0]} (exp 1)  max={mx[0]} (exp 7)  mean={me[0]} (exp 4)  count={ct[0]} (exp 4)")
print(f"group 1:  sum={s[1]} (exp 20)  min={mn[1]} (exp 2)  max={mx[1]} (exp 8)  mean={me[1]} (exp 5)  count={ct[1]} (exp 4)")
print()

assert abs(s[0]  - 16.0) < 1e-4, f"sum g0 wrong: {s[0]}"
assert abs(s[1]  - 20.0) < 1e-4, f"sum g1 wrong: {s[1]}"
assert abs(mn[0] -  1.0) < 1e-4, f"min g0 wrong: {mn[0]}"
assert abs(mn[1] -  2.0) < 1e-4, f"min g1 wrong: {mn[1]}"
assert abs(mx[0] -  7.0) < 1e-4, f"max g0 wrong: {mx[0]}"
assert abs(mx[1] -  8.0) < 1e-4, f"max g1 wrong: {mx[1]}"
assert abs(me[0] -  4.0) < 1e-4, f"mean g0 wrong: {me[0]}"
assert abs(me[1] -  5.0) < 1e-4, f"mean g1 wrong: {me[1]}"
assert abs(ct[0] -  4.0) < 1e-4, f"count g0 wrong: {ct[0]}"
assert abs(ct[1] -  4.0) < 1e-4, f"count g1 wrong: {ct[1]}"

print("ALL GPU ASSERTIONS PASSED")
