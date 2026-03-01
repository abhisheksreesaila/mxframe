"""Phase 1 notebook patch — wires group_min/max/mean/count into 04_custom_ops.ipynb."""
import json

# ── New _visit_aggregate_custom block ────────────────────────────────────────
# (replaces only the method body inside the class cell)

OLD_AGG = """\
    def _visit_aggregate_custom(self, plan, nodes, *, n_groups):
        upstream = self._visit_plan_custom(plan.input, nodes, n_groups=n_groups)
        out = {}
        grouped = bool(plan.group_by)
        for i, expr in enumerate(plan.aggs):
            name = expr._alias or f'agg_{i}'
            if grouped and n_groups > 0:
                if expr.op == "sum":
                    # Route to the Mojo group_sum kernel (float32 values + int32 ids)
                    val_node = ops.cast(self._visit_expr(expr.args[0], upstream), DType.float32)
                    gid_node = ops.cast(upstream["__group_ids__"], DType.int32)
                    out_type = TensorType(DType.float32, [n_groups], DeviceRef.CPU())
                    results = ops.custom(
                        name="group_sum",
                        values=[val_node, gid_node],
                        out_types=[out_type],
                        device=DeviceRef.CPU(),
                    )
                    out[name] = results[0]
                else:
                    raise NotImplementedError(
                        f"Grouped '{expr.op}' is not yet wired. "
                        f"Implement group_{expr.op}.mojo and register it in Phase 1."
                    )
            else:
                # Global (non-grouped) aggregation -- fall through to built-in MAX ops
                out[name] = self._visit_expr(expr, upstream)
        return out"""

NEW_AGG = """\
    def _visit_aggregate_custom(self, plan, nodes, *, n_groups):
        upstream = self._visit_plan_custom(plan.input, nodes, n_groups=n_groups)
        out = {}
        grouped = bool(plan.group_by)
        for i, expr in enumerate(plan.aggs):
            name = expr._alias or f'agg_{i}'
            if grouped and n_groups > 0:
                gid_node = ops.cast(upstream["__group_ids__"], DType.int32)
                out_type = TensorType(DType.float32, [n_groups], DeviceRef.CPU())
                if expr.op == "count":
                    # group_count kernel: takes only group_ids (no values needed)
                    results = ops.custom(
                        name="group_count",
                        values=[gid_node],
                        out_types=[out_type],
                        device=DeviceRef.CPU(),
                    )
                    out[name] = results[0]
                elif expr.op in ("sum", "min", "max", "mean"):
                    # group_sum / group_min / group_max / group_mean kernels
                    val_node = ops.cast(self._visit_expr(expr.args[0], upstream), DType.float32)
                    results = ops.custom(
                        name=f"group_{expr.op}",
                        values=[val_node, gid_node],
                        out_types=[out_type],
                        device=DeviceRef.CPU(),
                    )
                    out[name] = results[0]
                else:
                    raise NotImplementedError(
                        f"Grouped '{expr.op}' is not yet supported. "
                        f"Add a group_{expr.op}.mojo kernel and wire it in a future phase."
                    )
            else:
                # Global (non-grouped) aggregation -- fall through to built-in MAX ops
                out[name] = self._visit_expr(expr, upstream)
        return out"""

# ── New docstring (reflects Phase 1 completeness) ────────────────────────────
OLD_DOCSTRING = """\
    \"\"\"Compiles a LogicalPlan into a MAX Graph with custom Mojo kernels.

    Extends GraphCompiler with:
    1. Pre-filter   -- inherited _strip_filters applies Filter nodes in PyArrow.
    2. Group ids    -- PyArrow dictionary-encodes group-by keys into int32 ids.
    3. Kernel dispatch -- group_sum routed to the Mojo custom kernel.
    4. Keys in result -- group-by key columns are prepended to the output table.
    5. Guarded fallback -- grouped min/max/mean/count raise NotImplementedError
       rather than silently returning wrong global results.
    \"\"\""""

NEW_DOCSTRING = """\
    \"\"\"Compiles a LogicalPlan into a MAX Graph with custom Mojo kernels.

    Extends GraphCompiler with:
    1. Pre-filter   -- inherited _strip_filters applies Filter nodes in PyArrow.
    2. Group ids    -- PyArrow dictionary-encodes group-by keys into int32 ids.
    3. Kernel dispatch -- grouped sum/min/max/mean/count routed to Mojo kernels.
    4. Keys in result -- group-by key columns are prepended to the output table.
    \"\"\""""

# ── New tests cell (Phase 0 tests 1-4 retained; Test 5 replaced by Phase 1) ──
NEW_TESTS = """\
import pyarrow as pa
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame, Scan

kernels_path = (
    str(Path(__file__).parent.parent / "mxframe" / "kernels.mojopkg")
    if "__file__" in dir()
    else str(Path("/home/ablearn/mxdf/mxframe/kernels.mojopkg"))
)

compiler = CustomOpsCompiler(kernels_path)

# ── Test 1: Projection falls through to built-in ops ──
table = pa.table({'a': [1, 2, 3], 'b': [4, 5, 6]})
lf = LazyFrame(Scan(table))
result = compiler.compile_and_run(lf.select(col('a') + lit(10)).plan)
assert result.num_columns == 1
assert result.column(0).to_pylist() == [11, 12, 13], f'Projection failed: {result}'
print('✅ Test 1 passed: projection')

# ── Test 2: Global sum via built-in ops ──
table2 = pa.table({'x': [1.0, 2.0, 3.0, 4.0]})
lf2 = LazyFrame(Scan(table2))
result2 = compiler.compile_and_run(
    lf2.groupby().agg(col('x').sum().alias('total')).plan
)
assert abs(result2.column('total').to_pylist()[0] - 10.0) < 1e-6, f'Sum failed: {result2}'
print('✅ Test 2 passed: global sum aggregation')

# ── Test 3: Grouped sum via Mojo group_sum kernel -- with group keys in result ──
table3 = pa.table({
    'group': ['a', 'b', 'a', 'b', 'a'],
    'val':   [1.0, 2.0, 3.0, 4.0, 5.0],
})
lf3 = LazyFrame(Scan(table3))
result3 = compiler.compile_and_run(
    lf3.groupby('group').agg(col('val').sum().alias('total')).plan
)
assert 'group' in result3.column_names, f'Missing group key: {result3.column_names}'
assert result3.column_names[0] == 'group', f'Key should be first: {result3.column_names}'
groups = result3.column('group').to_pylist()
totals = result3.column('total').to_pylist()
result_dict = dict(zip(groups, totals))
assert abs(result_dict['a'] - 9.0) < 1e-6, f'Sum for a should be 9.0: {result_dict}'
assert abs(result_dict['b'] - 6.0) < 1e-6, f'Sum for b should be 6.0: {result_dict}'
print('✅ Test 3 passed: grouped sum with key columns in result (a=9, b=6)')

# ── Test 4: Filter removes rows before grouped aggregation ──
table4 = pa.table({
    'group': ['a', 'b', 'a', 'b', 'a'],
    'val':   [1.0, 2.0, 3.0, 4.0, 5.0],
    'flag':  [1,   1,   0,   1,   1  ],
})
lf4 = LazyFrame(Scan(table4))
result4 = compiler.compile_and_run(
    lf4.filter(col('flag') > lit(0)).groupby('group').agg(col('val').sum().alias('total')).plan
)
groups4 = result4.column('group').to_pylist()
totals4 = result4.column('total').to_pylist()
result_dict4 = dict(zip(groups4, totals4))
assert abs(result_dict4['a'] - 6.0) < 1e-6, f'Filtered sum for a should be 6.0: {result_dict4}'
assert abs(result_dict4['b'] - 6.0) < 1e-6, f'Filtered sum for b should be 6.0: {result_dict4}'
print('✅ Test 4 passed: filter removes rows before grouped aggregation')

# ─── Phase 1 tests ────────────────────────────────────────────────────────────
# Data: group x = [1.0, 2.0], group y = [3.0]
# min(x)=1  max(x)=2  mean(x)=1.5  count(x)=2
# min(y)=3  max(y)=3  mean(y)=3.0  count(y)=1
table5 = pa.table({'g': ['x', 'x', 'y'], 'v': [1.0, 2.0, 3.0]})
lf5 = LazyFrame(Scan(table5))

def _as_dict(result, key_col, val_col):
    return dict(zip(result.column(key_col).to_pylist(), result.column(val_col).to_pylist()))

# ── Test 5: Grouped min ──
r5 = compiler.compile_and_run(lf5.groupby('g').agg(col('v').min().alias('mn')).plan)
d5 = _as_dict(r5, 'g', 'mn')
assert abs(d5['x'] - 1.0) < 1e-5, f'min x should be 1.0: {d5}'
assert abs(d5['y'] - 3.0) < 1e-5, f'min y should be 3.0: {d5}'
print('✅ Test 5 passed: grouped min (x=1.0, y=3.0)')

# ── Test 6: Grouped max ──
r6 = compiler.compile_and_run(lf5.groupby('g').agg(col('v').max().alias('mx')).plan)
d6 = _as_dict(r6, 'g', 'mx')
assert abs(d6['x'] - 2.0) < 1e-5, f'max x should be 2.0: {d6}'
assert abs(d6['y'] - 3.0) < 1e-5, f'max y should be 3.0: {d6}'
print('✅ Test 6 passed: grouped max (x=2.0, y=3.0)')

# ── Test 7: Grouped mean ──
r7 = compiler.compile_and_run(lf5.groupby('g').agg(col('v').mean().alias('avg')).plan)
d7 = _as_dict(r7, 'g', 'avg')
assert abs(d7['x'] - 1.5) < 1e-5, f'mean x should be 1.5: {d7}'
assert abs(d7['y'] - 3.0) < 1e-5, f'mean y should be 3.0: {d7}'
print('✅ Test 7 passed: grouped mean (x=1.5, y=3.0)')

# ── Test 8: Grouped count ──
r8 = compiler.compile_and_run(lf5.groupby('g').agg(col('v').count().alias('cnt')).plan)
d8 = _as_dict(r8, 'g', 'cnt')
assert abs(d8['x'] - 2.0) < 1e-5, f'count x should be 2: {d8}'
assert abs(d8['y'] - 1.0) < 1e-5, f'count y should be 1: {d8}'
print('✅ Test 8 passed: grouped count (x=2, y=1)')

# ── Test 9: Multi-agg groupby (sum + mean in one call) ──
r9 = compiler.compile_and_run(
    lf5.groupby('g').agg(
        col('v').sum().alias('total'),
        col('v').mean().alias('avg'),
    ).plan
)
assert 'g' in r9.column_names and 'total' in r9.column_names and 'avg' in r9.column_names
d9_sum  = _as_dict(r9, 'g', 'total')
d9_mean = _as_dict(r9, 'g', 'avg')
assert abs(d9_sum['x']  - 3.0) < 1e-5, f'sum x should be 3.0: {d9_sum}'
assert abs(d9_mean['x'] - 1.5) < 1e-5, f'mean x should be 1.5: {d9_mean}'
print('✅ Test 9 passed: multi-agg groupby (sum + mean)')

print('\\nAll CustomOpsCompiler tests passed! 🎉')
"""

with open('nbs/04_custom_ops.ipynb') as f:
    nb = json.load(f)

patched = 0
for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    src = ''.join(cell['source'])
    if '_visit_aggregate_custom' in src and 'KERNELS_PATH' in src:
        # patch docstring
        if OLD_DOCSTRING in src:
            src = src.replace(OLD_DOCSTRING, NEW_DOCSTRING)
            patched += 1
            print("  ✓ patched class docstring")
        # patch method
        if OLD_AGG in src:
            src = src.replace(OLD_AGG, NEW_AGG)
            patched += 1
            print("  ✓ patched _visit_aggregate_custom")
        else:
            print("  ✗ OLD_AGG not found — printing first 200 chars of agg section:")
            idx = src.find('_visit_aggregate_custom')
            print(repr(src[idx:idx+300]))
        cell['source'] = src.splitlines(keepends=True)
    elif 'Test 5' in src and 'Phase 1' in src and 'NotImplementedError' in src:
        cell['source'] = NEW_TESTS.splitlines(keepends=True)
        patched += 1
        print("  ✓ patched tests cell (Phase 0+1 tests)")

with open('nbs/04_custom_ops.ipynb', 'w') as f:
    json.dump(nb, f, indent=1)
print(f"Saved 04_custom_ops.ipynb ({patched} changes)")
