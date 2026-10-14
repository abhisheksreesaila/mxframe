# 🛠️ Contributing to MXFrame

> Developer guide — architecture, kernel writing, adding queries, running tests.

---

## 📐 Architecture Overview

A query is a lazy plan built in Python. At `.compute(device="cpu")` it is validated, optimized, and then answered by one of three CPU layers:

1. **Native Mojo executor** (`native_exec.py` → `kernels_aot/libmxexec_cpu.so`, source `executor_cpu.mojo`): one ctypes call over the Arrow C Data Interface runs a whole aggregate over a single table (filter, group keys, aggregates, HAVING, ORDER BY, LIMIT). It also serves filter, gather, DISTINCT, COUNT(DISTINCT), top-k and string-slice calls for the layers below.
2. **Fused pipeline** (`pipeline_exec.py` → `libmxpipeline_cpu.so` / `libmxpipeline_gpu.so`): filter + group + aggregate in one pass, for aggregates the executor declines.
3. **Per-operator path** (`custom_ops.py` → `libmxkernels_aot.so` / `libmxkernels_aot_gpu.so`, plus `join_engine.py`, `libmxgroupkeys_cpu.so`, `libmxjoin_cpu.so`): everything else, one kernel per operator. Joins are built here first; the aggregate on the joined table then goes back to layers 1 and 2.

The [behind-the-scenes page](https://abhisheksreesaila.github.io/mxframe/internals.html) walks through each layer, its seams, when it declines and the environment switches (`MXFRAME_NATIVE=0`, `MXFRAME_FUSED_PIPELINE=0`, ...). The file-by-file layout is in [docs/api.md](docs/api.md#source-layout).

### The compiled libraries

All are built ahead of time, committed, and shipped in the wheel; nothing compiles at run time. CPU libraries target `x86-64-v3` (AVX2), so a wheel doesn't SIGILL on older machines.

| Library | Source | Build |
|---|---|---|
| `libmxexec_cpu.so` | `kernels_aot/executor_cpu.mojo` | `bash scripts/build_pipeline.sh exec` |
| `libmxpipeline_cpu.so`, `libmxpipeline_gpu.so` | `pipeline_cpu.mojo`, `pipeline_gpu.mojo` | `bash scripts/build_pipeline.sh cpu` / `gpu` |
| `libmxkernels_aot.so`, `libmxkernels_aot_gpu.so` | `kernels_aot.mojo`, `kernels_aot_gpu.mojo` | step 2 of `scripts/release.sh` |
| `libmxgroupkeys_cpu.so`, `libmxjoin_cpu.so` | `groupkeys_cpu.mojo`, `joinsemi_cpu.mojo` | same form: `pixi run mojo build kernels_aot/<src>.mojo -I kernels_aot --target-cpu x86-64-v3 --emit shared-lib -o kernels_aot/<lib>.so` |
| `kernels.mojoc` | `kernels/*.mojo` | `bash scripts/build_kernels.sh` (the MAX Graph custom ops) |

---

## 🔁 Development Workflow

### 1. Setup

```sh
git clone https://github.com/abhisheksreesaila/mxframe
cd mxframe
pixi install             # Python, Mojo and all dependencies into .pixi/envs/default/
pixi run setup-mxframe   # links the repo into site-packages as `mxframe`
```

Python changes take effect immediately; Mojo changes need the library rebuilt (table above).

### 2. Test

```sh
pixi run test        # pytest over tests/ (works from any checkout or worktree)
pixi run test-all    # the older scripts/_test_*.py suites, as CI runs them
pixi run python3 scripts/_check_gpu.py   # GPU sanity check
```

New tests go in `tests/`. A change to the engine should test both the answer and the path that produced it (`native_exec.last_path()`, `lf.last_compile_provenance`), so a silent decline can't pass.

### 3. Add an operator to the native executor

Follow the worked example on the [behind-the-scenes page](https://abhisheksreesaila.github.io/mxframe/internals.html#add-operator): plan node, optimizer, the Python fallback, the encoding in `native_exec.py`, the Mojo function in `executor_cpu.mojo`, the rebuild, and tests against `MXFRAME_NATIVE=0` and DuckDB.

### 4. Benchmark before and after

```sh
# TPC-H, query time only, fresh process per query, checked against DuckDB
pixi run python scripts/bench_user_felt.py --mode engine --sf 1 --engines duckdb,mx-cpu --name my-change
# seeded random queries over the same tables
pixi run python scripts/bench_random/run.py --sf 1 --n 100 --name random-sf1-my-change
```

Run on a quiet machine (load under 2, no scheduled jobs); see `docs/lessons.md`.

---

## ✏️ Writing a New Mojo Kernel (per-operator path)

This is the older per-operator kernel path (layer 3). New engine work usually goes into the native executor instead (see step 3 above).

### Step 1 — Write the kernel in `kernels/`

Create `kernels/my_kernel.mojo`:

```mojo
from math import ceildiv
from gpu import block_dim, block_idx, thread_idx
from os.atomic import Atomic
from runtime.asyncrt import DeviceContextPtr
from tensor import ManagedTensorSlice

comptime BLOCK_SIZE = 256

fn _my_kernel_cpu(
    out: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=1, io_spec=_, static_spec=_],
    inp: ManagedTensorSlice[dtype=DType.float32, rank=1, io_spec=_, static_spec=_],
):
    var n = inp.dim_size(0)
    for i in range(n):
        out[i] = inp[i] * 2.0          # example: double every element

fn _my_kernel_gpu(
    out: ManagedTensorSlice[mut=True, dtype=DType.float32, rank=1, io_spec=_, static_spec=_],
    inp: ManagedTensorSlice[dtype=DType.float32, rank=1, io_spec=_, static_spec=_],
    ctx: DeviceContextPtr,
) raises:
    var n = inp.dim_size(0)
    @parameter
    fn kernel(n_: Int):
        var i = block_dim.x * block_idx.x + thread_idx.x
        if i < UInt(n_):
            out[Int(i)] = inp[Int(i)] * 2.0
    ctx.get_device_context().enqueue_function_experimental[kernel](
        n, grid_dim=ceildiv(n, BLOCK_SIZE), block_dim=BLOCK_SIZE)
```

### Step 2 — Export in `kernels_aot/kernels_aot.mojo` (CPU)

```mojo
# In kernels_aot/kernels_aot.mojo, add:

@export
fn my_kernel_cpu(out_addr: Int, inp_addr: Int, n: Int):
    var out_ptr = _f32(out_addr)
    var inp_ptr = _f32(inp_addr)
    for i in range(n):
        out_ptr[i] = inp_ptr[i] * 2.0
```

### Step 3 — Export in `kernels_aot/kernels_aot_gpu.mojo` (GPU)

```mojo
@export
fn my_kernel_gpu(out_addr: Int, inp_addr: Int, n: Int):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _f32(out_addr)[i] = _f32(inp_addr)[i] * 2.0
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass
```

### Step 4 — Bind in `custom_ops.py`

In the `AOTKernelsGPU` class (or `AOTKernelsCPU`), add the binding:

```python
def _bind(self):
    # ... existing bindings ...
    self._lib.my_kernel_gpu.argtypes = [
        ctypes.c_int64,   # out_addr
        ctypes.c_int64,   # inp_addr
        ctypes.c_int,     # n
    ]
    self._lib.my_kernel_gpu.restype = None
```

And the call wrapper:

```python
def my_kernel(self, inp: np.ndarray) -> np.ndarray:
    n = len(inp)
    out = np.zeros(n, dtype=np.float32)
    self._lib.my_kernel_gpu(
        out.ctypes.data_as(ctypes.c_void_p).value,
        inp.ctypes.data_as(ctypes.c_void_p).value,
        ctypes.c_int(n),
    )
    return out
```

### Step 5 — Rebuild and test

```sh
pixi run mojo build --emit shared-lib kernels_aot/kernels_aot_gpu.mojo \
    -o kernels_aot/libmxkernels_aot_gpu.so
pixi run python3 scripts/_test_aot_smoke.py
```

---

## 🧬 How the Execute Path Works

When you call `.compute(device="gpu")`:

```
LazyFrame.compute("gpu")
    → custom_ops.CustomOpsCompiler._execute(plan, device="gpu")
    → walks LogicalPlan tree bottom-up
    → for each node:
        Filter  → _execute_filter_aot()     → filter_gather_f32_gpu()
        Groupby → _execute_grouped_aot()    → group_encode() + group_sum_f32_gpu()
        Join    → _execute_hash_join()      → join_count_gpu() + join_scatter_gpu()
        Sort    → _execute_sort()           → sort_indices_gpu() + gather rows
        Limit   → pa.Table.slice() (trivial)
```

All AOT calls follow this pattern:

```python
# 1. Get contiguous NumPy array (zero-copy from Arrow when possible)
arr = np.ascontiguousarray(column, dtype=np.float32)

# 2. Get raw pointer
ptr = arr.ctypes.data_as(ctypes.c_void_p).value

# 3. Call AOT function
self._aot_gpu.group_sum_f32(out_ptr, arr_ptr, labels_ptr, n_rows, n_groups)

# 4. Wrap result back as Arrow array
result = pa.array(out_np, type=pa.float32())
```

---

## 🧪 Test Structure

### `scripts/_test_aot_smoke.py` — AOT Kernel Unit Tests

Tests each kernel independently with small hand-crafted inputs:

```
✅ group_sum_f32       — 5 groups, 20 rows
✅ group_min_f32       — boundary values
✅ group_max_f32       — boundary values
✅ group_mean_f32      — float precision
✅ group_count         — empty groups
✅ join_count / join_scatter    — inner join correctness
✅ join_count_left / join_scatter_left  — left outer join, nulls
✅ filter_gather_f32   — sparse mask
✅ gather_rows_f32     — permutation
✅ sort_indices        — ascending/descending
✅ unique_mask         — consecutive duplicates
✅ masked_global_sum   — masked reduce
```

### `scripts/_test_phase6_tpch_tier2.py` — TPC-H Integration Tests

Runs all 22 TPC-H queries and checks output matches Pandas reference:

```
✅ Q1  — 8-agg groupby
✅ Q3  — 3-table join + top-10 sort
✅ Q4  — EXISTS semi-join
...
✅ Q22 — phone prefix anti-join
```

Each check uses `np.allclose(rtol=1e-2)` for float columns and exact match for integer/string columns.

### Adding a New Test

Add to `scripts/_test_aot_smoke.py`:

```python
def test_my_kernel():
    inp = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    expected = np.array([2.0, 4.0, 6.0], dtype=np.float32)
    result = aot.my_kernel(inp)
    assert np.allclose(result, expected), f"Expected {expected}, got {result}"
    print("✅ my_kernel")

test_my_kernel()
```

---

## 🔢 Caching Architecture

`custom_ops.py` maintains several module-level caches to avoid recomputing:

| Cache | Key | Value | Purpose |
|---|---|---|---|
| `_GROUP_ENCODE_CACHE` | `(col_names, n_rows, device)` | `(labels, n_groups)` | Group label arrays |
| `_JOIN_RESULT_CACHE` | `(id(left), id(right), col_names)` | `(left_idx, right_idx)` | Join index pairs |
| `_Q7_PREJOINED` | `(id(nation), id(supplier), id(customer))` | pre-joined tables | Stable cross-call join input |
| `_Q17_PRECOMP` | `(id(part), id(lineitem), n_rows)` | `(target_parts, avg_table)` | Cached sub-query results |

The caches are **identity-keyed** (`id(table)`). As long as you pass the same Python object across hot benchmark runs, the caches hit and the join overhead is eliminated.

---

## 🚫 What NOT to Do

| ❌ Don't | ✅ Do instead |
|---|---|
| `.to_pandas()` in a hot path | Use `pc.*` (PyArrow compute) or `np.asarray()` |
| `.to_pylist()` on large arrays | `np.asarray(arrow_array)` for numerics, `pc.*` for strings |
| Python loops over rows | NumPy `isin`, `searchsorted`, `argsort`, vectorized ops |
| `pd.merge()` on large tables | Mojo join kernels via LazyFrame |
| Building Python sets from Arrow arrays | `pc.is_in(arr, value_set=other_arr)` |

---

## 🌿 Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Stable — all tests pass, benchmarks verified |
| `dev` | Active development |
| `feature/xxx` | New kernel or feature work |
| `fix/xxx` | Bug fixes |

PRs require:
1. All `_test_phase*.py` tests passing
2. `_test_aot_smoke.py` passing
3. No regression in `bench_simple.py` vs previous `main`

---

## 📋 Code Style

- **Python**: PEP 8, 100-char line limit. No type annotations required in hot paths.
- **Mojo**: `snake_case` for variables and functions. `UPPER_CASE` for `comptime` constants.
- **Comments**: Explain *why*, not *what*. One comment per non-obvious block.
- **No docstrings** in hot-path functions (they add startup cost to the `.so`).