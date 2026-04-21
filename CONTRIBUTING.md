# 🛠️ Contributing to MXFrame

> Developer guide — architecture, kernel writing, adding queries, running tests.

---

## 📐 Architecture Overview

```
Python API (LazyFrame / SQL)
        │
        ▼
  LogicalPlan AST
        │
        ▼
   Optimizer         ← filter pushdown, join reordering
        │
        ▼
   Compiler          ← translates plan nodes → dispatch calls
        │
   ┌────┴─────────────────────────────┐
   │                                  │
   ▼                                  ▼
AOT Kernels (ctypes)          PyArrow fallback
libmxkernels_aot.so           (string ops, pc.* calls)
libmxkernels_aot_gpu.so
   │
   ├── CPU: direct ctypes call → Mojo function runs on CPU
   └── GPU: MAX Engine DeviceContext → kernel dispatched to GPU
```

### The Two Compile-Time Artifacts

| File | Built how | Used how |
|---|---|---|
| `kernels_aot/libmxkernels_aot.so` | `mojo build --emit shared-lib kernels_aot/kernels_aot.mojo` | `ctypes.CDLL()` at process start, called directly |
| `kernels_aot/libmxkernels_aot_gpu.so` | `mojo build --emit shared-lib kernels_aot/kernels_aot_gpu.mojo` | Loaded once, GPU kernels invoked via MAX DeviceContext |

**Key insight:** These `.so` files are compiled **once** at build time and shipped in the wheel.  
At runtime — `ctypes.CDLL()` takes ~1 ms. No JIT, no MLIR graph build, no recompilation ever.

---

## 🗂️ Directory Map

```
┌── __init__.py              Public API re-exports
├── lazy_expr.py             Expr AST nodes: col(), lit(), when(), arithmetic, agg
├── lazy_frame.py            LazyFrame, LazyGroupBy, Scan, .compute()
├── compiler.py              AST → MAX Graph (legacy) or AOT dispatch
├── custom_ops.py            AOT kernel dispatch, join implementations,
│                            group_encode, shape inference, caching
├── optimizer.py             Plan rewrites
├── plan_validation.py       Pre-run sanity checks
├── sql_frontend.py          sqlglot → LogicalPlan
│
├── kernels_v261/            Mojo kernel SOURCE (not shipped in wheel)
│   ├── __init__.mojo
│   ├── group_sum.mojo        scatter-add by group label (CPU + GPU)
│   ├── group_min.mojo        scatter-min
│   ├── group_max.mojo        scatter-max
│   ├── group_mean.mojo       scatter-mean
│   ├── group_count.mojo      scatter-count
│   ├── group_composite.mojo  multi-key groupby via composite int64 key
│   ├── join_count.mojo       inner join: count matches per left row
│   ├── join_scatter.mojo     inner join: emit (left_idx, right_idx) pairs
│   ├── join_count_left.mojo  left-outer join: count (min 1 per left row)
│   ├── join_scatter_left.mojo left-outer join: emit pairs, -1 for no-match
│   ├── filter_gather.mojo    compacted gather with mask
│   ├── gather_rows.mojo      permute rows by index array
│   ├── sort_indices.mojo     argsort
│   ├── unique_mask.mojo      mark first-in-sorted-run
│   └── masked_global_agg.mojo masked sum / sum-of-products
│
├── kernels_aot/             AOT entry point Mojo files + compiled .so files
│   ├── kernels_aot.mojo      @export CPU functions (ctypes ABI)
│   ├── kernels_aot_gpu.mojo  @export GPU functions (MAX DeviceContext ABI)
│   ├── libmxkernels_aot.so   ← COMPILED (tracked in git LFS or built locally)
│   └── libmxkernels_aot_gpu.so
│
└── scripts/
    ├── bench_simple.py       4-column benchmark: Pandas|Polars|MX CPU|MX GPU
    ├── benchmark_tpch.py     All 22 TPC-H query implementations + data generators
    ├── _test_aot_smoke.py    AOT kernel unit tests
    ├── _test_phase*.py       Integration test suites
    └── build_kernels.sh      Rebuild both .so files
```

---

## 🔁 Development Workflow

### 1. Setup

```bash
git clone https://github.com/abhisheksreesaila/mxframe
cd mxframe
pixi install          # installs Python + Mojo + all deps into .pixi/envs/default/
pixi run setup-mxframe   # creates mxframe symlink in site-packages
```

### 2. Make a Python Change

Edit any of the `.py` files directly (compiler, lazy_frame, custom_ops, etc.).  
The editable install means changes take effect immediately — no reinstall needed.

```bash
# Verify your change with smoke tests
pixi run python3 scripts/_test_aot_smoke.py

# Run specific query tests
pixi run python3 scripts/_test_phase6_tpch_tier2.py

# Quick benchmark check
pixi run python3 scripts/bench_simple.py --rows 100000 --runs 2 --queries 1,3,6
```

### 3. Add a New Mojo Kernel

See the [Kernel Writing Guide](#-writing-a-new-mojo-kernel) below.

After writing the kernel:

```bash
# Rebuild the CPU .so
pixi run mojo build --emit shared-lib kernels_aot/kernels_aot.mojo \
    -o kernels_aot/libmxkernels_aot.so

# Rebuild the GPU .so
pixi run mojo build --emit shared-lib kernels_aot/kernels_aot_gpu.mojo \
    -o kernels_aot/libmxkernels_aot_gpu.so

# Verify
pixi run python3 scripts/_test_aot_smoke.py
```

### 4. Run the Full Test Suite

```bash
# Unit tests (AOT kernel smoke)
pixi run python3 scripts/_test_aot_smoke.py

# Integration tests
pixi run python3 scripts/_test_phase0_compiler.py
pixi run python3 scripts/_test_phase1.py
pixi run python3 scripts/_test_phase4.py
pixi run python3 scripts/_test_phase5.py
pixi run python3 scripts/_test_phase6_tpch_tier2.py
pixi run python3 scripts/_test_phase_sql.py
pixi run python3 scripts/_test_phase_q5_q10.py

# Or run all at once
pixi run test-all

# GPU sanity check
pixi run python3 scripts/_check_gpu.py
```

### 5. Benchmark Before/After

```bash
# Full 22 queries, 1M rows, 3 hot runs
pixi run python3 scripts/bench_simple.py --rows 1000000 --runs 3

# Specific queries only
pixi run python3 scripts/bench_simple.py --rows 1000000 --runs 3 --queries 1,6,22
```

---

## ✏️ Writing a New Mojo Kernel

### Step 1 — Write the kernel in `kernels_v261/`

Create `kernels_v261/my_kernel.mojo`:

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

```bash
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
