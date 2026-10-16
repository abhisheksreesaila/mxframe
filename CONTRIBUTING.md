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

All are built ahead of time, committed, and shipped in the wheel; nothing compiles at run time. The native executor, pipeline, group-key and join libraries are built for `x86-64-v3` (AVX2), so a wheel doesn't SIGILL on older machines; `scripts/release.sh` builds the per-operator AOT kernels for Mojo's default target.

| Library | Source | Build (`bash scripts/build_pipeline.sh <target>`) |
|---|---|---|
| `libmxexec_cpu.so` | `executor_cpu.mojo` (imports `pipeline_cpu.mojo`) | `exec` |
| `libmxpipeline_cpu.so`, `libmxpipeline_gpu.so` | `pipeline_cpu.mojo`, `pipeline_gpu.mojo` | `cpu` (also rebuilds the executor) / `gpu` |
| `libmxkernels_aot.so`, `libmxkernels_aot_gpu.so` | `kernels_aot.mojo`, `kernels_aot_gpu.mojo` (imports `_atomic.mojo`) | `aot` (also step 2 of `scripts/release.sh`) |
| `libmxgroupkeys_cpu.so`, `libmxjoin_cpu.so` | `groupkeys_cpu.mojo`, `joinsemi_cpu.mojo` | `keys` / `join` |
| `kernels.mojoc` | `kernels/*.mojo` | `bash scripts/build_kernels.sh` (the MAX Graph custom ops) |

With no target the script rebuilds every library. Each build sets the relocatable RUNPATH and then runs `scripts/record_build_hash.py`, which writes `kernels_aot/<lib>.so.sources.json`: the SHA-256 of the library's own `.mojo` file and of every `kernels_aot/*.mojo` it imports (found by following `from X import` lines). Commit the `.so` and its `.sources.json` together. `tests/test_kernel_hashes.py` and the CI step `record_build_hash.py --check` fail with `libX.so is stale: <file> changed. Rebuild with bash scripts/build_pipeline.sh <target>` when a source changed without its library being rebuilt. A new library goes in the `LIBRARIES` table at the top of the script.

**After merging branches that edited `kernels_aot/*.mojo`, the hashes (and the libraries) no longer match the merged sources: run `bash scripts/build_pipeline.sh` once, with no argument, and commit the result.** Never run `record_build_hash.py` by hand without rebuilding first: it would record a hash for a library that was not built from those sources.

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

Copy an existing kernel rather than writing one from a template; the Mojo toolchain moves fast and the code in the repo is what builds today. The pattern, using grouped sum as the example:

| Step | CPU | GPU |
|---|---|---|
| 1. Export a C-ABI function (`@export`, `def … abi("C")`, buffers passed as integer addresses) | `group_sum_f32` in `kernels_aot/kernels_aot.mojo` | `group_sum_f32_gpu` in `kernels_aot/kernels_aot_gpu.mojo` (it calls the `run_group` helper; for a launch written out in full, `ctx.enqueue_function[...]` guarded by `if n > 0`, see `utf8_startswith_mask_gpu` in the same file) |
| 2. Bind it with ctypes (sizes are `c_int64`; Mojo `Int` is 64-bit) | `AOTKernels._bind` in `aot_kernels.py` (`self._fn(name, argtypes)`) | `AOTKernelsGPU._bind` in `aot_kernels.py` (`_bind_gpu(L, name, argtypes)`; arguments are device pointers, uploaded first) |
| 3. Call it from the operator path | `AOTKernels.group_sum` | the matching `AOTKernelsGPU` method |

The MAX Graph custom ops in `kernels/*.mojo` (e.g. `kernels/group_sum.mojo`, registered with `@register` from `extensibility`) are packaged into `kernels.mojoc` by `scripts/build_kernels.sh`.

### Rebuild and test

```sh
bash scripts/build_pipeline.sh aot   # builds, sets the RUNPATH (F-036), records source hashes (F-066)
pixi run python3 scripts/_test_aot_smoke.py
pixi run test
```

---

## 🚫 What NOT to Do

| ❌ Don't | ✅ Do instead |
|---|---|
| Import `pyarrow.compute` at module level (it costs tens of ms at import) | `from ._lazy_pc import pc`, which imports it on first use |
| `.to_pandas()` / `.to_pylist()` in a hot path | Keep Arrow buffers; hand them to Mojo over the Arrow C Data Interface, or `np.asarray()` for numerics |
| Python loops over rows | A native call, or vectorized NumPy |
| Special cases for one TPC-H query | General rules; check them on shuffled data and on the random suite (`scripts/bench_random/`) |
| A native path that might answer differently | Decline (return `None`) and let the next layer answer; test against `MXFRAME_NATIVE=0` and DuckDB |

---

## 🌿 Branches and PRs

Work on a branch off `main` and open a PR into `main`. Before a PR:
1. `pixi run test` and `pixi run test-all` pass (CI runs the `scripts/_test_*.py` suites).
2. Engine changes come with a before/after from `scripts/bench_user_felt.py --mode engine` (and the random suite when the change is general), run on a quiet machine.
3. Docs that quote numbers or describe the engine (`README.md`, `docs/site/`) still match; `tests/test_readme_api.py` and `tests/test_site_*.py` check most of it.

---

## 📋 Code Style

- **Python**: PEP 8, 100-char line limit. No type annotations required in hot paths.
- **Mojo**: `snake_case` for variables and functions. `UPPER_CASE` for `comptime` constants.
- **Comments**: Explain *why*, not *what*. One comment per non-obvious block.
- **No docstrings** in hot-path functions (they add startup cost to the `.so`).