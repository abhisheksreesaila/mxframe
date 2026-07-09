# 📋 Changelog

All notable changes to MXFrame are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.2.3] — 2026-07-09

### ⚡ Performance
- **Phase 4 — Device-resident GPU join pipeline** (`join_count_i32_gpu` + `join_scatter_i32_gpu`):
  GPU inner joins now bypass MAX Graph JIT entirely. Cold-start join latency drops from ~300 ms (MAX session init + JIT) to ~70 ms. `AOTKernelsGPU.hash_join()` caches key arrays via `_cached_upload`; `AOTKernelsGPU.gather_table()` gathers all numeric columns with a single shared device-resident index upload.
- **Phase 5 — GPU sort + top-k** (`sort_topk_i32_gpu`): `ORDER BY … LIMIT k` fuses sort and limit in a single AOT bitonic sort kernel, downloading only `k × 4` bytes instead of `N × 4`. `_apply_post_ops_custom` peek-ahead passes `topk=k` directly to the sort kernel.
- **Phase 3 — Vectorised rank mapping**: `_encode_sort_key` replaces the per-unique-value Python `for` loop with a single vectorised NumPy scatter (`rank_of[sorted_order] = np.arange(n_unique)`). ~1.5× speedup for sort-key encoding at 5 M rows.

### 🛠 Fixed
- **GPU sort+limit crash**: `_apply_sort_custom` was calling `_gpu_gather_table` which referenced `gather_f32` as a MAX Graph custom op — that kernel was never registered in the mojopkg. Any `ORDER BY` on `device="gpu"` would crash. Now routes through `_aot_gpu.gather_table` (AOT ctypes, no MAX Graph).
- `_apply_sort_custom` CPU path now uses `np.argpartition` (O(N) partial sort) when `topk < N`, instead of a full O(N log N) argsort.

### 🔨 Infrastructure
- **Mojo/MAX 26.4 migration**: all `kernels/*.mojo` and `kernels_aot/*.mojo` migrated from 26.2 API (`fn` → `def`, `from tensor import` → `from extensibility import`, `DeviceContextPtr` → `DeviceContext`, `enqueue_function_experimental` → `enqueue_function`).
- **`kernels/_atomic.mojo`** (new): drop-in replacement for the missing `std.os.atomic.Atomic`. Implements `fetch_add`, `min`, `max`, `compare_exchange` via MLIR `llvm.atomicrmw` / `llvm.cmpxchg` instructions — portable across NVIDIA and AMD; works in both `mojo package` (JIT) and `mojo build --emit shared-lib` (AOT) contexts.
- All compiled artifacts rebuilt with Mojo 1.0.0b2 (modular 26.4): `kernels.mojopkg`, `libmxkernels_aot.so`, `libmxkernels_aot_gpu.so`.

---

## [0.2.2] — 2026-07-08

### 🔨 Infrastructure
- First release targeting **Mojo/MAX 26.4** (`modular = "26.4.*"` in `pixi.toml`).
- `_atomic.mojo` shim introduced to replace the removed `std.os.atomic` module.
- All kernel source files migrated to 26.4 syntax; compiled artefacts rebuilt.
- `pyproject.toml` `runtime` extra pinned to `modular>=26.4` to prevent 26.2 wheels from being installed by Colab/pip users.

---

## [0.2.1] — 2026-06-xx

### 🔨 Infrastructure
- TestPyPI re-upload to validate wheel packaging. No functional changes.

---

## [0.2.0] — 2026-05-xx

### ✨ Added
- **Phase 1 — GPU buffer cache + masked global min/max**:
  - `masked_global_min_f32_gpu` / `masked_global_max_f32_gpu` Mojo AOT kernels.
  - `AOTKernelsGPU` device-buffer cache (`_cached_upload`, eviction, `clear_buf_cache`) — eliminates redundant PCIe uploads on repeated hot queries.
  - `group_mean_f32` GPU kernel (sum + count on GPU, divide on CPU).
  - Masked global min/max fast-paths wired in `custom_ops.py`.
- **Phase 2 — Masked global aggregation for any expression**:
  - `_compute_masked_global_agg` extended to handle any inner expression via `_eval_expr_arrow` (e.g. `a*(1-b)`, `case_when`, `startswith`, `isin`) — SUM reduction stays GPU-resident.
  - Join-only plans emit `"join_mojo_shortcut"` provenance (not `"pyarrow_shortcut"`) — Q19 restructured to single-pass join+filter+global_agg.
  - `audit_gpu_paths.py` updated: `join_mojo_shortcut` counted as Mojo-backed.
- **Phase 3 — GPU integer key encoding**:
  - `group_encode_i32_gpu` open-addressing hash-table kernel for integer group keys; replaces PyArrow `dictionary_encode` for integer-keyed group-by on GPU.
- **Audit tool** (`scripts/audit_gpu_paths.py`): classifies each of 22 TPC-H queries as `GPU-CLEAN` / `MOJO-CPU` / `FALLBACK` via `last_compile_provenance["path"]`.
- **Interactive visualiser** (`visualize/mxframe_pipeline.html`): step-through of how a Python query becomes a plan tree and dispatches to GPU kernels.

### ⚡ Performance
- **22/22 TPC-H queries GPU-CLEAN** (100% Mojo-backed on GPU, zero PyArrow fallbacks in the reduction and join hot paths).

---

## [0.1.2] — 2026-04-22

### 🛠 Fixed
- `pip install "mxframe[runtime]"` now gives a working install. The `[runtime]` extra pulls `modular>=25.5`, which provides the `max.engine` / `max.driver` modules that `mxframe.compiler` imports at module load. (Pixi users get `modular` from the Modular conda channel and should install `mxframe` without the extra.)
- `sqlglot` import in `mxframe.sql_frontend` is now lazy: `import mxframe` no longer requires the `[sql]` extra. Calling `mxframe.sql(…)` without `sqlglot` installed raises a clear `ImportError` with the install hint.
- `__init__.py` `__version__` now matches `pyproject.toml`.

---

## [0.1.1] — 2026-04-22

### 🛠 Fixed
- Packaging: `readme` now points to `README.md` so PyPI shows the full project page with benchmarks and quickstart (was accidentally showing the internal vision/architecture doc)
- CI: `AOTKernelsGPU` init no longer hard-fails on CPU-only runners (`CUDA driver unavailable` is caught, `self._aot_gpu` falls back to `None`)
- CI: `_find_kernels_path()` uses `Path(__file__).resolve().parent` directly, removing the stale `/home/ablearn/mxdf/…` dev fallback that surfaced in installed wheels
- Bug: `NameError: cache_key_count` in `_hash_join_mojo_cpu` MAX Graph fallback
- Tests: `scripts/_test_phase0_custom_ops.py` and `_test_phase1.py` no longer hard-code a developer's kernels path; they use the auto-detected default

---

## [0.1.0] — 2026-04-20

### ✨ Added
- **All 22 TPC-H queries** on CPU and GPU AOT paths
- **GPU LEFT JOIN** — `join_count_left` + `join_scatter_left` Mojo kernels wired to Python (Q13)
- `bench_simple.py` — clean 4-column benchmark (Pandas | Polars | MX CPU | MX GPU)
- `--queries` and `--runs` flags for targeted benchmarking

### ⚡ Performance
- **Q22** phone prefix anti-join: vectorized via `pc.utf8_slice_codeunits` + `pc.is_in` + `np.isin` — 3.6× faster than Polars CPU (was 10× **slower**)
- **Q20** semi-join chain: eliminated `.to_pandas()` Pandas detour — 8.7× faster than Polars (was comparable)
- **Q21** EXISTS+NOT EXISTS: replaced `.to_pandas().groupby.nunique()` with NumPy composite key dedup
- **Q18** large volume customers: replaced `.to_pylist()` semi-join with Mojo join — semi-join now uses AOT kernel path
- **Q15** argmax supplier: replaced `0.9999 * max_rev` tolerance hack with exact `pc.equal()` on float32
- **Q13** LEFT JOIN: removed `.to_pandas()` groupby detour — now fully on GPU AOT path (19ms CPU, 29ms GPU vs 25ms Polars)

### 🐛 Fixed
- `not use_gpu_filter` gate bug in `custom_ops.py` section 4.6 that blocked GPU filter path
- Q7 nation pre-join cache key now uses `id()` preventing cache misses on repeated hot calls

### 🏗️ Infrastructure
- `pyproject.toml` updated with full metadata, dependencies, optional extras, package-data for `.so` files
- `CONTRIBUTING.md` — developer guide with kernel writing tutorial
- `PUBLISHING.md` — step-by-step pip release guide
- GitHub Actions: `test.yml` (CI), `publish.yml` (PyPI via OIDC)

### 📊 Benchmark Summary (1M and 10M rows, all 22 queries, warm median of 3)

See the full table in [README.md § TPC-H Benchmark](README.md#-tpc-h-benchmark--all-22-queries),
sourced from `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv`.

- **Correctness:** 22/22 queries pass on CPU and GPU paths
- **MX CPU wins vs Polars:** **21/22** at 1 M, **18/22** at 10 M
  — headline: **Q9 128× · Q12 89× · Q7 42× · Q8 31× · Q17 24× · Q5 22×** (at 10 M)
- **MX GPU wins vs Polars:** **16/22** at 1 M, **15/22** at 10 M
  — headline: **Q12 26.5× · Q9 12× · Q8 10.8× · Q17 9.7× · Q7 4.7×** (at 10 M)
- **Remaining losses (Q4, Q6, Q13, Q21):** ops that still route through PyArrow fallback — kernel replacements tracked in `roadmap.md`

### 🛠 Fixed
- `pip install "mxframe[runtime]"` now gives a working install. The `[runtime]` extra pulls `modular>=25.5`, which provides the `max.engine` / `max.driver` modules that `mxframe.compiler` imports at module load. (Pixi users get `modular` from the Modular conda channel and should install `mxframe` without the extra.)
- `sqlglot` import in `mxframe.sql_frontend` is now lazy: `import mxframe` no longer requires the `[sql]` extra. Calling `mxframe.sql(…)` without `sqlglot` installed raises a clear `ImportError` with the install hint.
- `__init__.py` `__version__` now matches `pyproject.toml`.

---

## [0.1.1] — 2026-04-22

### 🛠 Fixed
- Packaging: `readme` now points to `README.md` so PyPI shows the full project page with benchmarks and quickstart (was accidentally showing the internal vision/architecture doc)
- CI: `AOTKernelsGPU` init no longer hard-fails on CPU-only runners (`CUDA driver unavailable` is caught, `self._aot_gpu` falls back to `None`)
- CI: `_find_kernels_path()` uses `Path(__file__).resolve().parent` directly, removing the stale `/home/ablearn/mxdf/…` dev fallback that surfaced in installed wheels
- Bug: `NameError: cache_key_count` in `_hash_join_mojo_cpu` MAX Graph fallback
- Tests: `scripts/_test_phase0_custom_ops.py` and `_test_phase1.py` no longer hard-code a developer's kernels path; they use the auto-detected default

---

## [0.1.0] — 2026-04-20

### ✨ Added
- **All 22 TPC-H queries** on CPU and GPU AOT paths
- **GPU LEFT JOIN** — `join_count_left` + `join_scatter_left` Mojo kernels wired to Python (Q13)
- `bench_simple.py` — clean 4-column benchmark (Pandas | Polars | MX CPU | MX GPU)
- `--queries` and `--runs` flags for targeted benchmarking

### ⚡ Performance
- **Q22** phone prefix anti-join: vectorized via `pc.utf8_slice_codeunits` + `pc.is_in` + `np.isin` — 3.6× faster than Polars CPU (was 10× **slower**)
- **Q20** semi-join chain: eliminated `.to_pandas()` Pandas detour — 8.7× faster than Polars (was comparable)
- **Q21** EXISTS+NOT EXISTS: replaced `.to_pandas().groupby.nunique()` with NumPy composite key dedup
- **Q18** large volume customers: replaced `.to_pylist()` semi-join with Mojo join — semi-join now uses AOT kernel path
- **Q15** argmax supplier: replaced `0.9999 * max_rev` tolerance hack with exact `pc.equal()` on float32
- **Q13** LEFT JOIN: removed `.to_pandas()` groupby detour — now fully on GPU AOT path (19ms CPU, 29ms GPU vs 25ms Polars)

### 🐛 Fixed
- `not use_gpu_filter` gate bug in `custom_ops.py` section 4.6 that blocked GPU filter path
- Q7 nation pre-join cache key now uses `id()` preventing cache misses on repeated hot calls

### 🏗️ Infrastructure
- `pyproject.toml` updated with full metadata, dependencies, optional extras, package-data for `.so` files
- `CONTRIBUTING.md` — developer guide with kernel writing tutorial
- `PUBLISHING.md` — step-by-step pip release guide
- GitHub Actions: `test.yml` (CI), `publish.yml` (PyPI via OIDC)

### 📊 Benchmark Summary (1M and 10M rows, all 22 queries, warm median of 3)

See the full table in [README.md § TPC-H Benchmark](README.md#-tpc-h-benchmark--all-22-queries),
sourced from `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv`.

- **Correctness:** 22/22 queries pass on CPU and GPU paths
- **MX CPU wins vs Polars:** **21/22** at 1 M, **18/22** at 10 M
  — headline: **Q9 128× · Q12 89× · Q7 42× · Q8 31× · Q17 24× · Q5 22×** (at 10 M)
- **MX GPU wins vs Polars:** **16/22** at 1 M, **15/22** at 10 M
  — headline: **Q12 26.5× · Q9 12× · Q8 10.8× · Q17 9.7× · Q7 4.7×** (at 10 M)
- **Remaining losses (Q4, Q6, Q13, Q21):** ops that still route through PyArrow fallback — kernel replacements tracked in `roadmap.md`