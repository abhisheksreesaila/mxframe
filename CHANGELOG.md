# 📋 Changelog

All notable changes to MXFrame are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.5.0] — 2026-07-11

Full narrative: [v0.5.0 release notes](RELEASE_NOTES_0.5.0.md).

### ⚡ Native GPU UTF-8 predicates
- Added AOT `utf8_startswith_mask_gpu`, `utf8_contains_mask_gpu`, `utf8_equal_mask_gpu`, and packed-literal `utf8_isin_mask_gpu` kernels over Arrow validity, offsets, and byte buffers.
- Supports sliced arrays, nulls, empty strings/patterns, duplicate literals, and multibyte UTF-8 while matching Arrow filter semantics.
- GPU expression dispatch covers string `==`, `!=`, `startswith`, literal `contains`, and non-null literal `isin`; unsupported/null-membership cases retain Arrow fallback.
- Added permanent CPU/GPU Phase 6 tests and `pixi run bench-utf8` comparisons against PyArrow and Polars.
- At 1M rows, packed `isin` and `contains` beat local Arrow and Polars; exact equality remains faster in Polars' optimized CPU path, so no universal string-performance claim is made.

### ⚡ Weak-query GPU fusion
- Q6 dispatches `range5_sum_product_f32_gpu` instead of materializing five masks and a product array.
- Q4 uses `semi_range_group_count_i32_gpu` for fused qualifying-key marking, semi-join probing, and grouped counts.
- Q13 uses `key_match_counts_i32_gpu` to emit one compact count per customer without materializing a left join.
- Q21 uses `single_late_supplier_counts_i32_gpu` for exact `(order, supplier)` deduplication and deterministic compact supplier counts.
- Representative 1M/10M GPU results versus Polars: Q4 **3.4/28.6 vs 20.2/105.4 ms**; Q13 **1.8/28.5 vs 26.5/442.0 ms**; Q21 **6.6/49.8 vs 30.2/99.1 ms**.

### 📊 Bounded 22-query benchmark
- Every query and engine now runs in its own process, releasing Arrow, Pandas, and CUDA allocations between measurements.
- Added per-engine timeouts, five-query progress groups, merged CSV output, and optional synchronized `cudf.pandas` workers.
- MX CPU beat Polars on **22/22 at 1M** and **18/22 at 10M**.
- MX GPU beat Polars on **15/17 comparable paths at 1M** and **14/17 at 10M**.
- Q8 GPU remains `N/A:JIT`; Q15/Q17/Q18/Q20 GPU workers exceeded the bounded timeout.
- cuDF was unavailable locally, so RAPIDS remains `N/A` and no RAPIDS parity claim is made.

### 🎬 Visualization and identity
- Q4, Q6, Q12, Q13, Q14, and Q21 visualize their logical rewrite, dispatch boundary, and step-by-step Mojo kernel algorithm.
- Added a cool-gray MX mark whose rising X stroke is shaped as orange lightning.
- Verified responsive rendering and zero horizontal overflow in Playwright.

### Remaining performance-parity work
- Replace Q8's JIT path and fix Q15/Q17/Q18/Q20 GPU timeouts.
- Improve Q1 and Q14 GPU pipelines at 10M.
- Run synchronized 1M/10M benchmarks on a compatible RAPIDS/cuDF environment.

### Remaining feature-parity work
- Native GPU UTF-8 dictionary construction and string/date gather.
- GPU windows and additional semi/anti/cross/as-of/range joins.
- Broader regex/string, datetime, nested-type, reshape, SQL, streaming, and I/O pushdown support.
- Cross-generation NVIDIA testing plus AMD and Apple Silicon validation.

---

## [0.4.0] — 2026-07-11

### 🔗 String and composite joins
- Join keys from both tables now use one shared Arrow dictionary before dispatch, so equal strings receive identical dense `int32` IDs regardless of encounter order.
- Multi-column joins jointly densify shared per-column codes, avoiding mixed-radix overflow and supporting mixed string/integer composite keys.
- SQL null semantics are preserved: null keys receive side-specific IDs and never match, including left joins.
- Compact nonnegative integer joins retain their direct `int32` fast path; sparse and non-integer keys are safely densified.
- Fixed GPU upload-cache aliasing by retaining each cached NumPy host owner, preventing recycled host pointers from returning stale device buffers during joined-table gather.
- String encoding runs in Arrow on CPU; resulting IDs use the Mojo AOT CPU/GPU join kernels. Native GPU UTF-8 dictionary construction remains future work.

---

## [0.3.0] — 2026-07-10

### 📚 Docs & Benchmarks
- README restructured: 543 lines → 119 lines. Intro, quick start, and 4-row teaser only — all detail moved to dedicated docs.
- New `docs/benchmarks.md`: full 22-query TPC-H tables (1 M + 10 M rows), Mojo kernel catalogue, current limitations, roadmap, and reproduce instructions.
- New `docs/api.md`: LazyFrame method reference, expression syntax, SQL frontend, supported operations, project structure.
- `docs/vision-and-architecture.md`: fixed garbled sections 2.5/2.6/3; success criteria table repaired; all milestones marked ✅.
- `docs/archive/` added to `.gitignore` — old dev notes moved there, not published.
- All stale `Mojo 0.26.2` / `MAX Graph shape-cached join` references updated to reflect v0.2.3 reality (Mojo 26.4, AOT ctypes everywhere).

### 📊 Benchmark updates
- Q5/Q7/Q8/Q10/Q13/Q19 re-measured with per-run join-cache clearing (fresh-join). Previous v0.2.0 numbers for those queries were warm join-cache hits.
- Summary corrected: CPU 20/22 (Q13 loses fresh-join), GPU 12/22 (Q5/Q7 no longer win at 1 M without cache warmup).
- Q8 GPU marked `—` (MAX Graph `year()` groupby JIT too slow to benchmark; Phase 6 target).

### 🔨 Infrastructure
- `modular >= 26.4` requirement added to `pyproject.toml` runtime deps and Dependencies table in README.
- `docs/archive/` gitignored; previously committed archive files removed from history.

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
- **`kernels/_atomic.mojo`** shim introduced to replace the removed `std.os.atomic` module.
- All kernel sources migrated to 26.4 syntax; compiled artefacts rebuilt.
- `pyproject.toml` `runtime` extra pinned to `modular>=26.4` to prevent 26.2 wheels from being installed by Colab/pip users.

---

## [0.2.1] — 2026-06-xx

### 🛠 Fixed
- TestPyPI re-upload to validate wheel packaging. No functional changes.

---

## [0.2.0] — 2026-05-xx

### ✨ Added
- **Phase 1 — GPU buffer cache + masked global min/max**:
  - `masked_global_min_f32_gpu` / `masked_global_max_f32_gpu` Mojo AOT kernels.
  - `AOTKernelsGPU` device-buffer cache (`_cached_upload`, eviction, `clear_buf_cache`) — eliminates redundant PCIe uploads on repeated hot queries.
  - `group_mean_f32` GPU kernel (sum + count on GPU, divide on CPU).
- **Phase 2 — Masked global aggregation for any expression**:
  - `_compute_masked_global_agg` extended to handle any inner expression via `_eval_expr_arrow` (e.g. `a*(1-b)`, `case_when`, `startswith`, `isin`) — SUM reduction stays GPU-resident.
  - Join-only plans emit `"join_mojo_shortcut"` provenance (not `"pyarrow_shortcut"`) — Q19 restructured to single-pass join+filter+global_agg.
- **Phase 3 — GPU integer key encoding**:
  - `group_encode_i32_gpu` open-addressing hash-table kernel for integer group keys; replaces PyArrow `dictionary_encode` for integer-keyed group-by on GPU.
- **Audit tool** (`scripts/audit_gpu_paths.py`): classifies each of 22 TPC-H queries as `GPU-CLEAN` / `MOJO-CPU` / `FALLBACK` via `last_compile_provenance["path"]`.

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
- Packaging: `readme` now points to `README.md` so PyPI shows the full project page with benchmarks and quickstart (was accidentally showing the internal vision/architecture doc).
- CI: `AOTKernelsGPU` init no longer hard-fails on CPU-only runners (`CUDA driver unavailable` is caught, `self._aot_gpu` falls back to `None`).
- CPU-only import and compute path validated in a clean venv without `modular`.

---

## [0.1.0] — 2026-04-21

Initial public release.