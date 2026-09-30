# Plan

<!-- Tickets, one per vertical slice (each independently shippable and testable).
Status is one of: todo, doing, review, done, blocked. Update it as work moves.
A blocked ticket gets a "Blocked:" line saying exactly what the person has to do.
Format:

## F-001 Tenant signup [done]
Brief: docs/brief.md · Design: docs/design/signup.md · Needs: none
- [x] a new visitor can create a tenant with an email and password
-->

Brief: docs/brief.md · Spec: docs/spec-auto-mode.md · Branch: `auto-mode`

Shared rules for every ticket:
- Headline metric = **user-felt**: fresh process, Parquet on disk, clock around read + query + result, no warm-up.
- `device="cpu"` and `device="gpu"` results and APIs keep working; new behaviour sits behind `device="auto"`.
- Tests: `pixi run test` (pytest via an `mxframe`-named symlink, works in worktrees; new tests go in `tests/`) plus `pixi run test-all` (legacy scripts) must pass.
- TPC-H data (gitignored) lives at `/home/abhisheksreesaila/Projects/mxframe/data/tpch_sf{1,10}`; worktrees read it from there.
- Use glossary terms from the spec: `Morsel`, `DeviceCross`, `Router`, `ResidencyTracker`, `Calibration`.

## Wave 1: foundations (parallel)

## F-001 User-felt TPC-H scoreboard [todo]
Needs: none · Files: `scripts/bench_user_felt.py`, `scripts/tpch_user/` (new)
- [ ] each (engine, query, scale) runs in a fresh subprocess; the clock covers Parquet read + query + materialized result; `import` time is recorded separately
- [ ] engines: pandas, Polars (lazy `scan_parquet`), DuckDB (reference), MXFrame `cpu`, `gpu`, `auto`
- [ ] all 22 queries at SF1 and SF10; every engine's result is checked against DuckDB (row count + numeric tolerance) and mismatches are shown, not hidden
- [ ] a secondary "repeat run in same process" number is recorded
- [ ] mx engines read through `mx.scan_parquet` (moved from F-010)
- [ ] writes `docs/bench/user-felt-<date>.md` and a CSV; a `--queries`/`--engines`/`--sf` CLI; per-run timeout so one hang can't stall the run

## F-002 Multi-core CPU kernels with float64 accumulation [done]
Needs: none · Files: `kernels_aot/kernels_aot.mojo`, `aot_kernels.py`, rebuilt `libmxkernels_aot.so`
- [x] grouped sum/min/max/mean/count and masked global aggs use all cores (`parallelize` over row morsels with per-thread partials, then merge)
- [x] sums and means accumulate in float64 (SF10 Q1 matches DuckDB to 1e-9 relative)
- [x] join build/probe and gather use multiple cores where the kernel is embarrassingly parallel
- [x] a thread-count knob (env `MXFRAME_CPU_THREADS`, default all cores) that later tickets can use to split cores between roles
- [x] microbench before/after at 6M and 60M rows recorded in the commit message

## F-003 Calibration store [done]
Needs: none · Files: `calibration.py` (new), `tests/test_calibration.py`
- [x] `mx.calibrate()` (quick, < 30 s) measures H2D/D2H bandwidth (pageable + pinned where available), launch overhead, and `time = α + c·n` per kernel per device
- [x] saved to `~/.mxframe/calibration.json` (overridable via `MXFRAME_CALIBRATION`), keyed by device name + driver + MXFrame version, schema-versioned
- [x] missing / corrupt / stale file → safe built-in defaults, never a crash; no GPU → CPU-only calibration
- [x] `observe(op, device, rows, ms)` refines the fit online (HyPE-style)

## F-004 First-run cost: lazy GPU init, no JIT on hot paths [done]
Needs: F-002 merged (both touch `aot_kernels.py`) · Files: `custom_ops.py`, `aot_kernels.py`
Why: fresh-process SF10 Q1 profile (2026-09-29): `device="cpu"` pays 0.25 s creating the CUDA context it never uses; `device="gpu"` pays **5.3 s** MAX Graph JIT compile for `group_composite` (18 `session.load` sites remain in `custom_ops.py`).
- [x] `device="cpu"` never initialises the GPU; GPU context + AOT GPU library initialise lazily, and under `auto` speculatively in a background thread so a CPU-bound query never waits
- [x] every operator the 22 TPC-H queries hit runs through AOT kernels on both devices: no MAX Graph compile in a fresh-process run (assert via a test that counts `session.load` calls)
- [x] measured: fresh-process SF10 Q1 on `gpu` loses the 5 s compile; small `auto` query not worse than `cpu`

## F-010 `mx.scan_parquet` with pushdown and dictionary strings [done]
Needs: none · Files: `io_parquet.py` (new), `lazy_frame.py` (source-resolution pass), group-key encoding in `custom_ops.py`
Why: Polars `scan_parquet` reads only needed columns; mxframe reading every column loses on I/O alone. SF10 Q1 spends 1.8 s dictionary-encoding string group keys that Parquet already stores dictionary-encoded.
- [x] `mx.scan_parquet(path)` returns a lazy frame; at compute, only columns referenced anywhere in the plan are read, and simple comparisons directly above the scan are pushed into the reader (row-group skipping)
- [x] low-cardinality string columns are read as Arrow dictionaries; group-key encoding uses the dictionary indices directly (no re-encode), unifying dictionaries across chunks
- [x] results equal the `pq.read_table` path (20 queries on dictionary inputs; Q1/Q3/Q6/Q12/Q14 end to end through scan_parquet; Q15/Q21 blocked by F-011)

## Wave 2: safe auto (Phase 1)

## F-005 Auto planner: whole-query choice, fallback, explain [done]
Needs: F-003 · Files: `auto_mode.py` (new), `lazy_frame.py` (compute/explain hooks)
- [x] cost model walks the plan, estimates rows per operator, and compares CPU-only against GPU (incl. copies and GPU-init if not ready); picks the lower
- [x] GPU error/OOM → rerun on CPU, warn once per shape, remember the shape; `strict=True` raises
- [x] `lf.explain(device="auto", costs=True)` prints decision, estimates per device, bytes crossing
- [x] `compute(profile=True)` records actual vs predicted for the query (feeding per-operator times into `calibration.observe` moved to F-007, which executes per operator)
- [x] no GPU present → everything CPU, zero errors; `auto` results equal `cpu` results on the test suite
- [x] GPU paths known to return wrong results (`auto_mode.KNOWN_BROKEN_GPU`: sort) or with no usable kernel (min/max > 8192 groups) are never chosen

## Wave 3: per-operator placement + co-processing (Phases 2–3)

## F-006 Router co-processing for grouped and global aggregates [done]
Open for Abhishek: the two unticked items below are not achievable on PCIe x8 for host data (measured); keep them as future-hardware goals or drop them.
Needs: F-002, F-003, F-005 · Files: `coproc.py` (new), hook in `custom_ops.py`
Done so far: `auto_mode.Placer` (cpu/gpu/split cost decision), `coproc.run_router` (morsel queue, prepare/compute overlap, fallible GPU worker), `gpu_stream.DeviceStager` (pinned slots, async H2D), `residency.ResidencyTracker` (LRU GPU copies, pins).
Physics on the reference box (measured 2026-09-29): GPU link is PCIe 4.0 **x8** — pinned H2D 13.4 GB/s, pageable 11 GB/s, stager ~9 GB/s; host DRAM (Ryzen 5900X, dual-channel DDR4) ~40–50 GB/s. So a scan-aggregate over host-resident data is ~4× faster on all CPU cores than shipping it to the GPU, and splitting a pure scan gains little (both devices read the same DRAM). The GPU earns its keep on (a) joins / high-cardinality group-bys (CPU latency-bound), (b) repeat queries over GPU-resident columns, (c) idle-link background upload of hot columns during CPU-bound queries so the next query finds them resident.
Design as built (measured 2026-09-30, SF10 Q1 fused stage, 60M rows): CPU 57 ms, GPU from host 353 ms, GPU resident 39 ms, CPU 40% + GPU 60% resident 25 ms; a Router streaming host data to the GPU was 67–160 ms (worse than CPU), and a dynamic morsel queue on resident data 42 ms (each GPU call has ~4.5 ms fixed cost). So `coproc_fused`: host data → CPU + background upload of the query's columns and group ids; resident data → static split, one GPU call + multi-core CPU, fraction balanced from calibration and refined by `observe`. Break-even (fresh process, public API, SF10 Q1, re-measured after moving all CUDA calls off the query thread): run 0 is ~1.0–1.2 s with or without co-processing (cold costs unrelated to it, see F-015); runs 1–2 pay the one-time GPU warm-up of the split (57–107 ms vs 51–96 ms); from run 3 on the split is 38–45 ms vs 51–60 ms CPU-only.
- [x] decomposable aggregations run on both devices at once; partial aggregates merged (Router, static split)
- [x] split ratio from calibration (refined online); `gpu_fraction=` overrides; results identical to `cpu` for 1, 2 and all threads (tests)
- [x] GPU failures inside a split are redone on CPU; CUDA errors disable the GPU; residency never serves a stale or freed device copy (pinned entries, owners held, zero-copy columns only)
- [x] `profile=True` / explain show both devices' busy time
- [ ] streaming host data to the GPU with copy/compute overlap beats CPU-only on this box — not achievable on PCIe x8 (measured above); revisit on PCIe 5 / Grace Hopper hardware
- [ ] cold (first-query) SF10 Q1 under `auto` beats `cpu` by ≥ 15% — not on this box for host data (see break-even); met for repeat queries

## F-007 Per-operator placement for joins (CPU filter → GPU join) [todo]
Needs: F-005 · Files: `auto_mode.py` placement pass, join dispatch in `custom_ops.py`
- [ ] placement pass assigns each operator a device; `DeviceCross` nodes shown in explain with bytes
- [ ] filters/projections before a crossing so only surviving rows and needed columns move
- [ ] join probe side can be split across CPU and GPU (Router) when both are worth it
- [ ] per-operator timings feed `calibration.observe` (moved from F-005)

## F-008 Fused pipeline kernel: vectorized interpreter [done]
Needs: F-002 · Files: `kernels_aot/pipeline*.mojo` (new), `pipeline_program.py` (new: Expr → bytecode), dispatch in `custom_ops.py`
Why: SF10 Q1 spends ~1.3 s in PyArrow expression eval, NumPy mask indexing and dtype copies around a ~0.5 s kernel; JIT-compiling a kernel per query shape would cost seconds on a first run.
Design (MonetDB/X100-style vectorized interpretation, AOT, no JIT):
- Python compiles the query's filter predicate, the aggregate input expressions and the aggregate list into a small bytecode program (LOAD_COL i, CONST k, ADD/SUB/MUL/DIV, CMP_{LT,LE,GT,GE,EQ,NE}, AND/OR/NOT, BETWEEN, IN_SET (dictionary codes), plus AGG_{SUM,MIN,MAX,COUNT} slots). Registers are vectors.
- CPU kernel: `parallelize` over morsels; within a morsel process vectors of ~1–2K rows: evaluate the predicate into a selection vector, evaluate expressions into cache-resident register vectors (SIMD), accumulate into per-thread float64 partials per group; merge partials at the end. Input columns are raw pointers (int32/int64/float32/float64 + int32 group ids, or none for global aggs).
- GPU kernel: same bytecode, one thread per row (bytecode is uniform across threads, so no divergence), block-level partials (shared memory for small group counts, global atomics otherwise).
- Both entry points take a row range [start, end), so the F-006 Router can hand morsels to either device, and return partials the Router merges.
- [x] Q1, Q6, Q14, Q19-style filters + arithmetic + grouped/global aggregates run through it on both devices; results match DuckDB (float64 accumulation)
- [x] fused stage on SF10 Q1 (60M rows, group ids ready) < 100 ms on `cpu`; cold end-to-end time is tracked in F-009
- [x] unsupported expressions fall back to the existing path, never wrong results

## F-011 Correctness: counts, Q15, Q21, nulls [todo]
Needs: F-002, F-008 merged (they own the count kernels) · Found by F-010
- [ ] grouped `count` returns int64, exact beyond 2^24 (SF10 Q1 `count_order` saturates at 16,777,216 today: float32)
- [ ] Q15 runs (fails today: NoneType float error) and Q21 runs (fails: join key types int64 vs int32; cast keys to a common type)
- [ ] nulls in filtered/grouped columns follow SQL semantics (filter drops null comparisons, null group key is its own group) or raise clearly
- [ ] float32 columns compared with float64 literals keep boundary rows (Q6 `l_discount <= 0.07` drops exact-grid rows today): compare in the column's precision, as DuckDB does on DECIMAL
- [ ] case_when derived columns on the CPU group path are float64
- [ ] a Limit below an Aggregate is applied (ignored today on both devices)
- [ ] multi-key `sort("k","j")` on GPU returns the right order (wrong today)
- [ ] `filter(k != "a")` with null keys doesn't raise
- [ ] `_JOIN_RESULT_CACHE` keyed by device (a GPU-computed join is reused by later CPU queries today)
- [x] Q6 matches DuckDB on cpu/gpu/auto (fused CMPC compares in the column type); "global_agg" removed from `KNOWN_BROKEN_GPU`
- [ ] GPU Q3 top-10 equals CPU; Q21 on `gpu` doesn't raise IndexError
- [ ] benchmark queries Q4, Q5, Q7, Q10, Q20 filter real Parquet dates correctly (they compare yyyymmdd ints against date columns and return 0 rows today)
- [ ] GPU min over a group whose only values are +inf returns +inf (identity is Float64.MAX today)
- [ ] a join whose output has a duplicate column name (same non-key name on both sides) can be joined again / queried: rename with a suffix or raise clearly (today: pyarrow KeyError 'Field exists 2 times')
- [ ] all 22 queries match DuckDB at SF1 and SF10 on `cpu`, `gpu`, `auto`

## F-012 Kernel health: crashes, hangs, per-call overhead [done]
Needs: F-002 merged · Found by F-003 calibration (2026-09-30)
- [x] GPU `group_min`/`group_max` no longer fault (`CUDA_ERROR_ILLEGAL_ADDRESS`) at >= 16k groups; test at 16k, 1.5M
- [x] `AOTKernelsGPU.group_encode_i32` terminates and is correct (hangs today for every input tried)
- [x] CPU `sort_indices` is O(n log n) (in-place rotate merge sort is O(n^2), hangs at 1e4 random rows); parallel radix/merge sort
- [x] GPU per-row cost measured with device-resident input is within 2x of memory bandwidth (calibrated today at 2–4 ms per million rows: slower than single-threaded CPU). Find where the time goes: per-call `DeviceContext()` creation, per-call upload/alloc in the Python wrappers, launch config
- [x] GPU wrappers accept device pointers (residency / Router) as well as NumPy arrays

## F-013 Calibration and masked-kernel refresh after F-012 [done]
Needs: F-012 merged · Found by re-running calibration (2026-09-30)
- [x] GPU masked global aggregates (`masked_global_*_gpu`) at memory speed: 73 ms per 30M rows today vs 7 ms on CPU (same seq_cst/CAS atomics issue F-012 fixed for group ops?)
- [x] CPU `sort_topk` probe times the engine's new radix sort, not PyArrow (`sort_topk@cpu` estimates 2.2 s at 30M)
- [x] a `pipeline` probe on both devices (fused kernel, Q1-shaped program, resident input on GPU) so co-processing starts from measured fits, not defaults
- [x] `join_total@cpu` is not a separate fit (planner prices CPU joins with `join`)

## F-014 Engine uses the fast kernels it has [todo]
Needs: F-011, F-013 merged · Found by F-013
- [ ] CPU sort in `custom_ops._apply_sort_custom` uses `AOTKernels.sort_indices` (parallel radix, ~69 ms / 10M) instead of `np.argsort` / `argpartition`; `compiler.py` post-op sort likewise
- [ ] `test_gpu_kernels.py::test_sort_end_to_end_single_and_multi_key_with_and_without_limit` is flaky on a shared GPU (fails ~1 in 3 even alone): find out whether it is memory pressure (cuMemAlloc 2) or a real ordering bug
- [ ] fused GPU pipeline kernel reaches ≥ 50% of memory bandwidth on the Q1 shape (77 GB/s today, 7.8 ms per 30M rows resident)

## F-015 Cold first query: 1.1 s vs 50 ms warm [done]
Needs: none · Found 2026-09-30: fresh process, SF10 lineitem already in memory, first `compute(device="auto")` of Q1 takes 1.0–1.2 s; the same query right after takes ~50 ms. The headline metric is the first run, so this is the biggest remaining gap.
- [x] profile the first query in a fresh process; list where the ~1 s goes. Fresh process, SF10 lineitem in memory (`dictionary="auto"`), first Q1-shaped `compute(device="cpu")`, ms:

  | cost item | before | after |
  |---|---|---|
  | `encode_key` x2: `pc.unique` unused-entry check | 300 | 0 (only present combinations become groups) |
  | `encode_key` x2: `combine_dictionary` (unify ~60-458 chunk dictionaries incl. remapping the rows) | 176 | ~1 (unify the tiny dictionaries only; skipped when shared) |
  | codes int8/int32 -> int32 `astype` | 119 | 0 (codes read in place, any width) |
  | Python/NumPy glue | 157 | ~5 |
  | `group_composite` | 99 | inside the pass below |
  | group ids in total | 851 | ~95 = page pre-fault of the 240 MB ids ~30 + composite/presence pass ~38 + dense remap ~18 |
  | `import max.engine` / `max.graph` (lazy now, never imported on CPU paths) | ~153 | 0 |
  | fused pipeline kernel | 44 | ~48 (same as warm) |
  | `dlopen` of the three kernel libraries (Mojo runtime load, first time) | ~30 | ~30 |
  | rest (plan, bind_chunks, calibration, ...) | ~20 | ~20 |
  | **total** | **1075** | **~170** |

  Finding: faulting the fresh 240 MB ids array from 24 Mojo threads inside the hot pass cost 190 ms vs 36 ms on touched pages; a separate parallel zero-fill first (~30 ms) fixes it.
- [x] first-query SF10 Q1 (table in memory) under 200 ms on `cpu` and `auto`, results unchanged. After the orchestrator's auto changes (c32636f), 7 fresh-process runs each, machine load 4-7: cpu 172 179 194 173 175 179 168 (median 175 ms); auto 229 194 181 176 179 179 173 (median 179 ms). Before: cpu 984-1116, auto 1082-1205. Q1 result identical to the old path (and to DuckDB on SF1 in `tests/test_group_keys_review.py`).
- [ ] follow-up (not done): a uint8 group-id variant in the fused kernel (<= 256 groups) would cut the 240 MB ids array to 60 MB and remove most of the ~95 ms.

## F-016 Engine correctness gaps outside TPC-H [todo]
Needs: F-011 merged · Found by the F-011 review (2026-09-30), all pre-existing
- [ ] `with_columns` / `select` after an aggregate is applied (today silently ignored: `agg(count n).with_columns((n/2).alias("n"))` returns n unchanged)
- [ ] a two-level groupby (aggregate over an aggregate) works (KeyError today)
- [ ] a global sum/min/max/mean over zero rows returns NULL as SQL does (0.0 today); count stays 0
- [ ] `count(col)` of a right-side column after a left join counts unmatched rows as 0 (counts them as 1 today: wrong answer)
- [ ] a groupby on a column produced by an earlier aggregate in the same plan works ("Field does not exist" today)
- [ ] a fuzz test of random plan shapes (project/filter/agg/join/sort/limit nested 2–3 deep) against DuckDB on cpu and auto

## F-017 Parquet read as fast as Polars [built, needs review; 1.5x target NOT met, see profile]
Needs: none · Found by the F-001 scoreboard (2026-09-30): SF10 `scan_parquet` of 3 lineitem columns with a date filter takes 689 ms in mx vs 149 ms in Polars.
- [~] profile the mx read path and close the gap to within 1.5x of Polars on SF1/SF10 lineitem scans with and without filters. Profiled; gap narrowed for selective filters, not closed (pyarrow's decoder is ~3x slower than Polars' own).
- [x] filters pushed to the reader drop rows while decoding ((row group, column) pieces on a thread pool; integer conjuncts; row groups fully covered by min/max skip the mask; unselective filters and files under 16M rows use the plain reader). Page-index skipping: not possible, pyarrow 22 has no row-range/page-skipping read API (`ParquetFile` has only read/read_row_group(s)/iter_batches).
- [x] results unchanged (tests/test_scan_parquet.py, test_fast_read.py, test_np_arrow.py, new tests/test_parquet_rowfilter.py; full suite green)

Profile (fresh process, SF10 exact lineitem, 60M rows / 58 row groups of ~1M rows, snappy; machine shared and loaded, medians of 7-9 interleaved runs):
- `l_shipdate` min/max is the same (8036..10561) in every row group, so statistics prune nothing for date ranges; the file is sorted by orderkey only. Only the mask-skipping "fully covered" case benefits from stats.
- Old path for `scan -> filter(shipdate range) -> select 3 cols` (~540-760 ms depending on load): ~40 ms `scan_parquet` (pq import + footer parsed, then parsed AGAIN at compute, +20-30 ms), ~225 ms `ParquetFile.read` of all 60M rows x 3 cols, ~300 ms engine filter+select over 60M rows. Dictionary-decision loop is 0.02 ms (not a cost); pre_buffer=True was slower (270 vs 230 ms).
- Raw decode floor: `pq.ParquetFile.read` of the 3 columns = ~225 ms (Polars does the whole scan+filter+select in ~77 ms, its no-filter 3-column read is ~165 ms). Arrow's decoder scales badly with threads on this box (user CPU x4 at 8 threads, x12 at 24: allocator/page-fault contention; glibc/jemalloc/mimalloc/system pool and mmap thresholds made no real difference); 6-8 decode threads is the sweet spot.
- Polars prefilters (decodes the filter column, then only materializes matching rows); pyarrow can only decode whole pages, so late materialization of the other columns is impossible here. The gain available to us is not materializing 60M rows for the engine: filter per (row group, column) piece right after decoding.
- Small things fixed: footer parsed once per process (`scan_parquet` primes the `_metadata` cache), `concurrent.futures` (10-40 ms to import) replaced by plain threads, `pyarrow.compute` not imported by the reader (NumPy gather), one chunk per plain column (like ParquetFile.read) so the engine does not re-concatenate.

Before/after, mx-auto end to end via the F-001 worker (fresh process, scan_parquet, median ms; base = 7367b47):
| query | base | F-017 | Polars |
|---|---|---|---|
| SF10 Q1 | 797 | 788 | 2771 |
| SF10 Q6 | 401 | 405 | 242 |
| SF10 Q14 | 1667 | 1659 | 235 |
| SF10 Q15 | 1124 | 460 | 160 |
| SF10 scan 3 cols + date filter (read+select only) | 763 | 452 | 82 |
| SF10 scan 3 cols, no filter | 354 | 300 | 170 |
SF1 is unchanged by design (6 row groups, 6M rows: the plain reader is already ~25 ms).

Open decision for the owner: closing the rest of the gap needs a different decoder. Polars' collect+to_arrow for the SF10 Q6 columns is ~140 ms after a ~77 ms `import polars`, vs ~360 ms for pyarrow. Options: (a) optional Polars-backed reader for numeric columns when polars is installed, (b) a native (Mojo/C++) Parquet page decoder in mxframe, (c) accept ~3x on raw Parquet scans and win elsewhere. Not attempted: it adds a dependency and a second type-mapping path.

## F-018 Join engine: order, lean materialization [todo]
Needs: F-011 merged · Found by the F-001 scoreboard: SF10 Q2 15.8x, Q9 9.4x, Q20 7.3x slower than Polars — joins dominate
- [ ] join reordering by estimated size/selectivity (filtered small sides first; e.g. Q2 joins 8M partsupp rows to supplier before the tiny filtered part table)
- [ ] late materialization: joins produce row-index vectors and gather only the columns later operators need, once, instead of materializing every column after every join (Q9: six chained joins over 60M rows)
- [ ] projection pushdown: only columns referenced above a join are carried through it
- [ ] per-operator GPU placement for large joins (F-007) on top of the lean join representation
- [ ] SF10 Q2/Q9/Q20 within 2x of Polars, results identical (DuckDB gate)

## F-019 Import time [todo]
Needs: none · Scoreboard: `import mxframe` ~700 ms in a fresh process (Polars ~70 ms, DuckDB ~40 ms)
- [ ] profile `python -X importtime -c "import mxframe"`; defer everything not needed to build a plan (pyarrow.parquet/csv, compute, numpy submodules, calibration, np_arrow) until first use
- [ ] `import mxframe` < 150 ms; the background warm-up still overlaps the rest

## F-009 Scoreboard report and README [todo]
Needs: F-001, F-005..F-008
- [ ] `auto` is within 10% of min(cpu, gpu) on ≥ 20/22 queries at SF1 and SF10 (moved from F-005)
- [ ] full SF1 + SF10 user-felt run on an otherwise idle machine; results + honest losses in `docs/benchmarks.md`
- [ ] README `device="auto"` section with the real numbers
