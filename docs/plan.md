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

## Core gate (agreed 2026-10-15): nothing new until cpu, gpu and auto all hold up

Abhishek, 2026-10-15: "the day we say GPU, CPU and auto work really well and 90% if not all of them are beating DuckDB, the core is ready; then the rings (vector, vision, Mojo API) stand on top of it."
Order, one stage at a time:
1. **CPU leftovers:** F-042, F-045, F-037, F-038, F-036, F-046a (partitioned hash join), F-033, F-034, then F-043's last check and F-025.
2. **GPU correctness:** F-055 (results only, no speed claims yet).
3. **Auto works:** F-053 (auto uses the native executor), then F-056 (data bigger than GPU memory never fails: spill or split to CPU, RAPIDS / BlazingSQL style), F-057 (morsel co-processing that wins), F-007 (per-operator placement for joins).
4. **Gate:** F-058, the scoreboard that says the core is ready.
Benchmarks run only on a quiet machine (see `docs/lessons.md`). With other agents running, do development and light SF1 correctness checks only.

## F-055 GPU correctness sweep [done]
Needs: none · Agreed 2026-10-15 · Report: `docs/bench/gpu-correctness-sf1.md`
- [x] all 22 TPC-H queries (SF1, stored and shuffled) on `device="gpu"` equal DuckDB (22/22 both; first and repeat run); every mismatch or crash gets a fix or a ticket: none found. Also on the GPU, per query: the path of every `compute()` and the GPU kernels it called (all 22 ran at least one GPU kernel; the join in a GPU session is the GPU hash join)
- [x] the random-query suite (`scripts/bench_random/`) runs on `device="gpu"` and checks every result against DuckDB (`--device gpu --gpu <uuid> --engines mx,duckdb`): 100/100 stored and shuffled, 94 ran a GPU kernel
- [x] F-014's flaky GPU sort test settled: memory pressure from other processes (about 0.5 GB free reproduces it), and behind it a real bug: every exported GPU function swallowed its errors (`except: pass`), so an out-of-memory sort or join returned garbage. All 31 now return a status and raise (`MemoryError` / `RuntimeError`), driver calls check their `CUresult`, `auto` reruns on the CPU (`tests/test_gpu_kernel_errors.py`); `is_sorted_topk` stays as a backstop
- [x] a GPU path that can't be made correct is listed in `auto_mode.KNOWN_BROKEN_GPU`: none needed, it stays empty
- [ ] open: statuses cover out-of-memory only (a real OOM sort and an oversized request are tested). A real fault such as `CUDA_ERROR_ILLEGAL_ADDRESS` kills the process (core dump) and no status comes back; the Mojo `ctx.synchronize()` of the other kernels is assumed to raise on OOM like the sort's does (not probed)

## F-056 Data bigger than GPU memory never fails [doing]
Needs: F-055 · Spec: `docs/spec-auto-mode.md` Phase 4 · Agreed 2026-10-15 ("96 GB on a 48 GB machine must not just say out of memory; trust is lost")
- [x] research first, written to `docs/research/gpu-out-of-core.md` and `docs/research/heavydb.md` (HeavyDB is the closest open-source GPU-native competitor; mine it for algorithms): how RAPIDS (cuDF spilling, dask-cudf, the cuDF-Polars streaming executor), BlazingSQL and HeavyDB handle a working set larger than the GPU, and which of their algorithms fit our morsel design
- [ ] `device="gpu"` and `"auto"` on a working set larger than free GPU memory finish with results equal to `cpu` (test with an artificially small GPU memory budget, so no 96 GB data is needed)
- [ ] the plan chooses up front (estimated size vs free memory): GPU, GPU in partitions, or CPU; a GPU OOM mid-query reruns the failed part on CPU and is remembered
- [ ] both GPUs on this box (2x RTX 3090, 24 GB each) are counted as separate memory pools, not one 48 GB pool
- [x] Step 1, never fail (`gpu_budget.py`; tests `tests/test_gpu_budget.py`, real-GPU ones opt-in with `MXFRAME_TEST_GPU_BUDGET=1`): per-GPU budget (driver free memory, capped by the test override `MXFRAME_GPU_MEM_BUDGET`; the process runs on one GPU, a second GPU is a second pool/process later); each GPU piece (fused aggregate, join pairs, sort / top-k, gather) is pre-checked at 90% of free memory -> CPU; an OOM halves the aggregate's chunk and retries once, then reruns the piece on the CPU and remembers the shape (piece level, and the query shape for auto); `device="gpu"` falls back with a warning, `strict=True` raises `MemoryError` with the estimate; provenance `gpu_pieces`, `explain` and the auto decision say when and why. Aggregate, join and sort/top-k over a fake 1 MB GPU equal DuckDB on `gpu` and `auto`
- Review fixes (step 1): a remembered shape is tried on the GPU again once 1.5x more memory is free (piece level and auto's `_failed_shapes`; non-OOM errors stay remembered), keyed by column names and dtypes; `forget_gpu_failures()` also forgets pieces; also guarded: GPU string filters, masked global sum/min/max/sum-product, range5 sum-product, group encode (the GPU left join already fell back to the CPU on any error; coproc and residency uploads are auto-only and covered); a join output past 2^31 rows is a distinct `JoinOutputTooLarge` (a `MemoryError` subclass, not remembered, not reported as OOM); the CPU join has the same limit, so it raises on every device
- Open after step 1: joins and sorts go straight to the CPU (no halving: a join could split its probe side, a sort could merge sorted runs); estimates are lower bounds for a join's output size (one match per left row; a bigger output is caught by the OOM path); the MAX-graph (JIT) paths are not guarded (they run only without the AOT libraries); a fake budget does not count memory already resident from earlier queries; nothing yet uses the second GPU
- Build order from the research (2026-10-15): (1) never fail: per-GPU budget with a test-only override, pre-check at 90% of free memory -> CPU, OOM -> halve the piece and retry once, then rerun the piece on CPU and remember the shape; pieces never change their inputs; (2) streaming probe through a resident hash table, for joins and many-group aggregates only, after a CPU filter + bitvector; (3) partition both sides so each pair fits one 24 GB GPU, the two GPUs as separate workers. Skip unified memory and generic host spilling. HeavyDB ideas: string filters run once over the dictionary (becomes an int IN-set), dense hash layouts sized from min/max with a group-count pre-pass
- [ ] the two GPUs are joined by an NVLink bridge (NV4 in `nvidia-smi topo -m`): measure GPU-to-GPU bandwidth and whether Mojo can use peer access before relying on it
- Note: 96 GB is also bigger than this box's 62 GB of RAM, so that case needs the CPU path to stream Parquet row groups and spill to disk too (a later ticket: the CPU rerun of a piece is exact but holds its inputs in RAM, so step 1 alone does not make 96 GB work here)
- Note from F-055: every GPU kernel returns a status and raises `MemoryError` on out-of-memory, so this ticket can rely on that for OOM only; a real GPU fault (illegal address) kills the process and returns no status

## F-057 Morsel co-processing that wins [todo]
Needs: F-053, F-055 · Builds on F-006 (`coproc.py`, `coproc_fused.py`, the Router) · Agreed 2026-10-15
- [ ] research first (added to `docs/spec-auto-mode.md`): morsel-driven parallelism (Leis 2014), HetExchange, "Pump Up the Volume" and later CPU+GPU co-processing work; what F-006 measured (PCIe x8: streaming host data loses to CPU, resident data splits win) decides which ideas are worth it on this hardware
- Research (2026-10-15, `docs/research/cpu-gpu-coprocessing.md`): keep F-006's design (host data -> CPU, resident data -> static split, one GPU call; Mordred 2022); split only if predicted time beats the faster device by 10% after per-call (4.5 ms), merge and contention costs; CPU threads take late GPU work; the second GPU as a third worker with its own resident share
- Check first (review of F-053): a `read_parquet` frame may never find its data resident: sources are re-read on each compute and residency is keyed by buffer addresses. Verify a repeat query finds its columns resident before relying on resident-data splits.
- [ ] the native CPU executor takes the CPU share of a split (today a split runs the older fused pipeline)
- [ ] a split is chosen only where calibration predicts it beats both single devices, and `explain` says why
- [ ] measured on a quiet machine: auto within 10% of min(cpu, gpu) on every query, faster than both on the shapes the split targets (high-volume aggregation on resident data)

- [ ] `auto_mode._unsplittable` rules out joins and unstable keys only; other shapes the fused pipeline declines (Project/Distinct between aggregate and scan, nulls, int64 past 2^53, dictionary-code filters) can still be predicted a split, then native declines every repeat: make the split predicate match the split path exactly (from the F-053 review)
## F-058 Core gate scoreboard [todo]
Needs: F-055, F-056, F-057 · Quiet machine only
- [ ] engine-only TPC-H SF1 + SF10 (stored and shuffled) and the random suite, for `cpu`, `gpu` (warm, labelled) and `auto`, against DuckDB and Polars, every result checked against DuckDB
- [ ] each device beats DuckDB on at least 90% of queries; the misses listed with a reason (close misses are fine for the gate; the long tail is chased later, after the rings start, per Abhishek 2026-10-15)
- [ ] when it passes: the rings below can start

## Rings (after the core gate, not before)
Proposed 2026-10-15, not yet briefed; each goes through `clarify` when its turn comes.
- Mojo front end for MXFrame: a Mojo plan builder over `mx_execute`, so MX App / DataLens can use MXFrame where there is no Python (iPhone). Depends on F-033, F-034, F-021 (every query native, native Parquet).
- Tables loaded once per process, shared read-only across sessions (MX App's `App.init_shared`, fh-saas dashboards).
- Embedding columns and filtered top-k vector search (CPU, then GPU), built on the `mx_filter_topk` pattern.
- Vision kernels, LLM operators, an MCP server: parked.

## Wave 1: foundations (parallel)

## F-001 User-felt TPC-H scoreboard [done]
Needs: none · Files: `scripts/bench_user_felt.py`, `scripts/tpch_user/` (new)
- [x] each (engine, query, scale) runs in a fresh subprocess; the clock covers Parquet read + query + materialized result; `import` time is recorded separately
- [x] engines: pandas, Polars (lazy `scan_parquet`), DuckDB (reference), MXFrame `cpu`, `gpu`, `auto`
- [x] all 22 queries at SF1 and SF10 (mx-gpu SF10 skipped this round: shared GPU memory, joins being rewritten in F-018); every engine's result is checked against DuckDB (row count + numeric tolerance) and mismatches are shown, not hidden
- [x] a secondary "repeat run in same process" number is recorded
- [x] mx engines read through `mx.scan_parquet` (moved from F-010)
- [x] writes `docs/bench/user-felt-<date>.md` and a CSV; a `--queries`/`--engines`/`--sf` CLI; per-run timeout so one hang can't stall the run

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
- [ ] per-operator GPU placement for large joins, on top of join_engine's index vectors (from F-018 / F-020)

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

## F-011 Correctness: counts, Q15, Q21, nulls [done]
Needs: F-002, F-008 merged (they own the count kernels) · Found by F-010
- [x] grouped `count` returns int64, exact beyond 2^24 (SF10 Q1 `count_order` saturates at 16,777,216 today: float32)
- [x] Q15 runs (fails today: NoneType float error) and Q21 runs (fails: join key types int64 vs int32; cast keys to a common type)
- [x] nulls in filtered/grouped columns follow SQL semantics (filter drops null comparisons, null group key is its own group) or raise clearly
- [x] float32 columns compared with float64 literals keep boundary rows (Q6 `l_discount <= 0.07` drops exact-grid rows today): compare in the column's precision, as DuckDB does on DECIMAL
- [x] case_when derived columns on the CPU group path are float64
- [x] a Limit below an Aggregate is applied (ignored today on both devices)
- [x] multi-key `sort("k","j")` on GPU returns the right order (wrong today)
- [x] `filter(k != "a")` with null keys doesn't raise
- [x] `_JOIN_RESULT_CACHE` keyed by device (a GPU-computed join is reused by later CPU queries today)
- [x] Q6 matches DuckDB on cpu/gpu/auto (fused CMPC compares in the column type); "global_agg" removed from `KNOWN_BROKEN_GPU`
- [x] GPU Q3 top-10 equals CPU; Q21 on `gpu` doesn't raise IndexError
- [x] benchmark queries Q4, Q5, Q7, Q10, Q20 filter real Parquet dates correctly (they compare yyyymmdd ints against date columns and return 0 rows today)
- [x] GPU min over a group whose only values are +inf returns +inf (identity is Float64.MAX today)
- [x] a join whose output has a duplicate column name (same non-key name on both sides) can be joined again / queried: rename with a suffix or raise clearly (today: pyarrow KeyError 'Field exists 2 times') -- done: the right column becomes `<name>_right` (`_right2`, ...)
- [x] all 22 queries match DuckDB at SF1 and SF10 on `cpu`, `gpu`, `auto` (132/132; SF10 gpu/auto ran on GPU 1 because another process held 17-21 GB of GPU 0)
- Build notes: `tests/test_tpch_duckdb.py` is the gate (`MXFRAME_TPCH_SF10=1` adds SF10). The DuckDB reference reads the same Parquet through views that restore DATE and DECIMAL(15,2), so it runs the official TPC-H text and gives the official answers (Q6 SF1 = 123141078.2283). The old Q6 "matches DuckDB" was DuckDB on raw float32 columns, which shares the truncation below.
- The data generator truncated DECIMAL to float32 (0.05 stored as 0.049999997): fixed in `scripts/gen_tpch_parquet.py`, regenerated as `data/tpch_sf1_exact` and `data/tpch_sf10_exact` (the older dirs are untouched; the harness prefers `_exact`). Other agents should read the `_exact` dirs.
- `run_q5_*`/`run_q8_*` (mxframe, pandas, Polars) take `supplier=` (keyword): TPC-H Q5 matches customer and supplier nation and Q8 measures the supplier's nation; the four/six-table forms could not.

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
- [x] CPU sort in `custom_ops._apply_sort_custom` uses `AOTKernels.sort_indices` (parallel radix, ~69 ms / 10M) instead of `np.argsort` / `argpartition`; `compiler.py` post-op sort likewise
- [ ] `tests/test_cpu_sort.py::test_sort_indices_is_stable_and_matches_numpy[1000000-True-300001-int64]` failed once in a loaded full-suite run (F-018 branch); not reproducible in 5,600 isolated/concurrent stress runs of the exact input with 1–64 threads. The engine now verifies radix results in O(n) and falls back to np.argsort with a warning. Suspect cross-test memory corruption (F-018 fixed an out-of-bounds join write); watch for the warning.
- [x] `test_gpu_kernels.py::test_sort_end_to_end_single_and_multi_key_with_and_without_limit` flaky on a shared GPU: settled in F-055: memory pressure from other processes (about 0.5 GB free reproduces it) and, behind it, a real bug: a sort kernel that runs out of memory returns a wrong order with no exception. The GPU top-k result is now verified and redone on the CPU (`sort_check.is_sorted_topk`); an honest `cuMemAlloc` failure remains under pressure (`docs/bench/gpu-correctness-sf1.md`)
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

## F-016 Engine correctness gaps outside TPC-H [done]
Needs: F-011 merged · Found by the F-011 review (2026-09-30), all pre-existing
- [x] `with_columns` / `select` after an aggregate is applied (`_materialize_barrier` in custom_ops; `with_columns` takes the aggregate's output names)
- [x] a two-level groupby (aggregate over an aggregate) works
- [x] a global sum/min/max/mean over zero rows returns NULL as SQL does; count stays 0 (fused, masked, Arrow paths)
- [x] `count(col)` of a right-side column after a left join counts unmatched rows as 0 (verified vs DuckDB, grouped and global, cpu and auto: tests/test_f016_plan_shapes.py)
- [x] a groupby on a column produced by an earlier aggregate in the same plan works
- [x] a fuzz test of random plan shapes against DuckDB on cpu and auto (tests/test_f016_fuzz.py; seed-reproducible; `-m slow` runs 300 seeds, `MXFRAME_FUZZ_SEEDS=n` more)
- [x] found by the fuzz and fixed: a Filter above a Project under an Aggregate (wrong/crash with nulls), HAVING on a global aggregate ignored, int column + int literal returned float32, float64 column + float literal was rounded to float32, int/int `/` was integer division, -0.0 and 0.0 sorted as different values
- [ ] follow-up (F-018/join owner): the CPU hash join segfaulted (`join_scatter`) on a many-to-many join of ~20k x 5k rows over 6 keys (16M+ output rows, then a second join); it should raise a MemoryError or handle it
- [ ] follow-up: `min`/`max`/`sum` of an integer column come back as float64 (DuckDB: integer); int keys vs float keys in a join are refused by design (`_common_key_type`), so joining on such a column raises TypeError where SQL casts

## F-017 Parquet read as fast as Polars [done]
Decided 2026-09-30 (Abhishek): (c) keep PyArrow for Parquet read/write for now; a native Mojo reader comes later (F-021). The 1.5x-of-Polars check is waived until then.
Decision needed (Abhishek): pyarrow's Parquet decoder is the floor (raw read of 3 SF10 lineitem columns ~225 ms vs Polars' whole scan+filter ~77 ms; pyarrow 22 has no page-skipping API). Options: (a) optional Polars-backed reader when polars is installed (soft dependency on a competitor; type-mapping care); (b) a native Mojo Parquet decoder in mxframe (large; could decode on the GPU like cuDF); (c) accept ~3x on raw scans. Until you choose: (c).
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


## F-018 Join engine: order, lean materialization [done]
Needs: F-011 merged · Found by the F-001 scoreboard: SF10 Q2 15.8x, Q9 9.4x, Q20 7.3x slower than Polars — joins dominate
- [x] join reordering by estimated size/selectivity (filtered small sides first; e.g. Q2 joins 8M partsupp rows to supplier before the tiny filtered part table)
- [x] late materialization: joins produce row-index vectors and gather only the columns later operators need, once, instead of materializing every column after every join (Q9: six chained joins over 60M rows)
- [x] projection pushdown: only columns referenced above a join are carried through it
- [ ] per-operator GPU placement for large joins — moved to F-007
- [ ] SF10 Q2/Q9/Q20 within 2x of Polars, results identical (DuckDB gate): Q9 met (1.9x, DuckDB gate green); Q2 (2.6x) and Q20 (3.6x) remain, the gap is Parquet decode, blocked by the F-017 decoder decision

## F-019 Import time [done]
Needs: none · Scoreboard reported ~700 ms under load; measured 172 ms idle → now ~117 ms median, with `import pyarrow` alone ~116 ms (the floor)
- [x] profile `python -X importtime -c "import mxframe"`; defer everything not needed to build a plan (pyarrow.parquet/csv, compute, numpy submodules, calibration, np_arrow) until first use
- [x] `import mxframe` < 150 ms; the background warm-up still overlaps the rest

## F-020 Remaining SF10 gaps after F-018 [done]
Needs: F-018 merged · From the F-018 report (2026-09-30)
- [x] Q21 (4.7x Polars at SF10): integer group/distinct keys are encoded by direct addressing in Mojo (`groupkeys_cpu.mojo`, `group_keys.encode_int`: first-appearance numbering like Arrow's hash; sparse composites ranked by the radix sort, `dense_sorted`; `first_rows` for Distinct). SF10 Q21 19.2 s -> 9.4 s (Polars 4.2 s), SF1 1.94 s -> 1.0 s. Wide-range keys still use Arrow's hash
- [x] Q10 (7.2x): string `==`/`!=`/range/isin filters are evaluated on dictionary codes inside the Parquet reader (also for columns the plan does not keep as dictionaries). SF10 Q10 2.47 s -> 1.21 s (Polars 0.34 s; the rest is Parquet decode, F-021)
- [x] GPU `hash_join` refuses an output past int32 before the prefix sum (`tests/test_join_gpu_limit.py`)
- [x] F-016 fuzz seed 5061 on gpu: a Project under a grouped aggregate kept the plan off the fused pipeline and the GPU fallback path summed float64 in float32; the Project is now folded in first (regression test; 200 large seeds x gpu/auto pass). Not fixed: that fallback path (`gpu_aot`) still rounds float64 to float32 for any plan the fused pipeline declines
- [ ] per-operator GPU placement for large joins — stays in F-007
- [x] rare Mojo-runtime crashes under load: not reproduced (see `scripts/stress_cpu_kernels.py`: 8 processes x 4 threads x 240 s, 320 fresh processes racing the warm-up thread, 1,440 fresh-process SF1 queries under 6-way load, all clean); slow stress tests added
- [x] a self-join of one `scan_parquet` frame (`a.join(a, on="k")`) raised KeyError 'x_right' when the renamed column was used: the scan now keeps the columns a join renames

## F-021 Native Mojo Parquet reader [todo]
Needs: none (later) · From the F-017 decision: PyArrow's decoder is the floor (~225 ms for 3 SF10 lineitem columns vs Polars' ~77 ms scan+filter)
- [ ] Mojo decoder for the common Parquet encodings (plain, dictionary/RLE, snappy) of numeric and string columns, on CPU and later GPU
- [ ] late materialization and page skipping using the page index
- [ ] replaces PyArrow on the query path; PyArrow stays as the fallback for anything unsupported

## F-022 GPU fallback precision and Q21 [done]
Needs: F-020 merged · From the F-020 report
- [x] the `gpu_aot` grouped-aggregate fallback accumulated float64 inputs in float32 whenever the fused pipeline declined a plan: a GPU session now runs that fallback on the CPU kernels (float64 accumulation; provenance path `cpu_aot`, device `cpu`, `gpu_declined`), and derived columns (`case_when`, isin, ...) stay float64 on both devices (`tests/test_gpu_fallback_precision.py`: declined plans give the same digits on cpu and gpu)
- [x] SF10 Q21: the (orderkey, suppkey) distincts are ranked by a counting sort on the leading key (`mx_ik_rank_bucketed`, `group_keys.dense_sorted(lead=...)`) instead of a radix sort of 60M composites: 1.26-1.76 s -> 0.33 s per distinct, identical ids/uniques/first rows (`tests/test_rank_bucketed.py`). Q21 user-felt SF10 9.4 s -> 8.3 s (Polars 4.5 s, same run, load ~5)
- [ ] Q21 rest (profile, SF10 cpu 8.4 s): Parquet read 1.6 s (F-021), `encode_int` 1.2 s, Arrow take/compute ~1.8 s, rebuilding 60M unique key columns in `_build_group_ids` 0.7 s. Next step: plan `distinct -> groupby(k) -> count` as one n_unique per key (no 60M-row distinct table) — moved to F-023
- [ ] `encode_int` hash/sort fallback for wide key ranges (> 4x rows): not seen in TPC-H; only if real data needs it
- Seen once during F-022: SF1 Q18 `auto` exited -11 (no stderr) in a full-suite run while a 60M-row benchmark loaded the machine; 16 reruns clean (the rare crash under load from F-020)
- `pixi run test-import` (`scripts/_test_base_import.py`) fails at HEAD before F-022 too: `mxframe.GraphCompiler` loads with MAX blocked

## F-023 Q21: distinct + count as one n_unique [done]
Needs: F-022 · From the F-022 profile
- [x] `select(a, b).distinct().groupby(a).agg(count)` runs as a per-key distinct count without materializing the distinct table: when `a` is non-decreasing (lineitem is stored by order key, and filters keep that), runs of equal keys are the groups and one native pass (`mx_count_distinct`) counts each run's distinct `b`. Unsorted keys, nulls, GPU devices and other shapes keep the old plan. `tests/test_count_distinct.py` (20). Reviewer PASS
- [x] SF10 Q21 within 1.5x Polars (engine-only, the headline since 2026-10-13): 3.44 s -> 0.96 s (Polars 4.12, DuckDB 2.80); SF1 279 -> 102 ms; both equal to DuckDB

## F-024 Engine-only scoreboard [done]
Needs: F-009 · Asked for 2026-10-12: the user-felt headline mixes Parquet reading with query speed; add a second number that isolates the engine. The user-felt number stays the headline.
- [x] `bench_user_felt.py --mode engine`: each engine first loads the columns its query uses into its own in-memory form (pandas DataFrame, Polars DataFrame, DuckDB in-memory table, an Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers query + materialized result only. Still a fresh process per (engine, query, sf) with no warm-up
- [x] the load time is recorded (`load_ms`), so read cost = user-felt minus engine-only is visible per engine
- [x] nothing in the timed region touches the Parquet files (test: the data directory is moved away after loading and every query still runs)
- [x] engine-only results equal the user-felt results for all 22 queries on every engine (SF0.01 generated in the test), and are checked against DuckDB in the run like the headline
- [x] the repeat run reloads the tables first, so the engine's caches keyed by table identity cannot make it look faster than a second, different query would be
- [x] the report says which mode it is (title, the clock sentence, methodology); old CSVs without `load_ms` render, and `--resume` into a CSV with a different header is refused
- [x] SF1 + SF10 engine-only run on a quiet machine (`docs/bench/engine-sf1.md`, `engine-sf10.md`), summarized next to the user-felt totals in `docs/benchmarks.md` and the README (2026-10-12 09:15, commit 302cc70; all 132 mx results match DuckDB). Findings:
  - About half of the gap to Polars is the engine and half the read (mx-auto/Polars without Q12 and Q19: SF1 2.09x user-felt vs 1.78x engine-only, SF10 1.91x vs 1.60x; engine ≈ 51% / 54% of the gap in ms).
  - Q12 and Q19 are slower on in-memory tables than through scan_parquet; cause not profiled (F-025).
  - The old "decode is most of the loss on Q3/Q10/Q12/Q14/Q19" claim in benchmarks.md is now scoped to what the data shows.

## F-025 Q12/Q19 slower on in-memory tables than through scan_parquet [todo, check again after F-031/F-032]
Needs: F-024 · From the engine-only run
- [ ] profile SF10 Q12 and Q19 on in-memory Arrow tables (`--mode engine`): mx-cpu Q12 1,827 ms engine-only vs 954 ms user-felt with the read included, Q19 2,271 vs 654 ms; Polars 173 / 487 ms. Candidates, unmeasured: the reader's filter pushdown hides a slow in-engine filter (Q12's `l_receiptdate` range, Q19's conditions), and Q12's plan joining before filtering
- [ ] fix what the profile shows: the same query on an in-memory table is no slower than through `scan_parquet`
- [ ] engine-only Q12 and Q19 within 2x Polars at SF10

## CPU track (agreed 2026-10-12)
Strong CPU at every size first, then warm GPU and large data on small GPUs, then `auto` on top. **Principle (2026-10-12): Python is a thin front end (API, SQL, planning); all heavy lifting runs in Mojo, as one call per query** (`docs/design/mojo-executor.md`, approved). Order: F-026 ✓ → F-027 → F-028 (runtime load only) → F-030 (M0 spike) → F-031 … F-034. F-025 and F-029 are folded into the executor milestones. Borrowing Polars' ideas (not its code) is fine. Baseline: `docs/bench/small-{userfelt,engine}.md` (~12k / 100k / 1M lineitem rows = TPC-H SF 0.002 / 0.017 / 0.17), plus `engine-sf{1,10}` and `final-sf{1,10}`.

## F-026 No pandas import on the query path [done]
Needs: none · From the CPU baseline: Q7, Q13, Q16, Q17, Q19, Q21 take 115–140 ms at 12k rows against 3–5 ms for the other queries, because some query paths make PyArrow import pandas (110–160 ms, once per process). Cause: PyArrow builds empty results with `pa.array([])`, which imports pandas. That happens on `ChunkedArray.combine_chunks()` / `to_numpy()` of a zero-chunk column (an intermediate a filter emptied), on masked NumPy arrays (the NULL of an aggregate over no rows), and on an empty `isin` list. Empty intermediates are common at 12k rows and rare at SF1, where these queries showed no such jump.
- [x] none of the 22 TPC-H queries imports pandas, mx-cpu, user-felt and engine-only, each in a fresh process (test over generated SF0.002 data)
- [x] the paths found get a pandas-free equivalent; nothing falls back to `pa.array` on the query path for inputs we produce ourselves
- [x] small-size engine-only rerun (`docs/bench/small-engine-f026.md`, load ~3): those six queries 3–8 ms at 12k rows (were 115–140). mx-cpu sum 571 → 104 ms at 12k (pandas 102, Polars 44) and 507 → 160 ms at 100k (pandas 177, Polars 68). Left for F-028: one-off ~23 ms first calls (Q6 at 12k, Q1 and Q21 at 100k)

## F-027 Empty intermediate results [done]
Needs: none · From the CPU baseline at ~12k rows
- [x] SF0.002 Q11 no longer crashes (`TypeError: None * float`) and Q17 is no longer WRONG on mx-cpu; root cause found and fixed in the engine (empty group / empty aggregate semantics), not worked around in the harness
  - Outcome, **accepted by Abhishek 2026-10-12** although it doesn't match the check as written: neither root cause was in the engine, which already returns SQL NULL over no rows. Q11 crashed in the benchmark's own mx query code (`None * fraction` when Germany has no suppliers; now `value > NULL` keeps no rows, as SQL does). Q17 failed because the DuckDB reference had `coalesce(sum, 0)`, which the published Q17 does not; the reference is now the published SQL and mx-cpu matches it exactly
- [x] aggregates over zero rows follow one documented rule (SQL: SUM/AVG/MIN/MAX of no rows = NULL, COUNT = 0), with tests: `tests/test_empty_aggregates.py`, on tables with and without nulls (the `_aggregate_with_nulls` path too), global and grouped, plus an all-NULL group
- [x] the harness settles Q19 at SF0.002 (pandas/Polars marked WRONG): for pandas and Polars only (their convention), a one-row aggregate of 0 against a NULL reference counts as equal, with a note on the row (`compare_tables(empty_sum_is_zero=True)`); the report's correctness methodology says so
- [x] all 22 queries correct on every engine at SF 0.002, 0.017 and 0.17: `docs/bench/small-userfelt-f027.md`, 198 results checked against DuckDB, 195 ok and 3 drift (Polars Q1, float32 accumulation, 3e-6 to 1.6e-3). mx-cpu 66/66 ok with no NULL-vs-0 note. `tests/test_tpch_small_correct.py` checks pandas, Polars and mx-cpu against DuckDB at 0.002 and 0.017, user-felt and engine-only paths

## F-028 One-time Mojo runtime load [done]
Unblocked 2026-10-12: Abhishek approved the toolchain upgrade (F-035); on Mojo 1.1 the runtime no longer links libNVPTX.
Done through F-035: library load 19 → 5.6 ms; the remaining first-query wait was the warm-up's `import pyarrow.compute`, now lazy.
Needs: F-026 · From the CPU baseline: engine-only median per query at 12k rows Polars 2.6 ms, mx-cpu 4.6 ms (before F-026), pandas 5.4 ms; user-felt read median mx 31 ms vs Polars 19 ms
- [ ] profile where a 12k-row query spends its time (planning, Arrow↔NumPy conversion, ctypes calls, result building) and write it down
  - First finding (2026-10-12): the ~23 ms first runs (Q6 at 12k, Q1 and Q21 at 100k, reproducible 23–25 ms in fresh workers vs 3–6 ms when warm) are the one-time Mojo runtime load. The first kernel library dlopen costs 22–42 ms, whichever library it is: `libNVPTX.so` (56 MB LLVM NVPTX backend, linked even by the CPU libraries) 11 ms, `libKGENCompilerRTShared.so` 5 ms, the rest runtime start-up. The import-time warm-up thread hides it only when the data load takes longer (Q1 load 27 ms → query 5 ms; load 8 ms → query 25 ms); with `MXFRAME_WARM=0` every first query pays 27–50 ms. Leads: build the CPU libraries without the NVPTX dependency; one CPU library instead of four; start the warm-up before `import pyarrow.compute` (29 ms) in `import mxframe` (89 ms)
- ~~cut the Python-side fixed costs~~ superseded by the Mojo executor (F-030+): Python glue is not where the fix belongs
- [ ] the one-time runtime load (~22 ms per process) off the first query: CPU libraries without the NVPTX dependency if the toolchain allows, one CPU library, warm-up started earlier; measured in fresh workers at 12k rows (Q6, Q21)
  - Investigated 2026-10-12, no code change shipped. The cost is the Mojo runtime itself: `libKGENCompilerRTShared.so` and `libAsyncRTMojoBindings.so` (Modular's packages) both link the 56 MB `libNVPTX.so`, so any Mojo-built library pays ~17–20 ms of relocation on first load (63k relocations, almost all data relocations: `RTLD_LAZY` doesn't help; `-Xlinker --as-needed` on our libraries doesn't either, the dependency isn't ours).
  - Moving the load can't hide it: dlopen holds the dynamic-loader lock that Python's own extension imports need. Preloading at the top of `import mxframe`, or first in the warm-up thread, turns the 12k-row first query from ~23 ms into ~5 ms but `import` from ~124 into ~143 ms; import + first query stays ~150 ms (engine-only) and ~155 ms (user-felt) either way. Shipping that would only move time from the headline column into the import column, so it stays out.
  - Remaining small-data cost to look at in M1: SF0.017 Q21 pays ~23 ms on its first run even with the runtime loaded (not this load)

## F-035 Toolchain: Mojo 1.1 / MAX 26.6 [done]
Needs: Abhishek's go (given 2026-10-12) · Unblocks F-028
- [x] `pixi.toml` `modular = "26.6.*"` (Mojo 1.1.0, Sept 2026; was 26.4 / Mojo 1.0.0b2)
- [x] all seven AOT libraries and the MAX Graph custom-op package build on 1.1. Changes are mechanical: `std.gpu` → `max.gpu`, `parallelize` from `max.algorithm` taking the worker as a value (`def worker(i: Int) {imm}:` closures), `memcpy` / `memset_zero` / `alloc` → `unsafe_*`, `InlineArray` → `Array`, `Atomic[DType.x]` → `Atomic[Scalar[DType.x]]`, pointer aliases `Pointer[T, MutUntrackedOrigin]` (struct fields can't hold `AnyOrigin`), `.address` → `._mlir_value`, `enqueue_function_experimental` → `enqueue_function`, GPU kernel arguments `Int64` instead of `Int` (Int is no longer `DevicePassable`); the package is `kernels.mojoc` (1.1 writes `.mojoc`, not `.mojopkg`)
- [x] `libNVPTX.so` is gone from the runtime's dependencies: dlopen of `libmxexec_cpu.so` 19 ms → 5.6 ms in a fresh process
- [x] `pixi run test` 2053 passed, 2 xfailed; `test0`, `test1`, `test4`, `test5`, `test6`, `test6-gpu`, `test-sql`, `test-q5q10` pass; `test-import` was failing before the upgrade too (it expected `GraphCompiler` to need MAX, which F-019 made lazy) and now checks that MAX is not imported eagerly
- [x] small-data scoreboards (12k and 100k rows) against the 1.0.0b2 runs: `docs/bench/small-{engine,user-felt}-mojo11.md` (run on the port before the lazy `pyarrow.compute` change). User-felt, sum of 22 first runs: mx-cpu 765 → 511 ms at 12k rows (pandas 526, Polars 443) and 835 → 587 ms at 100k (pandas 622, Polars 455). The first query of a process still paid ~10 ms: it waited for the warm-up thread, 29 of whose ~45 ms were `import pyarrow.compute`
- [x] `pyarrow.compute` loads on first use (`_lazy_pc.py`; the comparison tables hold names), not at import or in the warm-up. Fresh process, SF0.002 Q6 engine-only: load + first query 47 + 11 = 58 ms → 33 + 2 = 35 ms; Q1 (still on the Python path, which calls Arrow compute) 58 → 58 ms, the import moved into the query. The benchmark worker no longer pre-imports it for the mx engines (`tests/test_lazy_compute.py`)
- [x] review fixes: `MANIFEST.in` shipped a hand-kept module list that had gone stale (no `native_exec`, `io_parquet`, `join_engine`, `np_arrow`, ...) and `kernels.mojopkg`: now `include *.py` and `kernels.mojoc`; runtime extra `modular>=26.6`; `_lazy_pc` always imports through the module lock; `_LazyPcOps.get` returns functions and `.values()` / `.items()` raise
- Follow-up: ~1,300 deprecation warnings (positional `ptr[i]` → `ptr[unsafe_offset=i]`, `.free()` → `.unsafe_free()`, `@parameter` closures); harmless on 1.1, to be cleaned before Modular removes them

## F-036 Kernel libraries find the Mojo runtime wherever they are installed [done]
Needs: none · From the F-035 review
- [x] every `kernels_aot/*.so` carries a RUNPATH of the build machine's pixi env (`.../mxframe-mojo11/.pixi/envs/default/lib`, older builds `.claude/worktrees/agent-*`), and pixi sets no `LD_LIBRARY_PATH`: they find `libKGENCompilerRTShared.so` only while that directory exists. Use an `$ORIGIN`-relative or env-relative rpath (or preload the runtime from the active env before dlopen), and test it from a fresh checkout and an installed wheel
  - Done: `scripts/set_runpath.sh` (patchelf, now a pixi dependency) sets RUNPATH `$ORIGIN/../.pixi/envs/default/lib:$ORIGIN/../../modular/lib:$ORIGIN/../../../..` (checkout or worktree first, so a checkout prefers its own env over a directory beside it; then pip wheel, venv / conda env). Linux only; macOS dylibs are out of scope. The 0.6.0 wheel still has the absolute RUNPATH (CHANGELOG); the build scripts, `release.sh`, both workflows and CONTRIBUTING call it after every `mojo build`. The runtime libraries carry their own `$ORIGIN` RPATH, so finding `libKGENCompilerRTShared.so` is enough for the rest. `tests/test_runpath.py` reads each shipped `.so`'s RUNPATH, and copies a library into a venv-like and a pip-`modular`-like layout and dlopens it from `/usr/bin/python3` (no conda RPATH of its own), with a control that fails without the runtime. Not done: a fresh venv with the real `pip install modular` wheel (large download); its layout (`site-packages/modular/lib`) is taken from `max/_core`'s own RPATH
- [x] until then: rebuild the libraries in the checkout that will use them (`bash scripts/build_pipeline.sh` and the other targets)
  - No longer needed for the RUNPATH (it is relative); still rebuild a library after changing its source

## F-037 Integer arithmetic past 2^53 in the bytecode interpreter [done]
Needs: none · From the F-032 review (round 2); predates the executor (the fused pipeline has it too)
- [x] `(a * b) - lit(2**53) == lit(261134297)` with `a = b = 94906267` returns no rows (every row should match): column values are guarded (F-032, `E_INEXACT`), intermediate results are not. Bound each `*`, `+`, `-` result from its inputs' min / max and decline when it could pass 2^53, or give the interpreter int64 registers (typed vectors, design decision 4)
  - Built: `_Builder._arith` bounds every integer `+ - *` (and `case_when`, `year()`) as a lazy expression tree: column TYPE ranges first (no data read), the columns' real min / max (parallel, cached per compile) only when the type ranges could pass 2^53; constant folds use exact Python ints. Declines raise `Inexact` and the exact path answers. All entry points compile through it (`mx_execute`, `mx_filter`, `mx_filter_topk`, fused pipeline). No Mojo change.
  - Integer SUM (review round 1): a `sum` whose rows * max|value| could pass 2^53 declines; `_aggregate_with_nulls` (Arrow path) then sums exactly in int64, and `np_arrow.group_aggregate` sums integers in three 21-bit limbs instead of `np.bincount`'s float64 weights. Result type change: integer SUMs on that path (also nullable integer columns) are int64, were float64; sums that fit stay float64 natively until the follow-up makes integer SUM int64 everywhere.
  - Tests: `tests/test_int_arith_2_53.py`. Cost: integer `* + -` over int64 columns reads their min / max once per query (parallel) unless the type ranges suffice

## F-038 Two tests fail under load [done]
Needs: none · Seen 2026-10-12 while other test runs and benchmarks shared the machine; both pass alone (3/3)
- [x] `tests/test_cpu_kernels.py::test_thread_budget_changes_actual_parallelism` (wall-clock speedup assertion) and `tests/test_gpu_kernels.py::test_failed_hash_join_frees_its_scratch_buffers` (GPU memory measured while other processes hold GPU memory), and `tests/test_auto_cold.py::test_a_short_query_inside_the_window_restarts_it` (timing window, failed once at load average 7 during the M2e review): measure what each test means without depending on a quiet machine (CPU time or per-process GPU accounting), or mark them as needing a quiet machine
  - Done: the thread test counts the threads that ran CPU (`/proc/self/task/*/schedstat`): threads=1 gives 1, threads=4 gives 5, the same at load average 22; the GPU test reads this process's own GPU memory (`nvidia-smi --query-compute-apps`) instead of device-wide free memory; the window test runs on a fake clock (it was also order-dependent: alone it failed, because it relied on an earlier test having set `_last_end`). Nothing needed a `quiet_machine` marker. Same flavour, not changed: `test_background_work_waits_for_an_idle_gap` (real 50 ms sleeps between queries)

## F-029 CPU joins [superseded by F-032]
Needs: F-026 · From the engine-only run: at SF10 mx-auto is 2.6x–5.0x behind Polars on Q2, Q3, Q8, Q10, Q16, Q17, Q20 with reading out of the clock (F-007 is GPU placement; this ticket is the CPU join itself)
- [ ] profile the CPU join path on those queries (build, probe, materialization, key encoding)
- [ ] apply what the profile points at (Polars-style ideas: partitioned parallel build, join on dictionary codes / integer keys, late materialization of payload columns, join order by cardinality)
- [ ] engine-only SF10: each of those queries within 2x Polars

## F-030 Mojo executor M0: one call runs the plan (spike) [done]
Needs: design approved · Design: docs/design/mojo-executor.md · Time-box: a few days (took one session)
- [x] Arrow C Data Interface in (`_export_to_c` per chunk; Mojo reads the ten-word ArrowArray structs directly) and out (the executor writes the result into one arena that Python wraps zero-copy with `pa.foreign_buffer` and frees with `mx_exec_free`; no NumPy in either direction)
- [x] a serialized plan (an int64 buffer: shape, bytecode, columns, key kinds, outputs, sort) compiled once in Python (`native_exec.py`); columns chunked differently are cut at common boundaries with zero-copy slices
- [x] `mx_execute(plan, out)` in `libmxexec_cpu.so` (`kernels_aot/executor_cpu.mojo`): group keys (dictionary strings with per-chunk dictionaries merged in Mojo, int32, int64), filter + aggregate (the existing bytecode interpreter, run per 64k-row morsel together with the key encoding), mean / count, sorting the groups by key values (strings by their bytes), and the output columns including string keys. Q1 and Q6 run through that one call and match DuckDB (SF1 / SF10 max relative error 9e-15); `tests/test_native_exec_m0.py` (27 tests) covers shapes, chunking, slices, empty input, the multi-threaded per-morsel path with min/max, more than 4096 groups, a dictionary too big to merge (declined after the call), dictionaries repeating in non-adjacent chunks, invalid dictionary codes (declined, never read through) and an executor exception (falls back); with `MXFRAME_NATIVE=1` the whole suite gives 2039 passed and 8 failed: 4 tests assert the old fused path served the query, 4 demand bit-identical CPU vs GPU digits (the executor sums per morsel, ~1e-15 apart)
- [x] Python-side time per query 0.26–0.5 ms up to SF1, 1.5–1.9 ms at SF10 (one C Data export per chunk per column; measured in-process with `native_exec.last_timing()`, warm, not in the bench files). Engine-only, first run in a fresh process, **single runs at load average 1.7–6.6, so differences under ~30% are within noise** (`docs/bench/m0-native.md` vs `m0-today.md`; SF10 Q1 repeat runs agree, 62 vs 137 ms): SF10 Q1 66 ms vs 141 today (Polars 2,608, wrong), SF1 Q1 13 vs 21, SF1 Q6 6 vs 9, SF10 Q6 31 vs 30 (Polars 46); at 12k rows both ~24 ms, of which ~20 is the one-time Mojo runtime load (F-028)
- **Go for M1.** What M0 taught:
  - Fusing the key encoding with aggregation per morsel (no n_rows-sized intermediate) made Q1 2x faster than today's path; materializing a 240 MB group-id array cost more than the work. M1 should keep every operator inside morsel pipelines.
  - The one-time runtime load (F-028) now dominates small data: do it before or with M1.
  - Exporting one C Data struct per chunk per column costs ~2 ms at SF10; export streams or cache exports in M1.
  - When the executor becomes the default, the GPU session's fallback must use it too, or CPU and GPU digits differ in the last place.
  - Still float64 everywhere inside the interpreter; typed vectors and a hash-table aggregate (merged dictionaries are capped at 4096 strings, dense groups at 2^20) are M1.
  - Opt-in (`MXFRAME_NATIVE=1`) until M1 makes it the default.

## F-031 Mojo executor M1: single-table pipelines [done]
Reviewer PASS (round 3, 2026-10-12). Two decisions wait for Abhishek: the row order below, and moving the "within 1.5x Polars" check to M2.
Needs: F-030
- Finding that set the scope: joins are materialized before the aggregate, so the executor sees the final aggregate stage of every query as a single-table plan; M1 coverage is per query, not just Q1/Q6
- [x] `Limit? Sort? Having? Aggregate(Filter*(Scan))` in one `mx_execute` call (plan v2): grouped results without ORDER BY in key order on every grouping path (the hash path reorders its first-appearance groups); ORDER BY any mix of keys and aggregates (stable merge sort on exact sort values: string ranks by bytes, integer codes, or integer ranks when hash-path codes pass 2^53; ties in key order; top-k with a heap when a LIMIT under 1/8 of the groups follows); HAVING as AND-ed `aggregate <op> number`; NaN-aware min / max (the hidden NaN-count slot); a NaN among ORDER BY values declines (`-15`)
- [x] grouping for any key range: dense up to 2^20 combinations (fused per morsel up to 4096), compacted through a presence table + prefix sum up to 8x the rows (cap 2^27), otherwise a hash table on the tuple of per-key codes (groups in first-appearance order). Plain utf8 keys and the merge of per-chunk dictionaries hash the strings' bytes (the 4096-string cap and the linear merge are gone); strings sort by a merge sort of their bytes
- [x] string predicates on dictionary columns: Python concatenates the chunks' dictionaries (dictionary-sized work), the interpreter loads `code + chunk offset` (column type 5); no `unify_dictionaries`, no row-sized remap
- [x] on by default (`MXFRAME_NATIVE=0` turns it off); a GPU session answers plans its own pipeline declines through the executor (same digits as `device="cpu"`); a CPU+GPU co-processing policy keeps the split path
- [x] tests: `tests/test_native_exec_m1.py` (sort / limit / having / unordered / large and sparse key ranges / plain and dictionary string keys / string predicates / all 22 TPC-H queries equal to DuckDB with the executor on), M0 tests updated; full suite 2081 passed with the executor on by default; legacy suites pass
- [x] coverage (SF0.017): the final aggregate stage is native in 16/22 queries (M0: 6). Still declined: derived group keys (Q7, Q8, Q9: year of a date via CASE WHEN), string predicates on plain utf8 columns (Q12, Q14, Q19: join results are decoded, see below), a count(*)-only plan (Q13)
- [x] engine-only first runs, executor vs the old path (load average 3-5 during the runs, so under ~15% is noise; `docs/bench/m1-sf{1,10}-native{1,0}.md`): SF1 Q1 13 vs 40 ms, Q13 37 vs 49, the rest equal within noise (total 2699 vs 2734; Polars 1391); SF10 Q1 65 vs 160 ms (Polars 2,879), Q10 1105 vs 1208, Q11 83 vs 98, nothing slower beyond noise (total 21.0 vs 21.2 s; Polars 15.0 s)
- [ ] engine-only at 12k–1M rows within 1.5x Polars on the queries covered: **not met, and M1 can't meet it; proposed: move this check to M2 (Abhishek to confirm)**: on 20 of 22 queries the time is in the joins and the Python orchestration around them, which the final-stage executor does not touch (SF1: Q12 198 ms vs Polars 19, Q19 236 vs 44, Q2/Q8/Q11/Q14/Q15/Q17 5-7x). That is M2
- Found for M2: `io_parquet.resolve_plan` decodes every dictionary string column of every leaf when the plan has a join (and any column used in CASE WHEN): at SF10 that is ~60M `l_shipmode` strings for Q12 before the join starts. Joins should carry dictionaries as they are and let the executor read codes
- Order (needs Abhishek's OK): the executor returns unsorted GROUP BY results in key order (pandas' `groupby` default) and breaks ORDER BY ties by key order; the old path used first appearance. SQL leaves both undefined, and first appearance among *filtered* rows would need new interpreter support. Tests pin the new order (`test_unsorted_groups_come_out_in_key_order_on_every_path`, `test_order_by_ties_and_limit_follow_key_order`)
- Review round 2 (NEEDS WORK) fixed: string keys on the dense / compacted paths came out in dictionary-entry order; every grouping with a string key is now reordered by bytes (`test_string_keys_come_out_in_byte_order`)
- Review round 1 (NEEDS WORK) fixed: int64 group keys past 2^53 sorted through float64 lost order on the hash path (now ranked exactly); hash-path groups came out in first appearance while the other paths used key order (now key order everywhere); stale docstring; `last_path()` stale when MXFRAME_NATIVE=0
- Not done in M1 (deferred): typed int64 registers (sums are still float64; exact for TPC-H's magnitudes), cached C Data exports, parallel compaction / hashing (both are single-threaded loops today)

## F-032 Mojo executor M2: hash joins [done]
Needs: F-031 · Replaces F-029
- Profile first (SF1, warm code, fresh tables): Q12 spent 136 of 165 ms in the Python path's leaf filter (Arrow compute over 6M lineitem rows, 85 ms) and in decoding dictionary strings (`arrow_dict.decode`, 46 ms: `io_parquet.resolve_plan` decoded every dictionary column of every leaf when a plan has a join, before any filter). The join kernels themselves took ~10 ms. So M2 starts with the leaves, not the hash join
- [x] M2a, native leaf filters: a `Filter*(Scan)` leaf of a join is one `mx_filter` call. The predicate runs on the bytecode interpreter per 64k-row morsel in parallel (pass 1: mask, counts and string bytes per morsel), then the surviving rows of the columns the join needs are written into one arena (pass 2: fixed-width values, utf8, dictionary strings decoded on the way out). Dictionary predicates read codes + chunk offset (dtype 5, as in M1). Declines (nulls, other types, plain-string predicates) run the old path. `tests/test_native_exec_m2.py`
- [x] M2a, deferred decode: a dictionary column resolve_plan decoded only because the plan has a join stays encoded until its leaf's filter has run (`Scan._mx_decode_after`; `join_engine.leaf_table` decodes the surviving rows), on either path
- [x] M2b: deferred dictionary columns of a join's Filter*(Scan) inputs are decoded after the join's final gather (joined rows only); CASE WHEN and other uses above the join are deferred too (only the leaf's own filter needs them decoded first); scan_parquet reads deferrable columns as dictionaries. Other leaf shapes (Limit, Sort, Aggregate under a join) keep the up-front decode
- [x] numbers, engine-only first runs (`docs/bench/m2a-*`, `m2b-*`; load average 2-4): SF10 total 21.0 s (M1) → 16.2 s (M2a) → 15.7 s (M2b), Polars 15.0 s; MXFrame beats Polars on 6/22 and is within 2x on 14/22 (M1: 13/22). Q19 2,159 → 146 ms (Polars 487), Q12 1,788 → 511, Q10 1,105 → 559, Q14 345 → 189, Q20 459 → 286. SF1 total 2,699 → 2,209 ms (Polars 1,391). Worst left: Q17 3.8x, Q16 3.8x, Q8 3.5x at SF10
- [x] small data (12k / 100k rows, `docs/bench/m2b-small-*`): import + first query, sum of 22, user-felt: mx-cpu 3,503 → 3,245 ms at 12k (pandas 5,811, Polars 2,043, DuckDB 2,885) and 3,833 → 3,373 at 100k; mx-cpu beats pandas on 22/22 queries at both sizes. The "first run" column alone grew because `pyarrow.compute` (~25 ms) now loads in the first query that needs it instead of at import; join queries still need it for Arrow `take` in the join's gather (next item)
- [x] review round 1 (NEEDS WORK) fixed: int64 value columns past 2^53 were compared / summed in float64 by the bytecode interpreter (wrong rows); `mx_execute`, `mx_filter` and the legacy fused pipeline now decline such columns (`E_INEXACT`, the old path is exact); deferral limited to Filter*(Scan) join inputs; tests for left joins, other leaf shapes, a Parquet source, int64 past 2^53
- [x] native gather (`mx_gather`): the join's final gather is one native call (fixed-width, utf8, and deferred dictionary columns decoded in the gather); Arrow `take` only for other types or nulls. Q3 now runs without importing `pyarrow.compute`
- [x] review round 2 (PASS) follow-ups: the int64 range check runs in parallel; the fused pipeline's guard has a test; Python dictionary membership below 50k values; F-037 for arithmetic intermediates past 2^53. The m2a / m2b benchmark numbers above were taken before the int64 guard
- [x] numbers after M2c (`docs/bench/m2c-*`, load average 1.5-4): engine-only first runs, SF10 total 15.9 s (M1 21.0 s, Polars 15.0 s), beats Polars on 6/22, within 2x on 13/22; SF1 2.24 s (M1 2.70 s, Polars 1.39 s). User-felt at 12k / 100k rows, import + first query summed over 22: 3,225 / 3,250 ms (this afternoon 3,503 / 3,833; pandas 5,769 / 5,841, Polars 2,110 / 2,063); beats pandas on 22/22 at both sizes, all results equal to DuckDB. The native gather did not move the totals (within noise); it removes Arrow `take` and the `pyarrow.compute` import from join queries
- [x] review M2c (PASS); follow-up test for sliced, misaligned chunks and per-chunk dictionaries in the gather. Open from the review: work out each row's chunk once per morsel in `mx_gather` instead of per column (performance)
- [x] M2d, star joins in one scan (`tests/test_native_lookup.py`): a tree of inner joins runs as one `mx_filter` call over its largest Filter*(Scan) leaf (the fact table). Every other leaf belongs to the subtree of one of the fact's neighbours; each subtree is joined on its own (small tables, the existing path) and becomes a partner that Mojo indexes on its join key (1 or 2 integer columns; a direct array for compact ranges, else a hash table in 64 partitions, both built in parallel). While the fact's predicate runs, each surviving row looks up its partner rows; only joined rows are written, the fact's columns and the partners'. Conditions across subtrees (Q5's `c_nationkey = s_nationkey`, Q7's nation pair) run on the joined rows. Partner keys must be unique (foreign key to primary key); a duplicate returns `E_DUP` and the join runs leaf by leaf as before, as does anything else the call declines. `MXFRAME_STAR=0` turns it off
- [x] numbers after M2d (`docs/bench/m2d-engine.*`, engine-only first runs, 44/44 results equal to DuckDB, load average 1-7 during the run): SF10 total 15.9 s → **11.1 s** (Polars 15.0 s); beats Polars on 14/22 (M2c: 6), within 2x on 18/22 (M2c: 13). Q3 674 → 265 ms (Polars 311), Q5 493 → 211 (482), Q7 660 → 222 (1,105), Q8 427 → 229 (120), Q9 1,496 → 667 (905), Q10 546 → 282 (368), Q12 307 → 168 (173), Q18 1,980 → 825 (1,427), Q20 299 → 120 (137). SF1 total 2.24 s → 1.76 s (Polars 1.39 s). Every M2 join query (3, 4, 5, 7, 10, 12, 14, 19) is within 2x of Polars at SF10. Worst left at SF10: Q16 2.8x, Q2 2.8x, Q15 2.7x, Q17 2.5x; Q21 is the largest absolute gap (5.8 s vs 4.2 s, its DISTINCT + group-by over 60M rows: F-033)
- [x] review M2d round 1 (NEEDS WORK) fixed: a condition on a one-leaf partner (a renamed right column the optimizer does not push down) was dropped, now run on the joined rows; the register-count decline leaked the built maps (now checked before anything is allocated); a failed `calloc` declines (`E_NO_MEMORY`); plans without lookups go out as version 1 again. Tests added: composite key in the hash table, int32/int64/date32 keys, a condition inside a two-leaf subtree, no output column from the fact. Row order: a star join's rows come out in the fact table's order (the leaf-by-leaf join's order followed its join order); neither was ever promised, so a join with `.head(n)` and no sort may return different rows than before
- [x] M2e, semi joins: a partner that is `Distinct(Project(key columns)(Filter*(Scan)))` joined on exactly those keys (Q4's late orders) is a semi join: its keys are filtered natively (repeats and all) and looked up as a presence test (plan bit 16: repeated keys allowed, no outputs), so the distinct never runs. Leaves are now evaluated when first needed, so a semi partner is never materialized. Numbers (`docs/bench/m2e-engine.*`, mx-cpu only, 44/44 equal to DuckDB; Polars from `engine-sf10`): Q4 SF10 828 → 259 ms (Polars 962), SF1 106 → 33 ms; SF10 total 11.1 → 10.5 s, SF1 1.76 → 1.66 s
- [x] review M2e (PASS) follow-ups: the semi plan tests assert the star call returned (no silent fallback) and that a distinct over more than the keys is not taken as a semi join. Note: an unfiltered semi partner with keys too sparse for a direct array hashes every row, repeats included (~24 bytes per partner row)
- [ ] anti and left joins in the scan (Q13, Q21, Q22's shapes); partner keys that repeat (many-to-many); multi-pipeline plans (a partner that is itself an aggregate or distinct running inside the same native call). When a partner declines, the subtree joins already built are discarded and rebuilt by the fallback (performance only)
- [ ] Q3, Q4, Q5, Q7, Q10, Q12, Q14, Q19 fully native; SF10 engine-only within 2x Polars. Status after M2e: all eight within 2x of Polars at SF10 (Q3 0.85x, Q4 0.27x, Q5 0.44x, Q7 0.20x, Q10 0.77x, Q12 0.97x, Q14 1.5x, Q19 0.21x). Not yet "fully native": per query there are now 2-4 native calls (partner subtree joins, the star scan, the aggregate) with Python joining small partner subtrees and running cross-subtree residuals. Proposed (for Abhishek): count the 2x target as met and move "one call per query" plus anti / left joins to M3, where Q21 / Q22's subquery shapes are
- Decisions (Abhishek, 2026-10-13): GROUP BY without ORDER BY keeps key order (pandas' default). Target raised: **beat Polars on every one of the 22 queries, engine-only first run, at SF1 and SF10** (supersedes the 2x and M1's 1.5x checks). Losing after M2e: SF10 Q2, Q8, Q11, Q14, Q15, Q16, Q17, Q21; SF1 also Q3, Q9, Q12, Q18, Q20, Q22 (a ~30 ms fixed floor per query against Polars' 5-15 ms)
- [x] M2f, no `pyarrow.compute` in the query paths: its import (~22 ms per process: a docstring and signature for each of ~270 functions) was paid inside the first query by 15 of the 22 queries at SF1, through Arrow's own methods (`Array.cast`, `Table.filter`, `.take`, `.dictionary_encode`, `.is_null`). `_lazy_pc.pc` now calls generated functions straight on `pyarrow._compute` (its own `cast` / `take` / `fill_null`), engine code calls `pc.<fn>(x)` instead of Arrow methods, and the warm-up thread loads `pyarrow._compute` (6 ms) and the engine modules. Queries 1-21 no longer load it (Q22's benchmark code imports it itself). SF1 first runs: Q17 41 -> 17 ms, Q11 32 -> 13, Q15 46 -> 21, Q14 37 -> 15. `tests/test_lazy_compute.py`; review PASS (round 2: parity tests unload pyarrow.compute so the fast path is what they test). Still loading it, off the CPU query paths: `auto_mode._key_card` / `_sample_distinct` (~22 ms on the first `device="auto"` query), compiler fallbacks, calibration. Preloading `pyarrow.compute` in the warm-up thread instead was tried and dropped: when loading is fast it is still running as the query starts and slows it (Q1 SF1 13 -> 31 ms)
- [x] M2g, group-by and leaf fixes (`tests/test_native_groupby_keyed.py`, `test_native_lookup.py`): (1) a group-by with directly addressed keys (any group count, was <= 4096) numbers the groups per 1024-row block for the rows that pass the predicate only (`pipeline_keyed`: no group-id array over the whole table, keys of filtered-out rows never read), aggregating into per-worker partials; before, more than 4096 groups wrote a group id for all 60M rows first; (2) the integer key range scan runs in parallel; (3) a join leaf `.filter(...).select(key)` (a Project that only selects columns) is still a Filter*(Scan) leaf: its predicate runs natively on dictionary codes and its strings are not decoded up front (Q17's part, Q2). SF10 Q15's view group-by (100k groups) 158 -> ~63 ms; Q15 190 -> 66, Q17 145 -> 85, Q2 145 -> 28 ms
- [x] review M2g (PASS); its should-fix items done in M2h: a Parquet test for the `.filter().select(key)` leaf, the composite-key test now runs the keyed path (a second case covers the compacted one), and io_parquet sees through one Project like the join engine
- [x] M2h, distinct, string keys and star scans (`tests/test_native_distinct.py`, `test_native_lookup.py`):
  - native DISTINCT (`mx_distinct`): row tuples hashed in parallel, a per-block cache drops rows repeating an earlier row of their block (few distinct tuples would all land in one partition), the rest scattered to 64 partitions in row order, each partition keeps the first row of every tuple; the kept rows in row order are the first-appearance order the Python path returns, then one native gather. int32 / int64 / utf8 without nulls. Q16's distinct (1.2M rows) 112 -> 24 ms; 3M rows, 4 columns 1,638 ms (group-by route, tried first) -> 74
  - plain-string group keys interned in parallel (`_build_row_strings`: hash per block, 64 partitions, partition-major entry ids; string keys are ordered by their bytes, never by entry id). Q16's final group-by 76 -> 39 ms
  - a leaf column filtered in the leaf and only projected bare above the join keeps its dictionary until after the join (io_parquet `bad_expr`): Q16's part is filtered natively on codes. Q16's join 113 -> 54 ms
  - a partner subtree of 2+ leaves runs as a star of its own (recursively) before the leaf-by-leaf join; partners are probed most selective first. Q8's orders -> customer -> nation: 101 -> 36 ms
  - `mx_filter`: direct maps carry a presence bitmap (one bit per key, L2-resident), lookups are probed column-wise per 1024-row block with each map's words read once, and pass 1 records each morsel's kept rows so pass 2 copies only those (no row mask over the whole table). Q8's 60M-row lineitem scan 97 -> ~45 ms; Q8 SF10 212 -> ~100 ms (Polars 136)
- [x] numbers after M2h (`docs/bench/m2h-engine.*`, mx-cpu and Polars in one run, load average 6-7 at the start, 44/44 equal to DuckDB): SF10 beats Polars on **20/22** (total 7.3 s vs 15.1 s; losses Q11 33 vs 32 ms, Q16 142 vs 113), SF1 9/22 (total 0.99 s vs 1.37 s). Q21 SF10 5.6 -> 3.9 s (Polars 4.2), its distinct now native
- [x] review M2h (PASS) follow-ups: a test where the condition sits inside a nested star partner (and the nested call is asserted); join keys and DISTINCT subsets count in `bad_expr` again (only bare projected columns defer); kept-row offsets are uint16 (2 bytes per kept row, was 4); the presence bitmap is built only after the duplicate-key check passes
- [x] M2i, many groups and string keys:
  - the interpreter gives all workers' partials 128 MB (was 768) and uses at most rows / groups workers: every worker fills and merges a partial per group, so with many groups fewer workers win (Q18's 1.5M-group subquery at SF1 164 -> 84 ms, Polars 134; Q21 SF1 406 -> 324; Q13 SF1 12.7 -> 8)
  - string-key interning: a per-block cache sends repeats of an earlier row of the block to its entry (25 brands would otherwise land in a few partitions); Q16's final group-by 25 -> 15.5 ms
  - large per-row buffers (string interning, DISTINCT, the interpreter's partials) are huge-page allocations (`big_alloc`), string hashing / equality / copies 8 bytes at a time
- [ ] M2j, semi and anti joins and `str_slice` (`tests/test_semi_anti_join.py`, checked against DuckDB): `join(..., how="semi" | "anti")` as in Polars (the left rows with / without a partner, each once, left columns only, left row order; null keys never match: SQL EXISTS / NOT EXISTS); the optimizer pushes filters above them to the left side and reads only the keys from the right; `col(...).str_slice(offset, length)` (Polars' `str.slice`). TPC-H Q22 is now plain MXFrame (it used PyArrow / NumPy for the substring and NOT EXISTS, and with them loaded `pyarrow.compute`): SF1 34 -> 18 ms (Polars 14), SF10 132 ms (Polars 205). API decision taken overnight (Abhishek asked to proceed on my recommendations): names follow Polars (`how="semi"/"anti"`, `str_slice` for `str.slice`); open to rename. Semi / anti joins run in Python + NumPy over encoded keys for now (`join_engine._execute_filtering`); an anti lookup in `mx_filter` is the native next step
- [x] review M2i + M2j round 1 (NEEDS WORK) fixed: a semi / anti join reads only its keys from the right side whatever sits above it (io_parquet `_analyze`, optimizer projection pushdown); **SQL joins took the wrong type**: sqlglot keeps LEFT / RIGHT / FULL in `side`, not `kind`, so `LEFT JOIN` ran as an inner join (pre-existing, silent) and SEMI / ANTI did too; `sql_frontend._join_type` now maps INNER, LEFT [OUTER], [LEFT] SEMI, [LEFT] ANTI and raises for RIGHT / FULL / CROSS. Tests added: SQL join types, joins of joins (semi then inner, inner then anti on the inner join's right column, left then semi, a right side that is a join), the `_execute_hash_join` fallback, the GPU device (skipped without one), right keys only from Parquet, word-wise string hashing with block repeats. The duplicate `_big_alloc` is gone and 8-byte string loads / stores are explicitly unaligned. To check in the next SF10 run: the 128 MB partial budget leaves one worker above ~4M groups (Q18 / Q21-sized group-bys were unchanged at SF10 in spot checks)
- [x] M2k, threads and maps: kernels use one worker per physical core (hyperthread siblings counted once: memory-bound kernels gain nothing from the second thread; SF10 Q21 3.78 -> 3.54 s, Q13 120 -> 105 ms) and one per 128k input rows below that (SF1 Q8 22 -> 19 ms, Q11 9 -> 7); a direct map gets its presence bitmap only when keys are sparse in their range (< 1/8: Q8's filtered part, not Q12's orders, where every probe hits); `MXFRAME_CPU_THREADS` still overrides
- [x] review M2i + M2j round 2 (PASS); follow-ups done in M2l: `RIGHT SEMI` / `RIGHT ANTI JOIN` raise, `JOIN ... USING (k)` no longer crashes (sqlglot gives a list; pre-existing)
- [x] M2l, integer expression group keys (`tests/test_native_derived_keys.py`): a group-by on an integer expression of the row (TPC-H's order year, a CASE WHEN sum over a date; arithmetic of integer columns) runs in the native executor instead of declining: one native min / max pass measures the key's range, the key expression becomes a hidden aggregate slot of the program, and the keyed path reads its register per block (`K_EXPR`; directly addressed ranges only, others decline). The key comes back as int64 (the Python path returns float64). SF10 Q9 666 -> 439 ms (Polars 930), SF1 Q9 70 -> 56 ms (Polars 71); Q8 a few ms. Q7 still declines (an expression in its aggregate the bytecode compiler does not support)
- [x] review M2l (PASS) follow-ups: the integer check runs before the range pass (a float key no longer pays for a scan before declining); a test for expression keys too wide together (the executor declines, the Python path answers); the `slot_int` comment no longer overstates exactness
- [x] numbers after M2l (`docs/bench/m2l-engine.*`, mx-cpu, Polars and DuckDB in one run, load average 3 at the start with another project's browser tests running; 66 results, all mx results equal to DuckDB): SF10 total **6.45 s** (Polars 15.20, DuckDB 6.66), beats Polars on **21/22** (Q16 99 vs 98 ms) and DuckDB on 17/22; SF1 total **0.73 s** (Polars 1.34, DuckDB 2.31), beats Polars on 14/22 and DuckDB on 22/22. SF1 losses are 1-9 ms each (Q2, Q6, Q8, Q11, Q14, Q16, Q17, Q22): ~3-5 ms of Python per query and per-call start-up on small inputs. Polars' Q1 (2.8 s at SF10, flagged wrong) inflates its total: without Q1, SF10 is 6.39 / 12.40 / 6.44 s
- [x] M2m, sparse key ranges (`tests/test_native_groupby_keyed.py`): profiling the SF1 losses showed Python is ~2-3 ms per query, not the gap; the group-by kernel on small inputs with a wide key range was (32k rows over Q11's 200k part keys: every worker zeroed and merged a G x lanes partial, ~13 MB, and key codes were computed for all G groups with two divisions each). Now a direct key range of at least 16384 and more than 2x the input rows takes the existing compact path (only the groups present are numbered, in key order; never for expression keys), and key codes are computed for surviving groups only. SF1 first runs: Q11 14.6 -> 6.1 ms, Q17 29.5 -> 9.7, Q2 15.7 -> 9.3 (in-process). Engine-only mx-cpu run (`/tmp/m2bench/m2m-mx`, against Polars from the M2l run): SF1 734 -> 714 ms, 16/22; SF10 unchanged within noise, 22/22 with Q16 at 97.7 vs 98.4 ms (needs a same-run check). Reviewer PASS (follow-ups: a HAVING case, a note on how the branch is reached)
- [x] numbers after M2m + F-023 (`docs/bench/m2n-engine.*`, mx-cpu, Polars and DuckDB in one run, median load average 4.4; all 44 mx results equal DuckDB): SF10 total **3.82 s** (Polars 14.12, DuckDB 6.57), beats Polars on **21/21 verified** (Polars' Q1 wrong; closest Q16 95 vs 100 ms) and DuckDB on 18/22 (loses Q4, Q10, Q18, Q22); SF1 total **0.52 s** (Polars 1.33, DuckDB 2.20), beats Polars on 15/22 and DuckDB on 22/22
- [ ] beat Polars on 22/22 at SF1 and SF10: SF10 done (21/21 verified), SF1 15/22 (Q8, Q11, Q14, Q15, Q16, Q17, Q22, each 1-7 ms). Next levers: the per-call cost of the remaining small native calls, keeping dictionary columns encoded through a join when the consumer is in the same plan (Q12/Q16 re-encode strings the join decoded), count-distinct for Q16 (its DISTINCT runs in a separate compute), multi-chunk lookup partners (Q12 SF10 combines 15M rows), a quiet-machine rerun (engine-only, first run, mx-cpu and Polars timed in the same run)

## F-033 Mojo executor M3: distinct and subquery-shaped plans [todo]
Needs: F-032
- [ ] distinct; scalar results feeding a later filter inside one plan
- [ ] 18/22 TPC-H queries fully native

## F-034 Mojo executor M4: general strings [todo]
Needs: F-033
- [ ] LIKE / contains / substr and non-dictionary string keys in Mojo
- [ ] 22/22 TPC-H queries fully native; the Python execution path retired for them

## F-040 A group-by on an aliased column fails [done]
Needs: none · Found by the M2h review (predates F-032)
- [x] `LazyFrame(t).select(col("s").alias("b")).groupby("b")` returns the key as `b` with the right values. (On 2026-10-13 it no longer raised but silently returned the key as `s`: the kernel and fused group-by paths named a plain column key after its source and ignored the alias. A renamed key now takes the derived-key branch.) `tests/test_groupby_alias.py`: 3 paths, each checked to be the one that ran, plus GPU. Reviewer PASS.

## F-041 SQL: a SELECT alias on a group key is lost [done]
Needs: none · Found by the F-040 review
- [x] `SELECT s AS b, SUM(v) AS x FROM t GROUP BY s` and `... GROUP BY b` return `b` like DuckDB. Keys are matched by their translated expression, so one rule covers aliases, `t.s` vs `s`, ORDER BY on the original name or expression, and a key selected twice; result columns follow the SELECT list (a projection after ORDER BY / LIMIT). `tests/test_sql_group_alias.py`: 19 queries checked against DuckDB. Reviewer PASS (round 2)

## F-042 SQL front end gaps found in the F-041 review [done]
Needs: none · Pre-existing, found by the F-041 review
- [x] an aggregate without an alias is named `agg_0` (DuckDB: `sum(v)`) and its SELECT position is ignored
- [x] `SELECT kid ... JOIN ... ON k = kid GROUP BY kid` returns the column as `k`
- [x] `GROUP BY ALL` drops the keys
- [x] a duplicate output name (`SELECT x, SUM(v) AS x`) is accepted: is renamed like DuckDB's `duckdb.sql(q).arrow()` / `.df()` does (`x_1`); DuckDB does not raise (`con.execute(q).to_arrow_table()` keeps the duplicate names)
- [x] HAVING is dropped without an error (`... GROUP BY flag HAVING SUM(v) > 25000` gives no Filter); found by the F-046d review
- [x] `COUNT(DISTINCT x)` raises NotImplementedError
- [x] arithmetic on an aggregate (`SUM(v) * 2 AS y`) is not seen as an aggregate: raises "Aggregate.aggs cannot be empty", or the column is silently dropped when another aggregate is present
- built in `sql_frontend.py` only; `tests/test_sql_gaps.py` (166 cases, each checked against DuckDB; unsupported variants raise). Unaliased aggregates take DuckDB's text (`sum(v)`, `count_star()`, `count(DISTINCT v)`); an unaliased expression (`SUM(v) * 2`, `SUM(v + 1)`) raises asking for an alias. HAVING is a Filter above the Aggregate (it may use aggregates the SELECT list lacks, SELECT aliases and group keys). ORDER BY may use an aggregate or an expression over aggregates. `COUNT(DISTINCT x)` becomes `Aggregate(Distinct(Project(keys, x)))` (the native count-distinct path); alone it takes that path; mixed with other aggregates see round 2 (several COUNT(DISTINCT) and `SUM(DISTINCT ..)` raise NotImplementedError). A LEFT JOIN's right key can no longer be referenced (the join drops it; the old code silently read the left key, which has no NULLs); `SELECT s FROM t GROUP BY s` without any aggregate raises. Also checked on SF1 (lineitem / orders, 6 queries equal to DuckDB)
- round 2 (same ticket, from the coordinator): `SELECT DISTINCT` maps to the plan's Distinct (not with GROUP BY / aggregates, nor DISTINCT ON: NotImplementedError); `GROUP BY n` / `ORDER BY n` ordinals like DuckDB; an aggregate in WHERE raises; duplicate output names are renamed like DuckDB (`x`, `x_1`, `x_2`, first free name in SELECT order, also `SELECT a.k, b.k` over a join) instead of raising; COUNT(DISTINCT x) mixed with sum / min / max / count / avg is done by aggregating per (keys, x) and combining the partials per key (no join, so a NULL key stays a group); `t.v, w.v` where both joined tables have a column `v` (the join keeps one copy) now raises instead of returning the left one twice; an ORDER BY name that is a SELECT output name (`SELECT kid ... ORDER BY kid`) uses that column
- review fixes: `COUNT(*)` is `lit(1).count()` (rows) instead of a count of the first column in scope, which was low when that column had NULLs (HAVING COUNT(*) dropped the NULL group, mixed COUNT(DISTINCT) gave 0, a global count was short); the DataFrame API's `col(x).count()` skips NULLs and `lit(1).count()` counts rows on every path, so it shared no cause; the joined-column guard raises only for the qualifier whose copy the join dropped (`t.v` over `t JOIN w` works, `w.v` raises); `ORDER BY .. NULLS FIRST` raises NotImplementedError (the engine sorts NULLs last, which is DuckDB's default, so NULLS LAST works); tests with NULLs in the first column and keys, an all-NULL group and AVG over NULLs
- still open: more than one COUNT(DISTINCT ..) per query; `GROUP BY s` with no aggregate (SELECT DISTINCT is the form that works); unaliased expression group keys need an alias (DuckDB names them `(k + 1)`); `ORDER BY` an alias that names two SELECT columns (DuckDB's pick is unclear, we take the first)

## F-043 SF10: the four queries DuckDB still wins, with general fixes only [todo]
Needs: none · Agreed 2026-10-13 (Abhishek: focus on these, no TPC-H-only tuning)
- [x] Q18 (697 -> ~420 ms, DuckDB 395): too many groups for a partial table per worker -> all workers share the output, updated with atomic add / min / max behind a per-worker 4096-group cache (skewed keys stay local; under 1/16 hits it goes direct); compacted keys are encoded per block (no group id per row, no remap); survivors, HAVING and key decoding in parallel. Non-TPC-H: 20M rows / 5M groups random sum 608 -> ~390 ms, max 869 -> ~500, two keys 1181 -> ~620; skewed 8M rows: uniform 173 -> 61, hot key 22% 150 -> 57, 95% 29 -> 21
- [x] Q4 (224 -> ~120-140 ms, DuckDB 170): runtime join filter: a semi partner more than 2x the fact's rows is filtered through the fact's filtered keys first (skipped when the fact keeps over half its rows)
- [x] Q10 (187 -> ~170 ms, DuckDB 188): with an ORDER BY in the same call, hashed / string-keyed groups are not sorted into key order first; the keys are the sort's last tie-break columns (group-by 88 -> 48 ms)
- [x] Q22 (148 -> ~94 ms, DuckDB 117): native Mojo `str_slice` (new utf8 array) and utf8 `isin` up to 32 values (code points like Arrow)
- [ ] each change helps the general shape (checked on a non-TPC-H case), and no SF1 / SF10 query gets slower

## F-044 Random queries on the TPC-H data: performance beyond the 22 [done]
Needs: none · Agreed 2026-10-13 (Abhishek: don't overfit TPC-H)
- [x] a seeded generator of random queries over the TPC-H tables (filters, group-bys on random keys, joins, aggregates, sorts / limits, DISTINCT, count distinct), each run in MXFrame, Polars and DuckDB the engine-only way (`scripts/bench_random/`)
- [x] every MXFrame result checked against DuckDB: 250/250 equal (100 SF1, 100 SF1 shuffled, 50 SF10); no bug tickets needed
- [x] report: `docs/bench/random-sf1.md`, `random-sf1-shuffled.md`, `random-sf10.md`; the 22 on shuffled tables: `docs/bench/m2p-engine-shuffled.md` (both alongside `m2p-engine.md`, all on commit dc88dca, a loaded machine: median load 7-8 from other projects)
- [x] fixes it found: native top-k for ORDER BY ... LIMIT (a random top-N took 543 ms, now 14), native filter for WHERE-only plans, and a harness bug: DuckDB imported pyarrow inside its first query (~70 ms), which had inflated every DuckDB time
- findings (honest): vs Polars, MXFrame is faster on most random queries (geo. mean 1.0-2.7x) and all 21 verified TPC-H queries at SF10. vs DuckDB, once its import is staged: DuckDB is faster on most random queries (MXFrame 0.63-0.91x geo. mean) and on 13 of 22 TPC-H queries at SF10 (MXFrame's 22-query total is still lower, 3.27 s vs 5.31 s, from Q21 / Q9 / Q13 / Q1); on shuffled tables DuckDB's SF10 total is lower (5.78 s vs 8.64 s: Q21 4.98 s vs 2.88, Q9, Q18) -> F-046

## F-045 ORDER BY ties outside the native executor [done]
Needs: none · Found by the F-043 review
- [x] plans the native executor declines (e.g. date + dictionary-string group keys, or `MXFRAME_NATIVE=0`) break ORDER BY ties in a different order than the native path's key order and than DuckDB; make the fallback's tie-break key order too (matters for F-044's DuckDB checks). Native order: ORDER BY columns, then every group key ascending (strings by bytes). `GraphCompiler._key_tiebreak` appends the keys to a Sort directly above an Aggregate for every non-native path (fused, kernels, GPU, null-key, base compiler); `tests/test_orderby_ties.py`
  Open: unaliased derived keys get no tie-break; with a string key added, sort + LIMIT on fallback paths can't use the native top-k (numbers / dates only), so it ranks every row (correct, slower)

## F-059 ORDER BY ties with a Project between Sort and Aggregate [done]
Needs: F-045
- [x] `groupby(...).agg(...).with_columns(y=2*x).sort(y, desc).limit(10)` differs from DuckDB on both native settings (engine-wide; SQL is not affected because the translator puts the Sort directly on the Aggregate): the tie-break keys must reach a Sort that sits above a Project over the Aggregate. Done in `GraphCompiler._tiebreak_names` / `_key_tiebreak` (follows keys by name/alias through each Project) and an early call in `CustomOpsCompiler` before the materialize barrier, so the native path gets the extended sort too (no Mojo change). A Project that drops a key leaves nothing to tie-break on for that key: the remaining keys still break ties, and ties on the dropped key stay in input order (documented limit); Filter nodes between Sort and Aggregate are walked too. `tests/test_orderby_ties_project.py`

## F-061 Integer overflow past int64 is silent on the exact path [done]
Needs: F-037 · Found by the F-037 review (2026-10-16); high priority: silent wrong results, and F-037 routes more queries here
- [x] `pc.add` / `subtract` / `multiply` in the exact path (`custom_ops.py` ~620) wrap past 2^63 (`a*a*a` with a = 2^30 gives 0, rows dropped); DuckDB raises "Overflow in multiplication of INT64". Use Arrow's `_checked` kernels and raise the same way
- [x] `pc.sum` on int64 and the limb recombine in `np_arrow._int_group_sum` wrap at 2^63 (DuckDB sums int64 into a 128-bit integer): raise, or widen like DuckDB
- [x] tests vs DuckDB for each, both native settings
- Built: `np_arrow.checked_arith` (Arrow `*_checked` kernels, OverflowError "Overflow in multiplication of INT64") at every exact-path `+ - *` (`custom_ops`, `compiler._eval_predicate`); `np_arrow.int_sum` / `_int_group_sum` raise "Overflow in sum of INT64" (a total past int64 raises rather than widening: the SUM column stays int64). Found on the way: when every pipeline declined, `max(a*a*a)` ran in the compiled graph and wrapped to 0. `_aggregate_with_nulls(late=True)` now runs just before the compiled graph and the global shortcuts: `int_sum_inexact(..., preds)` bounds each predicate and aggregate on its own (an uncompilable one cannot hide another; an integer `+ - *` the compiler cannot take counts as unbounded) and sends it to the Arrow path. Group keys, `with_columns` / `select` and sorts on a projection were checked and found to run in Arrow (checked kernels), so they raise too (tests added, each fails on the pre-F-061 code). Known difference: int32 * int32 past int32 but within int64 returns the exact value natively, where DuckDB raises an INT32 overflow. Native declines past 2^53 (F-037). `tests/test_int_overflow.py` (grouped, global, after a join, nulls, GPU).

## F-062 Native merge for partner-row group keys (fused join + aggregate) [todo]
Needs: F-046a slice 3 · From its review (2026-10-16)
- [ ] group keys that are columns of one partner but not its unique key (TPC-H Q10's shape) leave several partner rows per key tuple; slice 3 declines that case (the first version merged them in NumPy, which was per-row Python work and wrong for negative key values). Merge natively in the executor: code the key tuples per partner row (negative and repeated values), then fold the partner-row partials into key-tuple groups
- [ ] tests vs DuckDB with negative and repeated key tuples, asserting the fused path; Q10 no slower fused than unfused on a quiet machine

## F-063 Q17 on device="auto" aborts the process under load [done]
Needs: none · Seen 2026-10-16 in a full-suite run (test_tpch_duckdb, Q17 on auto: exit -6, glibc `pthread_mutex_lock` assertion); passed on rerun. Same family as the rare -11 under load in F-020 / F-022. An abort can't be caught or fallen back from, so it is worse than a wrong answer
- [x] reproduced: `scripts/stress_auto_crash.py --runs 150 --queries 17` (fresh processes, SF1, auto, no extra load, machine load1 10-20): 2 of 150 failed, one `-11` and one `-6` (`tpp.c:83 __pthread_tpp_change_priority` assertion). A synthetic exit-during-warm-up loop gave 2 of 120 and 2 of 150
- [x] core dump (systemd-coredump, read with gdb): crashing thread is `mxframe-gpu-warmup` inside `cuDevicePrimaryCtxRetain` (libcuda, reached by ctypes from the CUDA init); the main thread is in libc `exit()` running static destructors (Arrow thread pool dispose, libcuda teardown). So the process exits while the daemon warm-up thread is still inside the driver; interpreter shutdown does not stop a thread in native code, and `exit()` destroys libcuda's mutexes under it (-6 when a destroyed mutex is locked, -11 otherwise). Not the Mojo runtime, not native executor workers (all idle in pthread_cond_wait). Q17 hits it because it chains two computes, so the 0.1 s idle gap that starts the warm-up lands just before the script ends
- [x] fix: `auto_mode._close_background_gpu_work` (atexit) starts no more background GPU work and joins the warm-up thread, `coproc_fused.close_background` joins resident-upload threads (`_start_thread` refuses to start any once exiting), and the `mxframe-warm` CPU thread. A short script that exits before the warm-up begins waits for nothing; one that exits mid-init now waits for it (up to ~0.4 s, cap 15 s)
- [x] tests (`tests/test_f063_exit_drain.py`): two deterministic ones (exit waits for an in-flight warm-up; none starts once exiting), both fail before the fix; `MXFRAME_STRESS=1` opt-in test, 300 real warm-ups exited at random moments 0.28-0.42 s into CUDA init: 0 failures after the fix (about 1.3% failed before, so 300 runs would show ~4). After the fix `stress_auto_crash.py --runs 200 --queries 17,11,22,1 --load 2`: 200 of 200 exit 0
- [ ] reviewer: a full-suite run on a loaded machine should show no more -6 / -11 in test_tpch_duckdb auto cases

## F-060 Exact integers without a pre-pass; integer SUM is int64 everywhere [todo]
Needs: F-037, F-046a slice 3 merged (same Mojo files) · From the F-037 review
- [ ] replace F-037's min/max pre-pass (int32*int32, int64*int64 and every int64 SUM read the data once per query) with a runtime overflow flag in the bytecode interpreter: compute, flag any |result| past 2^53 per morsel, and rerun on the exact path only when flagged
- [ ] integer SUM returns int64 on every path regardless of size or nulls (today: native double when it fits, int64 on the exact path), like DuckDB
- [ ] `mean` of large int64 values: integer accumulation, not float64
- [ ] integer `min` / `max` on the exact path return float64 (lossy past 2^53): return int64 (found by F-061)
- [ ] `pipeline_program._int_arith` treats an uncompilable expression as unbounded only when all its columns are integers; one mixing a string or float column with an overflowing int product might still reach the compiled graph (no repro found by the F-061 review); the runtime overflow flag should close it
- [ ] sums whose expression the compiler can't take keep the old float path today (`int_sum_inexact` returns False): cover them

## F-046 Close the CPU gaps to DuckDB (next release) [doing]
Needs: none · Agreed 2026-10-13 (Abhishek: for the next release, on the strong baseline). General fixes only; every slice measured on TPC-H (stored and shuffled) and the random suite, and must not slow any other query. MojoFrame (illinoisdata/MojoFrame, arXiv 2505.04080) checked: its techniques (integer ids for low-cardinality strings, tuple-hash group-by, factorize-then-join) are already covered here; the slices below follow the database literature instead
- [ ] F-046a joins: profile the random joins (0.64-0.86x DuckDB) and TPC-H Q3 / Q5 / Q8 / Q12 / Q17 / Q22; a partitioned parallel hash join in Mojo for build sides that are not direct-addressable and for the leaf-by-leaf fallback
- [x] F-046a slice 3 join + aggregate fusion: `Aggregate([Filter|Project]* inner-join tree)` plans that took a star scan (`mx_filter` writing every joined row) plus `mx_execute` on the joined table run as ONE `mx_execute` v3 call (`join_engine.fused_join_aggregate` -> `native_exec.try_execute_join`): per 1024-row block the fact leaf's own filter, the partner lookups (existing `_build_map` maps), the aggregate program's predicate over the survivors (cross-partner residuals such as Q5's c_nationkey = s_nationkey, filters above the join) with the partner columns gathered into block scratch, then the aggregate; no joined row is written. Group keys: fact columns as before; partner int columns, partner string columns (ranked in byte order by a dictionary-sized table, turned back to strings) and derived expressions are expression keys (derived ranges from one extra fused min / max pass). Declines to the old path with the same answer: hashed / compacted key ranges (> 2^20 groups), duplicate partner keys, int64 past 2^53 (fact predicate, fact or partner column), unsupported expressions; `MXFRAME_FUSED_JOIN_AGG=0` turns it off. Group keys past direct addressing (`K_PROW`): when every key is a column of ONE partner (or the fact column equal to its join key) AND the keys include the partner's unique join key, the group id is the partner's ROW (a synthetic partner column the front stage gathers) and the key columns are taken from the partner at those rows (Q3's shape); HAVING, ORDER BY and LIMIT run on that small result (top-k first narrows by the first sort column, ties included, so only the survivors' keys are materialized). Keys without the partner's unique key (several partner rows per key tuple, Q10's per-customer keys through orders) would need a merge of groups: declined for now (an earlier NumPy merge was removed: it mis-grouped negative key values and cost more than the old path), Q10 runs on the old path; a native merge is a separate ticket. A partner larger than 2^20 rows declines. Integer expression keys over partner columns only (Q8's year) are bounded by interval arithmetic instead of a second pass. Cheap checks (switches, device, library) run before any partner is built; a declined fused call hands its built partners to the old path (no second build); a join result the old path has cached is not asked again; accumulation is shared (atomic) for many groups, so a large key space with few survivors (Q11) does not zero and merge a partial per worker. SF1 vs DuckDB, all equal: Q3, Q5, Q8, Q11, Q12, Q14, Q19 fused, Q10 old path; random suite: all equal. `tests/test_fused_join_agg.py`. Open: the block lookup loop duplicates mx_filter's (unify after slice 2 changes the maps); a partner of more than 2^20 rows (SF10 Q3) still declines
- [x] F-046g SELECT DISTINCT on dictionary columns runs on unified codes, then decodes the distinct rows: random SF10 q30 375 -> 43 ms (DuckDB 27), SF1 q94 117 -> 11.5 ms (DuckDB 13)
- [ ] F-046c zone maps: per-chunk min / max to skip chunks a filter cannot match (deferred: TPC-H's and the random suite's filter columns aren't clustered, so the benchmarks couldn't show it; worth it for time-ordered data)
- [x] F-046d dictionary strings pass through joins as codes when the plan reads them above the join only in dictionary-native ways (bare group key, native predicates): the native join scan writes int32 codes of one unified dictionary (per-chunk remaps), the group-by uses them as they are; columns that reach the output are decoded as before. Random SF10 joins 387 -> 217 ms and 1597 -> 639 ms. `tests/test_dict_through_join.py`
- [x] F-046f single-table filters: for `[Sort/Limit/Distinct]* [Project] Filter+ Scan` plans, dictionary columns are no longer decoded for the whole table first; the native filter decodes only the rows that pass. Open: DISTINCT under an aggregate (q68's count-distinct shape) still decodes first
- [x] F-046b COUNT(DISTINCT) on unsorted keys: a radix-partitioned hash aggregate in `mx_count_distinct` (partitions of about 64k consecutive keys, pairs deduped and counted per key in cache, no atomics, output in key order; key ranges past max(2^26, 8 x rows) still fall back). Shuffled SF10 Q21 4.98 s -> 1.47 s (DuckDB 2.88). Costs 16 bytes of scratch per row (24 with string values); a key range holding most rows is one partition, so one worker
- [x] F-046e top-k fused with its filter: `mx_filter_topk` runs the predicate per block and offers passing rows to per-worker heaps (no filtered table); random SF10 WHERE + ORDER BY + LIMIT 420 -> 63 ms (DuckDB 50), 196 -> 63, 123 -> 76. Open: heaps hold int and float copies of every sort value (nt x k x S x 16 bytes)
- [x] numbers after F-046b/d/e/f/g (`docs/bench/m2q-engine*.md`, `random-*-m2q.md`, commit 1313971; median load 4-6): TPC-H SF10 MXFrame 3.20 s, DuckDB 5.11, Polars 14.24 (without Q21: 2.45 vs 2.30); shuffled SF10 MXFrame 4.99 s (was 8.64), DuckDB 5.52, Polars 15.24; vs DuckDB on random queries (geometric mean): SF1 0.63 -> 0.76x, SF1 shuffled 0.91 -> 1.02x, SF10 0.75 -> 0.96x; joins at SF10 0.64 -> 0.98x, top-N 0.41 -> 0.93x; still behind: COUNT(DISTINCT) 0.35-0.48x, SELECT DISTINCT 0.53-0.82x; all 250 MXFrame results equal DuckDB
- [x] F-046h COUNT(DISTINCT b) per a without decoding or one-worker stalls: on device="cpu" both columns stay dictionaries (single table, native filter, join output) and the native count reads int32 codes of one unified dictionary (a dictionary key is decoded and value-sorted after); a small (key range) x (value range) is counted in a byte map; fewer key partitions (or sorted runs) than 2 x workers are partitioned by the pair's hash instead. Random suite vs DuckDB, COUNT(DISTINCT) shape (geo. mean): SF1 0.43 -> 0.86x, SF1 shuffled 0.48 -> 1.02x, SF10 0.35 -> 0.94x; all queries SF1 0.76 -> 0.81x, SF1 shuffled 1.02 -> 1.19x, SF10 0.96 -> 1.02x (`docs/bench/random-*-m2r.md`, build before the review's small fixes; median load 3-5); all 250 MXFrame results equal DuckDB. `tests/test_count_distinct_codes.py`. Open: device="auto" still decodes the counted column
- [ ] remaining (next release): join + aggregate fusion (no joined rows written), partitioned hash join for TPC-H Q3 / Q5 / Q8 / Q11 / Q12 / Q17 / Q22 (0.44-0.69x DuckDB), SELECT DISTINCT (0.57-0.99x)
- [x] release numbers on one build, quiet machine (`docs/bench/m2s-engine*.md`, `random-*-m2s.md`; shuffled + random 2026-10-14 01:21-01:50 after load < 2 for 5 min; stored rows rerun 02:05 after load < 1 for 5 min, the first pass overlapped a snapper cleanup and was discarded): TPC-H SF10 MXFrame 3.15 s, DuckDB 5.08, Polars 14.40 (without Q21: 2.41 vs 2.27); shuffled SF10 4.97 / 5.55 / 15.06 (without Q21: 3.63 vs 2.79); SF1 0.47 / 0.63 / 1.31, shuffled 0.61 / 0.66 / 1.41; random vs DuckDB (geo. mean) SF1 0.83x, SF1 shuffled 1.21x, SF10 1.09x; all MXFrame results equal DuckDB. README, optimizations.md, RELEASE_NOTES_0.6.0.md updated

## Docs overhaul (brief: docs/briefs/docs-overhaul.md, approved 2026-10-14) · Branch: `docs-overhaul`

Shared rules: plain HTML + JS + one shared CSS in `docs/site/` (house style, Terminal Ledger); every page opens with "what you're looking at, how to read it"; numbers come from `docs/bench/*.csv` / `*.json`, never typed by hand (README headline table excepted, checked by a test).

## F-047 Site skeleton and GitHub Pages deploy [done]
Needs: design (`docs/design/docs-site.md`) · Files: `docs/site/` (new), `.github/workflows/pages.yml` (new)
- [x] `docs/site/index.html`: shared nav and `site.css`, light and dark, totals computed from the bench CSV; the page cards (thumbnail, one line on what each shows) are added as each page lands (F-049..F-052)
- [x] a Pages workflow deploys `docs/site/` on push to `main`; the bench CSV/JSON files are copied in at build
- [x] Pages on (2026-10-14, via `gh api`, build type workflow): https://abhisheksreesaila.github.io/mxframe/

## F-048 Simple README with API tables [done]
Needs: F-047 (links) · Files: `README.md`, `tests/test_readme_api.py` (new)
- [x] under ~120 lines: why (3 bullets), latest release (0.6.0 SF10 table + DuckDB caveat), install, a code + output "screenshot" (query, result, timing), API reference tables (LazyFrame methods, expressions, aggregations, SQL, devices), links to the pages; the long history moves to the pages / `optimizations.md` / `benchmarks.md`
- [x] a test runs every code sample and API call shown in the README; the headline numbers match `docs/bench/m2s-engine*.csv`

## F-049 Interactive benchmarks page [done]
Needs: F-047 · Files: `docs/site/benchmarks.html`, `docs/site/js/bench.js`
- [x] toggles: SF1 / SF10, stored / shuffled, engines; per-query bars and totals with and without Q21; random-query shapes vs DuckDB; correctness verdicts; run conditions (load, date, build)
- [x] reads the CSV/JSON in `docs/bench/` at runtime; tests replay bench.js under node against Python readings of the same files (typed run-conditions prose is not machine-checked)

## F-050 Behind the scenes (for porting) page [done]
Needs: F-047 · Files: `docs/site/internals.html`
- [x] the path a query takes, LazyFrame → plan → optimizer → native Mojo executor → fused pipeline → AOT kernels, as a clickable diagram; each layer shows its files, its seam (ctypes, Arrow C Data Interface), when it declines and who answers next, and the env switch (`MXFRAME_NATIVE`, `MXFRAME_FUSED_PIPELINE`)
- [x] "add an operator" walkthrough with the real file and function names; every path named on the page exists (checked by a test)

## F-051 How it got fast page [done]
Needs: F-047 · Files: `docs/site/how-it-got-fast.html` · Absorbs the open items of F-039
- [x] one card per optimization from `optimizations.md` (what was slow, what changed, before / after numbers, Mojo feature or plain engineering), the progress chart, per-query wins over time

## F-052 How device="auto" decides page [done]
Needs: F-047 · Files: `docs/site/auto-mode.html` (from `docs/auto-mode.html`)
- [x] restyled on `site.css`, "how to read this" header, numbers from the bench files

## F-053 device="auto" never uses the native executor [review]
Found 2026-10-14 while documenting auto (F-052); the review showed it holds on every machine. On every CPU decision `auto` sets a co-processing policy (`lazy_frame.py` ~299, whatever the GPU state; `MXFRAME_COPROC=0` doesn't help) and `native_exec.try_execute` declines ("CPU+GPU co-processing policy active"), so the default `.compute()` runs the older fused pipeline; every 0.6.0 number is `device="cpu"`. SF10 group-by probe: cpu `path=native`, auto `path=fused_pipeline_cpu`. auto's 22/22 and 21/22 results are from 2026-10-12, before the native executor; the cost model prices the CPU with the older kernels.
- [ ] re-run the user-felt TPC-H scoreboard with `mx-auto` next to `mx-cpu` on 0.6.0 (quiet machine)
- [x] auto sets the co-processing policy only when `auto_mode._cpu_path` predicts a CPU+GPU split beats the native CPU executor by 10% (or the caller passes `gpu_fraction`); every other CPU decision runs the native executor exactly as `device="cpu"`; `MXFRAME_COPROC=0` honoured. Split is priced on source rows (what the pipeline and upload read) and only predicted for shapes the split can take (no join below the aggregate, bare group keys; `coproc_fused.stable_group_keys` is shared with `pipeline_exec`). Decision: the native executor does NOT take the CPU share of a split (big change in native_exec, which another ticket is editing); a split stays on the older pipeline and is chosen only when it beats `pipeline@cpu * NATIVE_CPU_FACTOR`. Factor is 1.0: one warm Q1-shaped 6M-row measurement on a busy machine (load 12) gave native 7.7 ms vs fused 6.9 ms, so no native speed-up is assumed. Refresh it, and add a real native calibration probe, in the quiet-machine run. `explain(device="auto", costs=True)` prints `cpu path: ...` and why. Tests: `tests/test_f053_auto_native.py`
- [ ] README / site drop the "pass device=\"cpu\" for now" note once auto is within 10% of min(cpu, gpu) again
  - note: the README, site, auto-mode.md and CHANGELOG were already changed to describe the current behaviour honestly (auto runs the native executor on a CPU decision; no speed claim), pending the quiet-machine check above
- [ ] TODO quiet machine: `auto` within 10% of min(cpu, gpu) on >= 20/22 queries at SF1 and SF10 (median of fresh-process runs)

## F-054 Finish the docs [done]
Asked 2026-10-14 ("add license, finish docs, hold off on new dev")
- [x] `LICENSE` (Apache-2.0, official text), in the sdist via `MANIFEST.in`, linked from the README
- [x] `docs/api.md` rewritten against the code: constructors, joins, `compute()` arguments, inspection, SQL limits, devices, source layout
- [x] `CONTRIBUTING.md`: the three CPU layers, every compiled library and its build command, `pixi run test`, adding an operator, benchmarking
- [x] `docs/auto-mode.md` links the site page and says its numbers predate 0.6.0 (F-053); `docs/benchmarks.md` headline is 0.6.0; `docs/vision-and-architecture.md` points to the current architecture
- [x] kernel visualizer published on Pages and linked from the README and the site nav; CHANGELOG [Unreleased]

## F-039 The optimization story: README with visuals [done] (open items closed by F-051, the how-it-got-fast page)
Needs: F-032 (numbers from its final benchmark) · Asked by Abhishek 2026-10-13
- [x] draft done 2026-10-13 (review by Abhishek pending): `docs/optimizations.md`, `docs/img/{progress,per-query-sf10,per-query-sf1}.svg` from `scripts/plot_progress.py` (dataviz palette validated, light and dark), README section "How the CPU engine got fast"
- [ ] `docs/optimizations.md`: one section per optimization (M0 executor ... M2h distinct and later): what was slow and why, what the engine does now, a diagram (mermaid) of before / after, the before / after numbers from `docs/bench/`, and which Mojo feature it relies on (compile-time specialization of the keyed / plain pipeline, closures inlined into `parallelize`, SIMD widths from `simd_width_of`, AOT shared libraries over the Arrow C Data Interface) or that it is plain engineering (no `pyarrow.compute` import)
- [ ] a progress chart: SF10 / SF1 engine-only totals per milestone against Polars and DuckDB, and per-query wins over time
- [ ] README: a short "How MXFrame got fast" section linking to it, with the progress chart and the current scoreboard; claims only where a benchmark backs them

## F-009 Scoreboard report and README [done]
Handoff (2026-09-30): all engine work up to F-020 is merged on `auto-mode` (1928 tests green). This ticket is the final run on a QUIET machine (ask Abhishek to pause other jobs; record load average).
- Run from the repo root: `pixi run python scripts/bench_user_felt.py --sf 1 --engines pandas,polars,duckdb,mx-cpu,mx-auto,mx-gpu --timeout 120 --gpu 1 --name final-sf1` then `--sf 10 --engines polars,duckdb,mx-cpu,mx-auto,mx-gpu,pandas --timeout 300 --gpu 1 --name final-sf10` (add `--resume` to continue an interrupted run; exact-decimal data `data/tpch_sf{1,10}_exact` is the default; `scripts/tpch_user/official.py` checks the DuckDB reference against published answers).
- Before trusting timings: `pixi run test -q` green; `uptime` load < 2; no other GPU users (`nvidia-smi`).
- Report: `docs/bench/final-*.md` (+ CSV): headline first-run ms per engine, wins vs Polars/pandas/DuckDB for mx-auto, correctness verdicts, repeat-run column, honest losses with causes (Parquet decode on PyArrow — F-021; Q21 — F-022). Then the README `device="auto"` section with the real numbers and the co-processing result (repeat queries on resident data ~1.35x CPU-only; SF10 Q1 fused stage CPU 57 ms / GPU-resident 39 ms / split 25 ms).
Needs: F-001, F-005..F-008
- [x] `auto` is within 10% of min(cpu, gpu) on ≥ 20/22 queries at SF1 and SF10 (moved from F-005): SF10 22/22; SF1 21/22 on the median of 4 fresh-process runs (single runs 17–21/22, only 2 of 4 reach 20; the misses move between runs, ±10% noise on 50–400 ms queries; the 3 extra runs started at load 2.8–4.4). Reaching it needed three fixes for auto's first-query overhead: a cheaper `_key_card` sample, lazy `calibration` imports, and background GPU work that waits for a 100 ms idle gap (before them SF1 was 12/22)
- [x] full SF1 + SF10 user-felt run on an otherwise idle machine (2026-10-12 01:34, load1 0.13 at start, GPU 0 because GPU 1 drives the display): `docs/bench/final-sf1*.md`, `final-sf10.md`; summary and honest losses at the top of `docs/benchmarks.md`. All 132 mx results match DuckDB
- [x] README `device="auto"` section with the real numbers; the README performance section now leads with the user-felt table, and the old synthetic warm-run tables are marked historical
