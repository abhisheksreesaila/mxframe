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
- Tests: `pixi run pytest tests` (new tests go in `tests/`) plus `pixi run test-all` (legacy scripts) must pass.
- TPC-H data (gitignored) lives at `/home/abhisheksreesaila/Projects/mxframe/data/tpch_sf{1,10}`; worktrees read it from there.
- Use glossary terms from the spec: `Morsel`, `DeviceCross`, `Router`, `ResidencyTracker`, `Calibration`.

## Wave 1: foundations (parallel)

## F-001 User-felt TPC-H scoreboard [todo]
Needs: none · Files: `scripts/bench_user_felt.py`, `scripts/tpch_user/` (new)
- [ ] each (engine, query, scale) runs in a fresh subprocess; the clock covers Parquet read + query + materialized result; `import` time is recorded separately
- [ ] engines: pandas, Polars (lazy `scan_parquet`), DuckDB (reference), MXFrame `cpu`, `gpu`, `auto`
- [ ] all 22 queries at SF1 and SF10; every engine's result is checked against DuckDB (row count + numeric tolerance) and mismatches are shown, not hidden
- [ ] a secondary "repeat run in same process" number is recorded
- [ ] writes `docs/bench/user-felt-<date>.md` and a CSV; a `--queries`/`--engines`/`--sf` CLI; per-run timeout so one hang can't stall the run

## F-002 Multi-core CPU kernels with float64 accumulation [todo]
Needs: none · Files: `kernels_aot/kernels_aot.mojo`, `aot_kernels.py`, rebuilt `libmxkernels_aot.so`
- [ ] grouped sum/min/max/mean/count and masked global aggs use all cores (`parallelize` over row morsels with per-thread partials, then merge)
- [ ] sums and means accumulate in float64 (SF10 Q1 matches DuckDB to 1e-9 relative)
- [ ] join build/probe and gather use multiple cores where the kernel is embarrassingly parallel
- [ ] a thread-count knob (env `MXFRAME_CPU_THREADS`, default all cores) that later tickets can use to split cores between roles
- [ ] microbench before/after at 6M and 60M rows recorded in the commit message

## F-003 Calibration store [todo]
Needs: none · Files: `calibration.py` (new), `tests/test_calibration.py`
- [ ] `mx.calibrate()` (quick, < 30 s) measures H2D/D2H bandwidth (pageable + pinned where available), launch overhead, and `time = α + c·n` per kernel per device
- [ ] saved to `~/.mxframe/calibration.json` (overridable via `MXFRAME_CALIBRATION`), keyed by device name + driver + MXFrame version, schema-versioned
- [ ] missing / corrupt / stale file → safe built-in defaults, never a crash; no GPU → CPU-only calibration
- [ ] `observe(op, device, rows, ms)` refines the fit online (HyPE-style)

## F-004 First-run cost: lazy GPU init, no JIT on hot paths [todo]
Needs: F-002 merged (both touch `aot_kernels.py`) · Files: `custom_ops.py`, `aot_kernels.py`
Why: fresh-process SF10 Q1 profile (2026-09-29): `device="cpu"` pays 0.25 s creating the CUDA context it never uses; `device="gpu"` pays **5.3 s** MAX Graph JIT compile for `group_composite` (18 `session.load` sites remain in `custom_ops.py`).
- [ ] `device="cpu"` never initialises the GPU; GPU context + AOT GPU library initialise lazily, and under `auto` speculatively in a background thread so a CPU-bound query never waits
- [ ] every operator the 22 TPC-H queries hit runs through AOT kernels on both devices: no MAX Graph compile in a fresh-process run (assert via a test that counts `session.load` calls)
- [ ] measured: fresh-process SF10 Q1 on `gpu` loses the 5 s compile; small `auto` query not worse than `cpu`

## F-010 `mx.scan_parquet` with pushdown and dictionary strings [todo]
Needs: none · Files: `io_parquet.py` (new), `lazy_frame.py` (source-resolution pass), group-key encoding in `custom_ops.py`
Why: Polars `scan_parquet` reads only needed columns; mxframe reading every column loses on I/O alone. SF10 Q1 spends 1.8 s dictionary-encoding string group keys that Parquet already stores dictionary-encoded.
- [ ] `mx.scan_parquet(path)` returns a lazy frame; at compute, only columns referenced anywhere in the plan are read, and simple comparisons directly above the scan are pushed into the reader (row-group skipping)
- [ ] low-cardinality string columns are read as Arrow dictionaries; group-key encoding uses the dictionary indices directly (no re-encode), unifying dictionaries across chunks
- [ ] results equal the `pq.read_table` path on all 22 queries; F-001's `mx_read` helper switches to it

## Wave 2: safe auto (Phase 1)

## F-005 Auto planner: whole-query choice, fallback, explain [doing]
Needs: F-003 · Files: `auto_mode.py` (new), `lazy_frame.py` (compute/explain hooks)
- [ ] cost model walks the plan, estimates rows per operator, and compares CPU-only against GPU (incl. copies and GPU-init if not ready); picks the lower
- [ ] GPU error/OOM → rerun on CPU, warn once per shape, remember the shape; `strict=True` raises
- [ ] `lf.explain(device="auto", costs=True)` prints decision, estimates per device, bytes crossing
- [ ] `compute(profile=True)` returns predicted vs actual and feeds `calibration.observe`
- [ ] no GPU present → everything CPU, zero errors; `auto` results equal `cpu` results on the test suite

## Wave 3: per-operator placement + co-processing (Phases 2–3)

## F-006 Router co-processing for grouped and global aggregates [doing]
Needs: F-002, F-003, F-005 · Files: `coproc.py` (new), hook in `custom_ops.py`
- [ ] large decomposable aggregations split into morsels pulled by CPU workers and a GPU worker from one queue; partial aggregates merged
- [ ] GPU worker overlaps copy of morsel k+1 with compute of morsel k (2–3 staging buffers)
- [ ] split ratio from calibration; `gpu_fraction=` overrides; results identical to `cpu` for 1, 2 and all threads
- [ ] SF10 Q1 cold under `auto` beats both `cpu` and `gpu` by ≥ 15%, or the ticket records the measured break-even and why

## F-007 Per-operator placement for joins (CPU filter → GPU join) [todo]
Needs: F-005 · Files: `auto_mode.py` placement pass, join dispatch in `custom_ops.py`
- [ ] placement pass assigns each operator a device; `DeviceCross` nodes shown in explain with bytes
- [ ] filters/projections before a crossing so only surviving rows and needed columns move
- [ ] join probe side can be split across CPU and GPU (Router) when both are worth it

## F-008 Fused pipeline kernels [todo]
Needs: F-002 · Why: SF10 Q1 spends ~1.3 s in PyArrow expression eval, NumPy mask indexing and dtype copies around a ~0.5 s kernel · Files: new fused kernels in `kernels_aot/`, dispatch in `custom_ops.py`
- [ ] filter → arithmetic expression → grouped aggregate in one pass per morsel on CPU (multi-core) and GPU, no intermediate NumPy arrays (Crystal)
- [ ] covers Q1, Q6, Q14, Q19-style shapes

## F-009 Scoreboard report and README [todo]
Needs: F-001, F-005..F-008
- [ ] full SF1 + SF10 user-felt run on an otherwise idle machine; results + honest losses in `docs/benchmarks.md`
- [ ] README `device="auto"` section with the real numbers
