# Product brief — MXFrame auto mode

Status: approved
<!-- Approved by Abhishek on 2026-09-29 ("Spec = approved brief"). Full design: docs/spec-auto-mode.md -->

## Why (the mission)
- Why this exists, in one sentence: with `device="auto"`, MXFrame uses whatever compute the machine has (CPU only, or CPU and GPU together) and returns the answer faster than pandas or Polars would.
- Why now, and why you: Mojo kernels already run on CPU and on NVIDIA/AMD/Apple GPUs from one source; no open-source dataframe library co-processes a single query on CPU and GPU at once.
- If it disappeared tomorrow: users would pick one device by hand and leave compute idle, or use Polars (CPU) or cudf (NVIDIA only).

## Who it's for
- First user: a Python data person who runs a query and feels how long it takes. They never warm anything up.
- Trigger: an analytical query (filter, join, group-by) over millions of rows on a laptop or workstation.
- Today: pandas (slow, single core) or Polars (fast, CPU only).

## What it does
- Core job: run a dataframe query on the best mix of CPU and GPU, automatically.
- v1: (1) calibrated whole-query CPU/GPU choice with fallback and `explain`; (2) per-operator placement (filter on CPU, join/aggregate on GPU); (3) co-processing, where CPU threads and the GPU split one query's rows and merge the results.
- Not in scope: out-of-core/spill (Phase 4), multi-GPU (2.x), distributed, renaming the package.

## How we'll know it worked
- The number: **user-felt latency**. Fresh Python process, data starting as Parquet on disk, clock around load + query + result. No warm-up. pandas and Polars (lazy `scan_parquet`) are measured the same way; DuckDB is a reference. Repeat-run numbers are secondary.
- Done for v1: on TPC-H SF1 and SF10, `auto` beats Polars and pandas on most of the 22 queries with results matching DuckDB. `auto` is never meaningfully slower than the better of `cpu`/`gpu`, and on large scans + group-bys it beats both by using both devices.

## Constraints and risks
- Hardware: 2× RTX 3090 (24 GB), 24 cores, 62 GB RAM. One GPU for auto mode; the second is only for parallel test/bench jobs.
- `device="cpu"` and `device="gpu"` keep working; changes sit behind `auto`.
- Riskiest assumption: that splitting work across PCIe pays off. Measure it with the break-even sweep before building on it.

## Decisions and trade-offs
- Spec `docs/spec-auto-mode.md` is the design source; this brief is the summary.
- Headline metric is first-run end-to-end, not hot, so GPU init and graph compile count. Pro: honest. Con: the GPU path must hide or avoid its startup cost.
- Work lands on branch `auto-mode`, local commits, no push.

## Open questions
- (deferred) Name change, auto as the 1.0 default, calibration file location on shared servers.
