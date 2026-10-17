# Design: streamed execution over row-group batches (F-056 step 2)

Status: **building, 2026-10-16** · Ticket: F-056 step 2 · Builds on step 1 (`gpu_budget.py`), F-017 (row-group
pruning, pushed filters), F-057 (source tokens, residency), F-064/F-067 (reproducible mode)

Goal: a query over a `scan_parquet` source bigger than GPU memory **and bigger than RAM** finishes with the
result of `device="cpu"` on the same data (and DuckDB's), on `cpu`, `gpu` and `auto`, with memory bounded by
**one batch + the group state**, never by the file. "96 GB on a 48 GB GPU box with 62 GB of RAM must not say
out of memory."

## 1. The idea in one paragraph

When the scan's working set is bigger than the host budget, `compute()` does not read the file. It cuts the
file's (pruned) row groups into **batches** that fit the budget and runs the *same plan* once per batch, with
the streamed `ParquetScan` reading only that batch's row groups. The plan it runs is the plan's lowest
`Aggregate` turned into a **partial** aggregate (sum -> sum, count -> count, min -> min, max -> max,
mean -> sum + count). The partial results (one small table per batch) are merged by another aggregate over
the concatenated partials. What sat above the original aggregate (HAVING filter, project, sort, limit) then
runs on the merged table, through the normal `compute()`.

Nothing new runs on the data: each batch is an ordinary `compute(device)`, so filter pushdown, joins with
resident small tables, the native executor, the fused GPU pipeline, `gpu_budget`'s pre-check / OOM halving /
CPU rerun and auto's cost decision all apply per batch. "GPU when the batch fits the per-GPU budget, else the
CPU native executor" is `run_piece` (step 1) deciding per batch.

## 2. What streams, what declines

A plan streams when, after the optimizer:

1. it holds a lowest `Aggregate` **A** above the biggest `ParquetScan` leaf **L** (a single file);
2. between A and L there are only `Filter`, `Project` (no window / rank / lag / cum_sum) and `Join`s, with L
   on the probe side: either side of an inner join, the left of a left / semi / anti join. No `Sort`, `Limit`,
   `Distinct`, `Tail`, nested `Aggregate` (count distinct is a distinct below an aggregate: it declines);
3. every aggregate of A is a top-level `sum / mean / count / min / max` (arguments may be any expression
   without an aggregate in it), every group key is a column or an aliased expression;
4. L is read once in the plan (not a self-join of the big file);
5. the **other** leaves under A (the dimension tables: the build sides) estimate, together and times 1.5 (columns plus the join index built on them), under half the budget. They are read once, before the first batch, and stay resident for all batches.

Anything else runs the way it does today (no behaviour change when the data fits). When the data does not fit
**and** the plan declines, `compute()` warns once (`RuntimeWarning`, reason named) and runs as before, because
the alternative is a wrong answer. The declines are the step 3 list (section 8).

When does it stream: `estimate(A's leaves) * WORKING_FACTOR(3) > host budget`. The estimate is the decoded size
of the columns the plan reads over the row groups that survive pruning, from the Parquet footer (fixed-width
columns: rows x width; strings: uncompressed bytes + 4 per row). The factor covers the table, the filtered
copy and the group ids / gather buffers of the in-memory path.

## 3. Budgets

- **Host budget** (`MXFRAME_HOST_MEM_BUDGET`, "64M" / "2G": a test knob like `MXFRAME_GPU_MEM_BUDGET`):
  by default 70% of `MemAvailable` from `/proc/meminfo`. In **reproducible mode** it is 50% of `MemTotal`
  instead, a number that does not move with what other processes hold, so batch boundaries are a function of
  the file, the query and the machine, not of the moment.
- **GPU budget**: step 1's `gpu_budget.free_bytes()` (driver free memory, capped by the fake budget). On
  `gpu` / `auto` with a CUDA device the batch size is also capped at what one GPU piece may use
  (90% of free / 1.3), so a batch can run on the GPU. Reproducible mode ignores this cap (same reason).
- **Batch size** = `(budget / 2 - 1.5 * dimension bytes) / WORKING_FACTOR` (half of the budget is kept for the
  group state, the partials waiting to merge, and the merge's own copy), at least one row group. Row groups are
  added in file order until the next would pass that size.
- **Group state** (the merged partial table) over half the budget: raises `MemoryError` that names the number
  of groups, the size and "needs hash-partitioned groups (F-056 step 3)". Failing loudly and early beats the
  kernel's OOM killer. This is the one piece declined at run time.

## 4. Merge

Partial columns get hidden names (`__p0`, `__p1s`, `__p1c`); group keys keep their output names. Partials are
decoded (dictionary keys become plain strings, whose dictionaries differ per batch), empty ones are dropped
(a keyed aggregate of zero rows holds no groups; for a global aggregate the first one is kept so the result
still has its one row), and queued. The queue is merged when its rows pass `max(1M, merged state rows)`, so
the work is amortized O(total partial rows), then once more at the end. A merge is an ordinary `Aggregate`
plan over `Scan(concat(state, queue))` computed on the CPU (the partials are small):

| aggregate | partial (per batch) | merge | final |
|---|---|---|---|
| `sum(x)` | `sum(x)`; **cast to int64 when x is an integer expression** | `sum` | as merged |
| `count(x)` | `count(x)` | `sum`, cast int64 | |
| `min / max` | same | same | |
| `mean(x)` | `sum(x)` (int64 for integers), `count(x)` | `sum`, `sum` | `s / c` in float64, NULL where `c = 0` |

Why the cast: the engine returns a double for an integer sum it can prove below 2^53 and an int64 otherwise
(F-037 / F-060). Merging doubles could round past 2^53; merging int64 partials is exact, and the merge goes
through the same exactness gate. The type of x is read by a zero-row dry run of the plan's input (as
`LazyFrame.schema` does).

Rules that carry over, because the merge is the engine itself: NULL (an all-NULL group sums to NULL), NaN
(NaN is the largest value for min / max; a NaN sum is NaN), -0.0 / NaN key canonicalization, int64 sums
exact past 2^53 up to int64 overflow, the same overflow error.

**This is not `coproc_fused.Partials`.** That merge is the slot layout of the fused pipeline kernel, which has
no joins or projections around it. A streamed batch is a whole plan (joins to the resident dimensions,
expressions), so the partial is a table and the merge is an aggregate. (F-057's `range_partials` is the same
idea at a lower level: one batch's partial, merged with the others.) The two compose: a batch whose plan is a
fused-pipeline shape still runs through `Partials` inside its own compute, split between GPU and CPU.

**Reproducible mode** (`reproducible=True`): the batch plan is the same, each batch's float sums are the
engine's defined order (CPU), batches are merged **in file order** and the queue is merged in a fixed
order, with batch boundaries fixed as in section 3. Same file + same query + same machine budgets -> the same
bits. (A different budget is a different order of additions, and may differ in the last bits, as a different
`gpu_fraction` does.)

## 5. Reading a batch

The plan is not copied per batch: the streaming context (`io_parquet._stream`, a
`ContextVar` set around the batch computes) names the file and the row groups of the current batch.
`resolve_plan` reads that file through `_read_file(..., only=batch)`, which intersects the batch with the
groups the pushed filters leave (row-group pruning by statistics still happens per batch, and an emptied batch
reads nothing). Every other file in the plan is read once and cached in the context, so a dimension table is
decoded once, not per batch, and keeps one device copy (its token is unchanged). The token of a batch read
names its groups, so the GPU residency cache never mistakes two batches for one source.

Dictionary columns are decided per batch exactly as for a whole read. A batch that ends up with different
dictionaries is no problem: partials are decoded before they are queued.

## 6. Join to dimension tables (TPC-H shaped)

Q3 / Q5 / Q10-like plans are `Aggregate(Join(Join(lineitem, orders), customer))` with filters pushed to the
leaves. The biggest leaf (lineitem) streams; orders, customer, nation, ... are the resident side. Each batch's
compute is the existing join + aggregate path with the batch as the fact table (the star scan, or the fused
join + aggregate on the CPU, the leaf-by-leaf join on the GPU), so the build side is built from the
cached dimension tables per batch (the star scan indexes it in parallel; the cost is the F-046 join engine's
build, which the quiet-machine check below measures) while the fact table passes through once.
Q1 and Q6 are the join-free case. Sort / limit above (Q3, Q10 ORDER BY revenue DESC LIMIT) run on the merged
partials.

## 7. Tests (without 96 GB)

`MXFRAME_HOST_MEM_BUDGET` (new) and `MXFRAME_GPU_MEM_BUDGET` (step 1) make a ~200 MB file with many row groups
"bigger than memory". `tests/test_f056_streamed.py`:

- results equal DuckDB over the same file, on `cpu`, `gpu`, `auto`: Q1-shape (several aggregates, 4 groups,
  filter), Q6-shape (global), a fact x dims aggregate (Q3 / Q5-shape with ORDER BY / LIMIT above), a filter
  that prunes row groups, many groups (500k), NULLs, NaN, ints past 2^53, all batches empty, a batch emptied by
  the row filter, a key whose groups appear in some batches only;
- the path is asserted: `provenance["path"] == "streamed"`, batch count, the batches' devices, a declined
  shape warns and does not stream;
- memory is bounded: a subprocess reads the file through the streamed query and the Arrow pool's peak
  (`pa.default_memory_pool().max_memory()`) and `ru_maxrss` stay under the budget, while the same query
  without the budget reads the whole file (the contrast proves the number means something);
- reproducible mode: two runs are bit-identical on a float sum.

On `main` these fail: the budget variable is ignored, nothing streams, and the peak is the file.

## 8. Step 3 (out of scope here, written into the plan)

1. **Group state over the budget**: hash-partition the groups (filter by `hash(key) % P`, or range
   partition by key, one pass per partition) so each pass's state fits.
2. **Big-to-big joins**: partition both sides by the join key so each pair fits (Li et al. 2025;
   `docs/research/gpu-out-of-core.md`). A dimension over half the budget declines today.
3. **Sorts larger than memory**: sorted runs per batch + k-way merge; a streamed top-k is already free (a
   batch's top-k merges as a partial) but is not built: Sort / Limit below an aggregate declines, and a plan
   with no aggregate (pure filter / sort of a big file) is not streamed.
4. **The second GPU as a separate worker** taking alternate batches (own context, tracker, budget).
5. **A row group bigger than the budget** (one batch is at least one group): read it in column chunks /
   `iter_batches`.
6. **count distinct** and other non-decomposable aggregates (std, median): need the group partition of 1.
7. **Prefetch**: read batch i+1 while batch i computes.
8. Directory datasets (several files): each file is a natural batch; not wired.
9. Quiet-machine checks: the overhead of streaming vs in-memory when the data fits (force the budget on a
   file that fits, SF1 Q1 / Q6 / Q3), and a real 30+ GB run.

## 9. Result types follow the engine, not the batching

Partial integer sums are merged as int64; the *result* type of an integer sum is decided once, at the end, the way
the engine decides it for the whole table. In memory that decision looks at the table below the Filter chain: its
row count, the bound of each sum's argument (`col_bound` of the columns combined through the expression) and
whether Arrow's exact path has to answer (NULLs in a column the plan reads, dates, values past 2^53). Per batch the
compiler reports exactly those inputs to `pipeline_program.SumProbe` (a context variable set around the batches);
the probe adds the rows, takes the maximum of each column bound, ORs the flags, and `SumProbe.inexact()` asks the
engine's own `_Builder.within_exact` about the totals. int64 if inexact or Arrow answered, else a double. The
read-time row filter (`io_parquet._read_file`, which changes the rows below the Filter) is decided once for the whole
file and applied to every batch. min / max take the type of the original aggregate's zero-row schema. Known gap: the
probe sees the engine's inputs only where the batch reaches `_arrow_aggregate_reason` / `_compile`; a plan that
never does (a path that answers sums without either) gets a double.
