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
- **Group state** (the merged partial table): a merge whose input passes a quarter of the budget spills to hash
  partitions on disk instead (section 10; step 2 raised `MemoryError` here).

## 4. Merge

Partial columns get hidden names (`__p0`, `__p1s`, `__p1c`); group keys keep their output names. Partials are
decoded (dictionary keys become plain strings, whose dictionaries differ per batch), empty ones are dropped
(a keyed aggregate of zero rows holds no groups; for a global aggregate the first one is kept so the result
still has its one row), and queued. The queue is merged when its rows pass `max(1M, merged state rows)`, so
the work is amortized O(total partial rows), then once more at the end. A merge is an ordinary `Aggregate`
plan over `Scan(concat(state, queue))` computed on the CPU (the partials are small):

| aggregate | partial (per batch) | merge | final |
|---|---|---|---|
| `sum(x)` | `sum(x)` (int64 when x is an integer expression, F-060) | `sum` | as merged |
| `count(x)` | `count(x)` | `sum`, cast int64 | |
| `min / max` | same | same | |
| `mean(x)` | `sum(x)` (int64 for integers), `count(x)` | `sum`, `sum` | `s / c` in float64, NULL where `c = 0` |

Why int64: an integer sum is int64 on every path (F-060), so a partial and the merged total have the same type
whatever the batching, and merging int64 partials is exact. The type of x is read by a zero-row dry run of the
plan's input (as `LazyFrame.schema` does).

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

## 8. Step 3 (out of scope here, written into the plan; items 1, 3 (top-k only) and 5 are built, section 10)

1. **Group state over the budget**: hash-partition the groups (filter by `hash(key) % P`, or range
   partition by key, one pass per partition) so each pass's state fits.
2. **Big-to-big joins**: partition both sides by the join key so each pair fits (Li et al. 2025;
   `docs/research/gpu-out-of-core.md`). Built in step 3b, section 11.
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

## 9. Result types follow the input type, not the batching

Partial integer sums are int64 and so is the result: the engine adds integers in int64 on every path and the type of
a sum depends on the type of its argument only (F-060). Before that, an integer sum was a double when an F-037 bound
(rows x largest value, from the table below the Filter chain) proved the total fits in 2^53 and int64 otherwise, so a
batch that saw fewer rows or smaller values than the whole table could answer another type; `pipeline_program.SumProbe`
(a context variable around the batches that merged each batch's row count, column bounds and exact-path flags) existed
only to give the batches the type the whole table gives, and the end of `_State.result` cast the merged int64 to a
double when the probe said the sum was small. All of that is gone: there is nothing to agree on. min / max take the type
of the original aggregate's zero-row schema. The read-time row filter (`io_parquet._read_file`, which changes the rows
below the Filter) is still decided once for the whole file and applied to every batch.


## 10. Step 3a: spilled groups, row queries, oversized row groups

**Group state bigger than the budget** (`spill.py`, `streamed._State`). The engine's group-by holds about four times
its input (measured: 5 MB merged in, 20 MB peak), so a merge whose input (`merged + pending`) times
`MERGE_FACTOR = 4` passes the budget does not run: the state so far and every later partial are written instead
to 16 Arrow IPC streams under `MXFRAME_SPILL_DIR` (default `~/.cache/mxframe/spill`, never `$TMPDIR`: tmpfs is RAM) in `mxframe-spill-<host>-<pid namespace>-<pid>-*` (only that host and namespace sweep it). A row goes to partition `hash(keys) >> 33 % 16`. The hash
(`spill.hash_keys`) is splitmix64 over the bits of integers and floats (canonicalised first: -0.0 equals 0.0, every
NaN equals every NaN, as the engine groups them, F-077), the CRC32 of the text of anything else, NULL a value of its
own, and a per-level salt; no Python `hash()`, so the partition of a key is the same in every process and
reproducible mode gets the same partitions and the same output order. After the last batch, `pieces()` reads one
partition at a time into a fresh `_State` (the same merge, the same type rule as step 2: hidden columns, int64
partials, int64 sums), splitting a partition again with salt + 1 when
even it passes the budget (at most 6 levels: past that one group is the problem and it raises `MemoryError`). A
group is in exactly one partition, so partition outputs just concatenate. Each partition file is deleted once read
and the directory is removed in a `finally`, so an error leaves nothing behind. Provenance:
`streamed.spill = {partitions, bytes, depth}`.

**Above the aggregate.** The HAVING filter and projections (row-local nodes) run on every partition's output; what
is then above (Sort + Limit, Limit) is a *sink* over the pieces: a running top-(n + offset) or an early stop. The
sink's `result()` is placed where the aggregate / row-local part was and the original plan above it (Sort, Limit,
OFFSET) runs on that, so ties and offsets are the engine's own. A result that is all the groups has to fit the
budget: the API hands back one in-memory table.

**Plans that return rows** (no aggregate above the big file). The row-local part of the plan above the file (Filter,
Project without window functions, Join with the file on the probe side) is the batch plan; the sink is chosen by
what sits right above it:

| above the row-local part | per batch | at the end |
|---|---|---|
| Sort then Limit (Projects may sit between) | `Limit(Sort(part), n + offset)`, merged into the running best | the original Sort / Limit / Project on the best rows |
| Limit | `Limit(part, n + offset - rows so far)`; stops at n + offset | the original Limit (OFFSET) |
| anything else, or nothing | `part` | the original plan on the concatenation |

Collected rows must fit the budget (budget / 3 when an operator above will work on them, since it needs about
three times its input); otherwise `MemoryError`: "the result of this query is bigger than memory: the 100,000 rows
produced so far are 3.1 MB, about 12.4 MB in all, and only 6.0 MB fits ..." (or "the rows that the Sort above the big
file needs are ..."). The total is an extrapolation from the share of the file read so far. A bare Filter over the
file selects only the columns the plan reads. Provenance: `streamed.rows = {kind, batches_run}`.

**A row group bigger than a batch** is read in slices of rows (`io_parquet._next_slice`): `iter_batches` over a
`ParquetFile` opened with a 64 KB read buffer. Without the buffer Arrow reads each column chunk whole, and
`iter_batches` on an unbuffered file decodes the whole row group before slicing it (measured: 5.0 MB peak for a
4.8 MB group; 1.6 MB with the buffer and 32 KB pages). What is held is the slice plus about one data page per
column, so the bound depends on the file's page size (the writer default is 1 MB). The slices are batches like any
other (own source token, so residency never mistakes two slices for one source); the plan's Filter node filters
exactly (the read-time row filter is not applied to a slice). Provenance: `streamed.slices`.


## 11. Step 3b: Grace hash join

**Which join.** `streamed._grace_at` walks the chain from the aggregate (or the root) down to the biggest leaf and takes
the topmost join where (a) the big file is the right side of a left / semi / anti join (a build side cannot stream),
(b) the other side's leaves, times 1.5, pass half the budget, or (c) the other side reads the same file (self-join).
The nodes above it must still be row-local (Filter, Project without window functions, joins whose other side is
resident), the tables of those joins are the *resident* part, and everything below it is a *side*. The pair target
(`Plan.batch_target`) is the batch formula again: (budget / 2 - 1.5 x resident) / 3, capped by the GPU budget on
`gpu` / `auto`.

**Sides to partitions** (`grace.Grace`). Each side is streamed as a row query (a side with row-wise work above a big
file runs batch by batch exactly like step 3a's rows queries; a side that is a join of big tables is a nested `Grace`;
anything else, an aggregate over a big file or a side that fits, is an ordinary compute, which streams by itself), cut
to the columns the rest of the plan reads, decoded, and written through `spill.Partitioned` into 16 partitions by
`hash_keys(keys, salt)`. A join key that meets a float key on the other side hashes as a double (`as_float`), so 1
and 1.0 co-locate; floats are canonicalised (-0.0 = 0.0, one NaN), dictionaries are decoded, NULL is a value of its
own that never matches in the join. The partition files record their rows and bytes.

**Pairs.** For partition i: no left rows, or no right rows for inner / semi, means nothing to emit (files dropped).
Otherwise the side that is built is held whole in memory and the other is read in chunks of at most
`target - build bytes`: the right side for left / semi / anti, the smaller for an inner join, when it is at most half
the target. Each chunk's plan is the batch plan (partial aggregate or the rows sink's plan) with the join replaced by
`Join(Scan(chunk), Scan(build))`, computed by the ordinary `compute(device)`: the GPU if the pair fits the GPU
budget, else the CPU. A pair where no side fits is written again with `salt + 1` (16 sub-partitions each) and its
pairs processed the same way, at most `spill.MAX_DEPTH` levels; past that, one key is repeated too often and a
`MemoryError` says so (no partitioning can split one key). Partials go through `_State.add` (step 2's merge, spilling
if the groups are many), rows through the 3a sinks, so top-k / LIMIT / HAVING / ORDER BY above run as before and a
`LIMIT` stops reading pairs. Pairs run in partition order and chunks in file order, which keeps reproducible mode
deterministic.

**NOT IN.** `x NOT IN (SELECT ...)` is an anti join plus a Filter on a scalar subquery about the list (empty / has a
NULL / no NULL). The scalar is resolved before streaming (it is an aggregate over the big side and streams itself);
a NULL in the list makes the filter constant false, which `_resolve_scalars` now turns into an empty table of the
same columns instead of `Limit(0)` over the join, so the join never runs.

**Limits.** Memory per pair is the build side plus one chunk plus the join's own copies (the engine holds about three
times its input); the output of a join with many matches per key is not bounded (a pair could produce more rows than
it reads; the partial aggregate above it is what keeps it small). The sides' partition files are written one side
after the other, then read pair by pair, so the disk needs about the size of the columns of both sides. A join of
three big tables nests (sides are materialised to disk once per level); there is no cost-based order.

**NULL keys (review fix).** A NULL key never matches, but every NULL hashes to one partition at every salt, which no split
can divide. So they are never partitioned: right rows with any NULL key are dropped for every join kind; left ones are
dropped for inner / semi, and for left / anti written to their own file and emitted after the right side is partitioned,
each chunk joined against an empty right table (NULL columns for left, the row as is for anti). The right side is
therefore partitioned first. A `MemoryError` for a key that cannot split names the most repeated key and its rows.
An integer with no exact double joined with a float key raises the in-memory `TypeError: join key types differ`.


## 12. Step 3c: sink, external sort, count distinct (F-113)

**Sink.** `LazyFrame.sink_parquet(path, device="cpu", compression="zstd", **compute_args)` runs the query with a
`streamed.ParquetOut` in a context variable (`streamed._out`). A streamed query hands its result to the file in pieces
instead of building the table: the batches of a rows query with nothing row-wise-unfinished above it (a filter, a
projection, a join to resident tables, the pairs of a Grace join), the hash partitions of a spilled aggregate (after
HAVING / projections), and the key ranges of an external sort (below). Small pieces are buffered to row groups of
`budget / 16` bytes (at least 256 KB, at most 32 MB). The file is written under `<path>.mxframe-partial-<pid>` and
renamed when the query is complete, so an error leaves no partial file and keeps an old one. A query that was not streamed
(it fits) is written from its table; an empty result writes a file with the columns. Plain `compute()` is unchanged and
its "bigger than memory" `MemoryError` names `sink_parquet`. Provenance: `sink = {path, rows, streamed}`. SQL:
`COPY (SELECT ...) TO 'f.parquet' [(FORMAT PARQUET)]` runs the query through `sink_parquet` at once (on the CPU) and returns
the one-row `Count` table, as DuckDB does.

**Sorts larger than memory** (`extsort.py`; ORDER BY with no LIMIT over a rows query or over spilled groups; Sort + LIMIT
is still the running top-k of 3a). Not a heap merge: the comparison of two rows is the engine's `Sort` (F-070 / F-084: NaN
largest, NULLs last, DESC, strings, several keys, expressions), and calling it per row would be slow and could disagree.

1. Each batch is cut into pieces of `(budget - held) / 32` bytes (the engine's sort peaks at 4-5x its input, measured
   with `pa.default_memory_pool().max_memory()`), each piece is sorted on the **CPU** (the GPU sort rounds floats to
   float32) and written as a sorted *run* with a hidden `__seq` column (run number x 2^40 + position), plus 64 sampled rows
   kept in memory. `(keys, __seq)` is a strict total order every run already follows; ties keep the order of the file.
2. After the last batch the sorted samples give splitter rows so that a key range holds about `(budget - held) / 16` bytes
   (at most 256 ranges). Each run is loaded alone, sorted together with the splitters (tagged so that a splitter sorts
   before an equal row) which says how many of its rows come before each splitter, and its slices are appended to the range
   files. Because the splitters are real rows with their `__seq`, a key repeated a million times still splits.
3. Range by range the slices are read, sorted (a merge of k sorted pieces) and yielded without `__seq`. The ranges in order
   are the sorted result: `compute()` concatenates them (they must fit the same limit as any collected result, otherwise
   `MemoryError` naming the size and `sink_parquet`); a sink applies the row-local nodes above the Sort to each range and
   writes it. Nodes above the Sort that are not row-local (Distinct, Aggregate) collect first.

Limits: a sort can be about 256 ranges x `(budget - held) / 16`, i.e. 16x the budget (a larger one raises a `MemoryError`
that says so); a range that comes out bigger than `(budget - held) / 3` (sampling variance, 64 samples per run keep it
within 20-30%) also raises. Provenance: `streamed.rows = {kind: "sort", sort: {runs, ranges, bytes, rows}}`.

**count(distinct x) and other non-decomposable aggregates.**
- `Aggregate(Distinct(X), keys, aggs)` (the DataFrame form `select(a, b).distinct().groupby(a).agg(col(b).count())`)
  is rewritten, when step 2 would decline for the Distinct, as `Aggregate(Aggregate(X, every column of X, [count(1)]),
  keys, aggs)` (`_distinct_to_aggregate`): removing duplicate rows is grouping by all their columns, with the same rules
  for NaN, -0.0 and NULL. SQL `COUNT(DISTINCT x)` is already two aggregates (the front end groups by keys + x first; with other
  aggregates they are combined per key afterwards), so it needed no rewrite.
- The inner aggregate is the streamed one: its partials are per (keys, x) pair and, when the pairs pass the merge limit,
  go to hash partitions on disk by `hash(keys, x)` (3a), each deduplicated on its own. The outer aggregate is then a
  *second stage* (`_next_stage`): each partition's groups are the batches of another partial aggregate (CPU), merged by the
  same `_State` (which spills too when the per-key groups are many). Exact: a pair is
  in exactly one partition, and the outer count of a key is the sum of its partitions' counts. A group whose x are all NULL
  counts 0, as in DuckDB. When the pairs fit, the merged inner table goes to the in-memory outer aggregate as before.
- **median / quantile / stddev / variance**: the engine has none of them (in memory either: `mx.sql` raised a bare
  `KeyError` for them, now `NotImplementedError: Unsupported aggregate function`). They are new engine aggregates, not a
  streaming question: stddev / variance would stream as (count, sum, sum of squares) partials, but a sum of squares loses
  digits to cancellation, so they need a Welford / Chan merge in the engine first (follow-up); median needs the group
  partition of 3a plus an in-partition sort.
