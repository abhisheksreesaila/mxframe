# How omxcore got fast

This page tells the story of omxcore's CPU engine, one optimization at a time: what was slow, why,
what the engine does now, and what it bought. Every number comes from the engine-only TPC-H
benchmark in [`docs/bench/`](bench/):

- each query runs in a fresh Python process, with no warm-up;
- the data is already loaded into memory;
- the clock covers only the query and its materialized result;
- every result is checked against DuckDB.

It is also an argument: that **Mojo** is a good language for building a data engine. Each
section says which Mojo feature the optimization relies on, or says plainly when the win is
ordinary engineering that any language would get.

> Numbers: [`docs/bench/m2s-engine.md`](bench/m2s-engine.md) and its shuffled-rows twin
> (2026-10-14, mx-cpu, Polars and DuckDB in one run each; Ryzen 9 5900X, 12 cores; a quiet machine).
> Each milestone's own section quotes the run that was current when it landed.
> Earlier milestones' DuckDB figures included DuckDB's pyarrow import (see F-044 below); omxcore's
> own figures are unaffected. Charts: `pixi run python scripts/plot_progress.py`.

![omxcore total per milestone against Polars and DuckDB](img/progress.svg)

## The shape of the engine

```mermaid
flowchart LR
    U["Your code<br/>LazyFrame / SQL"] --> P["Python: plan,<br/>optimize, compile"]
    P -->|"one call per stage<br/>(Arrow C Data Interface, zero copy)"| M["Mojo executor<br/>(AOT shared library)"]
    M -->|"Arrow buffers<br/>(zero copy)"| R["pyarrow.Table"]
```

The split was agreed at the start of the work. **Python is a thin front end:** it parses,
plans, optimizes, and turns expressions into a small bytecode. **All the work that loops over
rows is done by Mojo:** filters, joins, group-bys, DISTINCT and gathers. Python hands Mojo
pointers to the Arrow buffers; Mojo reads them in place and writes its results into one block of
memory. Arrow then wraps that block without copying it.

### The Mojo features this relies on

| Feature | Where it is used |
|---|---|
| **Ahead-of-time shared libraries** (`mojo build --emit shared-lib`, `@export ... abi("C")`) | `libmxexec_cpu.so`, called from Python through `ctypes`: no JIT warm-up inside a query |
| **`parallelize` with capturing closures** | every pass is a closure over its inputs, run on all cores in a few lines (morsel-driven, work-stealing through an atomic counter) |
| **Compile-time parameters** (`def f[E: ..., keyed: Bool]`, `comptime if`) | one interpreter body compiled into specialized versions (with and without per-block group keys); the key encoder closure is inlined, no virtual call per row |
| **`SIMD[DType.float64, W]` with `W = simd_width_of[...]()`** | aggregation lanes and vector loads sized for the CPU it was compiled for (AVX2: 4 doubles) |
| **Systems-level control from a Python-like language** | raw pointers into Arrow buffers, `madvise` for huge pages, `calloc` for lazily zeroed lookup arrays |

## The milestones

```mermaid
timeline
    title SF10 engine-only total, 22 TPC-H queries (Polars: 14.4 s, DuckDB: 5.1 s)
    before the executor : 22.7 s
    M1 native aggregates : 21.0 s
    M2a native leaf filters : 16.2 s
    M2b decode after the join : 15.7 s
    M2d star joins in one scan : 11.1 s
    M2e semi joins : 10.5 s
    M2g keyed group-by : 9.8 s
    M2h native DISTINCT, leaner scans : 7.3 s
    M2i-M2l threads, semi/anti joins, expression keys : 6.45 s
    M2m-M2n sparse keys, count distinct on sorted keys : 3.82 s
    F-043 many groups, runtime join filter, string kernels : 3.27 s
    F-046 dictionary codes, unsorted COUNT(DISTINCT), filtered top-N : 3.15 s (shuffled rows 8.64 → 4.97 s)
```

### M0 / M1: one native call runs a whole aggregate

**Before:** each operator was a separate kernel launch with Python glue between them, and
intermediate results were copied back and forth.

**Now:** the whole `Filter → GroupBy → Aggregate → Sort/Limit/Having` pipeline is compiled into
bytecode (registers, constants, a predicate section, a body section, aggregate slots). The data
is read once, in 64k-row morsels. One Mojo interpreter evaluates the bytecode over 1,024 rows at
a time, with SIMD, on every core.

```mermaid
flowchart LR
    A["Arrow columns"] --> B["morsel of 64k rows"] --> C["predicate over 1024 rows<br/>→ selection vector"] --> D["expressions on selected rows"] --> E["per-worker partial sums"] --> F["merge"]
```

*Mojo:* `parallelize` over morsels and `SIMD` lanes in the accumulators.

### M2a / M2b / M2c: joins stop paying for strings and for Python

- **Native leaf filters (M2a).** Each table under a join is filtered in Mojo. Only the surviving
  rows of the columns the join needs are written out. Q19 at SF10: 2,159 → 146 ms.
- **Decode strings after the join (M2b).** A dictionary-encoded string column (for example
  `p_type`, with only 150 distinct values) stays encoded until the join has decided which rows
  survive. Q12 at SF10: 1,788 → 511 ms.
- **Native gather (M2c).** The join's final "take these rows" is one Mojo call instead of Arrow
  `take`.

### M2d: star joins in one scan

**Before:** every join was materialized one at a time: filter lineitem, join it to orders,
join that to customer, and so on.

**Now:** the largest table (the *fact* table) is scanned once. Every other table is turned into
a lookup map on its join key. While the fact table's filter runs, each surviving row looks up its
partners and only fully joined rows are written:

- **compact key ranges** use a direct array indexed by key;
- **sparse keys** use a hash table split into 64 partitions, built in parallel.

```mermaid
flowchart LR
    subgraph before["Before: leaf by leaf"]
      L1[lineitem] --> J1{join} --> J2{join} --> J3{join}
      O1[orders] --> J1
      C1[customer] --> J2
      N1[nation] --> J3
    end
    subgraph after["Now: one scan of the fact table"]
      L2["lineitem<br/>(filter + lookups per row)"] --> OUT[joined rows]
      O2[orders map] -.-> L2
      C2["customer→nation map"] -.-> L2
    end
```

SF10: Q3 674 → 265 ms, Q9 1,496 → 667 ms, Q18 1,980 → 825 ms. The total went from 15.9 s to
11.1 s.

### M2e: semi joins without the DISTINCT

A partner that is `DISTINCT(keys)` and is joined on exactly those keys (Q4: "orders that have a
late lineitem") only has to answer one question: does a match exist? The DISTINCT never runs.
Q4 at SF10: 966 → 259 ms.

### M2f: not importing `pyarrow.compute` (plain engineering)

Importing `pyarrow.compute` costs about 22 ms per process: it generates a docstring for each of
its roughly 270 functions. Arrow's own methods, such as `Array.cast` and `Table.filter`, import
it as a side effect, so 15 of the 22 queries paid that cost inside their first run.

omxcore now calls Arrow's compute core directly. At SF1: Q17 41 → 17 ms, Q11 32 → 13 ms. **Not a
Mojo win**, but it shows that at small scale, fixed costs matter as much as kernels do.

### M2g: a group-by specialized at compile time

**Before:** a group-by with more than 4,096 groups first wrote a group id for every row of the
table (60M rows at SF10), including rows the filter was about to drop, and then aggregated in a
second pass.

**Now:** group ids are computed per 1,024-row block, only for the rows that pass the filter, and
are folded straight into per-worker partial sums.

*Mojo:* the interpreter is written once as `_pipeline[E, keyed: Bool]`. The compiler emits one
version that reads a precomputed group-id array and one that calls the key encoder `E`, a closure
inlined into the row loop. Two specialized loops from one source, with no runtime dispatch. Q15's
100k-group view at SF10: 158 → 63 ms.

### M2h: native DISTINCT, nested stars, leaner scans

- **DISTINCT in Mojo.**
  1. Row tuples are hashed in parallel.
  2. A small per-block cache drops repeats early, so 25 distinct values don't all land on one
     thread.
  3. The remaining rows are spread over 64 partitions, keeping row order.
  4. Each partition keeps the first row of each tuple, which preserves first-appearance order
     exactly.

  Q16's DISTINCT went from 112 to 24 ms. On 3M rows with 4 columns it takes 74 ms; the
  first-attempt group-by route took 1,638 ms.
- **Nested star joins.** A partner that is itself a chain of tables (Q8's orders → customer →
  nation) becomes its own star scan.
- **Most selective lookup first**, a presence **bitmap** in front of each direct lookup array
  (one bit per key, small enough for L2 cache), and **kept-row lists** instead of a 60M-entry row
  mask. Q8 at SF10: 212 → about 100 ms.

The total went from 9.8 s to 7.3 s; **the engine beat Polars on 20 of the 22 queries at SF10.**

### M2i: fewer threads can be faster

With 1.5M groups, each of the 24 workers was zeroing and later merging its own 24 MB partial
table, 576 MB of memory traffic for 6M input rows. Capping the partial budget at 128 MB, and
the workers at rows / groups, makes Q18 at SF1 go from 164 to 84 ms (Polars: 134 ms). Plain
engineering: more parallelism isn't free.

### M2j: semi and anti joins in the API

`join(..., how="semi" | "anti")` and `col(...).str_slice(offset, length)` let TPC-H Q22
("customers who never ordered") be written in omxcore alone instead of dropping to PyArrow. SF1:
34 → 18 ms.

### M2k: one thread per core, not per hyperthread (plain engineering)

The machine has 12 cores and 24 hardware threads. These kernels are limited by memory bandwidth,
so the second hyperthread on a core only adds contention. Using one worker per physical core,
and fewer for small inputs (one per 128k rows), helped at both scales: SF10 Q21 3.78 → 3.54 s,
SF1 Q8 22 → 19 ms.

### M2l: grouping by a computed key

TPC-H's "order year" is computed from a date with a CASE WHEN. Grouping by it used to push the
whole aggregate back to Python and Arrow. Now:

1. One native min/max pass measures the key's range.
2. The key expression is compiled into the same bytecode program as a hidden aggregate slot.
3. The group-by reads that slot's register straight from the interpreter, per block.

*Mojo:* the key encoder is a closure that receives the interpreter's registers, and the compiler
inlines it into the specialized row loop. SF10 Q9 666 → 439 ms (Polars 930 ms).

### M2m: small inputs over wide key ranges

Profiling the SF1 losses showed that Python planning is only 2-3 ms per query. The real cost was
the group-by kernel. Q11 groups about 32k rows by part key, whose range is 200k. Every worker
zeroed and merged a partial table sized for all 200k keys (about 13 MB), and then computed key
codes for every one of them.

Now a key range of at least 16,384 that is more than twice the row count takes the compacted path:
only the keys that occur are numbered, still in key order. Key codes are computed only for groups
that survive. SF1 first runs, measured in-process right after the change: Q11 14.6 → 6.1 ms,
Q17 29.5 → 9.7 ms, Q2 15.7 → 9.3 ms (in the benchmark run below, Q11 took 10.9 ms: small queries
move by several ms from run to run). Plain
engineering: the work should be proportional to the data, not the key range.

### M2n: COUNT(DISTINCT) on sorted keys (F-023)

TPC-H Q21 asks, per order, how many different suppliers delivered it (and how many were late). It
was written as `select(order, supplier).distinct().groupby(order).count()`: a 60M-row DISTINCT,
then a gather of the distinct table, then a second group-by. That was 2.5 s of Q21's 3.4 s at SF10.

But lineitem is stored by order key, and a filter keeps that order. When a key never decreases,
equal keys sit next to each other in **runs**, and each run is one group. So the engine checks the
order (one cheap pass), then counts the distinct suppliers inside each run, a handful of rows,
in one parallel pass. No hash table over 60M rows, no distinct table, no second group-by.

```mermaid
flowchart LR
    subgraph before["Before: three steps"]
      A1["DISTINCT (order, supplier)<br/>hash 60M rows"] --> A2["gather the<br/>distinct rows"] --> A3["GROUP BY order<br/>COUNT"]
    end
    subgraph after["Now: one pass over runs"]
      B1["is order key<br/>non-decreasing?"] -->|yes| B2["per run of equal keys:<br/>count distinct suppliers"]
      B1 -->|no| A1
    end
```

SF10 Q21: 3.44 s before (M2l run) → 0.95 s (Polars 3.93 s, DuckDB 2.81 s in the same run as the 0.95). SF1: 279 → 102 ms. Polars and DuckDB use
the same idea when they know a column is sorted; omxcore checks at run time.

*Mojo:* the kernel is a few `parallelize` closures over blocks that start at run boundaries, and
reuses the value hashing and comparison helpers written for DISTINCT, including strings.

### F-043: four general fixes for the queries DuckDB led

- **Millions of groups on every core.** A group-by with millions of groups (Q18's 15M orders) ran on
  one worker, because each worker's partial table would have taken gigabytes. Now all workers
  update one shared output with atomic add, min and max. A small per-worker cache of 4,096
  groups sits in front, so a hot key (skewed data) stays local. Compacted keys are encoded per
  block, with no group id per row. Q18 at SF10: 697 → about 400 ms. Non-TPC-H, 20M rows over 5M
  groups: sum 608 → about 390 ms; 95% of rows on one key 29 → 21 ms.
  *Mojo:* `Atomic[Scalar[DType.float64]]` add, min and max, and a compile-time `shared` parameter
  on the interpreter loop.
- **Runtime join filter.** A semi-join partner much larger than the filtered side (Q4's late
  lineitems) is filtered through that side's keys first, so it never materializes rows nobody
  can use. Q4: 224 → about 120 ms.
- **One sort, not two.** With an ORDER BY in the same call, groups are no longer sorted into key
  order first; the keys become the sort's last tie-break columns. Q10's group-by: 88 → 48 ms.
- **String kernels.** `str_slice` and a string `IN (...)` run natively, counting UTF-8 code points
  the way Arrow does. Q22: 148 → about 94 ms.

### F-044: random queries, and a fairness bug in our own benchmark

To check that none of this is TPC-H tuning, a seeded generator writes random queries over the
TPC-H tables and runs each one in omxcore, Polars and DuckDB, on the stored rows and on
shuffled rows. Shapes: filters, group-bys, 1-3 joins, top-N, DISTINCT, COUNT(DISTINCT) and HAVING.
Every result is checked against DuckDB, and **all 250 omxcore results matched**.

It found two blind spots and fixed them:
- **ORDER BY ... LIMIT had no fast path.** A native top-k now keeps a heap of the best k rows per
  worker (one query: 543 → 14 ms).
- **WHERE-only queries were filtered by Arrow.** They now go through the native filter.

It also found **a bug in the benchmark that favoured omxcore.** DuckDB imports pyarrow inside its
first `.arrow()` call, about 70 ms per process. That had been counted in every DuckDB time, and
it explained most of the earlier "faster than DuckDB on 22/22 at SF1". The import is now staging,
like omxcore's own.

### F-046: closing the gaps the random queries found

F-044's random queries showed where DuckDB still led: joins, top-N with a filter, SELECT
DISTINCT, COUNT(DISTINCT), and anything on shuffled rows. F-046 targeted those query shapes, not
TPC-H queries. Its numbers are judged on the random suite and on shuffled tables first.

- **Dictionary strings stay as codes** (F-046d, f, g, h).
  - A Parquet string column arrives as a dictionary: small integer codes plus one list of
    distinct values. Decoding it back into strings for every row was the single biggest cost on
    string-heavy queries.
  - Codes now pass through joins, single-table filters, SELECT DISTINCT and COUNT(DISTINCT).
  - Chunks that carry their own dictionaries are remapped to one shared dictionary.
  - Only the rows that reach the output are decoded.
  - Random SF10 examples (measured when each slice landed):
    - a join with a string group key: 1,597 → 639 ms;
    - a SELECT DISTINCT: 375 → 43 ms.
- **COUNT(DISTINCT) without sorted keys** (F-046b, h). M2n's sorted-run count only works when
  the table is stored by the group key. On unsorted keys the count now picks one of three layouts:
  - a radix-partitioned hash aggregate (partitions of about 64k consecutive keys, deduplicated in
    cache, with no atomics);
  - a byte map when (key range) x (value range) is small;
  - partitions by the hash of the (key, value) pair when there are too few keys to keep every
    core busy.

  Shuffled SF10 Q21: 4.98 → 1.34 s in this run (DuckDB 2.77 s). The random COUNT(DISTINCT) queries at SF10
  went from 0.31x to 0.89x of DuckDB's speed.
- **WHERE + ORDER BY + LIMIT in one pass** (F-046e).
  - The predicate runs per block, and rows that pass go straight into the per-worker top-k heaps.
    No filtered table is built.
  - Random SF10, when it landed: 420 → 63 ms (DuckDB 50 ms).

*Mojo:* the dictionary remaps, the byte map and the pair partitions are `parallelize` closures
over raw Arrow buffers. The byte map's per-key counts use `Atomic[DType.int64]` adds.

**Result (one build, quiet machine, `docs/bench/m2s-*`, `random-*-m2s.md`):**

| | before F-046 (F-044 run) | after F-046 |
|---|---:|---:|
| TPC-H SF10, rows shuffled: omxcore total (DuckDB) | 8.64 s (5.78 s) | **4.97 s** (5.55 s) |
| TPC-H SF10, stored rows: omxcore total (DuckDB) | 3.27 s (5.31 s) | 3.15 s (5.08 s) |
| random SF10 vs DuckDB, geometric mean | 0.75x | **1.09x** |
| random SF1 shuffled vs DuckDB | 0.91x | **1.21x** |
| random SF1 vs DuckDB | 0.63x | 0.83x |

The stored-rows TPC-H total moved little (3.27 → 3.15 s, within run-to-run variation). No F-046
work targeted it.

## Where it stands

![SF10: omxcore against Polars per query](img/per-query-sf10.svg)

| total of 22 queries | omxcore (CPU) | Polars | DuckDB |
|---|---:|---:|---:|
| SF10, engine-only | **3.15 s** | 14.40 s | 5.08 s |
| SF1, engine-only | **0.47 s** | 1.31 s | 0.63 s |
| SF10, rows shuffled | **4.97 s** | 15.06 s | 5.55 s |
| SF1, rows shuffled | **0.61 s** | 1.41 s | 0.66 s |

- **Against Polars:**
  - TPC-H: faster on 20 of the 21 verified queries at SF10 (Polars is 3% faster on Q16;
    Polars' Q1 result is wrong) and on 11 of 22 at SF1.
  - Random queries: faster at SF10 (43 of 50, 3.8x geometric mean), on shuffled SF1 (97 of 99,
    3.1x), and on stock SF1 (57 of 99, 1.3x).
- **Against DuckDB:**
  - **The lower TPC-H totals rest on Q21:** 0.74 s against 2.81 s at SF10 on stored rows, and
    1.34 s against 2.77 s shuffled. Leaving Q21 out, DuckDB's SF10 total is lower:
    - stored rows: 2.27 s against omxcore's 2.41 s;
    - shuffled rows: 2.79 s against 3.63 s.
  - DuckDB is faster on 12 of the 22 queries at SF10. The biggest gaps are joins: Q3, Q5, Q8,
    Q11, Q12, Q17 and Q22 run at 0.44–0.69x DuckDB's speed.
  - **Random queries** (geometric mean of DuckDB time / omxcore time):

    | | SF1 | SF1 shuffled | SF10 |
    |---|---:|---:|---:|
    | all queries | 0.83x | **1.21x** | **1.09x** |
    | SELECT DISTINCT (where DuckDB leads most) | 0.57x | 0.99x | 0.62x |

  - **Next:** joins on build sides that are not direct-addressable, SELECT DISTINCT, and the
    TPC-H join queries above.
- **Correctness:** omxcore's results equal DuckDB's on all 44 TPC-H runs, the 44 shuffled ones,
  and the 250 random queries.

![SF1: omxcore against Polars per query](img/per-query-sf1.svg)

### Read these numbers honestly

- **Engine-only is the headline.** Loading data is staging and stays outside the clock (as in the
  TPC-H power test, the H2O db-benchmark and ClickBench); the clock covers the first run of the
  query in a fresh process. The secondary user-felt scoreboard (import and Parquet read included,
  `docs/benchmarks.md`) has not been re-run since these optimizations. omxcore still reads
  Parquet through PyArrow, so that gap remains (F-021).
- **omxcore's kernel warm-up is staged.** `import omxcore` warms the kernel libraries in a
  background thread that finishes during the untimed load; Polars and DuckDB have no such step.
- **Polars' Q1 is an outlier.** At SF10 its first run takes 2.7 s, and its result differs from
  DuckDB's (float32 accumulation). Leaving Q1 out, SF10 totals are omxcore 3.09 s, Polars 11.68 s,
  DuckDB 4.94 s.
- **DuckDB is timed without its pyarrow import** (staging, like omxcore's import); earlier
  versions of this page counted it and overstated omxcore's lead.
- **TPC-H's stored order helps omxcore on Q21.** On stored rows Q21 uses the sorted-run
  count-distinct (0.74 s at SF10). Shuffled, it uses the hash layouts from F-046 (1.34 s). DuckDB
  takes about 2.8 s either way.
- **One build, a quiet machine.** Numbers come from `docs/bench/m2s-engine*.md` and
  `random-*-m2s.md`.
  - Shuffled TPC-H and the three random suites ran 01:21–01:50 on 2026-10-14. They started after
    the load average had stayed under 2 for 5 minutes.
  - The first stored-rows TPC-H pass overlapped a scheduled snapshot cleanup, and every engine
    came out slower. It was discarded.
  - The stored-rows run shown here was rerun at 02:05, after 5 minutes under load 1. The only
    processes sampled during it besides the benchmark were a browser at about a quarter of one
    core and, in the last sample, one short process at 98% of one core.

  SF1 queries take a few milliseconds, so single SF1 figures can move by 10–20% between runs.
- **Some wins are not Mojo.** M2f (imports), M2i (fewer workers) and M2k (physical cores) are
  ordinary engineering. The Mojo-specific part is everything that runs per row: the interpreter,
  the lookups, the hash tables, the gathers, DISTINCT and the run-based count, compiled ahead of
  time from code that reads like Python. M2m (work proportional to the data) is engineering too.
  So are F-046's ideas: keeping dictionary codes, the byte map and the fused filter + top-k are
  algorithms any engine could use. Mojo's part there is again the parallel per-row loops.
