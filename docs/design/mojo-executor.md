# Design: the whole plan executes in Mojo

Status: **approved (shape and principle), 2026-10-12** · Replaces the "shave Python" parts of F-028 and F-029

## The principle (Abhishek, 2026-10-12)

> Python is just a simple lightweight front end that does the planning, parses SQL, etc. All the heavy
> lifting has to be Mojo; else what is the point.

So Python owns the API, SQL parsing, planning, optimizing and compiling the plan, and nothing that touches
row data. Every operation that loops over rows runs in Mojo: filters, expressions, joins, aggregates,
sorts, distinct, string matching, and in time Parquet decoding (F-021). The Python execution path is a
transitional fallback, retired query by query as the native executor covers it; new engine work goes
into Mojo, not into the Python path.

## Why

MXFrame today is a Python engine that calls Mojo kernels: ~15,800 lines of Python (planner, optimizer, join
orchestration, expression evaluation, Arrow/NumPy glue) and ~7,600 lines of Mojo kernels, called one at a
time through ctypes. The benchmarks show where that costs us:

- Where one Mojo call does the whole query, MXFrame wins: SF10 Q1 runs in 145 ms against Polars' 2,879 ms
  (engine-only).
- Where it loses, the time goes to Python between kernels: planning, Arrow ↔ NumPy conversions,
  intermediate tables, PyArrow side effects. At 12k–100k rows that overhead *is* the query
  (mx-cpu median 4–5 ms vs Polars 2–3 ms). At SF10 the multi-step Python orchestration of joins is a
  large part of the 2.6x–5x gap on join queries.

Polars does it the other way round: Python only builds the plan, and one call hands it to Rust, which runs
it start to finish. This design adopts that shape with Mojo as the executor.

## The shape

```
LazyFrame (Python API, unchanged)
   │  optimize (existing optimizer: pushdown, pruning, join order)
   ▼
Physical plan compiler (Python)          ← the only Python on the hot path, runs once per query
   │  flat byte buffer: operators + expression bytecode + constants
   ▼
mx_execute(plan, inputs, outputs)        ← ONE call into Mojo
   │  pipelines, morsel-parallel, typed vectors, hash tables, sort
   ▼
Arrow arrays out (C Data Interface)      ← zero copy back to pyarrow.Table
```

Anything the compiler cannot express returns `None` and the query runs on today's Python path. A plan is
either fully native or not at all; never a wrong result. Coverage is measured per TPC-H query.

## Decisions

1. **Boundary: the Arrow C Data Interface, both ways.** Inputs are exported with `_export_to_c` (buffer
   pointers, offsets, validity bitmaps, dictionaries); results come back as Arrow arrays built in Mojo and
   imported with `_import_from_c`. This removes the Arrow → NumPy → Arrow round trips and the PyArrow
   calls that drag in pandas. Nothing uses the C Data Interface today; the existing kernels take raw
   pointers to NumPy buffers.

2. **Plan format: pipelines of operators, as in DuckDB and Polars' streaming engine.** A plan is a list of
   pipelines; each pipeline is a source, a chain of streaming operators, and a sink. Sinks that must see
   all their input (hash-join build, grouped aggregate, sort, distinct) end a pipeline, and their result
   is the source of a later one. Python serializes this as a flat buffer, like today's bytecode header.

3. **Execution: morsel-driven, push-based, on all cores.** Each pipeline is split into morsels (~64k rows)
   that worker threads pull from a shared counter; operators process a vector at a time (the X100 style the
   fused pipeline already uses). Thread-local sink state (partial aggregates, partitioned hash tables)
   merges at the end.

4. **Types: typed vectors, not float64 for everything.** Today's pipeline widens every value to float64,
   which is fine for aggregates but wrong for int64 keys above 2^53 and for exact equality. Registers
   become typed (int32 / int64 / float32 / float64 / bool / dictionary code), with float64 accumulation
   for sums and means kept as today (it is what gives the 1e-12 agreement with DuckDB).

5. **Strings: dictionary codes first.** Comparisons, `isin` and joins on low-cardinality strings work on
   dictionary codes, as the pipeline already does (literals are translated to codes in Python, once).
   General string functions (LIKE patterns, `substr`, string keys that are not dictionary-encoded) are a
   later milestone; until then such plans fall back.

6. **Operators, in build order.** Scan (Arrow buffers, column projection) · filter · project (expression
   bytecode) · grouped aggregate with a hash table on composite keys (today group ids are computed outside
   the pipeline and passed in) · global aggregate · hash join inner / left / semi / anti (build sink plus
   probe operator) · sort and top-k · distinct · limit.

7. **Reuse, don't restart.** The expression bytecode, `CMPC` literal semantics, the `IN_SET` dictionary
   test, the group-key encoders (`groupkeys_cpu.mojo`), the join kernels and the radix sort already exist
   and are tested; the executor wraps them as operators instead of rewriting them.

8. **One library, loaded once.** The executor lives in one CPU library, built without the NVPTX
   dependency if the toolchain allows (it is 11 of the ~22 ms one-time load, F-028).

9. **GPU later, same plan.** The GPU executor consumes the same plan buffer; co-processing and `auto` then
   decide per pipeline, not per kernel call. Not part of this work.

## Milestones (each its own ticket, each shippable, each measured)

| | What | Queries fully native | Exit check |
|---|---|---|---|
| M0 | Spike: Arrow C Data Interface in and out; Q6 and Q1 through one `mx_execute` call | 1, 6 | same results; Python-side overhead per query < 0.5 ms; numbers vs today at 12k / 1M / SF10 |
| M1 | Single-table pipelines: scan, filter, project, global + grouped aggregate (hash table in Mojo), sort, top-k, limit | 1, 6 + single-table stages of others | engine-only at 12k–1M within 1.5x Polars on those queries |
| M2 | Hash joins: inner, semi, anti, left; multi-pipeline plans | 3, 4, 5, 7, 10, 12, 14, 19 | SF10 engine-only join queries within 2x Polars (F-029's target) |
| M3 | Distinct, subquery-shaped plans (scalar results feeding a filter) | 11, 15, 17, 18, 20, 21, 22 | 18/22 queries fully native |
| M4 | General strings: LIKE, contains, substr, string keys | 2, 9, 13, 16 | 22/22 native |

The scoreboards (user-felt and engine-only, 12k → SF10) are rerun after each milestone. The Python path
stays as the fallback throughout and is only retired once a query is native and verified.

## Risks

- **Size.** This is the largest change to the engine so far: weeks, not days. The milestones are ordered so
  each one pays off on its own (M0–M1 alone should close most of the small-data gap).
- **Mojo 1.0.0b2.** Threading, generic code and FFI ergonomics are early. The existing kernels already use
  `parallelize` and C ABI exports, so the basics are proven; anything that blocks gets reported, not
  hacked around. Upgrading the toolchain (needed later for Marrow / parquet.mojo, F-021) may come first if
  M0 shows the old version is the bottleneck.
- **Two engines during the transition.** Every native query is checked against DuckDB and against the
  Python path in tests until the Python path for it is removed.
- **Debuggability.** Errors inside one big Mojo call are harder to locate than in Python. The executor
  returns error codes with the operator index, and an `explain` shows which pipelines ran natively.

## Not in scope

GPU execution of the plan (later, same format) · out-of-core / small-GPU-memory work (GPU track) ·
a native Parquet reader (F-021) · changes to the public `LazyFrame` API.

## Questions for Abhishek

1. ~~Approve the shape?~~ Approved, with the principle above.
2. Start with M0 as a time-boxed spike (a few days) before committing to M1–M4?
3. Keep F-028's cheap item (runtime load, ~22 ms per process) as a separate small ticket now, since it pays
   off regardless?
