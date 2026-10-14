# MXFrame 0.6.0: a native Mojo CPU engine

The CPU engine is now a native Mojo executor. Each query stage (the scan, filter, joins, group-by,
DISTINCT, top-N and COUNT(DISTINCT)) runs in one call to an ahead-of-time compiled shared library,
directly on Arrow memory. Python only plans the query.

## Numbers (engine-only, first run, one build, quiet machine)

Loading data is staging. The clock covers the query and its materialized result, in a fresh
process, on the first run. Every MXFrame result equals DuckDB's.

| TPC-H, total of 22 queries | MXFrame (CPU) | Polars | DuckDB |
|---|---:|---:|---:|
| SF10 | **3.15 s** | 14.40 s | 5.08 s |
| SF1 | **0.47 s** | 1.31 s | 0.63 s |
| SF10, rows shuffled | **4.97 s** | 15.06 s | 5.55 s |
| SF1, rows shuffled | **0.61 s** | 1.41 s | 0.66 s |

Seeded random queries over the TPC-H tables (filters, group-bys, joins, top-N, DISTINCT,
COUNT(DISTINCT), HAVING). Each figure is the geometric mean of DuckDB time / MXFrame time, so
above 1x means MXFrame is faster:

| | SF1 | SF1, rows shuffled | SF10 |
|---|---:|---:|---:|
| all queries | 0.83x | 1.21x | 1.09x |

**Read honestly:**
- **Much of the TPC-H lead over DuckDB comes from Q21.** Leaving Q21 out, DuckDB's SF10 total is
  lower: 2.27 s against 2.41 s on stored rows, and 2.79 s against 3.63 s on shuffled rows.
- DuckDB is still faster on 12 of the 22 TPC-H queries at SF10, mostly joins.
- On random queries DuckDB still leads at SF1 (0.83x overall). At both SF1 and SF10 it also leads
  on SELECT DISTINCT, COUNT(DISTINCT), top-N and joins.
- MXFrame is faster than Polars on 20 of the 21 verified TPC-H queries at SF10.

Details: [How the CPU engine got fast](README.md#-how-the-cpu-engine-got-fast-v060-2026-10-14)
and [docs/optimizations.md](docs/optimizations.md).

## What's new

- **Native Mojo executor** (`libmxexec_cpu.so`, over the Arrow C Data Interface):
  - a bytecode interpreter for predicates and expressions;
  - a group-by specialized at compile time;
  - joins fused into the scan (star, semi, anti and nested);
  - native DISTINCT, ORDER BY ... LIMIT, WHERE + ORDER BY + LIMIT in one pass, and
    COUNT(DISTINCT);
  - one thread per physical core.
- **Dictionary strings stay as codes** through joins, filters, SELECT DISTINCT and
  COUNT(DISTINCT). Only the rows that reach the output are decoded.
- **Many-group aggregation** into one shared output; a runtime join filter for large semi-join
  partners; native `str_slice` and utf8 `isin`.
- **Parquet, SQL and planning:**
  - `mx.scan_parquet` with projection and filter pushdown;
  - an `mx.sql` front end;
  - the `device="auto"` planner with explain and fallback.
- **Benchmarks:**
  - an engine-only TPC-H scoreboard (`scripts/bench_user_felt.py --mode engine [--shuffle]`);
  - a seeded random-query suite (`scripts/bench_random/`);
  - all results checked against DuckDB.

Full list: [CHANGELOG.md](CHANGELOG.md).

## Known limits

- MXFrame still reads Parquet through PyArrow (F-021). Times that include the read are reported
  separately and have not been re-run.
- The COUNT(DISTINCT) on dictionary codes applies to `device="cpu"`. `device="auto"` decodes the
  counted column first.
- `HAVING` in `mx.sql` is dropped without an error (open bug).
