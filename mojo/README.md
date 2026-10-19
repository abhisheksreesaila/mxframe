# The Mojo package (no Python)

MX frame's Python-free API (`docs/mxframe-mojo-api.md`, brief `docs/briefs/mojo-api.md`).

- **Source:** the API in `mojo/mxframe/`. The native kernels stay in `kernels_aot/` (the Python path builds its
  `.so` files from them); `scripts/build_mojo_pkg.sh` copies them into the package as `mxframe/kernels/` with their
  imports made package-relative. Edit kernels in `kernels_aot/`, never in `build/`.
- **Build:** `pixi run bash scripts/build_mojo_pkg.sh` -> `build/mxframe.mojoc`. A program uses it with
  `mojo build app.mojo -I build`, plus `-Xlinker -lzstd -Xlinker -lsnappy` (Parquet codecs, C libraries) and their
  `-L` / `-rpath` (the pixi env's `lib/` here).
- **Tests:** `pixi run mojo-test` builds the package, then builds and runs each `mojo/tests/test_*.mojo` (a `main()`
  that raises on failure). A test binary that links libpython fails.
- **Lifetimes:** Mojo frees a value at its last use. A raw address taken from a `List` is dangling once the list
  is no longer used: keep the owner alive past the call (`_ = owner^`), or better, pass owned types, not addresses.

Why the kernels are copied, not imported from `-I kernels_aot`: a `.mojoc` does not carry modules that were only on
the `-I` path (a consumer then fails with "unable to locate module"), and a shared library that only re-imports
`@export` functions exports nothing, so the `.so` builds must keep compiling the original files.

## SQL (F-150, F-152)

`sql(catalog, "SELECT ...")` runs a query over one table, a Frame or a Parquet scan registered by name (aggregates, or rows: `SELECT *`,
expressions, `DISTINCT`, `ORDER BY`, `LIMIT` / `OFFSET`) and returns an owned `Table` plus the planner's notes and EXPLAIN
(`docs/mxframe-mojo-api.md` section 8). The modules, in the order a query passes them: `sql.mojo` (lexer, parser, syntax tree), `bind.mojo`
(Catalog, typed tree, binder), `plan.mojo` (the logical plan: Scan, Filter, Aggregate, Project, Sort, Limit), `planner.mojo` (`Planner` trait,
`RulePlanner`, `Stats`, `PlannerOptions`; the group strategy and the sink), `sqlprog.mojo` (the kernel program in SQL's mode), then the
executor: `exec.mojo` (Sources: table, frame mask + delta, scan row groups; shared helpers), `sqlexec.mojo` (the aggregate Sinks: dense,
hashed, holistic; `sql`), `rows.mojo` (the row Sinks: materialize through the kernel's block hook, top-k heaps), `post.mojo` (HAVING, the
select list, ORDER BY and LIMIT after aggregation, run as a row query over the groups); `sqleval.mojo` holds the result column type and the
exact AVG. `mojo/tools/sql_probe.mojo` runs one SQL string over Parquet files and prints typed JSON (tests/test_mojo_sql.py and
tests/test_mojo_rows.py compare it, and `mx`, with DuckDB); `mojo/tests/test_sql.mojo` and `test_sql_parse.mojo` run in `pixi run mojo-test`.

## `mx`, the agent command line (F-151)

`mojo/cli/mx.mojo` is `mx sql | explain | schema` over the SQL above: JSON in, JSON out, exit 1 for a query error and 2 for a usage error
(`docs/mxframe-mojo-api.md`, "The `mx` command line"). `mojo/mxframe/output.mojo` writes a Table as JSON, JSON lines or CSV (exact decimals,
ISO dates). Build: `pixi run bash scripts/build_mx.sh` → `build/mx` (it fails if libpython or Mojo's Python interop is linked). tests/mx_oracle.py
runs a SQL string through `build/mx` and DuckDB and compares typed values; tests/test_mx.py uses it.

## Build switches (F-138b): the package on an app with no GPU, thread pool or codecs

Three compile-time switches, read when the app builds (`mojo build ... -D NAME`), not when the package is built. Each
is **on for every target but Linux** (an Apple object needs no flags) and `-D` turns it on for Linux too, which is how
`pixi run mojo-test` tests the other side (`mojo/tests/switches/test_switches.mojo`). **Android counts as Linux** to
Mojo: an Android build gets the GPU code, the thread pool and the codecs unless it passes all three `-D` flags.

| Switch | Off (Linux default) | On |
|---|---|---|
| `MXFRAME_NO_GPU` (`HAS_GPU` in `gpu.mojo`) | `DeviceTable`, `compute(GPU)`, the GPU kernel | `gpu_available()` is False; `DeviceTable(...)` and `compute(GPU)` raise "no GPU on this build"; AUTO is the CPU. No `AsyncRT_Device*` symbol. |
| `MXFRAME_SERIAL` (`kernels/_fpenv.mojo`) | the thread pool (`par`, `par_n`) | tasks run in a loop on the calling thread, `core_count()` is 1, no `AsyncRT_*` symbol |
| `MXFRAME_NO_CODECS` (`parquet.mojo`) | zstd and snappy (C libraries) | `write_parquet` saves UNCOMPRESSED whatever `compression` says; `read_parquet` refuses a zstd / snappy column with an error naming it. No `ZSTD_*` / `snappy_*` symbol |

AUTO and the GPU: `compute(AUTO)` goes to the CPU when the GPU path fails for one of two reasons only, an allocation
failure for the query's buffers or a program the kernel declines (-5 shared memory, -6 group table), and says which in
`QueryResult.fallback_reason`. A launch error or an illegal address raises on AUTO too. `compute(GPU)` never falls back.

**Link check:** `pixi run bash scripts/check_mac_link.sh` compiles `mojo/tools/mac_link_probe.mojo` over the package
for `arm64-apple-macosx14.0`, links it with `ld64.lld` against the app's `apple/sdk/macos/*.tbd` and its
`mojoui/hosts/mojo_rt.mojo` stand-ins, and fails on any AsyncRT, Tracy, KGEN (beyond the stand-ins), zstd, snappy,
Python or CUDA symbol. It prints the libSystem calls the package adds to the app's list (file I/O: `open`, `read`,
`mmap`, `rename`, ...): the app's stub and its `ALLOWED` set need them (F-138c).
