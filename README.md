<img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-hero.svg" alt="MX frame, the Mojo engine: a data engine kernel apps and agents build on. Streamed 20M-row query 0.14 s (DuckDB 0.18 s); NOT EXISTS 9 ms (DuckDB 35 ms); median at 1M groups 398 ms (DuckDB 2,208 ms); a 1.6 GB file in 72 MB. Informal timings on a shared machine." width="100%"> <a href="https://github.com/abhisheksreesaila/mxframe/blob/main/docs/briefs/positioning.md"><img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-roadmap.svg" alt="Roadmap: DuckDB-class SQL that uses your GPU and links into any app with no Python. What omxcore can be: a faster drop-in for DuckDB, GPU-native analytics on one machine, an embedded engine for apps, the data engine for AI agents, the compute half of a lakehouse, live dataframes for simulation, ML data prep. The path: 1. SQL and joins, then the TPC-H 22 gate (we are here); 2. quiet-machine scoreboard; 3. close the gaps; 4. omxcore 0.7.0 with an MCP server; 5. joins bigger than RAM and the SQL long tail. Built for agents (plugins on the core, planned for M10 after 0.7.0): plug and play with your CSV and Parquet files, GPU-native vector search, agent search, classification, an MCP server and agent memory; mx with JSON, a 12 ms start and no Python works today." width="100%"></a> <img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-family.svg" alt="The omx family: omxapp (UI built once, native on Linux, macOS, iPhone, Android and the web) over omxcore (the data engine, CPU and GPU) over your CSV and Parquet files. The app and its engine compile into one native binary, with no Python and no server; omxplorer is the first app built this way." width="100%">

# MXFrame

**A data engine written in Mojo, for apps and agents: CPU, GPU and an auto mode that picks between them. Faster than DuckDB on the TPC-H SF10 total.** Soon renamed **omxcore** ([where it's going](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/briefs/positioning.md)).<br>
**Built for agents, plug and play with the CSV and Parquet files you already have:** `mx` speaks JSON with no Python today; GPU vector search, agent search, decisions, memory and an MCP server are planned next (M10).

[![PyPI](https://img.shields.io/pypi/v/mxframe?color=f97316)](https://pypi.org/project/mxframe/) [![Mojo](https://img.shields.io/badge/engine-Mojo%201.1-ff4f00)](https://github.com/abhisheksreesaila/mxframe/tree/main/mojo) [![Mojo API](https://img.shields.io/badge/Mojo%20API-no%20Python-22c55e)](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/mxframe-mojo-api.md) [![GPU](https://img.shields.io/badge/GPU-NVIDIA-76b900)](https://abhisheksreesaila.github.io/mxframe/auto-mode.html) [![Python](https://img.shields.io/badge/python-3.12%2B-3776ab)](https://github.com/abhisheksreesaila/mxframe/blob/main/pyproject.toml) [![Docs](https://img.shields.io/badge/docs-interactive-a855f7)](https://abhisheksreesaila.github.io/mxframe/)

🔥 **[The Mojo engine: what it does and how it got fast →](https://abhisheksreesaila.github.io/mxframe/engine.html)** ([long form](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/mojo-engine.md): every number, and where it is still behind) · 📖 **[Interactive docs](https://abhisheksreesaila.github.io/mxframe/)**: [benchmarks](https://abhisheksreesaila.github.io/mxframe/benchmarks.html) · [how `device="auto"` decides](https://abhisheksreesaila.github.io/mxframe/auto-mode.html) · [behind the scenes](https://abhisheksreesaila.github.io/mxframe/internals.html) · [how it got fast](https://abhisheksreesaila.github.io/mxframe/how-it-got-fast.html)

## 🔥 Two ways in

| | 🐍 **Python package** (`pip install mxframe`) | 🔥 **Mojo API** (`from mxframe import ...`) |
|---|---|---|
| For | notebooks, pipelines, SQL | apps and agents compiled in Mojo, e.g. [omxplorer](https://github.com/abhisheksreesaila/mxapp) |
| Python | Arrow in, Arrow out | **none**: no libpython, links on Linux and macOS (arm64) |
| Devices | 🟦 CPU · 🟩 GPU · 🟪 auto (a cost model calibrated on your machine) | 🟦 CPU · 🟩 GPU · 🟪 AUTO (GPU when a device copy is attached, CPU fallback with its reason) |
| Bigger than memory | 💾 streams files, spills big joins to disk (Grace hash join) | 💾 streams Parquet a row group at a time: a 1.6 GB file in 72 MB, with edits and save over it |
| Data | Parquet, Arrow, SQL | CSV (parallel, typed, import report), Parquet (zstd / snappy), edits, undo, formulas |

<img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-hood.svg" alt="Under the hood: modern algorithms, all built and in use today (morsel-driven parallelism, radix-partitioned hash joins without atomics, parallel radix sort and select, HyperLogLog-sized hash sets, bounded top-k heaps, SIMD bulk Parquet decode, two-phase parallel dictionaries, a planner from exact statistics, reproducible float sums) times Mojo, compiled ahead of time with SIMD types, compile-time specialisation, no garbage collector and the same kernels on CPU and GPU." width="100%"> <img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-wins.svg" alt="Where the Mojo engine leads DuckDB: SQL, joins and aggregates, from 11.9x (1M groups out) to 1.3x (join + group sum). Informal timings." width="100%"> <img src="https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/engine-behind.svg" alt="Where the Mojo engine is still behind DuckDB: one-to-many LEFT join 2.6x, full sort 2.4x, ORDER BY LIMIT 2.1x, probe-side filter 1.5x, streamed with 20 row groups 1.3-1.4x, two-column key 1.4x, 8 groups 1.3x. Informal timings." width="100%">

```mermaid
flowchart LR
  F[("📄 CSV · Parquet")]:::io --> T["🗃️ Table / SharedTable"]:::core
  T --> E["✏️ Frame: edits · undo · merges · formulas"]:::core --> Q["⚡ Query: where · group_by · agg"]:::q
  T --> Q
  Q -- AUTO --> C["🟦 CPU kernel, every core"]:::cpu
  Q -- AUTO --> G["🟩 GPU kernel, DeviceTable"]:::gpu
  C & G --> R["📊 QueryResult"]:::out
  classDef io fill:#0ea5e9,stroke:#0369a1,color:#fff
  classDef core fill:#f97316,stroke:#c2410c,color:#fff
  classDef q fill:#a855f7,stroke:#7e22ce,color:#fff
  classDef cpu fill:#2563eb,stroke:#1e40af,color:#fff
  classDef gpu fill:#16a34a,stroke:#166534,color:#fff
  classDef out fill:#eab308,stroke:#a16207,color:#000
```

```mojo
from mxframe import read_csv, CsvOptions, Query, Frame, col, sum, count, AUTO

var csv = read_csv("orders.csv", CsvOptions())       # parallel, typed, with an import report
var orders = csv.take_table().share()
var region = orders.column("region")
var out = (Query(orders).where(col("qty") > 2)
           .group_by(col("region"), groups=len(orders.dictionary(region)))
           .agg(sum(col("revenue")), count()).compute(AUTO))      # out.device: where it ran
var f = Frame(orders)                                 # a session's edits over shared data, no copy
f.set_text(f.column("revenue"), 0, "99.00"); f.formula("net = revenue - qty")
print(Query(f).agg(sum(col("net"))).compute().values(0)[0])     # decimals are exact scaled integers
```

Runnable as [`mojo/tools/readme_example.mojo`](https://github.com/abhisheksreesaila/mxframe/blob/main/mojo/tools/readme_example.mojo) (checked by `pixi run mojo-test`). Reference: [Mojo API](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/mxframe-mojo-api.md) · [build switches](https://github.com/abhisheksreesaila/mxframe/blob/main/mojo/README.md) · build the package with `pixi run bash scripts/build_mojo_pkg.sh` → `build/mxframe.mojoc`.

## Latest release: 0.6.0

The native Mojo CPU executor ([release notes](https://github.com/abhisheksreesaila/mxframe/blob/main/RELEASE_NOTES_0.6.0.md)). All 22 TPC-H queries, query time only (fresh process, data already loaded, no warm-up), every MXFrame result equal to DuckDB's:

| Run | MXFrame | DuckDB | Polars |
|---|---|---|---|
| SF10 | 3.15 s | 5.08 s | 14.40 s |
| SF1 | 0.47 s | 0.63 s | 1.31 s |
| SF10 shuffled | 4.97 s | 5.55 s | 15.06 s |
| SF1 shuffled | 0.61 s | 0.66 s | 1.41 s |

Honestly: Q21 carries the lead. Without it DuckDB is ahead, 2.27 s to 2.41 s at SF10 and 2.79 s to 3.63 s at SF10 shuffled, and DuckDB is faster on 12 of the 22 SF10 queries, mostly join-heavy ones. On seeded random queries over the same tables (geometric mean of DuckDB's time over MXFrame's), MXFrame is 1.09x at SF10 (50 queries), 0.83x at SF1 (100) and 1.21x at SF1 shuffled (100). [Every query, interactively →](https://abhisheksreesaila.github.io/mxframe/benchmarks.html)

## Install and run

```sh
pip install "mxframe[runtime,sql]"   # pixi: pixi add --pypi "mxframe[sql]" (MAX comes from conda)
```

```python
import mxframe as mx

lf = mx.scan_parquet("lineitem.parquet")
out = (lf.filter(mx.col("l_quantity") < 24).groupby("l_returnflag")
         .agg(mx.col("l_extendedprice").sum().alias("revenue"))
         .sort("l_returnflag").compute(device="cpu"))   # a pyarrow Table
print(out.to_pandas())
```

Nothing runs until `.compute()`; then the plan is optimized and executed in one native call. The 0.6.0 numbers above are `device="cpu"`; the default, `device="auto"`, runs the same native executor whenever it decides on the CPU, and `explain(device="auto", costs=True)` says why. On main (F-058, quiet machine) auto took 2.75 s against cpu 2.67 s on all of SF10. Extras: `runtime` (the MAX engine; pixi users get it from conda), `sql` (sqlglot), `polars`. The GPU path is validated on NVIDIA; MXFrame caps the Mojo runtime's GPU memory pool at 25% on import (`MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT`).

## API reference

Examples use `lf` (a `LazyFrame` over columns `id, dept, x, price, name, day`) and `lf2` (`id, region`). Every example below is run by `tests/test_readme_api.py`.

**Sources and running**

| Call | What it does | Example |
|---|---|---|
| `mx.scan_parquet(path)` | Lazy Parquet scan (file or directory) | `mx.scan_parquet("t.parquet")` |
| `mx.LazyFrame(table)` | Wrap a `pyarrow.Table` | `mx.LazyFrame(t)` |
| `mx.sql(query, **tables)` | SQL over named tables | `mx.sql("SELECT dept, SUM(x) AS s FROM t GROUP BY dept", t=t)` |
| `.compute(device)` | Run: `"auto"` (default), `"cpu"`, `"gpu"` | `lf.compute(device="cpu")` |
| `.explain()` | The optimized plan, as text | `lf.filter(col("x") > 2).explain()` |

**LazyFrame**

| Method | What it does | Example |
|---|---|---|
| `.filter(expr)` | Keep rows where `expr` is true | `lf.filter(col("x") > 10)` |
| `.select(*cols)` | Pick or compute columns | `lf.select("id", (col("price") * 2).alias("p2"))` |
| `.with_columns(*exprs)` | Add or replace columns | `lf.with_columns((col("x") + 1).alias("x1"))` |
| `.groupby(*keys).agg(*aggs)` | Grouped aggregation | `lf.groupby("dept").agg(col("price").sum().alias("s"))` |
| `.join(other, on, how)` | `how`: inner, left, semi, anti | `lf.join(lf2, on="id", how="left")` |
| `.sort(*by, descending)` | Order rows | `lf.sort("price", descending=True)` |
| `.limit(n)` / `.head(n)` | First n rows | `lf.sort("x").limit(2)` |
| `.distinct(*subset)` | Unique rows | `lf.select("dept").distinct()` |
| COUNT(DISTINCT b) per a | distinct, then count | `lf.select("dept", "x").distinct().groupby("dept").agg(col("x").count().alias("n"))` |
| `.to_pandas()` / `.to_polars()` | Run and convert | `lf.to_pandas()` |

**Expressions**

| Expression | What it does | Example |
|---|---|---|
| `col(name)`, `lit(v)` | A column, a constant | `col("price") * lit(1.1)` |
| `+ - * /`, `== != < <= > >=` | Arithmetic, comparison | `(col("price") - col("x")) > 0` |
| `& \| ~` | And, or, not | `(col("x") > 2) & ~(col("dept") == "ops")` |
| `.isin(values)` | Membership | `col("dept").isin(["eng", "hr"])` |
| `.between(lo, hi)` | Inclusive range | `col("x").between(2, 15)` |
| `.startswith(s)`, `.contains(s)` | String predicates | `col("name").startswith("al")` |
| `.str_slice(offset, length)` | Substring | `col("name").str_slice(0, 2)` |
| `.year()` | Year of a date | `col("day").year()` |
| `when(cond, then, otherwise)` | Conditional (no `otherwise` is NULL) | `when(col("x") > 10, col("price"), 0.0)` |
| `.sum() .mean() .min() .max() .count()` | Aggregates (in `.agg`) | `lf.groupby("dept").agg(col("x").mean().alias("m"))` |
| `.over(*by, order_by)` | Window | `lf.select("id", col("price").sum().over("dept").alias("w"))` |
| `.rank() .dense_rank() .lag(n) .lead(n) .cum_sum()` | Window functions | `lf.select("id", col("price").cum_sum().over("dept", order_by="id").alias("c"))` |
| `.alias(name)` | Name the result | `lf.groupby("dept").agg(col("x").sum().alias("total"))` |

SQL covers `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN … ON` and `CASE WHEN`. Known limits in 0.6.0: SQL `COUNT(DISTINCT …)` isn't supported (use the DataFrame form above), and SQL `HAVING` is dropped without an error.

## More

[Benchmarks](https://abhisheksreesaila.github.io/mxframe/benchmarks.html) ([tables](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/benchmarks.md)) · [Behind the scenes](https://abhisheksreesaila.github.io/mxframe/internals.html) · [How it got fast](https://abhisheksreesaila.github.io/mxframe/how-it-got-fast.html) ([long form](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/optimizations.md)) · [How `device="auto"` decides](https://abhisheksreesaila.github.io/mxframe/auto-mode.html) · [API notes](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/api.md) · [Architecture](https://github.com/abhisheksreesaila/mxframe/blob/main/docs/vision-and-architecture.md) · [Kernel visualizer](https://abhisheksreesaila.github.io/mxframe/visualize/mxframe_pipeline.html) · [Changelog](https://github.com/abhisheksreesaila/mxframe/blob/main/CHANGELOG.md) · [0.6.0 notes](https://github.com/abhisheksreesaila/mxframe/blob/main/RELEASE_NOTES_0.6.0.md) · [Contributing](https://github.com/abhisheksreesaila/mxframe/blob/main/CONTRIBUTING.md)

Build from source with [pixi](https://pixi.sh): `git clone https://github.com/abhisheksreesaila/mxframe && cd mxframe && pixi install && pixi run test`. License: [Apache-2.0](https://github.com/abhisheksreesaila/mxframe/blob/main/LICENSE).
