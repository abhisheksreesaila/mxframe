# User-felt TPC-H scoreboard - 2026-09-30

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | **78** | 249 | 266 | - |
| Q21 suppliers waiting | **392** | 1,942 | 1,971 | - |
| **sum over the 2 queries valid for every engine** | **471** | **2,191** | **2,237** |  |

Fastest verified engine per query: Polars 2, mx-cpu 0, mx-auto 0.

### Wins at SF1 (headline, both results verified)

- **mx-cpu beats Polars on 0/2** (2 comparable)
- **mx-auto beats Polars on 0/2** (2 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q21 suppliers waiting | mx-auto | 1,971 | 392 | 5.0x |
| Q10 returned items top20 | mx-auto | 266 | 78 | 3.4x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q21 suppliers waiting | mx-cpu | 1,942 | 392 | 5.0x |
| Q10 returned items top20 | mx-cpu | 249 | 78 | 3.2x |

### SF1: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q10 | 354 | 380 | 399 |
| Q21 | 463 | 2,069 | 2,096 |
| **sum (2 queries)** | **817** | **2,448** | **2,495** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q10 | 45 | 198 | 206 |
| Q21 | 349 | 1,893 | 1,909 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 173 | 129 | 129 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 1.8e-07, mx-cpu 2.1e-16, mx-auto 2.1e-16.


## Scale factor 10: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | **339** | 2,468 | 2,522 | - |
| Q21 suppliers waiting | **4,309** | 19,220 | 19,199 | - |
| **sum over the 2 queries valid for every engine** | **4,647** | **21,688** | **21,721** |  |

Fastest verified engine per query: Polars 2, mx-cpu 0, mx-auto 0.

### Wins at SF10 (headline, both results verified)

- **mx-cpu beats Polars on 0/2** (2 comparable)
- **mx-auto beats Polars on 0/2** (2 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | mx-auto | 2,522 | 339 | 7.4x |
| Q21 suppliers waiting | mx-auto | 19,199 | 4,309 | 4.5x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | mx-cpu | 2,468 | 339 | 7.3x |
| Q21 suppliers waiting | mx-auto | 19,199 | 4,309 | 4.5x |

### SF10: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q10 | 409 | 2,592 | 2,647 |
| Q21 | 4,380 | 19,347 | 19,330 |
| **sum (2 queries)** | **4,789** | **21,939** | **21,976** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q10 | 305 | 2,022 | 2,415 |
| Q21 | 4,038 | 19,059 | 19,331 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF10: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 71 | 126 | 128 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 1.1e-07, mx-cpu 1.7e-16, mx-auto 1.7e-16.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 0.9, max 4.3 over 12 runs; load average at report time: 1.9 3.7 5.4), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `7d0b4a8`. Last invocation took 2 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

### Query semantics

All engines and the DuckDB reference compute the same query. Where it differs from the official TPC-H text the difference is listed here; this is not a TPC-certified benchmark.

- Q2: region EUROPE and nation joined by key (n_regionkey = 3, no region table); 'BRASS' matched anywhere in p_type (TPC-H: suffix)
- Q5: region ASIA by n_regionkey = 2 (no region table)
- Q7: shipping year from l_shipdate; nations FRANCE/GERMANY by name
- Q8: supplier nation BRAZIL by nationkey 2, region AMERICA by n_regionkey = 1 (no region table)
- Q10: no c_name/c_acctbal/c_address/c_phone/c_comment output columns; groups by (o_custkey, n_name); ties on revenue broken by o_custkey
- Q13: the `o_comment NOT LIKE '%special%requests%'` predicate is NOT applied (o_comment is not read)
- Q16: supplier filter is `s_comment LIKE '%Complaints%'` (TPC-H: '%Customer%Complaints%')
- Q18: sorted by sum_qty DESC, l_orderkey (TPC-H: o_totalprice DESC, o_orderdate); no c_name/o_orderdate/o_totalprice output columns
- Q20: Canada by nationkey 3 (no nation table)
- Q21: Saudi Arabia by nationkey 20 (no nation table)
- Q22: the c_acctbal column is float32 in the data, so the average-balance threshold and the sums are computed on float32 inputs (the DuckDB reference sums them in DOUBLE)

**Work the mx engines do outside mxframe** (PyArrow / NumPy inside the timed region; all other queries are pure mxframe):

- Q13: PyArrow: appends the zero-order customer row (customers minus customers with orders); no left join: works around the engine's left-join count bug (F-016)
- Q22: PyArrow/NumPy: phone-prefix substring, NOT EXISTS anti-join

