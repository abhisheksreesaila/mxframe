# User-felt TPC-H scoreboard - 2026-09-30

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **30** | 58 | 65 | - |
| Q3 3-join+top10 | **66** | 182 | 216 | - |
| Q5 4-join+groupby | **90** | 153 | 164 | - |
| Q7 volume shipping | **140** | 757 | 748 | - |
| Q8 market share | **72** | 651 | 655 | - |
| Q9 6-join profit | **129 †2e-6** | 1,013 | 1,018 | - |
| Q10 returned items top20 | **55** | 235 | 249 | - |
| Q14 promo revenue | **40** | 139 | 145 | - |
| Q20 potential promotion | **47** | 185 | 210 | - |
| Q21 suppliers waiting | **396** | 1,925 | 1,951 | - |
| **sum over the 10 queries valid for every engine** | **1,066** | **5,298** | **5,421** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF1 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (10 comparable)
- **mx-auto beats Polars on 0/10** (10 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-auto | 655 | 72 | 9.1x |
| Q9 6-join profit | mx-auto | 1,018 | 129 | 7.9x |
| Q7 volume shipping | mx-auto | 748 | 140 | 5.3x |
| Q21 suppliers waiting | mx-auto | 1,951 | 396 | 4.9x |
| Q10 returned items top20 | mx-auto | 249 | 55 | 4.6x |
| Q20 potential promotion | mx-auto | 210 | 47 | 4.5x |
| Q14 promo revenue | mx-auto | 145 | 40 | 3.6x |
| Q3 3-join+top10 | mx-auto | 216 | 66 | 3.3x |
| Q2 min-cost supplier | mx-auto | 65 | 30 | 2.1x |
| Q5 4-join+groupby | mx-auto | 164 | 90 | 1.8x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-cpu | 651 | 72 | 9.0x |
| Q9 6-join profit | mx-cpu | 1,013 | 129 | 7.8x |
| Q7 volume shipping | mx-auto | 748 | 140 | 5.3x |
| Q21 suppliers waiting | mx-cpu | 1,925 | 396 | 4.9x |
| Q10 returned items top20 | mx-cpu | 235 | 55 | 4.3x |
| Q20 potential promotion | mx-cpu | 185 | 47 | 3.9x |
| Q14 promo revenue | mx-cpu | 139 | 40 | 3.4x |
| Q3 3-join+top10 | mx-cpu | 182 | 66 | 2.8x |
| Q2 min-cost supplier | mx-cpu | 58 | 30 | 1.9x |
| Q5 4-join+groupby | mx-cpu | 153 | 90 | 1.7x |

### SF1: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 102 | 187 | 191 |
| Q3 | 135 | 321 | 342 |
| Q5 | 163 | 281 | 290 |
| Q7 | 211 | 887 | 872 |
| Q8 | 145 | 776 | 782 |
| Q9 | 200 †2e-6 | 1,137 | 1,144 |
| Q10 | 126 | 360 | 373 |
| Q14 | 112 | 267 | 270 |
| Q20 | 118 | 310 | 337 |
| Q21 | 469 | 2,049 | 2,075 |
| **sum (10 queries)** | **1,780** | **6,576** | **6,675** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 21 | 28 | 28 |
| Q3 | 53 | 137 | 164 |
| Q5 | 68 | 116 | 117 |
| Q7 | 103 | 711 | 732 |
| Q8 | 53 | 625 | 627 |
| Q9 | 99 | 933 | 940 |
| Q10 | 48 | 192 | 200 |
| Q14 | 27 | 105 | 107 |
| Q20 | 43 | 147 | 163 |
| Q21 | 350 | 1,871 | 1,897 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 71 | 127 | 125 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 1.9e-06, mx-cpu 3.5e-09, mx-auto 3.5e-09.


## Scale factor 10: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **99** | 250 | 263 | - |
| Q3 3-join+top10 | **482** | 1,298 | 1,339 | - |
| Q5 4-join+groupby | **769** | 1,209 | 1,219 | - |
| Q7 volume shipping | **1,210 †3e-6** | 5,296 | 5,291 | - |
| Q8 market share | **530** | 6,113 | 6,073 | - |
| Q9 6-join profit | **1,323 †7e-6** | 10,357 | 10,234 | - |
| Q10 returned items top20 | **344** | 2,116 | 2,120 | - |
| Q14 promo revenue | **161** | 399 | 406 | - |
| Q20 potential promotion | **240** | 854 | 874 | - |
| Q21 suppliers waiting | **4,094** | 18,977 | 18,975 | - |
| **sum over the 10 queries valid for every engine** | **9,252** | **46,868** | **46,794** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF10 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (10 comparable)
- **mx-auto beats Polars on 0/10** (10 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-auto | 6,073 | 530 | 11.4x |
| Q9 6-join profit | mx-auto | 10,234 | 1,323 | 7.7x |
| Q10 returned items top20 | mx-auto | 2,120 | 344 | 6.2x |
| Q21 suppliers waiting | mx-auto | 18,975 | 4,094 | 4.6x |
| Q7 volume shipping | mx-auto | 5,291 | 1,210 | 4.4x |
| Q20 potential promotion | mx-auto | 874 | 240 | 3.6x |
| Q3 3-join+top10 | mx-auto | 1,339 | 482 | 2.8x |
| Q2 min-cost supplier | mx-auto | 263 | 99 | 2.7x |
| Q14 promo revenue | mx-auto | 406 | 161 | 2.5x |
| Q5 4-join+groupby | mx-auto | 1,219 | 769 | 1.6x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q8 market share | mx-auto | 6,073 | 530 | 11.4x |
| Q9 6-join profit | mx-auto | 10,234 | 1,323 | 7.7x |
| Q10 returned items top20 | mx-cpu | 2,116 | 344 | 6.1x |
| Q21 suppliers waiting | mx-auto | 18,975 | 4,094 | 4.6x |
| Q7 volume shipping | mx-auto | 5,291 | 1,210 | 4.4x |
| Q20 potential promotion | mx-cpu | 854 | 240 | 3.6x |
| Q3 3-join+top10 | mx-cpu | 1,298 | 482 | 2.7x |
| Q2 min-cost supplier | mx-cpu | 250 | 99 | 2.5x |
| Q14 promo revenue | mx-cpu | 399 | 161 | 2.5x |
| Q5 4-join+groupby | mx-cpu | 1,209 | 769 | 1.6x |

### SF10: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 170 | 373 | 392 |
| Q3 | 552 | 1,421 | 1,466 |
| Q5 | 840 | 1,336 | 1,343 |
| Q7 | 1,284 †3e-6 | 5,422 | 5,415 |
| Q8 | 604 | 6,238 | 6,200 |
| Q9 | 1,392 †7e-6 | 10,482 | 10,360 |
| Q10 | 415 | 2,240 | 2,244 |
| Q14 | 231 | 526 | 536 |
| Q20 | 310 | 979 | 1,001 |
| Q21 | 4,167 | 19,111 | 19,101 |
| **sum (10 queries)** | **9,965** | **48,130** | **48,059** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 72 | 215 | 217 |
| Q3 | 379 | 1,211 | 1,248 |
| Q5 | 641 | 1,143 | 933 |
| Q7 | 1,006 | 5,149 | 5,173 |
| Q8 | 404 | 5,980 | 5,782 |
| Q9 | 1,077 | 10,374 | 9,967 |
| Q10 | 312 | 2,014 | 2,061 |
| Q14 | 135 | 350 | 350 |
| Q20 | 191 | 778 | 796 |
| Q21 | 3,939 | 18,864 | 18,942 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF10: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 71 | 125 | 126 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 6.9e-06, mx-cpu 2.8e-09, mx-auto 6.7e-07.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.0, max 6.1 over 60 runs; load average at report time: 2.6 3.5 4.5), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `07252ed`. Last invocation took 4 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

