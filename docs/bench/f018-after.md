# User-felt TPC-H scoreboard - 2026-09-30

> **Note (F-018 review):** Q2's harness query differs between the before and after runs: after, `scripts/tpch_user/q_mx.py` Q2 selects the columns the rest of the query reads. Harness narrowing alone takes SF10 mx-cpu Q2 from about 1,727 to about 600 ms; the join engine takes it from about 600 to 255 ms.

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **29** | 59 | 66 | - |
| Q3 3-join+top10 | **69** | 179 | 217 | - |
| Q5 4-join+groupby | **92** | 154 | 170 | - |
| Q7 volume shipping | **135** | 191 | 197 | - |
| Q8 market share | **68** | 196 | 201 | - |
| Q9 6-join profit | **132 †2e-6** | 291 | 312 | - |
| Q10 returned items top20 | **57** | 231 | 241 | - |
| Q14 promo revenue | **38** | 139 | 145 | - |
| Q20 potential promotion | **51** | 186 | 211 | - |
| Q21 suppliers waiting | **393** | 1,916 | 1,947 | - |
| **sum over the 10 queries valid for every engine** | **1,063** | **3,544** | **3,707** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF1 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (10 comparable)
- **mx-auto beats Polars on 0/10** (10 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q21 suppliers waiting | mx-auto | 1,947 | 393 | 5.0x |
| Q10 returned items top20 | mx-auto | 241 | 57 | 4.2x |
| Q20 potential promotion | mx-auto | 211 | 51 | 4.2x |
| Q14 promo revenue | mx-auto | 145 | 38 | 3.8x |
| Q3 3-join+top10 | mx-auto | 217 | 69 | 3.2x |
| Q8 market share | mx-auto | 201 | 68 | 3.0x |
| Q9 6-join profit | mx-auto | 312 | 132 | 2.4x |
| Q2 min-cost supplier | mx-auto | 66 | 29 | 2.3x |
| Q5 4-join+groupby | mx-auto | 170 | 92 | 1.9x |
| Q7 volume shipping | mx-auto | 197 | 135 | 1.5x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q21 suppliers waiting | mx-cpu | 1,916 | 393 | 4.9x |
| Q10 returned items top20 | mx-cpu | 231 | 57 | 4.0x |
| Q20 potential promotion | mx-cpu | 186 | 51 | 3.7x |
| Q14 promo revenue | mx-cpu | 139 | 38 | 3.6x |
| Q8 market share | mx-cpu | 196 | 68 | 2.9x |
| Q3 3-join+top10 | mx-cpu | 179 | 69 | 2.6x |
| Q9 6-join profit | mx-cpu | 291 | 132 | 2.2x |
| Q2 min-cost supplier | mx-cpu | 59 | 29 | 2.0x |
| Q5 4-join+groupby | mx-cpu | 154 | 92 | 1.7x |
| Q7 volume shipping | mx-cpu | 191 | 135 | 1.4x |

### SF1: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 100 | 183 | 192 |
| Q3 | 139 | 303 | 342 |
| Q5 | 164 | 279 | 296 |
| Q7 | 205 | 315 | 321 |
| Q8 | 141 | 320 | 327 |
| Q9 | 202 †2e-6 | 416 | 436 |
| Q10 | 129 | 356 | 366 |
| Q14 | 110 | 263 | 270 |
| Q20 | 123 | 313 | 337 |
| Q21 | 463 | 2,043 | 2,071 |
| **sum (10 queries)** | **1,776** | **4,791** | **4,958** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 19 | 27 | 31 |
| Q3 | 53 | 134 | 162 |
| Q5 | 74 | 115 | 120 |
| Q7 | 108 | 148 | 154 |
| Q8 | 51 | 152 | 154 |
| Q9 | 103 | 245 | 248 |
| Q10 | 48 | 189 | 196 |
| Q14 | 29 | 106 | 107 |
| Q20 | 41 | 147 | 162 |
| Q21 | 356 | 1,873 | 1,894 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 71 | 124 | 125 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 1.9e-06, mx-cpu 3.5e-09, mx-auto 3.5e-09.


## Scale factor 10: headline (first run in a fresh process), ms

| query | Polars | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q2 min-cost supplier | **97** | 255 | 260 | - |
| Q3 3-join+top10 | **484** | 1,296 | 1,342 | - |
| Q5 4-join+groupby | **772** | 1,191 | 1,215 | - |
| Q7 volume shipping | **1,204 †3e-6** | 1,271 | 1,273 | - |
| Q8 market share | **522** | 1,528 | 1,575 | - |
| Q9 6-join profit | **1,319 †7e-6** | 2,632 | 2,556 | - |
| Q10 returned items top20 | **346** | 2,479 | 2,127 | - |
| Q14 promo revenue | **171** | 402 | 404 | - |
| Q20 potential promotion | **240** | 856 | 889 | - |
| Q21 suppliers waiting | **4,044** | 18,955 | 19,001 | - |
| **sum over the 10 queries valid for every engine** | **9,199** | **30,864** | **30,642** |  |

Fastest verified engine per query: Polars 10, mx-cpu 0, mx-auto 0.

### Wins at SF10 (headline, both results verified)

- **mx-cpu beats Polars on 0/10** (10 comparable)
- **mx-auto beats Polars on 0/10** (10 comparable)

### Where mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | mx-auto | 2,127 | 346 | 6.1x |
| Q21 suppliers waiting | mx-auto | 19,001 | 4,044 | 4.7x |
| Q20 potential promotion | mx-auto | 889 | 240 | 3.7x |
| Q8 market share | mx-auto | 1,575 | 522 | 3.0x |
| Q3 3-join+top10 | mx-auto | 1,342 | 484 | 2.8x |
| Q2 min-cost supplier | mx-auto | 260 | 97 | 2.7x |
| Q14 promo revenue | mx-auto | 404 | 171 | 2.4x |
| Q9 6-join profit | mx-auto | 2,556 | 1,319 | 1.9x |
| Q5 4-join+groupby | mx-auto | 1,215 | 772 | 1.6x |
| Q7 volume shipping | mx-auto | 1,273 | 1,204 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF10 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q10 returned items top20 | mx-auto | 2,127 | 346 | 6.1x |
| Q21 suppliers waiting | mx-cpu | 18,955 | 4,044 | 4.7x |
| Q20 potential promotion | mx-cpu | 856 | 240 | 3.6x |
| Q8 market share | mx-cpu | 1,528 | 522 | 2.9x |
| Q3 3-join+top10 | mx-cpu | 1,296 | 484 | 2.7x |
| Q2 min-cost supplier | mx-cpu | 255 | 97 | 2.6x |
| Q14 promo revenue | mx-cpu | 402 | 171 | 2.4x |
| Q9 6-join profit | mx-auto | 2,556 | 1,319 | 1.9x |
| Q5 4-join+groupby | mx-cpu | 1,191 | 772 | 1.5x |
| Q7 volume shipping | mx-cpu | 1,271 | 1,204 | 1.1x |

### SF10: import + first run, ms (headline + engine import time)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 169 | 383 | 383 |
| Q3 | 557 | 1,423 | 1,468 |
| Q5 | 845 | 1,316 | 1,339 |
| Q7 | 1,276 †3e-6 | 1,396 | 1,400 |
| Q8 | 593 | 1,667 | 1,702 |
| Q9 | 1,389 †7e-6 | 2,756 | 2,681 |
| Q10 | 419 | 2,618 | 2,252 |
| Q14 | 242 | 527 | 529 |
| Q20 | 311 | 981 | 1,013 |
| Q21 | 4,114 | 19,079 | 19,127 |
| **sum (10 queries)** | **9,915** | **32,147** | **31,894** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q2 | 79 | 213 | 213 |
| Q3 | 365 | 1,208 | 1,233 |
| Q5 | 634 | 1,132 | 936 |
| Q7 | 1,021 | 1,184 | 1,171 |
| Q8 | 393 | 1,453 | 1,323 |
| Q9 | 1,104 | 2,643 | 2,167 |
| Q10 | 309 | 2,008 | 2,036 |
| Q14 | 136 | 350 | 356 |
| Q20 | 194 | 788 | 814 |
| Q21 | 3,838 | 18,830 | 18,930 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF10: import time, ms (median over queries; not in the headline)

|  | Polars | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 71 | 125 | 125 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 6.9e-06, mx-cpu 2.8e-09, mx-auto 7.3e-07.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.8, max 8.4 over 60 runs; load average at report time: 4.7 4.2 4.0), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `205d5ec`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

