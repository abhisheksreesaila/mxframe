# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | **81** | 329 | 83 | - |
| Q2 min-cost supplier | **60** | 389 | 64 | - |
| Q3 3-join+top10 | 179 | 386 | **174** | - |
| Q4 EXISTS semi-join | **188** | 386 | 211 | - |
| Q5 4-join+groupby | **159** | 361 | 159 | - |
| Q6 filter+global sum | 55 | 271 | **54** | - |
| Q7 volume shipping | 203 | 386 | **198** | - |
| Q8 market share | **195** | 494 | 200 | - |
| Q9 6-join profit | **300** | 480 | 302 | - |
| Q10 returned items top20 | **235** | 428 | 236 | - |
| Q11 important stock | **47** | 368 | 48 | - |
| Q12 shipping modes | **276** | 485 | 311 | - |
| Q13 customer distribution | **169** | 379 | 182 | PyArrow/NumPy |
| Q14 promo revenue | **120** | 351 | 145 | - |
| Q15 top supplier | **81** | 423 | 92 | - |
| Q16 part/supplier | **90** | 336 | 91 | - |
| Q17 small-qty revenue | **117** | 321 | 121 | - |
| Q18 large volume top100 | 401 | 464 | **393** | - |
| Q19 discounted revenue | 281 | 497 | **274** | - |
| Q20 potential promotion | **184** | 387 | 193 | - |
| Q21 suppliers waiting | 894 | 994 | **884** | - |
| Q22 global sales opportunity | **53** | 264 | 63 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **4,368** | **9,179** | **4,476** |  |

Fastest verified engine per query: mx-cpu 16, mx-gpu 0, mx-auto 6.

### Wins at SF1 (headline, both results verified)

- 
- 
- 

### SF1: import + first run, ms (headline + engine import time)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 208 | 456 | 207 |
| Q2 | 186 | 519 | 192 |
| Q3 | 309 | 512 | 300 |
| Q4 | 315 | 511 | 337 |
| Q5 | 287 | 486 | 287 |
| Q6 | 180 | 396 | 181 |
| Q7 | 328 | 516 | 324 |
| Q8 | 320 | 626 | 332 |
| Q9 | 424 | 606 | 427 |
| Q10 | 362 | 555 | 363 |
| Q11 | 173 | 493 | 175 |
| Q12 | 403 | 614 | 439 |
| Q13 | 296 | 505 | 309 |
| Q14 | 246 | 478 | 270 |
| Q15 | 208 | 548 | 218 |
| Q16 | 223 | 463 | 217 |
| Q17 | 244 | 447 | 247 |
| Q18 | 526 | 592 | 521 |
| Q19 | 406 | 624 | 416 |
| Q20 | 333 | 513 | 321 |
| Q21 | 1,019 | 1,121 | 1,011 |
| Q22 | 178 | 389 | 189 |
| **sum (22 queries)** | **7,174** | **11,970** | **7,283** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 58 | 72 | 59 |
| Q2 | 28 | 29 | 30 |
| Q3 | 137 | 122 | 140 |
| Q4 | 156 | 149 | 156 |
| Q5 | 113 | 97 | 114 |
| Q6 | 31 | 41 | 32 |
| Q7 | 150 | 134 | 152 |
| Q8 | 153 | 130 | 154 |
| Q9 | 248 | 204 | 260 |
| Q10 | 191 | 177 | 195 |
| Q11 | 17 | 14 | 17 |
| Q12 | 244 | 234 | 259 |
| Q13 | 37 | 34 | 42 |
| Q14 | 110 | 103 | 107 |
| Q15 | 56 | 67 | 59 |
| Q16 | 53 | 47 | 57 |
| Q17 | 106 | 94 | 110 |
| Q18 | 355 | 217 | 370 |
| Q19 | 350 | 242 | 237 |
| Q20 | 146 | 142 | 151 |
| Q21 | 877 | 781 | 853 |
| Q22 | 27 | 26 | 27 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| import | 126 | 127 | 127 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 6.5e-15, mx-auto 7.0e-15, mx-gpu 6.8e-15.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 4.5, max 5.1 over 66 runs; load average at report time: 4.3 4.9 4.3), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.4.0, mojo 1.0.0b2; mxframe at git `5a2d331`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

