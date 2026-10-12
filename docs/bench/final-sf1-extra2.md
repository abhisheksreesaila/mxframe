# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 83 | 336 | **80** | - |
| Q2 min-cost supplier | **60** | 387 | 64 | - |
| Q3 3-join+top10 | **181** | 365 | 190 | - |
| Q4 EXISTS semi-join | **202** | 411 | 208 | - |
| Q5 4-join+groupby | **154** | 349 | 163 | - |
| Q6 filter+global sum | **52** | 261 | 58 | - |
| Q7 volume shipping | **194** | 396 | 204 | - |
| Q8 market share | **194** | 488 | 201 | - |
| Q9 6-join profit | **297** | 467 | 304 | - |
| Q10 returned items top20 | 269 | 441 | **234** | - |
| Q11 important stock | **44** | 367 | 45 | - |
| Q12 shipping modes | **279** | 483 | 287 | - |
| Q13 customer distribution | **170** | 368 | 176 | PyArrow/NumPy |
| Q14 promo revenue | 143 | 380 | **140** | - |
| Q15 top supplier | **89** | 414 | 92 | - |
| Q16 part/supplier | **89** | 333 | 91 | - |
| Q17 small-qty revenue | **116** | 321 | 117 | - |
| Q18 large volume top100 | **406** | 480 | 411 | - |
| Q19 discounted revenue | 271 | 488 | **251** | - |
| Q20 potential promotion | **185** | 398 | 194 | - |
| Q21 suppliers waiting | 899 | 1,010 | **876** | - |
| Q22 global sales opportunity | **55** | 263 | 61 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **4,431** | **9,208** | **4,449** |  |

Fastest verified engine per query: mx-cpu 17, mx-gpu 0, mx-auto 5.

### Wins at SF1 (headline, both results verified)

- 
- 
- 

### SF1: import + first run, ms (headline + engine import time)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 209 | 462 | 204 |
| Q2 | 185 | 515 | 190 |
| Q3 | 306 | 494 | 318 |
| Q4 | 329 | 539 | 334 |
| Q5 | 281 | 477 | 287 |
| Q6 | 179 | 389 | 186 |
| Q7 | 321 | 524 | 333 |
| Q8 | 321 | 617 | 330 |
| Q9 | 424 | 593 | 430 |
| Q10 | 394 | 567 | 361 |
| Q11 | 179 | 493 | 171 |
| Q12 | 404 | 609 | 413 |
| Q13 | 295 | 492 | 307 |
| Q14 | 269 | 506 | 274 |
| Q15 | 217 | 541 | 219 |
| Q16 | 215 | 460 | 219 |
| Q17 | 241 | 446 | 244 |
| Q18 | 532 | 605 | 535 |
| Q19 | 396 | 613 | 377 |
| Q20 | 309 | 522 | 320 |
| Q21 | 1,027 | 1,137 | 1,001 |
| Q22 | 181 | 390 | 189 |
| **sum (22 queries)** | **7,212** | **11,991** | **7,241** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 58 | 69 | 61 |
| Q2 | 28 | 29 | 29 |
| Q3 | 133 | 123 | 140 |
| Q4 | 154 | 145 | 160 |
| Q5 | 119 | 97 | 116 |
| Q6 | 31 | 40 | 32 |
| Q7 | 147 | 140 | 152 |
| Q8 | 152 | 133 | 156 |
| Q9 | 253 | 210 | 250 |
| Q10 | 190 | 178 | 191 |
| Q11 | 15 | 14 | 16 |
| Q12 | 242 | 232 | 252 |
| Q13 | 39 | 38 | 45 |
| Q14 | 106 | 105 | 108 |
| Q15 | 55 | 66 | 60 |
| Q16 | 55 | 46 | 55 |
| Q17 | 105 | 89 | 108 |
| Q18 | 354 | 215 | 368 |
| Q19 | 242 | 235 | 236 |
| Q20 | 148 | 139 | 152 |
| Q21 | 866 | 768 | 833 |
| Q22 | 28 | 27 | 31 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| import | 126 | 126 | 127 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 6.4e-15, mx-auto 6.4e-15, mx-gpu 6.8e-15.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.7, max 7.2 over 66 runs; load average at report time: 5.4 5.2 4.3), so timings are indicative.**
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

