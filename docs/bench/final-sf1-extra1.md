# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: headline (first run in a fresh process), ms

| query | mx-cpu | mx-gpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 82 | 346 | **79** | - |
| Q2 min-cost supplier | **60** | 427 | 63 | - |
| Q3 3-join+top10 | **179** | 373 | 189 | - |
| Q4 EXISTS semi-join | **190** | 408 | 193 | - |
| Q5 4-join+groupby | **158** | 350 | 161 | - |
| Q6 filter+global sum | 53 | 272 | **53** | - |
| Q7 volume shipping | **194** | 420 | 199 | - |
| Q8 market share | **193** | 495 | 203 | - |
| Q9 6-join profit | 298 | 478 | **296** | - |
| Q10 returned items top20 | **233** | 425 | 238 | - |
| Q11 important stock | **45** | 363 | 46 | - |
| Q12 shipping modes | 282 | 482 | **268** | - |
| Q13 customer distribution | **172** | 366 | 174 | PyArrow/NumPy |
| Q14 promo revenue | **119** | 381 | 122 | - |
| Q15 top supplier | 90 | 420 | **76** | - |
| Q16 part/supplier | **89** | 309 | 92 | - |
| Q17 small-qty revenue | 120 | 326 | **119** | - |
| Q18 large volume top100 | **405** | 482 | 408 | - |
| Q19 discounted revenue | **274** | 487 | 283 | - |
| Q20 potential promotion | **187** | 387 | 192 | - |
| Q21 suppliers waiting | **903** | 990 | 905 | - |
| Q22 global sales opportunity | **54** | 268 | 60 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **4,380** | **9,254** | **4,420** |  |

Fastest verified engine per query: mx-cpu 16, mx-gpu 0, mx-auto 6.

### Wins at SF1 (headline, both results verified)

- 
- 
- 

### SF1: import + first run, ms (headline + engine import time)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 207 | 474 | 204 |
| Q2 | 189 | 553 | 189 |
| Q3 | 304 | 500 | 314 |
| Q4 | 318 | 532 | 320 |
| Q5 | 283 | 477 | 286 |
| Q6 | 180 | 404 | 179 |
| Q7 | 322 | 545 | 324 |
| Q8 | 318 | 622 | 327 |
| Q9 | 424 | 604 | 424 |
| Q10 | 359 | 552 | 362 |
| Q11 | 172 | 493 | 173 |
| Q12 | 408 | 607 | 394 |
| Q13 | 301 | 490 | 303 |
| Q14 | 249 | 508 | 247 |
| Q15 | 215 | 546 | 204 |
| Q16 | 215 | 436 | 218 |
| Q17 | 247 | 451 | 244 |
| Q18 | 537 | 607 | 533 |
| Q19 | 399 | 611 | 408 |
| Q20 | 313 | 513 | 319 |
| Q21 | 1,031 | 1,118 | 1,029 |
| Q22 | 182 | 395 | 185 |
| **sum (22 queries)** | **7,171** | **12,038** | **7,186** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other eng have no equivalent.

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 58 | 72 | 60 |
| Q2 | 27 | 29 | 30 |
| Q3 | 135 | 127 | 140 |
| Q4 | 154 | 146 | 157 |
| Q5 | 114 | 96 | 113 |
| Q6 | 33 | 41 | 31 |
| Q7 | 151 | 133 | 149 |
| Q8 | 154 | 132 | 157 |
| Q9 | 253 | 209 | 259 |
| Q10 | 191 | 178 | 194 |
| Q11 | 16 | 15 | 16 |
| Q12 | 241 | 234 | 248 |
| Q13 | 37 | 36 | 42 |
| Q14 | 105 | 104 | 107 |
| Q15 | 57 | 69 | 60 |
| Q16 | 56 | 48 | 56 |
| Q17 | 105 | 89 | 108 |
| Q18 | 350 | 212 | 365 |
| Q19 | 239 | 236 | 236 |
| Q20 | 149 | 140 | 152 |
| Q21 | 874 | 773 | 839 |
| Q22 | 28 | 29 | 29 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF1: import time, ms (median over queries; not in the headline)

|  | mx-cpu | mx-gpu | mx-auto |
|---|---:|---:|---:|
| import | 127 | 126 | 126 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 6.2e-15, mx-auto 6.8e-15, mx-gpu 7.0e-15.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double.
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 5.5, max 7.2 over 66 runs; load average at report time: 7.2 5.5 4.4), so timings are indicative.**
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

