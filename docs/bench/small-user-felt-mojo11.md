# User-felt TPC-H scoreboard - 2026-10-12

> **Note (F-035):** run on the uncommitted Mojo 1.1 port (modular 26.6, Mojo 1.1.0), *before* the lazy `pyarrow.compute` change; the "mxframe at git 9b5d66e" line below is the base commit, not 26.4 code. mx engines still pre-imported `pyarrow.compute` in this run.


**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 34 | 19 †3e-6 | 101 | **18** | - |
| Q2 min-cost supplier | 28 | **18** | 94 | 24 | - |
| Q3 3-join+top10 | 27 | **22** | 100 | 24 | - |
| Q4 EXISTS semi-join | **20** | 23 | 94 | 23 | - |
| Q5 4-join+groupby | 25 | **23** | 98 | 24 | - |
| Q6 filter+global sum | **16** | 18 | 84 | 19 | - |
| Q7 volume shipping | 29 | **23** | 106 | 23 | - |
| Q8 market share | 27 | **17** | 91 | 23 | - |
| Q9 6-join profit | 31 | **23** | 96 | 25 | - |
| Q10 returned items top20 | 25 | **18** | 90 | 24 | - |
| Q11 important stock | **19** | 20 | 91 | 22 | - |
| Q12 shipping modes | **21** | 23 | 88 | 24 | - |
| Q13 customer distribution | 23 | 20 | 96 | **19** | PyArrow/NumPy |
| Q14 promo revenue | 24 | **21** | 95 | 22 | - |
| Q15 top supplier | 20 | **20** | 93 | 23 | - |
| Q16 part/supplier | 25 | **19** | 107 | 20 | - |
| Q17 small-qty revenue | 22 | **17** | 89 | 25 | - |
| Q18 large volume top100 | 21 | **20** | 91 | 23 | - |
| Q19 discounted revenue | 22 | **16** | 88 | 27 | - |
| Q20 potential promotion | 23 | **22** | 88 | 23 | - |
| Q21 suppliers waiting | 21 | **21** | 96 | 26 | - |
| Q22 global sales opportunity | 21 | **21** | 100 | 29 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **526** | **443** | **2,079** | **511** |  |

Fastest verified engine per query: pandas 4, Polars 16, DuckDB 0, mx-cpu 2.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 3/22** (22 comparable); **mx-cpu beats pandas on 12/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 324 | 94 †3e-6 | 147 | 148 |
| Q2 | 292 | 91 | 137 | 184 |
| Q3 | 324 | 102 | 143 | 164 |
| Q4 | 305 | 104 | 138 | 153 |
| Q5 | 322 | 102 | 144 | 165 |
| Q6 | 296 | 90 | 124 | 149 |
| Q7 | 294 | 98 | 151 | 151 |
| Q8 | 289 | 90 | 129 | 155 |
| Q9 | 326 | 100 | 137 | 155 |
| Q10 | 271 | 97 | 129 | 153 |
| Q11 | 259 | 95 | 131 | 162 |
| Q12 | 271 | 99 | 130 | 156 |
| Q13 | 303 | 96 | 137 | 153 |
| Q14 | 345 | 105 | 137 | 182 |
| Q15 | 280 | 92 | 137 | 150 |
| Q16 | 304 | 100 | 151 | 185 |
| Q17 | 273 | 99 | 128 | 160 |
| Q18 | 259 | 93 | 130 | 153 |
| Q19 | 269 | 89 | 127 | 153 |
| Q20 | 262 | 94 | 127 | 151 |
| Q21 | 260 | 94 | 135 | 154 |
| Q22 | 308 | 98 | 142 | 167 |
| **sum (22 queries)** | **6,436** | **2,123** | **2,993** | **3,503** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 11 | 14 | 13 | 2.4 |
| Q2 | 11 | 12 | 17 | 3.3 |
| Q3 | 8.5 | 4.3 | 13 | 3.0 |
| Q4 | 4.6 | 15 | 12 | 2.5 |
| Q5 | 9.1 | 7.7 | 14 | 3.4 |
| Q6 | 2.2 | 4.4 | 13 | 1.5 |
| Q7 | 13 | 5.1 | 14 | 4.6 |
| Q8 | 12 | 11 | 18 | 4.9 |
| Q9 | 14 | 14 | 17 | 5.2 |
| Q10 | 8.3 | 12 | 15 | 3.7 |
| Q11 | 3.7 | 7.3 | 10 | 1.9 |
| Q12 | 5.6 | 12 | 13 | 2.7 |
| Q13 | 4.8 | 9.0 | 11 | 2.0 |
| Q14 | 4.3 | 7.1 | 18 | 3.8 |
| Q15 | 4.8 | 12 | 12 | 2.5 |
| Q16 | 9.5 | 4.6 | 15 | 3.1 |
| Q17 | 6.1 | 16 | 12 | 2.5 |
| Q18 | 5.7 | 15 | 15 | 3.3 |
| Q19 | 7.9 | 12 | 14 | 4.2 |
| Q20 | 8.4 | 14 | 16 | 3.7 |
| Q21 | 6.2 | 16 | 18 | 5.3 |
| Q22 | 8.5 | 17 | 15 | 5.8 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 265 | 76 | 41 | 131 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 42 | 25 †2e-4 | 98 | **23** | - |
| Q2 min-cost supplier | 28 | **18** | 148 | 23 | - |
| Q3 3-join+top10 | 26 | **18** | 93 | 26 | - |
| Q4 EXISTS semi-join | 22 | **20** | 94 | 24 | - |
| Q5 4-join+groupby | 29 | **21** | 97 | 26 | - |
| Q6 filter+global sum | 19 | **19** | 91 | 19 | - |
| Q7 volume shipping | 35 | **23** | 101 | 26 | - |
| Q8 market share | 31 | **22** | 99 | 27 | - |
| Q9 6-join profit | 36 | **24** | 106 | 31 | - |
| Q10 returned items top20 | 29 | **22** | 95 | 29 | - |
| Q11 important stock | 20 | **14** | 110 | 22 | - |
| Q12 shipping modes | 25 | **20** | 94 | 25 | - |
| Q13 customer distribution | 22 | 20 | 97 | **19** | PyArrow/NumPy |
| Q14 promo revenue | 20 | **19** | 88 | 23 | - |
| Q15 top supplier | 33 | **20** | 99 | 26 | - |
| Q16 part/supplier | 28 | **17** | 100 | 28 | - |
| Q17 small-qty revenue | 22 | **19** | 90 | 21 | - |
| Q18 large volume top100 | 27 | **22** | 96 | 31 | - |
| Q19 discounted revenue | 39 | **19** | 93 | 29 | - |
| Q20 potential promotion | 28 | **22** | 96 | 28 | - |
| Q21 suppliers waiting | 38 | **30** | 116 | 48 | - |
| Q22 global sales opportunity | 22 | **21** | 102 | 30 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **622** | **455** | **2,202** | **587** |  |

Fastest verified engine per query: pandas 0, Polars 20, DuckDB 0, mx-cpu 2.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 2/22** (22 comparable); **mx-cpu beats pandas on 13/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 325 | 114 †2e-4 | 143 | 200 |
| Q2 | 315 | 92 | 189 | 152 |
| Q3 | 268 | 90 | 132 | 170 |
| Q4 | 262 | 94 | 133 | 153 |
| Q5 | 362 | 97 | 138 | 207 |
| Q6 | 297 | 98 | 136 | 181 |
| Q7 | 360 | 100 | 159 | 171 |
| Q8 | 284 | 108 | 140 | 167 |
| Q9 | 306 | 99 | 151 | 166 |
| Q10 | 306 | 103 | 136 | 172 |
| Q11 | 307 | 90 | 155 | 156 |
| Q12 | 271 | 93 | 141 | 183 |
| Q13 | 289 | 94 | 138 | 147 |
| Q14 | 267 | 92 | 129 | 152 |
| Q15 | 356 | 96 | 142 | 171 |
| Q16 | 325 | 97 | 141 | 167 |
| Q17 | 273 | 95 | 129 | 166 |
| Q18 | 292 | 95 | 136 | 158 |
| Q19 | 304 | 93 | 142 | 179 |
| Q20 | 307 | 99 | 145 | 172 |
| Q21 | 284 | 110 | 157 | 221 |
| Q22 | 295 | 95 | 142 | 223 |
| **sum (22 queries)** | **6,655** | **2,145** | **3,154** | **3,833** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 24 | 20 | 21 | 4.3 |
| Q2 | 12 | 9.0 | 17 | 4.1 |
| Q3 | 9.9 | 16 | 17 | 5.4 |
| Q4 | 6.5 | 16 | 16 | 5.3 |
| Q5 | 14 | 6.6 | 16 | 6.1 |
| Q6 | 2.8 | 2.4 | 11 | 3.0 |
| Q7 | 18 | 14 | 19 | 6.6 |
| Q8 | 16 | 12 | 18 | 7.5 |
| Q9 | 18 | 8.6 | 21 | 9.4 |
| Q10 | 12 | 6.9 | 18 | 7.2 |
| Q11 | 3.9 | 10 | 14 | 2.4 |
| Q12 | 10 | 14 | 17 | 6.5 |
| Q13 | 7.1 | 13 | 14 | 2.8 |
| Q14 | 4.9 | 13 | 14 | 3.4 |
| Q15 | 6.7 | 3.6 | 14 | 4.8 |
| Q16 | 9.1 | 12 | 17 | 4.8 |
| Q17 | 7.7 | 14 | 17 | 4.1 |
| Q18 | 11 | 13 | 18 | 9.1 |
| Q19 | 21 | 15 | 19 | 8.5 |
| Q20 | 11 | 6.2 | 16 | 6.1 |
| Q21 | 18 | 17 | 26 | 24 |
| Q22 | 7.6 | 14 | 15 | 3.5 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 271 | 76 | 41 | 144 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.3e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 7.6, max 9.8 over 176 runs; load average at report time: 7.6 5.1 4.7), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `9b5d66e`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

