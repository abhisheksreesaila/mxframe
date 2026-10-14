# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 25 | 20 †3e-6 | 88 | **18** | - |
| Q2 min-cost supplier | 26 | **18** | 93 | 46 | - |
| Q3 3-join+top10 | 22 | **19** | 88 | 45 | - |
| Q4 EXISTS semi-join | 19 | **19** | 90 | 42 | - |
| Q5 4-join+groupby | 24 | **20** | 106 | 45 | - |
| Q6 filter+global sum | **16** | 16 | 85 | 18 | - |
| Q7 volume shipping | 27 | **21** | 90 | 47 | - |
| Q8 market share | 26 | **18** | 93 | 47 | - |
| Q9 6-join profit | 27 | **20** | 90 | 48 | - |
| Q10 returned items top20 | 23 | **17** | 89 | 43 | - |
| Q11 important stock | **17** | 19 | 87 | 47 | - |
| Q12 shipping modes | 21 | **20** | 88 | 46 | - |
| Q13 customer distribution | 18 | **15** | 89 | 41 | PyArrow/NumPy |
| Q14 promo revenue | **18** | 19 | 86 | 44 | - |
| Q15 top supplier | 20 | **19** | 103 | 46 | - |
| Q16 part/supplier | 22 | **20** | 95 | 47 | - |
| Q17 small-qty revenue | 19 | **19** | 83 | 48 | - |
| Q18 large volume top100 | 20 | **20** | 91 | 42 | - |
| Q19 discounted revenue | 23 | **20** | 85 | 58 | - |
| Q20 potential promotion | 25 | **19** | 96 | 51 | - |
| Q21 suppliers waiting | 24 | **23** | 102 | 50 | - |
| Q22 global sales opportunity | 21 | **15** | 89 | 43 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **484** | **417** | **2,004** | **962** |  |

Fastest verified engine per query: pandas 3, Polars 18, DuckDB 0, mx-cpu 1.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 266 | 93 †3e-6 | 126 | 120 |
| Q2 | 262 | 90 | 132 | 148 |
| Q3 | 258 | 92 | 126 | 148 |
| Q4 | 253 | 93 | 129 | 143 |
| Q5 | 262 | 94 | 149 | 150 |
| Q6 | 252 | 88 | 124 | 124 |
| Q7 | 262 | 93 | 129 | 148 |
| Q8 | 263 | 90 | 133 | 149 |
| Q9 | 264 | 92 | 128 | 152 |
| Q10 | 261 | 91 | 128 | 149 |
| Q11 | 254 | 93 | 127 | 152 |
| Q12 | 258 | 92 | 129 | 148 |
| Q13 | 251 | 89 | 132 | 143 |
| Q14 | 262 | 91 | 126 | 146 |
| Q15 | 257 | 95 | 143 | 151 |
| Q16 | 258 | 93 | 134 | 152 |
| Q17 | 255 | 94 | 122 | 150 |
| Q18 | 255 | 93 | 133 | 143 |
| Q19 | 306 | 95 | 124 | 165 |
| Q20 | 304 | 100 | 137 | 165 |
| Q21 | 291 | 102 | 145 | 153 |
| Q22 | 257 | 89 | 129 | 144 |
| **sum (22 queries)** | **5,811** | **2,043** | **2,885** | **3,245** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 9.9 | 13 | 13 | 2.2 |
| Q2 | 10.0 | 14 | 16 | 3.6 |
| Q3 | 6.6 | 14 | 15 | 2.9 |
| Q4 | 4.5 | 16 | 15 | 2.3 |
| Q5 | 8.7 | 10 | 14 | 3.6 |
| Q6 | 2.0 | 7.2 | 11 | 1.5 |
| Q7 | 12 | 15 | 17 | 4.8 |
| Q8 | 11 | 12 | 17 | 4.4 |
| Q9 | 12 | 13 | 19 | 5.4 |
| Q10 | 8.4 | 12 | 17 | 3.6 |
| Q11 | 3.3 | 10 | 12 | 1.9 |
| Q12 | 5.8 | 12 | 14 | 3.1 |
| Q13 | 4.3 | 3.9 | 14 | 2.3 |
| Q14 | 3.8 | 10 | 13 | 2.6 |
| Q15 | 4.5 | 9.2 | 11 | 3.3 |
| Q16 | 7.7 | 13 | 15 | 3.6 |
| Q17 | 5.2 | 6.0 | 13 | 3.0 |
| Q18 | 5.8 | 18 | 16 | 2.8 |
| Q19 | 7.8 | 16 | 14 | 5.3 |
| Q20 | 7.9 | 7.6 | 17 | 4.3 |
| Q21 | 7.5 | 18 | 17 | 5.7 |
| Q22 | 6.5 | 14 | 14 | 2.1 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 237 | 73 | 39 | 103 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 40 | 24 †2e-4 | 91 | **20** | - |
| Q2 min-cost supplier | 28 | **20** | 93 | 48 | - |
| Q3 3-join+top10 | 26 | **21** | 91 | 52 | - |
| Q4 EXISTS semi-join | 22 | **20** | 93 | 48 | - |
| Q5 4-join+groupby | 28 | **20** | 92 | 47 | - |
| Q6 filter+global sum | **17** | 19 | 88 | 19 | - |
| Q7 volume shipping | 34 | **22** | 95 | 52 | - |
| Q8 market share | 29 | **21** | 92 | 48 | - |
| Q9 6-join profit | 33 | **21** | 99 | 53 | - |
| Q10 returned items top20 | 29 | **22** | 89 | 48 | - |
| Q11 important stock | **18** | 18 | 85 | 46 | - |
| Q12 shipping modes | 26 | **22** | 92 | 50 | - |
| Q13 customer distribution | 20 | **16** | 86 | 44 | PyArrow/NumPy |
| Q14 promo revenue | 20 | **16** | 88 | 47 | - |
| Q15 top supplier | 22 | **20** | 88 | 43 | - |
| Q16 part/supplier | 24 | **21** | 97 | 48 | - |
| Q17 small-qty revenue | 21 | **17** | 88 | 44 | - |
| Q18 large volume top100 | 26 | **24** | 96 | 52 | - |
| Q19 discounted revenue | 39 | **20** | 87 | 50 | - |
| Q20 potential promotion | 26 | **21** | 90 | 58 | - |
| Q21 suppliers waiting | 34 | **25** | 115 | 72 | - |
| Q22 global sales opportunity | 26 | **20** | 95 | 43 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **587** | **450** | **2,032** | **1,028** |  |

Fastest verified engine per query: pandas 2, Polars 19, DuckDB 0, mx-cpu 1.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 2/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 274 | 96 †2e-4 | 130 | 124 |
| Q2 | 272 | 92 | 133 | 168 |
| Q3 | 282 | 97 | 130 | 185 |
| Q4 | 259 | 94 | 133 | 151 |
| Q5 | 265 | 94 | 131 | 153 |
| Q6 | 253 | 92 | 128 | 120 |
| Q7 | 271 | 94 | 134 | 159 |
| Q8 | 266 | 93 | 132 | 150 |
| Q9 | 266 | 94 | 139 | 155 |
| Q10 | 265 | 94 | 127 | 150 |
| Q11 | 256 | 91 | 124 | 153 |
| Q12 | 267 | 96 | 130 | 152 |
| Q13 | 256 | 91 | 126 | 148 |
| Q14 | 256 | 88 | 128 | 149 |
| Q15 | 261 | 93 | 130 | 146 |
| Q16 | 263 | 93 | 137 | 151 |
| Q17 | 255 | 90 | 127 | 150 |
| Q18 | 264 | 95 | 136 | 153 |
| Q19 | 276 | 92 | 126 | 159 |
| Q20 | 263 | 93 | 130 | 163 |
| Q21 | 267 | 97 | 157 | 187 |
| Q22 | 302 | 96 | 136 | 148 |
| **sum (22 queries)** | **5,858** | **2,055** | **2,906** | **3,373** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 23 | 16 | 19 | 3.3 |
| Q2 | 11 | 17 | 17 | 4.5 |
| Q3 | 10 | 19 | 19 | 6.8 |
| Q4 | 6.7 | 17 | 18 | 5.5 |
| Q5 | 12 | 17 | 18 | 5.8 |
| Q6 | 2.7 | 7.7 | 11 | 2.0 |
| Q7 | 18 | 17 | 19 | 7.8 |
| Q8 | 13 | 16 | 21 | 7.1 |
| Q9 | 15 | 18 | 22 | 8.5 |
| Q10 | 12 | 6.1 | 18 | 7.0 |
| Q11 | 3.7 | 17 | 14 | 2.4 |
| Q12 | 10 | 17 | 15 | 6.8 |
| Q13 | 6.4 | 15 | 14 | 3.4 |
| Q14 | 4.6 | 8.4 | 14 | 4.3 |
| Q15 | 5.6 | 13 | 14 | 3.8 |
| Q16 | 8.5 | 13 | 16 | 5.8 |
| Q17 | 5.9 | 10 | 15 | 3.8 |
| Q18 | 10 | 18 | 20 | 8.4 |
| Q19 | 20 | 15 | 17 | 7.9 |
| Q20 | 9.6 | 15 | 18 | 14 |
| Q21 | 18 | 23 | 26 | 33 |
| Q22 | 7.4 | 14 | 14 | 2.4 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 237 | 72 | 40 | 104 |

### SF0.017: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 2.30e-04 | max relative error 2.30e-04 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 7.6e-08, Polars 2.3e-04, mx-cpu 5.2e-15.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), then runs read+query+materialize once (`first_ms`, the headline) and once more in the same process (`repeat_ms`).
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Column pruning**: nobody passes a column list. Polars (`scan_parquet`), DuckDB and MXFrame (`mx.scan_parquet`, one lazy scan per table; low-cardinality strings are read as dictionaries) prune to the columns each query references; pandas uses `pd.read_parquet(columns=...)` with the columns the query uses (it has no lazy reader).
- **Engines**: pandas (eager), Polars (lazy `scan_parquet` + `collect()`), DuckDB (SQL over `read_parquet`, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 8.0, max 10.3 over 176 runs; load average at report time: 5.6 5.7 5.1), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `7576e6c`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

