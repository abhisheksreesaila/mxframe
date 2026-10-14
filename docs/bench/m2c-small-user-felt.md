# User-felt TPC-H scoreboard - 2026-10-12

**Headline = what a user feels:** a fresh Python process, data starting as Parquet on disk, the clock around *Parquet read + query + materialized result*. No warm-up. `import` time is reported separately (and summed in the "import + first" tables). Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 120s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 0.002: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 27 | 66 †3e-6 | 87 | **18** | - |
| Q2 min-cost supplier | 30 | **25** | 91 | 44 | - |
| Q3 3-join+top10 | 23 | **16** | 92 | 24 | - |
| Q4 EXISTS semi-join | **19** | 19 | 88 | 46 | - |
| Q5 4-join+groupby | 24 | **18** | 88 | 45 | - |
| Q6 filter+global sum | **16** | 18 | 92 | 17 | - |
| Q7 volume shipping | 27 | **21** | 94 | 47 | - |
| Q8 market share | 27 | **18** | 94 | 47 | - |
| Q9 6-join profit | 32 | **21** | 94 | 49 | - |
| Q10 returned items top20 | 24 | **18** | 89 | 24 | - |
| Q11 important stock | **17** | 19 | 84 | 45 | - |
| Q12 shipping modes | 21 | **19** | 87 | 45 | - |
| Q13 customer distribution | 18 | **18** | 91 | 42 | PyArrow/NumPy |
| Q14 promo revenue | 18 | **18** | 85 | 46 | - |
| Q15 top supplier | 20 | **14** | 87 | 45 | - |
| Q16 part/supplier | 22 | **22** | 92 | 46 | - |
| Q17 small-qty revenue | 19 | **19** | 85 | 50 | - |
| Q18 large volume top100 | **20** | 20 | 93 | 46 | - |
| Q19 discounted revenue | 23 | **21** | 87 | 53 | - |
| Q20 potential promotion | 23 | **19** | 89 | 45 | - |
| Q21 suppliers waiting | 21 | **21** | 103 | 48 | - |
| Q22 global sales opportunity | 21 | **15** | 88 | 43 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **491** | **467** | **1,981** | **918** |  |

Fastest verified engine per query: pandas 4, Polars 17, DuckDB 0, mx-cpu 1.

### Wins at SF0.002 (headline, both results verified)

- **mx-cpu beats Polars on 2/22** (22 comparable); **mx-cpu beats pandas on 1/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.002: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 267 | 161 †3e-6 | 128 | 120 |
| Q2 | 268 | 99 | 129 | 146 |
| Q3 | 260 | 92 | 133 | 131 |
| Q4 | 258 | 92 | 128 | 149 |
| Q5 | 261 | 91 | 129 | 148 |
| Q6 | 254 | 91 | 131 | 121 |
| Q7 | 262 | 93 | 133 | 149 |
| Q8 | 267 | 94 | 135 | 150 |
| Q9 | 272 | 95 | 133 | 157 |
| Q10 | 259 | 93 | 127 | 127 |
| Q11 | 256 | 91 | 123 | 147 |
| Q12 | 255 | 91 | 126 | 164 |
| Q13 | 255 | 93 | 131 | 145 |
| Q14 | 255 | 93 | 124 | 150 |
| Q15 | 259 | 86 | 126 | 149 |
| Q16 | 259 | 96 | 130 | 149 |
| Q17 | 273 | 91 | 126 | 154 |
| Q18 | 255 | 94 | 132 | 151 |
| Q19 | 276 | 98 | 127 | 157 |
| Q20 | 259 | 90 | 131 | 149 |
| Q21 | 278 | 93 | 142 | 151 |
| Q22 | 259 | 91 | 127 | 163 |
| **sum (22 queries)** | **5,769** | **2,110** | **2,851** | **3,225** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.002: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 9.9 | 11 | 14 | 1.5 |
| Q2 | 10 | 9.9 | 17 | 4.5 |
| Q3 | 7.2 | 15 | 14 | 3.3 |
| Q4 | 3.9 | 14 | 15 | 2.4 |
| Q5 | 8.4 | 15 | 17 | 3.8 |
| Q6 | 2.4 | 8.1 | 11 | 1.2 |
| Q7 | 12 | 14 | 16 | 5.3 |
| Q8 | 11 | 12 | 17 | 4.8 |
| Q9 | 11 | 17 | 17 | 6.2 |
| Q10 | 8.3 | 14 | 17 | 3.7 |
| Q11 | 3.3 | 12 | 13 | 1.9 |
| Q12 | 6.4 | 11 | 14 | 3.1 |
| Q13 | 4.1 | 9.7 | 13 | 2.2 |
| Q14 | 4.0 | 8.4 | 13 | 2.7 |
| Q15 | 4.4 | 6.4 | 13 | 3.0 |
| Q16 | 8.0 | 11 | 16 | 4.0 |
| Q17 | 5.0 | 8.9 | 13 | 2.6 |
| Q18 | 5.8 | 15 | 14 | 3.1 |
| Q19 | 8.2 | 9.5 | 14 | 4.7 |
| Q20 | 8.2 | 15 | 17 | 4.4 |
| Q21 | 6.4 | 15 | 17 | 5.6 |
| Q22 | 6.7 | 7.6 | 14 | 2.4 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.002: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 238 | 73 | 39 | 103 |

### SF0.002: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.28e-06 | max relative error 3.28e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): pandas 1.6e-07, Polars 3.3e-06, mx-cpu 4.7e-15.


## Scale factor 0.017: headline (first run in a fresh process), ms

| query | pandas | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 40 | 23 †2e-4 | 90 | **20** | - |
| Q2 min-cost supplier | 27 | **18** | 89 | 48 | - |
| Q3 3-join+top10 | 26 | **21** | 90 | 26 | - |
| Q4 EXISTS semi-join | 22 | **21** | 94 | 46 | - |
| Q5 4-join+groupby | 29 | **21** | 96 | 47 | - |
| Q6 filter+global sum | 17 | **15** | 84 | 18 | - |
| Q7 volume shipping | 35 | **22** | 95 | 49 | - |
| Q8 market share | 29 | **20** | 96 | 50 | - |
| Q9 6-join profit | 32 | **22** | 97 | 56 | - |
| Q10 returned items top20 | 28 | **19** | 92 | 25 | - |
| Q11 important stock | **18** | 18 | 92 | 42 | - |
| Q12 shipping modes | 25 | **19** | 86 | 47 | - |
| Q13 customer distribution | 20 | **17** | 91 | 41 | PyArrow/NumPy |
| Q14 promo revenue | 20 | **20** | 84 | 46 | - |
| Q15 top supplier | 21 | **17** | 87 | 48 | - |
| Q16 part/supplier | 24 | **20** | 93 | 48 | - |
| Q17 small-qty revenue | 22 | **20** | 90 | 44 | - |
| Q18 large volume top100 | 26 | **23** | 93 | 50 | - |
| Q19 discounted revenue | 38 | **21** | 90 | 48 | - |
| Q20 potential promotion | 26 | **21** | 94 | 56 | - |
| Q21 suppliers waiting | 34 | **27** | 107 | 68 | - |
| Q22 global sales opportunity | 22 | **21** | 92 | 44 | PyArrow/NumPy |
| **sum over the 22 queries valid for every engine** | **582** | **446** | **2,023** | **968** |  |

Fastest verified engine per query: pandas 1, Polars 20, DuckDB 0, mx-cpu 1.

### Wins at SF0.017 (headline, both results verified)

- **mx-cpu beats Polars on 1/22** (22 comparable); **mx-cpu beats pandas on 3/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF0.017: import + first run, ms (headline + engine import time)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 278 | 95 †2e-4 | 128 | 121 |
| Q2 | 264 | 92 | 129 | 152 |
| Q3 | 265 | 93 | 131 | 128 |
| Q4 | 259 | 95 | 133 | 151 |
| Q5 | 268 | 94 | 137 | 149 |
| Q6 | 257 | 90 | 122 | 121 |
| Q7 | 296 | 95 | 135 | 151 |
| Q8 | 266 | 93 | 135 | 152 |
| Q9 | 274 | 97 | 136 | 158 |
| Q10 | 267 | 94 | 131 | 127 |
| Q11 | 256 | 91 | 131 | 145 |
| Q12 | 260 | 92 | 125 | 152 |
| Q13 | 256 | 91 | 132 | 143 |
| Q14 | 258 | 92 | 122 | 148 |
| Q15 | 259 | 92 | 126 | 154 |
| Q16 | 259 | 92 | 133 | 150 |
| Q17 | 256 | 92 | 129 | 163 |
| Q18 | 263 | 99 | 134 | 154 |
| Q19 | 275 | 95 | 129 | 152 |
| Q20 | 266 | 96 | 133 | 160 |
| Q21 | 274 | 100 | 147 | 171 |
| Q22 | 264 | 94 | 132 | 148 |
| **sum (22 queries)** | **5,841** | **2,063** | **2,890** | **3,250** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries while the user's code runs. Its cost is in neither `import` (it returns immediately) nor the headline unless it contends with the query for cores; the other engines have no equivalent.

### SF0.017: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| Q1 | 23 | 16 | 18 | 3.3 |
| Q2 | 11 | 9.8 | 19 | 4.8 |
| Q3 | 10 | 16 | 19 | 6.7 |
| Q4 | 6.6 | 15 | 16 | 5.3 |
| Q5 | 12 | 16 | 18 | 6.0 |
| Q6 | 2.6 | 7.5 | 13 | 1.8 |
| Q7 | 19 | 18 | 19 | 8.3 |
| Q8 | 13 | 15 | 20 | 7.6 |
| Q9 | 16 | 15 | 22 | 11 |
| Q10 | 11 | 15 | 18 | 7.0 |
| Q11 | 3.8 | 3.9 | 12 | 2.5 |
| Q12 | 10 | 12 | 17 | 5.2 |
| Q13 | 6.4 | 16 | 14 | 2.5 |
| Q14 | 4.8 | 15 | 14 | 4.2 |
| Q15 | 5.7 | 14 | 14 | 4.1 |
| Q16 | 8.2 | 15 | 17 | 5.8 |
| Q17 | 6.3 | 14 | 14 | 4.4 |
| Q18 | 10 | 19 | 18 | 9.0 |
| Q19 | 22 | 15 | 17 | 6.2 |
| Q20 | 9.4 | 14 | 17 | 14 |
| Q21 | 18 | 21 | 26 | 23 |
| Q22 | 6.8 | 15 | 14 | 2.6 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). Module-level result caches in `custom_ops.py` make the mx repeat numbers unrealistically fast; that is why the headline is a fresh process.

### SF0.017: import time, ms (median over queries; not in the first-run times)

|  | pandas | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|---:|
| import | 238 | 73 | 39 | 103 |

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
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 9.8, max 11.0 over 176 runs; load average at report time: 7.9 6.0 5.3), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `b685c7f`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

