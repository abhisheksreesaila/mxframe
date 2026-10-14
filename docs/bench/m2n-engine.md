# Engine-only TPC-H scoreboard - 2026-10-13

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 275 †3e-3 | 122 | **11** | - |
| Q2 min-cost supplier | 9.2 | 80 | **8.1** | - |
| Q3 3-join+top10 | 31 | 86 | **18** | - |
| Q4 EXISTS semi-join | 75 | 89 | **26** | - |
| Q5 4-join+groupby | 47 | 85 | **19** | - |
| Q6 filter+global sum | 6.5 | 76 | **6.0** | - |
| Q7 volume shipping | 105 | 91 | **21** | - |
| Q8 market share | **14** | 84 | 17 | - |
| Q9 6-join profit | 65 †2e-6 | 128 | **53** | - |
| Q10 returned items top20 | 36 | 93 | **25** | - |
| Q11 important stock | **4.6** | 76 | 11 | - |
| Q12 shipping modes | 18 | 83 | **18** | - |
| Q13 customer distribution | 52 | 98 | **7.8** | PyArrow/NumPy |
| Q14 promo revenue | **9.7** | 84 | 12 | - |
| Q15 top supplier | **8.2** | 79 | 9.5 | - |
| Q16 part/supplier | **19** | 96 | 24 | - |
| Q17 small-qty revenue | **6.8** | 78 | 10 | - |
| Q18 large volume top100 | 131 | 104 | **75** | - |
| Q19 discounted revenue | 40 | 90 | **15** | - |
| Q20 potential promotion | 14 | 82 | **14** | - |
| Q21 suppliers waiting | 355 | 308 | **101** | - |
| Q22 global sales opportunity | **13** | 86 | 16 | - |
| **sum over the 22 queries valid for every engine** | **1,333** | **2,200** | **520** |  |

Fastest verified engine per query: Polars 7, DuckDB 0, mx-cpu 15.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 15/22** (22 comparable); **mx-cpu beats DuckDB on 22/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 351 †3e-3 | 161 | 112 |
| Q2 | 83 | 120 | 109 |
| Q3 | 104 | 124 | 122 |
| Q4 | 146 | 130 | 127 |
| Q5 | 120 | 124 | 124 |
| Q6 | 79 | 116 | 114 |
| Q7 | 176 | 129 | 123 |
| Q8 | 87 | 123 | 118 |
| Q9 | 136 †2e-6 | 166 | 172 |
| Q10 | 107 | 131 | 126 |
| Q11 | 76 | 113 | 117 |
| Q12 | 92 | 122 | 120 |
| Q13 | 123 | 139 | 112 |
| Q14 | 81 | 122 | 112 |
| Q15 | 79 | 118 | 111 |
| Q16 | 91 | 135 | 126 |
| Q17 | 79 | 115 | 112 |
| Q18 | 203 | 142 | 178 |
| Q19 | 110 | 128 | 115 |
| Q20 | 85 | 121 | 114 |
| Q21 | 426 | 346 | 204 |
| Q22 | 84 | 124 | 118 |
| **sum (22 queries)** | **2,916** | **3,049** | **2,786** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 78 | 130 | 53 |
| Q2 | 32 | 65 | 27 |
| Q3 | 54 | 115 | 79 |
| Q4 | 49 | 101 | 71 |
| Q5 | 63 | 109 | 92 |
| Q6 | 33 | 63 | 38 |
| Q7 | 61 | 114 | 92 |
| Q8 | 76 | 137 | 119 |
| Q9 | 86 | 176 | 143 |
| Q10 | 70 | 122 | 84 |
| Q11 | 26 | 47 | 17 |
| Q12 | 63 | 133 | 79 |
| Q13 | 32 | 38 | 37 |
| Q14 | 45 | 84 | 84 |
| Q15 | 41 | 70 | 45 |
| Q16 | 25 | 47 | 27 |
| Q17 | 44 | 73 | 81 |
| Q18 | 44 | 68 | 69 |
| Q19 | 75 | 164 | 100 |
| Q20 | 57 | 112 | 100 |
| Q21 | 56 | 96 | 77 |
| Q22 | 28 | 33 | 26 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 207 | 18 | 9.8 |
| Q2 | 7.0 | 10 | 6.4 |
| Q3 | 27 | 16 | 13 |
| Q4 | 67 | 14 | 23 |
| Q5 | 40 | 14 | 12 |
| Q6 | 5.9 | 4.7 | 4.3 |
| Q7 | 86 | 17 | 15 |
| Q8 | 13 | 15 | 11 |
| Q9 | 63 | 58 | 43 |
| Q10 | 34 | 21 | 20 |
| Q11 | 4.2 | 4.7 | 5.0 |
| Q12 | 18 | 13 | 14 |
| Q13 | 41 | 27 | 6.1 |
| Q14 | 7.2 | 13 | 6.8 |
| Q15 | 8.5 | 9.1 | 7.5 |
| Q16 | 15 | 22 | 21 |
| Q17 | 5.6 | 8.9 | 6.6 |
| Q18 | 120 | 33 | 62 |
| Q19 | 38 | 20 | 12 |
| Q20 | 14 | 12 | 10 |
| Q21 | 321 | 232 | 93 |
| Q22 | 12 | 13 | 14 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 71 | 38 | 102 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.8e-15, Polars 3.1e-03.


## Scale factor 10: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 2,584 WRONG | 220 | **66** | - |
| Q2 min-cost supplier | 51 | 97 | **19** | - |
| Q3 3-join+top10 | 304 | 159 | **121** | - |
| Q4 EXISTS semi-join | 955 | **171** | 207 | - |
| Q5 4-join+groupby | 492 | 141 | **112** | - |
| Q6 filter+global sum | 48 | 100 | **30** | - |
| Q7 volume shipping | 1,052 †3e-6 | 154 | **116** | - |
| Q8 market share | 142 | 134 | **108** | - |
| Q9 6-join profit | 854 †7e-6 | 503 | **424** | - |
| Q10 returned items top20 | 342 | **188** | 195 | - |
| Q11 important stock | 30 | 88 | **27** | - |
| Q12 shipping modes | 163 | 162 | **118** | - |
| Q13 customer distribution | 652 | 307 | **101** | PyArrow/NumPy |
| Q14 promo revenue | 83 | 133 | **53** | - |
| Q15 top supplier | 74 | 137 | **46** | - |
| Q16 part/supplier | 100 | 184 | **95** | - |
| Q17 small-qty revenue | 67 | 109 | **63** | - |
| Q18 large volume top100 | 1,458 | **381** | 700 | - |
| Q19 discounted revenue | 439 | 153 | **67** | - |
| Q20 potential promotion | 121 | 122 | **66** | - |
| Q21 suppliers waiting | 3,929 | 2,811 | **954** | - |
| Q22 global sales opportunity | 181 | **115** | 130 | - |
| **sum over the 21 queries valid for every engine** | **11,536** | **6,348** | **3,751** |  |

Fastest verified engine per query: Polars 0, DuckDB 4, mx-cpu 18.

### Wins at SF10 (engine-only, both results verified)

- **mx-cpu beats Polars on 21/22** (21 comparable); **mx-cpu beats DuckDB on 18/22** (22 comparable)

### SF10: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,654 WRONG | 258 | 167 |
| Q2 | 124 | 136 | 135 |
| Q3 | 375 | 196 | 222 |
| Q4 | 1,028 | 209 | 308 |
| Q5 | 563 | 180 | 232 |
| Q6 | 118 | 139 | 133 |
| Q7 | 1,125 †3e-6 | 192 | 219 |
| Q8 | 219 | 174 | 212 |
| Q9 | 925 †7e-6 | 541 | 525 |
| Q10 | 415 | 226 | 295 |
| Q11 | 102 | 126 | 128 |
| Q12 | 238 | 199 | 218 |
| Q13 | 722 | 347 | 201 |
| Q14 | 154 | 174 | 155 |
| Q15 | 145 | 176 | 147 |
| Q16 | 175 | 225 | 198 |
| Q17 | 141 | 150 | 165 |
| Q18 | 1,532 | 418 | 801 |
| Q19 | 510 | 193 | 172 |
| Q20 | 193 | 160 | 182 |
| Q21 | 4,000 | 2,849 | 1,055 |
| Q22 | 252 | 153 | 232 |
| **sum (21 queries)** | **13,057** | **7,162** | **5,937** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF10: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 521 | 602 | 504 |
| Q2 | 82 | 223 | 116 |
| Q3 | 316 | 476 | 666 |
| Q4 | 258 | 341 | 603 |
| Q5 | 351 | 503 | 682 |
| Q6 | 199 | 318 | 273 |
| Q7 | 382 | 558 | 721 |
| Q8 | 468 | 721 | 1,024 |
| Q9 | 515 | 812 | 1,194 |
| Q10 | 418 | 472 | 676 |
| Q11 | 51 | 83 | 85 |
| Q12 | 434 | 485 | 681 |
| Q13 | 75 | 83 | 208 |
| Q14 | 268 | 445 | 774 |
| Q15 | 241 | 384 | 385 |
| Q16 | 68 | 160 | 101 |
| Q17 | 235 | 392 | 741 |
| Q18 | 202 | 245 | 594 |
| Q19 | 562 | 757 | 884 |
| Q20 | 353 | 549 | 931 |
| Q21 | 328 | 431 | 636 |
| Q22 | 78 | 131 | 240 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF10: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| Q1 | 2,022 WRONG | 138 | 64 |
| Q2 | 45 | 31 | 16 |
| Q3 | 246 | 85 | 118 |
| Q4 | 839 | 96 | 195 |
| Q5 | 408 | 69 | 100 |
| Q6 | 43 | 30 | 28 |
| Q7 | 905 | 84 | 105 |
| Q8 | 93 | 59 | 95 |
| Q9 | 768 | 442 | 380 |
| Q10 | 308 | 115 | 185 |
| Q11 | 27 | 16 | 20 |
| Q12 | 153 | 77 | 116 |
| Q13 | 550 | 224 | 98 |
| Q14 | 60 | 59 | 48 |
| Q15 | 59 | 47 | 44 |
| Q16 | 84 | 115 | 86 |
| Q17 | 37 | 27 | 60 |
| Q18 | 1,468 | 311 | 612 |
| Q19 | 407 | 85 | 66 |
| Q20 | 100 | 47 | 62 |
| Q21 | 3,609 | 2,728 | 941 |
| Q22 | 169 | 38 | 125 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF10: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu |
|---|---:|---:|---:|
| import | 72 | 38 | 101 |

### SF10: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | wrong | 5.48e-02 | max relative error 5.48e-02 / repeat run WRONG |
| Q7 | polars | drift | 2.75e-06 | max relative error 2.75e-06 |
| Q9 | polars | drift | 6.92e-06 | max relative error 6.92e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.1e-14, Polars 5.5e-02.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


**mx-gpu was not run at SF10** (decision for this round: GPU memory is shared with other jobs and the join engine is being rewritten in F-018); SF10 tables have no mx-gpu column.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 1 (NVIDIA GeForce RTX 3090), GPU-2b0cad02-4be1-3933-8210-861d920e6f28; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 4.4, max 9.8 over 132 runs; load average at report time: 9.1 5.8 4.8), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `3536370`. Last invocation took 3 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

