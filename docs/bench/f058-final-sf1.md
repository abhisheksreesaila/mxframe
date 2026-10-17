# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 245 †3e-3 | 20 | 11 | **11** | - |
| Q2 min-cost supplier | **7.4** | 11 | 9.6 | 9.3 | - |
| Q3 3-join+top10 | 31 | **16** | 24 | 24 | - |
| Q4 EXISTS semi-join | 73 | **17** | 18 | 17 | - |
| Q5 4-join+groupby | 48 | 16 | **13** | 18 | - |
| Q6 filter+global sum | 6.2 | **5.9** | 11 | 6.7 | - |
| Q7 volume shipping | 99 | 18 | **16** | 20 | - |
| Q8 market share | **14** | 16 | 16 | 19 | - |
| Q9 6-join profit | 68 †2e-6 | 56 | **35** | 39 | - |
| Q10 returned items top20 | 37 | 23 | **20** | 26 | - |
| Q11 important stock | **4.0** | 6.5 | 15 | 15 | - |
| Q12 shipping modes | 20 | **14** | 14 | 14 | - |
| Q13 customer distribution | 51 | 27 | **7.7** | 8.0 | PyArrow/NumPy |
| Q14 promo revenue | 9.4 | 15 | 7.2 | **7.1** | - |
| Q15 top supplier | **8.2** | 8.3 | 10 | 11 | - |
| Q16 part/supplier | **15** | 25 | 27 | 31 | - |
| Q17 small-qty revenue | **7.0** | 9.3 | 10 | 11 | - |
| Q18 large volume top100 | 136 | **39** | 53 | 62 | - |
| Q19 discounted revenue | 40 | 20 | **14** | 15 | - |
| Q20 potential promotion | 15 | **13** | 16 | 23 | - |
| Q21 suppliers waiting | 359 | 226 | 79 | **78** | - |
| Q22 global sales opportunity | **15** | 17 | 17 | 17 | - |
| **sum over the 22 queries valid for every engine** | **1,307** | **618** | **445** | **482** |  |

Fastest verified engine per query: Polars 7, DuckDB 6, mx-cpu 6, mx-auto 3.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 13/22** (22 comparable); **mx-cpu beats DuckDB on 10/22** (22 comparable)
- **mx-auto beats Polars on 13/22** (22 comparable); **mx-auto beats DuckDB on 8/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 15 | 4.0 | 3.8x |
| Q16 part/supplier | mx-auto | 31 | 15 | 2.1x |
| Q17 small-qty revenue | mx-auto | 11 | 7.0 | 1.5x |
| Q20 potential promotion | mx-auto | 23 | 15 | 1.5x |
| Q8 market share | mx-auto | 19 | 14 | 1.4x |
| Q15 top supplier | mx-auto | 11 | 8.2 | 1.3x |
| Q2 min-cost supplier | mx-auto | 9.3 | 7.4 | 1.3x |
| Q22 global sales opportunity | mx-auto | 17 | 15 | 1.1x |
| Q6 filter+global sum | mx-auto | 6.7 | 6.2 | 1.1x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 15 | 4.0 | 3.8x |
| Q16 part/supplier | mx-cpu | 27 | 15 | 1.8x |
| Q17 small-qty revenue | mx-cpu | 10 | 7.0 | 1.5x |
| Q15 top supplier | mx-cpu | 10 | 8.2 | 1.3x |
| Q2 min-cost supplier | mx-auto | 9.3 | 7.4 | 1.3x |
| Q8 market share | mx-cpu | 16 | 14 | 1.2x |
| Q22 global sales opportunity | mx-auto | 17 | 15 | 1.1x |
| Q6 filter+global sum | mx-auto | 6.7 | 6.2 | 1.1x |
| Q20 potential promotion | mx-cpu | 16 | 15 | 1.0x |

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 319 †3e-3 | 131 | 114 | 115 |
| Q2 | 80 | 123 | 115 | 111 |
| Q3 | 108 | 125 | 131 | 125 |
| Q4 | 147 | 127 | 120 | 121 |
| Q5 | 120 | 125 | 116 | 120 |
| Q6 | 82 | 116 | 114 | 116 |
| Q7 | 176 | 130 | 120 | 126 |
| Q8 | 87 | 126 | 118 | 122 |
| Q9 | 139 †2e-6 | 168 | 137 | 143 |
| Q10 | 114 | 137 | 122 | 134 |
| Q11 | 78 | 115 | 120 | 129 |
| Q12 | 94 | 160 | 123 | 119 |
| Q13 | 124 | 139 | 109 | 115 |
| Q14 | 82 | 126 | 113 | 111 |
| Q15 | 81 | 117 | 112 | 113 |
| Q16 | 87 | 135 | 132 | 134 |
| Q17 | 81 | 118 | 116 | 116 |
| Q18 | 209 | 148 | 156 | 170 |
| Q19 | 113 | 131 | 118 | 117 |
| Q20 | 88 | 122 | 118 | 128 |
| Q21 | 432 | 336 | 185 | 180 |
| Q22 | 88 | 130 | 121 | 122 |
| **sum (22 queries)** | **2,928** | **3,085** | **2,730** | **2,788** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 69 | 148 | 54 | 55 |
| Q2 | 32 | 76 | 34 | 29 |
| Q3 | 58 | 127 | 76 | 81 |
| Q4 | 49 | 121 | 75 | 72 |
| Q5 | 62 | 119 | 89 | 90 |
| Q6 | 34 | 84 | 36 | 37 |
| Q7 | 63 | 125 | 97 | 95 |
| Q8 | 74 | 154 | 121 | 123 |
| Q9 | 86 | 180 | 144 | 142 |
| Q10 | 73 | 137 | 85 | 86 |
| Q11 | 26 | 59 | 19 | 17 |
| Q12 | 64 | 134 | 80 | 80 |
| Q13 | 32 | 46 | 30 | 35 |
| Q14 | 47 | 87 | 93 | 91 |
| Q15 | 42 | 83 | 46 | 43 |
| Q16 | 27 | 59 | 26 | 26 |
| Q17 | 44 | 89 | 89 | 89 |
| Q18 | 44 | 82 | 72 | 78 |
| Q19 | 76 | 179 | 99 | 94 |
| Q20 | 56 | 122 | 102 | 106 |
| Q21 | 54 | 114 | 77 | 80 |
| Q22 | 26 | 42 | 27 | 23 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 199 | 17 | 9.5 | 10 |
| Q2 | 6.0 | 10.0 | 7.3 | 7.6 |
| Q3 | 28 | 15 | 17 | 17 |
| Q4 | 62 | 15 | 13 | 14 |
| Q5 | 39 | 15 | 10 | 11 |
| Q6 | 6.1 | 4.8 | 4.6 | 4.8 |
| Q7 | 88 | 17 | 15 | 14 |
| Q8 | 13 | 15 | 12 | 13 |
| Q9 | 58 | 56 | 29 | 30 |
| Q10 | 33 | 24 | 15 | 17 |
| Q11 | 4.1 | 4.4 | 6.5 | 6.8 |
| Q12 | 20 | 13 | 13 | 12 |
| Q13 | 43 | 25 | 5.7 | 6.1 |
| Q14 | 7.4 | 14 | 5.0 | 5.6 |
| Q15 | 8.2 | 8.3 | 8.1 | 8.3 |
| Q16 | 15 | 22 | 24 | 24 |
| Q17 | 5.7 | 8.5 | 7.6 | 7.5 |
| Q18 | 117 | 33 | 41 | 46 |
| Q19 | 38 | 19 | 12 | 12 |
| Q20 | 16 | 12 | 12 | 18 |
| Q21 | 328 | 233 | 60 | 59 |
| Q22 | 14 | 13 | 15 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| import | 73 | 110 | 103 | 104 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.11e-03 | max relative error 3.11e-03 |
| Q9 | polars | drift | 1.88e-06 | max relative error 1.88e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 2.3e-15, mx-auto 2.3e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 2.0, max 2.7 over 88 runs; load average at report time: 2.7 4.7 8.8), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8e001f3`. Last invocation took 1 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

