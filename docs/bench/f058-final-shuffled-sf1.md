# Engine-only TPC-H scoreboard (rows shuffled) - 2026-10-17

**Rows shuffled:** after the untimed load every table's rows are put in a random order (seeded per table, the same for every engine), so no engine gains from TPC-H's stored order (lineitem by order key, ...).

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | Polars | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 166 †3e-3 | 19 | 10 | **9.4** | - |
| Q2 min-cost supplier | **8.3** | 10.0 | 9.3 | 9.8 | - |
| Q3 3-join+top10 | 25 | **16** | 24 | 24 | - |
| Q4 EXISTS semi-join | 75 | **16** | 17 | 17 | - |
| Q5 4-join+groupby | 40 | 15 | **14** | 17 | - |
| Q6 filter+global sum | 9.4 | **4.9** | 5.9 | 6.4 | - |
| Q7 volume shipping | 114 | 17 | **16** | 21 | - |
| Q8 market share | **11** | 15 | 15 | 19 | - |
| Q9 6-join profit | 56 †2e-6 | **35** | 40 | 43 | - |
| Q10 returned items top20 | 28 | **20** | 23 | 29 | - |
| Q11 important stock | **4.0** | 5.8 | 8.7 | 9.2 | - |
| Q12 shipping modes | 64 | **13** | 17 | 19 | - |
| Q13 customer distribution | 49 | 26 | **7.6** | 8.2 | PyArrow/NumPy |
| Q14 promo revenue | 7.9 | 11 | **7.5** | 7.8 | - |
| Q15 top supplier | 10 | **9.1** | 11 | 11 | - |
| Q16 part/supplier | **17** | 22 | 29 | 29 | - |
| Q17 small-qty revenue | **7.1** | 8.8 | 9.5 | 11 | - |
| Q18 large volume top100 | 113 | **79** | 96 | 106 | - |
| Q19 discounted revenue | 75 | 18 | **13** | 14 | - |
| Q20 potential promotion | 17 | **14** | 14 | 21 | - |
| Q21 suppliers waiting | 374 | 226 | **136** | 138 | - |
| Q22 global sales opportunity | **12** | 14 | 17 | 17 | - |
| **sum over the 22 queries valid for every engine** | **1,282** | **615** | **540** | **585** |  |

Fastest verified engine per query: Polars 6, DuckDB 9, mx-cpu 6, mx-auto 1.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats Polars on 15/22** (22 comparable); **mx-cpu beats DuckDB on 8/22** (22 comparable)
- **mx-auto beats Polars on 13/22** (22 comparable); **mx-auto beats DuckDB on 6/22** (22 comparable)

### Where mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-auto | 9.2 | 4.0 | 2.3x |
| Q8 market share | mx-auto | 19 | 11 | 1.8x |
| Q16 part/supplier | mx-auto | 29 | 17 | 1.7x |
| Q17 small-qty revenue | mx-auto | 11 | 7.1 | 1.6x |
| Q22 global sales opportunity | mx-auto | 17 | 12 | 1.4x |
| Q20 potential promotion | mx-auto | 21 | 17 | 1.3x |
| Q2 min-cost supplier | mx-auto | 9.8 | 8.3 | 1.2x |
| Q15 top supplier | mx-auto | 11 | 10 | 1.1x |
| Q10 returned items top20 | mx-auto | 29 | 28 | 1.0x |

### Where best of mx-cpu / mx-gpu / mx-auto loses to Polars at SF1 (worst first)

| query | engine | MXFrame ms | Polars ms | slower by |
|---|---:|---:|---:|---:|
| Q11 important stock | mx-cpu | 8.7 | 4.0 | 2.2x |
| Q16 part/supplier | mx-auto | 29 | 17 | 1.7x |
| Q8 market share | mx-cpu | 15 | 11 | 1.4x |
| Q22 global sales opportunity | mx-auto | 17 | 12 | 1.4x |
| Q17 small-qty revenue | mx-cpu | 9.5 | 7.1 | 1.4x |
| Q2 min-cost supplier | mx-cpu | 9.3 | 8.3 | 1.1x |
| Q15 top supplier | mx-cpu | 11 | 10 | 1.1x |

### SF1: import + first run, ms (engine-only + engine import time)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 242 †3e-3 | 130 | 117 | 112 |
| Q2 | 81 | 119 | 114 | 113 |
| Q3 | 98 | 127 | 129 | 128 |
| Q4 | 151 | 128 | 133 | 122 |
| Q5 | 113 | 124 | 117 | 125 |
| Q6 | 83 | 114 | 111 | 109 |
| Q7 | 189 | 128 | 117 | 123 |
| Q8 | 84 | 125 | 119 | 124 |
| Q9 | 133 †2e-6 | 147 | 141 | 147 |
| Q10 | 104 | 134 | 125 | 134 |
| Q11 | 77 | 117 | 115 | 114 |
| Q12 | 138 | 128 | 120 | 121 |
| Q13 | 122 | 136 | 112 | 112 |
| Q14 | 82 | 123 | 113 | 114 |
| Q15 | 83 | 119 | 119 | 117 |
| Q16 | 89 | 132 | 136 | 132 |
| Q17 | 82 | 119 | 116 | 114 |
| Q18 | 186 | 189 | 207 | 209 |
| Q19 | 149 | 127 | 119 | 117 |
| Q20 | 91 | 125 | 119 | 124 |
| Q21 | 448 | 342 | 243 | 245 |
| Q22 | 85 | 124 | 122 | 125 |
| **sum (22 queries)** | **2,909** | **3,056** | **2,863** | **2,884** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 66 | 149 | 56 | 58 |
| Q2 | 33 | 82 | 28 | 27 |
| Q3 | 56 | 134 | 84 | 83 |
| Q4 | 48 | 115 | 73 | 70 |
| Q5 | 61 | 119 | 90 | 91 |
| Q6 | 34 | 83 | 36 | 41 |
| Q7 | 59 | 119 | 93 | 93 |
| Q8 | 76 | 154 | 120 | 129 |
| Q9 | 83 | 188 | 137 | 144 |
| Q10 | 66 | 129 | 85 | 85 |
| Q11 | 26 | 65 | 24 | 24 |
| Q12 | 68 | 152 | 76 | 75 |
| Q13 | 30 | 52 | 36 | 32 |
| Q14 | 40 | 97 | 93 | 91 |
| Q15 | 42 | 85 | 49 | 49 |
| Q16 | 27 | 66 | 25 | 25 |
| Q17 | 42 | 84 | 88 | 88 |
| Q18 | 47 | 84 | 76 | 77 |
| Q19 | 72 | 173 | 100 | 97 |
| Q20 | 59 | 131 | 107 | 101 |
| Q21 | 53 | 110 | 84 | 81 |
| Q22 | 26 | 53 | 23 | 25 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| Q1 | 151 | 18 | 8.6 | 9.4 |
| Q2 | 6.6 | 9.7 | 7.1 | 7.9 |
| Q3 | 22 | 15 | 16 | 19 |
| Q4 | 68 | 15 | 13 | 14 |
| Q5 | 35 | 15 | 11 | 11 |
| Q6 | 8.8 | 5.1 | 4.1 | 4.5 |
| Q7 | 97 | 16 | 14 | 13 |
| Q8 | 10.0 | 13 | 12 | 12 |
| Q9 | 54 | 33 | 32 | 35 |
| Q10 | 28 | 20 | 16 | 18 |
| Q11 | 3.6 | 5.0 | 6.0 | 7.2 |
| Q12 | 66 | 13 | 14 | 16 |
| Q13 | 42 | 23 | 6.0 | 6.7 |
| Q14 | 8.1 | 13 | 5.0 | 5.6 |
| Q15 | 11 | 8.9 | 8.5 | 11 |
| Q16 | 13 | 25 | 25 | 26 |
| Q17 | 5.4 | 7.8 | 6.9 | 7.0 |
| Q18 | 91 | 76 | 88 | 98 |
| Q19 | 73 | 18 | 12 | 14 |
| Q20 | 16 | 11 | 11 | 14 |
| Q21 | 346 | 234 | 85 | 85 |
| Q22 | 12 | 16 | 15 | 15 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | Polars | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|---:|
| import | 74 | 110 | 105 | 104 |

### SF1: correctness against DuckDB and failures

| query | engine | verdict | max rel. error | note |
|---|---|---|---:|---|
| Q1 | polars | drift | 3.13e-03 | max relative error 3.13e-03 |
| Q9 | polars | drift | 1.86e-06 | max relative error 1.86e-06 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): Polars 3.1e-03, mx-cpu 2.6e-15, mx-auto 3.6e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 3.7, max 5.8 over 88 runs; load average at report time: 4.0 4.6 8.4), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `8e001f3`. Last invocation took 2 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

