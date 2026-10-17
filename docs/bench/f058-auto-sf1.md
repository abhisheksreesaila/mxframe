# Engine-only TPC-H scoreboard - 2026-10-17

**Engine-only:** a fresh Python process; each engine first loads the columns the query uses from Parquet into its own in-memory form (pandas / Polars DataFrame, DuckDB table, Arrow table under an MXFrame LazyFrame), outside the clock; the clock then covers *query + materialized result* only. No warm-up run. Loading data is staging; this is the headline measure (since 2026-10-13). The user-felt scoreboard (clock around read + query) is the secondary number. Load and `import` times are reported separately. Numbers are milliseconds; **bold** = fastest verified engine for that query.

Legend: `†` float32 accumulation drift (result within 1e-2 relative of DuckDB, error shown but outside the strict 1e-6 check), `WRONG` = differs from DuckDB (rows/keys, relative error above 1e-2, or rows not in the query's ORDER BY order), `?` = **unchecked** (the DuckDB reference failed for that query, so the result was not verified; it never counts as fastest or as a win), `TIMEOUT` / `CRASH` / `ERR` = did not produce a result (timeout 300s per subprocess). `outside mx` = the mx engines do part of that query with PyArrow/NumPy because the LazyFrame API cannot express it (details under Query semantics); every other mx query is pure mxframe.


## Scale factor 1: engine-only (query on loaded data, first run in a fresh process), ms

| query | DuckDB | mx-cpu | mx-auto | outside mx |
|---|---:|---:|---:|---:|
| Q1 filter+groupby+8 aggs | 20 | **14** | 18 | - |
| Q2 min-cost supplier | 13 | **13** | 16 | - |
| Q3 3-join+top10 | **16** | 27 | 34 | - |
| Q4 EXISTS semi-join | **16** | 20 | 23 | - |
| Q5 4-join+groupby | **16** | 17 | 20 | - |
| Q6 filter+global sum | **5.6** | 14 | 13 | - |
| Q7 volume shipping | **20** | 21 | 24 | - |
| Q8 market share | **15** | 21 | 23 | - |
| Q9 6-join profit | 56 | **38** | 41 | - |
| Q10 returned items top20 | **23** | 24 | 29 | - |
| Q11 important stock | **6.9** | 17 | 24 | - |
| Q12 shipping modes | **14** | 18 | 21 | - |
| Q13 customer distribution | 27 | **11** | 17 | PyArrow/NumPy |
| Q14 promo revenue | 15 | **11** | 13 | - |
| Q15 top supplier | **8.8** | 13 | 19 | - |
| Q16 part/supplier | **26** | 31 | 33 | - |
| Q17 small-qty revenue | **10.0** | 14 | 17 | - |
| Q18 large volume top100 | **33** | 57 | 65 | - |
| Q19 discounted revenue | 21 | **17** | 20 | - |
| Q20 potential promotion | **12** | 18 | 25 | - |
| Q21 suppliers waiting | 219 | **80** | 87 | - |
| Q22 global sales opportunity | **16** | 26 | 24 | - |
| **sum over the 22 queries valid for every engine** | **609** | **521** | **606** |  |

Fastest verified engine per query: DuckDB 15, mx-cpu 7, mx-auto 0.

### Wins at SF1 (engine-only, both results verified)

- **mx-cpu beats DuckDB on 7/22** (22 comparable)
- **mx-auto beats DuckDB on 6/22** (22 comparable)

### SF1: import + first run, ms (engine-only + engine import time)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 152 | 150 | 133 |
| Q2 | 129 | 118 | 121 |
| Q3 | 131 | 139 | 138 |
| Q4 | 128 | 126 | 129 |
| Q5 | 126 | 119 | 122 |
| Q6 | 117 | 117 | 120 |
| Q7 | 130 | 131 | 128 |
| Q8 | 124 | 126 | 132 |
| Q9 | 165 | 144 | 144 |
| Q10 | 132 | 131 | 137 |
| Q11 | 121 | 153 | 158 |
| Q12 | 128 | 152 | 132 |
| Q13 | 140 | 114 | 119 |
| Q14 | 126 | 114 | 116 |
| Q15 | 122 | 118 | 123 |
| Q16 | 136 | 138 | 145 |
| Q17 | 123 | 151 | 124 |
| Q18 | 143 | 160 | 170 |
| Q19 | 130 | 123 | 133 |
| Q20 | 123 | 122 | 129 |
| Q21 | 329 | 187 | 191 |
| Q22 | 125 | 133 | 126 |
| **sum (22 queries)** | **3,079** | **2,965** | **2,970** |

> `import mxframe` starts a background thread that warms the CPU kernel libraries. In this mode it finishes during the untimed load, so the mx first runs start with warm kernel libraries; the other engines' loads also start their thread pools / connections, but have no equivalent of that thread.

### SF1: load time, ms (Parquet -> in-memory tables, outside the clock)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 150 | 54 | 57 |
| Q2 | 79 | 38 | 28 |
| Q3 | 135 | 86 | 81 |
| Q4 | 120 | 72 | 73 |
| Q5 | 114 | 91 | 89 |
| Q6 | 84 | 32 | 37 |
| Q7 | 122 | 101 | 94 |
| Q8 | 148 | 126 | 123 |
| Q9 | 180 | 146 | 147 |
| Q10 | 131 | 87 | 89 |
| Q11 | 65 | 17 | 18 |
| Q12 | 154 | 76 | 78 |
| Q13 | 48 | 37 | 37 |
| Q14 | 88 | 92 | 94 |
| Q15 | 82 | 47 | 52 |
| Q16 | 58 | 24 | 26 |
| Q17 | 92 | 89 | 90 |
| Q18 | 81 | 75 | 76 |
| Q19 | 175 | 104 | 101 |
| Q20 | 129 | 108 | 107 |
| Q21 | 111 | 83 | 78 |
| Q22 | 54 | 28 | 27 |

> Each engine reads only the columns the query uses, but no engine can skip row groups with the query's filters here (Polars, DuckDB and MXFrame push filters into the reader in the user-felt run).

### SF1: repeat run in the same process, ms (secondary: warm caches / JIT / result caches)

| query | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| Q1 | 19 | 10 | 11 |
| Q2 | 9.8 | 7.0 | 7.2 |
| Q3 | 18 | 17 | 21 |
| Q4 | 16 | 12 | 13 |
| Q5 | 15 | 11 | 11 |
| Q6 | 5.0 | 4.5 | 4.9 |
| Q7 | 20 | 14 | 16 |
| Q8 | 14 | 13 | 13 |
| Q9 | 55 | 31 | 31 |
| Q10 | 22 | 16 | 18 |
| Q11 | 5.7 | 6.4 | 10 |
| Q12 | 14 | 14 | 14 |
| Q13 | 24 | 6.3 | 8.3 |
| Q14 | 13 | 5.3 | 5.4 |
| Q15 | 8.8 | 7.8 | 12 |
| Q16 | 23 | 26 | 26 |
| Q17 | 9.2 | 7.5 | 8.0 |
| Q18 | 31 | 41 | 47 |
| Q19 | 19 | 12 | 13 |
| Q20 | 11 | 11 | 18 |
| Q21 | 238 | 62 | 65 |
| Q22 | 15 | 15 | 16 |

> The repeat result is verified against DuckDB like the first one (`WRONG`/`?` marks as above). The tables are reloaded before the repeat, so result caches keyed by table identity (`custom_ops.py`) cannot answer it; it still runs with warm kernels, JIT and thread pools.

### SF1: import time, ms (median over queries; not in the first-run times)

|  | DuckDB | mx-cpu | mx-auto |
|---|---:|---:|---:|
| import | 111 | 106 | 105 |

Largest relative error vs DuckDB (over all queries where rows/keys matched): mx-cpu 1.8e-15, mx-auto 1.8e-15.


## Sanity check of the DuckDB reference against published TPC-H SF1 answers

The harness compares every engine with DuckDB, so the reference itself is checked against the published SF1 answers (full result, or the top rows for LIMIT queries): Q1 ok, Q2 ok, Q3 ok, Q4 ok, Q5 ok, Q6 ok, Q7 ok, Q8 ok, Q9 ok, Q10 ok, Q11 ok, Q12 ok, Q14 ok, Q15 ok, Q17 ok, Q18 ok, Q19 ok, Q20 ok, Q21 ok, Q22 ok.


## Methodology

- **One subprocess per (engine, query, scale factor)**; the worker imports the engine (timed as `import_ms`), loads the query's columns into memory (`load_ms`, outside the clock), then runs query+materialize once (`first_ms`) and, after reloading the tables, once more in the same process (`repeat_ms`). Reloading gives the repeat new table objects, so caches keyed by table identity cannot answer it.
- **Data** (exact): DuckDB `dbgen` Parquet (dates as int32 days since epoch, decimals as float32), read from the OS page cache (files are read once before each scale factor so no engine pays a cold disk).
- **Loading** (untimed): every engine is given the columns the query uses. pandas `pd.read_parquet(columns=...)`, Polars `pl.read_parquet(columns=...)`, DuckDB `CREATE TABLE ... AS SELECT <columns> FROM read_parquet(...)` (its native in-memory storage), MXFrame `mx.read_parquet_table(columns=..., dictionary="auto")` (the `scan_parquet` reader; low-cardinality strings as dictionaries).
- **Engines**: pandas, Polars (`df.lazy()` + `collect()`), DuckDB (the same SQL over the loaded tables, also the correctness reference), MXFrame `device="cpu"|"gpu"|"auto"` through the public `LazyFrame` API over the loaded Arrow tables; multi-stage queries `.compute()` a small intermediate and wrap it with `mx.LazyFrame(table)`. Work the API cannot express is done with PyArrow/NumPy inside the timed region and listed per query below.
- **What "loaded" means for MXFrame**: the Arrow tables stay as read (chunked, in host memory). Inside the clock MXFrame still combines chunks, hands buffers to its kernels and, on the GPU, uploads them; the other engines start from their own native in-memory format.
- **Correctness**: each result is compared with DuckDB (row count; integer/string columns exactly; float columns at 1e-6 relative, matched by key columns; LIMIT queries tolerate rows that tie at the cutoff within 1e-6) and then checked to be in the query's ORDER BY order. Inputs are float32; an engine that accumulates in float32 drifts on large sums (`†`), DuckDB sums in double. A one-row aggregate over no rows is NULL in SQL (DuckDB, MXFrame) and 0 in pandas and Polars; for pandas and Polars the comparison counts their 0 as the same answer (it happens only on tiny data).
- **Hardware**: AMD Ryzen 9 5900X 12-Core Processor (24 cores); GPU: NVIDIA GeForce RTX 3090, 610.57.04, 24576 MiB (nvidia-smi index 0 (NVIDIA GeForce RTX 3090), GPU-950714f3-1dcb-82c4-6f0b-241024860932; the workers get `CUDA_VISIBLE_DEVICES=<that UUID>`, so it is the same card nvidia-smi calls by that index). **The machine was shared with other jobs during this run (1-minute load average at measurement time: median 1.8, max 4.2 over 66 runs; load average at report time: 4.2 3.8 3.3), so timings are indicative.**
- **Versions**: pandas 3.0.2, polars 1.34.0, duckdb 1.5.1, pyarrow 22.0.0, numpy 2.4.3, max 26.6.0, mojo 1.1.0; mxframe at git `fcf5af0`. Last invocation took 0 min. Data root: `/home/abhisheksreesaila/Projects/mxframe/data`.

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

