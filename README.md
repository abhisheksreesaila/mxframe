<img src="visualize/mx-spark.svg" alt="MXFrame MX lightning mark" width="72">

# MXFrame

**A DataFrame engine written in Mojo: Polars-style API, faster than DuckDB on the TPC-H SF10 total.**

[![PyPI](https://img.shields.io/pypi/v/mxframe)](https://pypi.org/project/mxframe/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-lightgrey)](pyproject.toml)
[![Docs](https://img.shields.io/badge/docs-interactive-6C3E78)](https://abhisheksreesaila.github.io/mxframe/)

📖 **[Interactive docs](https://abhisheksreesaila.github.io/mxframe/)**: [benchmarks](https://abhisheksreesaila.github.io/mxframe/benchmarks.html) · [how `device="auto"` decides](https://abhisheksreesaila.github.io/mxframe/auto-mode.html) · [behind the scenes](https://abhisheksreesaila.github.io/mxframe/internals.html) · [how it got fast](https://abhisheksreesaila.github.io/mxframe/how-it-got-fast.html)

## Why MXFrame?

- **Fast on one machine.** A query plan runs in one call into a native Mojo executor, on every CPU core. It ran all 22 TPC-H queries at SF10 in 3.15 s, against 5.08 s for DuckDB and 14.40 s for Polars.
- **Nothing new to learn.** A lazy, Polars-style DataFrame API plus SQL, over Apache Arrow. Results come back as `pyarrow` tables.
- **The GPU when you have one.** `compute(device="auto")` measures your machine and splits work between the GPU and the CPU.

## Latest release: 0.6.0

The native Mojo CPU executor ([release notes](RELEASE_NOTES_0.6.0.md)). All 22 TPC-H queries, query time only (fresh process, data already loaded, no warm-up), every MXFrame result equal to DuckDB's:

| Run | MXFrame | DuckDB | Polars |
|---|---|---|---|
| SF10 | 3.15 s | 5.08 s | 14.40 s |
| SF1 | 0.47 s | 0.63 s | 1.31 s |
| SF10 shuffled | 4.97 s | 5.55 s | 15.06 s |
| SF1 shuffled | 0.61 s | 0.66 s | 1.41 s |

Honestly: Q21 carries the SF10 lead (without it DuckDB is ahead, 2.27 s to 2.41 s), and DuckDB is faster on 12 of the 22 queries, mostly join-heavy ones. On 250 random queries MXFrame is 1.09x DuckDB's speed at SF10 and 0.83x at SF1. [Every query, interactively →](https://abhisheksreesaila.github.io/mxframe/benchmarks.html)

## Install and run

```sh
pip install "mxframe[runtime]"     # pixi users: pixi add mxframe (MAX comes from conda)
```

```python
import mxframe as mx

lf = mx.scan_parquet("lineitem.parquet")
out = (lf.filter(mx.col("l_quantity") < 24)
         .groupby("l_returnflag")
         .agg(mx.col("l_extendedprice").sum().alias("revenue"))
         .sort("l_returnflag")
         .compute())          # a pyarrow Table
print(out.to_pandas())
```

```
  l_returnflag       revenue
0            A  1.222429e+10      # TPC-H SF1 lineitem, 6M rows
1            N  2.520421e+10
2            R  1.223160e+10
```

Nothing runs until `.compute()`; then the plan is optimized and executed in one native call.

## API reference

Examples use `lf` (a `LazyFrame` over columns `id, dept, x, price, name, day`) and `lf2` (`id, region`). Every example below is run by `tests/test_readme_api.py`.

**Sources and running**

| Call | What it does | Example |
|---|---|---|
| `mx.scan_parquet(path)` | Lazy Parquet scan (file or directory) | `mx.scan_parquet("t.parquet")` |
| `mx.LazyFrame(table)` | Wrap a `pyarrow.Table` | `mx.LazyFrame(t)` |
| `mx.sql(query, **tables)` | SQL over named tables | `mx.sql("SELECT dept, SUM(x) AS s FROM t GROUP BY dept", t=t)` |
| `.compute(device)` | Run: `"auto"` (default), `"cpu"`, `"gpu"` | `lf.compute(device="cpu")` |
| `.explain()` | The optimized plan, as text | `lf.filter(col("x") > 2).explain()` |

**LazyFrame**

| Method | What it does | Example |
|---|---|---|
| `.filter(expr)` | Keep rows where `expr` is true | `lf.filter(col("x") > 10)` |
| `.select(*cols)` | Pick or compute columns | `lf.select("id", (col("price") * 2).alias("p2"))` |
| `.with_columns(*exprs)` | Add or replace columns | `lf.with_columns((col("x") + 1).alias("x1"))` |
| `.groupby(*keys).agg(*aggs)` | Grouped aggregation | `lf.groupby("dept").agg(col("price").sum().alias("s"))` |
| `.join(other, on, how)` | `how`: inner, left, semi, anti | `lf.join(lf2, on="id", how="left")` |
| `.sort(*by, descending)` | Order rows | `lf.sort("price", descending=True)` |
| `.limit(n)` / `.head(n)` | First n rows | `lf.sort("x").limit(2)` |
| `.distinct(*subset)` | Unique rows | `lf.select("dept").distinct()` |
| COUNT(DISTINCT b) per a | distinct, then count | `lf.select("dept", "x").distinct().groupby("dept").agg(col("x").count().alias("n"))` |
| `.to_pandas()` / `.to_polars()` | Run and convert | `lf.to_pandas()` |

**Expressions**

| Expression | What it does | Example |
|---|---|---|
| `col(name)`, `lit(v)` | A column, a constant | `col("price") * lit(1.1)` |
| `+ - * /`, `== != < <= > >=` | Arithmetic, comparison | `(col("price") - col("x")) > 0` |
| `& \| ~` | And, or, not | `(col("x") > 2) & ~(col("dept") == "ops")` |
| `.isin(values)` | Membership | `col("dept").isin(["eng", "hr"])` |
| `.between(lo, hi)` | Inclusive range | `col("x").between(2, 15)` |
| `.startswith(s)`, `.contains(s)` | String predicates | `col("name").startswith("al")` |
| `.str_slice(offset, length)` | Substring | `col("name").str_slice(0, 2)` |
| `.year()` | Year of a date | `col("day").year()` |
| `when(cond, then, otherwise)` | Conditional | `when(col("x") > 10, col("price"), 0.0)` |
| `.sum() .mean() .min() .max() .count()` | Aggregates (in `.agg`) | `lf.groupby("dept").agg(col("x").mean().alias("m"))` |
| `.over(*by, order_by)` | Window | `lf.select("id", col("price").sum().over("dept").alias("w"))` |
| `.rank() .dense_rank() .lag(n) .lead(n) .cum_sum()` | Window functions | `lf.select("id", col("price").cum_sum().over("dept", order_by="id").alias("c"))` |
| `.alias(name)` | Name the result | `lf.groupby("dept").agg(col("x").sum().alias("total"))` |

SQL covers `SELECT`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN … ON` and `CASE WHEN`. Known limits in 0.6.0: SQL `COUNT(DISTINCT …)` isn't supported (use the DataFrame form above), and SQL `HAVING` is dropped without an error.

## More

| | |
|---|---|
| [Benchmarks (interactive)](https://abhisheksreesaila.github.io/mxframe/benchmarks.html) | Every TPC-H and random query, against DuckDB and Polars; full tables in [docs/benchmarks.md](docs/benchmarks.md) |
| [Behind the scenes](https://abhisheksreesaila.github.io/mxframe/internals.html) | The path a query takes, for porting or extending the engine |
| [How it got fast](https://abhisheksreesaila.github.io/mxframe/how-it-got-fast.html) | Each optimization, and which wins are Mojo; long form in [docs/optimizations.md](docs/optimizations.md) |
| [How `device="auto"` decides](https://abhisheksreesaila.github.io/mxframe/auto-mode.html) | CPU, GPU or both, from costs measured on your machine |
| [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) | Release history; dev setup and writing Mojo kernels |

Build from source with [pixi](https://pixi.sh): `git clone https://github.com/abhisheksreesaila/mxframe && cd mxframe && pixi install && pixi run test`.

License: Apache 2.0.
