# MXFrame User Guide

MXFrame is a lazy DataFrame library backed by [Modular MAX](https://docs.modular.com/max/) graph compilation and Mojo kernels. It provides a Polars-style API that compiles to high-performance CPU/GPU kernels at query time.

---

## Installation

### Development / editable install (current)

```bash
cd /path/to/mxdf_v2
pixi install          # sets up the conda/mojo environment
pip install -e .      # installs mxframe into the pixi Python env
```

### Running scripts

Always run scripts via `pixi run` to ensure the MAX SDK load-library paths are active:

```bash
pixi run python3 scripts/benchmark_tpch.py
pixi run python3 my_analysis.py
```

Running with plain `python3` works for PyArrow-only paths but Mojo kernel compilation (grouped aggregations, sort, join) will fail without the pixi environment variables.

---

## Quick Start

```python
import pyarrow as pa
import mxframe as mx
from mxframe import col, lit

# 1. Create a LazyFrame from a PyArrow Table
table = pa.table({
    "region": ["A", "B", "A", "B", "A"],
    "sales":  [10.0, 20.0, 30.0, 40.0, 50.0],
    "qty":    [1, 2, 3, 4, 5],
})
lf = mx.from_arrow(table)

# 2. Filter rows
result = lf.filter(col("sales") > lit(15.0)).compute()

# 3. Select / compute new columns
result = lf.select(
    col("region"),
    (col("sales") * col("qty")).alias("revenue"),
).compute()

# 4. GroupBy + aggregate
result = (
    lf
    .groupby("region")
    .agg(
        col("sales").sum().alias("total_sales"),
        col("qty").mean().alias("avg_qty"),
        col("sales").count().alias("n"),
    )
    .sort("total_sales", descending=True)
    .compute()
)

# 5. Execute — specify device
result = lf.groupby("region").agg(col("sales").sum()).compute(device="cpu")   # CPU
result = lf.groupby("region").agg(col("sales").sum()).compute(device="gpu")   # GPU (raises if unavailable)
result = lf.groupby("region").agg(col("sales").sum()).compute(device="auto")  # GPU if available + N > 10k rows
```

---

## Core API

### Creating a LazyFrame

```python
import mxframe as mx
import pyarrow as pa

lf = mx.from_arrow(pa_table)          # from PyArrow Table
lf = mx.from_pandas(df)               # from pandas DataFrame
lf = mx.from_polars(pl_df)            # from Polars DataFrame
lf = mx.read_csv("path/to/file.csv")  # read CSV
lf = mx.read_parquet("path.parquet")  # read Parquet
```

Or directly:

```python
from mxframe.lazy_frame import LazyFrame
lf = LazyFrame(pa_table)
```

### Expressions (`col`, `lit`, `when`)

```python
from mxframe import col, lit, when

col("price")               # reference a column
lit(1.5)                   # scalar literal
col("price") * lit(0.9)    # arithmetic: +, -, *, /
col("x") > lit(10)         # comparison: >, >=, <, <=, ==, !=
col("flag") & col("active")  # boolean and
col("flag") | col("other")   # boolean or
~col("flag")                 # boolean not

# Conditional (CASE WHEN)
when(col("x") > lit(0), col("x"), lit(0.0)).alias("positive_x")

# String operations
col("name").startswith("PROMO")
col("name").contains("STEEL")
col("code").isin(["A", "B", "C"])

# Range check
col("discount").between(0.05, 0.07)

# Date extraction (YYYYMMDD integer columns)
col("l_shipdate").year()

# Alias
col("price").alias("list_price")
(col("price") * lit(1.1)).alias("price_with_tax")
```

### Aggregations

```python
col("x").sum()
col("x").min()
col("x").max()
col("x").mean()
col("x").count()
```

### LazyFrame methods

| Method | Description |
|--------|-------------|
| `.filter(predicate)` | Remove rows not matching predicate |
| `.select(*exprs)` | Project / compute columns (replaces all) |
| `.with_columns(*exprs)` | Add / replace columns (keeps existing) |
| `.groupby(*keys)` | Return a LazyGroupBy |
| `.sort(*by, descending=False)` | Sort by one or more columns |
| `.limit(n)` | Take first N rows |
| `.head(n=5)` | Alias for `.limit(n)` |
| `.tail(n=5)` | Take last N rows |
| `.distinct(*subset)` | Deduplicate rows (optionally on subset) |
| `.join(other, left_on, right_on, how="inner")` | Hash join two LazyFrames |
| `.compute(device="auto")` | Execute and return `pa.Table` |
| `.explain()` | Print logical plan |
| `.schema` | Infer output schema without full execution |
| `.to_pandas()` | Compute and convert to pandas |
| `.to_polars()` | Compute and convert to Polars |
| `.to_arrow()` | Compute and return PyArrow Table |

### GroupBy aggregation

```python
lf.groupby("region", "year").agg(
    col("revenue").sum().alias("total"),
    col("qty").mean().alias("avg_qty"),
    col("revenue").count().alias("n"),
    col("price").min().alias("min_price"),
    col("price").max().alias("max_price"),
)

# Global aggregation (no group keys)
lf.groupby().agg(col("revenue").sum())
```

### Window / analytic functions

```python
# Partitioned window — broadcast group aggregate to every row
col("sales").sum().over("region")
col("sales").mean().over("region")

# Ranking within partition
col("score").rank().over("region", order_by="score")
col("score").dense_rank().over("region", order_by="score")
col("score").row_number().over("region", order_by="date")

# Running totals and offsets
col("sales").cum_sum().over("region", order_by="date")
col("price").lag(1).over("region", order_by="date")
col("price").lead(1).over("region", order_by="date")

# Global (no partition)
col("sales").rank()
col("sales").cum_sum()
```

### Joins

```python
orders = LazyFrame(orders_table)
customers = LazyFrame(customers_table)

joined = orders.join(customers, left_on="o_custkey", right_on="c_custkey")
# how= "inner" (default), "left"

# Chained joins (TPC-H style)
result = (
    LazyFrame(orders)
    .join(LazyFrame(customers), left_on="o_custkey", right_on="c_custkey")
    .join(LazyFrame(lineitem),  left_on="o_orderkey", right_on="l_orderkey")
    .filter(col("c_mktsegment") == lit("BUILDING"))
    .groupby("o_orderkey", "o_orderdate")
    .agg(
        (col("l_extendedprice") * (lit(1) - col("l_discount"))).sum().alias("revenue")
    )
    .sort("revenue", descending=True)
    .limit(10)
    .compute()
)
```

### SQL frontend

```python
import mxframe as mx

result = mx.sql(
    """
    SELECT l_returnflag, l_linestatus,
           SUM(l_quantity)      AS sum_qty,
           SUM(l_extendedprice) AS sum_base_price,
           COUNT(*)             AS count_order
    FROM lineitem
    WHERE l_shipdate <= 10471
    GROUP BY l_returnflag, l_linestatus
    ORDER BY l_returnflag, l_linestatus
    """,
    lineitem=lineitem_table,   # pa.Table
).compute()
```

Supported SQL:
- `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`
- `INNER JOIN`, `LEFT JOIN … ON`
- `SUM`, `AVG`, `MIN`, `MAX`, `COUNT(*)`
- `CASE WHEN … THEN … ELSE … END`
- `BETWEEN`, `IN (…)`, `NOT IN`, `LIKE 'prefix%'`
- `AND`, `OR`, `NOT`
- Arithmetic: `+`, `-`, `*`, `/`

---

## Device Selection

```python
# Auto: CPU for N ≤ 10 000 rows, GPU for N > 10 000 when a supported GPU is present
result = lf.compute(device="auto")   # default

# Force CPU
result = lf.compute(device="cpu")

# Force GPU — raises RuntimeError if no supported GPU (A10/A100/H100 etc.)
result = lf.compute(device="gpu")
```

Consumer GPUs (RTX 30xx/40xx) are not currently supported by MAX Engine. Use `device="cpu"` or `device="auto"` on those.

---

## Performance Tips

| Tip | Reason |
|-----|--------|
| Use `pixi run python3` | Required for Mojo JIT compilation |
| Filter before groupby | Filters are pushed down eagerly in PyArrow |
| Use `.between()` instead of `(col >= lo) & (col <= hi)` on single columns | Slightly cleaner IR |
| `device="auto"` at scale (>100K rows) will use GPU automatically | No code changes needed |
| Reuse a `LazyFrame` object across `.compute()` calls | Graph is cached — second call is much faster |
| Call `from mxframe.custom_ops import clear_cache` after changing kernel source | Clears compiled graph cache |

---

## Debugging

```python
# Print the logical plan
print(lf.explain())

# Print with optimization trace
print(lf.explain(optimized=True))

# See runtime execution provenance after compute()
result = lf.compute(verbose=True)
print(lf.last_compile_provenance)
# e.g. {'device': 'gpu', 'rows': 1000000, 'path': 'grouped_mojo', 'execute_ms': 12.4, ...}

# Clear compiled model cache (needed after editing .mojo kernel files)
from mxframe.custom_ops import clear_cache
clear_cache()
```

---

## All 22 TPC-H Queries

MXFrame passes all 22 TPC-H queries against DuckDB reference results. Each query is available in both the DataFrame API and SQL frontend.

| Tier | Queries | Key Capabilities |
|------|---------|-----------------|
| 1 | Q1, Q6 | filter + groupby + agg + sort |
| 2 | Q12, Q14 | 2-table join + CASE WHEN + isin |
| 3 | Q3, Q5, Q10 | 3–6 table join + limit |
| 4 | Q4, Q11, Q15, Q17, Q18, Q20, Q21, Q22 | subquery patterns via two-pass join |
| 5 | Q2, Q7, Q8, Q9, Q13, Q16, Q19 | string ops + year extraction + complex aggs |

See `scripts/benchmark_tpch.py` for runnable implementations of all queries.
