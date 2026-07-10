# 🔤 MXFrame — API Reference

---

## LazyFrame

```python
from mxframe import LazyFrame, Scan, col, lit, when

lf = LazyFrame(Scan(arrow_table))
```

| Method | Description | Example |
|---|---|---|
| `.filter(expr)` | Row filter | `.filter(col("x") > lit(10))` |
| `.select(*cols)` | Column projection | `.select("a", "b", col("c").alias("d"))` |
| `.with_columns(*exprs)` | Add/replace columns | `.with_columns((col("a") * lit(2)).alias("a2"))` |
| `.groupby(*keys)` | Start grouped aggregation | `.groupby("dept", "region")` |
| `.agg(*exprs)` | Aggregation expressions (follows `.groupby`) | `.agg(col("x").sum().alias("s"))` |
| `.join(other, left_on, right_on, how)` | Hash join | `.join(lf2, "id", "fk_id", how="inner")` |
| `.sort(expr, descending)` | Sort rows | `.sort(col("revenue"), descending=True)` |
| `.limit(n)` | Take first N rows | `.limit(100)` |
| `.distinct()` | Deduplicate rows | `.distinct()` |
| `.compute(device)` | Execute the plan | `.compute(device="gpu")` |

`how` options: `"inner"` (default), `"left"`.

---

## Expressions

```python
from mxframe import col, lit, when
```

### Arithmetic & comparison

```python
col("price") * (lit(1.0) - col("discount"))   # arithmetic
col("date") >= lit(19940101)                   # comparison
(col("x") > lit(0)) & (col("y") < lit(100))   # boolean AND
col("flag") | col("other")                     # boolean OR
~col("active")                                 # boolean NOT
```

### String predicates

```python
col("phone").startswith("13")
col("comment").contains("special")
col("type").isin(["PROMO", "BRAND"])
```

### Date extraction

```python
col("orderdate").year()    # extract year as int32
```

### Conditional

```python
when(col("nation") == lit("BRAZIL"), col("revenue"), lit(0.0))
```

### Aggregations *(inside `.agg()`)*

```python
col("salary").sum()
col("salary").mean()
col("salary").min()
col("salary").max()
col("id").count()
```

### Aliasing

```python
col("x").sum().alias("total_x")
```

---

## SQL Frontend

```python
from mxframe.sql_frontend import sql

result = sql("""
    SELECT dept, SUM(salary) AS total, COUNT(*) AS n
    FROM employees
    WHERE age > 30
    GROUP BY dept
    ORDER BY total DESC
""", employees=arrow_table)
```

Pass tables as keyword arguments matching the table names in the SQL string.
Supported SQL: `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN … ON`, `CASE WHEN`.

---

## Supported Operations

| Category | Operations |
|---|---|
| **Filter** | `>`, `>=`, `<`, `<=`, `==`, `!=`, `&`, `|`, `~`, `isin`, `startswith`, `contains` |
| **Aggregation** | `sum`, `mean`, `min`, `max`, `count` |
| **Groupby** | Single key, multi-key, composite key |
| **Join** | Inner, Left outer |
| **Sort** | Single/multi column, ascending/descending |
| **Date** | `year()` extraction |
| **Projection** | `select`, `with_columns`, `alias`, arithmetic expressions |
| **Semi-join** | Via unique-key inner join |
| **Anti-join** | Via `isin` + invert |
| **Distinct** | Full row deduplication |
| **SQL** | `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN` |

---

## Device Selection

```python
result = lf.compute(device="cpu")   # default — works everywhere
result = lf.compute(device="gpu")   # requires NVIDIA/AMD/Apple Silicon + MAX runtime
```

The GPU path uses Mojo's `DeviceContext` — same source compiles to PTX (NVIDIA), HSA/ROCm (AMD), or Metal (Apple Silicon).

---

## Running Tests

```bash
# AOT kernel smoke tests (26 tests)
pixi run python3 scripts/_test_aot_smoke.py

# All 22 TPC-H correctness checks
pixi run python3 scripts/_test_phase6_tpch_tier2.py

# GPU device check
pixi run python3 scripts/_check_gpu.py

# Full 22-query benchmark (1M rows)
pixi run python3 scripts/bench_simple.py --rows 1000000 --runs 3
```

---

## Project Structure

```
mxframe/
├── __init__.py            ← Public API (LazyFrame, Scan, col, lit, when, sql)
├── lazy_frame.py          ← LazyFrame, LazyGroupBy, Scan
├── lazy_expr.py           ← Expr, col(), lit(), when()
├── compiler.py            ← LogicalPlan → MAX Graph compiler
├── custom_ops.py          ← Dispatch: AOT kernels / PyArrow fallback
├── optimizer.py           ← Plan rewrites (filter pushdown, join reordering)
├── plan_validation.py     ← Pre-execution plan checks
├── sql_frontend.py        ← SQL → LogicalPlan via sqlglot
│
├── kernels_aot/           ← Pre-compiled AOT shared libraries
│   ├── libmxkernels_aot.so      ← CPU kernels (ctypes-callable)
│   └── libmxkernels_aot_gpu.so  ← GPU kernels (CUDA/ROCm/Metal)
│
├── kernels/               ← Mojo kernel source (build time only)
│   ├── group_sum.mojo, group_min.mojo, group_max.mojo ...
│   ├── join_scatter.mojo, join_count.mojo
│   ├── join_scatter_left.mojo, join_count_left.mojo
│   └── filter_gather.mojo, unique_mask.mojo ...
│
├── kernels_aot/
│   ├── kernels_aot.mojo         ← CPU AOT entry points
│   └── kernels_aot_gpu.mojo     ← GPU AOT entry points
│
└── scripts/
    ├── benchmark_tpch.py        ← All 22 TPC-H query implementations
    ├── bench_simple.py          ← 4-column benchmark (Pandas|Polars|MX CPU|MX GPU)
    ├── _test_aot_smoke.py       ← AOT kernel smoke tests
    └── quickstart.py            ← Minimal hello-world example
```
