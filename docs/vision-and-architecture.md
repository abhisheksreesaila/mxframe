# 🔭 MXFrame — Vision & Architecture

---

## 1. Vision

**MXFrame is a GPU-native DataFrame library built on MAX Graph and Mojo.**

It aims to be as easy to use as pandas/polars, but with a fundamentally different engine: every operation compiles to optimized Mojo code that runs on CPU *or* GPU — no CUDA, no Rapids retrofit, no Python interpreter in the hot path.

### Core Beliefs

| Belief | What it means |
|--------|--------------|
| **Mojo does all the work** | Python is the API surface. Mojo is the compute engine. MAX Graph is the translation layer that bridges the two. |
| **GPU-native, not GPU-retrofitted** | Unlike Rapids/cuDF which bolt GPU onto pandas, MXFrame is designed GPU-first. CPU is the fallback, not the other way around. |
| **Lazy by default** | Users build a logical plan (like Polars). The plan is compiled into a MAX Graph, optimized, then executed in one shot. No row-by-row Python. |
| **Prove it with benchmarks** | TPC-H benchmarks are the yardstick. If MXFrame can run TPC-H competitively, it's useful for real workloads. |
| **Developer-friendly** | Clean API, great docs, small notebooks, emoji-rich explanations. If someone can use pandas, they can use MXFrame. |

### Differentiators vs. Existing Libraries

| Library | How MXFrame differs |
|---------|-------------------|
| **pandas** | pandas is eager, single-threaded, CPU-only. MXFrame is lazy, compiled, GPU-capable. |
| **Polars** | Polars is Rust-based, CPU-only (GPU via cudf-polars plugin). MXFrame is Mojo-based, GPU-native from day one. |
| **cuDF / Rapids** | cuDF requires NVIDIA Rapids ecosystem, CUDA toolkit, complex setup. MXFrame uses MAX/Mojo — single `pixi install`. |
| **Spark** | Spark is distributed, JVM-heavy, high latency. MXFrame is single-node, zero-overhead, instant compilation. |

---

## 2. Architecture

### 2.1 The Five Layers (Two Frontends, One Plan)

The key insight: **SQL and the DataFrame API are just two frontends** that produce
the **same `LogicalPlan` AST**. The compiler and Mojo kernels don't care which
frontend built the plan — they see identical plan trees either way.

```
┌───────────────────────────┐  ┌──────────────────────────┐
│   🐍 DATAFRAME API        │  │   📝 SQL FRONTEND        │
│                           │  │                          │
│  df.filter(col('x') > 30) │  │  mx.sql("""              │
│    .groupby('dept')       │  │    SELECT dept,          │
│    .agg(col('sal').sum()) │  │           sum(sal)       │
│    .compute()             │  │    FROM t                │
│                           │  │    WHERE x > 30          │
│  Classes: LazyFrame,      │  │    GROUP BY dept         │
│  LazyGroupBy, Expr        │  │  """, t=arrow_table)     │
│  File: lazy_expr.py,      │  │                          │
│        lazy_frame.py      │  │  Parser: sqlglot         │
│                           │  │  File: sql_frontend.py   │
└─────────────┬─────────────┘  └────────────┬─────────────┘
              │ builds                       │ parses & builds
              └──────────────┬───────────────┘
                             ▼
┌─────────────────────────────────────────────────────────┐
│                 📐 LOGICAL PLAN LAYER                   │
│                                                         │
│   Aggregate(group_by=[col('dept')], aggs=[sum(salary)]) │
│       └─ Project(exprs=[col('name'), col('salary')*1.1])│
│           └─ Filter(predicate=col('age') > 30)          │
│               └─ Scan(table=arrow_table)                │
│                                                         │
│   ** Same plan tree regardless of frontend **           │
│                                                         │
│   Nodes: Scan, Filter, Project, Aggregate,              │
│          Sort, Join, Limit, Distinct  (planned)         │
└──────────────────────┬──────────────────────────────────┘
                       │ compiled by
                       ▼
┌─────────────────────────────────────────────────────────┐
│               ⚡ MAX GRAPH COMPILER LAYER               │
│                                                         │
│   GraphCompiler walks the plan tree and emits:          │
│   - Built-in MAX ops (ops.add, ops.sum, ops.gather)     │
│   - Custom Mojo kernels (ops.custom("group_sum", ...))  │
│                                                         │
│   The compiler decides:                                 │
│   - CPU vs GPU device placement                         │
│   - Which kernels to fuse                               │
│   - How to handle data layout & types                   │
│                                                         │
│   Classes: GraphCompiler, CustomOpsCompiler              │
│   File: compiler.py, custom_ops.py                      │
└──────────────────────┬──────────────────────────────────┘
                       │ dispatches to
                       ▼
┌─────────────────────────────────────────────────────────┐
│                  🔥 MOJO KERNEL LAYER                   │
│                                                         │
│   Pure Mojo code compiled to native CPU/GPU binaries.   │
│                                                         │
│   CPU Kernels:          GPU Kernels:                    │
│   - group_sum           - group_sum (warp reduction)    │
│   - group_min           - masked_sum (fused)            │
│   - group_max           - fused_group_agg               │
│   - group_mean          - fused_q1_full                 │
│   - group_count         - sort_kernel                   │
│   - scatter / gather    - hash_join                     │
│                                                         │
│   Files: mxframe/kernels/*.mojo                         │
│   Built to: mxframe/kernels.mojopkg                     │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow (End to End)

```
PyArrow Table
    │
    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  LazyFrame   │───►│ LogicalPlan  │───►│  MAX Graph   │
│  (Python)    │    │  (AST tree)  │    │  (compiled)  │
└──────────────┘    └──────────────┘    └──────┬───────┘
                                               │
                                    ┌──────────┴──────────┐
                                    ▼                     ▼
                              ┌──────────┐          ┌──────────┐
                              │ CPU Mojo │          │ GPU Mojo │
                              │ Kernels  │          │ Kernels  │
                              └────┬─────┘          └────┬─────┘
                                   │                     │
                                   └──────────┬──────────┘
                                              ▼
                                      PyArrow Table
                                       (result)
```

### 2.3 Device Strategy

The compiler decides CPU vs GPU based on:


| Factor | CPU | GPU |The strategy is **automatic with manual override**:

|--------|-----|-----|

| Data size | Small (< 100K rows) | Large (> 100K rows) || Kernel exists? | Fallback to MAX built-in ops | Only if Mojo GPU kernel is registered |

| Operation type | Scatter/gather, string ops | Reductions, arithmetic || GPU available? | Always | Only if `driver.GPU()` succeeds |

```python
# Automatic (default)
result = df.compute()

# Force GPU
result = df.compute(device="gpu")

# Force CPU
result = df.compute(device="cpu")
```

### 2.4 The Compiler's Job

The `GraphCompiler` (and its subclass `CustomOpsCompiler`) does:

1. **Walk the plan tree** bottom-up (Scan → Filter → Project → Aggregate)
2. **For each node**, emit MAX Graph ops:
   - `Scan` → map column names to `graph.inputs`
   - `Filter` → emit comparison ops, then `ops.gather` to select matching rows
   - `Project` → emit arithmetic/function ops for each expression
   - `Aggregate` → emit `ops.custom("group_sum", ...)` or `ops.sum(axis=0)`
3. **Load custom Mojo kernels** via `custom_extensions=[Path("kernels.mojopkg")]`
4. **Execute** via `session.load(graph)` → `model.execute(*inputs)`
5. **Convert results** back to PyArrow via `tensor.to_numpy()` → `pa.array()`

### 2.5 SQL Frontend — Same Plan, Different Syntax

SQL support works by parsing SQL text into our existing `LogicalPlan` nodes:

```
SQL string
    │
    ▼  (sqlglot parses)
SQLGlot AST
    │
    ▼  (sql_frontend.py translates)
LogicalPlan tree  ←── identical to what DataFrame API builds
    │
    ▼  (compiler compiles)

MAX Graph → Mojo kernels → result### 3.1 Syntax — Polars-Inspired, Simplified

```

## 3. API Design Philosophy

We use [`sqlglot`](https://github.com/tobymao/sqlglot) (already in `pixi.toml`) for parsing.

It handles SQL dialects, produces a clean AST, and we only need to write a---

translator from `sqlglot.Expression` → our `LogicalPlan` nodes.

- **Session caching** — compiled graphs are cached, subsequent runs are instant

**Why this works:** Our `LogicalPlan` is the universal intermediate representation.- **Mixed ops** — combine built-in MAX ops with custom Mojo kernels in one graph

Whether the user writes:- **Device placement** — the graph runtime handles CPU↔GPU data transfers

- **Fusion opportunities** — MAX can fuse adjacent ops into single kernel launches

```python- **Automatic memory management** — tensors flow through the graph, no manual alloc/free

# DataFrame APIMAX Graph provides:

df.filter(col('x') > 30).groupby('y').agg(col('z').sum().alias('total')).compute()

### 2.6 Why MAX Graph (Not Direct Mojo FFI)?

# SQL API

mx.sql("SELECT y, sum(z) AS total FROM t WHERE x > 30 GROUP BY y", t=table)completely unaware of which frontend was used.

```The compiler, the Mojo kernels, the GPU dispatch — everything downstream is



Both produce:```

```        └─ Scan(table)

Aggregate(group_by=[col('y')], aggs=[sum(col('z')).alias('total')])    └─ Filter(predicate=col('x') > lit(30))

```python
import mxframe as mx

# Read data
df = mx.read_csv("data.csv")          # returns LazyFrame
df = mx.from_arrow(arrow_table)        # from PyArrow
df = mx.from_pandas(pandas_df)         # from pandas

# Lazy operations (build plan, don't execute)
result = (
    df
    .filter(mx.col('age') > 30)
    .select(
        mx.col('name'),
        (mx.col('salary') * 1.1).alias('adjusted_salary'),
    )
    .groupby('department')
    .agg(
        mx.col('salary').sum().alias('total_salary'),
        mx.col('salary').mean().alias('avg_salary'),
        mx.col('name').count().alias('headcount'),
    )
    .sort('total_salary', descending=True)
    .limit(10)
    .compute()   # ← executes everything, returns PyArrow Table
)
```

### 3.2 Design Rules

1. **Lazy by default** — `.compute()` triggers execution. Everything before it builds a plan.
2. **Expressions are composable** — `col('x') + lit(10)` builds an AST, not a value.
3. **Alias is explicit** — every computed column needs `.alias('name')`.
4. **No mutation** — each method returns a new `LazyFrame`. Original is untouched.
5. **PyArrow is the interchange** — input and output are always `pa.Table`. Users can convert to pandas/polars from there.

### 3.3 What We Borrow from Polars

- `col()` / `lit()` expression syntax
- `.filter()` / `.select()` / `.groupby()` / `.agg()` method chain
- Lazy evaluation model
- `.alias()` for naming computed columns

### 3.4 What We Do Differently

- **No Rust** — Mojo replaces Rust as the systems language
- **MAX Graph** — we compile to a graph IR, not direct function calls
- **GPU-native** — GPU is a first-class target, not a plugin
- **Simpler scope** — we start with the 80% of operations that cover 95% of use cases

---

## 4. TPC-H as the North Star

TPC-H has 22 queries. We target them incrementally:

| Query | What it tests | Required ops |
|-------|--------------|--------------|
| **Q1** | Grouped aggregation + filter | `filter`, `groupby`, `sum`, `count`, `mean` |
| **Q6** | Simple scan + filter + sum | `filter`, `sum` |
| **Q3** | 3-way join + groupby + sort + limit | `join`, `groupby`, `sum`, `sort`, `limit` |
| **Q5** | 6-way join + groupby + sort | `join` (multiple), `groupby`, `sum`, `sort` |
| **Q10** | Join + filter + groupby + sort + limit | All of the above |

**Q1 and Q6 are the first milestones.** They validate the core pipeline (filter → aggregate) without joins.

---

## 5. Success Criteria

| Milestone | Definition of Done |
|-----------|-------------------|
| **v0.1 — Core Pipeline** | `filter` + `select` + `groupby` + `sum/min/max/mean/count` all compile to MAX Graph and produce correct results on CPU |
| **v0.2 — GPU Path** | Same ops run on GPU with automatic device selection |
| **v0.3 — TPC-H Q1 & Q6** | Both queries run correctly and beat pandas in speed |
| **v0.4 — Joins & Sort** | Hash join, sort, limit plan nodes working |
| **v0.5 — TPC-H Q3** | 3-way join query runs correctly |
| **v1.0 — Public Release** | 10+ TPC-H queries, docs site, pip-installable, benchmarks published |