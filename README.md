# 🚀 MXFrame

> **GPU-accelerated DataFrames — Python ergonomics, Mojo speed, every GPU.**

MXFrame is a DataFrame query engine that pairs a Polars-style Python API with
pre-compiled Mojo AOT kernels. The same code runs on **NVIDIA, AMD, and Apple Silicon** —
no CUDA required, no JIT compilation at query time.

[![TPC-H](https://img.shields.io/badge/TPC--H-22%2F22%20queries-brightgreen)](docs/benchmarks.md)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## ✨ Why MXFrame?

| | pandas | Polars | cuDF (Rapids) | **MXFrame** |
|---|---|---|---|---|
| GPU support | ❌ | ❌ | ✅ NVIDIA only | ✅ **Any GPU** |
| Compiled kernels | ❌ | ✅ Rust | ✅ CUDA | ✅ **Mojo AOT** |
| Install complexity | pip | pip | CUDA + Rapids stack | **pixi install** |
| TPC-H competitive | ❌ | ✅ | ✅ | ✅ |
| Cross-vendor | ❌ | ❌ | ❌ | ✅ NVIDIA/AMD/Apple |

MXFrame is the **cuDF architecture without the CUDA lock-in**.  
The kernels are compiled **once** to a `.so` at build time — loaded in ~1 ms at process start,
then pure dispatch on every query. No per-query JIT tax.

---

## ⚡ Quick Start

### Install (Development)

```bash
# 1. Install pixi (Modular's package manager)
curl -fsSL https://pixi.sh/install.sh | bash

# 2. Clone and set up
git clone https://github.com/abhisheksreesaila/mxframe
cd mxframe
pixi install

# 3. Verify GPU is working
pixi run python3 scripts/_check_gpu.py
```

### Your First Query

```python
import pyarrow as pa
from mxframe import LazyFrame, Scan, col, lit

# Create an Arrow table (or load from Parquet/CSV)
data = pa.table({
    "dept":   pa.array(["eng", "eng", "mkt", "mkt", "eng"]),
    "salary": pa.array([120.0, 95.0, 80.0, 110.0, 130.0], pa.float32()),
    "age":    pa.array([32, 28, 35, 29, 40], pa.int32()),
})

# Build a lazy query plan — nothing executes yet
result = (
    LazyFrame(Scan(data))
    .filter(col("age") > lit(28))
    .groupby("dept")
    .agg(
        col("salary").sum().alias("total_salary"),
        col("salary").mean().alias("avg_salary"),
        col("age").count().alias("headcount"),
    )
    .sort(col("total_salary"), descending=True)
    .compute(device="gpu")   # or "cpu"
)

print(result.to_pandas())
```

Output:
```
  dept  total_salary  avg_salary  headcount
0  eng         345.0       115.0          3
1  mkt         110.0       110.0          1
```

---

## 📊 TPC-H Benchmark — all 22 queries

> **Hardware:** NVIDIA RTX 3090 (sm_86) · AMD 12-core CPU · Mojo 0.26.2 AOT kernels
> **Baselines:** Polars 1.29+ · Pandas 3.0 · MXFrame CPU path · MXFrame GPU path
> **Data:** TPC-H schema synthetic data (numpy RNG, fixed seed)
> **Methodology:** 1 warmup run + **median of 3 timed runs**, per engine. Warmup primes every cache an app would have primed on query #2 in production — so these numbers reflect steady-state dispatch, not first-call JIT cost.
> **Source of truth:** [`scripts/bench_results_1M.csv`](scripts/bench_results_1M.csv) and
> [`scripts/bench_results_10M.csv`](scripts/bench_results_10M.csv) — committed in repo, reproducible via
> [`scripts/benchmark_all_22.py`](scripts/benchmark_all_22.py)

### How the kernels dispatch

| Device | Path | Coverage |
|---|---|---|
| **CPU** | 100% ctypes into pre-compiled `libmxkernels_aot.so` | All 22 queries — group aggs, masked aggs, inner + left joins, gather, filter, sort, unique |
| **GPU** | ctypes into pre-compiled `libmxkernels_aot_gpu.so` | All grouped aggs (sum/min/max/count) + masked global aggs |
| **GPU** | MAX Graph, shape-cached model | Hash joins only — compiled once per `(n_left, n_right)` shape, cached for the session |

No per-query JIT on CPU. GPU aggregations skip MAX Graph entirely. GPU joins compile a model once per shape and reuse it — the warmup run primes that cache.

### 1 M rows — warm median of 3 runs

All times in **milliseconds · lower is better**. Speedup columns = `Polars / MXFrame`; **bold** = MXFrame wins.

| Query | Description | MX CPU | MX GPU | Polars | Pandas | CPU vs Polars | GPU vs Polars |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1  | Filter + 8 aggregations       | **11.2** | 94.4     | 35.6  | 117.4 | **3.2×**   | 0.4× |
| Q2  | Min-cost supplier             | **6.6**  | 15.6     | 16.3  | 9.5   | **2.5×**   | **1.0×** |
| Q3  | 3-table join + agg            | **5.6**  | **15.9** | 19.2  | 21.8  | **3.4×**   | **1.2×** |
| Q4  | Order priority                | 15.0     | 192.5    | 15.0  | 29.9  | 1.0×       | 0.1× |
| Q5  | Multi-join + groupby          | **0.6**  | **4.1**  | 22.6  | 22.0  | **37.7×**  | **5.5×** |
| Q6  | Masked global agg             | **7.9**  | 13.2     | 10.4  | 7.7   | **1.3×**   | 0.8× |
| Q7  | Shipping volume               | **0.6**  | **7.3**  | 29.9  | 19.1  | **49.8×**  | **4.1×** |
| Q8  | Market share                  | **0.9**  | **4.7**  | 20.2  | 10.3  | **22.4×**  | **4.3×** |
| Q9  | Product profit (6-table join) | **0.6**  | **6.6**  | 39.9  | 17.8  | **66.5×**  | **6.0×** |
| Q10 | Customer revenue              | **3.6**  | **11.0** | 32.6  | 19.3  | **9.1×**   | **3.0×** |
| Q11 | Important stock               | **0.5**  | **2.7**  | 7.4   | 3.0   | **14.8×**  | **2.7×** |
| Q12 | 2-table join + agg            | **0.8**  | **3.5**  | 24.5  | 586.2 | **30.6×**  | **7.0×** |
| Q13 | Customer distribution         | **16.2** | 30.2     | 27.5  | 33.5  | **1.7×**   | 0.9× |
| Q14 | Promo revenue                 | **1.4**  | **1.2**  | 6.9   | 244.0 | **4.9×**   | **5.8×** |
| Q15 | Top-supplier revenue          | **1.3**  | 11.8     | 9.9   | 6.4   | **7.6×**   | 0.8× |
| Q16 | Part/supplier relationships   | **2.1**  | **6.3**  | 16.9  | 6.8   | **8.0×**   | **2.7×** |
| Q17 | Small-qty order               | **0.3**  | **2.7**  | 7.9   | 4.7   | **26.3×**  | **2.9×** |
| Q18 | Large-volume customers        | **4.2**  | **22.3** | 33.4  | 16.8  | **8.0×**   | **1.5×** |
| Q19 | Discounted revenue            | **10.9** | **11.2** | 19.6  | 22.4  | **1.8×**   | **1.8×** |
| Q20 | Potential part promo          | **4.3**  | **5.6**  | 30.0  | 9.6   | **7.0×**   | **5.4×** |
| Q21 | Suppliers who kept (EXISTS)   | **26.0** | 64.1     | 31.3  | 28.6  | **1.2×**   | 0.5× |
| Q22 | Global sales opportunity      | **7.6**  | **16.5** | 25.4  | 56.7  | **3.3×**   | **1.5×** |

**1 M summary:** MX CPU beats Polars on **21/22** queries (Q4 tied); MX GPU beats Polars on **16/22** queries. Headline CPU wins: **Q9 66×**, **Q7 50×**, **Q5 38×**, **Q12 31×**, **Q17 26×**. Headline GPU wins: **Q12 7×**, **Q9 6×**, **Q14 5.8×**, **Q5 5.5×**, **Q20 5.4×**.

### 10 M rows — warm median of 3 runs

| Query | Description | MX CPU | MX GPU | Polars | Pandas | CPU vs Polars | GPU vs Polars |
|---|---|---:|---:|---:|---:|---:|---:|
| Q1  | Filter + 8 aggregations       | **361.0**  | 1190.7   | 946.5  | 1771.7 | **2.6×**   | 0.8× |
| Q2  | Min-cost supplier             | **5.7**    | **11.9** | 15.4   | 7.6    | **2.7×**   | **1.3×** |
| Q3  | 3-table join + agg            | **57.7**   | **67.2** | 72.2   | 581.8  | **1.3×**   | **1.1×** |
| Q4  | Order priority                | 301.5      | 492.2    | 113.8  | 807.6  | 0.4×       | 0.2× |
| Q5  | Multi-join + groupby          | **2.8**    | **13.8** | 60.7   | 332.9  | **21.7×**  | **4.4×** |
| Q6  | Masked global agg             | 399.6      | 523.2    | 92.0   | 246.4  | 0.2×       | 0.2× |
| Q7  | Shipping volume               | **1.8**    | **16.4** | 76.4   | 392.5  | **42.4×**  | **4.7×** |
| Q8  | Market share                  | **1.3**    | **3.8**  | 40.9   | 55.1   | **31.5×**  | **10.8×** |
| Q9  | Product profit                | **0.7**    | **7.4**  | 89.7   | 431.3  | **128.1×** | **12.1×** |
| Q10 | Customer revenue              | **39.2**   | **45.0** | 131.1  | 216.2  | **3.3×**   | **2.9×** |
| Q11 | Important stock               | **0.4**    | **3.1**  | 6.6    | 2.5    | **16.5×**  | **2.1×** |
| Q12 | 2-table join + agg            | **1.3**    | **4.4**  | 116.4  | 6853.6 | **89.5×**  | **26.5×** |
| Q13 | Customer distribution         | 385.1      | 396.1    | 285.8  | 463.9  | 0.7×       | 0.7× |
| Q14 | Promo revenue                 | **14.4**   | **16.9** | 29.7   | 2719.2 | **2.1×**   | **1.8×** |
| Q15 | Top-supplier revenue          | **2.7**    | 57.4     | 16.1   | 30.0   | **6.0×**   | 0.3× |
| Q16 | Part/supplier relationships   | **2.7**    | **6.5**  | 16.3   | 6.8    | **6.0×**   | **2.5×** |
| Q17 | Small-qty order               | **0.6**    | **1.5**  | 14.6   | 17.0   | **24.3×**  | **9.7×** |
| Q18 | Large-volume customers        | **46.8**   | 69.4     | 63.4   | 242.8  | **1.4×**   | 0.9× |
| Q19 | Discounted revenue            | **100.4**  | **97.9** | 112.9  | 234.8  | **1.1×**   | **1.2×** |
| Q20 | Potential part promo          | **32.3**   | **37.1** | 39.9   | 52.9   | **1.2×**   | **1.1×** |
| Q21 | Suppliers who kept            | 756.0      | 705.3    | 85.9   | 216.6  | 0.1×       | 0.1× |
| Q22 | Global sales opportunity      | **54.1**   | **59.2** | 132.8  | 1292.1 | **2.5×**   | **2.2×** |

**10 M summary:** MX CPU beats Polars on **18/22** queries; MX GPU beats Polars on **15/22**. Headline CPU wins scale beautifully: **Q9 128×**, **Q12 89×**, **Q7 42×**, **Q8 32×**, **Q17 24×**, **Q5 22×**. Headline GPU wins: **Q12 26.5×**, **Q9 12×**, **Q8 10.8×**, **Q17 9.7×**, **Q7 4.7×**.

### Where MXFrame loses (same at both scales)

- **Q4, Q6, Q13, Q21** — operations where our kernel path falls back to PyArrow compute or does extra passes. These are the focus of the next milestone (see [`roadmap.md`](roadmap.md)).

### What the numbers mean

- **Correctness ✅** — all 22 queries return results that round-trip through Pandas and match Polars output.
- **Coverage ✅** — every TPC-H query has a CPU AOT path; all group aggs/masked aggs have GPU AOT paths; GPU joins use shape-cached MAX Graph models.
- **No JIT tax in steady state** — after the first query of each shape warms the GPU join model cache, every subsequent call is pure dispatch. The CPU path has no JIT at all.
- **Why GPU doesn't always win** — GPU wins scale with workload size and kernel coverage. At 10 M, GPU crushes Polars on the join-heavy queries (Q8/Q9/Q12) where Mojo's shape-cached kernels pay off. Where GPU loses, it's either PCIe overhead on tiny outputs (Q1, Q6) or ops that still route through PyArrow fallback (Q4, Q13, Q21).



---

## 🔁 Reproducing the Benchmark

To run the benchmark with **official TPC-H data** (generated by DuckDB's
faithful port of the TPC-H `dbgen` tool):

```bash
# Step 1 — generate TPC-H data (requires: pip install duckdb)
#   SF=1  →  ~6M lineitem rows,  ~200 MB Parquet
#   SF=0.1 → ~600K rows, quick sanity check
pixi run python3 scripts/gen_tpch_parquet.py --sf 1

# Step 2 — run the 22-query benchmark against real data
pixi run python3 scripts/bench_real_tpch.py --data-dir data/tpch_sf1 --runs 3
```

The generated `data/tpch_sf1/` directory contains 8 Parquet files (one per
TPC-H table) that you can inspect, share, or version-control.

### Scale factor guide

| `--sf` | lineitem rows | approx size | use case |
|---|---|---|---|
| 0.01 | ~60K | 2 MB | smoke test / CI |
| 0.1 | ~600K | 20 MB | local dev |
| 1 | ~6M | 200 MB | standard published benchmark |
| 10 | ~60M | 2 GB | stress test |

### Data lineage & legal note

- Data is generated by **DuckDB's TPC-H extension** — a faithful port of the
  official TPC-H `dbgen` v3.0.1 with the same value distributions (uniform,
  Zipfian, pseudo-random vocab).
- **TPC-H® is a trademark of the Transaction Processing Performance Council.**
  These results are an _independent, non-certified_ benchmark. They may _not_
  be reported as "TPC-H results" without formal TPC certification.
- Reference: <https://www.tpc.org/tpch/>

---

## 🔤 API Reference

### LazyFrame

```python
from mxframe import LazyFrame, Scan, col, lit, when

lf = LazyFrame(Scan(arrow_table))
```

| Method | Description | Example |
|---|---|---|
| `.filter(expr)` | Row filter | `.filter(col("x") > lit(10))` |
| `.select(*cols)` | Column projection | `.select("a", "b", col("c").alias("d"))` |
| `.with_columns(*exprs)` | Add/replace columns | `.with_columns((col("a") * lit(2)).alias("a2"))` |
| `.groupby(*keys)` | Start grouped agg | `.groupby("dept", "region")` |
| `.join(other, left_on, right_on, how)` | Hash join | `.join(lf2, "id", "fk_id", how="inner")` |
| `.sort(expr, descending)` | Sort rows | `.sort(col("revenue"), descending=True)` |
| `.limit(n)` | Take first N rows | `.limit(100)` |
| `.distinct()` | Deduplicate rows | `.distinct()` |
| `.compute(device)` | Execute the plan | `.compute(device="gpu")` |

### Expressions (`col`, `lit`, `when`)

```python
# Arithmetic
col("price") * (lit(1.0) - col("discount"))

# Comparison
col("date") >= lit(19940101)

# Boolean combine
(col("x") > lit(0)) & (col("y") < lit(100))

# Conditional
when(col("nation") == lit("BRAZIL"), col("revenue"), lit(0.0))

# String
col("phone").startswith("13")

# Date parts
col("orderdate").year()    # extract year as int32

# Aggregations (inside .agg())
col("salary").sum()
col("salary").mean()
col("salary").min()
col("salary").max()
col("id").count()
```

### SQL Frontend

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

---

## 🔧 Supported Operations

| Category | Operations |
|---|---|
| **Filter** | `>`, `>=`, `<`, `<=`, `==`, `!=`, `&`, `\|`, `~`, `isin`, `startswith`, `contains` |
| **Aggregation** | `sum`, `mean`, `min`, `max`, `count` |
| **Groupby** | Single key, multi-key, composite key |
| **Join** | Inner, Left outer |
| **Sort** | Single/multi column, ascending/descending |
| **Window** | `year()` date part extraction |
| **Projection** | `select`, `with_columns`, `alias`, arithmetic expressions |
| **Semi-join** | Via unique-key inner join |
| **Anti-join** | Via `pc.is_in` + `pc.invert` |
| **Distinct** | Full row deduplication |
| **SQL** | `SELECT`, `FROM`, `WHERE`, `GROUP BY`, `ORDER BY`, `LIMIT`, `JOIN` |

---

## 📁 Project Structure

```
mxframe/
├── __init__.py            ← Public API (LazyFrame, Scan, col, lit, when, sql)
├── lazy_frame.py          ← LazyFrame, LazyGroupBy, Scan
├── lazy_expr.py           ← Expr, col(), lit(), when()
├── compiler.py            ← LogicalPlan → MAX Graph compiler
├── custom_ops.py          ← Dispatch: AOT kernels / MAX Graph / PyArrow fallback
├── optimizer.py           ← Plan rewrites (filter pushdown, join reordering)
├── plan_validation.py     ← Pre-execution plan checks
├── sql_frontend.py        ← SQL → LogicalPlan via sqlglot
│
├── kernels_aot/           ← Pre-compiled AOT shared libraries
│   ├── libmxkernels_aot.so      ← CPU kernels (ctypes-callable)
│   └── libmxkernels_aot_gpu.so  ← GPU kernels (CUDA/ROCm/Metal)
│
├── kernels_v261/          ← Mojo kernel source (build time only)
│   ├── group_sum.mojo, group_min.mojo, group_max.mojo ...
│   ├── join_scatter.mojo, join_count.mojo
│   ├── join_scatter_left.mojo, join_count_left.mojo
│   └── filter_gather.mojo, gather_rows.mojo, unique_mask.mojo ...
│
├── kernels_aot/
│   ├── kernels_aot.mojo         ← CPU AOT entry points
│   └── kernels_aot_gpu.mojo     ← GPU AOT entry points
│
├── scripts/
│   ├── bench_simple.py          ← Clean 4-col benchmark (Pandas|Polars|MX CPU|MX GPU)
│   ├── benchmark_tpch.py        ← All 22 TPC-H query implementations
│   ├── _test_aot_smoke.py       ← AOT kernel smoke tests
│   └── quickstart.py            ← Minimal hello-world example
│
└── docs/
    ├── vision-and-architecture.md
    ├── CONTRIBUTING.md          ← Developer guide
    └── PUBLISHING.md            ← pip release steps
```

---

## 🖥️ Device Selection

```python
# CPU (default, works everywhere)
result = lf.compute(device="cpu")

# GPU (requires NVIDIA/AMD/Apple Silicon with MAX runtime)
result = lf.compute(device="gpu")
```

The GPU path uses Mojo's `DeviceContext` — the same source compiles to:
- **PTX** on NVIDIA (CUDA-compatible)
- **HSA/ROCm** on AMD
- **Metal** on Apple Silicon

---

## 🧪 Running Tests

```bash
# Smoke tests — AOT kernels
pixi run python3 scripts/_test_aot_smoke.py

# All TPC-H correctness checks
pixi run python3 scripts/_test_phase6_tpch_tier2.py

# GPU check
pixi run python3 scripts/_check_gpu.py

# Full 22-query benchmark
pixi run python3 scripts/bench_simple.py --rows 1000000 --runs 3
```

---

## 📦 Dependencies

| Package | Required | Purpose |
|---|---|---|
| `pyarrow >= 14` | ✅ | Column storage, zero-copy NumPy bridge |
| `numpy >= 1.24` | ✅ | Vectorized pre/post processing |
| `pandas >= 2.0` | ✅ | Reference implementations, Pandas bridge |
| `modular >= 26.2` | GPU only | MAX Engine runtime, Mojo GPU dispatch |
| `polars >= 0.20` | optional | Polars bridge + benchmark comparison |
| `sqlglot >= 25` | optional | SQL frontend parsing |

---

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the developer guide, kernel writing guidelines,
and how to add new TPC-H queries.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).
