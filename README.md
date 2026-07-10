# 🚀 MXFrame

> **GPU-accelerated DataFrames — Python ergonomics, Mojo speed, every GPU.**

MXFrame is a DataFrame query engine that pairs a Polars-style Python API with
pre-compiled Mojo AOT kernels. The same code runs on **NVIDIA, AMD, and Apple Silicon** —
no CUDA required, no JIT compilation at query time.

[![TPC-H](https://img.shields.io/badge/TPC--H-22%2F22%20queries-brightgreen)](docs/benchmarks.md)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-lightgrey)](pyproject.toml)
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
Kernels are compiled **once** to a `.so` at build time — loaded in ~1 ms, then pure dispatch on every query.

---

## ⚡ Quick Start

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

```python
import pyarrow as pa
from mxframe import LazyFrame, Scan, col, lit

data = pa.table({
    "dept":   pa.array(["eng", "eng", "mkt", "mkt", "eng"]),
    "salary": pa.array([120.0, 95.0, 80.0, 110.0, 130.0], pa.float32()),
    "age":    pa.array([32, 28, 35, 29, 40], pa.int32()),
})

result = (
    LazyFrame(Scan(data))
    .filter(col("age") > lit(28))
    .groupby("dept")
    .agg(
        col("salary").sum().alias("total_salary"),
        col("age").count().alias("headcount"),
    )
    .sort(col("total_salary"), descending=True)
    .compute(device="gpu")   # or "cpu"
)
print(result.to_pandas())
```

```
  dept  total_salary  headcount
0  eng         345.0          3
1  mkt         110.0          1
```

---

## 📊 Performance — TPC-H at 1 M rows *(v0.2.3)*

MX CPU beats Polars on **20/22** queries. GPU advantage compounds at 10 M+ rows where kernel parallelism dominates PCIe upload cost.

| Query | MX CPU | MX GPU | Polars | CPU vs Polars |
|---|---:|---:|---:|---:|
| Q9 · Product profit (6-table join) | **0.6 ms** | 6.6 ms | 39.9 ms | **67×** |
| Q12 · 2-table join + agg           | **0.5 ms** | 3.4 ms | 23.2 ms | **46×** |
| Q3 · 3-table join + agg            | **2.6 ms** | 8.7 ms | 23.3 ms | **9×** |
| Q1 · Filter + 8 aggregations       | **10.7 ms** | 42.1 ms | 33.3 ms | **3.1×** |

→ [Full 22-query tables (1 M + 10 M rows), methodology, and reproduce instructions](docs/benchmarks.md)

---

## 📚 Docs

| | |
|---|---|
| [**API Reference**](docs/api.md) | LazyFrame, expressions, SQL frontend, supported operations, running tests |
| [**Benchmarks**](docs/benchmarks.md) | Full TPC-H tables, kernel catalogue, limitations, roadmap, reproduce instructions |
| [**Architecture**](docs/vision-and-architecture.md) | Design philosophy, internal layers, how MAX Graph fits in |
| [**Contributing**](CONTRIBUTING.md) | Dev setup, writing Mojo kernels, adding queries |

---

## 📦 Dependencies

| Package | Required | Purpose |
|---|---|---|
| `pyarrow >= 14` | ✅ | Column storage, zero-copy NumPy bridge |
| `numpy >= 1.24` | ✅ | Vectorized pre/post processing |
| `pandas >= 2.0` | ✅ | Reference implementations |
| `modular >= 26.4` | GPU only | MAX Engine runtime, Mojo GPU dispatch |
| `polars >= 0.20` | optional | Benchmark comparison |
| `sqlglot >= 25` | optional | SQL frontend parsing |

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE).
