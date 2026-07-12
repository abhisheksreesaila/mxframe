<img src="visualize/mx-spark.svg" alt="MXFrame MX lightning mark" width="72">

# 🚀 MXFrame

> **GPU-accelerated DataFrames — Python ergonomics, Mojo AOT kernels.**

MXFrame is a DataFrame query engine that pairs a Polars-style Python API with pre-compiled Mojo AOT kernels. The current GPU build is validated on **NVIDIA**; AMD and Apple Silicon validation remain on the roadmap.

[![TPC-H](https://img.shields.io/badge/TPC--H-22%2F22%20queries-brightgreen)](docs/benchmarks.md)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](pyproject.toml)
[![Platform](https://img.shields.io/badge/platform-linux--x86__64-lightgrey)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)

---

## ✨ Why MXFrame?

| | pandas | Polars | cuDF (RAPIDS) | **MXFrame** |
|---|---|---|---|---|
| GPU support | ❌ | ❌ | ✅ NVIDIA | ✅ **NVIDIA validated** |
| Compiled kernels | ❌ | ✅ Rust | ✅ CUDA | ✅ **Mojo AOT** |
| Install complexity | pip | pip | CUDA + RAPIDS stack | **pip or pixi** |
| TPC-H coverage | reference | ✅ | ✅ | ✅ **22/22** |
| Portable kernel design | ❌ | CPU | CUDA | ✅ NVIDIA today; AMD/Apple planned |

MXFrame uses a portable Mojo kernel architecture rather than CUDA-specific source. Kernels compile once to shared libraries and dispatch directly at runtime.

### What changed in v0.4 and v0.5

- **v0.4.0 — correct string/composite joins:** shared dictionaries assign identical dense IDs across both join sides; mixed composite keys and SQL null semantics are preserved.
- **v0.5.0 — native GPU UTF-8 predicates:** `startswith`, literal `contains`, string equality/inequality, and packed non-null literal `isin` operate directly on Arrow offsets and byte buffers.
- **v0.5.0 — weak-query fusion:** dedicated compact GPU paths remove expanded intermediates from Q4, Q6, Q13, and Q21.
- **v0.5.0 — bounded benchmarking and visualization:** every query/engine runs in an isolated process; Q4, Q6, Q12, Q13, Q14, and Q21 have step-by-step kernel visualizations.

See [v0.5.0 release notes](RELEASE_NOTES_0.5.0.md) for measured gains, limitations, and the parity roadmap.

---

## ⚡ Quick Start

```sh
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

## 📊 Performance — bounded v0.5.0 benchmark

RTX 3090, warm median of three runs. Every query and engine ran in its own process so Arrow, Pandas, and CUDA allocations were released between measurements.

- **1M:** MX CPU beat Polars on **22/22**; MX GPU beat Polars on **15/17 comparable** paths.
- **10M:** MX CPU beat Polars on **18/22**; MX GPU beat Polars on **14/17 comparable** paths.
- Q8 GPU remains `N/A:JIT`; Q15, Q17, Q18, and Q20 GPU workers exceeded the bounded timeout.
- cuDF was unavailable on this machine, so RAPIDS remains `N/A` and no RAPIDS parity claim is made.

### 1M representative queries

| Query | MX CPU | MX GPU | Polars | Best MX speedup |
|---|---:|---:|---:|---:|
| Q4 · Order priority | 12.8 ms | **3.4 ms** | 20.2 ms | **5.9× GPU** |
| Q6 · Discounted revenue | **6.7 ms** | 6.8 ms | 10.8 ms | **1.6× CPU** |
| Q12 · Shipping modes | **0.6 ms** | 6.1 ms | 26.6 ms | **44.3× CPU** |
| Q13 · Customer distribution | 20.4 ms | **1.8 ms** | 26.5 ms | **14.7× GPU** |
| Q21 · Waiting suppliers | 27.1 ms | **6.6 ms** | 30.2 ms | **4.6× GPU** |

### 10M representative queries

| Query | MX CPU | MX GPU | Polars | Best MX speedup |
|---|---:|---:|---:|---:|
| Q4 · Order priority | 265.1 ms | **28.6 ms** | 105.4 ms | **3.7× GPU** |
| Q9 · Product profit | **2.0 ms** | 16.5 ms | 109.7 ms | **54.9× CPU** |
| Q12 · Shipping modes | **1.5 ms** | 8.2 ms | 128.0 ms | **85.3× CPU** |
| Q13 · Customer distribution | 368.8 ms | **28.5 ms** | 442.0 ms | **15.5× GPU** |
| Q21 · Waiting suppliers | 1068.7 ms | **49.8 ms** | 99.1 ms | **2.0× GPU** |

Full matrices: [1M CSV](scripts/bench_results_1M.csv) · [10M CSV](scripts/bench_results_10M.csv) · [methodology and kernel catalogue](docs/benchmarks.md)

## What remains for parity

**Performance parity:** replace Q8's JIT path; fix Q15/Q17/Q18/Q20 GPU timeouts; improve Q1 and Q14 GPU execution; run synchronized 1M/10M measurements on a compatible RAPIDS/cuDF environment.

**Feature parity:** native GPU UTF-8 dictionary construction and string/date gather; GPU windows; semi/anti/as-of/range joins; broader string/regex, datetime, nested-type, reshape, SQL, streaming, and I/O pushdown coverage.

---

## 📚 Docs

| | |
|---|---|
| [**v0.5.0 Release Notes**](RELEASE_NOTES_0.5.0.md) | v0.4/v0.5 gains, benchmark evidence, known gaps, and parity roadmap |
| [**API Reference**](docs/api.md) | LazyFrame, expressions, SQL frontend, supported operations, running tests |
| [**Benchmarks**](docs/benchmarks.md) | Full TPC-H tables, kernel catalogue, limitations, roadmap, reproduce instructions |
| [**Architecture**](docs/vision-and-architecture.md) | Design philosophy, internal layers, how MAX Graph fits in |
| [**Kernel Visualizer**](visualize/mxframe_pipeline.html) | Interactive plans, dispatch boundaries, GPU threads, and output masks |
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