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

## 🧠 How `device="auto"` works

Give MXFrame a CPU and a GPU and `compute(device="auto")` decides where each query runs
from costs it measures on your machine. First scans run on the CPU while the table's
columns copy to the GPU in the background; after that, queries split their rows so the
GPU and every CPU core work at the same time. On TPC-H SF10 Q1 the split finishes the
aggregation in 25 ms, against 57 ms on the CPU alone and 39 ms on the GPU alone.

On the final user-felt TPC-H run, `auto` is within 10% of the better of `cpu` and `gpu` on
22/22 queries at SF10, and on 21/22 at SF1 (median of 4 runs; single runs 17–21). Each query ran in a fresh process from Parquet on disk with
no warm-up, and every result matched DuckDB. On the five SF10 queries where the GPU wins from a
cold start (Q5, Q7, Q9, Q18, Q21), `auto` stays within 5% of it. At SF1, where CUDA start-up
would cost more than the query, it stays on the CPU.

➡️ **[Read how it works](docs/auto-mode.md)** · [visual version](https://htmlpreview.github.io/?https://github.com/abhisheksreesaila/mxframe/blob/main/docs/auto-mode.html)

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

**GPU memory.** Importing `mxframe.aot_kernels` sets `MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE_PERCENT=25` unless you have set it yourself. Left at its default, the Mojo runtime reserves about 90% of the free GPU memory the first time a kernel allocates a scratch buffer, which starves mxframe's own allocations at TPC-H SF10 and every other process on the GPU. The variable is process-wide: other MAX / Mojo users in the same Python process get the 25% pool too. Set it to a larger value before importing mxframe if a kernel needs more scratch than a quarter of the GPU.

---

## 📊 Performance: user-felt TPC-H (2026-10-12)

Each engine runs each query in a fresh process. The data starts as Parquet on disk, and the
clock covers read + query + result with no warm-up. Every result is checked against DuckDB.
Ryzen 9 5900X, RTX 3090.

| sum of 22 queries, ms | pandas | Polars | DuckDB | MXFrame `auto` |
|---|---:|---:|---:|---:|
| SF1 | 6,394 | **2,014** | 3,336 | 4,479 |
| SF10 | 82,081 | 17,075 | **10,394** | 28,892 |

MXFrame `auto` beats pandas on 14/22 (SF1) and 19/22 (SF10). It beats DuckDB on 7/22 at SF1.
It is behind Polars on most queries and behind DuckDB on every SF10 query. The main causes are
the PyArrow Parquet decoder (F-021) and join-heavy queries (F-007). Full tables and the honest
losses: [docs/benchmarks.md](docs/benchmarks.md#final-user-felt-scoreboard-2026-10-12-f-009).

The section below is the older warm-run benchmark on synthetic data: 1 warm-up, then the median
of 3 runs. Its numbers are not comparable with the table above.

### Historical: 📊 Performance — bounded v0.5.0 benchmark

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