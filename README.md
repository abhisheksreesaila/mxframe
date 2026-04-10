# mxframe

## 🎯 Vision

**mxframe** bridges the gap between Python DataFrames and
high-performance GPU compute. By leveraging PyArrow’s columnar memory
format and MAX Engine’s accelerated kernels, we enable:

- **Zero-copy data sharing** between PyArrow and MAX tensors
- **Seamless CPU/GPU execution** with a unified API  
- **Production-ready performance** without leaving Python

### 🚀 What’s in v0.0.1

This initial release provides a **lazy DataFrame engine** with compiled
execution:

| Feature | Description |
|----|----|
| `LazyFrame` class | Lazy DataFrame backed by PyArrow with logical plan |
| `col()` / `lit()` | Expression DSL for filters, projections, aggregations |
| Filter / Join / GroupBy | Core relational operators |
| Sort / Limit / Distinct | Query shaping operators |
| CPU & GPU execution | `.compute(device="cpu")` or `device="gpu"` |
| Custom Mojo kernels | Compiled joins and aggregations via MAX Engine |

## ⚡ Quick Start

``` python
import pyarrow as pa
from mxframe.lazy_expr import col, lit
from mxframe.lazy_frame import LazyFrame

# Create a LazyFrame from a PyArrow table
table = pa.table({
    'price': [10.5, 20.3, 15.8, 42.0, 8.9],
    'quantity': [100, 200, 150, 50, 300],
    'category': ['A', 'B', 'A', 'B', 'A'],
})
lf = LazyFrame(table)

# Filter + compute
result = lf.filter(col("price") > lit(10.0)).compute()
print(result.to_pandas())
```

       price  quantity category
    0   10.5       100        A
    1   20.3       200        B
    2   15.8       150        A
    3   42.0        50        B

### GroupBy + Aggregation

Lazy expressions for groupby aggregations:

``` python
# GroupBy aggregation
result = (
    lf.groupby("category")
      .agg(col("price").sum().alias("total_price"),
           col("quantity").sum().alias("total_qty"))
      .compute()
)
print(result.to_pandas())
```

      category  total_price  total_qty
    0        A    35.199997      550.0
    1        B    62.299999      250.0

### Join + Sort + Limit

Combine two tables with a join, then sort and limit:

``` python
# Join two tables
orders = pa.table({"oid": [1, 2, 3], "cid": [10, 20, 10], "amount": [100.0, 200.0, 150.0]})
customers = pa.table({"cid": [10, 20], "name": ["Alice", "Bob"]})

result = (
    LazyFrame(orders)
    .join(LazyFrame(customers), on="cid")
    .sort("amount", descending=True)
    .limit(2)
    .compute()
)
print(result.to_pandas())
```

       oid  cid  amount   name
    0    2   20   200.0    Bob
    1    3   10   150.0  Alice

## 📦 Installation

Install from [PyPI](https://pypi.org/project/mxframe/):

``` sh
pip install mxframe
```

Or install latest from GitHub:

``` sh
pip install git+https://github.com/abhisheksreesaila/mxframe.git
```

### Requirements

- Python 3.10+
- PyArrow
- MAX Engine SDK

## 🛠️ Developer Guide

``` sh
# Clone the repository
git clone https://github.com/abhisheksreesaila/mxframe.git
cd mxframe

# Install in development mode
pip install -e .
```

For more detailed developer instructions, architecture, and guidelines, please refer to our **[Developer Guide](DEVELOPER.md)** adjacent to this README.

## 📚 Documentation

> *API Reference and detailed documentation are currently being migrated from our initial prototypes to a new format and will be available soon.*

## 🗺️ Roadmap

- [x] Lazy evaluation and query optimization
- [x] Filter / Join / GroupBy / Sort / Limit / Distinct
- [x] CPU & GPU execution via MAX Engine
- [x] Custom Mojo kernels for TPC-H queries
- [x] Multi-threaded CPU execution
- [x] Window functions
- [x] I/O connectors (Parquet, CSV)

## 📄 License

Apache 2.0
