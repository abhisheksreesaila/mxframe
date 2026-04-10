# mxframe Developer Guide

Welcome to the `mxframe` developer documentation! This guide will help you understand the architecture, set up your local development environment, and contribute to the codebase.

## 🏗️ Architecture Overview

The `mxframe` framework consists of two major layers:
1. **Python Frontend (Logical & Physical Planning):** Built entirely in Python. Users write standard functional expressions (`col()`, `lit()`) that construct a `LazyFrame`. Operations are stored entirely out of order in a lazy execution graph.
2. **Execution Engine (MAX Graph & Mojo Kernels):** The framework compiles the final logical plan directly into the MAX Graph. Crucially, any heavy lifting (like aggregates, complex joins, GPU processing) runs directly in high-performance **Mojo kernels**.
   - These compute kernels run with zero-copy overhead directly traversing PyArrow array memory.

### Directory Structure

- **`mxframe/`**: Core Python package.
  - `lazy_frame.py` / `lazy_expr.py`: The user-facing API surface.
  - `optimizer.py`: Logical plan simplifications.
  - `compiler.py` / `custom_ops.py`: Compiles the logical plan into a MAX Graph, injecting Mojo capabilities.
  - `kernels_v261/`: Pure Mojo implementations of the operators. Used to generate the `kernels.mojopkg`.
- **`scripts/`**: Development utilities like building wrappers or running automated kernel builds.
- **`benchmarks/`**: Benchmarking notebooks where performance evaluations (e.g., TPC-H Q1 & Q3 tests) are actively verified.

---

## 🛠️ Setting Up the Development Environment

We use [`pixi`](https://pixi.sh) to manage Python and Mojo dependencies predictably across platforms.

### 1. Install `pixi`
If you do not have `pixi` installed, follow the [official instructions](https://pixi.sh/latest/#installation).

### 2. Bootstrap the Environment
Clone the repository and spin up the environment:
```sh
git clone https://github.com/abhisheksreesaila/mxframe.git
cd mxframe

# Pixi will automatically download Python, PyArrow, Modular MAX SDK, and other tools
pixi install

# Enter the isolated environment
pixi shell
```

### 3. Install the Python bindings locally
Install the framework in editable mode to ensure Python module changes take effect immediately:
```sh
pip install -e .
```

---

## 🚀 Compiling Mojo Kernels

When you make changes to any hardware-level operation written in Mojo (in `mxframe/kernels_v261/*.mojo`), you **must** recompile the `.mojopkg` so the Python Graph compiler can see it.

```sh
# Helper script that packages the kernels folder correctly
bash scripts/build_kernels.sh
```
*Note: Make sure your `pixi.toml`'s modular channel is active, this script relies on the `mojo package` command.*

---

## 🧭 How to Add a New Operator

When creating a new DataFrame operation (e.g., expanding statistical tools or vector manipulation), follow these steps:

1. **Add the Expression & API in Python**: 
   - Define your operator token in `lazy_expr.py` (e.g., `Expr.std()`).
   - If it changes DataFrame shape (like GroupBy/Join), append it logically in `lazy_frame.py`.
2. **Compile to MAX Graph**: 
   - Open `custom_ops.py` (or `compiler.py`).
   - Write the traversal logic to turn the abstract token into a MAX tensor operator.
3. **Build the Custom CPU/GPU execution (Mojo)**: 
   - If MAX does not support it natively on the target device, build a `.mojo` kernel in `kernels_v261/`.
   - Re-run `scripts/build_kernels.sh`.
   - Update `custom_ops.py` to route logic correctly using `--custom_extensions` during `MAXGraph` injection.

---

## 🧪 Running Benchmarks

Performance testing is critical since the entire goal of `mxframe` is matching or beating heavily optimized low-level frameworks. Check out the internal TPC-H benchmarks:

1. Start Jupyter:
   ```sh
   pixi run jupyter notebook
   ```
2. Navigate to `benchmarks/` and execute `bench_q1.ipynb` and `bench_q3.ipynb`. These validate both execution correctness and custom Mojo kernel timing.
