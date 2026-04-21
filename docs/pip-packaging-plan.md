# 🚀 MXFrame — pip Install Packaging Plan

> Goal: `pip install mxframe` installs a working CPU+GPU DataFrame library backed by Mojo kernels.

---

## Current State

| Item | Status |
|------|--------|
| All 22 TPC-H queries | ✅ Passing |
| Window functions | ✅ Implemented |
| SQL frontend | ✅ Implemented |
| GPU support | ✅ Working |
| Mojo kernels compiled to `kernels.mojopkg` | ✅ Built |
| `kernels.mojopkg` wired into runtime | ⚠️ Currently uses `kernels_v261/` dir (dev mode) |
| Proper `pyproject.toml` (a) structure | ⚠️ Needs restructure |
| PyPI-ready wheel | ❌ Not yet built |
| MAX dependency declaration | ❌ Not yet specified |

---

## Step 1 — Restructure to a clean src layout

**Why:** The current `package-dir = "."` maps `mxframe` to the project root, dragging in `scripts/`, `llm-ctx/`, `kernels_v261/`, docs etc. into the deployed package. A proper `src/` layout fixes this.

```
mxdf_v2/
├── src/
│   └── mxframe/
│       ├── __init__.py
│       ├── compiler.py
│       ├── custom_ops.py
│       ├── lazy_expr.py
│       ├── lazy_frame.py
│       ├── optimizer.py
│       ├── plan_validation.py
│       ├── sql_frontend.py
│       └── kernels.mojopkg        ← pre-built, included in wheel
├── kernels_v261/                  ← stays as dev-only build source
├── scripts/
├── pyproject.toml
└── pixi.toml
```

**Action:**
```bash
mkdir -p src/mxframe
cp __init__.py compiler.py custom_ops.py lazy_expr.py \
   lazy_frame.py optimizer.py plan_validation.py sql_frontend.py \
   src/mxframe/
cp kernels.mojopkg src/mxframe/
```

---

## Step 2 — Update `pyproject.toml`

Replace the current minimal config with a full spec:

```toml
[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "mxframe"
version = "0.1.0"
description = "GPU-accelerated DataFrames backed by Modular MAX Engine and Mojo kernels"
readme = "README.md"
license = { text = "Apache-2.0" }
requires-python = ">=3.12"
keywords = ["dataframe", "gpu", "mojo", "max", "tpch", "analytics"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: Apache Software License",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "Topic :: Database",
]
dependencies = [
    "pyarrow>=14.0",
    "numpy>=1.24",
    "sqlglot>=25.0",
    # MAX Engine — install from modular conda or PyPI when available:
    # "modular>=24.6",   # uncomment once MAX is on PyPI
]

[project.urls]
Homepage = "https://github.com/YOUR_ORG/mxframe"
Documentation = "https://mxframe.readthedocs.io"

[tool.setuptools.packages.find]
where = ["src"]

[tool.setuptools.package-data]
mxframe = ["kernels.mojopkg"]
```

---

## Step 3 — Update `KERNELS_PATH` to use `.mojopkg` in production

In `custom_ops.py`, the current default points to `kernels_v261/` directory. For a deployed wheel that has no `kernels_v261/`, it must point to `kernels.mojopkg`.

```python
# custom_ops.py  (after move to src/mxframe/)
KERNELS_PATH = str(Path(__file__).parent / "kernels.mojopkg")
```

The `custom_extensions=[Path(self.kernels_path)]` in the Graph builder accepts both a directory of `.mojo` files AND a pre-built `.mojopkg` file, so no other code changes are needed.

**Dev workflow** (optional — for kernel iteration):
```python
# Override at runtime during kernel development
import os
os.environ["MXFRAME_KERNELS_PATH"] = "/path/to/kernels_v261"
```

Update `KERNELS_PATH` in `custom_ops.py`:
```python
KERNELS_PATH = os.environ.get(
    "MXFRAME_KERNELS_PATH",
    str(Path(__file__).parent / "kernels.mojopkg")
)
```

---

## Step 4 — Handle the MAX Engine dependency

MAX Engine is not yet on PyPI as a pure `pip install` package. There are three paths:

### Option A — Modular conda channel (recommended for now)
```bash
pixi add modular --channel https://conda.modular.com/max
# or
conda install -c https://conda.modular.com/max modular
```
Document this as a prerequisite in README.

### Option B — Max PyPI release (track Modular release calendar)
Modular is working on a PyPI release. Once `modular` lands on PyPI, add to `dependencies`:
```toml
dependencies = ["pyarrow>=14.0", "numpy>=1.24", "sqlglot>=25.0", "max>=24.6"]
```

### Option C — Graceful import guard
Add to `__init__.py`:
```python
try:
    from max import engine, driver
except ImportError:
    raise ImportError(
        "mxframe requires the Modular MAX Engine. Install it via:\n"
        "  conda install -c https://conda.modular.com/max modular\n"
        "  or follow https://docs.modular.com/max/install"
    )
```

---

## Step 5 — Build the wheel

```bash
# Install build tool
pip install build

# Build wheel + sdist
python -m build

# Output: dist/mxframe-0.1.0-py3-none-any.whl
#           dist/mxframe-0.1.0.tar.gz
```

The wheel is `py3-none-any` (pure Python + pre-compiled `.mojopkg`) — no C extensions, no platform-specific compiled code in the wheel itself. MAX Engine handles platform specifics at runtime via JIT.

---

## Step 6 — Test clean install

```bash
# In a fresh environment with MAX already installed
pip install dist/mxframe-0.1.0-py3-none-any.whl

python -c "
import mxframe as mx
import pyarrow as pa
import numpy as np

t = pa.table({'x': np.random.randn(1000).astype('float32'),
              'g': np.random.randint(0, 4, 1000).astype('int32')})
result = mx.from_arrow(t).groupby('g').agg(mx.col('x').sum()).compute()
print('Install OK:', result)
"
```

---

## Step 7 — Publish to PyPI

```bash
pip install twine

# Test PyPI first
twine upload --repository testpypi dist/*
pip install --index-url https://test.pypi.org/simple/ mxframe

# Production PyPI
twine upload dist/*
```

---

## Step 8 — CI / CD (GitHub Actions)

```yaml
# .github/workflows/release.yml
name: Release
on:
  push:
    tags: ["v*"]

jobs:
  build-and-publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: prefix-dev/setup-pixi@v0.8.0
        with:
          pixi-version: latest
      - name: Install pixi deps
        run: pixi install
      - name: Build kernels
        run: pixi run bash scripts/build_kernels.sh
      - name: Build wheel
        run: pixi run python -m build
      - name: Publish to PyPI
        uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

---

## Summary Checklist

```
[ ] Step 1:  Create src/mxframe/ layout, copy Python files + kernels.mojopkg
[ ] Step 2:  Rewrite pyproject.toml with proper metadata and package-data
[ ] Step 3:  Update KERNELS_PATH to use kernels.mojopkg (+ env var override)
[ ] Step 4:  Add MAX import guard to __init__.py with clear install instructions
[ ] Step 5:  pip install build && python -m build
[ ] Step 6:  Smoke test in a clean env — all 22 TPC-H queries must pass
[ ] Step 7:  twine upload to TestPyPI, verify, then production PyPI
[ ] Step 8:  Add GitHub Actions release workflow
```

**Estimated effort:** 1 session (~2-3 hours)

The hardest part is Step 4 (MAX dependency) — specifically ensuring that end-users who only have `pip install mxframe` also have a working MAX runtime. Until MAX is on PyPI, the README prerequisite is the pragmatic approach.
