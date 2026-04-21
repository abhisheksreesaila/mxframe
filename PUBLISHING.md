# 📦 Publishing MXFrame to PyPI

> Step-by-step guide to building and releasing a new version.

---

## 🔑 One-Time Setup

### 1. Create PyPI accounts

- [PyPI](https://pypi.org/account/register/) — production
- [TestPyPI](https://test.pypi.org/account/register/) — staging

### 2. Configure OIDC Trusted Publishing (recommended — no API tokens needed)

In your PyPI project settings → **Add a new publisher**:

| Field | Value |
|---|---|
| Owner | `abhisheksreesaila` |
| Repository | `mxframe` |
| Workflow filename | `publish.yml` |
| Environment | `pypi` |

Do the same on TestPyPI with environment name `testpypi`.

### 3. Create GitHub Environments

In your repo → **Settings → Environments**:

- Create `testpypi` (no protection rules)
- Create `pypi` (add **Required reviewers** = yourself for safety)

### 4. Install local build tools

```bash
pip install build twine
```

---

## 🔄 Release Checklist

### Before Every Release

- [ ] All tests pass: `pixi run test-all`  
- [ ] Benchmark shows no regression vs previous release  
- [ ] `pyproject.toml` version bumped  
- [ ] `CHANGELOG.md` updated (see format below)  
- [ ] Pre-built `.so` files are current (rebuild if kernels changed)  

---

## 🏗️ Step 1 — Rebuild the AOT Kernels

Always rebuild before packaging to ensure the `.so` files match the source:

```bash
# CPU kernels
pixi run mojo build --emit shared-lib kernels_aot/kernels_aot.mojo \
    -o kernels_aot/libmxkernels_aot.so

# GPU kernels  
pixi run mojo build --emit shared-lib kernels_aot/kernels_aot_gpu.mojo \
    -o kernels_aot/libmxkernels_aot_gpu.so

# Verify sizes look reasonable
ls -lh kernels_aot/*.so
```

---

## 🔢 Step 2 — Bump the Version

Edit `pyproject.toml`:

```toml
[project]
version = "0.2.0"   # ← bump this
```

Use [Semantic Versioning](https://semver.org/):

| Change type | Example | Version bump |
|---|---|---|
| New kernel / query | Added Q15 GPU path | `0.1.x → 0.2.0` (minor) |
| Bug fix | Fixed Q22 float precision | `0.1.0 → 0.1.1` (patch) |
| Breaking API change | Renamed `.compute()` arg | `0.x → 1.0.0` (major) |

---

## 📝 Step 3 — Update CHANGELOG

Append to `CHANGELOG.md`:

```markdown
## [0.2.0] — 2026-04-07

### ✨ Added
- GPU path for Q13 LEFT JOIN (join_count_left + join_scatter_left kernels)
- Q22 vectorized phone prefix via pc.utf8_slice_codeunits

### 🐛 Fixed
- Q15 argmax used 0.9999 tolerance hack → now exact float32 equality
- Q18 used to_pylist() for semi-join → replaced with Mojo join

### ⚡ Performance
- Q20: 8.7× faster than Polars (was comparable) — removed Pandas detour
- Q21: eliminated .to_pandas().groupby.nunique() → NumPy composite key
```

---

## 🧪 Step 4 — Test the Build Locally

```bash
# Build wheel + sdist
python -m build

# Inspect what's inside the wheel
unzip -l dist/mxframe-*.whl | head -40

# Check that .so files are included
unzip -l dist/mxframe-*.whl | grep '\.so'

# Twine checks metadata
twine check dist/*
```

Expected output from `twine check`:
```
Checking dist/mxframe-0.2.0-<platform-tag>.whl: PASSED
```

---

## 🚀 Step 5 — Publish to TestPyPI First

```bash
# Upload to TestPyPI
twine upload --repository testpypi dist/*

# Test install in a clean env
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    mxframe==0.2.0

# Quick sanity check
python3 -c "from mxframe import LazyFrame, col, lit; print('OK')"
```

---

## 🎉 Step 6 — Publish to PyPI

Once TestPyPI looks good:

```bash
twine upload dist/*
```

Or use the **GitHub Actions** workflow (recommended):

1. Go to your repo on GitHub
2. **Releases → Draft a new release**
3. Click **Create a new tag** → `v0.2.0`
4. Write release notes (paste from CHANGELOG)
5. Click **Publish release**

The `publish.yml` workflow fires automatically and uploads to PyPI.

---

## 🔁 Via GitHub Actions (Preferred)

### Test release to TestPyPI

```
GitHub → Actions → "📦 Publish to PyPI" → Run workflow
  └── target: testpypi
```

### Production release

```
GitHub → Releases → Create new release → tag v0.2.0 → Publish
  → publish.yml fires automatically
  → Builds wheel with Mojo-compiled kernels
  → Uploads to PyPI via OIDC (no secrets needed)
```

---

## 🔍 Verifying the Release

After publishing, verify from a fresh environment:

```bash
# In a new venv or conda env
pip install mxframe==0.2.0

python3 - <<'EOF'
import pyarrow as pa
import numpy as np
from mxframe import LazyFrame, Scan, col, lit

data = pa.table({
    "x": pa.array([1, 2, 3, 4, 5], pa.int32()),
    "y": pa.array([10.0, 20.0, 30.0, 40.0, 50.0], pa.float32()),
})

result = (
    LazyFrame(Scan(data))
    .filter(col("x") > lit(2))
    .groupby("x")
    .agg(col("y").sum().alias("total"))
    .compute(device="cpu")
)
print(result.to_pandas())
print("✅ mxframe install verified")
EOF
```

---

## ⚠️ Known Limitations for pip Install

| Limitation | Status | Notes |
|---|---|---|
| GPU requires Modular MAX runtime | ⚠️ Not pip-installable | User must install `modular` via pixi or conda separately |
| `.so` files are Linux-only (x86_64) | ⚠️ | Need separate builds for ARM64, macOS, Windows |
| Python 3.12 only (tested) | ⚠️ | May work on 3.10/3.11 but untested |

**For GPU support**, users need:
```bash
# Install Modular MAX runtime
curl -ssL https://magic.modular.com | bash
magic install modular
```

Then the GPU path activates automatically.

---

## 📐 Wheel Platform Tags

The release wheel must be platform-specific because it bundles native Linux `.so` files.
The default GitHub Actions build should now produce a Linux-tagged wheel rather than `py3-none-any`.
For a production release with broader Linux compatibility, use `manylinux` via `cibuildwheel`:

```bash
# Build with manylinux (requires Docker on Linux)
pip install cibuildwheel
cibuildwheel --platform linux

# This produces manylinux wheels:
# dist/mxframe-0.2.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

Future goal: add `cibuildwheel` step to `publish.yml` to auto-build platform wheels.
