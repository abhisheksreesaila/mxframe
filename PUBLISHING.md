# v0.5.0 release workflow

The authoritative local workflow is `scripts/release.sh`.

```bash
# Validate tests, rebuild GPU AOT, build wheel/sdist, run twine checks,
# and verify compiled libraries are present. Does not publish or tag.
scripts/release.sh

# After committing the validated release tree:
scripts/release.sh --tag       # annotated v0.5.0 tag + GitHub release
scripts/release.sh --upload    # PyPI upload
```

The script refuses `--tag` and `--upload` from a dirty worktree. GitHub Actions with PyPI trusted publishing remains the preferred production upload path when `publish.yml` is configured.

Release artifacts and evidence:

- `RELEASE_NOTES_0.5.0.md` — GitHub release body;
- `CHANGELOG.md` — concise version history;
- `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv` — bounded benchmark matrices;
- `dist/mxframe-0.5.0-py3-none-linux_x86_64.whl` — Linux platform wheel.

The sections below provide background and manual alternatives. Substitute the current version (`0.5.0`) in any older example.

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

```sh
pip install build twine
```

---

## 🔄 Release Checklist

### Before Every Release

- [ ] All tests pass: `pixi run test-all`
- [ ] GPU integration passes: `pixi run test6-gpu`
- [ ] Benchmark shows no regression versus the previous release
- [ ] `pyproject.toml` and `mxframe.__version__` agree
- [ ] `CHANGELOG.md` and the release-note artifact are updated
- [ ] Pre-built CPU/GPU `.so` files are current
- [ ] `twine check dist/*` passes

---

## 🏗️ Step 1 — Rebuild the AOT Kernels

Always rebuild before packaging to ensure the `.so` files match the source:

```sh
# CPU kernels
pixi run mojo build kernels_aot/kernels_aot.mojo -I kernels_aot \
    --emit shared-lib -o kernels_aot/libmxkernels_aot.so

# GPU kernels
pixi run mojo build kernels_aot/kernels_aot_gpu.mojo -I kernels_aot \
    --emit shared-lib -o kernels_aot/libmxkernels_aot_gpu.so

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

```sh
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

```sh
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

```sh
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

```sh
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

## ⚠️ Known limitations for pip install

| Limitation | Status | Notes |
|---|---|---|
| GPU requires Modular MAX runtime | Optional extra | `pip install "mxframe[runtime]==0.5.0"`; Pixi users should use the Modular channel environment instead. |
| Bundled `.so` files target Linux x86_64 | Current release target | Separate ARM64, macOS, and Windows artifacts are not published yet. |
| Python 3.12+ | Required | The current wheel is tagged `py3` but validated with Python 3.12. |
| GPU validation | NVIDIA RTX 3090 | AMD and Apple Silicon validation remains future work. |

For a pip-based GPU installation:

```bash
pip install "mxframe[runtime]==0.5.0"
```

For development and reproducible Mojo builds, use `pixi install` from the repository.

```sh
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

```sh
# Build with manylinux (requires Docker on Linux)
pip install cibuildwheel
cibuildwheel --platform linux

# This produces manylinux wheels:
# dist/mxframe-0.2.0-cp312-cp312-manylinux_2_17_x86_64.whl
```

Future goal: add `cibuildwheel` step to `publish.yml` to auto-build platform wheels.