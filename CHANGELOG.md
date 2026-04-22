# 📋 Changelog

All notable changes to MXFrame are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [0.1.1] — 2026-04-22

### 🛠 Fixed
- Packaging: `readme` now points to `README.md` so PyPI shows the full project page with benchmarks and quickstart (was accidentally showing the internal vision/architecture doc)
- CI: `AOTKernelsGPU` init no longer hard-fails on CPU-only runners (`CUDA driver unavailable` is caught, `self._aot_gpu` falls back to `None`)
- CI: `_find_kernels_path()` uses `Path(__file__).resolve().parent` directly, removing the stale `/home/ablearn/mxdf/…` dev fallback that surfaced in installed wheels
- Bug: `NameError: cache_key_count` in `_hash_join_mojo_cpu` MAX Graph fallback
- Tests: `scripts/_test_phase0_custom_ops.py` and `_test_phase1.py` no longer hard-code a developer's kernels path; they use the auto-detected default

---

## [0.1.0] — 2026-04-20

### ✨ Added
- **All 22 TPC-H queries** on CPU and GPU AOT paths
- **GPU LEFT JOIN** — `join_count_left` + `join_scatter_left` Mojo kernels wired to Python (Q13)
- `bench_simple.py` — clean 4-column benchmark (Pandas | Polars | MX CPU | MX GPU)
- `--queries` and `--runs` flags for targeted benchmarking

### ⚡ Performance
- **Q22** phone prefix anti-join: vectorized via `pc.utf8_slice_codeunits` + `pc.is_in` + `np.isin` — 3.6× faster than Polars CPU (was 10× **slower**)
- **Q20** semi-join chain: eliminated `.to_pandas()` Pandas detour — 8.7× faster than Polars (was comparable)
- **Q21** EXISTS+NOT EXISTS: replaced `.to_pandas().groupby.nunique()` with NumPy composite key dedup
- **Q18** large volume customers: replaced `.to_pylist()` semi-join with Mojo join — semi-join now uses AOT kernel path
- **Q15** argmax supplier: replaced `0.9999 * max_rev` tolerance hack with exact `pc.equal()` on float32
- **Q13** LEFT JOIN: removed `.to_pandas()` groupby detour — now fully on GPU AOT path (19ms CPU, 29ms GPU vs 25ms Polars)

### 🐛 Fixed
- `not use_gpu_filter` gate bug in `custom_ops.py` section 4.6 that blocked GPU filter path
- Q7 nation pre-join cache key now uses `id()` preventing cache misses on repeated hot calls

### 🏗️ Infrastructure
- `pyproject.toml` updated with full metadata, dependencies, optional extras, package-data for `.so` files
- `CONTRIBUTING.md` — developer guide with kernel writing tutorial
- `PUBLISHING.md` — step-by-step pip release guide
- GitHub Actions: `test.yml` (CI), `publish.yml` (PyPI via OIDC)

### 📊 Benchmark Summary (1M and 10M rows, all 22 queries, warm median of 3)

See the full table in [README.md § TPC-H Benchmark](README.md#-tpc-h-benchmark--all-22-queries),
sourced from `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv`.

- **Correctness:** 22/22 queries pass on CPU and GPU paths
- **MX CPU wins vs Polars:** **21/22** at 1 M, **18/22** at 10 M
  — headline: **Q9 128× · Q12 89× · Q7 42× · Q8 31× · Q17 24× · Q5 22×** (at 10 M)
- **MX GPU wins vs Polars:** **16/22** at 1 M, **15/22** at 10 M
  — headline: **Q12 26.5× · Q9 12× · Q8 10.8× · Q17 9.7× · Q7 4.7×** (at 10 M)
- **Remaining losses (Q4, Q6, Q13, Q21):** ops that still route through PyArrow fallback — kernel replacements tracked in `roadmap.md`
