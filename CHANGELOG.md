# 📋 Changelog

All notable changes to MXFrame are documented here.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

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

### 📊 Benchmark Summary (1M and 10M rows, all 22 queries)

See the full table in [README.md § TPC-H Benchmark](README.md#-tpc-h-benchmark--all-22-queries),
sourced from `scripts/bench_results_1M.csv` and `scripts/bench_results_10M.csv`.

- **Correctness:** 22/22 queries pass on CPU and GPU paths
- **MX CPU wins vs Polars (1 M, one-shot):** 11/22 — headline wins on Q11 (3.8×), Q18 (3.2×), Q17 (2.8×), Q2 (2.4×), Q7 (1.9×), Q21 (1.8×)
- **MX GPU wins vs Polars (10 M, warm cache):** 6/22 — headline wins on Q17 (5.5×), Q5 (4.1×), Q16 (2.5×), Q14 (2.2×), Q19 (1.3×), Q2 (1.3×)
- **Outstanding work:** Q8/Q9/Q12 dominated by join kernel without radix partitioning (tracked in `roadmap.md`)
