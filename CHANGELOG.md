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

### 📊 Benchmark Summary (1M rows, hot min)

| Category | Queries | MX CPU vs Polars |
|---|---|---|
| Multi-table joins (Q5,Q7,Q8,Q9) | GPU-heavy | **35–77× faster** |
| Date filter + join (Q12) | Mixed | **25× faster** |
| Groupby + agg (Q1, Q11, Q17) | Compute-heavy | **3–25× faster** |
| String/semi-join (Q4, Q16, Q22) | Complex | **3–8× faster** |
| Global reduce (Q6) | Tiny output | ~1× (tied) |

**MX CPU faster than Polars: 20/22 queries**  
**MX GPU faster than Polars: 16/22 queries**
