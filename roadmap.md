# 🗺️ MXFrame — Roadmap

> Ordered plan from current state to TPC-H competitive DataFrame library.

---

## Phase 0: Foundation Fixes 🔧
**Goal:** Make what exists actually correct before adding new features.
**Estimated effort:** 1-2 sessions

| # | Task | Notebook | Priority |
|---|------|----------|----------|
| 0.1 | **Fix `Filter` to actually remove rows** — replace mask-multiply with PyArrow pre-filter before graph entry (correct) or `ops.gather` inside graph (pure graph) | `03_compiler.ipynb` | 🔴 Critical |
| 0.2 | **Add `.count()` to `Expr`** — `Expr("count", self)` + compile to `ops.sum` of a ones-tensor or shape-based constant | `01_lazy_expr.ipynb`, `03_compiler.ipynb` | 🔴 Critical |
| 0.3 | **Fix grouped aggregation fallthrough** — `min/max/mean/count` with `groupby()` must error or dispatch correctly, not silently produce wrong results | `04_custom_ops.ipynb` | 🔴 Critical |
| 0.4 | **Notebook path fix** — ensure `KERNELS_PATH` works both in installed package and notebook contexts | `04_custom_ops.ipynb` | ✅ Done |
| 0.5 | **Return group-by keys in output** — the grouped result should include the key column(s), not just aggregated values | `04_custom_ops.ipynb` | 🟡 Important |

---

## Phase 1: Complete Grouped Aggregation (CPU) 🧮
**Goal:** All standard aggregations work with `groupby()` on CPU.
**Estimated effort:** 2-3 sessions

| # | Task | Details |
|---|------|---------|
| 1.1 | **Write `group_min.mojo` kernel** | Scatter-min by group IDs. Similar pattern to `group_sum.mojo`. |
| 1.2 | **Write `group_max.mojo` kernel** | Scatter-max by group IDs. |
| 1.3 | **Write `group_count.mojo` kernel** | Count elements per group. |
| 1.4 | **Write `group_mean.mojo` kernel** | Either `group_sum / group_count` or fused single-pass. |
| 1.5 | **Wire all kernels in `CustomOpsCompiler`** | Extend `_visit_aggregate_custom` to dispatch `min/max/mean/count`. |
| 1.6 | **Rebuild `kernels.mojopkg`** | Update `__init__.mojo`, run `build_kernels.sh`. |
| 1.7 | **Test with multi-column groupby** | `groupby('a', 'b').agg(col('x').sum(), col('y').mean())` |

**Milestone check:** `groupby().agg(sum, min, max, mean, count)` all produce correct results ✅

---

## Phase 2: GPU Path 🚀
**Goal:** Same operations run on GPU with automatic or manual device selection.
**Estimated effort:** 2-3 sessions

| # | Task | Details |
|---|------|---------|
| 2.1 | **Add GPU device detection** | `driver.GPU()` in compiler `__init__`, store available devices. |
| 2.2 | **Parameterize `DeviceRef`** | Replace hardcoded `DeviceRef.CPU()` with device selection logic. |
| 2.3 | **Ensure `group_sum.mojo` GPU path works** | It already has a warp-reduction GPU kernel — test it on real GPU. |
| 2.4 | **Add GPU variants of `group_min/max/count/mean`** | Warp-reduction pattern, same as `group_sum`. |
| 2.5 | **Add `.compute(device="gpu")` API** | Let users force GPU execution. |
| 2.6 | **Auto-device heuristic** | Data > 100K rows → GPU if available, else CPU. |
| 2.7 | **Benchmark: CPU vs GPU** | Time `groupby().agg(sum)` for 1M, 10M, 100M rows. |

**Milestone check:** Same query, same results, 10x faster on GPU for large data ✅

---

## Phase 3: TPC-H Q1 & Q6 🏆
**Goal:** Run TPC-H Q1 and Q6 end-to-end, benchmark against Polars/pandas.
**Estimated effort:** 2 sessions

### Q6 (simpler — no groupby)

```sql
SELECT sum(l_extendedprice * l_discount) AS revenue
FROM lineitem
WHERE l_shipdate >= '1994-01-01'
  AND l_shipdate < '1995-01-01'
  AND l_discount BETWEEN 0.05 AND 0.07
  AND l_quantity < 24;
```

| # | Task |
|---|------|
| 3.1 | **Add date comparison support** — date columns as int32 (days since epoch), compare with `lit()` |
| 3.2 | **Add `&` (and) boolean combinator** — `Expr("and", lhs, rhs)` → `ops.mul` or bitwise and |
| 3.3 | **Add BETWEEN sugar** — `col('x').between(lo, hi)` → `(col('x') >= lo) & (col('x') <= hi)` |
| 3.4 | **Run Q6** — filter + multiply + sum. Should already work with Phase 0+1 fixes. |
| 3.5 | **Benchmark Q6** — MXFrame vs Polars vs pandas vs DuckDB. |

### Q1 (grouped aggregation)

```sql
SELECT l_returnflag, l_linestatus,
       sum(l_quantity), sum(l_extendedprice),
       sum(l_extendedprice * (1 - l_discount)),
       sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)),
       avg(l_quantity), avg(l_extendedprice), avg(l_discount),
       count(*)
FROM lineitem
WHERE l_shipdate <= '1998-09-02'
GROUP BY l_returnflag, l_linestatus
ORDER BY l_returnflag, l_linestatus;
```

| # | Task |
|---|------|
| 3.6 | **Add `Sort` plan node** — `@dataclass class Sort(LogicalPlan)` + compile to `ops.sort` or PyArrow post-sort |
| 3.7 | **Wire the existing `fused_q1_full.mojo`** — it already computes all 24 aggregations for Q1 |
| 3.8 | **Run Q1** — filter + groupby(2 cols) + 10 aggregations + sort. |
| 3.9 | **Benchmark Q1** — MXFrame vs Polars vs pandas vs DuckDB, SF1/SF10. |

**Milestone check:** Q1 and Q6 produce correct results, MXFrame beats pandas, competitive with Polars ✅

---

## Phase 4: Sort, Limit, Distinct 📋
**Goal:** Add remaining single-table operations.
**Estimated effort:** 1-2 sessions

| # | Task | Details |
|---|------|---------|
| 4.1 | **`Sort` plan node** | `Sort(input, by=[col('x')], descending=[False])` |
| 4.2 | **`Limit` plan node** | `Limit(input, n=10)` → slice first N rows |
| 4.3 | **`Distinct` plan node** | `Distinct(input, subset=['col1'])` → unique rows |
| 4.4 | **Mojo sort kernel** | Radix sort or merge sort for GPU. Use `ops.sort` for CPU. |
| 4.5 | **LazyFrame API** | `.sort()`, `.limit()`, `.distinct()` methods |

---

## Phase 5: Joins 🔗
**Goal:** Hash join for two-table operations.
**Estimated effort:** 3-4 sessions

| # | Task | Details |
|---|------|---------|
| 5.1 | **`Join` plan node** | `Join(left, right, on=['key'], how='inner')` |
| 5.2 | **CPU hash join** | Build hash table on smaller side, probe with larger. PyArrow-based initially. |
| 5.3 | **Mojo hash join kernel** | Move hash table build + probe into Mojo for speed. |
| 5.4 | **GPU hash join** (stretch) | Mojo GPU kernel for join probe. |
| 5.5 | **Multi-table LazyFrame API** | `df1.join(df2, on='key', how='inner')` |
| 5.6 | **TPC-H Q3** | 3-way join + groupby + sort + limit. First join-heavy benchmark. |

---

## Phase 6: SQL Frontend 📝
**Goal:** Let users write SQL queries that compile to the same LogicalPlan → MAX Graph pipeline.
**Estimated effort:** 2-3 sessions
**Key insight:** SQL is just another frontend. `sqlglot` parses SQL → we translate to our `LogicalPlan` nodes → same compiler, same kernels, same results.

| # | Task | Details |
|---|------|--------|
| 6.1 | **SQL parser integration** | `sqlglot` (already in `pixi.toml`) parses SQL string → `sqlglot.Expression` AST |
| 6.2 | **`sql_frontend.py` translator** | Walk `sqlglot` AST → emit `Scan`, `Filter`, `Project`, `Aggregate` plan nodes |
| 6.3 | **`mx.sql()` public API** | `mx.sql("SELECT ...", t1=table1, t2=table2)` → `LazyFrame` with plan built from SQL |
| 6.4 | **SELECT + WHERE + GROUP BY** | Cover Tier 1 SQL (Q1, Q6 equivalents) |
| 6.5 | **JOIN support in SQL** | Translate `FROM t1 JOIN t2 ON ...` → `Join` plan node (requires Phase 5) |
| 6.6 | **ORDER BY + LIMIT** | Translate to `Sort` + `Limit` plan nodes |
| 6.7 | **Re-run TPC-H Q1, Q6 via SQL** | Prove SQL frontend produces identical results to DataFrame API |
| 6.8 | **Notebook: `nbs/05_sql_frontend.ipynb`** | Document & export `sql_frontend.py` |

**Progressive SQL support:** We don't need all SQL at once. Start with simple SELECT/WHERE/GROUP BY (matches Phase 0-3 capabilities), then add JOIN/ORDER BY/LIMIT as those plan nodes become available.

---

## Phase 7: I/O & Usability 📂
**Goal:** Read/write common formats, pandas-like convenience.
**Estimated effort:** 2 sessions

| # | Task |
|---|------|
| 7.1 | `mx.read_csv()` — wraps PyArrow CSV reader → `LazyFrame(Scan(...))` |
| 7.2 | `mx.read_parquet()` — wraps PyArrow Parquet reader |
| 7.3 | `mx.from_pandas()` / `mx.from_polars()` — conversion helpers |
| 7.4 | `.to_pandas()` / `.to_polars()` — on computed results |
| 7.5 | `.head()`, `.tail()`, `.describe()` — convenience methods |
| 7.6 | `__repr__` for `LazyFrame` — show the plan tree, not the data |
| 7.7 | `.explain()` — print the logical plan for debugging |

---

## Phase 8: Polish & Release 🎁
**Goal:** Documentation, packaging, benchmarks site.
**Estimated effort:** 2 sessions

| # | Task |
|---|------|
| 8.1 | nbdev docs site with all notebooks rendered |
| 8.2 | README with installation, quickstart, benchmarks |
| 8.3 | PyPI package (`pip install mxframe`) |
| 8.4 | Benchmark suite: TPC-H Q1, Q3, Q6 at SF1, SF10 |
| 8.5 | Comparison charts: MXFrame vs pandas vs Polars vs DuckDB |
| 8.6 | GitHub Actions CI: `nbdev_test` on every push |

---

## TPC-H Progressive Validation Map 🧭

Every phase unlocks new TPC-H queries. We run them as **validation gates** — if a query passes, we've proven that set of capabilities actually works end-to-end.

### TPC-H Query Complexity Tiers

| Tier | Queries | Required Capabilities |
|------|---------|----------------------|
| **Tier 1 — Scan + Filter + Agg** | Q1, Q6 | `filter`, `groupby`, `sum`, `avg`, `count` |
| **Tier 2 — Simple Join (2 tables)** | Q12, Q14 | Tier 1 + `join` (2 tables), `sort` |
| **Tier 3 — Multi-Join + Sort + Limit** | Q3, Q5, Q10 | Tier 2 + `join` (3-6 tables), `limit` |
| **Tier 4 — Subqueries + Exists** | Q4, Q11, Q15, Q17, Q18, Q20, Q21, Q22 | Tier 3 + subqueries, `exists`, `in` |
| **Tier 5 — String Ops + Complex** | Q2, Q7, Q8, Q9, Q13, Q16, Q19 | Tier 4 + `like`, `substring`, `case when` |

### Phase → TPC-H Mapping

| Phase | What it unlocks | TPC-H Queries to Test | Pass Criteria |
|-------|----------------|----------------------|---------------|
| **Phase 0+1** | Correct filter + grouped agg (CPU) | **Q6** (filter+sum), **Q1** (grouped agg) | Correct results vs DuckDB |
| **Phase 2** | GPU execution | Re-run **Q1, Q6** on GPU | ✅ Same results, faster than CPU |
| **Phase 3** | Date ops, boolean combinators | **Q1, Q6** benchmarked | ✅ Beat pandas, competitive with Polars |
| **Phase 4** | Sort, limit, distinct | **Q1** with ORDER BY, **Q6** verified | ✅ Full Q1 including sort |
| **Phase 5** | Joins | **Q3** (3-way join), **Q12** (2-way), **Q14** (2-way) | ✅ Correct results, benchmark |
| **Phase 5+** | CASE WHEN + isin + startswith | **Q12, Q14** with CASE WHEN | ✅ Exact match DuckDB |
| **Phase 5+** | More joins | **Q5** (6-way), **Q10** (4-way) | ✅ Correct results vs DuckDB |
| **Phase 6** | SQL frontend | **Q1, Q6, Q3, Q12, Q14** via `mx.sql()` | ✅ All match DuckDB |
| **Phase 7** | I/O | `read_csv`, `read_parquet`, `from_pandas` | ✅ Done |
| **Phase 8** | Polish & Release | README, PyPI, scoreboard | ⬜ Next |
| **Phase 9** | GPU Performance | GPU-native join gather, warm-start caching | ✅ Done |

### Running TPC-H Validation

Each validation creates a benchmark notebook: `nbs/bench_qNN.ipynb`

```python
# Pattern for every TPC-H validation notebook:

# 1. Load TPC-H data (SF1 or SF10)
lineitem = mx.read_parquet("data/tpch/sf1/lineitem.parquet")

# 2. Run MXFrame query
mx_result = lineitem.filter(...).groupby(...).agg(...).compute()

# 3. Run DuckDB reference query (ground truth)
duck_result = duckdb.sql("SELECT ... FROM lineitem ...").arrow()

# 4. Assert correctness
assert_tables_equal(mx_result, duck_result)

# 5. Benchmark timing
# MXFrame CPU: X ms
# MXFrame GPU: Y ms
# Polars:      Z ms
# DuckDB:      W ms
# pandas:      V ms
```

### Scoreboard

We maintain a living scoreboard in `docs/tpch-scoreboard.md`:

```
| Query | Correct? | MXFrame CPU | MXFrame GPU | Polars | DuckDB | pandas |
|-------|----------|-------------|-------------|--------|--------|--------|
| Q1    | ❌        | —           | —           | —      | —      | —      |
| Q3    | ❌        | —           | —           | —      | —      | —      |
| Q6    | ❌        | —           | —           | —      | —      | —      |
| ...   |          |             |             |        |        |        |
```

This gets updated every time we pass a new query. It's both motivation and proof.

---

## Summary Timeline

```
Phase 0: Foundation Fixes          ████████████████████  ✅ DONE
Phase 1: Grouped Aggregation (CPU) ████████████████████  ✅ DONE — Q6, Q1
Phase 2: GPU Path                  ████████████████████  ✅ DONE — GPU Q1/Q6
Phase 3: TPC-H Q1 & Q6            ████████████████████  ✅ DONE — benchmarks
Phase 4: Sort / Limit / Distinct   ████████████████████  ✅ DONE — full Q1 ORDER BY
Phase 5: Joins                     ████████████████████  ✅ DONE — Q3,Q12,Q14,Q5,Q10
Phase 5+: Tier 2 CASE WHEN/isin   ████████████████████  ✅ DONE — Q12/Q14 CASE WHEN
Phase 6: SQL Frontend              ████████████████████  ✅ DONE — Q1,Q6,Q3,Q12,Q14 via SQL
Phase 7: I/O & Usability           ████████████████████  ✅ DONE — read_csv/parquet/pandas
Phase 9: GPU Join Gather + Warmup  ████████████████████  ✅ DONE — GPU gather, mx.warmup()
Phase 8: Polish & Release          ░░░░░░░░░░░░░░░░░░░░  ← NEXT — README, scoreboard, PyPI
```

> **Status as of 2026-04-04:** All feature phases (0–7) are complete. Full TPC-H Tier 1, 2, and 3
> pass (Q1, Q3, Q5, Q6, Q10, Q12, Q14) with correct results vs DuckDB ground truth.
> SQL frontend (`mx.sql()`) supports SELECT/JOIN/WHERE/CASE WHEN/IN/LIKE/GROUP BY/ORDER BY/LIMIT.

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-02-21 | Stay on Python 3.12 | `cudf-polars-cu12` has no 3.14 wheels |
| 2026-02-21 | Polars-style API, not pandas | Lazy evaluation aligns with MAX Graph compilation model |
| 2026-02-21 | PyArrow as interchange format | Zero-copy to/from MAX tensors; universal compatibility |
| 2026-02-21 | Phase 0 first (fix bugs) | Correctness before features. Grouped min/max/mean is broken. |
| 2026-02-21 | GPU before TPC-H | GPU is the differentiator; prove it works early |
| 2026-02-21 | Progressive TPC-H validation | Each phase unlocks new TPC-H queries as validation gates; maintain a living scoreboard |
| 2026-02-21 | SQL frontend via sqlglot | SQL is just another frontend to the same LogicalPlan; sqlglot already in deps; enables TPC-H validation via raw SQL |
