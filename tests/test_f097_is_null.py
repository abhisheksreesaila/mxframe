"""F-097: IS NULL / IS NOT NULL skip Parquet row groups by their null counts, and run in the native bytecode.

Part 1: `scan_parquet` pushes `is_null` / `~is_null` to the reader. A row group with null_count 0 cannot hold a row
for IS NULL; one whose null_count equals its row count cannot hold a row for IS NOT NULL; a file without
statistics is read whole. The rows read are counted from the resolved scan (`scan.table.num_rows`), as in
tests/test_scan_parquet.py.

Part 2: the bytecode reads the Arrow validity bitmap (OP_ISNULL), so a filter or an aggregate whose only use of a
nullable column is an IS [NOT] NULL test stays on the native executor, on sliced and multi-chunk columns.

Every answer is compared with DuckDB's.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import LazyFrame, col
from mxframe import gpu_stream
from mxframe.io_parquet import resolve_plan
from mxframe.lazy_frame import Scan
from mxframe.custom_ops import CustomOpsCompiler

GPU = gpu_stream.cuda_available()
DEVICES = ["cpu"] + (["gpu"] if GPU else [])
GROUP = 500


def _leaf(plan):
    while not isinstance(plan, Scan):
        plan = plan.input
    return plan


def _rows(t):
    cols = [t.column(n).to_pylist() for n in t.column_names]
    cols = [["nan" if isinstance(v, float) and v != v else v for v in c] for c in cols]      # NaN == NaN here
    return sorted(zip(*cols), key=lambda r: tuple("" if v is None else str(v) for v in r))


def _duck(table, sql):
    con = duckdb.connect()
    con.register("t", table)
    out = con.execute(sql).arrow()
    return _rows(out.read_all() if hasattr(out, "read_all") else out)


# ── part 1: row groups ──────────────────────────────────────────────────────────────────────

NULL_GROUPS = (2, 5)          # row groups of `v` that hold NULLs
ALL_NULL_GROUP = 3            # row group of `w` that holds nothing else


def _table(n_groups=8):
    n = n_groups * GROUP
    ids = np.arange(n, dtype=np.int64)
    g = ids // GROUP
    v = [None if (gi in NULL_GROUPS and i % 7 == 0) else float(i) for i, gi in zip(ids.tolist(), g.tolist())]
    w = [None if gi == ALL_NULL_GROUP else i for i, gi in zip(ids.tolist(), g.tolist())]
    s = [None if gi in NULL_GROUPS and i % 5 == 0 else f"s{i % 4}" for i, gi in zip(ids.tolist(), g.tolist())]
    nan = [float("nan") if i % 11 == 0 else float(i) for i in ids.tolist()]       # NaN is a value, not NULL
    return pa.table({"id": pa.array(ids), "v": pa.array(v, pa.float64()), "w": pa.array(w, pa.int64()),
                     "s": pa.array(s, pa.string()), "nan": pa.array(nan, pa.float64())})


@pytest.fixture
def pq_file(tmp_path):
    p = str(tmp_path / "t.parquet")
    pq.write_table(_table(), p, row_group_size=GROUP)
    return p


def _read(path, predicate):
    lf = mx.scan_parquet(path).filter(predicate)
    return _leaf(resolve_plan(lf.plan)).table.num_rows, lf


def _expect(path, predicate, sql):
    n_read, lf = _read(path, predicate)
    got = _rows(lf.compute(device="cpu"))
    assert got == _duck(pq.read_table(path), sql)
    return n_read, got


def test_is_null_reads_only_row_groups_with_nulls(pq_file):
    n_read, got = _expect(pq_file, col("v").is_null(), "select * from t where v is null")
    assert n_read == 2 * GROUP                       # 2 of 8 row groups
    assert len(got) == 143                           # the multiples of 7 in ids 1000..1499 and 2500..2999


def test_is_not_null_skips_all_null_row_groups(pq_file):
    n_read, got = _expect(pq_file, ~col("w").is_null(), "select * from t where w is not null")
    assert n_read == 7 * GROUP                       # the all-NULL group is skipped
    assert len(got) == 7 * GROUP


def test_is_null_on_an_all_null_group_reads_just_it(pq_file):
    n_read, got = _expect(pq_file, col("w").is_null(), "select * from t where w is null")
    assert n_read == 1 * GROUP and len(got) == GROUP


def test_is_not_null_keeps_groups_with_some_nulls(pq_file):
    n_read, _ = _expect(pq_file, ~col("v").is_null(), "select * from t where v is not null")
    assert n_read == 8 * GROUP                       # no group is all NULL: nothing to skip


def test_string_column_is_pruned_too(pq_file):
    n_read, _ = _expect(pq_file, col("s").is_null(), "select * from t where s is null")
    assert n_read == 2 * GROUP


def test_nan_is_not_null(pq_file):
    n_read, got = _expect(pq_file, col("nan").is_null(), "select * from t where nan is null")
    assert n_read == 0 and got == []                 # no NULL anywhere: every row group is skipped
    n_read, got = _expect(pq_file, ~col("nan").is_null(), "select * from t where nan is not null")
    assert n_read == 8 * GROUP and len(got) == 8 * GROUP


def test_composes_with_and(pq_file):
    n_read, got = _expect(pq_file, col("v").is_null() & (col("id") >= 5 * GROUP),
                          "select * from t where v is null and id >= 2500")
    assert n_read == 1 * GROUP                       # null groups {2, 5} intersect id groups {5, 6, 7}
    n_read, _ = _expect(pq_file, (col("id") < 2 * GROUP) & col("v").is_null(),
                        "select * from t where id < 1000 and v is null")
    assert n_read == 0
    n_read, _ = _expect(pq_file, col("v").is_null() & ~col("w").is_null(),
                        "select * from t where v is null and w is not null")
    assert n_read == 2 * GROUP


def test_or_is_not_pushed_and_stays_right(pq_file):
    n_read, _ = _expect(pq_file, col("v").is_null() | (col("id") < 10),
                        "select * from t where v is null or id < 10")
    assert n_read == 8 * GROUP


def test_missing_statistics_never_skip(tmp_path):
    p = str(tmp_path / "nostats.parquet")
    pq.write_table(_table(), p, row_group_size=GROUP, write_statistics=False)
    for pred, sql in ((col("v").is_null(), "select * from t where v is null"),
                      (~col("w").is_null(), "select * from t where w is not null")):
        n_read, _ = _expect(p, pred, sql)
        assert n_read == 8 * GROUP


@pytest.mark.parametrize("name", ["l", "st"])
def test_list_and_struct_columns_are_not_pruned_by_leaf_counts(tmp_path, name):
    n = 2000
    t = pa.table({"id": pa.array(np.arange(n)),
                  "l": pa.array([[1, None]] * n, pa.list_(pa.int64())),
                  "st": pa.array([{"a": None}] * n, pa.struct([("a", pa.int64())]))})
    p = str(tmp_path / "nested.parquet")
    pq.write_table(t, p, row_group_size=250)
    n_read, lf = _read(p, ~col(name).is_null())
    assert n_read == n                                # every row is NOT NULL: nothing may be skipped
    assert lf.compute(device="cpu").num_rows == n
    n_read, lf = _read(p, col(name).is_null())
    assert n_read == n and lf.compute(device="cpu").num_rows == 0


def test_filters_expression_maps_null_tests():
    from mxframe.io_parquet import _filters_expression
    e = _filters_expression([("v", "isnull", None), ("w", "notnull", None)])
    assert "is_null(v" in str(e) and "invert(is_null(w" in str(e)


# ── part 2: the native bytecode ─────────────────────────────────────────────────────────────

N = 30_000


def _nullable_table():
    rng = np.random.default_rng(7)
    n = N
    dates = np.arange(n, dtype=np.int32) + 18000

    def holes(p):
        return (rng.random(n) < p).tolist()

    hx, hs, hd, hk = holes(0.1), holes(0.2), holes(0.05), holes(0.3)
    return pa.table({
        "i": pa.array(np.arange(n, dtype=np.int32)),
        "g": pa.array((np.arange(n) % 5).astype(np.int32)),
        "a": pa.array(rng.integers(0, 100, n).astype(np.int32)),
        "x": pa.array([None if m else float(v) for v, m in zip(rng.random(n), hx)], pa.float64()),
        "s": pa.array([None if m else f"str{v}" for v, m in zip(rng.integers(0, 9, n).tolist(), hs)], pa.string()),
        "d": pa.array([None if m else int(v) for v, m in zip(dates, hd)], pa.date32()),
        "k": pa.array([None if m else int(v) for v, m in zip(rng.integers(0, 6, n), hk)], pa.int64()),
        "e": pa.array([None if m else f"c{v}" for v, m in
                       zip(rng.integers(0, 4, n).tolist(), holes(0.25))], pa.string()).dictionary_encode(),
    })


def _shapes(t):
    """The same rows as one chunk, as many chunks cut at odd places, and as a slice (a validity bitmap whose
    first bit is not a multiple of 8)."""
    cuts = [0, 1, 7, 8001, 8002, 16003, 23999, t.num_rows]
    chunked = pa.concat_tables([t.slice(a, b - a) for a, b in zip(cuts, cuts[1:])])
    assert chunked.column("x").num_chunks == len(cuts) - 1
    sliced = pa.concat_tables([t.slice(0, 5), t.slice(5, 29_995)]).slice(3, N - 11)
    return {"one": t, "chunks": chunked, "sliced": sliced}


@pytest.fixture(scope="module")
def tables():
    return _shapes(_nullable_table())


@pytest.fixture(params=["one", "chunks", "sliced"])
def table(request, tables):
    return tables[request.param]


def _run_filter(table, build, device="cpu"):
    nx._filter_last.update(path=None, decline=None)
    lf = build(LazyFrame(table))
    return lf.compute(device=device), nx.last_filter()


FILTERS = [
    ("x is null", lambda: col("x").is_null(), "x is null"),
    ("x is not null", lambda: ~col("x").is_null(), "x is not null"),
    ("string null", lambda: col("s").is_null(), "s is null"),
    ("date null", lambda: ~col("d").is_null(), "d is not null"),
    ("int null", lambda: col("k").is_null(), "k is null"),
    ("dictionary null", lambda: col("e").is_null(), "e is null"),
    ("and a compare", lambda: col("x").is_null() & (col("a") > 50), "x is null and a > 50"),
    ("or a compare", lambda: col("s").is_null() | (col("a") < 5), "s is null or a < 5"),
    ("two null tests", lambda: col("x").is_null() & ~col("k").is_null(), "x is null and k is not null"),
    ("or of null tests", lambda: col("d").is_null() | col("k").is_null(), "d is null or k is null"),
    ("not of an or", lambda: ~(col("x").is_null() | (col("a") > 90)), "not (x is null or a > 90)"),
]


@pytest.mark.parametrize("name,pred,sql", FILTERS, ids=[f[0] for f in FILTERS])
def test_row_filter_is_native(table, name, pred, sql):
    got, (path, why) = _run_filter(table, lambda lf: lf.filter(pred()).select("i", "g", "a"))
    assert _rows(got) == _duck(table, f"select i, g, a from t where {sql}")
    assert (path, why) == ("native_filter", None)


def test_filter_leaves_nulls_of_the_tested_column_out_of_the_answer_checked(table):
    # the tested column is also an output: the native filter does not gather nulls, the answer is still right
    got, (path, _) = _run_filter(table, lambda lf: lf.filter(col("x").is_null()))
    assert _rows(got) == _duck(table, "select * from t where x is null")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,pred,sql", FILTERS, ids=[f[0] for f in FILTERS])
def test_grouped_count_is_native_on_cpu(table, device, name, pred, sql):
    seen = []
    run = CustomOpsCompiler._compile_and_run

    def recording(self, *a, **k):
        out = run(self, *a, **k)
        seen.append((self.last_compile_provenance or {}).get("path"))
        return out

    CustomOpsCompiler._compile_and_run = recording
    try:
        lf = LazyFrame(table).filter(pred()).groupby("g").agg(col("i").count().alias("n"), col("a").sum().alias("s"))
        got = lf.compute(device=device)
    finally:
        CustomOpsCompiler._compile_and_run = run
    assert _rows(got) == _duck(table, f"select g, count(i) n, sum(a) s from t where {sql} group by g")
    if device == "cpu":
        assert "native" in seen, seen


def test_a_compare_on_the_nullable_column_is_still_right(table):
    got, _ = _run_filter(table, lambda lf: lf.filter(col("x").is_null() | (col("x") > 0.5)).select("i"))
    assert _rows(got) == _duck(table, "select i from t where x is null or x > 0.5")
    got, _ = _run_filter(table, lambda lf: lf.filter(~(col("x") > 0.5)).select("i"))
    assert _rows(got) == _duck(table, "select i from t where not (x > 0.5)")


def test_a_column_without_nulls_in_this_table(tables):
    t = tables["one"].select(["i", "g", "a"])
    got, (path, _) = _run_filter(t, lambda lf: lf.filter(col("a").is_null()).select("i"))
    assert got.num_rows == 0
    got, _ = _run_filter(t, lambda lf: lf.filter(~col("a").is_null()).select("i"))
    assert got.num_rows == N


def test_sql_is_null_stays_native(table):
    nx._filter_last.update(path=None, decline=None)
    got = mx.sql("select i, g from t where x is null and a > 10", t=table).compute(device="cpu")
    assert _rows(got) == _duck(table, "select i, g from t where x is null and a > 10")
    assert nx.last_filter() == ("native_filter", None)
