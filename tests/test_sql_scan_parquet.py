"""F-092: mx.sql accepts LazyFrames (scan_parquet, read_parquet, any plan) as tables, keeps scan pushdown,
and matches DuckDB over the same file."""
import subprocess

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
from omxcore import col
from omxcore.io_parquet import resolve_plan
from omxcore.lazy_frame import Scan, _resolve_scalars
from omxcore.optimizer import optimize_plan


def _devices():
    out = ["cpu"]
    try:
        if subprocess.run(["nvidia-smi", "-L"], capture_output=True).returncode == 0:
            out += ["gpu", "auto"]
    except OSError:
        pass
    return out


@pytest.fixture(params=_devices())
def dev(request):
    return request.param


N = 4000


@pytest.fixture
def path(tmp_path):
    rng = np.random.default_rng(7)
    t = pa.table({
        "id": pa.array(np.arange(N, dtype=np.int64)),
        "grp": pa.array(rng.choice(["alpha", "beta", "gamma", "delta"], N)),
        "qty": pa.array(rng.integers(1, 50, N).astype(np.int64)),
        "price": pa.array(rng.integers(1, 1000, N).astype(np.float64)),
        "note": pa.array([f"free text {i}" for i in range(N)]),
    })
    p = str(tmp_path / "t.parquet")
    pq.write_table(t, p, row_group_size=500)          # 8 row groups
    return p


@pytest.fixture
def path2(tmp_path):
    t = pa.table({"grp": ["alpha", "beta", "gamma", "delta"], "w": [1, 2, 3, 4],
                  "unused": ["a", "b", "c", "d"]})
    p = str(tmp_path / "u.parquet")
    pq.write_table(t, p)
    return p


def _duck(q, mem=None, **files):
    con = duckdb.connect()
    for name, f in files.items():
        con.execute(f"CREATE VIEW {name} AS SELECT * FROM read_parquet('{f}')")
    for name, t in (mem or {}).items():
        con.register(name, t)
    return con.execute(q).to_arrow_table()


def _same(got: pa.Table, want: pa.Table, ordered: bool):
    assert got.column_names == want.column_names
    g = list(zip(*[got.column(n).to_pylist() for n in got.column_names]))
    w = list(zip(*[want.column(n).to_pylist() for n in want.column_names]))
    if not ordered:
        g, w = sorted(g, key=str), sorted(w, key=str)
    assert g == w


def _leaves(plan):
    out = []

    def walk(p):
        if isinstance(p, Scan):
            out.append(p)
        for a in ("input", "left", "right"):
            if hasattr(p, a):
                walk(getattr(p, a))
    walk(plan)
    return out


def _read_plan(lf):
    """The plan compute() reads from: scalar subqueries run, optimizer applied, Parquet sources resolved."""
    return resolve_plan(optimize_plan(_resolve_scalars(lf.plan, "cpu", False)).plan)


@pytest.fixture(params=["scan", "read"])
def src(request):
    return {"scan": mx.scan_parquet, "read": mx.read_parquet}[request.param]


def test_filter_and_group_by(path, src, dev):
    q = "SELECT grp, SUM(qty) AS s, COUNT(*) AS n FROM t WHERE id >= 1000 AND qty > 10 GROUP BY grp"
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=False)


def test_order_by_limit(path, src, dev):
    q = "SELECT id, price FROM t WHERE qty > 25 ORDER BY price DESC, id LIMIT 7"
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=True)


def test_join_of_two_scans(path, path2, src, dev):
    q = "SELECT u.grp, SUM(t.price * u.w) AS s FROM t JOIN u ON t.grp = u.grp WHERE t.id < 2000 GROUP BY u.grp"
    got = mx.sql(q, t=src(path), u=src(path2)).compute(device=dev)
    _same(got, _duck(q, t=path, u=path2), ordered=False)


def test_scan_joined_with_in_memory_table(path, src, dev):
    mem = pa.table({"grp": ["alpha", "beta"], "w": [10, 20]})
    q = "SELECT t.grp, COUNT(*) AS n, SUM(w) AS sw FROM t JOIN m ON t.grp = m.grp GROUP BY t.grp"
    got = mx.sql(q, t=src(path), m=mem).compute(device=dev)
    _same(got, _duck(q, mem={"m": mem}, t=path), ordered=False)


def test_subquery(path, src, dev):
    q = ("SELECT grp, COUNT(*) AS n FROM t WHERE qty > (SELECT AVG(qty) FROM t) "
         "AND id IN (SELECT id FROM t WHERE id < 2000) GROUP BY grp")
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=False)


def test_derived_table_and_with(path, src, dev):
    q = ("WITH big AS (SELECT grp, qty FROM t WHERE qty > 40) "
         "SELECT d.grp, SUM(d.q) AS s FROM (SELECT grp, qty AS q FROM big) d GROUP BY d.grp")
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=False)


def test_self_join(path, src, dev):
    q = "SELECT a.grp, COUNT(*) AS n FROM t a JOIN t b ON a.id = b.id WHERE a.id < 100 GROUP BY a.grp"
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=False)


def test_any_lazyframe_plan_is_a_table(path, dev):
    lf = mx.scan_parquet(path).filter(col("qty") > 30).select("id", "grp", "qty")
    q = "SELECT grp, SUM(qty) AS s FROM v GROUP BY grp"
    want = _duck("SELECT grp, SUM(qty) AS s FROM (SELECT id, grp, qty FROM t WHERE qty > 30) v GROUP BY grp", t=path)
    _same(mx.sql(q, v=lf).compute(device=dev), want, ordered=False)


def test_star_over_scan(path, dev):
    q = "SELECT * FROM t WHERE id < 5 ORDER BY id"
    _same(mx.sql(q, t=mx.scan_parquet(path)).compute(device=dev), _duck(q, t=path), ordered=True)


# -- pushdown survives the SQL front end ------------------------------------

def test_row_groups_skipped_and_columns_pruned(path):
    lf = mx.sql("SELECT grp, COUNT(*) AS n FROM t WHERE id >= 3500 GROUP BY grp", t=mx.scan_parquet(path))
    (leaf,) = _leaves(_read_plan(lf))
    assert leaf.table.num_rows == 500                        # 1 of 8 row groups read
    assert sorted(leaf.table.column_names) == ["grp", "id"]  # qty, price, note never read
    out = lf.compute(device="cpu")
    assert sum(out.column("n").to_pylist()) == 500


def test_pushdown_through_join_and_subquery(path, path2):
    lf = mx.sql("SELECT u.grp, SUM(t.price) AS s FROM t JOIN u ON t.grp = u.grp "
                "WHERE t.id >= 3500 AND t.qty > (SELECT MIN(qty) FROM t) GROUP BY u.grp",
                t=mx.scan_parquet(path), u=mx.scan_parquet(path2))
    leaves = _leaves(_read_plan(lf))
    assert all("unused" not in l.table.column_names and "note" not in l.table.column_names for l in leaves)
    assert min(l.table.num_rows for l in leaves) == 4        # u: whole file
    assert any(l.table.num_rows == 500 for l in leaves)      # t: pushed filter read one row group


# -- regressions: the same query text over different sources ----------------

def _write(tmp_path, name, seed):
    rng = np.random.default_rng(seed)
    t = pa.table({"id": pa.array(np.arange(600, dtype=np.int64)),
                  "grp": pa.array(rng.choice(["a", "b", "c"], 600)),
                  "qty": pa.array(rng.integers(1, 50, 600).astype(np.int64))})
    p = str(tmp_path / name)
    pq.write_table(t, p, row_group_size=200)
    return p


def test_same_join_query_over_two_files_back_to_back(tmp_path, src, dev):
    q = ("SELECT t.grp, SUM(t.qty * u.w) AS s, COUNT(*) AS n FROM t JOIN u ON t.grp = u.grp "
         "WHERE t.id >= 100 GROUP BY t.grp")
    wp = str(tmp_path / "w.parquet")
    pq.write_table(pa.table({"grp": ["a", "b", "c"], "w": [1, 2, 3]}), wp)
    for p in (_write(tmp_path, "a.parquet", 1), _write(tmp_path, "b.parquet", 2), _write(tmp_path, "a.parquet", 1)):
        _same(mx.sql(q, t=src(p), u=src(wp)).compute(device=dev), _duck(q, t=p, u=wp), ordered=False)


def test_same_file_with_different_pre_filters(path, dev):
    q = "SELECT grp, SUM(qty) AS s, COUNT(*) AS n FROM v GROUP BY grp"
    for lo in (3000, 100, 3900):
        v = mx.scan_parquet(path).filter(col("id") >= lo)
        want = _duck(f"SELECT grp, SUM(qty) AS s, COUNT(*) AS n FROM (SELECT * FROM t WHERE id >= {lo}) v GROUP BY grp",
                     t=path)
        _same(mx.sql(q, v=v).compute(device=dev), want, ordered=False)


@pytest.mark.parametrize("how", ["JOIN", "LEFT JOIN"])
def test_self_join_filtered_on_one_side_only(path, src, dev, how):
    q = (f"SELECT x.grp, COUNT(*) AS n, COUNT(y.id) AS ny FROM t x {how} t y "
         "ON x.id = y.id WHERE x.id < 300 GROUP BY x.grp")
    _same(mx.sql(q, t=src(path)).compute(device=dev), _duck(q, t=path), ordered=False)
    q2 = (f"SELECT x.grp, COUNT(*) AS n FROM t x {how} t y ON x.id = y.id "
          "WHERE x.qty > 25 AND x.id < 300 GROUP BY x.grp")
    _same(mx.sql(q2, t=src(path)).compute(device=dev), _duck(q2, t=path), ordered=False)


def test_aggregated_lazyframe_as_table(path, dev):
    agg = mx.scan_parquet(path).groupby("grp").agg(col("qty").sum().alias("total"), col("id").count().alias("n"))
    q = "SELECT grp, total, n FROM g WHERE total > 0 ORDER BY total DESC, grp"
    want = _duck("SELECT grp, total, n FROM (SELECT grp, SUM(qty) AS total, COUNT(id) AS n FROM t GROUP BY grp) g "
                 "WHERE total > 0 ORDER BY total DESC, grp", t=path)
    _same(mx.sql(q, g=agg).compute(device=dev), want, ordered=True)


def test_renamed_lazyframe_as_table(path, dev):
    lf = mx.scan_parquet(path).select(col("id").alias("key"), col("qty").alias("amount"), "grp")
    q = "SELECT grp, SUM(amount) AS s FROM r WHERE key < 2000 GROUP BY grp"
    want = _duck("SELECT grp, SUM(qty) AS s FROM t WHERE id < 2000 GROUP BY grp", t=path)
    _same(mx.sql(q, r=lf).compute(device=dev), want, ordered=False)
