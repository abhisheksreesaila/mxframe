"""F-092: mx.sql accepts LazyFrames (scan_parquet, read_parquet, any plan) as tables, keeps scan pushdown,
and matches DuckDB over the same file."""
import subprocess

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
from mxframe import col
from mxframe.io_parquet import resolve_plan
from mxframe.lazy_frame import Scan, _resolve_scalars
from mxframe.optimizer import optimize_plan


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
