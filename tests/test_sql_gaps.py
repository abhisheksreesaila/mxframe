"""F-042: SQL front end gaps found in the F-041 review. Unnamed aggregates take DuckDB's names and
keep their SELECT position, a joined key keeps the name the query wrote, GROUP BY ALL, HAVING (a
Filter above the Aggregate), COUNT(DISTINCT x) and arithmetic on aggregates. Every variant is checked
against DuckDB; duplicate output names and constructs we cannot translate raise instead of changing
meaning."""
from decimal import Decimal

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
from mxframe.np_arrow import array as A


def _t(n=5_000, seed=11):
    r = np.random.default_rng(seed)
    s = np.array(["a", "b", "c", "d"])[r.integers(0, 4, n)].tolist()
    x = r.integers(0, 6, n).tolist()
    for i in range(0, n, 17):
        x[i] = None                       # COUNT(DISTINCT x) skips nulls
    return pa.table({
        "s": pa.array(s, pa.string()),
        "k": A(r.integers(0, 7, n).astype(np.int64)),
        "v": A(r.integers(1, 100, n).astype(np.int64)),
        "x": pa.array(x, pa.int64()),
    })


def _u():
    return pa.table({"kid": A(np.arange(7, dtype=np.int64)), "name": pa.array(list("pqrstuv"), pa.string())})


def _norm(x):
    if isinstance(x, (int, float, Decimal)) and not isinstance(x, bool):
        return round(float(x), 6)
    return x


def _w():
    return pa.table({"k": A(np.arange(7, dtype=np.int64)), "v": A(np.arange(7, dtype=np.int64) * 100)})


def _n():
    """NULLs in the first column and in the group keys; group 'c' has only NULL values, NULL keys
    form a group of their own."""
    return pa.table({
        "x": pa.array([None, 1, 1, 2, None, 3, 3, 3, 4, None], pa.int64()),
        "g": pa.array([None, "a", "a", "b", None, "a", "b", "b", "c", "c"], pa.string()),
        "v": pa.array([5, 1, 2, None, None, 3, 4, 5, None, None], pa.int64()),
    })


def _run(q):
    t, u, w, tn = _t(), _u(), _w(), _n()
    exp = duckdb.sql(q).arrow()
    if isinstance(exp, pa.RecordBatchReader):
        exp = exp.read_all()
    got = mx.sql(q, t=t, u=u, w=w, tn=tn).compute(device="cpu")
    return got, exp


def _check(q):
    got, exp = _run(q)
    assert got.column_names == exp.column_names
    cols = exp.column_names
    g = list(zip(*[[_norm(x) for x in got.column(c).to_pylist()] for c in cols]))
    e = list(zip(*[[_norm(x) for x in exp.column(c).to_pylist()] for c in cols]))
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e
    assert len(e) > 0 or "HAVING" in q


NAMING = [
    "SELECT SUM(v), s FROM t GROUP BY s",
    "SELECT s, SUM(v), COUNT(*), COUNT(v), AVG(v), MIN(v), MAX(v) FROM t GROUP BY s",
    "SELECT SUM(t.v) FROM t",
    "SELECT COUNT(*), SUM(V) FROM t",
    "SELECT MAX(v), s, MIN(v) AS lo FROM t GROUP BY s",
    "SELECT s AS b, SUM(v) FROM t GROUP BY s ORDER BY b",
    "SELECT s, SUM(v) FROM t GROUP BY s ORDER BY SUM(v) DESC",
    "SELECT s, SUM(v) FROM t GROUP BY s ORDER BY SUM(v) DESC, s LIMIT 2",
    "SELECT s FROM t GROUP BY s ORDER BY SUM(v) DESC",
]

JOIN_KEYS = [
    "SELECT kid FROM t JOIN u ON k = kid",
    "SELECT kid, name FROM t JOIN u ON k = kid WHERE v > 90",
    "SELECT kid, SUM(v) AS x FROM t JOIN u ON k = kid GROUP BY kid",
    "SELECT kid, SUM(v) FROM t JOIN u ON k = kid GROUP BY kid",
    "SELECT k, SUM(v) AS x FROM t JOIN u ON k = kid GROUP BY k",
    "SELECT kid AS id, SUM(v) AS x FROM t JOIN u ON k = kid GROUP BY kid",
    "SELECT u.kid, SUM(v) AS x FROM t JOIN u ON t.k = u.kid GROUP BY u.kid",
    "SELECT kid, SUM(v) AS x FROM t LEFT JOIN u ON k = kid GROUP BY kid",      # the right key: NULL where unmatched (F-086)
]

GROUP_ALL = [
    "SELECT s, SUM(v) AS x FROM t GROUP BY ALL",
    "SELECT SUM(v) AS x, s FROM t GROUP BY ALL",
    "SELECT s AS b, k + 1 AS k1, SUM(v) AS x FROM t GROUP BY ALL",
    "SELECT s, k, COUNT(*) AS n, MAX(v) AS m FROM t GROUP BY ALL ORDER BY s, k",
    "SELECT s, SUM(v) FROM t GROUP BY ALL",
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY ALL",
    "SELECT SUM(v) AS x FROM t GROUP BY ALL",
    "SELECT s, SUM(v) AS x FROM t WHERE v > 50 GROUP BY ALL HAVING SUM(v) > 100000",
]

HAVING = [
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING SUM(v) > 125000",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING x > 125000",
    "SELECT s, SUM(v) FROM t GROUP BY s HAVING SUM(v) > 125000",
    "SELECT s FROM t GROUP BY s HAVING SUM(v) > 125000",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING AVG(v) > 50.5",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING COUNT(*) > 1200 AND s <> 'a'",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING s = 'a' OR SUM(v) > 125000",
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY s HAVING b = 'a'",
    "SELECT s AS b, SUM(v) AS x FROM t GROUP BY s HAVING s = 'a'",
    "SELECT k + 1 AS k1, SUM(v) AS x FROM t GROUP BY k + 1 HAVING k + 1 > 3",
    "SELECT SUM(v) AS x FROM t HAVING SUM(v) > 1",
    "SELECT SUM(v) AS x FROM t HAVING SUM(v) > 1000000000",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING SUM(v) > 125000 ORDER BY x DESC LIMIT 2",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING SUM(v) * 2 > 250000",
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY s HAVING y > 250000",
    "SELECT name, SUM(v) AS x FROM t JOIN u ON k = kid GROUP BY name HAVING SUM(v) > 35000",
    "SELECT s, SUM(v) AS x FROM t WHERE v > 20 GROUP BY s HAVING MAX(v) = 99 AND MIN(v) >= 21",
]

COUNT_DISTINCT = [
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY s",
    "SELECT s, COUNT(DISTINCT k) FROM t GROUP BY s",
    "SELECT COUNT(DISTINCT k) AS n FROM t",
    "SELECT COUNT(DISTINCT k) FROM t",
    "SELECT COUNT(DISTINCT x) AS n FROM t",
    "SELECT s, COUNT(DISTINCT x) AS n FROM t GROUP BY s",
    "SELECT k, COUNT(DISTINCT s) AS n FROM t GROUP BY k",
    "SELECT s, COUNT(DISTINCT s) AS n FROM t GROUP BY s",
    "SELECT s AS b, COUNT(DISTINCT k) AS n FROM t GROUP BY s",
    "SELECT COUNT(DISTINCT k) AS n, s FROM t GROUP BY s",
    "SELECT s, k, COUNT(DISTINCT x) AS n FROM t GROUP BY s, k",
    "SELECT k + 1 AS k1, COUNT(DISTINCT x) AS n FROM t GROUP BY k + 1",
    "SELECT s, COUNT(DISTINCT k + 1) AS n FROM t GROUP BY s",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t WHERE v > 90 GROUP BY s",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY s HAVING COUNT(DISTINCT k) > 6",
    "SELECT s FROM t GROUP BY s HAVING COUNT(DISTINCT x) > 5",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY s ORDER BY n DESC, s LIMIT 3",
    "SELECT s, COUNT(DISTINCT k) * 10 AS n FROM t GROUP BY s",
    "SELECT name, COUNT(DISTINCT v) AS n FROM t JOIN u ON k = kid GROUP BY name",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY ALL",
]

ARITHMETIC = [
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY s",
    "SELECT s, SUM(v) AS x, SUM(v) * 2 AS y FROM t GROUP BY s",
    "SELECT s, SUM(v) * 2 AS y, SUM(v) AS x FROM t GROUP BY s",
    "SELECT SUM(v) * 2 AS y, s FROM t GROUP BY s",
    "SELECT s, SUM(v) / COUNT(*) AS m FROM t GROUP BY s",
    "SELECT s, SUM(v) * 2 + MAX(k) AS z, MIN(v) AS lo FROM t GROUP BY s",
    "SELECT s, 100 * SUM(v) / SUM(k + 1) AS pct FROM t GROUP BY s",
    "SELECT SUM(v) * 2 AS y FROM t",
    "SELECT SUM(v) * 2 AS y, COUNT(*) AS n FROM t",
    "SELECT s, (SUM(v) - MIN(v)) * 2 AS y FROM t GROUP BY s",
    "SELECT s, -SUM(v) AS y FROM t GROUP BY s",
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY s ORDER BY y DESC",
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY s ORDER BY SUM(v) * 2 DESC",
    "SELECT s AS b, SUM(v) * 2 AS y FROM t GROUP BY s ORDER BY b",
    "SELECT s, SUM(v) * k AS y FROM t GROUP BY s, k",
    "SELECT s, CASE WHEN SUM(v) > 125000 THEN 1 ELSE 0 END AS big FROM t GROUP BY s",
    "SELECT SUM(CASE WHEN v > 50 THEN v ELSE 0 END) * 2 AS y FROM t",
]


MIXED_DISTINCT = [
    "SELECT s, COUNT(DISTINCT k) AS n, SUM(v) AS x FROM t GROUP BY s",
    "SELECT s, SUM(v), COUNT(DISTINCT k) FROM t GROUP BY s",
    "SELECT COUNT(DISTINCT k) AS n, SUM(v) AS x, MIN(v) AS lo, MAX(v) AS hi, COUNT(*) AS c, COUNT(v) AS cv FROM t",
    "SELECT s, COUNT(DISTINCT x) AS n, AVG(v) AS a, COUNT(*) AS c FROM t GROUP BY s",
    "SELECT s, COUNT(DISTINCT k) AS n, SUM(v) * 2 AS y FROM t GROUP BY s",
    "SELECT s, k, COUNT(DISTINCT x) AS n, MAX(v) AS m FROM t GROUP BY s, k",
    "SELECT s AS b, COUNT(DISTINCT k) AS n, SUM(v) AS x FROM t GROUP BY s HAVING SUM(v) > 125000",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY s HAVING SUM(v) > 125000",
    "SELECT s, COUNT(DISTINCT k) AS n, SUM(v) AS x FROM t GROUP BY s ORDER BY x DESC LIMIT 2",
    "SELECT s, COUNT(DISTINCT x) AS n, SUM(x) AS sx FROM t GROUP BY s",
    "SELECT x, COUNT(DISTINCT k) AS n, SUM(v) AS sv FROM t GROUP BY x",   # null group key
    "SELECT name, COUNT(DISTINCT v) AS n, SUM(v) AS x FROM t JOIN u ON k = kid GROUP BY name",
]

DISTINCT = [
    "SELECT DISTINCT s FROM t",
    "SELECT DISTINCT s, k FROM t",
    "SELECT DISTINCT s AS b FROM t ORDER BY b",
    "SELECT DISTINCT s, k FROM t ORDER BY s, k LIMIT 5",
    "SELECT DISTINCT k FROM t ORDER BY k DESC LIMIT 3",
    "SELECT DISTINCT x FROM t",
    "SELECT DISTINCT k + 1 AS k1 FROM t WHERE v > 50",
    "SELECT DISTINCT name FROM t JOIN u ON k = kid",
    "SELECT DISTINCT name, s FROM t JOIN u ON k = kid ORDER BY name, s",
    "SELECT DISTINCT kid FROM t JOIN u ON k = kid ORDER BY kid LIMIT 4",
]

ORDINALS = [
    "SELECT s, SUM(v) AS x FROM t GROUP BY 1",
    "SELECT SUM(v) AS x, s FROM t GROUP BY 2",
    "SELECT s, k, SUM(v) AS x FROM t GROUP BY 1, 2 ORDER BY 1, 2",
    "SELECT s, SUM(v) AS x FROM t GROUP BY 1 ORDER BY 2 DESC",
    "SELECT s, SUM(v) AS x FROM t GROUP BY s ORDER BY 2 DESC, 1",
    "SELECT s, SUM(v) FROM t GROUP BY 1 ORDER BY 2",
    "SELECT k + 1 AS k1, SUM(v) AS x FROM t GROUP BY 1 ORDER BY 1",
    "SELECT s, SUM(v) * 2 AS y FROM t GROUP BY 1 ORDER BY 2 DESC",
    "SELECT s, k FROM t ORDER BY 2 DESC, 1 LIMIT 5",
    "SELECT s AS b, v FROM t WHERE v > 95 ORDER BY 2, 1",
    "SELECT DISTINCT s, k FROM t ORDER BY 2, 1",
    "SELECT s, COUNT(DISTINCT k) AS n FROM t GROUP BY 1 ORDER BY 2 DESC, 1",
]

RENAMED = [
    "SELECT s, SUM(v) AS s FROM t GROUP BY s",
    "SELECT x, SUM(v) AS x FROM t GROUP BY x",
    "SELECT s, SUM(v), s FROM t GROUP BY s",
    "SELECT s, SUM(v) AS s_1, SUM(k) AS s FROM t GROUP BY s",
    "SELECT SUM(v) AS x_1, SUM(v) AS x, SUM(k) AS x FROM t",
    "SELECT SUM(v), SUM(v) FROM t",
    "SELECT s, s FROM t",
    "SELECT k, v AS k FROM t",
    "SELECT t.k, u.kid, t.k FROM t JOIN u ON t.k = u.kid",
    "SELECT t.k, w.k FROM t JOIN w ON t.k = w.k",
    "SELECT t.v FROM t JOIN w ON t.k = w.k WHERE t.v > 95",
    "SELECT t.s, SUM(t.v) AS x FROM t JOIN w ON t.k = w.k GROUP BY t.s",
    "SELECT kid, name FROM t JOIN u ON k = kid ORDER BY kid, name LIMIT 5",
]


NULLS = [
    "SELECT g, SUM(v) AS s FROM tn GROUP BY g HAVING COUNT(*) > 1",
    "SELECT COUNT(*) AS c, COUNT(x) AS cx, COUNT(v) AS cv FROM tn",
    "SELECT COUNT(DISTINCT x) AS n, COUNT(*) AS c FROM tn",
    "SELECT g, COUNT(*) AS c, COUNT(x) AS cx, COUNT(v) AS cv FROM tn GROUP BY g",
    "SELECT g, COUNT(DISTINCT x) AS n, COUNT(*) AS c FROM tn GROUP BY g",
    "SELECT g, COUNT(DISTINCT x) AS n, COUNT(*) AS c, COUNT(v) AS cv, SUM(v) AS s, AVG(v) AS a, MIN(v) AS lo, MAX(v) AS hi FROM tn GROUP BY g",
    "SELECT COUNT(DISTINCT g) AS n, SUM(v) AS s, AVG(v) AS a, COUNT(*) AS c FROM tn",
    "SELECT g, AVG(v) AS a, SUM(v) AS s FROM tn GROUP BY g",
    "SELECT x, COUNT(DISTINCT g) AS n, COUNT(*) AS c, AVG(v) AS a FROM tn GROUP BY x",
    "SELECT g, x FROM tn ORDER BY g, x",
    "SELECT g, x FROM tn ORDER BY g DESC, x DESC",
    "SELECT g, x FROM tn ORDER BY g NULLS LAST, x NULLS LAST",
    "SELECT g, COUNT(*) AS c FROM tn GROUP BY g ORDER BY g DESC NULLS LAST",
]


@pytest.mark.parametrize("q", NULLS)
def test_nulls_match_duckdb(q):
    _check(q)


def test_count_star_counts_rows_in_the_dataframe_api():
    from mxframe import LazyFrame, col, lit
    got = LazyFrame(_n()).groupby("g").agg(lit(1).count().alias("c"), col("x").count().alias("cx")) \
        .compute(device="cpu").to_pydict()
    assert sorted(zip(got["g"], got["c"], got["cx"]), key=repr) == \
        sorted([(None, 2, 0), ("a", 3, 3), ("b", 3, 3), ("c", 2, 1)], key=repr)


@pytest.mark.parametrize("q", MIXED_DISTINCT + DISTINCT + ORDINALS + RENAMED)
def test_matches_duckdb_round2(q):
    _check(q)


@pytest.mark.parametrize("q", NAMING + JOIN_KEYS + GROUP_ALL + HAVING + COUNT_DISTINCT + ARITHMETIC)
def test_matches_duckdb(q):
    _check(q)


def test_having_is_a_filter_above_the_aggregate():
    """HAVING is never dropped: the plan has a Filter over the Aggregate."""
    from mxframe.lazy_frame import Filter, Aggregate
    lf = mx.sql("SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING SUM(v) > 125000", t=_t())
    p = lf.plan
    while not isinstance(p, Filter):
        p = p.input
    assert isinstance(p.input, Aggregate)


def test_count_distinct_reaches_the_native_count(monkeypatch):
    import mxframe.native_exec as nx
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    rcs = []
    real = nx._call_count_distinct

    def counting(*a):
        rc = real(*a)
        rcs.append(rc)
        return rc

    monkeypatch.setattr(nx, "_call_count_distinct", counting)
    got, exp = _run("SELECT k, COUNT(DISTINCT v) AS n FROM t GROUP BY k")
    assert sorted(zip(got.column("k").to_pylist(), got.column("n").to_pylist())) == \
        sorted(zip(exp.column("k").to_pylist(), exp.column("n").to_pylist()))
    assert rcs == [0]


@pytest.mark.parametrize("q, exc, msg", [
    ("SELECT s, SUM(v) * 2 FROM t GROUP BY s", ValueError, "alias"),
    ("SELECT s, SUM(v + 1) FROM t GROUP BY s", ValueError, "alias"),
    ("SELECT s, k, SUM(v) AS x FROM t GROUP BY s", ValueError, "GROUP BY"),
    ("SELECT s, SUM(v) AS x FROM t HAVING SUM(v) > 1", ValueError, "GROUP BY"),
    ("SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING k > 1", ValueError, "GROUP BY"),
    ("SELECT s FROM t HAVING s = 'a'", ValueError, "HAVING"),
    ("SELECT s FROM t GROUP BY s", ValueError, "aggregate"),
    ("SELECT s FROM t GROUP BY ALL", ValueError, "aggregate"),
    ("SELECT SUM(DISTINCT v) AS x FROM t", NotImplementedError, "DISTINCT"),
    ("SELECT AVG(DISTINCT v) AS x FROM t", NotImplementedError, "DISTINCT"),
    ("SELECT COUNT(DISTINCT s, k) AS n FROM t", NotImplementedError, "DISTINCT"),
    ("SELECT COUNT(DISTINCT k) AS n, COUNT(DISTINCT x) AS m FROM t", NotImplementedError, "COUNT\\(DISTINCT"),
    ("SELECT s, STDDEV(v) AS d FROM t GROUP BY s", NotImplementedError, "STDDEV"),
    ("SELECT s, SUM(v) AS x FROM t GROUP BY s HAVING STDDEV(v) > 1", NotImplementedError, "STDDEV"),
    ("SELECT s FROM t WHERE SUM(v) > 1", ValueError, "WHERE"),
    ("SELECT s, SUM(v) AS x FROM t WHERE COUNT(*) > 1 GROUP BY s", ValueError, "WHERE"),
    ("SELECT DISTINCT ON (s) s, k FROM t", NotImplementedError, "DISTINCT ON"),
    ("SELECT DISTINCT s, SUM(v) AS x FROM t GROUP BY s", NotImplementedError, "DISTINCT"),
    ("SELECT s, k FROM t ORDER BY 3", ValueError, "position"),
    ("SELECT s, SUM(v) AS x FROM t GROUP BY 2", ValueError, "aggregate"),
    ("SELECT t.v, w.v FROM t JOIN w ON t.k = w.k", ValueError, "more than one joined table"),
    ("SELECT w.v FROM t JOIN w ON t.k = w.k", ValueError, "more than one joined table"),
    ("SELECT g FROM tn ORDER BY g NULLS FIRST", NotImplementedError, "NULLS FIRST"),
    ("SELECT g, COUNT(*) AS c FROM tn GROUP BY g ORDER BY g DESC NULLS FIRST", NotImplementedError, "NULLS FIRST"),
])
def test_unsupported_raises(q, exc, msg):
    with pytest.raises(exc, match=msg):
        mx.sql(q, t=_t(), u=_u(), w=_w(), tn=_n())
