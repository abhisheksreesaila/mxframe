"""F-072: MIN / MAX of a string column (dictionary or plain), ordered by bytes like DuckDB.

A dictionary column's distinct values are ranked once in byte order; the executor takes min / max of the
integer ranks (native) and the ranks are turned back into strings. Every answer is compared with DuckDB;
each shape asserts the path that ran (native executor, fused join, or Arrow's null semantics)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import LazyFrame, col
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

WORDS = ["b", "a", "é", "zz", "", "Z", "ab", "ÿ", "日本", "á", "~", "B"]   # '' sorts first; 'é' < 'ÿ' < '日本' in bytes; 'Z' < 'a'


def _dict_col(values, n_chunks, seed=0):
    """A dictionary column cut into chunks that each carry their own dictionary, in their own order."""
    r = np.random.default_rng(seed)
    cuts = np.linspace(0, len(values), n_chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vals = list(values[a:b])
        d = sorted({v for v in vals if v is not None}, key=lambda _: r.random())
        idx = {v: i for i, v in enumerate(d)}
        ind = pa.array([None if v is None else idx[v] for v in vals], pa.int32())
        parts.append(pa.DictionaryArray.from_arrays(ind, pa.array(d)))
    return pa.chunked_array(parts)


def _table(n=120_000, seed=5, chunks=4, kind="dict", nulls=False, groups=40):
    r = np.random.default_rng(seed)
    s = [WORDS[i] for i in r.integers(0, len(WORDS), n)]
    k = r.integers(0, groups, n).astype(np.int32)
    if nulls:
        for i in np.flatnonzero(k == 3):          # group 3 holds only NULL strings
            s[i] = None
        for i in r.integers(0, n, n // 20):
            s[i] = None
    scol = _dict_col(s, chunks, seed) if kind == "dict" else pa.chunked_array([pa.array(s)])
    return pa.table({"k": A(k), "s": scol, "v": A(r.random(n))})


def _duck(sql, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, pa.table({n: (t.column(n).cast(pa.string()) if pa.types.is_dictionary(t.column(n).type)
                                         else t.column(n)) for n in t.column_names}))
    return sorted(tuple(r) for r in con.execute(sql).fetchall())


def _run(lf):
    """(result, provenance of the run): which path answered."""
    return lf.compute(device="cpu"), lf.last_compile_provenance


def _rows(tbl):
    return sorted(zip(*[tbl.column(n).to_pylist() for n in tbl.column_names]))


@pytest.mark.parametrize("kind", ["dict", "plain"])
def test_grouped_min_max_native(kind):
    t = _table(kind=kind)
    got, prov = _run(mx.sql("SELECT k, MIN(s) lo, MAX(s) hi FROM t GROUP BY k", t=t))
    assert got.schema.field("lo").type == pa.string() and got.schema.field("hi").type == pa.string()
    assert _rows(got) == _duck("SELECT k, MIN(s), MAX(s) FROM t GROUP BY k", t=t)
    assert prov["path"] == "native"


def test_sql_select_k_max_s_group_by_k():
    t = _table()
    got, prov = _run(mx.sql("SELECT k, MAX(s) FROM t GROUP BY k", t=t))
    assert _rows(got) == _duck("SELECT k, MAX(s) FROM t GROUP BY k", t=t)
    assert prov["path"] == "native"


@pytest.mark.parametrize("kind", ["dict", "plain"])
def test_global_min_max_native(kind):
    t = _table(kind=kind)
    got, prov = _run(mx.sql("SELECT MIN(s) lo, MAX(s) hi FROM t", t=t))
    assert _rows(got) == _duck("SELECT MIN(s), MAX(s) FROM t", t=t) == [("", "日本")]
    assert prov["path"] == "native"


def test_dataframe_api_with_other_aggregates_and_filter():
    t = _table()
    got, prov = _run(LazyFrame(t).filter(col("v") < 0.5).groupby("k")
                     .agg(col("s").max().alias("hi"), col("v").sum().alias("sv"), col("v").count().alias("n")))
    exp = _duck("SELECT k, MAX(s), SUM(v), COUNT(v) FROM t WHERE v < 0.5 GROUP BY k", t=t)
    g = _rows(got)
    assert [(r[0], r[1], r[3]) for r in g] == [(r[0], r[1], r[3]) for r in exp]
    assert all(abs(a[2] - b[2]) < 1e-6 * max(1, abs(b[2])) for a, b in zip(g, exp))
    assert prov["path"] == "native"


def test_nulls_are_ignored_and_an_all_null_group_is_null():
    t = _table(nulls=True)
    got, prov = _run(mx.sql("SELECT k, MIN(s) lo, MAX(s) hi FROM t GROUP BY k", t=t))
    assert {r[0]: r[1:] for r in _rows(got)}[3] == (None, None)
    assert _rows(got) == _duck("SELECT k, MIN(s), MAX(s) FROM t GROUP BY k", t=t)
    assert prov["path"] == "arrow_null_semantics"          # the executor declines NULLs; Arrow's hash aggregate answers
    glob, _ = _run(mx.sql("SELECT MIN(s) lo, MAX(s) hi FROM t", t=t))
    assert _rows(glob) == _duck("SELECT MIN(s), MAX(s) FROM t", t=t) == [("", "日本")]


def test_all_null_column_gives_null():
    t = pa.table({"k": A(np.array([1, 1, 2], np.int32)),
                  "s": pa.array([None, None, None], pa.string()).dictionary_encode()})
    got = mx.sql("SELECT k, MAX(s) m FROM t GROUP BY k", t=t).compute(device="cpu")
    assert _rows(got) == [(1, None), (2, None)]
    assert mx.sql("SELECT MIN(s) m FROM t", t=t).compute(device="cpu").column("m").to_pylist() == [None]


def test_where_that_removes_every_row():
    t = _table(n=1000)
    got = mx.sql("SELECT MAX(s) m FROM t WHERE v > 5", t=t).compute(device="cpu")
    assert got.column("m").to_pylist() == [None]
    got = mx.sql("SELECT k, MAX(s) m FROM t WHERE v > 5 GROUP BY k", t=t).compute(device="cpu")
    assert got.num_rows == 0


def test_having_and_order_by_on_the_string_aggregate():
    t = _table()
    got = mx.sql("SELECT k, MAX(s) m FROM t GROUP BY k HAVING MAX(s) > 'zz' ORDER BY k LIMIT 5", t=t).compute(device="cpu")
    exp = _duck("SELECT k, MAX(s) FROM t GROUP BY k HAVING MAX(s) > 'zz' ORDER BY k LIMIT 5", t=t)
    assert _rows(got) == exp and len(exp) == 5
    got = mx.sql("SELECT k, MIN(s) m FROM t GROUP BY k ORDER BY m DESC, k LIMIT 3", t=t).compute(device="cpu")
    assert [tuple(r) for r in zip(*[got.column(n).to_pylist() for n in got.column_names])] == \
        [tuple(r) for r in duckdb.connect().register("t", pa.table({"k": t.column("k"), "s": t.column("s").cast(pa.string())})
                                                     ).execute("SELECT k, MIN(s) m FROM t GROUP BY k ORDER BY m DESC, k LIMIT 3").fetchall()]


def _join_tables(chunks=3, n=100_000):
    r = np.random.default_rng(11)
    fact = pa.table({"pk": A(r.integers(0, 300, n).astype(np.int64)),
                     "g": A(r.integers(0, 9, n).astype(np.int32)), "v": A(r.random(n))})
    names = [WORDS[i] for i in r.integers(0, len(WORDS), 300)]
    part = pa.table({"pk": A(np.arange(300, dtype=np.int64)), "s": _dict_col(names, chunks, 2),
                     "t": pa.chunked_array([pa.array(names)])})
    return fact, part


@pytest.mark.parametrize("scol", ["s", "t"])
def test_partner_column_through_a_fused_join(scol):
    fact, part = _join_tables()
    got, prov = _run(LazyFrame(fact).join(LazyFrame(part), on="pk").groupby("g")
                     .agg(col(scol).max().alias("hi"), col(scol).min().alias("lo")))
    exp = _duck(f"SELECT g, MAX({scol}), MIN({scol}) FROM fact JOIN part USING (pk) GROUP BY g", fact=fact, part=part)
    assert _rows(got) == exp
    assert prov["path"] == "native" and prov["fused_join"] is True


def test_partner_column_global_and_sql_join():
    fact, part = _join_tables()
    got, prov = _run(mx.sql("SELECT g, MAX(s) m FROM fact JOIN part ON fact.pk = part.pk GROUP BY g", fact=fact, part=part))
    assert prov["fused_join"] is True
    assert _rows(got) == _duck("SELECT g, MAX(s) FROM fact JOIN part ON fact.pk = part.pk GROUP BY g", fact=fact, part=part)
    got, prov = _run(mx.sql("SELECT MIN(s) m FROM fact JOIN part ON fact.pk = part.pk", fact=fact, part=part))
    assert prov["path"] == "native"
    assert _rows(got) == _duck("SELECT MIN(s) FROM fact JOIN part ON fact.pk = part.pk", fact=fact, part=part)


def test_clashing_name_after_a_join_is_found_by_its_renamed_form():
    # both sides have `s`: the right copy comes out as s_right
    fact, part = _join_tables()
    fact = fact.append_column("s", _dict_col([WORDS[i % len(WORDS)] for i in range(fact.num_rows)], 2))
    got, prov = _run(LazyFrame(fact).join(LazyFrame(part), on="pk").groupby("g")
                     .agg(col("s_right").max().alias("hi"), col("s").min().alias("lo")))
    exp = _duck("SELECT g, MAX(part.s), MIN(fact.s) FROM fact JOIN part USING (pk) GROUP BY g", fact=fact, part=part)
    assert _rows(got) == exp
    assert prov["path"] == "native"             # materialized join, then the executor on the ranks


def test_numeric_min_max_unchanged():
    t = _table()
    got = mx.sql("SELECT k, MIN(v) lo, MAX(v) hi FROM t GROUP BY k", t=t).compute(device="cpu")
    assert _rows(got) == _duck("SELECT k, MIN(v), MAX(v) FROM t GROUP BY k", t=t)
    assert got.schema.field("lo").type == pa.float64()


def test_parquet_scan(tmp_path):
    import pyarrow.parquet as pq
    t = _table()
    p = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({"k": t.column("k"), "s": t.column("s").cast(pa.string()), "v": t.column("v")}),
                   p, row_group_size=30_000)
    got = mx.scan_parquet(p).groupby("k").agg(col("s").max().alias("m")).compute(device="cpu")
    assert _rows(got) == _duck("SELECT k, MAX(s) FROM t GROUP BY k", t=t)
