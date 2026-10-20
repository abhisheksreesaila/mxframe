"""F-114: SQL ``IN`` / ``NOT IN`` lists holding a NULL literal.

Before: ``x NOT IN (1, 2, NULL)`` and ``x IN (1, NULL)`` raised "Expected scalar literal in IN list, got Null".

DuckDB's rule (probed): ``x IN (.., NULL)`` is TRUE where x matches a non-NULL value, else NULL, never FALSE;
``x NOT IN (.., NULL)`` is FALSE where x matches, else NULL, never TRUE; a NULL x gives NULL. So in WHERE the NOT IN
keeps nothing, and the difference shows in SELECT, under NOT, and with OR / AND / CASE / IS NULL above it.

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off.
"""
import datetime as dt

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import gpu_stream

GPU = gpu_stream.cuda_available()
NAN = float("nan")
N = 12


def _table():
    return pa.table({
        "id": pa.array(range(N), pa.int64()),
        "i": pa.array([None if k % 4 == 3 else k % 5 for k in range(N)], pa.int64()),            # NULL on 3, 7, 11
        "j": pa.array([k % 5 for k in range(N)], pa.int32()),                                    # no NULLs
        "f": pa.array([None if k % 5 == 1 else NAN if k == 4 else k % 3 * 0.5 for k in range(N)], pa.float64()),
        "r": pa.array([None if k % 4 == 2 else k % 3 * 0.5 for k in range(N)], pa.float32()),
        "s": pa.array(["a", "b", None, "a", "c", "b", "", None, "c", "a", "b", "d"]),
        "ds": pa.array(["a", None, "c", "a", None, "b", "c", "a", "b", "c", "a", "b"]).dictionary_encode(),
        "d": pa.array([None if k % 5 == 2 else dt.date(2024, 1, 1) + dt.timedelta(days=k % 4) for k in range(N)],
                      pa.date32()),
        "g": pa.array([k % 3 for k in range(N)], pa.int32()),
    })


T = _table()
CON = duckdb.connect()
CON.register("_t", T)
CON.execute("create table t as select * from _t")


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev


def _rows(x):
    cols = [x.column(i).to_pylist() for i in range(x.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(x.num_rows)]


def _same(a, b):
    return len(a) == len(b) and all(
        len(x) == len(y) and all(p == q or (p != p and q != q) for p, q in zip(x, y)) for x, y in zip(a, b))


QUERIES = {
    # SELECT shows the three values
    "int_in_null": "select id, i in (1, 2, null) as c from t order by id",
    "int_not_in_null": "select id, i not in (1, 2, null) as c from t order by id",
    "int_in_only_null": "select id, i in (null) as c from t order by id",
    "int_not_in_only_null": "select id, i not in (null) as c from t order by id",
    "null_first": "select id, i not in (null, 1, 2) as c from t order by id",
    "two_nulls": "select id, i in (1, null, null) as c from t order by id",
    "int32_not_in_null": "select id, j not in (0, 3, null) as c from t order by id",
    "float_in_null": "select id, f in (0.5, null) as c from t order by id",
    "float_not_in_null": "select id, f not in (0.0, 1.0, null) as c from t order by id",
    "float_nan": "select id, f in (cast('nan' as double), null) as c from t order by id",
    "float32_not_in_null": "select id, r not in (0.5, null) as c from t order by id",
    "string_in_null": "select id, s in ('a', 'b', null) as c from t order by id",
    "string_not_in_null": "select id, s not in ('a', 'b', null) as c from t order by id",
    "empty_string": "select id, s in ('', null) as c from t order by id",
    "dictionary_in_null": "select id, ds in ('a', null) as c from t order by id",
    "dictionary_not_in_null": "select id, ds not in ('a', 'c', null) as c from t order by id",
    # WHERE: IN keeps the matches, NOT IN keeps nothing
    "where_in": "select id from t where i in (1, 2, null) order by id",
    "where_not_in": "select id from t where i not in (1, 2, null) order by id",
    "where_not_in_no_match": "select id from t where j not in (9, null) order by id",
    "where_string_in": "select id from t where s in ('a', null) order by id",
    "where_string_not_in": "select id from t where s not in ('a', null) order by id",
    "where_dictionary_in": "select id from t where ds in ('a', null) order by id",
    "where_dictionary_not_in": "select id from t where ds not in ('a', null) order by id",
    # NOT / OR / AND above the list: where NULL and FALSE differ
    "not_in": "select id from t where not (i in (1, 2, null)) order by id",
    "not_not_in": "select id from t where not (i not in (1, 2, null)) order by id",
    "or_other": "select id from t where i not in (1, null) or g = 1 order by id",
    "and_other": "select id from t where i in (1, 2, null) and g = 1 order by id",
    "or_in": "select id from t where i in (1, null) or j in (2, null) order by id",
    "not_or": "select id from t where not (i in (1, null) or g = 1) order by id",
    "is_null_of_in": "select id, (i in (1, 2, null)) is null as c from t order by id",
    "is_not_null_of_not_in": "select id from t where (j not in (0, null)) is not null order by id",
    "case_on_in": "select id, case when i in (1, null) then 'y' else 'n' end as c from t order by id",
    "case_on_not_in": "select id, case when j not in (1, null) then 'y' else 'n' end as c from t order by id",
    "coalesce_of_in": "select id, coalesce(i in (1, null), false) as c from t order by id",
    # aggregates over them
    "count_where_in": "select count(*) as c from t where i in (1, null)",
    "count_where_not_in": "select count(*) as c from t where j not in (1, null)",
    "group_where_in": "select g, count(*) as c from t where j in (1, 2, null) group by g order by g",
    "sum_case_in": "select g, sum(case when j in (1, null) then 1 else 0 end) as c from t group by g order by g",
}


@pytest.mark.parametrize("name", list(QUERIES))
def test_matches_duckdb(cfg, name):
    q = QUERIES[name]
    want = _rows(CON.execute(q).to_arrow_table())
    got = _rows(mx.sql(q, t=T).compute(device=cfg))
    assert _same(got, want), (got, want)


def test_three_values():
    got = _rows(mx.sql("select id, j not in (1, 2, null) as c from t order by id", t=T).compute())
    assert got == [(0, None), (1, False), (2, False), (3, None), (4, None),
                   (5, None), (6, False), (7, False), (8, None), (9, None), (10, None), (11, False)]
    got = _rows(mx.sql("select id, j in (1, 2, null) as c from t order by id", t=T).compute())
    assert got == [(0, None), (1, True), (2, True), (3, None), (4, None),
                   (5, None), (6, True), (7, True), (8, None), (9, None), (10, None), (11, True)]


def test_native_executor_does_not_drop_the_null(monkeypatch):
    """Group-by over a filter / CASE on a NULL list: the native executor must decline or follow DuckDB, not ignore the NULL."""
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    for q in ("select count(*) as c from t where j not in (1, 2, null)",
              "select g, sum(case when j in (1, null) then 1 else 0 end) as c from t group by g order by g",
              "select g, count(*) as c from t where not (j in (1, null)) group by g order by g"):
        assert _rows(mx.sql(q, t=T).compute()) == _rows(CON.execute(q).to_arrow_table()), q


def test_scan_parquet_pushdown_keeps_three_values(tmp_path):
    """Statistics / row filters pushed into a file scan must not drop the NULL: ``i NOT IN (.., NULL)`` keeps nothing."""
    path = str(tmp_path / "t.parquet")
    pq.write_table(T, path, row_group_size=4)
    for q in ("select id from t where i in (1, 2, null) order by id",
              "select id from t where i not in (1, 2, null) order by id",
              "select id from t where not (j in (1, null)) order by id",
              "select id from t where s in ('a', null) order by id",
              "select id from t where s not in ('a', null) order by id",
              "select id from t where j in (1, null) or j in (2, null) order by id"):
        want = _rows(CON.execute(q).to_arrow_table())
        assert _rows(mx.sql(q, t=mx.scan_parquet(path)).compute()) == want, q
