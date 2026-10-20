"""F-046f / F-046g: in a single-table query (WHERE, a selection of plain columns, maybe DISTINCT),
dictionary string columns are no longer decoded for the whole table first. The native filter
decodes only the rows that pass; under a DISTINCT it keeps their codes, the DISTINCT runs on the
codes and only the distinct rows are decoded. Results and types as before (strings), checked
against DuckDB; a filter that reads the column as plain values still decodes first."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import LazyFrame, col, lit
from omxcore.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


def _table(n=400_000, seed=3, chunks=4):
    r = np.random.default_rng(seed)
    pri = np.array(["1-URGENT", "2-HIGH", "3-MEDIUM", "4-NOT SPECIFIED", "5-LOW"])[r.integers(0, 5, n)]
    st = np.array(["F", "O", "P"])[r.integers(0, 3, n)]
    t = pa.table({"pri": pa.array(pri.tolist()), "st": pa.array(st.tolist()),
                  "x": A(r.random(n)), "k": A(r.integers(0, 50, n).astype(np.int32))})
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    parts = [t.slice(a, b - a) for a, b in zip(cuts[:-1], cuts[1:])]
    enc = lambda p: pa.table({"pri": p.column("pri").combine_chunks().dictionary_encode(),
                              "st": p.column("st").combine_chunks().dictionary_encode(),
                              "x": p.column("x"), "k": p.column("k")})
    return pa.concat_tables([enc(p) for p in parts]), t


@pytest.fixture
def decodes(monkeypatch):
    import omxcore.arrow_dict as ad
    log = []
    real = ad.decode
    monkeypatch.setattr(ad, "decode", lambda arr: (log.append(len(arr)), real(arr))[1])
    return log


def _duck(sql, t):
    con = duckdb.connect()
    con.register("t", t)
    return sorted(con.execute(sql).fetchall())


def test_distinct_of_two_dictionary_columns(decodes):
    d, plain = _table()
    got = (LazyFrame(d).filter(col("pri") == lit("4-NOT SPECIFIED")).select("pri", "st").distinct()
           .compute(device="cpu"))
    assert got.schema.types == [pa.string(), pa.string()]
    assert sorted(zip(*got.to_pydict().values())) == _duck(
        "SELECT DISTINCT pri, st FROM t WHERE pri = '4-NOT SPECIFIED'", plain)
    assert max(decodes, default=0) < 100                          # only the distinct rows were decoded


def test_filter_and_select_decodes_only_the_rows_that_pass(decodes):
    d, plain = _table()
    got = LazyFrame(d).filter(col("x") < lit(0.01)).select("st", "k").compute(device="cpu")
    assert got.schema.field("st").type == pa.string()
    assert sorted(zip(*got.to_pydict().values())) == _duck("SELECT st, k FROM t WHERE x < 0.01", plain)
    assert max(decodes, default=0) < 50_000


def test_distinct_order_is_first_appearance(decodes):
    d, plain = _table()
    got = LazyFrame(d).filter(col("k") < lit(10)).select("st", "pri").distinct().compute(device="cpu")
    seen, exp = set(), []
    m = plain.column("k").to_numpy() < 10
    for s, p in zip(np.array(plain.column("st").to_pylist())[m], np.array(plain.column("pri").to_pylist())[m]):
        if (s, p) not in seen:
            seen.add((s, p))
            exp.append((s, p))
    assert list(zip(*got.to_pydict().values())) == exp


def test_a_value_read_of_the_column_decodes_first():
    d, plain = _table()
    from omxcore import when
    got = (LazyFrame(d).filter(when(col("st") == lit("F"), lit(1), lit(0)) == lit(1)).select("st", "k").distinct()
           .compute(device="cpu"))
    assert sorted(zip(*got.to_pydict().values())) == _duck("SELECT DISTINCT st, k FROM t WHERE st = 'F'", plain)


def test_from_parquet(tmp_path, decodes):
    import pyarrow.parquet as pq
    d, plain = _table()
    pq.write_table(plain, tmp_path / "t.parquet", row_group_size=100_000)
    got = (mx.scan_parquet(str(tmp_path / "t.parquet")).filter(col("x") > lit(0.9)).select("pri", "st").distinct()
           .compute(device="cpu"))
    assert sorted(zip(*got.to_pydict().values())) == _duck("SELECT DISTINCT pri, st FROM t WHERE x > 0.9", plain)
    assert got.schema.types == [pa.string(), pa.string()]


def test_other_routes_decode(decodes):
    d, plain = _table()
    # a renamed deferred column, with Sort / Limit / Tail after it
    got = LazyFrame(d).filter(col("x") < lit(0.001)).select(col("st").alias("s2"), col("k")).sort("s2").limit(7).compute(device="cpu")
    assert got.schema.field("s2").type == pa.string() and got.num_rows == 7
    got = LazyFrame(d).filter(col("x") < lit(0.001)).select("pri", "k").tail(3).compute(device="cpu")
    assert got.schema.field("pri").type == pa.string() and got.num_rows == 3
    # empty results, plain and DISTINCT
    for lf in (LazyFrame(d).filter(col("x") < lit(-1.0)).select("pri"),
               LazyFrame(d).filter(col("x") < lit(-1.0)).select("pri").distinct()):
        got = lf.compute(device="cpu")
        assert got.num_rows == 0 and got.schema.field("pri").type == pa.string()
    # DISTINCT that keeps every row
    got = LazyFrame(d).filter(col("x") < lit(0.0005)).select("pri", "st", "x").distinct().compute(device="cpu")
    assert got.num_rows == int((plain.column("x").to_numpy() < 0.0005).sum())
    assert got.schema.field("st").type == pa.string()


def test_metadata_kept_without_a_select():
    d, plain = _table()
    d = d.replace_schema_metadata({b"m": b"1"})
    got = LazyFrame(d).filter(col("x") < lit(0.01)).compute(device="cpu")
    assert got.schema.metadata == {b"m": b"1"}
    assert got.schema.field("pri").type == pa.string()


def test_non_cpu_device_takes_the_old_route():
    d, plain = _table(n=50_000)
    got = LazyFrame(d).filter(col("x") < lit(0.1)).select("pri", "st").distinct().compute(device="auto")
    assert sorted(zip(*got.to_pydict().values())) == _duck("SELECT DISTINCT pri, st FROM t WHERE x < 0.1", plain)
    assert got.schema.types == [pa.string(), pa.string()]
