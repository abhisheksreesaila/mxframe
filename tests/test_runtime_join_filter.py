"""F-043: a semi-join partner much larger than the filtered fact table is filtered by the fact's
keys first (a runtime join filter), so the partner never materializes rows the fact cannot use.
TPC-H Q4: the late lineitems of the orders in range, not all 38M late lineitems. Same answers as
DuckDB for one- and two-column keys, unsorted keys and repeated fact keys."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
import mxframe.join_engine as je
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")


@pytest.fixture
def reduced(monkeypatch):
    """Partner filters that ran through the fact's keys (a semi lookup) and succeeded."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    log = []
    real = nx.try_filter

    def spy(comp, table, preds, needed, lookups=(), **kw):
        out = real(comp, table, preds, needed, lookups, **kw)
        if "pk" in table.column_names and lookups and out is not None:     # the partner, by the fact's keys
            log.append(1)
        return out

    monkeypatch.setattr(nx, "try_filter", spy)
    return log


def _tables(seed=3, fact_n=40_000, part_n=400_000, dup=False):
    r = np.random.default_rng(seed)
    fk = r.permutation(fact_n * 3)[:fact_n].astype(np.int64)
    if dup:
        fk = r.integers(0, fact_n, fact_n).astype(np.int64)
    fact = pa.table({"k": A(fk), "k2": A((fk % 5).astype(np.int32)), "d": A(r.integers(0, 1000, fact_n).astype(np.int32)),
                     "g": A(r.integers(0, 4, fact_n).astype(np.int32))})
    pk = r.integers(0, fact_n * 3, part_n).astype(np.int64)
    part = pa.table({"pk": A(pk), "pk2": A((pk % 5).astype(np.int32)), "a": A(r.integers(0, 50, part_n).astype(np.int32)),
                     "b": A(r.integers(0, 50, part_n).astype(np.int32))})
    return fact, part


def _duck(sql, **t):
    con = duckdb.connect()
    for k, v in t.items():
        con.register(k, v)
    return sorted(con.execute(sql).fetchall())


@pytest.mark.parametrize("dup", [False, True])
def test_one_key(reduced, dup):
    fact, part = _tables(dup=dup)
    sub = LazyFrame(part).filter(col("a") < col("b")).select("pk").distinct()
    got = (LazyFrame(fact).filter(col("d") < lit(100)).join(sub, left_on="k", right_on="pk")
           .groupby("g").agg(col("g").count().alias("n")).compute(device="cpu"))
    exp = _duck("SELECT g, COUNT(*) FROM fact WHERE d < 100 AND k IN (SELECT pk FROM part WHERE a < b) GROUP BY g",
                fact=fact, part=part)
    assert sorted(zip(got.column("g").to_pylist(), got.column("n").to_pylist())) == exp
    assert reduced == [1]


def test_two_keys(reduced):
    fact, part = _tables(seed=5)
    sub = LazyFrame(part).filter(col("a") > lit(10)).select("pk", "pk2").distinct()
    got = (LazyFrame(fact).filter(col("d") >= lit(900)).join(sub, left_on=["k", "k2"], right_on=["pk", "pk2"])
           .groupby("g").agg(col("g").count().alias("n")).compute(device="cpu"))
    exp = _duck("SELECT g, COUNT(*) FROM fact WHERE d >= 900 AND (k, k2) IN (SELECT pk, pk2 FROM part WHERE a > 10) "
                "GROUP BY g", fact=fact, part=part)
    assert sorted(zip(got.column("g").to_pylist(), got.column("n").to_pylist())) == exp
    assert reduced == [1]


def _check(fact, part, mx_fact, duck_where):
    sub = LazyFrame(part).filter(col("a") < col("b")).select("pk").distinct()
    got = (mx_fact(LazyFrame(fact)).join(sub, left_on="k", right_on="pk")
           .groupby("g").agg(col("g").count().alias("n")).compute(device="cpu"))
    exp = _duck(f"SELECT g, COUNT(*) FROM fact WHERE {duck_where} AND k IN (SELECT pk FROM part WHERE a < b) "
                "GROUP BY g", fact=fact, part=part)
    assert sorted(zip(got.column("g").to_pylist(), got.column("n").to_pylist())) == exp


def test_left_alone_when_it_cannot_pay(reduced):
    # the partner is not much larger than the fact
    fact, part = _tables(fact_n=40_000, part_n=50_000)
    _check(fact, part, lambda f: f.filter(col("d") < lit(100)), "d < 100")
    # the fact is not filtered
    fact, part = _tables()
    _check(fact, part, lambda f: f, "TRUE")
    # the fact's filter keeps most rows
    _check(fact, part, lambda f: f.filter(col("d") < lit(900)), "d < 900")
    assert reduced == []


def test_int32_fact_keys_against_int64_partner_keys(reduced):
    fact, part = _tables()
    fact = fact.set_column(0, "k", A(fact.column("k").to_numpy().astype(np.int32)))
    _check(fact, part, lambda f: f.filter(col("d") < lit(100)), "d < 100")
    assert reduced == [1]


def test_fact_filter_matches_nothing(reduced):
    fact, part = _tables()
    _check(fact, part, lambda f: f.filter(col("d") < lit(0)), "d < 0")


@pytest.mark.parametrize("side", ["fact", "partner"])
def test_null_keys(reduced, side):
    fact, part = _tables()
    if side == "fact":
        k = fact.column("k").to_pylist()
        fact = fact.set_column(0, "k", pa.array([None if i % 7 == 0 else v for i, v in enumerate(k)], pa.int64()))
    else:
        k = part.column("pk").to_pylist()
        part = part.set_column(0, "pk", pa.array([None if i % 7 == 0 else v for i, v in enumerate(k)], pa.int64()))
    _check(fact, part, lambda f: f.filter(col("d") < lit(100)), "d < 100")
