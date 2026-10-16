"""F-046a slice 3: join + aggregate fusion (`mx_execute` version 3).

`Aggregate([Filter|Project]* (inner-join tree))` plans that used to run as a star scan (`mx_filter` writes
every joined row) followed by `mx_execute` on the joined table run as ONE native call: per 1024-row block
the fact table's own filter, the partner lookups, the cross-partner residuals and filters above the join
(in the aggregate program's predicate) and the aggregate, with no joined row written. Every result is
compared with DuckDB; every test asserts the fused call ran and `mx_filter` did not (a counter around the
native calls), and a plan that does not fit declines to the old path with the same answer."""
import ctypes

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit, when
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

NAMES = [f"n{i}" for i in range(7)]


def _dict_col(values, n_chunks=1, seed=0):
    """A dictionary column cut into chunks that each carry their own dictionary, in their own order."""
    r = np.random.default_rng(seed)
    cuts = np.linspace(0, len(values), n_chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vals = list(values[a:b])
        d = sorted(set(vals), key=lambda _: r.random())
        idx = {v: i for i, v in enumerate(d)}
        parts.append(pa.DictionaryArray.from_arrays(A(np.array([idx[v] for v in vals], np.int32)), pa.array(d)))
    return pa.chunked_array(parts)


def _tables(n=120_000, seed=7, chunks=1, dup_free=True):
    r = np.random.default_rng(seed)
    fact = pa.table({
        "k": A(r.integers(-20, 5200, n).astype(np.int64)),              # dim1 key (some unmatched)
        "k2": A(r.integers(0, 330, n).astype(np.int32)),                 # dim2 key
        "k3": A(r.integers(0, 70, n).astype(np.int32)),                  # dim3 key
        "d": A(r.integers(0, 100, n).astype(np.int32)),
        "g": A(r.integers(0, 6, n).astype(np.int32)),
        "v": A(np.round(r.random(n) * 100, 2)),
        "q": A(r.integers(1, 50, n).astype(np.int32)),
        "f": _dict_col([["A", "B", "C"][i] for i in r.integers(0, 3, n)], chunks, seed),
    })
    m1 = 5000
    dim1 = pa.table({
        "dk": A(np.arange(m1, dtype=np.int64)),
        "cls": A(r.integers(0, 9, m1).astype(np.int32)),
        "w": A(np.round(r.random(m1), 3)),
        "nm": _dict_col([NAMES[i] for i in r.integers(0, 7, m1)], 1, seed + 1),
        "day": A((r.integers(1992, 1999, m1) * 10000 + r.integers(1, 13, m1) * 100 + r.integers(1, 29, m1)).astype(np.int32)),
    })
    dim2 = pa.table({
        "ek": A(np.arange(300, dtype=np.int32)),
        "cls2": A(r.integers(0, 9, 300).astype(np.int32)),
        "w2": A(np.round(r.random(300), 3)),
    })
    dim3 = pa.table({
        "fk": A(np.arange(60, dtype=np.int32)),
        "nat": A(r.integers(0, 5, 60).astype(np.int32)),
        "w3": A(np.round(r.random(60), 3)),
        "reg": pa.array([f"r{i % 4}" for i in range(60)], type=pa.string()),
    })
    return fact, dim1, dim2, dim3


@pytest.fixture
def calls(monkeypatch):
    """Counts of native calls: `exec` holds the plan version of every `mx_execute`; `filter` counts every
    `mx_filter` and `star` those that join (plan version 2: the star scan that writes the joined rows; a
    plain leaf filter of a partner is not one)."""
    log = {"exec": [], "filter": 0, "star": 0, "rc3": []}
    real_call, real_filter = nx._call, nx._call_filter

    def call(lib, plan_addr, out_addr):
        ver = int((ctypes.c_int64 * 2).from_address(plan_addr)[1])
        log["exec"].append(ver)
        rc = real_call(lib, plan_addr, out_addr)
        if ver == 3:
            log["rc3"].append(rc)           # < 0: the fused call declined
        return rc

    def filt(lib, plan_addr, out_addr):
        log["filter"] += 1
        log["star"] += int((ctypes.c_int64 * 2).from_address(plan_addr)[1]) == 2
        return real_filter(lib, plan_addr, out_addr)

    monkeypatch.setattr(nx, "_call", call)
    monkeypatch.setattr(nx, "_call_filter", filt)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return log


def _duck(sql, **t):
    con = duckdb.connect()
    for k, v in t.items():
        con.register(k, v)
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _rows(t, cols=None):
    cols = cols or t.column_names
    d = t.select(cols).to_pydict()
    return [tuple(d[c][i] for c in cols) for i in range(t.num_rows)]


def _norm(rows, ordered):
    rows = [tuple(float(x) if isinstance(x, (float, np.floating)) else x for x in r) for r in rows]
    return rows if ordered else sorted(rows, key=lambda r: tuple((x is None, str(type(x)), x) for x in r))


def _same(got, exp, ordered=False, rel=1e-9):
    got, exp = _norm(got, ordered), _norm(exp, ordered)
    assert len(got) == len(exp), (len(got), len(exp))
    for g, e in zip(got, exp):
        assert len(g) == len(e)
        for a, b in zip(g, e):
            if isinstance(a, float) and isinstance(b, float):
                assert a == pytest.approx(b, rel=rel, abs=1e-9), (g, e)
            else:
                assert a == b, (g, e)


def _fused(calls):
    assert 3 in calls["exec"], f"the fused call did not run: {calls}, {nx.last_decline()}"
    assert calls["star"] == 0 and calls["rc3"] and min(calls["rc3"]) >= 0, calls


def _declined(calls, code=None):
    """The fused call was not made, or returned a negative code (`code`), and the old path answered."""
    assert all(rc < 0 for rc in calls["rc3"]), calls
    if code is not None:
        assert calls["rc3"] == [code], calls


def _run(plan, calls, device="cpu"):
    got = plan.compute(device=device)
    return got


@pytest.mark.parametrize("chunks", [1, 3])
def test_grouped_by_a_fact_key_over_one_partner(calls, chunks):
    fact, dim1, _, _ = _tables(chunks=chunks)
    got = (LazyFrame(fact).filter(col("d") < lit(70))
           .join(LazyFrame(dim1).filter(col("w") > lit(0.2)), left_on="k", right_on="dk")
           .groupby("g").agg((col("v") * col("w")).sum().alias("s"), col("q").count().alias("n"),
                             col("v").max().alias("mx"), col("w").min().alias("mn"), col("v").mean().alias("av"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT g, SUM(v * w), COUNT(q), MAX(v), MIN(w), AVG(v) FROM fact JOIN dim1 ON k = dk "
                "WHERE d < 70 AND w > 0.2 GROUP BY g", fact=fact, dim1=dim1)
    _same(_rows(got, ["g", "s", "n", "mx", "mn", "av"]), exp)
    assert got.schema.field("g").type == pa.int32()


def test_global_aggregate_over_two_partners(calls):
    fact, dim1, dim2, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim2).filter(col("w2") < lit(0.8)), left_on="k2", right_on="ek")
           .groupby().agg((col("v") * col("w") * col("w2")).sum().alias("s"), col("v").count().alias("n"),
                          col("w2").max().alias("m"), col("w").mean().alias("a"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT SUM(v * w * w2), COUNT(v), MAX(w2), AVG(w) FROM fact JOIN dim1 ON k = dk "
                "JOIN dim2 ON k2 = ek WHERE w2 < 0.8", fact=fact, dim1=dim1, dim2=dim2)
    _same(_rows(got), exp)


def test_global_aggregate_that_matches_nothing(calls):
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1).filter(col("w") > lit(5.0)), left_on="k", right_on="dk")
           .groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    exp = _duck("SELECT SUM(v), COUNT(v) FROM fact JOIN dim1 ON k = dk WHERE w > 5.0", fact=fact, dim1=dim1)
    assert _rows(got) == exp == [(None, 0)]


def test_having_shaped_filter_above_the_aggregate(calls):
    fact, dim1, dim2, _ = _tables()
    sums = sorted(r[0] for r in _duck("SELECT SUM(v * w2) FROM fact JOIN dim1 ON k = dk JOIN dim2 ON k2 = ek GROUP BY g",
                                      fact=fact, dim1=dim1, dim2=dim2))
    cut = (sums[2] + sums[3]) / 2                        # three groups pass
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim2), left_on="k2", right_on="ek")
           .groupby("g").agg((col("v") * col("w2")).sum().alias("s"), col("v").count().alias("n"))
           .filter(col("s") > lit(cut)).compute(device="cpu"))
    exp = _duck(f"SELECT g, SUM(v * w2) s, COUNT(v) n FROM fact JOIN dim1 ON k = dk JOIN dim2 ON k2 = ek "
                f"GROUP BY g HAVING SUM(v * w2) > {cut!r}", fact=fact, dim1=dim1, dim2=dim2)
    assert len(exp) == 3
    _fused(calls)
    _same(_rows(got, ["g", "s", "n"]), exp)


@pytest.mark.parametrize("partners", [1, 2, 3])
def test_order_by_and_limit_over_partners(calls, partners):
    fact, dim1, dim2, dim3 = _tables()
    lf = LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
    sql = "FROM fact JOIN dim1 ON k = dk"
    if partners >= 2:
        lf = lf.join(LazyFrame(dim2), left_on="k2", right_on="ek")
        sql += " JOIN dim2 ON k2 = ek"
    if partners >= 3:
        lf = lf.join(LazyFrame(dim3), left_on="k3", right_on="fk")
        sql += " JOIN dim3 ON k3 = fk"
    got = (lf.groupby("g", "d").agg((col("v") * col("w")).sum().alias("rev"), col("v").count().alias("n"))
           .sort("rev", "g", descending=[True, False]).limit(7).compute(device="cpu"))
    _fused(calls)
    exp = _duck(f"SELECT g, d, SUM(v * w) rev, COUNT(v) n {sql} GROUP BY g, d ORDER BY rev DESC, g LIMIT 7",
                fact=fact, dim1=dim1, dim2=dim2, dim3=dim3)
    _same(_rows(got, ["g", "d", "rev", "n"]), exp, ordered=True)


def test_group_keys_on_partner_columns_int_and_dictionary(calls):
    fact, dim1, dim2, dim3 = _tables(chunks=2)
    got = (LazyFrame(fact).filter(col("d") > lit(10))
           .join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim3), left_on="k3", right_on="fk")
           .groupby("nm", "cls", "nat", "f").agg((col("v") * col("w3")).sum().alias("s"), col("v").count().alias("n"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT nm, cls, nat, f, SUM(v * w3), COUNT(v) FROM fact JOIN dim1 ON k = dk JOIN dim3 ON k3 = fk "
                "WHERE d > 10 GROUP BY nm, cls, nat, f", fact=fact, dim1=dim1, dim3=dim3)
    _same(_rows(got, ["nm", "cls", "nat", "f", "s", "n"]), exp)
    assert got.schema.field("nm").type == pa.string()
    assert got.schema.field("cls").type == pa.int32()


def test_partner_string_key_comes_out_in_byte_order(calls):
    fact, dim1, _, dim3 = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim3), left_on="k3", right_on="fk")
           .groupby("reg").agg(col("v").sum().alias("s")).compute(device="cpu"))        # plain utf8 partner column
    _fused(calls)
    exp = _duck("SELECT reg, SUM(v) FROM fact JOIN dim3 ON k3 = fk GROUP BY reg ORDER BY reg", fact=fact, dim3=dim3)
    _same(_rows(got), exp, ordered=True)


def test_cross_partner_residual(calls):
    fact, dim1, dim2, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim2), left_on="k2", right_on="ek")
           .filter(col("cls") == col("cls2"))
           .groupby("g").agg((col("v") * col("w")).sum().alias("s"), col("v").count().alias("n"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT g, SUM(v * w), COUNT(v) FROM fact JOIN dim1 ON k = dk JOIN dim2 ON k2 = ek "
                "WHERE cls = cls2 GROUP BY g", fact=fact, dim1=dim1, dim2=dim2)
    assert exp
    _same(_rows(got, ["g", "s", "n"]), exp)


def test_filter_above_the_join_across_partners(calls):
    fact, dim1, dim2, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim2), left_on="k2", right_on="ek")
           .filter((col("w") + col("w2") > lit(1.2)) & (col("v") < lit(60.0)))
           .groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT SUM(v), COUNT(v) FROM fact JOIN dim1 ON k = dk JOIN dim2 ON k2 = ek "
                "WHERE w + w2 > 1.2 AND v < 60.0", fact=fact, dim1=dim1, dim2=dim2)
    _same(_rows(got), exp)


def test_case_when_on_a_partner_dictionary_column(calls):
    fact, dim1, _, _ = _tables(chunks=2)
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .groupby("f").agg(when(col("nm").isin(["n1", "n3"]), col("v"), lit(0.0)).sum().alias("hi"),
                             when(col("nm") == lit("n5"), lit(1.0), lit(0.0)).sum().alias("c5"),
                             col("v").sum().alias("tot"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT f, SUM(CASE WHEN nm IN ('n1', 'n3') THEN v ELSE 0.0 END), "
                "SUM(CASE WHEN nm = 'n5' THEN 1.0 ELSE 0.0 END), SUM(v) FROM fact JOIN dim1 ON k = dk GROUP BY f",
                fact=fact, dim1=dim1)
    _same(_rows(got, ["f", "hi", "c5", "tot"]), exp)


def _year(c, y0=1992, y1=1998):
    """Year of an integer YYYYMMDD column as a sum of CASE WHENs (what the TPC-H queries do)."""
    e = when(col(c) >= lit((y0 + 1) * 10000 + 101), lit(1), lit(0))
    for y in range(y0 + 2, y1 + 1):
        e = e + when(col(c) >= lit(y * 10000 + 101), lit(1), lit(0))
    return e + lit(y0)


def test_derived_group_keys_over_partner_columns(calls):
    fact, dim1, _, dim3 = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim3), left_on="k3", right_on="fk")
           .with_columns(_year("day").alias("y"), (col("cls") * lit(10) + col("nat")).alias("cn"))
           .groupby("y", "cn", "g").agg(col("v").sum().alias("s")).compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT day // 10000 y, cls * 10 + nat cn, g, SUM(v) "
                "FROM fact JOIN dim1 ON k = dk JOIN dim3 ON k3 = fk GROUP BY 1, 2, 3", fact=fact, dim1=dim1, dim3=dim3)
    _same(_rows(got, ["y", "cn", "g", "s"]), exp)


def test_semi_partner_keys_and_a_fact_filter_with_dictionary(calls):
    fact, dim1, _, _ = _tables(chunks=3)
    got = (LazyFrame(fact).filter(col("f") == lit("B"))
           .join(LazyFrame(dim1).filter(col("cls") < lit(4)).select("dk"), left_on="k", right_on="dk")
           .groupby("g").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT g, SUM(v), COUNT(v) FROM fact WHERE f = 'B' AND k IN (SELECT dk FROM dim1 WHERE cls < 4) GROUP BY g",
                fact=fact, dim1=dim1)
    _same(_rows(got, ["g", "s", "n"]), exp)


# ── declines: the same answer, from the old path ─────────────────────────────────────────────────

def _plan(fact, dim1, dim2):
    return (LazyFrame(fact).filter(col("d") < lit(80))
            .join(LazyFrame(dim1), left_on="k", right_on="dk")
            .join(LazyFrame(dim2), left_on="k2", right_on="ek")
            .groupby("g", "cls").agg((col("v") * col("w2")).sum().alias("s"), col("v").count().alias("n")))


def test_switch_off_gives_the_same_answer_through_the_star_scan(calls, monkeypatch):
    fact, dim1, dim2, _ = _tables()
    fused = _plan(fact, dim1, dim2).compute(device="cpu")
    _fused(calls)
    calls["exec"].clear()
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")
    old = _plan(fact, dim1, dim2).compute(device="cpu")
    assert 3 not in calls["exec"] and calls["star"] > 0, calls
    _same(_rows(fused), _rows(old))


def test_duplicate_partner_keys_decline(calls):
    fact, dim1, dim2, _ = _tables()
    dup = pa.concat_tables([dim2, dim2.slice(0, 5)])
    got = (LazyFrame(fact).join(LazyFrame(dup), left_on="k2", right_on="ek")
           .groupby("g").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    _declined(calls, -18)                                  # E_DUP: the partner's key is not unique
    exp = _duck("SELECT g, SUM(v), COUNT(v) FROM fact JOIN dup ON k2 = ek GROUP BY g", fact=fact, dup=dup)
    _same(_rows(got, ["g", "s", "n"]), exp)


def test_left_join_declines(calls):
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1).filter(col("w") > lit(0.5)), left_on="k", right_on="dk", how="left")
           .groupby("g").agg(col("v").sum().alias("s"), col("w").count().alias("n")).compute(device="cpu"))
    assert calls["exec"] == [2] or 3 not in calls["exec"], calls          # a left join is not a star
    exp = _duck("SELECT g, SUM(v), COUNT(w) FROM fact LEFT JOIN (SELECT * FROM dim1 WHERE w > 0.5) d ON k = dk GROUP BY g",
                fact=fact, dim1=dim1)
    _same(_rows(got, ["g", "s", "n"]), exp)


def test_int64_past_2_53_declines_with_the_exact_answer(calls, monkeypatch):
    """The interpreter computes in float64: an int64 column past +-2^53 (a fact predicate, a fact value
    column, a partner value column) must decline, never round (F-037)."""
    n = 60_000
    r = np.random.default_rng(3)
    big = np.full(n, (1 << 53) + 1, dtype=np.int64)
    big[::5] = 7
    fact = pa.table({"k": A((np.arange(n) % 500).astype(np.int64)), "big": A(big), "v": A(np.ones(n)),
                     "g": A((np.arange(n) % 3).astype(np.int32))})
    pbig = np.full(500, (1 << 53) + 1, dtype=np.int64)
    pbig[::4] = 9
    dim = pa.table({"dk": A(np.arange(500, dtype=np.int64)), "pbig": A(pbig), "w": A(r.random(500))})
    cases = {
        "fact predicate": (LazyFrame(fact).filter(col("big") > lit(1 << 53)).join(LazyFrame(dim), left_on="k", right_on="dk")
                           .groupby("g").agg(col("v").sum().alias("s")),
                           "SELECT g, SUM(v) FROM fact JOIN dim ON k = dk WHERE big > 9007199254740992 GROUP BY g"),
        "fact value": (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
                       .groupby("g").agg(col("big").max().alias("m")),
                       "SELECT g, MAX(big) FROM fact JOIN dim ON k = dk GROUP BY g"),
        "partner value": (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
                          .groupby("g").agg(col("pbig").max().alias("m")),
                          "SELECT g, MAX(pbig) FROM fact JOIN dim ON k = dk GROUP BY g"),
        "partner predicate": (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
                              .filter(col("pbig") > lit(1 << 53)).groupby("g").agg(col("v").sum().alias("s")),
                              "SELECT g, SUM(v) FROM fact JOIN dim ON k = dk WHERE pbig > 9007199254740992 GROUP BY g"),
    }
    for name, (plan, sql) in cases.items():
        calls["exec"].clear()
        got = plan.compute(device="cpu")
        if name == "partner predicate":
            # the partner's own filter declines (its leaf filter sees the int64) and runs in Arrow; the
            # fused call then reads no int64 column and is exact
            assert calls["rc3"] == [0], (name, calls)
        else:
            _declined(calls, -16)                          # E_INEXACT
        calls["rc3"].clear()
        if "MAX" in sql:
            # today's path folds a max over int64 into float64 whatever the join (a limit of the old
            # path, not of this one): the fused decline must give exactly that answer
            monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")
            old = plan.compute(device="cpu")
            monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "1")
            assert got.to_pydict() == old.to_pydict()
        else:
            _same(_rows(got), _duck(sql, fact=fact, dim=dim))


def test_a_wide_group_key_range_declines(calls):
    """k x g x d spans 3 million codes: past direct addressing the executor returns E_BAD_KEY (-11), the
    old path answers (hashed keys, until the fused call takes them)."""
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .groupby("k", "g", "d").agg(col("v").sum().alias("s")).compute(device="cpu"))
    _declined(calls, -11)
    exp = _duck("SELECT k, g, d, SUM(v) FROM fact JOIN dim1 ON k = dk GROUP BY k, g, d", fact=fact, dim1=dim1)
    _same(_rows(got, ["k", "g", "d", "s"]), exp)
    assert calls["star"] > 0, calls


def test_global_aggregate_over_a_fact_filter_that_keeps_nothing(calls):
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).filter(col("d") > lit(1000)).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n"), col("w").max().alias("m"))
           .compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT SUM(v), COUNT(v), MAX(w) FROM fact JOIN dim1 ON k = dk WHERE d > 1000", fact=fact, dim1=dim1)
    assert _rows(got) == exp == [(None, 0, None)]




def test_a_repeated_query_is_answered_from_the_join_cache(calls, monkeypatch):
    """The old path keeps a join's result for the next query; asking the fused call then would only add cost."""
    import mxframe.custom_ops as co
    fact, dim1, dim2, _ = _tables()
    # tables holding exactly the columns the query reads (the cache is keyed by table identity, and column
    # pruning would make a new table per query)
    fact, dim1, dim2 = fact.select(["k", "k2", "g", "v"]), dim1.select(["dk", "w"]), dim2.select(["ek"])
    plan = lambda: (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
                    .join(LazyFrame(dim2), left_on="k2", right_on="ek").groupby("g").agg(col("w").sum().alias("s")))
    co._JOIN_RESULT_CACHE.clear()
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")
    first = plan().compute(device="cpu")                    # fills the cache
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "1")
    calls["exec"].clear()
    again = plan().compute(device="cpu")
    assert 3 not in calls["exec"], calls
    _same(_rows(again), _rows(first))
    co._JOIN_RESULT_CACHE.clear()
    calls.update(exec=[], star=0, filter=0, rc3=[])
    plan().compute(device="cpu")
    _fused(calls)                                           # nothing cached: fused


def test_a_declined_fused_call_does_not_build_the_partners_again(calls, monkeypatch):
    import mxframe.join_engine as je
    built = []
    real = je._star_setup
    monkeypatch.setattr(je, "_star_setup", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .groupby("k", "g", "d").agg(col("v").sum().alias("s")).compute(device="cpu"))
    _declined(calls, -11)
    assert len(built) == 1, built
    assert got.num_rows > 0


def test_nothing_is_built_when_the_executor_cannot_run_the_aggregate(calls, monkeypatch):
    import mxframe.join_engine as je
    built = []
    real = je._star_setup
    monkeypatch.setattr(je, "_star_setup", lambda *a, **k: (built.append(1), real(*a, **k))[1])
    fact, dim1, _, _ = _tables()
    plan = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk").groupby("g").agg(col("v").sum().alias("s")))
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    plan.compute(device="cpu")
    assert built == [] and calls["exec"] == []


# ── group keys past direct addressing: group by the partner's row (K_PROW), merged when keys repeat ───────

def _wide_dim(m=6000, seed=11):
    """A partner whose key columns multiply to 40 million codes, with only ~3500 distinct tuples (so several
    partner rows share one tuple)."""
    r = np.random.default_rng(seed)
    return pa.table({
        "dk": A(np.arange(m, dtype=np.int64)),
        "a": A((r.integers(0, 50, m) * 40_000).astype(np.int64)),
        "b": A(r.integers(-5, 6, m).astype(np.int32)),                 # negative values among the keys
        "nm": _dict_col([NAMES[i] for i in r.integers(0, 7, m)], 1, seed),
        "w": A(np.round(r.random(m), 3)),
    })


def test_keys_with_the_partners_unique_key_group_by_partner_row(calls):
    """TPC-H Q3's shape: the fact's join key plus partner columns; one group per partner row, no merge."""
    fact, dim1, _, _ = _tables()
    got = (LazyFrame(fact).filter(col("d") < lit(60)).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .groupby("k", "day", "cls").agg((col("v") * col("w")).sum().alias("rev"), col("v").count().alias("n"))
           .sort("rev", descending=True).limit(10).compute(device="cpu"))
    _fused(calls)
    assert calls["rc3"] == [0], calls
    exp = _duck("SELECT k, day, cls, SUM(v * w) rev, COUNT(v) n FROM fact JOIN dim1 ON k = dk WHERE d < 60 "
                "GROUP BY k, day, cls ORDER BY rev DESC, k, day, cls LIMIT 10", fact=fact, dim1=dim1)
    _same(_rows(got, ["k", "day", "cls", "rev", "n"]), exp, ordered=True)
    assert got.schema.field("k").type == pa.int64() and got.schema.field("cls").type == pa.int32()


def test_negative_key_values_after_the_first_key_group_by_partner_row(calls):
    """Keys with the partner's unique key (so one group per partner row), a negative value in a non-first key,
    in every order."""
    fact, _, _, _ = _tables()
    dim = _wide_dim()
    fact = fact.set_column(0, "k", A((np.asarray(fact.column("k").to_numpy()) % 6000).astype(np.int64)))
    for keys in (["k", "b", "a"], ["b", "k", "a"], ["a", "b", "k"]):
        calls.update(exec=[], star=0, filter=0, rc3=[])
        got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
               .groupby(*keys).agg((col("v") * col("w")).sum().alias("s"), col("v").count().alias("n"),
                                   col("v").mean().alias("av")).compute(device="cpu"))
        _fused(calls)
        assert calls["rc3"] == [0], (keys, calls)
        exp = _duck(f"SELECT {', '.join(keys)}, SUM(v * w) s, COUNT(v) n, AVG(v) av FROM fact JOIN dim ON k = dk "
                    f"GROUP BY {', '.join(keys)} ORDER BY {', '.join(keys)}", fact=fact, dim=dim)
        _same(_rows(got, keys + ["s", "n", "av"]), exp, ordered=True)


@pytest.mark.parametrize("keys", [["a", "b", "nm"], ["b", "a"], ["a", "b"]])
def test_partner_keys_without_the_unique_key_decline_with_the_right_groups(calls, keys):
    """Several partner rows share a key tuple (TPC-H Q10's per-customer keys through orders): grouping by
    partner row would need a merge, so the fused call declines and the old path answers (negative key
    values included)."""
    fact, _, _, _ = _tables()
    dim = _wide_dim()
    fact = fact.set_column(0, "k", A((np.asarray(fact.column("k").to_numpy()) % 6000).astype(np.int64)))
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .groupby(*keys).agg((col("v") * col("w")).sum().alias("s"), col("v").count().alias("n"),
                               col("v").mean().alias("av")).compute(device="cpu"))
    _declined(calls)
    exp = _duck(f"SELECT {', '.join(keys)}, SUM(v * w) s, COUNT(v) n, AVG(v) av FROM fact JOIN dim ON k = dk "
                f"GROUP BY {', '.join(keys)} ORDER BY {', '.join(keys)}", fact=fact, dim=dim)
    assert len(exp) > 100
    _same(_rows(got, keys + ["s", "n", "av"]), exp, ordered=True)


@pytest.mark.parametrize("having", [False, True])
def test_top_k_by_a_count_with_heavy_ties_follows_the_key_tie_break(calls, having):
    """ORDER BY count LIMIT 7 where most counts are equal: the rows that tie at the boundary come out in
    key order, as DuckDB's `ORDER BY n DESC, keys` gives them."""
    fact, _, _, _ = _tables()
    dim = _wide_dim()
    n = fact.num_rows
    k = (12 + np.arange(n) % 138) * 40                               # 138 keys, about 870 rows each
    k[: 12 * 1000] = np.arange(12 * 1000) % 12 * 40                  # 12 keys with exactly 1000 more: they tie for the top 7
    k = k.astype(np.int64)
    fact = fact.set_column(0, "k", A(k))
    lf = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
          .groupby("k", "b", "a").agg(col("v").count().alias("n"), col("v").sum().alias("s")))
    sql = "SELECT k, b, a, COUNT(v) n, SUM(v) s FROM fact JOIN dim ON k = dk GROUP BY k, b, a"
    if having:
        lf, sql = lf.filter(col("n") > lit(3)), sql + " HAVING COUNT(v) > 3"
    got = lf.sort("n", descending=True).limit(7).compute(device="cpu")
    _fused(calls)
    exp = _duck(sql + " ORDER BY n DESC, k, b, a LIMIT 7", fact=fact, dim=dim)
    _same(_rows(got, ["k", "b", "a", "n", "s"]), exp, ordered=True)


def test_integer_sums_over_fact_and_partner_columns(calls):
    """Sums of integer columns (and products of them) are checked against 2^53 from the joined row count:
    the fused compile must see the same bounds as the single-table one, and not fail on them."""
    fact, dim1, dim2, _ = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim1), left_on="k", right_on="dk")
           .join(LazyFrame(dim2), left_on="k2", right_on="ek")
           .groupby("g").agg(col("q").sum().alias("sq"), col("cls").sum().alias("sc"),
                             (col("q") * col("cls2")).sum().alias("sp")).compute(device="cpu"))
    _fused(calls)
    exp = _duck("SELECT g, SUM(q), SUM(cls), SUM(q * cls2) FROM fact JOIN dim1 ON k = dk JOIN dim2 ON k2 = ek GROUP BY g",
                fact=fact, dim1=dim1, dim2=dim2)
    _same(_rows(got, ["g", "sq", "sc", "sp"]), exp)
