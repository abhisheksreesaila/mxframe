"""F-069: float64 group keys and key tuples wider than direct addressing run in the native executor as a
partitioned hash aggregate (single table and the fused join + aggregate), instead of declining.

Every result is compared with DuckDB (built from a native DuckDB table: a registered Arrow table treats NaN
differently) and every test asserts the path: the native hooks ran, none declined, and the executor
reports that it grouped by hash (`nx.last_group_path()`), so a result that came back right through the
Arrow fallback cannot pass. Float keys: bit-exact equality, every NaN in one group, -0.0 with 0.0 (one
group, reported as +0.0), NaN after +inf in key order, as DuckDB. A NULL in a group key still declines
(`test_a_null_float_key_declines_and_the_answer_stays_right`)."""
import ctypes
import math
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from conftest import TPCH_DATA
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

NAN = float("nan")
NEG_NAN = np.uint64(0xFFF8000000000001).view(np.float64).item()          # a NaN with another sign and payload
PALETTE = [0.0, -0.0, NAN, NEG_NAN, math.inf, -math.inf, 1.5, 1.5000000000000002, 1e300, -1e-300, 5e-324, -7.25]


@pytest.fixture
def hooks(monkeypatch):
    """(hook, ran, decline, group path) of every aggregate hook call."""
    log = []
    for name, tag in (("try_execute", "execute"), ("try_execute_join", "join")):
        real = getattr(nx, name)

        def wrap(*a, _real=real, _tag=tag, **k):
            r = _real(*a, **k)
            log.append((_tag, r is not None, None if r is not None else nx.last_decline(),
                        nx.last_group_path() if r is not None else None))
            return r
        monkeypatch.setattr(nx, name, wrap)
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    return log


def _duck(sql, **tables):
    con = duckdb.connect()
    for name, t in tables.items():
        con.register("_" + name, t)
        con.execute(f"create table {name} as select * from _{name}")
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _rows(t):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    return [tuple(c[r] for c in cols) for r in range(t.num_rows)]


def _eq(x, y):
    if x is None or y is None:
        return x is None and y is None
    if isinstance(x, float) or isinstance(y, float):
        if math.isnan(x) or math.isnan(y):
            return math.isnan(x) and math.isnan(y)
        if math.isinf(x) or math.isinf(y):
            return x == y
        return x == pytest.approx(y, rel=1e-9, abs=1e-12)
    return x == y


def _key(r):
    return tuple((x is None, isinstance(x, float) and math.isnan(x), 0 if x is None or (isinstance(x, float) and math.isnan(x)) else x)
                 for x in r)


def _same(got, exp, ordered=False):
    if not ordered:
        got, exp = sorted(got, key=_key), sorted(exp, key=_key)
    assert len(got) == len(exp), (len(got), len(exp))
    for g, e in zip(got, exp):
        assert len(g) == len(e) and all(_eq(x, y) for x, y in zip(g, e)), (g, e)


def _hashed(hooks, join=None):
    """Every aggregate hook call that did work ran (a zero-row schema probe declines 'empty input'), and the
    last grouped one went by hash."""
    live = [h for h in hooks if h[1] or h[2] != "empty input"]
    assert live, "no native aggregate call was made"
    assert all(h[1] for h in live), live
    assert live[-1][3] == "hash", live
    if join is not None:
        assert (live[-1][0] == "join") == join, live


def _sql(sql, hooks, ordered=False, join=None, **tables):
    got = _rows(mx.sql(sql, **tables).compute(device="cpu"))
    _same(got, _duck(sql, **tables), ordered)
    _hashed(hooks, join)
    return got


def _float_table(n=60_000, chunks=3, seed=69):
    r = np.random.default_rng(seed)
    f = np.array(PALETTE)[r.integers(0, len(PALETTE), n)]
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    cols = {"f": pa.chunked_array([A(f[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]),
            "g": A(r.integers(0, 4, n).astype(np.int32)),
            "w": A(r.integers(-10**12, 10**12, n).astype(np.int64)),
            "v": A(np.round(r.random(n) * 100, 2)),
            "s": pa.array([f"s{i}" for i in r.integers(0, 5, n)])}
    return pa.table(cols)


# ── float keys, single table ─────────────────────────────────────────────────────────────────────

def test_float_key_with_nan_signed_zero_and_infinities(hooks):
    t = _float_table()
    got = _sql("select f, sum(v) as sv, count(*) as n, min(v) as lo, max(v) as hi, avg(v) as m from t group by f",
               hooks, t=t)
    assert len(got) == len(PALETTE) - 2        # the two NaNs are one group, -0.0 joins 0.0
    nan_group = [r for r in got if math.isnan(r[0])]
    zero_group = [r for r in got if r[0] == 0.0]
    assert len(nan_group) == 1 and len(zero_group) == 1


def test_float_key_values_one_ulp_apart_are_different_groups(hooks):
    t = _float_table()
    got = _sql("select f, count(*) as n from t group by f", hooks, t=t)
    assert {r[0] for r in got if r[0] > 1.0 and r[0] < 2.0} == {1.5, 1.5000000000000002}


def test_float_key_result_comes_back_as_float64_in_key_order_nan_last(hooks):
    t = _float_table()
    got = mx.sql("select f, count(*) as n from t group by f", t=t).compute(device="cpu")
    assert got.schema.field("f").type == pa.float64()
    keys = got.column("f").to_pylist()
    assert keys[-1] != keys[-1] and keys[-2] == math.inf and keys[0] == -math.inf        # NaN after +inf, as DuckDB
    assert keys[1:-1] == sorted(keys[1:-1])
    _hashed(hooks)


def test_float_key_with_filter_having_order_by_and_limit(hooks):
    t = _float_table()
    _sql("select f, sum(v) as sv from t where g < 3 and v > 20 group by f having sum(v) > 100000 order by f",
         hooks, ordered=True, t=t)
    hooks.clear()
    got = _sql("select f, sum(v) as sv, count(*) as n from t group by f order by sv desc, f limit 3",
               hooks, ordered=True, t=t)
    assert len(got) == 3
    hooks.clear()
    _sql("select f, count(*) as n from t group by f order by f desc limit 4", hooks, ordered=True, t=t)


def test_float_key_next_to_integer_string_and_wide_keys(hooks):
    t = _float_table()
    _sql("select f, g, sum(v) as sv from t group by f, g", hooks, t=t)
    hooks.clear()
    _sql("select f, s, count(*) as n, max(v) as hi from t group by f, s", hooks, t=t)
    hooks.clear()
    _sql("select w, f, sum(v) as sv from t group by w, f", hooks, t=t)


def test_float32_key_keeps_its_type_and_the_nan_zero_and_infinity_rules(hooks):
    t = _float_table()
    t = t.set_column(0, "f", pa.chunked_array([c.cast(pa.float32()) for c in t.column("f").chunks]))
    got = mx.sql("select f, sum(v) as sv, count(*) as n from t group by f", t=t).compute(device="cpu")
    assert got.schema.field("f").type == pa.float32()
    _same(_rows(got), _duck("select f, sum(v), count(*) from t group by f", t=t))
    _hashed(hooks)


def test_float_key_through_the_dataframe_api(hooks):
    t = _float_table()
    got = _rows(LazyFrame(t).filter(col("g") < lit(3)).groupby("f").agg(col("v").sum().alias("sv"), col("v").count().alias("n"))
                .compute(device="cpu"))
    _same(got, _duck("select f, sum(v), count(v) from t where g < 3 group by f", t=t))
    _hashed(hooks)


def test_float_key_when_the_filter_keeps_nothing(hooks):
    t = _float_table()
    got = _sql("select f, sum(v) as sv from t where v < 0 group by f", hooks, t=t)
    assert got == []


def test_a_float_key_under_another_name_and_a_wide_expression_key(hooks):
    t = _float_table()
    _sql("select f as x, sum(v) as sv from t group by x", hooks, t=t)
    hooks.clear()
    _sql("select g, w + g as k, count(*) as n from t group by g, k", hooks, t=t)


def test_a_null_float_key_declines_and_the_answer_stays_right(hooks):
    t = _float_table(n=3000)
    f = [x if x == x and x != 0.0 else 2.5 for x in t.column("f").to_pylist()]      # no NaN / -0.0: the Arrow answer is DuckDB's
    f[5] = f[17] = None
    t = t.set_column(0, "f", pa.array(f, pa.float64()))
    got = _rows(mx.sql("select f, count(*) as n from t group by f", t=t).compute(device="cpu"))
    _same(got, _duck("select f, count(*) from t group by f", t=t))
    assert not any(h[1] and h[3] == "hash" for h in hooks), hooks        # a NULL key is not hashed natively


# ── keys wider than direct addressing ────────────────────────────────────────────────────────────

def _wide_table(n=80_000, chunks=2, seed=70):
    r = np.random.default_rng(seed)
    pool = np.concatenate([[-2**63, 2**63 - 1, 0, -1], r.integers(-2**62, 2**62, 400)]).astype(np.int64)
    cuts = np.linspace(0, n, chunks + 1).astype(int)
    a = pool[r.integers(0, len(pool), n)]
    b = pool[r.integers(0, len(pool), n)]
    c = r.integers(-10**15, 10**15, 40)[r.integers(0, 40, n)].astype(np.int64)
    ch = lambda x: pa.chunked_array([A(x[i:j]) for i, j in zip(cuts[:-1], cuts[1:])])
    return pa.table({"a": ch(a), "b": ch(b), "c": ch(c), "v": A(np.round(r.random(n) * 100, 2)),
                     "s": pa.array([f"k{i}" for i in r.integers(0, 30, n)])})


def test_int64_pair_spanning_the_whole_int64_range(hooks):
    t = _wide_table()
    got = _sql("select a, b, sum(v) as sv, count(*) as n from t group by a, b", hooks, t=t)
    assert len(got) > 5000
    assert any(r[0] == -2**63 for r in got) and any(r[0] == 2**63 - 1 for r in got)


def test_three_and_four_wide_key_columns(hooks):
    t = _wide_table()
    _sql("select a, b, c, min(v) as lo, max(v) as hi from t group by a, b, c", hooks, t=t)
    hooks.clear()
    _sql("select a, b, c, s, count(*) as n from t where v > 10 group by a, b, c, s", hooks, t=t)


def test_wide_keys_with_having_order_by_and_limit(hooks):
    t = _wide_table()
    _sql("select a, b, sum(v) as sv from t group by a, b having count(*) > 1 order by sv desc, a, b limit 10",
         hooks, ordered=True, t=t)
    hooks.clear()
    _sql("select a, c, sum(v) as sv from t group by a, c order by a, c limit 25", hooks, ordered=True, t=t)


def test_many_groups_over_a_million(hooks):
    r = np.random.default_rng(71)
    n = 2_400_000
    k = r.integers(0, 1_500_000, n).astype(np.int64) * 7919 - 10**12          # about 1.2M distinct, a wide range
    cuts = [0, 800_000, 1_700_000, n]
    t = pa.table({"k": pa.chunked_array([A(k[a:b]) for a, b in zip(cuts[:-1], cuts[1:])]),
                  "m": A(r.integers(0, 3, n).astype(np.int32)), "v": A(np.round(r.random(n) * 10, 2))})
    got = _sql("select k, m, sum(v) as sv, count(*) as n from t group by k, m having count(*) > 1", hooks, t=t)
    assert len(got) > 300_000
    hooks.clear()
    got = _sql("select k, sum(v) as sv from t group by k order by sv desc, k limit 7", hooks, ordered=True, t=t)
    assert len(got) == 7
    hooks.clear()
    got = _rows(mx.sql("select k, sum(v) as sv from t group by k", t=t).compute(device="cpu"))
    assert len(got) > 1_000_000
    _same(got, _duck("select k, sum(v) from t group by k", t=t))
    _hashed(hooks)


# ── reproducible mode on the hashed path ─────────────────────────────────────────────────────────

def _bits(t, name):
    return np.ascontiguousarray(t.column(name).to_numpy()).view(np.int64)


def _repro_table(n, groups, seed=72):
    r = np.random.default_rng(seed)
    keys = np.concatenate([r.random(groups) * 1000, [NAN, math.inf, -math.inf]])
    return pa.table({"f": A(keys[r.integers(0, len(keys), n)]), "p": A(r.random(n) * 1000), "d": A(r.random(n))})


def _revenue(lf):
    return lf.groupby("f").agg((col("p") * (lit(1.0) - col("d"))).sum().alias("tot"), col("p").mean().alias("av"))


@pytest.mark.parametrize("groups", [40, 9_000, 400_000])
def test_reproducible_float_key_sums_are_bit_identical_across_runs_and_thread_counts(hooks, monkeypatch, groups):
    plan = _revenue(LazyFrame(_repro_table(1_500_000, groups)))
    runs = []
    for threads in ("8", "8", "8", "3", "16"):
        monkeypatch.setenv("MXFRAME_CPU_THREADS", threads)
        runs.append(plan.compute(device="cpu", reproducible=True))
    _hashed(hooks)
    for other in runs[1:]:
        for name in ("tot", "av"):
            assert int((_bits(runs[0], name) != _bits(other, name)).sum()) == 0, (groups, name)
        assert _bits(runs[0], "f").tolist() == _bits(other, "f").tolist()


@pytest.mark.parametrize("groups", [9_000, 100_000])
def test_reproducible_sums_follow_the_defined_order_per_morsel_then_in_morsel_order(hooks, monkeypatch, groups):
    """The same NumPy reference as F-064's tests (each group's rows added in row order within a morsel, the
    morsels' sub-totals in morsel order), bit for bit (one lane per group, as F-064's path for more than 1,024 groups)."""
    n, morsel = 1_000_000, 65536
    t = _repro_table(n, groups)
    f = t.column("f").to_numpy()
    f = np.where(f == f, f, 3.0e9)            # NaN key -> an ordinary code for the reference
    _, code = np.unique(f, return_inverse=True)
    v = t.column("p").to_numpy() * (1.0 - t.column("d").to_numpy())
    want = np.zeros(code.max() + 1)
    for lo in range(0, n, morsel):
        sub = np.zeros_like(want)
        np.add.at(sub, code[lo:lo + morsel], v[lo:lo + morsel])
        want += sub
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    got = _revenue(LazyFrame(t)).compute(device="cpu", reproducible=True)
    _hashed(hooks)
    order = np.argsort(np.where(got.column("f").to_numpy() == got.column("f").to_numpy(), got.column("f").to_numpy(), 3.0e9))
    assert (_bits(got, "tot")[order] != want.view(np.int64)).sum() == 0


# ── the fused join + aggregate ───────────────────────────────────────────────────────────────────

@pytest.fixture
def calls(monkeypatch):
    """Plan version of every `mx_execute` (3 = the fused join + aggregate) and the return code of the fused ones."""
    log = {"exec": [], "rc3": []}
    real = nx._call

    def call(lib, plan_addr, out_addr):
        ver = int((ctypes.c_int64 * 2).from_address(plan_addr)[1])
        log["exec"].append(ver)
        rc = real(lib, plan_addr, out_addr)
        if ver == 3:
            log["rc3"].append(rc)
        return rc
    monkeypatch.setattr(nx, "_call", call)
    return log


def _fused(calls):
    assert 3 in calls["exec"] and calls["rc3"] and min(calls["rc3"]) >= 0, (calls, nx.last_decline())


def _star(n=150_000, seed=73):
    r = np.random.default_rng(seed)
    m = 4000
    fact = pa.table({"k": A(r.integers(0, m + 50, n).astype(np.int64)),
                     "f": A(np.array(PALETTE)[r.integers(0, len(PALETTE), n)]),
                     "a": A(r.integers(-2**62, 2**62, 300)[r.integers(0, 300, n)].astype(np.int64)),
                     "b": A(r.integers(-2**62, 2**62, 300)[r.integers(0, 300, n)].astype(np.int64)),
                     "d": A(r.integers(0, 100, n).astype(np.int32)),
                     "v": A(np.round(r.random(n) * 100, 2))})
    dim = pa.table({"dk": A(np.arange(m, dtype=np.int64)),
                    "w": A(np.round(r.random(m) * 50, 3)),
                    "big": A(np.arange(m, dtype=np.int64) * 10**12 - 2 * 10**15),
                    "nm": pa.array([f"n{i % 90}" for i in range(m)])})
    return fact, dim


def test_fused_join_grouped_by_a_float_fact_key(hooks, calls):
    fact, dim = _star()
    got = _rows(LazyFrame(fact).join(LazyFrame(dim).filter(col("w") > lit(5.0)), left_on="k", right_on="dk")
                .groupby("f").agg((col("v") * col("w")).sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    _same(got, _duck("select f, sum(v * w), count(v) from fact join dim on k = dk where w > 5.0 group by f", fact=fact, dim=dim))
    _fused(calls)
    _hashed(hooks, join=True)


def test_fused_join_grouped_by_a_float_partner_column(hooks, calls):
    fact, dim = _star()
    got = _rows(LazyFrame(fact).filter(col("d") < lit(60)).join(LazyFrame(dim), left_on="k", right_on="dk")
                .groupby("w").agg(col("v").sum().alias("s"), col("v").max().alias("hi")).compute(device="cpu"))
    _same(got, _duck("select w, sum(v), max(v) from fact join dim on k = dk where d < 60 group by w", fact=fact, dim=dim))
    assert len(got) > 3000
    _fused(calls)
    _hashed(hooks, join=True)


def test_fused_join_grouped_by_a_float32_partner_column(hooks, calls):
    fact, dim = _star()
    dim = dim.set_column(1, "w", dim.column("w").cast(pa.float32()))
    got = mx.sql("select w, sum(v) as s, count(*) as n from fact, dim where k = dk and d < 60 group by w",
                 fact=fact, dim=dim).compute(device="cpu")
    assert got.schema.field("w").type == pa.float32()
    _same(_rows(got), _duck("select w, sum(v), count(*) from fact, dim where k = dk and d < 60 group by w", fact=fact, dim=dim))
    _fused(calls)
    _hashed(hooks, join=True)


def test_fused_join_grouped_by_wide_keys_of_the_fact_and_the_partner(hooks, calls):
    fact, dim = _star()
    got = _rows(LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
                .groupby("a", "b", "big").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    _same(got, _duck("select a, b, big, sum(v), count(v) from fact join dim on k = dk group by a, b, big", fact=fact, dim=dim))
    assert len(got) > 50_000
    _fused(calls)
    _hashed(hooks, join=True)


def test_fused_join_wide_and_float_keys_with_having_order_by_and_limit(hooks, calls):
    fact, dim = _star()
    plan = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").groupby("a", "f", "nm")
            .agg(col("v").sum().alias("s")).filter(col("s") > lit(150.0)).sort("s", descending=[True]).limit(12))
    got = _rows(plan.compute(device="cpu"))
    exp = _duck("select a, f, nm, sum(v) s from fact join dim on k = dk group by a, f, nm having sum(v) > 150.0 "
                "order by s desc limit 12", fact=fact, dim=dim)
    assert len(got) == 12 == len(exp)
    for g, e in zip(got, exp):
        assert _eq(g[3], e[3])
    _fused(calls)
    _hashed(hooks, join=True)


def test_reproducible_fused_wide_key_sums_are_bit_identical(hooks, calls, monkeypatch):
    fact, dim = _star(n=900_000)
    plan = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk").groupby("a", "b", "big")
            .agg((col("v") * col("w")).sum().alias("tot"), col("v").mean().alias("av")))
    runs = []
    for threads in ("8", "8", "3", "16"):
        monkeypatch.setenv("MXFRAME_CPU_THREADS", threads)
        runs.append(plan.compute(device="cpu", reproducible=True))
    _fused(calls)
    for other in runs[1:]:
        for name in ("tot", "av"):
            assert int((_bits(runs[0], name) != _bits(other, name)).sum()) == 0, name


# ── the memory bound ─────────────────────────────────────────────────────────────────────────────

def test_a_record_budget_passed_declines_to_the_path_that_fits_with_the_same_answer(hooks, monkeypatch):
    """Flushed records grow with rows when groups are many; past MXFRAME_HASH_RECORD_MB the executor returns
    E_NO_MEMORY (-19) and the Arrow path answers."""
    r = np.random.default_rng(5)
    n = 600_000
    t = pa.table({"a": A(r.integers(-2**62, 2**62, 300_000)[r.integers(0, 300_000, n)]), "v": A(r.random(n))})
    sql = "select a, sum(v) as sv, count(*) as n from t group by a"
    monkeypatch.setenv("MXFRAME_HASH_RECORD_MB", "2")
    got = _rows(mx.sql(sql, t=t).compute(device="cpu"))
    _same(got, _duck(sql, t=t))
    assert any(h[2] and "-19" in h[2] for h in hooks), hooks
    hooks.clear()
    monkeypatch.setenv("MXFRAME_HASH_RECORD_MB", "2000")
    _same(_rows(mx.sql(sql, t=t).compute(device="cpu")), _duck(sql, t=t))
    _hashed(hooks)


@pytest.mark.parametrize("value", ["abc", "", "0", "-5", "nan"])
def test_an_unusable_record_limit_variable_is_ignored(monkeypatch, value):
    monkeypatch.delenv("MXFRAME_HASH_RECORD_MB", raising=False)
    default = nx._record_limit()
    monkeypatch.setenv("MXFRAME_HASH_RECORD_MB", value)
    assert nx._record_limit() == default > 0
    monkeypatch.setenv("MXFRAME_HASH_RECORD_MB", "3")
    assert nx._record_limit() == 3 << 20


# ── TPC-H Q10 and Q18 as SQL text ────────────────────────────────────────────────────────────────

SF = "0.17"
DATA = TPCH_DATA / f"tpch_sf{SF}_exact"
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


@pytest.mark.skipif(not (DATA / "lineitem.parquet").exists(), reason="TPC-H small data not present")
@pytest.mark.parametrize("q", [10, 18])
def test_tpch_sql_float_and_wide_group_keys_equal_duckdb_with_no_native_decline(q, hooks, monkeypatch):
    """Every aggregate hook ran and none declined, and the answer is DuckDB's. (Fully native in the coverage
    script's sense is a stricter check: SF1 Q10 still dictionary-encodes four partner string columns in Arrow.)"""
    from tpch_sql import run
    from tpch_sql.queries import SQL
    text, d = SQL[q], run.data_dir(SF)
    tables = run.load_tables(text, d)
    got = mx.sql(text, **tables).compute(device="cpu")
    assert run.compare(got, run.duck_result(text, d), text) == ""
    assert got.num_rows > 0
    live = [h for h in hooks if h[1] or h[2] != "empty input"]
    assert live and all(h[1] for h in live), live
