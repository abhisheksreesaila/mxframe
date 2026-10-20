"""F-071: a dictionary column of a NESTED partner star (Q8's supplier x nation: n_name inside a CASE) stays
int32 codes through the partner build and into the fused join aggregate, which reads it as a dictionary
(CASE, group key, predicate). Nothing runs Arrow compute over >= 1000 rows (the coverage script's
criterion), the partner is never re-encoded from plain strings, and when the fused call declines the old
path still sees strings, not codes. Every answer is compared with DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import omxcore as mx
import omxcore.native_exec as nx
from omxcore import _lazy_pc
from omxcore.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

N_FACT, N_SUP, N_NAT = 60_000, 10_000, 25
MIN_ROWS = 1000


def _tables(n_sup=N_SUP):
    r = np.random.default_rng(71)
    names = [f"NATION{i:02d}" for i in range(N_NAT)]
    nat = pa.table({"nk": A(np.arange(N_NAT, dtype=np.int64)),
                    "n_name": pa.array(names).dictionary_encode(),
                    "rk": A((np.arange(N_NAT) % 5).astype(np.int64))})
    sup = pa.table({"sk": A(np.arange(n_sup, dtype=np.int64)),
                    "nk": A(r.integers(0, N_NAT, n_sup).astype(np.int64))})
    part = pa.table({"pk": A(np.arange(500, dtype=np.int64)), "pt": A((np.arange(500) % 7).astype(np.int32)),
                      "pnk": A((np.arange(500) % N_NAT).astype(np.int64))})
    n_fact = max(N_FACT, 2 * n_sup)            # the largest leaf is the star's fact
    fact = pa.table({"sk": A(r.integers(0, n_sup, n_fact).astype(np.int64)),
                     "pk": A(r.integers(0, 500, n_fact).astype(np.int64)),
                     "v": A(np.round(r.uniform(1, 100, n_fact), 2))})
    return {"fact": fact, "sup": sup, "nat": nat, "part": part}


def _duck(sql, tables):
    con = duckdb.connect()
    for k, t in tables.items():
        con.register(k, pa.table({n: (t.column(n).cast(pa.string()) if pa.types.is_dictionary(t.column(n).type)
                                      else t.column(n)) for n in t.column_names}))
    return [tuple(r) for r in con.execute(sql).fetchall()]


@pytest.fixture
def spy(monkeypatch):
    """What ran: Arrow compute calls over >= MIN_ROWS rows, and the partner columns the fused call received."""
    log = {"arrow": [], "partner_types": [], "hooks": []}
    import pyarrow._compute as _c
    real_call = _c.call_function

    def note(name, args):
        if name == "take" and len(args) > 1:
            args = args[1:2]
        n = 0
        for a in args:
            try:
                n = max(n, len(a))
            except TypeError:
                pass
        if n >= MIN_ROWS:
            log["arrow"].append(f"{name}[{n}]")

    monkeypatch.setattr(_c, "call_function", lambda name, args, *a, **k: (note(name, args), real_call(name, args, *a, **k))[1])
    rw, rr = _lazy_pc._wrapper, _lazy_pc._real

    def logged(name, f):
        if f is None or name in _lazy_pc._REBUILT:
            return f
        return lambda *args, **kw: (note(name, args), f(*args, **kw))[1]
    monkeypatch.setattr(_lazy_pc, "_wrapper", lambda name: logged(name, rw(name)))
    monkeypatch.setattr(_lazy_pc, "_real", lambda name: logged(name, rr(name)))
    _lazy_pc.pc._cache.clear()
    real_join = nx.try_execute_join

    def join(comp, plan, fj, *a, **k):
        log["partner_types"] = [str(lk.build.schema.field(n).type) for lk in fj.lookups
                                for n in lk.build.column_names if n.endswith("n_name")]
        out = real_join(comp, plan, fj, *a, **k)
        log["hooks"].append("join:ran" if out is not None else f"join:declined({nx.last_decline()})")
        return out

    monkeypatch.setattr(nx, "try_execute_join", join)
    yield log
    _lazy_pc.pc._cache.clear()


def _eq(a, b):
    return a == pytest.approx(b, rel=1e-9) if isinstance(a, float) else a == b


def _rows(t):
    return sorted([tuple(t.column(i)[r].as_py() for i in range(t.num_columns)) for r in range(t.num_rows)], key=repr)


def _check(sql, spy, tables=None):
    tables = tables or _tables()
    exp = sorted(_duck(sql, tables), key=repr)
    got = _rows(mx.sql(sql, **tables).compute(device="cpu"))
    assert len(got) == len(exp) and all(_eq(x, y) for g, e in zip(got, exp) for x, y in zip(g, e)), (got[:3], exp[:3])
    assert spy["hooks"] == ["join:ran"], spy["hooks"]
    assert spy["arrow"] == [], spy["arrow"]
    assert spy["partner_types"] and all("dictionary" in t for t in spy["partner_types"]), spy["partner_types"]
    return got


FROM = "from fact, sup, nat, part where fact.sk = sup.sk and sup.nk = nat.nk and fact.pk = part.pk and pt < 5"


def test_nested_partner_dictionary_in_a_case(spy):
    got = _check("select pt, sum(case when n_name = 'NATION03' then v else 0 end) / sum(v) as share "
                 + FROM + " group by pt", spy)
    assert len(got) == 5 and all(0 < r[1] < 0.2 for r in got)


def test_nested_partner_dictionary_as_a_group_key(spy):
    got = _check("select n_name, sum(case when n_name = 'NATION03' then v else 0 end) as s, count(*) as c "
                 + FROM + " group by n_name", spy)
    assert len(got) == N_NAT and got[0][0] == "NATION00"


def test_nested_partner_dictionary_in_a_predicate(spy):
    got = _check("select pt, sum(v) as s " + FROM + " and (case when n_name in ('NATION01', 'NATION02') then 1 else 0 end = 1 or pt = 2) group by pt", spy)
    assert len(got) == 5


def test_nested_partner_beyond_dict_max(spy):
    # SF10's supplier: more partner ROWS than _DICT_MAX, but the dictionary is 25 values: only distinct values count
    tables = _tables(n_sup=nx._DICT_MAX + 34_464)
    got = _check("select pt, sum(case when n_name = 'NATION03' then v else 0 end) as s " + FROM + " group by pt",
                 spy, tables)
    assert len(got) == 5


def test_fused_decline_hands_strings_to_the_old_path(spy, monkeypatch):
    # an aggregate the bytecode compiler declines (a string function of the partner's column): the old path
    # must get plain strings, not codes
    tables = _tables()
    sql = ("select pt, sum(case when substr(n_name, 7, 2) = '03' then v else 0 end) as s " + FROM + " group by pt")
    exp = sorted(_duck(sql, tables), key=repr)
    monkeypatch.setenv("MXFRAME_STRICT_NATIVE", "0")
    kept = []
    real_filter = nx.try_filter

    def filt(comp, table, preds, needed, lookups=(), keep_dict=(), **kw):
        for lk in lookups:
            kept.extend(sorted(lk.keep))
        return real_filter(comp, table, preds, needed, lookups, keep_dict=keep_dict, **kw)

    monkeypatch.setattr(nx, "try_filter", filt)
    spy_join = nx.try_execute_join

    def join_then_forget(*a, **k):                       # only what runs AFTER the fused call declined counts
        out = spy_join(*a, **k)
        kept.clear()
        return out

    monkeypatch.setattr(nx, "try_execute_join", join_then_forget)
    got = _rows(mx.sql(sql, **tables).compute(device="cpu"))
    assert len(got) == len(exp) and all(_eq(x, y) for g, e in zip(got, exp) for x, y in zip(g, e))
    assert spy["hooks"], "the fused call must have been reached (and declined)"
    assert any("dictionary" in t for t in spy["partner_types"]), spy["partner_types"]   # widened before the decline
    assert all(h.startswith("join:declined") for h in spy["hooks"]), spy["hooks"]
    assert not [k for k in kept if k.endswith("n_name")], kept      # the old path's partners hold strings


def test_nested_partner_dictionary_of_a_table_joined_twice(spy):
    # Q8 joins nation twice: the second copy's n_name is renamed (n_name_right) by the join, and that is
    # the name the plan reads: the dictionary must still be deferred and kept as codes
    got = _check("select pt, sum(case when n2.n_name = 'NATION03' then v else 0 end) / sum(v) as share "
                 "from fact, sup, part, nat n1, nat n2 where fact.sk = sup.sk and sup.nk = n2.nk "
                 "and fact.pk = part.pk and part.pnk = n1.nk and n1.rk = 1 group by pt", spy)
    assert len(got) == 7
