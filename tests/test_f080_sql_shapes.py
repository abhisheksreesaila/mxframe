"""F-080 / F-076: the SQL shapes that used to leave the native path, checked against DuckDB on small tables.

- Q5: a comma join whose keys come from two tables already joined (`l_suppkey = s_suppkey AND c_nationkey =
  s_nationkey`) is one key plus a filter;
- Q9: a composite key spread over leaves that the joins below equate (`ps_partkey = l_partkey` after
  `p_partkey = l_partkey`) is read from one leaf;
- Q22: `substring(c_phone, 1, 2)` in WHERE is a column computed below the Filter;
- Q7: a group key renamed by a subquery (`n1.n_name AS supp_nation`) groups like a plain column;
- Q10: plain string columns fixed by a unique key of their table are taken by partner row, not encoded
  (and a key that is NOT unique keeps the old path, still equal to DuckDB)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

N_L, N_O, N_C, N_S, N_P, N_N = 80_000, 20_000, 3_000, 200, 500, 7


def _tables(dup_customer_key=False):
    r = np.random.default_rng(80)
    nation = pa.table({"n_nationkey": A(np.arange(N_N, dtype=np.int64)),
                       "n_name": pa.array([f"NATION{i}" for i in range(N_N)])})
    ck = np.arange(N_C, dtype=np.int64)
    if dup_customer_key:
        ck[1] = ck[0]                                  # two customers share a key and differ in name
    customer = pa.table({"c_custkey": A(ck), "c_nationkey": A(r.integers(0, N_N, N_C).astype(np.int64)),
                         "c_name": pa.array([f"Customer#{i:06d}" for i in range(N_C)]),
                         "c_phone": pa.array([f"{10 + i % 25}-{i:07d}" for i in range(N_C)])})
    orders = pa.table({"o_orderkey": A(np.arange(N_O, dtype=np.int64)),
                       "o_custkey": A(r.integers(0, N_C, N_O).astype(np.int64))})
    supplier = pa.table({"s_suppkey": A(np.arange(N_S, dtype=np.int64)),
                         "s_nationkey": A(r.integers(0, N_N, N_S).astype(np.int64))})
    part = pa.table({"p_partkey": A(np.arange(N_P, dtype=np.int64)),
                     "p_name": pa.array([("green " if i % 3 == 0 else "") + f"part{i}" for i in range(N_P)])})
    ps = pa.table({"ps_partkey": A(np.repeat(np.arange(N_P, dtype=np.int64), 4)),
                   "ps_suppkey": A((np.repeat(np.arange(N_P), 4) + np.tile(np.arange(4), N_P)) % N_S),
                   "ps_cost": A(np.round(r.uniform(1, 50, N_P * 4), 2))})
    pk = r.integers(0, N_P, N_L)
    sk = (pk + r.integers(0, 4, N_L)) % N_S             # a (part, supplier) pair that exists in partsupp
    lineitem = pa.table({"l_orderkey": A(r.integers(0, N_O, N_L).astype(np.int64)),
                         "l_partkey": A(pk.astype(np.int64)), "l_suppkey": A(sk.astype(np.int64)),
                         "l_price": A(np.round(r.uniform(1, 100, N_L), 2))})
    return {"nation": nation, "customer": customer, "orders": orders, "supplier": supplier, "part": part,
            "partsupp": ps, "lineitem": lineitem}


def _duck(sql, tables):
    con = duckdb.connect()
    for k, t in tables.items():
        con.register(k, t)
    return [tuple(r) for r in con.execute(sql).fetchall()]


def _mx(sql, tables):
    out = mx.sql(sql, **tables).compute(device="cpu")
    return out, [tuple(r.values()) for r in out.to_pylist()]


def _same(got, want, ordered=True):
    assert len(got) == len(want)
    if not ordered:
        got, want = sorted(got, key=repr), sorted(want, key=repr)
    for g, w in zip(got, want):
        assert len(g) == len(w)
        for a, b in zip(g, w):
            assert a == pytest.approx(b, rel=1e-6) if isinstance(b, float) else a == b


Q5 = """select n_name, sum(l_price) as revenue
from customer, orders, lineitem, supplier, nation
where c_custkey = o_custkey and l_orderkey = o_orderkey and l_suppkey = s_suppkey
  and c_nationkey = s_nationkey and s_nationkey = n_nationkey
group by n_name order by revenue desc"""

Q9 = """select n_name as nation, sum(l_price - ps_cost) as profit
from part, supplier, lineitem, partsupp, nation
where s_suppkey = l_suppkey and ps_suppkey = l_suppkey and ps_partkey = l_partkey
  and p_partkey = l_partkey and s_nationkey = n_nationkey and p_name like '%green%'
group by n_name order by n_name"""

Q22 = """select cc, count(*) as numcust, sum(c_nationkey) as tot
from (select substring(c_phone from 1 for 2) as cc, c_nationkey from customer
      where substring(c_phone from 1 for 2) in ('13', '31', '23', '29')) as t
group by cc order by cc"""

Q7 = """select supp_nation, cust_nation, sum(volume) as revenue
from (select n1.n_name as supp_nation, n2.n_name as cust_nation, l_price as volume
      from supplier, lineitem, orders, customer, nation n1, nation n2
      where s_suppkey = l_suppkey and o_orderkey = l_orderkey and c_custkey = o_custkey
        and s_nationkey = n1.n_nationkey and c_nationkey = n2.n_nationkey
        and ((n1.n_name = 'NATION1' and n2.n_name = 'NATION2') or (n1.n_name = 'NATION2' and n2.n_name = 'NATION1'))
     ) as shipping
group by supp_nation, cust_nation order by supp_nation, cust_nation"""

Q10 = """select c_custkey, c_name, sum(l_price) as revenue, n_name
from customer, orders, lineitem, nation
where c_custkey = o_custkey and l_orderkey = o_orderkey and c_nationkey = n_nationkey
group by c_custkey, c_name, n_name order by revenue desc limit 20"""


@pytest.mark.parametrize("name,sql", [("q5", Q5), ("q9", Q9), ("q22", Q22), ("q7", Q7)])
def test_shape_equals_duckdb(name, sql):
    t = _tables()
    out, got = _mx(sql, t)
    want = _duck(sql, t)
    assert want, "the query must return rows"
    _same(got, want)
    assert out.column_names == {"q5": ["n_name", "revenue"], "q9": ["nation", "profit"],
                                "q22": ["cc", "numcust", "tot"],
                                "q7": ["supp_nation", "cust_nation", "revenue"]}[name]


@pytest.fixture
def spy(monkeypatch):
    """Which native routes ran: the dependent-key route and the declines of the fused call."""
    log = {"dependent": 0, "declines": []}
    real = getattr(nx, "_try_dependent_keys", lambda *a, **k: None)

    def dep(*a, **k):
        r = real(*a, **k)
        log["dependent"] += r is not None
        return r
    monkeypatch.setattr(nx, "_try_dependent_keys", dep, raising=False)
    real_join = nx.try_execute_join

    def join(*a, **k):
        r = real_join(*a, **k)
        if r is None:
            log["declines"].append(nx.last_decline())
        return r
    monkeypatch.setattr(nx, "try_execute_join", join)
    return log


@pytest.mark.parametrize("name,sql", [("q5", Q5), ("q9", Q9), ("q7", Q7)])
def test_shape_stays_in_the_fused_join(spy, name, sql):
    _mx(sql, _tables())
    assert spy["declines"] == []


def test_strings_fixed_by_a_unique_key_are_taken_by_partner_row(spy):
    t = _tables()
    _, got = _mx(Q10, t)
    _same(got, _duck(Q10, t))
    assert spy["dependent"] == 1 and spy["declines"] == []


def test_a_key_that_is_not_unique_keeps_every_string_in_the_group(spy):
    t = _tables(dup_customer_key=True)
    _, got = _mx(Q10, t)
    want = _duck(Q10, t)
    assert {r[1] for r in got if r[0] == 0} <= {"Customer#000000", "Customer#000001"}
    _same(got, want)
    assert spy["dependent"] == 0


# ---- Q13: COUNT(right column) per left key over a LEFT JOIN -------------------------------------

Q13 = """select c_count, count(*) as custdist
from (select c_custkey, count(o_orderkey) as c_count
      from customer left outer join orders on c_custkey = o_custkey and o_orderkey % 5 <> 0
      group by c_custkey) as c_orders
group by c_count order by custdist desc, c_count desc"""


@pytest.fixture
def counted(monkeypatch):
    """How many times the left-join counting route answered."""
    import mxframe.join_engine as je
    log = {"ran": 0}
    real = getattr(je, "left_join_counts", lambda *a, **k: None)

    def wrapped(*a, **k):
        r = real(*a, **k)
        log["ran"] += r is not None
        return r
    monkeypatch.setattr(je, "left_join_counts", wrapped, raising=False)
    return log


def test_left_join_counts_equal_duckdb_with_customers_that_have_no_orders(counted):
    t = _tables()
    _, got = _mx(Q13, t)
    want = _duck(Q13, t)
    assert any(r[0] == 0 for r in want), "the data must have customers without orders"
    _same(got, want)
    assert counted["ran"] == 1


def test_left_join_counts_with_a_duplicate_left_key_keep_the_old_path(counted):
    t = _tables(dup_customer_key=True)
    _, got = _mx(Q13, t)
    _same(got, _duck(Q13, t))
    assert counted["ran"] == 0


def test_left_join_counts_skip_nulls_of_the_counted_column(counted):
    t = _tables()
    vals = t["orders"].column("o_orderkey").to_pylist()
    t["orders"] = t["orders"].set_column(0, "o_orderkey", pa.array([None if v % 3 == 0 else v for v in vals], type=pa.int64()))
    _, got = _mx(Q13, t)
    _same(got, _duck(Q13, t))


def test_left_join_count_per_key_result_columns():
    t = _tables()
    sql = ("select c_custkey, count(o_orderkey) as n from customer left join orders on c_custkey = o_custkey "
           "group by c_custkey order by c_custkey")
    out, got = _mx(sql, t)
    assert out.column_names == ["c_custkey", "n"]
    assert str(out.schema.field("n").type) == "int64"
    _same(got, _duck(sql, t))


# ---- review fixes ----------------------------------------------------------------------------

def test_group_keys_from_another_partner_next_to_dependent_strings(spy):
    sql = ("select c_custkey, c_name, s_nationkey, sum(l_price) as revenue "
           "from customer, orders, lineitem, supplier "
           "where c_custkey = o_custkey and o_orderkey = l_orderkey and l_suppkey = s_suppkey "
           "group by c_custkey, c_name, s_nationkey order by revenue desc limit 20")
    t = _tables()
    _, got = _mx(sql, t)
    _same(got, _duck(sql, t))


@pytest.mark.parametrize("sql", [
    "select ps_partkey from partsupp where ps_suppkey not in (select s_suppkey as x from supplier where s_nationkey = 1) order by 1, ps_suppkey",
    "select ps_partkey from partsupp where ps_suppkey in (select S_SUPPKEY from supplier where s_nationkey = 1) order by 1, ps_suppkey",
    "select ps_partkey from partsupp where ps_suppkey not in (select S_SUPPKEY from supplier where s_nationkey = 1) order by 1, ps_suppkey",
    "select ps_partkey from partsupp where ps_suppkey in (select s_suppkey as x from supplier where s_nationkey = 1) order by 1, ps_suppkey",
])
def test_in_subquery_item_spelling_and_alias(sql):
    t = _tables()
    _, got = _mx(sql, t)
    _same(got, _duck(sql, t))


@pytest.mark.parametrize("name,sql,mod", [
    ("right where", "select c_custkey, count(o_orderkey) as n from customer left join orders on c_custkey = o_custkey "
                    "where o_orderkey % 2 = 0 group by c_custkey order by c_custkey", None),
    ("inner", "select c_custkey, count(o_orderkey) as n from customer join orders on c_custkey = o_custkey "
              "group by c_custkey order by c_custkey", None),
    ("count star", "select c_custkey, count(*) as n from customer left join orders on c_custkey = o_custkey "
                   "group by c_custkey order by c_custkey", None),
    ("with sum", "select c_custkey, count(o_orderkey) as n, sum(o_orderkey) as s from customer left join orders "
                 "on c_custkey = o_custkey group by c_custkey order by c_custkey", None),
    ("null left key", "select c_custkey, count(o_orderkey) as n from customer left join orders on c_custkey = o_custkey "
                      "group by c_custkey order by c_custkey", "null_key"),
])
def test_left_join_count_declines_keep_the_answer(counted, name, sql, mod):
    t = _tables()
    if mod == "null_key":
        k = t["customer"].column("c_custkey").to_pylist()
        k[5] = None
        t["customer"] = t["customer"].set_column(0, "c_custkey", pa.array(k, type=pa.int64()))
    _, got = _mx(sql, t)
    _same(got, _duck(sql, t))
    assert counted["ran"] == 0
