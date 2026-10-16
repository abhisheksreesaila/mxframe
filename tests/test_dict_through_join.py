"""F-046d: a dictionary string column that the plan reads above a join only in dictionary-native
ways (a bare group key) passes through the join as codes: the native join scan writes int32 codes
of one unified dictionary (chunks with different dictionaries remapped), the group-by uses them
as they are, and nothing decodes the joined rows. A column that also reaches the output is decoded
as before. Every result is compared with DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

MODES = ["A", "N", "R", "Z"]



@pytest.fixture
def star_scan(monkeypatch):
    """These tests check the star scan's own mechanics (what it writes, which columns it keeps coded);
    an aggregate over the join would otherwise skip it (F-046a slice 3, tests/test_fused_join_agg.py)."""
    monkeypatch.setenv("MXFRAME_FUSED_JOIN_AGG", "0")

def _dict_chunks(values, n_chunks, r):
    """A dictionary column whose chunks each carry their own dictionary, in their own order."""
    cuts = np.linspace(0, len(values), n_chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vals = values[a:b]
        d = sorted(set(vals), key=lambda _: r.random())
        idx = {v: i for i, v in enumerate(d)}
        parts.append(pa.DictionaryArray.from_arrays(pa.array([idx[v] for v in vals], pa.int32()), pa.array(d)))
    return pa.chunked_array(parts)


def _tables(n=300_000, seed=2, chunks=4):
    r = np.random.default_rng(seed)
    import random
    rr = random.Random(seed)
    flag = [MODES[i] for i in r.integers(0, 4, n)]
    fact = pa.table({"k": A(r.integers(0, 5000, n).astype(np.int64)), "flag": _dict_chunks(flag, chunks, rr),
                     "v": A(r.random(n)), "d": A(r.integers(0, 100, n).astype(np.int32))})
    names = [f"name{i % 37}" for i in range(5000)]
    dim = pa.table({"dk": A(np.arange(5000, dtype=np.int64)),
                    "nm": _dict_chunks(names, 1, rr), "w": A(r.integers(0, 9, 5000).astype(np.int32))})
    return fact, dim


def _duck(sql, **t):
    con = duckdb.connect()
    for k, v in t.items():
        con.register(k, v)
    return sorted(tuple(float(x) if isinstance(x, float) else x for x in row) for row in con.execute(sql).fetchall())


def _rows(t, cols):
    d = t.select(cols).to_pydict()
    return sorted(tuple(float(x) if isinstance(x, float) else x for x in row) for row in zip(*[d[c] for c in cols]))


@pytest.fixture
def coded(monkeypatch):
    """Dictionary columns the native join scan was asked to keep as codes."""
    log = []
    real = nx.try_filter

    def spy(comp, table, preds, needed, lookups=(), keep_dict=(), **kw):
        log.extend(sorted(keep_dict))
        for lk in lookups:
            log.extend(sorted(lk.keep))
        return real(comp, table, preds, needed, lookups, keep_dict=keep_dict, **kw)

    monkeypatch.setattr(nx, "try_filter", spy)
    return log


def _approx(got, exp):
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        for a, b in zip(g, e):
            assert a == pytest.approx(b, rel=1e-9) if isinstance(a, float) else a == b


@pytest.mark.parametrize("chunks", [1, 4])
def test_fact_dictionary_group_key_through_a_star_join(star_scan, coded, chunks):
    fact, dim = _tables(chunks=chunks)
    got = (LazyFrame(fact).filter(col("d") < lit(70)).join(LazyFrame(dim).filter(col("w") > lit(2)),
                                                            left_on="k", right_on="dk")
           .groupby("flag").agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu"))
    exp = _duck("SELECT flag, SUM(v), COUNT(v) FROM fact JOIN dim ON k = dk WHERE d < 70 AND w > 2 GROUP BY flag",
                fact=fact, dim=dim)
    _approx(_rows(got, ["flag", "s", "n"]), exp)
    assert got.schema.field("flag").type == pa.string()           # the output is decoded as before
    assert "flag" in coded


def test_partner_dictionary_group_key(star_scan, coded):
    fact, dim = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .groupby("nm", "flag").agg(col("v").sum().alias("s")).compute(device="cpu"))
    exp = _duck("SELECT nm, flag, SUM(v) FROM fact JOIN dim ON k = dk GROUP BY nm, flag", fact=fact, dim=dim)
    _approx(_rows(got, ["nm", "flag", "s"]), exp)
    assert any(c.endswith("nm") for c in coded)


def test_a_column_also_in_the_output_is_decoded(coded):
    fact, dim = _tables()
    got = (LazyFrame(fact).filter(col("d") < lit(3)).join(LazyFrame(dim), left_on="k", right_on="dk")
           .select("k", "flag", "nm").compute(device="cpu"))
    exp = _duck("SELECT k, flag, nm FROM fact JOIN dim ON k = dk WHERE d < 3", fact=fact, dim=dim)
    assert _rows(got, ["k", "flag", "nm"]) == exp
    assert got.schema.field("flag").type == pa.string()
    assert "flag" not in coded


def test_leaf_by_leaf_path_keeps_codes(coded, monkeypatch):
    import mxframe.join_engine as je
    import mxframe.arrow_dict as ad
    monkeypatch.setattr(je, "_star_fact", lambda comp, fl: None)       # force the leaf-by-leaf join
    decoded = []
    real = ad.decode
    monkeypatch.setattr(ad, "decode", lambda arr: (decoded.append(len(arr)), real(arr))[1])
    fact, dim = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .groupby("flag").agg(col("w").sum().alias("s")).compute(device="cpu"))
    exp = _duck("SELECT flag, SUM(w) FROM fact JOIN dim ON k = dk GROUP BY flag", fact=fact, dim=dim)
    _approx(_rows(got, ["flag", "s"]), [tuple(float(x) if not isinstance(x, str) else x for x in r) for r in exp])
    assert not any(n >= 100_000 for n in decoded)          # the joined rows' codes were never decoded


def test_predicate_on_the_kept_column_above_the_join(coded):
    fact, dim = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .filter(col("flag").isin(["A", "Z"])).groupby("flag").agg(col("v").count().alias("n"))
           .compute(device="cpu"))
    exp = _duck("SELECT flag, COUNT(v) FROM fact JOIN dim ON k = dk WHERE flag IN ('A', 'Z') GROUP BY flag",
                fact=fact, dim=dim)
    assert _rows(got, ["flag", "n"]) == exp


def test_a_column_in_an_expression_above_the_join_is_not_kept(coded):
    from mxframe import when
    fact, dim = _tables()
    got = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
           .groupby("w").agg(when(col("flag") == lit("A"), col("v"), lit(0.0)).sum().alias("s"))
           .compute(device="cpu"))
    exp = _duck("SELECT w, SUM(CASE WHEN flag = 'A' THEN v ELSE 0 END) FROM fact JOIN dim ON k = dk GROUP BY w",
                fact=fact, dim=dim)
    _approx(_rows(got, ["w", "s"]), [tuple(float(x) for x in r) for r in exp])
    assert "flag" not in coded


@pytest.mark.parametrize("device", ["auto", "cpu"])
def test_from_parquet_and_auto_device(coded, tmp_path, device):
    import pyarrow.parquet as pq
    import mxframe as mx
    fact, dim = _tables(chunks=1)
    pq.write_table(fact.cast(pa.schema([pa.field("k", pa.int64()), pa.field("flag", pa.string()),
                                        pa.field("v", pa.float64()), pa.field("d", pa.int32())])),
                   tmp_path / "fact.parquet", row_group_size=60_000)
    pq.write_table(dim.cast(pa.schema([pa.field("dk", pa.int64()), pa.field("nm", pa.string()),
                                       pa.field("w", pa.int32())])), tmp_path / "dim.parquet")
    got = (mx.scan_parquet(str(tmp_path / "fact.parquet")).filter(col("d") < lit(50))
           .join(mx.scan_parquet(str(tmp_path / "dim.parquet")), left_on="k", right_on="dk")
           .groupby("flag", "nm").agg(col("v").sum().alias("s")).compute(device=device))
    exp = _duck("SELECT flag, nm, SUM(v) FROM fact JOIN dim ON k = dk WHERE d < 50 GROUP BY flag, nm",
                fact=fact.cast(pa.schema([pa.field("k", pa.int64()), pa.field("flag", pa.string()),
                                          pa.field("v", pa.float64()), pa.field("d", pa.int32())])),
                dim=dim.cast(pa.schema([pa.field("dk", pa.int64()), pa.field("nm", pa.string()),
                                        pa.field("w", pa.int32())])))
    _approx(_rows(got, ["flag", "nm", "s"]), exp)
    assert got.schema.field("flag").type == pa.string()


def test_join_cache_keeps_coded_and_plain_results_apart(coded):
    # same tables and columns: one plan keeps flag coded (group key), the next needs it plain
    from mxframe import when
    fact, dim = _tables()
    a = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
         .groupby("flag").agg(col("w").sum().alias("s")).compute(device="cpu"))
    b = (LazyFrame(fact).join(LazyFrame(dim), left_on="k", right_on="dk")
         .groupby("w").agg(when(col("flag") == lit("A"), col("v"), lit(0.0)).sum().alias("s"))
         .compute(device="cpu"))
    exp = _duck("SELECT w, SUM(CASE WHEN flag = 'A' THEN v ELSE 0 END) FROM fact JOIN dim ON k = dk GROUP BY w",
                fact=fact, dim=dim)
    _approx(_rows(b, ["w", "s"]), [tuple(float(x) for x in r) for r in exp])
    assert a.num_rows == 4


def test_filtered_and_nested_partners_keep_their_string_group_key_as_codes(monkeypatch):
    """F-033: a partner leaf with a filter (TPC-H Q5's `nation WHERE region = ..`) and a partner that is itself a
    star (orders -> customer -> nation) deliver the name as codes; the aggregate groups by them and no Arrow
    `dictionary_encode` runs over the partner's rows."""
    from mxframe._lazy_pc import pc
    r = np.random.default_rng(4)
    n = 200_000
    facts = pa.table({"ok": A(r.integers(0, 20_000, n).astype(np.int64)), "v": A(r.random(n))})
    orders = pa.table({"o_k": A(np.arange(20_000, dtype=np.int64)), "o_c": A(r.integers(0, 3000, 20_000).astype(np.int32))})
    cust = pa.table({"c_k": A(np.arange(3000, dtype=np.int32)), "c_n": A(r.integers(0, 25, 3000).astype(np.int32))})
    names = pa.array([f"NATION{i:02d}" for i in range(25)])
    nat = pa.table({"n_k": A(np.arange(25, dtype=np.int32)), "n_r": A((np.arange(25) // 5).astype(np.int32)),
                    "n_name": pa.DictionaryArray.from_arrays(pa.array(np.arange(25, dtype=np.int32)), names)})
    plain_nat = pa.table({"n_k": nat.column("n_k"), "n_r": nat.column("n_r"), "n_name": names})
    encoded = []
    real = pc.dictionary_encode
    monkeypatch.setitem(pc._cache, "dictionary_encode", lambda *a, **k: (encoded.append(len(a[0])), real(*a, **k))[1])
    got = (LazyFrame(facts).join(LazyFrame(orders), left_on="ok", right_on="o_k")
           .join(LazyFrame(cust), left_on="o_c", right_on="c_k")
           .join(LazyFrame(nat).filter(col("n_r") == lit(2)), left_on="c_n", right_on="n_k")
           .groupby("n_name").agg(col("v").sum().alias("s"), col("v").count().alias("n"))
           .sort("n_name").compute(device="cpu"))
    want = _duck("""SELECT n_name, SUM(v), COUNT(*) FROM facts JOIN orders ON ok = o_k JOIN cust ON o_c = c_k
                    JOIN nat ON c_n = n_k WHERE n_r = 2 GROUP BY n_name ORDER BY n_name""",
                 facts=facts, orders=orders, cust=cust, nat=plain_nat)
    _approx(_rows(got, ["n_name", "s", "n"]), want)
    assert got.num_rows == 5
    assert not [e for e in encoded if e >= 1000], encoded
