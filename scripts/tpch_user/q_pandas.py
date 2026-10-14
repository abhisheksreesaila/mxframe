"""pandas implementations: what a knowledgeable pandas user writes (pd.read_parquet with
`columns=`, filter early, merge, groupby).  Same semantics as q_mx.py / q_duckdb.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pyarrow as pa

from . import spec as S
from .readers import pandas_read, table_path


def _year(days: pd.Series) -> np.ndarray:
    return days.to_numpy().astype("datetime64[D]").astype("datetime64[Y]").astype(np.int64) + 1970


def _df(**cols) -> pd.DataFrame:
    return pd.DataFrame(cols)


def q1(t):
    li = t["lineitem"]
    f = li[li.l_shipdate <= S.Q1_CUTOFF].copy()
    f["disc_price"] = f.l_extendedprice * (1.0 - f.l_discount)
    f["charge"] = f.disc_price * (1.0 + f.l_tax)
    return (f.groupby(["l_returnflag", "l_linestatus"], as_index=False)
            .agg(sum_qty=("l_quantity", "sum"), sum_base_price=("l_extendedprice", "sum"),
                 sum_disc_price=("disc_price", "sum"), sum_charge=("charge", "sum"),
                 avg_qty=("l_quantity", "mean"), avg_price=("l_extendedprice", "mean"),
                 avg_disc=("l_discount", "mean"), count_order=("l_quantity", "count"))
            .sort_values(["l_returnflag", "l_linestatus"]).reset_index(drop=True))


def q2(t):
    n, pt, s, ps = t["nation"], t["part"], t["supplier"], t["partsupp"]
    eu = s.merge(n[n.n_regionkey == S.Q2_REGIONKEY][["n_nationkey", "n_name"]],
                 left_on="s_nationkey", right_on="n_nationkey")
    tgt = pt[(pt.p_size == 15) & pt.p_type.str.contains("BRASS", regex=False)]
    m = (ps.merge(eu, left_on="ps_suppkey", right_on="s_suppkey")
           .merge(tgt, left_on="ps_partkey", right_on="p_partkey"))
    mn = m.groupby("ps_partkey")["ps_supplycost"].min().rename("min_supplycost").reset_index()
    r = m.merge(mn, on="ps_partkey")
    r = r[r.ps_supplycost == r.min_supplycost]
    return (r[S.OUTPUT[2]].sort_values(["s_acctbal", "n_name", "s_name", "ps_partkey"],
                                       ascending=[False, True, True, True])
            .head(100).reset_index(drop=True))


def q3(t):
    c, o, l = t["customer"], t["orders"], t["lineitem"]
    c = c[c.c_mktsegment == "BUILDING"]
    o = o[o.o_orderdate < S.Q3_DATE]
    l = l[l.l_shipdate > S.Q3_DATE]
    m = o.merge(c, left_on="o_custkey", right_on="c_custkey").merge(
        l, left_on="o_orderkey", right_on="l_orderkey")
    m["revenue"] = m.l_extendedprice * (1.0 - m.l_discount)
    return (m.groupby(["o_orderkey", "o_orderdate", "o_shippriority"], as_index=False)["revenue"].sum()
            .sort_values(["revenue", "o_orderdate", "o_orderkey"], ascending=[False, True, True]).head(10)
            .reset_index(drop=True))


def q4(t):
    o, l = t["orders"], t["lineitem"]
    keys = l.loc[l.l_commitdate < l.l_receiptdate, "l_orderkey"].unique()
    f = o[(o.o_orderdate >= S.Q4_LO) & (o.o_orderdate < S.Q4_HI) & o.o_orderkey.isin(keys)]
    return (f.groupby("o_orderpriority").size().reset_index(name="order_count")
            .sort_values("o_orderpriority").reset_index(drop=True))


def q5(t):
    n, s, c, o, l = t["nation"], t["supplier"], t["customer"], t["orders"], t["lineitem"]
    o = o[(o.o_orderdate >= S.Q5_LO) & (o.o_orderdate < S.Q5_HI)]
    m = (l.merge(o, left_on="l_orderkey", right_on="o_orderkey")
          .merge(c, left_on="o_custkey", right_on="c_custkey")
          .merge(s, left_on="l_suppkey", right_on="s_suppkey"))
    m = m[m.c_nationkey == m.s_nationkey].merge(
        n[n.n_regionkey == S.Q5_REGIONKEY], left_on="c_nationkey", right_on="n_nationkey")
    m["revenue"] = m.l_extendedprice * (1.0 - m.l_discount)
    return m.groupby("n_name", as_index=False)["revenue"].sum().sort_values(
        "revenue", ascending=False).reset_index(drop=True)


def q6(t):
    l = t["lineitem"]
    f = l[(l.l_shipdate >= S.Q6_LO) & (l.l_shipdate < S.Q6_HI) & (l.l_discount >= 0.05)
          & (l.l_discount <= 0.07) & (l.l_quantity < 24)]
    return _df(revenue=[float((f.l_extendedprice * f.l_discount).sum())])


def q7(t):
    n, s, c, o, l = t["nation"], t["supplier"], t["customer"], t["orders"], t["lineitem"]
    l = l[(l.l_shipdate >= S.Q7_LO) & (l.l_shipdate <= S.Q7_HI)]
    sn = s.merge(n.rename(columns={"n_nationkey": "s_nk", "n_name": "supp_nation"}),
                 left_on="s_nationkey", right_on="s_nk")[["s_suppkey", "supp_nation"]]
    cn = c.merge(n.rename(columns={"n_nationkey": "c_nk", "n_name": "cust_nation"}),
                 left_on="c_nationkey", right_on="c_nk")[["c_custkey", "cust_nation"]]
    m = (l.merge(o, left_on="l_orderkey", right_on="o_orderkey")
          .merge(cn, left_on="o_custkey", right_on="c_custkey")
          .merge(sn, left_on="l_suppkey", right_on="s_suppkey"))
    a, b = S.Q7_NATIONS
    m = m[((m.supp_nation == a) & (m.cust_nation == b)) | ((m.supp_nation == b) & (m.cust_nation == a))].copy()
    m["l_year"] = _year(m.l_shipdate)
    m["revenue"] = m.l_extendedprice * (1.0 - m.l_discount)
    return (m.groupby(["supp_nation", "cust_nation", "l_year"], as_index=False)["revenue"].sum()
            .sort_values(["supp_nation", "cust_nation", "l_year"]).reset_index(drop=True))


def q8(t):
    n, pt, s, c, o, l = t["nation"], t["part"], t["supplier"], t["customer"], t["orders"], t["lineitem"]
    c_am = c[c.c_nationkey.isin(n[n.n_regionkey == S.Q8_REGIONKEY].n_nationkey)]
    pt = pt[pt.p_type == "ECONOMY ANODIZED STEEL"]
    o = o[(o.o_orderdate >= S.Q8_LO) & (o.o_orderdate <= S.Q8_HI)]
    m = (l.merge(pt, left_on="l_partkey", right_on="p_partkey")
          .merge(s, left_on="l_suppkey", right_on="s_suppkey")
          .merge(o, left_on="l_orderkey", right_on="o_orderkey")
          .merge(c_am, left_on="o_custkey", right_on="c_custkey"))
    m["o_year"] = _year(m.o_orderdate)
    m["total_vol"] = m.l_extendedprice * (1.0 - m.l_discount)
    m["brazil_vol_sum"] = m.total_vol.where(m.s_nationkey == S.Q8_NATIONKEY, 0.0)
    r = m.groupby("o_year", as_index=False)[["total_vol", "brazil_vol_sum"]].sum()
    r["mkt_share"] = r.brazil_vol_sum / r.total_vol
    return r.sort_values("o_year").reset_index(drop=True)


def q9(t):
    n, s, ps, pt, o, l = t["nation"], t["supplier"], t["partsupp"], t["part"], t["orders"], t["lineitem"]
    pt = pt[pt.p_name.str.contains("green", regex=False)]
    m = (l.merge(pt, left_on="l_partkey", right_on="p_partkey")
          .merge(ps, left_on=["l_partkey", "l_suppkey"], right_on=["ps_partkey", "ps_suppkey"])
          .merge(s, left_on="l_suppkey", right_on="s_suppkey")
          .merge(n, left_on="s_nationkey", right_on="n_nationkey")
          .merge(o, left_on="l_orderkey", right_on="o_orderkey"))
    m["o_year"] = _year(m.o_orderdate)
    m["profit"] = m.l_extendedprice * (1.0 - m.l_discount) - m.ps_supplycost * m.l_quantity
    return (m.groupby(["n_name", "o_year"], as_index=False)["profit"].sum()
            .rename(columns={"profit": "sum_profit"}).sort_values(["n_name", "o_year"], ascending=[True, False])
            .reset_index(drop=True))


def q10(t):
    n, c, o, l = t["nation"], t["customer"], t["orders"], t["lineitem"]
    l = l[l.l_returnflag == "R"]
    o = o[(o.o_orderdate >= S.Q10_LO) & (o.o_orderdate < S.Q10_HI)]
    m = (l.merge(o, left_on="l_orderkey", right_on="o_orderkey")
          .merge(c, left_on="o_custkey", right_on="c_custkey")
          .merge(n, left_on="c_nationkey", right_on="n_nationkey"))
    m["revenue"] = m.l_extendedprice * (1.0 - m.l_discount)
    return (m.groupby(["o_custkey", "n_name"], as_index=False)["revenue"].sum()
            .sort_values(["revenue", "o_custkey"], ascending=[False, True]).head(20).reset_index(drop=True))


def q11(t):
    s, ps = t["supplier"], t["partsupp"]
    ger = s.loc[s.s_nationkey == S.Q11_NATIONKEY, "s_suppkey"]
    f = ps[ps.ps_suppkey.isin(ger)].copy()
    f["value"] = f.ps_supplycost * f.ps_availqty
    thr = f["value"].sum() * S.q11_fraction()
    g = f.groupby("ps_partkey", as_index=False)["value"].sum()
    return g[g.value > thr].sort_values("value", ascending=False).reset_index(drop=True)


def q12(t):
    o, l = t["orders"], t["lineitem"]
    f = l[l.l_shipmode.isin(["MAIL", "SHIP"]) & (l.l_commitdate < l.l_receiptdate)
          & (l.l_shipdate < l.l_commitdate) & (l.l_receiptdate >= S.Q12_LO)
          & (l.l_receiptdate < S.Q12_HI)]
    m = f.merge(o, left_on="l_orderkey", right_on="o_orderkey")
    hi = m.o_orderpriority.isin(["1-URGENT", "2-HIGH"])
    m = m.assign(high_line_count=hi.astype("int64"), low_line_count=(~hi).astype("int64"))
    return (m.groupby("l_shipmode", as_index=False)[["high_line_count", "low_line_count"]].sum()
            .sort_values("l_shipmode").reset_index(drop=True))


def q13(t):
    c, o = t["customer"], t["orders"]
    m = c.merge(o, left_on="c_custkey", right_on="o_custkey", how="left")
    cc = m.groupby("c_custkey", as_index=False)["o_orderkey"].count().rename(
        columns={"o_orderkey": "c_count"})
    return (cc.groupby("c_count", as_index=False)["c_custkey"].count()
            .rename(columns={"c_custkey": "custdist"})
            .sort_values(["custdist", "c_count"], ascending=[False, False]).reset_index(drop=True))


def q14(t):
    pt, l = t["part"], t["lineitem"]
    l = l[(l.l_shipdate >= S.Q14_LO) & (l.l_shipdate < S.Q14_HI)]
    m = l.merge(pt, left_on="l_partkey", right_on="p_partkey")
    rev = m.l_extendedprice * (1.0 - m.l_discount)
    promo = rev.where(m.p_type.str.startswith("PROMO"), 0.0)
    return _df(promo_revenue=[100.0 * float(promo.sum()) / float(rev.sum())])


def q15(t):
    s, l = t["supplier"], t["lineitem"]
    f = l[(l.l_shipdate >= S.Q15_LO) & (l.l_shipdate < S.Q15_HI)]
    rev = (f.l_extendedprice * (1.0 - f.l_discount)).groupby(f.l_suppkey).sum().rename(
        "total_revenue").reset_index()
    top = rev[rev.total_revenue == rev.total_revenue.max()]
    r = s.merge(top, left_on="s_suppkey", right_on="l_suppkey")
    return r[S.OUTPUT[15]].sort_values("s_suppkey").reset_index(drop=True)


def q16(t):
    p, s, ps = t["part"], t["supplier"], t["partsupp"]
    bad = s.loc[s.s_comment.str.contains("Complaints", regex=False), "s_suppkey"]
    pf = p[(p.p_brand != "Brand#45") & ~p.p_type.str.startswith("MEDIUM POLISHED")
           & p.p_size.isin(S.Q16_SIZES)]
    j = ps.merge(pf, left_on="ps_partkey", right_on="p_partkey")
    j = j[~j.ps_suppkey.isin(bad)]
    d = j[["p_brand", "p_type", "p_size", "ps_suppkey"]].drop_duplicates()
    return (d.groupby(["p_brand", "p_type", "p_size"]).size().reset_index(name="supplier_cnt")
            .sort_values(["supplier_cnt", "p_brand", "p_type", "p_size"], ascending=[False, True, True, True])
            .reset_index(drop=True))


def q17(t):
    p, l = t["part"], t["lineitem"]
    pf = p.loc[(p.p_brand == "Brand#23") & (p.p_container == "MED BOX"), ["p_partkey"]]
    m = l.merge(pf, left_on="l_partkey", right_on="p_partkey")
    avg = m.groupby("l_partkey")["l_quantity"].mean().rename("avg_qty").reset_index()
    m2 = m.merge(avg, on="l_partkey")
    return _df(revenue=[float(m2.loc[m2.l_quantity < 0.2 * m2.avg_qty, "l_extendedprice"].sum()) / 7.0])


def q18(t):
    c, o, l = t["customer"], t["orders"], t["lineitem"]
    q = l.groupby("l_orderkey")["l_quantity"].sum()
    big = q.index[q > 300.0]
    lf = l[l.l_orderkey.isin(big)]
    m = (lf.merge(o, left_on="l_orderkey", right_on="o_orderkey")
           .merge(c, left_on="o_custkey", right_on="c_custkey"))
    return (m.groupby(["o_custkey", "l_orderkey"], as_index=False)["l_quantity"].sum()
            .rename(columns={"l_quantity": "sum_qty"})
            .sort_values(["sum_qty", "l_orderkey"], ascending=[False, True]).head(100).reset_index(drop=True))


def q19(t):
    p, l = t["part"], t["lineitem"]
    m = l.merge(p, left_on="l_partkey", right_on="p_partkey")
    cond = (
        ((m.p_brand == "Brand#12") & m.p_container.isin(S.CONTAINERS_SM)
         & (m.l_quantity >= 1) & (m.l_quantity <= 11) & (m.p_size >= 1) & (m.p_size <= 5))
        | ((m.p_brand == "Brand#23") & m.p_container.isin(S.CONTAINERS_MD)
           & (m.l_quantity >= 10) & (m.l_quantity <= 20) & (m.p_size >= 1) & (m.p_size <= 10))
        | ((m.p_brand == "Brand#34") & m.p_container.isin(S.CONTAINERS_LG)
           & (m.l_quantity >= 20) & (m.l_quantity <= 30) & (m.p_size >= 1) & (m.p_size <= 15)))
    f = m[cond & m.l_shipmode.isin(["AIR", "AIR REG"]) & (m.l_shipinstruct == "DELIVER IN PERSON")]
    return _df(revenue=[float((f.l_extendedprice * (1.0 - f.l_discount)).sum())])


def q20(t):
    p, s, ps, l = t["part"], t["supplier"], t["partsupp"], t["lineitem"]
    forest = p.loc[p.p_name.str.startswith("forest"), "p_partkey"]
    lf = l[(l.l_shipdate >= S.Q20_LO) & (l.l_shipdate < S.Q20_HI) & l.l_partkey.isin(forest)]
    q = lf.groupby(["l_partkey", "l_suppkey"], as_index=False)["l_quantity"].sum().rename(
        columns={"l_quantity": "sum_qty"})
    m = ps[ps.ps_partkey.isin(forest)].merge(
        q, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
    ex = m.loc[m.ps_availqty > 0.5 * m.sum_qty, "ps_suppkey"].unique()
    r = s[(s.s_nationkey == S.Q20_NATIONKEY) & s.s_suppkey.isin(ex)]
    return r[["s_name", "s_address"]].sort_values("s_name").reset_index(drop=True)


def q21(t):
    s, o, l = t["supplier"], t["orders"], t["lineitem"]
    saudi = s.loc[s.s_nationkey == S.Q21_NATIONKEY, "s_suppkey"]
    fkeys = o.loc[o.o_orderstatus == "F", "o_orderkey"]
    ns = l.groupby("l_orderkey")["l_suppkey"].nunique()
    multi = ns.index[ns >= 2]
    late = l[l.l_receiptdate > l.l_commitdate]
    nl = late.groupby("l_orderkey")["l_suppkey"].nunique()
    single = nl.index[nl == 1]
    q = late[late.l_orderkey.isin(fkeys) & late.l_orderkey.isin(single) & late.l_orderkey.isin(multi)
             & late.l_suppkey.isin(saudi)]
    return (q.groupby("l_suppkey").size().reset_index(name="numwait")
            .sort_values(["numwait", "l_suppkey"], ascending=[False, True]).head(100)
            .reset_index(drop=True))


def q22(t):
    c, o = t["customer"], t["orders"]
    cc = c.c_phone.str.slice(0, 2)
    tgt = c[cc.isin(S.Q22_CODES)].assign(cntrycode=cc[cc.isin(S.Q22_CODES)])
    avg = tgt.loc[tgt.c_acctbal > 0, "c_acctbal"].mean()
    tgt = tgt[~tgt.c_custkey.isin(o.o_custkey) & (tgt.c_acctbal > avg)]
    return (tgt.groupby("cntrycode", as_index=False)
            .agg(numcust=("c_custkey", "count"), totacctbal=("c_acctbal", "sum"))
            .sort_values("cntrycode").reset_index(drop=True))


QUERIES = {i: globals()[f"q{i}"] for i in range(1, 23)}


def run(q: int, data_dir: str, device=None):
    """Timed unit: Parquet read + query + materialized DataFrame."""
    S.set_scale_from(data_dir)
    t = {n: pandas_read(table_path(data_dir, n), cols) for n, cols in S.TABLES[q].items()}
    return QUERIES[q](t)


def load_mem(q: int, data_dir: str) -> dict:
    """Engine-only mode, untimed: the columns the query uses, as DataFrames."""
    S.set_scale_from(data_dir)
    return {n: pandas_read(table_path(data_dir, n), cols) for n, cols in S.TABLES[q].items()}


def run_mem(q: int, tables: dict, device=None):
    """Engine-only timed unit: query + materialized DataFrame on loaded tables."""
    return QUERIES[q](tables)


def to_arrow(result) -> pa.Table:
    return pa.Table.from_pandas(result, preserve_index=False)
