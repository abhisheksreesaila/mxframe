"""Polars LAZY implementations: pl.scan_parquet + query + .collect() (projection / predicate
pushdown is automatic).  Same semantics as q_mx.py / q_duckdb.py."""
from __future__ import annotations

import polars as pl

from . import spec as S
from .readers import table_path


def _rev():
    return pl.col("l_extendedprice") * (1.0 - pl.col("l_discount"))


def q1(t):
    return (
        t["lineitem"].filter(pl.col("l_shipdate") <= S.Q1_CUTOFF)
        .with_columns(_rev().alias("disc_price"),
                      (_rev() * (1.0 + pl.col("l_tax"))).alias("charge"))
        .group_by("l_returnflag", "l_linestatus")
        .agg(pl.col("l_quantity").sum().alias("sum_qty"),
             pl.col("l_extendedprice").sum().alias("sum_base_price"),
             pl.col("disc_price").sum().alias("sum_disc_price"),
             pl.col("charge").sum().alias("sum_charge"),
             pl.col("l_quantity").mean().alias("avg_qty"),
             pl.col("l_extendedprice").mean().alias("avg_price"),
             pl.col("l_discount").mean().alias("avg_disc"),
             pl.col("l_quantity").count().alias("count_order"))
        .sort("l_returnflag", "l_linestatus")
    )


def q2(t):
    n, pt, s, ps = t["nation"], t["part"], t["supplier"], t["partsupp"]
    eu_nk = n.filter(pl.col("n_regionkey") == S.Q2_REGIONKEY).select("n_nationkey")
    eu = s.join(eu_nk, left_on="s_nationkey", right_on="n_nationkey", how="semi")
    tgt = pt.filter((pl.col("p_size") == 15) & pl.col("p_type").str.contains("BRASS", literal=True))
    m = (ps.join(eu, left_on="ps_suppkey", right_on="s_suppkey")
           .join(tgt, left_on="ps_partkey", right_on="p_partkey"))
    mn = m.group_by("ps_partkey").agg(pl.col("ps_supplycost").min().alias("min_supplycost"))
    return (m.join(mn, on="ps_partkey").filter(pl.col("ps_supplycost") == pl.col("min_supplycost"))
            .select(S.OUTPUT[2]).sort(["s_acctbal", "s_name"], descending=[True, False]).head(100))


def q3(t):
    c = t["customer"].filter(pl.col("c_mktsegment") == "BUILDING")
    o = t["orders"].filter(pl.col("o_orderdate") < S.Q3_DATE)
    l = t["lineitem"].filter(pl.col("l_shipdate") > S.Q3_DATE)
    return (o.join(c, left_on="o_custkey", right_on="c_custkey")
            .join(l, left_on="o_orderkey", right_on="l_orderkey")
            .group_by("o_orderkey", "o_orderdate", "o_shippriority")
            .agg(_rev().sum().alias("revenue"))
            .sort(["revenue", "o_orderdate"], descending=[True, False]).head(10))


def q4(t):
    keys = (t["lineitem"].filter(pl.col("l_commitdate") < pl.col("l_receiptdate"))
            .select("l_orderkey").unique())
    return (t["orders"].filter((pl.col("o_orderdate") >= S.Q4_LO) & (pl.col("o_orderdate") < S.Q4_HI))
            .join(keys, left_on="o_orderkey", right_on="l_orderkey", how="semi")
            .group_by("o_orderpriority").agg(pl.len().alias("order_count")).sort("o_orderpriority"))


def q5(t):
    o = t["orders"].filter((pl.col("o_orderdate") >= S.Q5_LO) & (pl.col("o_orderdate") < S.Q5_HI))
    return (t["lineitem"].join(o, left_on="l_orderkey", right_on="o_orderkey")
            .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
            .join(t["nation"], left_on="c_nationkey", right_on="n_nationkey")
            .group_by("n_name").agg(_rev().sum().alias("revenue"))
            .sort("revenue", descending=True))


def q6(t):
    return (t["lineitem"].filter(
        (pl.col("l_shipdate") >= S.Q6_LO) & (pl.col("l_shipdate") < S.Q6_HI)
        & (pl.col("l_discount") >= 0.05) & (pl.col("l_discount") <= 0.07) & (pl.col("l_quantity") < 24))
        .select((pl.col("l_extendedprice") * pl.col("l_discount")).sum().alias("revenue")))


def _year(c):
    return pl.col(c).cast(pl.Date).dt.year()


def q7(t):
    n = t["nation"]
    sn = t["supplier"].join(n.select(pl.col("n_nationkey").alias("s_nk"),
                                     pl.col("n_name").alias("supp_nation")),
                            left_on="s_nationkey", right_on="s_nk").select("s_suppkey", "supp_nation")
    cn = t["customer"].join(n.select(pl.col("n_nationkey").alias("c_nk"),
                                     pl.col("n_name").alias("cust_nation")),
                            left_on="c_nationkey", right_on="c_nk").select("c_custkey", "cust_nation")
    l = t["lineitem"].filter((pl.col("l_shipdate") >= S.Q7_LO) & (pl.col("l_shipdate") <= S.Q7_HI))
    return (l.join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .join(cn, left_on="o_custkey", right_on="c_custkey")
            .join(sn, left_on="l_suppkey", right_on="s_suppkey")
            .group_by("supp_nation", "cust_nation", _year("l_shipdate").alias("l_year"))
            .agg(_rev().sum().alias("revenue"))
            .sort("supp_nation", "cust_nation", "l_year"))


def q8(t):
    n = t["nation"].filter(pl.col("n_regionkey") == S.Q8_REGIONKEY).select("n_nationkey")
    c_am = t["customer"].join(n, left_on="c_nationkey", right_on="n_nationkey", how="semi")
    pt = t["part"].filter(pl.col("p_type") == "ECONOMY ANODIZED STEEL")
    return (t["lineitem"].join(pt, left_on="l_partkey", right_on="p_partkey")
            .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .join(c_am, left_on="o_custkey", right_on="c_custkey")
            .with_columns(_year("o_orderdate").alias("o_year"), _rev().alias("volume"))
            .with_columns(pl.when(pl.col("c_nationkey") == S.Q8_NATIONKEY)
                          .then(pl.col("volume")).otherwise(0.0).alias("brazil_vol"))
            .group_by("o_year")
            .agg(pl.col("volume").sum().alias("total_vol"), pl.col("brazil_vol").sum().alias("brazil_vol_sum"))
            .with_columns((pl.col("brazil_vol_sum") / pl.col("total_vol")).alias("mkt_share"))
            .sort("o_year"))


def q9(t):
    pt = t["part"].filter(pl.col("p_name").str.contains("green", literal=True))
    return (t["lineitem"].join(pt, left_on="l_partkey", right_on="p_partkey")
            .join(t["partsupp"], left_on=["l_partkey", "l_suppkey"], right_on=["ps_partkey", "ps_suppkey"])
            .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
            .join(t["nation"], left_on="s_nationkey", right_on="n_nationkey")
            .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .group_by("n_name", _year("o_orderdate").alias("o_year"))
            .agg((_rev() - pl.col("ps_supplycost") * pl.col("l_quantity")).sum().alias("sum_profit"))
            .sort("n_name", "o_year"))


def q10(t):
    l = t["lineitem"].filter(pl.col("l_returnflag") == "R")
    o = t["orders"].filter((pl.col("o_orderdate") >= S.Q10_LO) & (pl.col("o_orderdate") < S.Q10_HI))
    return (l.join(o, left_on="l_orderkey", right_on="o_orderkey")
            .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
            .join(t["nation"], left_on="c_nationkey", right_on="n_nationkey")
            .group_by("o_custkey", "n_name").agg(_rev().sum().alias("revenue"))
            .sort("revenue", descending=True).head(20))


def q11(t):
    ger = t["supplier"].filter(pl.col("s_nationkey") == S.Q11_NATIONKEY).select("s_suppkey")
    g = (t["partsupp"].join(ger, left_on="ps_suppkey", right_on="s_suppkey", how="semi")
         .group_by("ps_partkey")
         .agg((pl.col("ps_supplycost") * pl.col("ps_availqty")).sum().alias("value")))
    return (g.filter(pl.col("value") > pl.col("value").sum() * S.Q11_FRACTION)
            .sort("value", descending=True))


def q12(t):
    l = t["lineitem"].filter(
        pl.col("l_shipmode").is_in(["MAIL", "SHIP"]) & (pl.col("l_commitdate") < pl.col("l_receiptdate"))
        & (pl.col("l_shipdate") < pl.col("l_commitdate"))
        & (pl.col("l_receiptdate") >= S.Q12_LO) & (pl.col("l_receiptdate") < S.Q12_HI))
    hi = pl.col("o_orderpriority").is_in(["1-URGENT", "2-HIGH"])
    return (l.join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .group_by("l_shipmode")
            .agg(hi.cast(pl.Int64).sum().alias("high_line_count"),
                 (~hi).cast(pl.Int64).sum().alias("low_line_count"))
            .sort("l_shipmode"))


def q13(t):
    return (t["customer"].join(t["orders"], left_on="c_custkey", right_on="o_custkey", how="left")
            .group_by("c_custkey").agg(pl.col("o_orderkey").count().alias("c_count"))
            .group_by("c_count").agg(pl.len().alias("custdist"))
            .sort("custdist", "c_count", descending=[True, True]))


def q14(t):
    l = t["lineitem"].filter((pl.col("l_shipdate") >= S.Q14_LO) & (pl.col("l_shipdate") < S.Q14_HI))
    return (l.join(t["part"], left_on="l_partkey", right_on="p_partkey")
            .select(pl.when(pl.col("p_type").str.starts_with("PROMO")).then(_rev()).otherwise(0.0)
                    .sum().alias("promo_revenue"),
                    _rev().sum().alias("total_revenue")))


def q15(t):
    r = (t["lineitem"].filter((pl.col("l_shipdate") >= S.Q15_LO) & (pl.col("l_shipdate") < S.Q15_HI))
         .group_by("l_suppkey").agg(_rev().sum().alias("total_revenue")))
    top = r.filter(pl.col("total_revenue") == pl.col("total_revenue").max())
    return (t["supplier"].join(top, left_on="s_suppkey", right_on="l_suppkey")
            .select(S.OUTPUT[15]).sort("s_suppkey"))


def q16(t):
    bad = t["supplier"].filter(pl.col("s_comment").str.contains("Complaints", literal=True)).select("s_suppkey")
    p = t["part"].filter((pl.col("p_brand") != "Brand#45")
                         & ~pl.col("p_type").str.starts_with("MEDIUM POLISHED")
                         & pl.col("p_size").is_in(S.Q16_SIZES))
    return (t["partsupp"].join(p, left_on="ps_partkey", right_on="p_partkey")
            .join(bad, left_on="ps_suppkey", right_on="s_suppkey", how="anti")
            .group_by("p_brand", "p_type", "p_size")
            .agg(pl.col("ps_suppkey").n_unique().alias("supplier_cnt"))
            .sort("supplier_cnt", descending=True))


def q17(t):
    p = t["part"].filter((pl.col("p_brand") == "Brand#23") & (pl.col("p_container") == "MED BOX")).select("p_partkey")
    m = t["lineitem"].join(p, left_on="l_partkey", right_on="p_partkey")
    avg = m.group_by("l_partkey").agg(pl.col("l_quantity").mean().alias("avg_qty"))
    return (m.join(avg, on="l_partkey").filter(pl.col("l_quantity") < 0.2 * pl.col("avg_qty"))
            .select((pl.col("l_extendedprice").sum() / 7.0).alias("revenue")))


def q18(t):
    big = (t["lineitem"].group_by("l_orderkey").agg(pl.col("l_quantity").sum().alias("sq"))
           .filter(pl.col("sq") > 300.0).select("l_orderkey"))
    return (t["lineitem"].join(big, on="l_orderkey", how="semi")
            .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
            .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
            .group_by("o_custkey", "l_orderkey").agg(pl.col("l_quantity").sum().alias("sum_qty"))
            .sort("sum_qty", descending=True).head(100))


def q19(t):
    def cond(brand, cont, qlo, qhi, shi):
        return ((pl.col("p_brand") == brand) & pl.col("p_container").is_in(cont)
                & (pl.col("l_quantity") >= qlo) & (pl.col("l_quantity") <= qhi)
                & (pl.col("p_size") >= 1) & (pl.col("p_size") <= shi))
    return (t["lineitem"].join(t["part"], left_on="l_partkey", right_on="p_partkey")
            .filter(cond("Brand#12", S.CONTAINERS_SM, 1, 11, 5) | cond("Brand#23", S.CONTAINERS_MD, 10, 20, 10)
                    | cond("Brand#34", S.CONTAINERS_LG, 20, 30, 15))
            .select(_rev().sum().alias("revenue")))


def q20(t):
    forest = t["part"].filter(pl.col("p_name").str.contains("forest", literal=True)).select("p_partkey")
    q = (t["lineitem"].filter((pl.col("l_shipdate") >= S.Q20_LO) & (pl.col("l_shipdate") < S.Q20_HI))
         .join(forest, left_on="l_partkey", right_on="p_partkey", how="semi")
         .group_by("l_partkey", "l_suppkey").agg(pl.col("l_quantity").sum().alias("sum_qty")))
    ex = (t["partsupp"].join(forest, left_on="ps_partkey", right_on="p_partkey", how="semi")
          .join(q, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
          .filter(pl.col("ps_availqty") > 0.5 * pl.col("sum_qty")).select("ps_suppkey").unique())
    return (t["supplier"].filter(pl.col("s_nationkey") == S.Q20_NATIONKEY)
            .join(ex, left_on="s_suppkey", right_on="ps_suppkey", how="semi")
            .select("s_name", "s_address").sort("s_name"))


def q21(t):
    late = t["lineitem"].filter(pl.col("l_receiptdate") > pl.col("l_commitdate"))
    single = (late.group_by("l_orderkey").agg(pl.col("l_suppkey").n_unique().alias("n_late"))
              .filter(pl.col("n_late") == 1).select("l_orderkey"))
    f = t["orders"].filter(pl.col("o_orderstatus") == "F").select("o_orderkey")
    s = t["supplier"].filter(pl.col("s_nationkey") == S.Q21_NATIONKEY).select("s_suppkey")
    return (late.join(f, left_on="l_orderkey", right_on="o_orderkey", how="semi")
            .join(single, on="l_orderkey", how="semi")
            .join(s, left_on="l_suppkey", right_on="s_suppkey", how="semi")
            .group_by("l_suppkey").agg(pl.len().alias("numwait"))
            .sort(["numwait", "l_suppkey"], descending=[True, False]).head(100))


def q22(t):
    c = (t["customer"].with_columns(pl.col("c_phone").str.slice(0, 2).alias("cntrycode"))
         .filter(pl.col("cntrycode").is_in(S.Q22_CODES)))
    avg = pl.col("c_acctbal").filter(pl.col("c_acctbal") > 0).mean()
    return (c.filter(pl.col("c_acctbal") > avg)
            .join(t["orders"].select("o_custkey").unique(), left_on="c_custkey", right_on="o_custkey", how="anti")
            .group_by("cntrycode")
            .agg(pl.len().alias("numcust"), pl.col("c_acctbal").sum().alias("totacctbal"))
            .sort("cntrycode"))


QUERIES = {i: globals()[f"q{i}"] for i in range(1, 23)}


def run(q: int, data_dir: str, device=None):
    """Timed unit: lazy scan + query + collect (a materialized DataFrame)."""
    t = {n: pl.scan_parquet(table_path(data_dir, n)) for n in S.TABLES[q]}
    return QUERIES[q](t).collect()


def to_arrow(result):
    return result.to_arrow()
