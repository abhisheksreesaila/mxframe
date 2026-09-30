"""MXFrame implementations of the 22 queries, fed by Parquet -> PyArrow tables.

Adapted from scripts/benchmark_tpch.py so they are correct on the REAL dbgen data
(int32 days-since-epoch dates, int64 keys).  Only the public LazyFrame API is used (the
benchmark script's hand-written direct-GPU-kernel shortcuts are NOT used, since a user
cannot call those).  Anything the LazyFrame layer cannot express is done with PyArrow
inside the timed region, exactly like a user would.  No caches, no id()-keyed state.

`.year()` on an int32 column assumes YYYYMMDD in this engine, so the year of a
days-since-epoch date is computed with PyArrow (`_year`).
"""
from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from mxframe import LazyFrame, col, lit, when
from mxframe.lazy_frame import Scan

from . import spec as S
from .readers import mx_read, table_path


def _lf(t: pa.Table) -> LazyFrame:
    return LazyFrame(Scan(t))


def _year(arr) -> pa.ChunkedArray:
    return pc.cast(pc.year(pc.cast(arr, pa.date32())), pa.int32())


def _with_nation(tbl: pa.Table, key: str, nation: pa.Table, name: str) -> pa.Table:
    idx = pc.index_in(tbl[key], value_set=nation["n_nationkey"])
    return tbl.append_column(name, pc.take(nation["n_name"], idx))


def q1(t, dev):
    return (
        _lf(t["lineitem"])
        .filter(col("l_shipdate") <= lit(S.Q1_CUTOFF))
        .groupby("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("sum_disc_price"),
            (col("l_extendedprice") * (lit(1.0) - col("l_discount")) * (lit(1.0) + col("l_tax")))
            .sum().alias("sum_charge"),
            col("l_quantity").mean().alias("avg_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_discount").mean().alias("avg_disc"),
            col("l_quantity").count().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .compute(device=dev)
    )


def q2(t, dev):
    nation, part, supplier, partsupp = t["nation"], t["part"], t["supplier"], t["partsupp"]
    eu = nation.filter(pc.equal(nation["n_regionkey"], S.Q2_REGIONKEY))["n_nationkey"]
    eu_sup = supplier.filter(pc.is_in(supplier["s_nationkey"], value_set=eu.combine_chunks()))
    target = part.filter(pc.and_(pc.equal(part["p_size"], 15),
                                 pc.match_substring(part["p_type"], pattern="BRASS")))
    cand = (partsupp.join(eu_sup, keys="ps_suppkey", right_keys="s_suppkey", join_type="inner")
            .join(target.select(["p_partkey", "p_mfgr"]), keys="ps_partkey", right_keys="p_partkey", join_type="inner"))
    if cand.num_rows == 0:
        return cand.select(S.OUTPUT[2])
    min_costs = (_lf(cand).groupby("ps_partkey")
                 .agg(col("ps_supplycost").min().alias("min_supplycost")).compute(device=dev))
    return (
        _lf(cand)
        .join(_lf(min_costs), left_on="ps_partkey", right_on="ps_partkey")
        .filter(col("ps_supplycost") == col("min_supplycost"))
        .select("s_acctbal", "s_name", "ps_partkey", "p_mfgr", "s_address", "s_phone", "s_comment")
        .sort(col("s_acctbal"), descending=True)
        .limit(100)
        .compute(device=dev)
    )


def q3(t, dev):
    lf_c = _lf(t["customer"]).filter(col("c_mktsegment") == lit("BUILDING"))
    lf_o = _lf(t["orders"]).filter(col("o_orderdate") < lit(S.Q3_DATE))
    lf_l = _lf(t["lineitem"]).filter(col("l_shipdate") > lit(S.Q3_DATE))
    return (
        lf_o.join(lf_c, left_on="o_custkey", right_on="c_custkey")
        .join(lf_l, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderkey", "o_orderdate", "o_shippriority")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort("revenue", descending=True)
        .limit(10)
        .compute(device=dev)
    )


def q4(t, dev):
    orders, lineitem = t["orders"], t["lineitem"]
    late = lineitem.filter(pc.less(lineitem["l_commitdate"], lineitem["l_receiptdate"]))
    keys = pa.table({"j_orderkey": pc.unique(late["l_orderkey"])})
    return (
        _lf(orders)
        .filter((col("o_orderdate") >= lit(S.Q4_LO)) & (col("o_orderdate") < lit(S.Q4_HI)))
        .join(_lf(keys), left_on="o_orderkey", right_on="j_orderkey")
        .groupby("o_orderpriority")
        .agg(col("o_orderpriority").count().alias("order_count"))
        .sort(col("o_orderpriority"))
        .compute(device=dev)
    )


def q5(t, dev):
    return (
        _lf(t["lineitem"])
        .join(_lf(t["orders"]).filter((col("o_orderdate") >= lit(S.Q5_LO)) &
                                      (col("o_orderdate") < lit(S.Q5_HI))),
              left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(t["customer"]), left_on="o_custkey", right_on="c_custkey")
        .join(_lf(t["nation"]), left_on="c_nationkey", right_on="n_nationkey")
        .groupby("n_name")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort(col("revenue"), descending=True)
        .compute(device=dev)
    )


def q6(t, dev):
    return (
        _lf(t["lineitem"])
        .filter((col("l_shipdate") >= lit(S.Q6_LO)) & (col("l_shipdate") < lit(S.Q6_HI))
                & (col("l_discount") >= lit(0.05)) & (col("l_discount") <= lit(0.07))
                & (col("l_quantity") < lit(24.0)))
        .groupby()
        .agg((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        .compute(device=dev)
    )


def q7(t, dev):
    li = t["lineitem"]
    li = li.filter(pc.and_(pc.greater_equal(li["l_shipdate"], S.Q7_LO),
                           pc.less_equal(li["l_shipdate"], S.Q7_HI)))
    li = li.append_column("l_year", _year(li["l_shipdate"]))
    sup = _with_nation(t["supplier"], "s_nationkey", t["nation"], "supp_nation")
    cust = _with_nation(t["customer"], "c_nationkey", t["nation"], "cust_nation")
    return (
        _lf(li)
        .join(_lf(t["orders"]), left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(cust), left_on="o_custkey", right_on="c_custkey")
        .join(_lf(sup), left_on="l_suppkey", right_on="s_suppkey")
        .groupby("supp_nation", "cust_nation", "l_year")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort(col("supp_nation"))
        .compute(device=dev)
    )


def q8(t, dev):
    nation, customer, orders = t["nation"], t["customer"], t["orders"]
    am = nation.filter(pc.equal(nation["n_regionkey"], S.Q8_REGIONKEY))["n_nationkey"].combine_chunks()
    cust_am = customer.filter(pc.is_in(customer["c_nationkey"], value_set=am))
    orders = orders.append_column("o_year", _year(orders["o_orderdate"]))
    rev = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
    result = (
        _lf(t["lineitem"])
        .join(_lf(t["part"]).filter(col("p_type") == lit("ECONOMY ANODIZED STEEL")),
              left_on="l_partkey", right_on="p_partkey")
        .join(_lf(orders), left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(cust_am), left_on="o_custkey", right_on="c_custkey")
        .with_columns(
            rev.alias("volume"),
            when(col("c_nationkey") == lit(S.Q8_NATIONKEY), rev, lit(0.0)).alias("brazil_vol"),
        )
        .groupby("o_year")
        .agg(col("volume").sum().alias("total_vol"), col("brazil_vol").sum().alias("brazil_vol_sum"))
        .sort(col("o_year"))
        .compute(device=dev)
    )
    tv = result["total_vol"].to_pylist()
    bv = result["brazil_vol_sum"].to_pylist()
    return result.append_column("mkt_share", pa.array([b / v if v else 0.0 for b, v in zip(bv, tv)],
                                                      type=pa.float64()))


def q9(t, dev):
    orders = t["orders"]
    orders = orders.append_column("o_year", _year(orders["o_orderdate"]))
    return (
        _lf(t["lineitem"])
        .join(_lf(t["part"]).filter(col("p_name").contains("green")),
              left_on="l_partkey", right_on="p_partkey")
        .join(_lf(t["partsupp"]), left_on=["l_partkey", "l_suppkey"],
              right_on=["ps_partkey", "ps_suppkey"])
        .join(_lf(t["supplier"]), left_on="l_suppkey", right_on="s_suppkey")
        .join(_lf(t["nation"]), left_on="s_nationkey", right_on="n_nationkey")
        .join(_lf(orders), left_on="l_orderkey", right_on="o_orderkey")
        .groupby("n_name", "o_year")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))
              - col("ps_supplycost") * col("l_quantity")).sum().alias("sum_profit"))
        .sort(col("n_name"))
        .compute(device=dev)
    )


def q10(t, dev):
    return (
        _lf(t["lineitem"]).filter(col("l_returnflag") == lit("R"))
        .join(_lf(t["orders"]).filter((col("o_orderdate") >= lit(S.Q10_LO)) &
                                      (col("o_orderdate") < lit(S.Q10_HI))),
              left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(t["customer"]), left_on="o_custkey", right_on="c_custkey")
        .join(_lf(t["nation"]), left_on="c_nationkey", right_on="n_nationkey")
        .groupby("o_custkey", "n_name")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .sort(col("revenue"), descending=True)
        .limit(20)
        .compute(device=dev)
    )


def q11(t, dev):
    sup = t["supplier"]
    ger = sup.filter(pc.equal(sup["s_nationkey"], S.Q11_NATIONKEY)).select(["s_suppkey"])
    grouped = (
        _lf(t["partsupp"])
        .join(_lf(ger), left_on="ps_suppkey", right_on="s_suppkey")
        .groupby("ps_partkey")
        .agg((col("ps_supplycost") * col("ps_availqty")).sum().alias("value"))
        .compute(device=dev)
    )
    total = pc.sum(grouped["value"]).as_py() or 0.0
    thr = total * S.Q11_FRACTION
    out = grouped.filter(pc.greater(grouped["value"], pa.scalar(float(thr), grouped["value"].type)))
    return out.sort_by([("value", "descending")])


def q12(t, dev):
    hi = ["1-URGENT", "2-HIGH"]
    return (
        _lf(t["lineitem"])
        .join(_lf(t["orders"]), left_on="l_orderkey", right_on="o_orderkey")
        .filter(col("l_shipmode").isin(["MAIL", "SHIP"]))
        .filter(col("l_commitdate") < col("l_receiptdate"))
        .filter(col("l_shipdate") < col("l_commitdate"))
        .filter(col("l_receiptdate") >= lit(S.Q12_LO))
        .filter(col("l_receiptdate") < lit(S.Q12_HI))
        .groupby("l_shipmode")
        .agg(
            when(col("o_orderpriority").isin(hi), lit(1), lit(0)).sum().alias("high_line_count"),
            when(~col("o_orderpriority").isin(hi), lit(1), lit(0)).sum().alias("low_line_count"),
        )
        .sort(col("l_shipmode"))
        .compute(device=dev)
    )


def q13(t, dev):
    joined = (_lf(t["customer"])
              .join(_lf(t["orders"]), left_on="c_custkey", right_on="o_custkey", how="left")
              .compute(device=dev))
    has_order = pc.cast(pc.is_valid(joined["o_orderkey"]), pa.float32())
    c_counts = (
        _lf(pa.table({"c_custkey": joined["c_custkey"], "has_order": has_order}))
        .groupby("c_custkey").agg(col("has_order").sum().alias("c_count_f")).compute(device=dev)
    )
    per_cust = pa.table({"c_custkey": c_counts["c_custkey"],
                         "c_count": pc.cast(c_counts["c_count_f"], pa.int32())})
    return (
        _lf(per_cust).groupby("c_count").agg(col("c_custkey").count().alias("custdist"))
        .sort(col("custdist"), descending=True).compute(device=dev)
    )


def q14(t, dev):
    revenue = col("l_extendedprice") * (lit(1.0) - col("l_discount"))
    promo = when(col("p_type").startswith("PROMO"), revenue, lit(0.0))
    return (
        _lf(t["lineitem"])
        .join(_lf(t["part"]), left_on="l_partkey", right_on="p_partkey")
        .filter(col("l_shipdate") >= lit(S.Q14_LO))
        .filter(col("l_shipdate") < lit(S.Q14_HI))
        .groupby()
        .agg(promo.sum().alias("promo_revenue"), revenue.sum().alias("total_revenue"))
        .compute(device=dev)
    )


def q15(t, dev):
    view = (
        _lf(t["lineitem"])
        .filter((col("l_shipdate") >= lit(S.Q15_LO)) & (col("l_shipdate") < lit(S.Q15_HI)))
        .groupby("l_suppkey")
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("total_revenue"))
        .compute(device=dev)
    )
    mx = pc.max(view["total_revenue"])
    top = view.filter(pc.equal(view["total_revenue"], mx))
    return (
        _lf(t["supplier"])
        .join(_lf(top), left_on="s_suppkey", right_on="l_suppkey")
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort(col("s_suppkey"))
        .compute(device=dev)
    )


def q16(t, dev):
    sup = t["supplier"]
    complaint = pc.unique(sup.filter(pc.match_substring(sup["s_comment"], pattern="Complaints"))
                          ["s_suppkey"]).to_pylist()
    deduped = (
        _lf(t["part"])
        .filter((col("p_brand") != lit("Brand#45")) & (~col("p_type").startswith("MEDIUM POLISHED"))
                & col("p_size").isin(S.Q16_SIZES))
        .join(_lf(t["partsupp"]), left_on="p_partkey", right_on="ps_partkey")
        .filter(~col("ps_suppkey").isin(complaint))
        .select("p_brand", "p_type", "p_size", "ps_suppkey")
        .distinct()
        .compute(device=dev)
    )
    return (
        _lf(deduped).groupby("p_brand", "p_type", "p_size")
        .agg(col("ps_suppkey").count().alias("supplier_cnt"))
        .sort(col("supplier_cnt"), descending=True).compute(device=dev)
    )


def q17(t, dev):
    part, lineitem = t["part"], t["lineitem"]
    target = part.filter(pc.and_(pc.equal(part["p_brand"], "Brand#23"),
                                 pc.equal(part["p_container"], "MED BOX"))).select(["p_partkey"])
    avg_table = (
        _lf(lineitem).join(_lf(target), left_on="l_partkey", right_on="p_partkey")
        .groupby("l_partkey").agg(col("l_quantity").mean().alias("avg_qty")).compute(device=dev)
    )
    res = (
        _lf(lineitem).join(_lf(target), left_on="l_partkey", right_on="p_partkey")
        .join(_lf(avg_table), left_on="l_partkey", right_on="l_partkey")
        .filter(col("l_quantity") < col("avg_qty") * lit(0.2))
        .groupby().agg(col("l_extendedprice").sum().alias("revenue")).compute(device=dev)
    )
    total = res["revenue"][0].as_py() if res.num_rows else None
    return pa.table({"revenue": [(total or 0.0) / 7.0]})


def q18(t, dev):
    qty = (_lf(t["lineitem"]).groupby("l_orderkey")
           .agg(col("l_quantity").sum().alias("sum_qty")).compute(device=dev))
    valid = qty.filter(pc.greater(qty["sum_qty"], pa.scalar(300.0, qty["sum_qty"].type))
                       ).select(["l_orderkey"])
    return (
        _lf(t["lineitem"])
        .join(_lf(valid), left_on="l_orderkey", right_on="l_orderkey")
        .join(_lf(t["orders"]), left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(t["customer"]), left_on="o_custkey", right_on="c_custkey")
        .groupby("o_custkey", "l_orderkey")
        .agg(col("l_quantity").sum().alias("sum_qty"))
        .sort(col("sum_qty"), descending=True)
        .limit(100)
        .compute(device=dev)
    )


def q19(t, dev):
    sm = ((col("p_brand") == lit("Brand#12")) & col("p_container").isin(S.CONTAINERS_SM)
          & (col("l_quantity") >= lit(1.0)) & (col("l_quantity") <= lit(11.0))
          & (col("p_size") >= lit(1)) & (col("p_size") <= lit(5)))
    md = ((col("p_brand") == lit("Brand#23")) & col("p_container").isin(S.CONTAINERS_MD)
          & (col("l_quantity") >= lit(10.0)) & (col("l_quantity") <= lit(20.0))
          & (col("p_size") >= lit(1)) & (col("p_size") <= lit(10)))
    lg = ((col("p_brand") == lit("Brand#34")) & col("p_container").isin(S.CONTAINERS_LG)
          & (col("l_quantity") >= lit(20.0)) & (col("l_quantity") <= lit(30.0))
          & (col("p_size") >= lit(1)) & (col("p_size") <= lit(15)))
    return (
        _lf(t["lineitem"]).join(_lf(t["part"]), left_on="l_partkey", right_on="p_partkey")
        .filter(sm | md | lg)
        .groupby()
        .agg((col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("revenue"))
        .compute(device=dev)
    )


def q20(t, dev):
    part, supplier, partsupp, lineitem = t["part"], t["supplier"], t["partsupp"], t["lineitem"]
    empty = pa.table({"s_name": pa.array([], pa.string()), "s_address": pa.array([], pa.string())})
    max_supp = int(supplier.num_rows) + 1
    forest = pc.unique(part.filter(pc.match_substring(part["p_name"], pattern="forest"))["p_partkey"])
    if len(forest) == 0:
        return empty
    canada = supplier.filter(pc.equal(supplier["s_nationkey"], S.Q20_NATIONKEY))
    li = lineitem.filter(pc.and_(
        pc.and_(pc.greater_equal(lineitem["l_shipdate"], S.Q20_LO),
                pc.less(lineitem["l_shipdate"], S.Q20_HI)),
        pc.is_in(lineitem["l_partkey"], value_set=forest)))
    if li.num_rows == 0:
        return empty
    lk = np.asarray(li["l_partkey"]).astype(np.int64)
    sk = np.asarray(li["l_suppkey"]).astype(np.int64)
    comp = pa.table({"composite_key": pa.array(lk * max_supp + sk, type=pa.int64()),
                     "qty": li["l_quantity"]})
    agg = (_lf(comp).groupby("composite_key").agg(col("qty").sum().alias("sum_qty"))
           .compute(device=dev))
    ck = np.asarray(agg["composite_key"]).astype(np.int64)
    sq = np.asarray(agg["sum_qty"])
    ps = partsupp.filter(pc.is_in(partsupp["ps_partkey"], value_set=forest))
    ps_ck = (np.asarray(ps["ps_partkey"]).astype(np.int64) * max_supp
             + np.asarray(ps["ps_suppkey"]).astype(np.int64))
    ps_av = np.asarray(ps["ps_availqty"])
    order = np.argsort(ck)
    sck = ck[order]
    ins = np.clip(np.searchsorted(sck, ps_ck), 0, len(sck) - 1)
    match = sck[ins] == ps_ck
    excess = match & (ps_av > 0.5 * sq[order[ins]])
    keys = pa.array(np.unique(np.asarray(ps["ps_suppkey"])[excess]).astype(np.int64))
    res = canada.filter(pc.is_in(canada["s_suppkey"], value_set=keys))
    return res.select(["s_name", "s_address"]).sort_by([("s_name", "ascending")])


def q21(t, dev):
    supplier, orders, lineitem = t["supplier"], t["orders"], t["lineitem"]
    saudi = supplier.filter(pc.equal(supplier["s_nationkey"], S.Q21_NATIONKEY)).select(["s_suppkey"])
    f_orders = orders.filter(pc.equal(orders["o_orderstatus"], "F")).select(["o_orderkey"])
    late = lineitem.filter(pc.greater(lineitem["l_receiptdate"], lineitem["l_commitdate"]))
    max_supp = int(supplier.num_rows) + 1
    ok = np.asarray(late["l_orderkey"]).astype(np.int64)
    sk = np.asarray(late["l_suppkey"]).astype(np.int64)
    pairs = np.unique(ok * max_supp + sk)
    u_ok, u_cnt = np.unique(pairs // max_supp, return_counts=True)
    single = pa.table({"l_orderkey": pa.array(u_ok[u_cnt == 1].astype(np.int64))})
    return (
        _lf(late)
        .join(_lf(f_orders), left_on="l_orderkey", right_on="o_orderkey")
        .join(_lf(single), left_on="l_orderkey", right_on="l_orderkey")
        .join(_lf(saudi), left_on="l_suppkey", right_on="s_suppkey")
        .groupby("l_suppkey")
        .agg(col("l_orderkey").count().alias("numwait"))
        .sort(col("numwait"), descending=True)
        .limit(100)
        .compute(device=dev)
    )


def q22(t, dev):
    customer, orders = t["customer"], t["orders"]
    code = pc.utf8_slice_codeunits(customer["c_phone"], 0, 2)
    in_t = pc.is_in(code, value_set=pa.array(S.Q22_CODES))
    tc = customer.filter(in_t)
    tcode = code.filter(in_t)
    bal = tc["c_acctbal"]
    avg_bal = float(pc.mean(bal.filter(pc.greater(bal, pa.scalar(0.0, bal.type)))).as_py() or 0.0)
    no_order = pa.array(~np.isin(np.asarray(tc["c_custkey"]), np.asarray(orders["o_custkey"])))
    nc = tc.filter(no_order)
    ncode = tcode.filter(no_order)
    keep = pc.greater(nc["c_acctbal"], pa.scalar(avg_bal, bal.type))
    filtered = nc.filter(keep).append_column("cntrycode", ncode.filter(keep))
    return (
        _lf(filtered).groupby("cntrycode")
        .agg(col("c_custkey").count().alias("numcust"), col("c_acctbal").sum().alias("totacctbal"))
        .sort(col("cntrycode")).compute(device=dev)
    )


QUERIES = {i: globals()[f"q{i}"] for i in range(1, 23)}


# Tables this file touches with PyArrow string/nation kernels before mxframe sees them: read plain.
PLAIN = {2: {"part"}, 7: {"nation"}, 16: {"supplier"}, 17: {"part"}, 20: {"part", "supplier"},
         21: {"orders"}, 22: {"customer"}}


def load(q: int, data_dir: str) -> dict:
    return {n: mx_read(table_path(data_dir, n), cols, dictionary=False if n in PLAIN.get(q, ()) else "auto")
            for n, cols in S.TABLES[q].items()}


def run(q: int, data_dir: str, device: str):
    """Timed unit: Parquet read + query + materialized result (a pa.Table)."""
    return QUERIES[q](load(q, data_dir), device)


def to_arrow(result) -> pa.Table:
    return result
