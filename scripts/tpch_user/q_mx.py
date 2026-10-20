"""omxcore implementations of the 22 queries, written the way a user of the public API would.

Every table is a `mx.scan_parquet(path)` LazyFrame (column pruning and simple filter pushdown
are the engine's job), every query is a chain of LazyFrame operations ending in `.compute(device=)`
that returns a `pa.Table`.  Multi-stage queries materialize a small intermediate with `.compute()`
and wrap it again with `mx.LazyFrame(table)`.  The benchmark script's hand-written direct-GPU-kernel
shortcuts are NOT used, since a user cannot call those.

What the public API cannot express is done with PyArrow / NumPy inside the timed region and is
declared per query in `spec.OUTSIDE_MX` (the report prints it, so nobody mistakes those queries for
pure-mxframe ones).  The Parquet files store dates as int32 days since 1970-01-01, and
`Expr.year()` assumes YYYYMMDD ints or date32, so `_year` derives the year with a sum of
`when(date >= 1 January of year, 1, 0)` terms, which is plain omxcore.
"""
from __future__ import annotations

import pyarrow as pa

import omxcore as mx
from omxcore import LazyFrame, col, lit, when

from omxcore import np_arrow

from . import spec as S
from .readers import table_path

def _year(c: str, y0: int, y1: int):
    """Year of an int32 days-since-epoch column, for years y0..y1 (omxcore expression)."""
    e = when(col(c) >= lit(S.days(y0 + 1, 1, 1)), lit(1), lit(0))
    for y in range(y0 + 2, y1 + 1):
        e = e + when(col(c) >= lit(S.days(y, 1, 1)), lit(1), lit(0))
    return e + lit(y0)


def _rev():
    return col("l_extendedprice") * (lit(1.0) - col("l_discount"))


def _scalar(tbl: pa.Table, name: str = None):
    return tbl.column(name or tbl.column_names[0])[0].as_py()


def q1(t, dev):
    return (
        t["lineitem"]
        .filter(col("l_shipdate") <= lit(S.Q1_CUTOFF))
        .groupby("l_returnflag", "l_linestatus")
        .agg(
            col("l_quantity").sum().alias("sum_qty"),
            col("l_extendedprice").sum().alias("sum_base_price"),
            _rev().sum().alias("sum_disc_price"),
            (_rev() * (lit(1.0) + col("l_tax"))).sum().alias("sum_charge"),
            col("l_quantity").mean().alias("avg_qty"),
            col("l_extendedprice").mean().alias("avg_price"),
            col("l_discount").mean().alias("avg_disc"),
            col("l_quantity").count().alias("count_order"),
        )
        .sort("l_returnflag", "l_linestatus")
        .compute(device=dev)
    )


def q2(t, dev):
    eu = (t["nation"].filter(col("n_regionkey") == lit(S.Q2_REGIONKEY))
          .select("n_nationkey", "n_name"))
    eu_sup = t["supplier"].join(eu, left_on="s_nationkey", right_on="n_nationkey")
    target = (t["part"].filter((col("p_size") == lit(15)) & col("p_type").contains("BRASS"))
              .select("p_partkey", "p_mfgr"))
    # the intermediate is materialized with .compute(), so name the columns the rest of the query
    # reads (a user would): otherwise every partsupp column, ps_comment included, is read and carried
    cand = (t["partsupp"].join(eu_sup, left_on="ps_suppkey", right_on="s_suppkey")
            .join(target, left_on="ps_partkey", right_on="p_partkey")
            .select("ps_partkey", "ps_supplycost", "s_acctbal", "s_name", "n_name", "p_mfgr",
                    "s_address", "s_phone", "s_comment")
            .compute(device=dev))
    c = LazyFrame(cand)
    min_costs = c.groupby("ps_partkey").agg(col("ps_supplycost").min().alias("min_supplycost"))
    return (
        c.join(min_costs, on="ps_partkey")
        .filter(col("ps_supplycost") == col("min_supplycost"))
        .select("s_acctbal", "s_name", "n_name", "ps_partkey", "p_mfgr", "s_address", "s_phone",
                "s_comment")
        .sort("s_acctbal", "n_name", "s_name", "ps_partkey", descending=[True, False, False, False])
        .limit(100)
        .compute(device=dev)
    )


def q3(t, dev):
    lf_c = t["customer"].filter(col("c_mktsegment") == lit("BUILDING"))
    lf_o = t["orders"].filter(col("o_orderdate") < lit(S.Q3_DATE))
    lf_l = t["lineitem"].filter(col("l_shipdate") > lit(S.Q3_DATE))
    return (
        lf_o.join(lf_c, left_on="o_custkey", right_on="c_custkey")
        .join(lf_l, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderkey", "o_orderdate", "o_shippriority")
        .agg(_rev().sum().alias("revenue"))
        .sort("revenue", "o_orderdate", "o_orderkey", descending=[True, False, False])
        .limit(10)
        .compute(device=dev)
    )


def q4(t, dev):
    late_orders = (t["lineitem"].filter(col("l_commitdate") < col("l_receiptdate"))
                   .select("l_orderkey").distinct())
    return (
        t["orders"]
        .filter((col("o_orderdate") >= lit(S.Q4_LO)) & (col("o_orderdate") < lit(S.Q4_HI)))
        .join(late_orders, left_on="o_orderkey", right_on="l_orderkey")
        .groupby("o_orderpriority")
        .agg(col("o_orderpriority").count().alias("order_count"))
        .sort("o_orderpriority")
        .compute(device=dev)
    )


def q5(t, dev):
    asia = t["nation"].filter(col("n_regionkey") == lit(S.Q5_REGIONKEY))
    return (
        t["lineitem"]
        .join(t["orders"].filter((col("o_orderdate") >= lit(S.Q5_LO)) &
                                 (col("o_orderdate") < lit(S.Q5_HI))),
              left_on="l_orderkey", right_on="o_orderkey")
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .filter(col("c_nationkey") == col("s_nationkey"))
        .join(asia, left_on="c_nationkey", right_on="n_nationkey")
        .groupby("n_name")
        .agg(_rev().sum().alias("revenue"))
        .sort("revenue", descending=True)
        .compute(device=dev)
    )


def q6(t, dev):
    return (
        t["lineitem"]
        .filter((col("l_shipdate") >= lit(S.Q6_LO)) & (col("l_shipdate") < lit(S.Q6_HI))
                & (col("l_discount") >= lit(0.05)) & (col("l_discount") <= lit(0.07))
                & (col("l_quantity") < lit(24.0)))
        .groupby()
        .agg((col("l_extendedprice") * col("l_discount")).sum().alias("revenue"))
        .compute(device=dev)
    )


def q7(t, dev):
    sup = t["supplier"].join(
        t["nation"].select(col("n_nationkey").alias("sn_key"), col("n_name").alias("supp_nation")),
        left_on="s_nationkey", right_on="sn_key")
    cust = t["customer"].join(
        t["nation"].select(col("n_nationkey").alias("cn_key"), col("n_name").alias("cust_nation")),
        left_on="c_nationkey", right_on="cn_key")
    return (
        t["lineitem"]
        .filter((col("l_shipdate") >= lit(S.Q7_LO)) & (col("l_shipdate") <= lit(S.Q7_HI)))
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(cust, left_on="o_custkey", right_on="c_custkey")
        .join(sup, left_on="l_suppkey", right_on="s_suppkey")
        .filter(((col("supp_nation") == lit(S.Q7_NATIONS[0])) & (col("cust_nation") == lit(S.Q7_NATIONS[1])))
                | ((col("supp_nation") == lit(S.Q7_NATIONS[1])) & (col("cust_nation") == lit(S.Q7_NATIONS[0]))))
        .with_columns(_year("l_shipdate", 1995, 1996).alias("l_year"))
        .groupby("supp_nation", "cust_nation", "l_year")
        .agg(_rev().sum().alias("revenue"))
        .sort("supp_nation", "cust_nation", "l_year")
        .compute(device=dev)
    )


def q8(t, dev):
    am = t["nation"].filter(col("n_regionkey") == lit(S.Q8_REGIONKEY)).select("n_nationkey")
    cust_am = t["customer"].join(am, left_on="c_nationkey", right_on="n_nationkey")
    orders = t["orders"].filter((col("o_orderdate") >= lit(S.Q8_LO)) & (col("o_orderdate") <= lit(S.Q8_HI)))
    by_year = (
        t["lineitem"]
        .join(t["part"].filter(col("p_type") == lit("ECONOMY ANODIZED STEEL")),
              left_on="l_partkey", right_on="p_partkey")
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(cust_am, left_on="o_custkey", right_on="c_custkey")
        .with_columns(
            _year("o_orderdate", 1995, 1996).alias("o_year"),
            _rev().alias("volume"),
            when(col("s_nationkey") == lit(S.Q8_NATIONKEY), _rev(), lit(0.0)).alias("brazil_vol"),
        )
        .groupby("o_year")
        .agg(col("volume").sum().alias("total_vol"), col("brazil_vol").sum().alias("brazil_vol_sum"))
        .compute(device=dev)
    )
    # (a select() directly on top of an aggregate is not executed by the engine: wrap the result)
    return (LazyFrame(by_year)
            .select(col("o_year"), col("total_vol"), col("brazil_vol_sum"),
                    (col("brazil_vol_sum") / col("total_vol")).alias("mkt_share"))
            .sort("o_year").compute(device=dev))


def q9(t, dev):
    return (
        t["lineitem"]
        .join(t["part"].filter(col("p_name").contains("green")),
              left_on="l_partkey", right_on="p_partkey")
        .join(t["partsupp"], left_on=["l_partkey", "l_suppkey"],
              right_on=["ps_partkey", "ps_suppkey"])
        .join(t["supplier"], left_on="l_suppkey", right_on="s_suppkey")
        .join(t["nation"], left_on="s_nationkey", right_on="n_nationkey")
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .with_columns(_year("o_orderdate", 1992, 1998).alias("o_year"))
        .groupby("n_name", "o_year")
        .agg((_rev() - col("ps_supplycost") * col("l_quantity")).sum().alias("sum_profit"))
        .sort("n_name", "o_year", descending=[False, True])
        .compute(device=dev)
    )


def q10(t, dev):
    return (
        t["lineitem"].filter(col("l_returnflag") == lit("R"))
        .join(t["orders"].filter((col("o_orderdate") >= lit(S.Q10_LO)) &
                                 (col("o_orderdate") < lit(S.Q10_HI))),
              left_on="l_orderkey", right_on="o_orderkey")
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .join(t["nation"], left_on="c_nationkey", right_on="n_nationkey")
        .groupby("o_custkey", "n_name")
        .agg(_rev().sum().alias("revenue"))
        .sort("revenue", "o_custkey", descending=[True, False])
        .limit(20)
        .compute(device=dev)
    )


def q11(t, dev):
    ger = t["supplier"].filter(col("s_nationkey") == lit(S.Q11_NATIONKEY)).select("s_suppkey")
    base = t["partsupp"].join(ger, left_on="ps_suppkey", right_on="s_suppkey")
    value = col("ps_supplycost") * col("ps_availqty")
    # HAVING value > (SELECT sum(value) * fraction ...): the subquery is a native plan of its own (no rows:
    # NULL, and `value > NULL` keeps nothing, as in SQL), its result a literal of the grouped plan's HAVING
    return (
        base.groupby("ps_partkey")
        .agg(value.sum().alias("value"))
        .filter(col("value") > base.scalar(value.sum()) * lit(S.q11_fraction()))
        .sort("value", descending=True)
        .compute(device=dev)
    )


def q12(t, dev):
    hi = ["1-URGENT", "2-HIGH"]
    return (
        t["lineitem"]
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
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
        .sort("l_shipmode")
        .compute(device=dev)
    )


def q13(t, dev):
    # mx `count()` on the right side of a left join counts the unmatched (null) rows as 1, so
    # count orders per customer with an inner aggregate and add the zero-order customers.
    per_cust = (t["orders"].groupby("o_custkey")
                .agg(col("o_orderkey").count().alias("c_count")).compute(device=dev))
    with_orders = (LazyFrame(per_cust).groupby("c_count")
                   .agg(col("o_custkey").count().alias("custdist")).compute(device=dev))
    n_cust = _scalar(t["customer"].groupby().agg(col("c_custkey").count().alias("n")).compute(device=dev))
    zero = n_cust - per_cust.num_rows
    if zero > 0:
        with_orders = pa.concat_tables([
            with_orders,
            pa.table({"c_count": np_arrow.array([0.0], with_orders.schema.field("c_count").type),
                      "custdist": np_arrow.array([float(zero)], with_orders.schema.field("custdist").type)}),
        ])
    return (LazyFrame(with_orders).sort("custdist", "c_count", descending=[True, True])
            .compute(device=dev))


def q14(t, dev):
    promo = when(col("p_type").startswith("PROMO"), _rev(), lit(0.0))
    sums = (
        t["lineitem"]
        .join(t["part"], left_on="l_partkey", right_on="p_partkey")
        .filter(col("l_shipdate") >= lit(S.Q14_LO))
        .filter(col("l_shipdate") < lit(S.Q14_HI))
        .groupby()
        .agg(promo.sum().alias("promo_sum"), _rev().sum().alias("total_sum"))
        .compute(device=dev)
    )
    return (LazyFrame(sums).select((lit(100.0) * col("promo_sum") / col("total_sum")).alias("promo_revenue"))
            .compute(device=dev))


def q15(t, dev):
    view = (
        t["lineitem"]
        .filter((col("l_shipdate") >= lit(S.Q15_LO)) & (col("l_shipdate") < lit(S.Q15_HI)))
        .groupby("l_suppkey")
        .agg(_rev().sum().alias("total_revenue"))
        .compute(device=dev)
    )
    # the view is computed once: `total_revenue == max(total_revenue)` compares float sums, which two
    # separate runs of the aggregate may add in a different order (and differ in the last bit)
    v = LazyFrame(view)
    return (
        t["supplier"]
        .join(v.filter(col("total_revenue") == v.scalar(col("total_revenue").max())),
              left_on="s_suppkey", right_on="l_suppkey")
        .select("s_suppkey", "s_name", "s_address", "s_phone", "total_revenue")
        .sort("s_suppkey")
        .compute(device=dev)
    )


def q16(t, dev):
    complaint = (t["supplier"].filter(col("s_comment").contains("Complaints")).select("s_suppkey")
                 .compute(device=dev)["s_suppkey"].to_pylist())
    deduped = (
        t["part"]
        .filter((col("p_brand") != lit("Brand#45")) & (~col("p_type").startswith("MEDIUM POLISHED"))
                & col("p_size").isin(S.Q16_SIZES))
        .join(t["partsupp"], left_on="p_partkey", right_on="ps_partkey")
        .filter(~col("ps_suppkey").isin(complaint))
        .select("p_brand", "p_type", "p_size", "ps_suppkey")
        .distinct()
        .compute(device=dev)
    )
    return (
        LazyFrame(deduped).groupby("p_brand", "p_type", "p_size")
        .agg(col("ps_suppkey").count().alias("supplier_cnt"))
        .sort("supplier_cnt", "p_brand", "p_type", "p_size", descending=[True, False, False, False])
        .compute(device=dev)
    )


def q17(t, dev):
    target = (t["part"].filter((col("p_brand") == lit("Brand#23")) & (col("p_container") == lit("MED BOX")))
              .select("p_partkey"))
    m = LazyFrame(t["lineitem"].join(target, left_on="l_partkey", right_on="p_partkey")
                  .select("l_partkey", "l_quantity", "l_extendedprice").compute(device=dev))
    avg = m.groupby("l_partkey").agg(col("l_quantity").mean().alias("avg_qty"))
    total = (
        m.join(avg, on="l_partkey")
        .filter(col("l_quantity") < col("avg_qty") * lit(0.2))
        .groupby().agg(col("l_extendedprice").sum().alias("revenue"))
        .compute(device=dev)
    )
    return LazyFrame(total).select((col("revenue") / lit(7.0)).alias("revenue")).compute(device=dev)


def q18(t, dev):
    big = (t["lineitem"].groupby("l_orderkey").agg(col("l_quantity").sum().alias("sum_qty"))
           .filter(col("sum_qty") > lit(300.0)).select("l_orderkey"))
    return (
        t["lineitem"]
        .join(big, on="l_orderkey")
        .join(t["orders"], left_on="l_orderkey", right_on="o_orderkey")
        .join(t["customer"], left_on="o_custkey", right_on="c_custkey")
        .groupby("o_custkey", "l_orderkey")
        .agg(col("l_quantity").sum().alias("sum_qty"))
        .sort("sum_qty", "l_orderkey", descending=[True, False])
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
        t["lineitem"].join(t["part"], left_on="l_partkey", right_on="p_partkey")
        .filter((sm | md | lg) & col("l_shipmode").isin(["AIR", "AIR REG"])
                & (col("l_shipinstruct") == lit("DELIVER IN PERSON")))
        .groupby()
        .agg(_rev().sum().alias("revenue"))
        .compute(device=dev)
    )


def q20(t, dev):
    forest = t["part"].filter(col("p_name").startswith("forest")).select("p_partkey")
    shipped = (
        t["lineitem"]
        .filter((col("l_shipdate") >= lit(S.Q20_LO)) & (col("l_shipdate") < lit(S.Q20_HI)))
        .join(forest, left_on="l_partkey", right_on="p_partkey")
        .groupby("l_partkey", "l_suppkey")
        .agg(col("l_quantity").sum().alias("sum_qty"))
    )
    excess = (
        t["partsupp"]
        .join(forest, left_on="ps_partkey", right_on="p_partkey")
        .join(shipped, left_on=["ps_partkey", "ps_suppkey"], right_on=["l_partkey", "l_suppkey"])
        .filter(col("ps_availqty") > col("sum_qty") * lit(0.5))
        .select("ps_suppkey")
        .distinct()
    )
    return (
        t["supplier"].filter(col("s_nationkey") == lit(S.Q20_NATIONKEY))
        .join(excess, left_on="s_suppkey", right_on="ps_suppkey")
        .select("s_name", "s_address")
        .sort("s_name")
        .compute(device=dev)
    )


def q21(t, dev):
    late = t["lineitem"].filter(col("l_receiptdate") > col("l_commitdate"))
    single = (late.select("l_orderkey", "l_suppkey").distinct()
              .groupby("l_orderkey").agg(col("l_suppkey").count().alias("n_late"))
              .filter(col("n_late") == lit(1)).select("l_orderkey"))
    multi = (t["lineitem"].select("l_orderkey", "l_suppkey").distinct()
             .groupby("l_orderkey").agg(col("l_suppkey").count().alias("n_supp"))
             .filter(col("n_supp") >= lit(2)).select("l_orderkey"))
    f_orders = t["orders"].filter(col("o_orderstatus") == lit("F")).select("o_orderkey")
    saudi = t["supplier"].filter(col("s_nationkey") == lit(S.Q21_NATIONKEY)).select("s_suppkey")
    return (
        late
        .join(f_orders, left_on="l_orderkey", right_on="o_orderkey")
        .join(single, on="l_orderkey")
        .join(multi, on="l_orderkey")
        .join(saudi, left_on="l_suppkey", right_on="s_suppkey")
        .groupby("l_suppkey")
        .agg(col("l_orderkey").count().alias("numwait"))
        .sort("numwait", "l_suppkey", descending=[True, False])
        .limit(100)
        .compute(device=dev)
    )


def q22(t, dev):
    c = (t["customer"].select("c_custkey", "c_phone", "c_acctbal")
         .with_columns(col("c_phone").str_slice(0, 2).alias("cntrycode"))
         .filter(col("cntrycode").isin(S.Q22_CODES)))
    avg_bal = c.filter(col("c_acctbal") > lit(0.0)).scalar(col("c_acctbal").mean())
    return (
        c.filter(col("c_acctbal") > avg_bal)
        .join(t["orders"], left_on="c_custkey", right_on="o_custkey", how="anti")
        .groupby("cntrycode")
        .agg(col("c_custkey").count().alias("numcust"), col("c_acctbal").sum().alias("totacctbal"))
        .sort("cntrycode").compute(device=dev)
    )


QUERIES = {i: globals()[f"q{i}"] for i in range(1, 23)}


def load(q: int, data_dir: str) -> dict:
    """Lazy scans (footer only); the engine reads just the columns each query references."""
    S.set_scale_from(data_dir)
    return {n: mx.scan_parquet(table_path(data_dir, n)) for n in S.TABLES[q]}


def run(q: int, data_dir: str, device: str):
    """Timed unit: Parquet read + query + materialized result (a pa.Table)."""
    return QUERIES[q](load(q, data_dir), device)


def load_mem(q: int, data_dir: str) -> dict:
    """Engine-only mode, untimed: the columns the query uses as Arrow tables, read by the same
    reader `scan_parquet` uses (low-cardinality strings as dictionaries)."""
    S.set_scale_from(data_dir)
    return {n: mx.read_parquet_table(table_path(data_dir, n), columns=cols, dictionary="auto")
            for n, cols in S.TABLES[q].items()}


def run_mem(q: int, tables: dict, device: str):
    """Engine-only timed unit: query + materialized result on in-memory tables."""
    return QUERIES[q]({n: LazyFrame(t) for n, t in tables.items()}, device)


def to_arrow(result) -> pa.Table:
    return result
