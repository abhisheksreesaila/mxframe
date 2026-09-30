"""What each of the 22 queries reads, returns and how it is compared.

Dates in the Parquet files are int32 days since 1970-01-01, DECIMAL columns are float32
(see scripts/gen_tpch_parquet.py).  All engines use the same constants below.
"""
from datetime import date


def days(y: int, m: int, d: int) -> int:
    return (date(y, m, d) - date(1970, 1, 1)).days


# ---- constants shared by every engine ------------------------------------------------------
Q1_CUTOFF = days(1998, 9, 2)                        # 10471
Q3_DATE = days(1995, 3, 15)                         # 9204
Q4_LO, Q4_HI = days(1993, 7, 1), days(1993, 10, 1)
Q5_LO, Q5_HI = days(1994, 1, 1), days(1995, 1, 1)
Q6_LO, Q6_HI = days(1994, 1, 1), days(1995, 1, 1)
Q7_LO, Q7_HI = days(1995, 1, 1), days(1996, 12, 31)  # inclusive
Q10_LO, Q10_HI = days(1993, 10, 1), days(1994, 1, 1)
Q12_LO, Q12_HI = days(1994, 1, 1), days(1995, 1, 1)
Q14_LO, Q14_HI = days(1995, 9, 1), days(1995, 10, 1)
Q15_LO, Q15_HI = days(1996, 1, 1), days(1996, 4, 1)
Q20_LO, Q20_HI = days(1994, 1, 1), days(1995, 1, 1)

Q11_NATIONKEY = 7            # GERMANY
Q11_FRACTION = 0.0001
Q20_NATIONKEY = 3            # CANADA
Q21_NATIONKEY = 20           # SAUDI ARABIA
Q2_REGIONKEY = 3             # EUROPE
Q8_REGIONKEY = 1             # AMERICA
Q8_NATIONKEY = 2             # BRAZIL
Q16_SIZES = [49, 14, 23, 45, 19, 3, 36, 9]
Q22_CODES = ["13", "31", "23", "29", "30", "18", "17"]
CONTAINERS_SM = ["SM CASE", "SM BOX", "SM PACK", "SM PKG"]
CONTAINERS_MD = ["MED BAG", "MED BOX", "MED PKG", "MED PACK"]
CONTAINERS_LG = ["LG CASE", "LG BOX", "LG PACK", "LG PKG"]

# ---- columns each query reads (both mx and pandas read only these; Polars/DuckDB prune) ----
TABLES: dict[int, dict[str, list[str]]] = {
    1: {"lineitem": ["l_shipdate", "l_returnflag", "l_linestatus", "l_quantity",
                     "l_extendedprice", "l_discount", "l_tax"]},
    2: {"nation": ["n_nationkey", "n_regionkey"],
        "part": ["p_partkey", "p_size", "p_type", "p_mfgr"],
        "supplier": ["s_suppkey", "s_name", "s_acctbal", "s_address", "s_phone", "s_comment",
                     "s_nationkey"],
        "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost"]},
    3: {"customer": ["c_custkey", "c_mktsegment"],
        "orders": ["o_orderkey", "o_custkey", "o_orderdate", "o_shippriority"],
        "lineitem": ["l_orderkey", "l_shipdate", "l_extendedprice", "l_discount"]},
    4: {"orders": ["o_orderkey", "o_orderdate", "o_orderpriority"],
        "lineitem": ["l_orderkey", "l_commitdate", "l_receiptdate"]},
    5: {"nation": ["n_nationkey", "n_name"],
        "customer": ["c_custkey", "c_nationkey"],
        "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
        "lineitem": ["l_orderkey", "l_extendedprice", "l_discount"]},
    6: {"lineitem": ["l_shipdate", "l_discount", "l_quantity", "l_extendedprice"]},
    7: {"nation": ["n_nationkey", "n_name"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "customer": ["c_custkey", "c_nationkey"],
        "orders": ["o_orderkey", "o_custkey"],
        "lineitem": ["l_orderkey", "l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]},
    8: {"nation": ["n_nationkey", "n_regionkey"],
        "part": ["p_partkey", "p_type"],
        "customer": ["c_custkey", "c_nationkey"],
        "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
        "lineitem": ["l_orderkey", "l_partkey", "l_extendedprice", "l_discount"]},
    9: {"nation": ["n_nationkey", "n_name"],
        "supplier": ["s_suppkey", "s_nationkey"],
        "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost"],
        "part": ["p_partkey", "p_name"],
        "orders": ["o_orderkey", "o_orderdate"],
        "lineitem": ["l_orderkey", "l_partkey", "l_suppkey", "l_quantity", "l_extendedprice",
                     "l_discount"]},
    10: {"nation": ["n_nationkey", "n_name"],
         "customer": ["c_custkey", "c_nationkey"],
         "orders": ["o_orderkey", "o_custkey", "o_orderdate"],
         "lineitem": ["l_orderkey", "l_returnflag", "l_extendedprice", "l_discount"]},
    11: {"supplier": ["s_suppkey", "s_nationkey"],
         "partsupp": ["ps_partkey", "ps_suppkey", "ps_supplycost", "ps_availqty"]},
    12: {"orders": ["o_orderkey", "o_orderpriority"],
         "lineitem": ["l_orderkey", "l_shipmode", "l_commitdate", "l_receiptdate", "l_shipdate"]},
    13: {"customer": ["c_custkey"], "orders": ["o_orderkey", "o_custkey"]},
    14: {"part": ["p_partkey", "p_type"],
         "lineitem": ["l_partkey", "l_extendedprice", "l_discount", "l_shipdate"]},
    15: {"supplier": ["s_suppkey", "s_name", "s_address", "s_phone"],
         "lineitem": ["l_suppkey", "l_shipdate", "l_extendedprice", "l_discount"]},
    16: {"part": ["p_partkey", "p_brand", "p_type", "p_size"],
         "supplier": ["s_suppkey", "s_comment"],
         "partsupp": ["ps_partkey", "ps_suppkey"]},
    17: {"part": ["p_partkey", "p_brand", "p_container"],
         "lineitem": ["l_partkey", "l_quantity", "l_extendedprice"]},
    18: {"customer": ["c_custkey"],
         "orders": ["o_orderkey", "o_custkey"],
         "lineitem": ["l_orderkey", "l_quantity"]},
    19: {"part": ["p_partkey", "p_brand", "p_container", "p_size"],
         "lineitem": ["l_partkey", "l_quantity", "l_extendedprice", "l_discount"]},
    20: {"part": ["p_partkey", "p_name"],
         "supplier": ["s_suppkey", "s_name", "s_address", "s_nationkey"],
         "partsupp": ["ps_partkey", "ps_suppkey", "ps_availqty"],
         "lineitem": ["l_partkey", "l_suppkey", "l_quantity", "l_shipdate"]},
    21: {"supplier": ["s_suppkey", "s_nationkey"],
         "orders": ["o_orderkey", "o_orderstatus"],
         "lineitem": ["l_orderkey", "l_suppkey", "l_commitdate", "l_receiptdate"]},
    22: {"customer": ["c_custkey", "c_phone", "c_acctbal"], "orders": ["o_custkey"]},
}

# ---- canonical output columns (results are renamed / reordered to these before comparing) --
OUTPUT: dict[int, list[str]] = {
    1: ["l_returnflag", "l_linestatus", "sum_qty", "sum_base_price", "sum_disc_price",
        "sum_charge", "avg_qty", "avg_price", "avg_disc", "count_order"],
    2: ["s_acctbal", "s_name", "ps_partkey", "p_mfgr", "s_address", "s_phone", "s_comment"],
    3: ["o_orderkey", "o_orderdate", "o_shippriority", "revenue"],
    4: ["o_orderpriority", "order_count"],
    5: ["n_name", "revenue"],
    6: ["revenue"],
    7: ["supp_nation", "cust_nation", "l_year", "revenue"],
    8: ["o_year", "total_vol", "brazil_vol_sum", "mkt_share"],
    9: ["n_name", "o_year", "sum_profit"],
    10: ["o_custkey", "n_name", "revenue"],
    11: ["ps_partkey", "value"],
    12: ["l_shipmode", "high_line_count", "low_line_count"],
    13: ["c_count", "custdist"],
    14: ["promo_revenue", "total_revenue"],
    15: ["s_suppkey", "s_name", "s_address", "s_phone", "total_revenue"],
    16: ["p_brand", "p_type", "p_size", "supplier_cnt"],
    17: ["revenue"],
    18: ["o_custkey", "l_orderkey", "sum_qty"],
    19: ["revenue"],
    20: ["s_name", "s_address"],
    21: ["l_suppkey", "numwait"],
    22: ["cntrycode", "numcust", "totacctbal"],
}

# LIMIT-N queries: index (in OUTPUT) of the ORDER BY metric, so boundary ties are tolerated.
TOPN_METRIC: dict[int, int] = {2: 0, 3: 3, 10: 2, 18: 2, 21: 1}

TITLES = {
    1: "filter+groupby+8 aggs", 2: "min-cost supplier", 3: "3-join+top10", 4: "EXISTS semi-join",
    5: "4-join+groupby", 6: "filter+global sum", 7: "volume shipping", 8: "market share",
    9: "6-join profit", 10: "returned items top20", 11: "important stock", 12: "shipping modes",
    13: "customer distribution", 14: "promo revenue", 15: "top supplier", 16: "part/supplier",
    17: "small-qty revenue", 18: "large volume top100", 19: "discounted revenue",
    20: "potential promotion", 21: "suppliers waiting", 22: "global sales opportunity",
}

# Where the shared implementation differs from the official TPC-H text (all engines and the
# DuckDB reference compute the SAME simplified query, so the comparison is apples-to-apples).
DEVIATIONS = {
    2: "no region-name join (nation.regionkey = 3), 'BRASS' matched anywhere in p_type, no tie-break columns",
    4: "'EXISTS' expressed as commit < receipt semi-join (as in spec)",
    5: "no supplier-nation = customer-nation condition, no region filter (all nations)",
    7: "all nation pairs (no FRANCE/GERMANY filter)",
    8: "market share by CUSTOMER nation (Brazil, nationkey 2), all order years, customers in region 1",
    10: "no c_name/c_acctbal/... output columns; groups by (o_custkey, n_name)",
    11: "Germany = nationkey 7, threshold fraction 0.0001 (not 0.0001/SF)",
    14: "returns promo_revenue and total_revenue, not the 100*ratio",
    15: "top supplier = exact equality with max revenue",
    17: "no rounding; sum(l_extendedprice)/7",
    18: "sorted by sum_qty only; no o_totalprice/o_orderdate/c_name columns",
    19: "no l_shipmode / l_shipinstruct predicates",
    20: "no partsupp/lineitem correlation on nation join beyond supplier nationkey 3 (Canada)",
    21: "orders with exactly one late supplier, no 'other supplier on the order' EXISTS",
    22: "as TPC-H apart from avg over positive balances computed in float32",
}
