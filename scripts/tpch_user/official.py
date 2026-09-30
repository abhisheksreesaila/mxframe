"""Sanity check of the DuckDB reference itself against the published TPC-H SF1 answers.

The truncated-DECIMAL data bug (F-011) made Q6 come out at 8.865e7 instead of 123141078.2283
on every engine, DuckDB included, so "all engines agree with DuckDB" is not proof of
correctness.  Only queries whose text the harness follows (see spec.DEVIATIONS) are listed;
"first" = only the top rows of a LIMIT query are compared.
"""
from __future__ import annotations

import pyarrow as pa

OFFICIAL_SF1: dict[int, dict] = {
    1: {"cols": ["l_returnflag", "l_linestatus", "sum_qty", "count_order"], "rtol": 1e-6,
        "rows": [("A", "F", 37734107, 1478493), ("N", "F", 991417, 38854),
                 ("N", "O", 74476040, 2920374), ("R", "F", 37719753, 1478870)]},
    2: {"cols": ["s_acctbal", "s_name", "n_name", "ps_partkey", "p_mfgr"], "rtol": 1e-6, "first": True,
        "rows": [(9938.53, "Supplier#000005359", "UNITED KINGDOM", 185358, "Manufacturer#4")]},
    3: {"cols": ["o_orderkey", "revenue"], "rtol": 1e-5, "rows": [(2456423, 406181.0111)], "first": True},
    4: {"cols": ["o_orderpriority", "order_count"], "rtol": 0,
        "rows": [("1-URGENT", 10594), ("2-HIGH", 10476), ("3-MEDIUM", 10410),
                 ("4-NOT SPECIFIED", 10556), ("5-LOW", 10487)]},
    5: {"cols": ["n_name", "revenue"], "rtol": 1e-5,
        "rows": [("INDONESIA", 55502041.1697), ("VIETNAM", 55295086.9967), ("CHINA", 53724494.2566),
                 ("INDIA", 52035512.0002), ("JAPAN", 45410175.6954)]},
    6: {"cols": ["revenue"], "rtol": 1e-5, "rows": [(123141078.2283,)]},
    7: {"cols": ["supp_nation", "cust_nation", "l_year", "revenue"], "rtol": 1e-5, "subset": True,
        "rows": [("FRANCE", "GERMANY", 1995, 54639732.73), ("FRANCE", "GERMANY", 1996, 54633083.31),
                 ("GERMANY", "FRANCE", 1995, 52531746.67), ("GERMANY", "FRANCE", 1996, 52520549.02)]},
    8: {"cols": ["o_year", "mkt_share"], "rtol": 1e-4,
        "rows": [(1995, 0.03443589), (1996, 0.04148552)]},
    10: {"cols": ["o_custkey", "n_name", "revenue"], "rtol": 1e-5, "first": True,
         "rows": [(57040, "JAPAN", 734235.2455)]},
    12: {"cols": ["l_shipmode", "high_line_count", "low_line_count"], "rtol": 0,
         "rows": [("MAIL", 6202, 9324), ("SHIP", 6200, 9262)]},
    15: {"cols": ["s_suppkey", "total_revenue"], "rtol": 1e-5, "rows": [(8449, 1772627.2087)]},
    14: {"cols": ["promo_revenue"], "rtol": 1e-5, "rows": [(16.3807786,)]},
    17: {"cols": ["revenue"], "rtol": 1e-4, "rows": [(348406.05,)]},
    19: {"cols": ["revenue"], "rtol": 1e-5, "rows": [(3083843.0578,)]},
    18: {"cols": ["o_custkey", "l_orderkey", "sum_qty"], "rtol": 1e-6, "subset": True,
         "rows": [(128120, 4722021, 323.0)]},
    21: {"cols": ["l_suppkey", "numwait"], "rtol": 0, "first2": True,
         "rows": [(2829, 20), (5808, 18)]},
    22: {"cols": ["cntrycode", "numcust", "totacctbal"], "rtol": 1e-4,
         "rows": [("13", 888, 6737713.99), ("17", 861, 6460573.72), ("18", 964, 7236687.40),
                  ("23", 892, 6701457.95), ("29", 948, 7158866.63), ("30", 909, 6808436.13),
                  ("31", 922, 6806670.18)]},
}


def check_official(q: int, tbl: pa.Table, official: dict | None = None) -> str | None:
    """None if no published answer is listed for the query; else "ok" or the mismatch.
"""
    entry = (official or OFFICIAL_SF1).get(q)
    if entry is None:
        return None
    cols, rtol = entry["cols"], entry["rtol"]
    got = [tuple(r[c] for c in cols) for r in tbl.select(cols).to_pylist()]
    want = entry["rows"]
    if entry.get("first") or entry.get("first2"):
        got = got[:len(want)]
    if entry.get("subset"):
        got = [g for g in got if g[:3] in {w[:3] for w in want}]
    if len(got) != len(want):
        return f"{len(got)} matching rows, the published answer has {len(want)}"
    for g, w in zip(sorted(got, key=repr), sorted(want, key=repr)):
        for gv, wv in zip(g, w):
            if isinstance(wv, str) or isinstance(gv, str):
                bad = gv != wv
            else:
                bad = abs(float(gv) - float(wv)) > max(rtol, 1e-12) * abs(float(wv))
            if bad:
                return f"row {w}: got {g}"
    return "ok"
