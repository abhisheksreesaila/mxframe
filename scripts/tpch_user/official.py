"""Sanity check of the DuckDB reference itself against the published TPC-H SF1 answers.

The truncated-DECIMAL data bug (F-011) made Q6 come out at 8.865e7 instead of 123141078.2283
on every engine, DuckDB included, so "all engines agree with DuckDB" is not proof of
correctness.  The answers come from DuckDB's tpch extension (`tpch_answers()`, scale factor 1),
so nothing is typed by hand.  Only queries whose text the harness follows (spec.DEVIATIONS)
are checked: Q13, Q16 and Q18's order differ, so Q13/Q16 are skipped and Q18 is checked as
"our rows appear in the answer" (`subset`).
"""
from __future__ import annotations

import pyarrow as pa

# query -> {our column: answer column}, rtol for float columns, optional flags
#   subset: every published row must appear in ours (ours may hold more/other rows, e.g. Q18 sorts
#           differently and returns different columns)
#   suffix_int: answer column holds 'Supplier#000002829' where ours holds the integer 2829
CHECKS: dict[int, dict] = {
    1: {"map": {"l_returnflag": "l_returnflag", "l_linestatus": "l_linestatus", "sum_qty": "sum_qty",
                "sum_disc_price": "sum_disc_price", "count_order": "count_order"}, "rtol": 1e-6},
    2: {"map": {"s_acctbal": "s_acctbal", "s_name": "s_name", "n_name": "n_name",
                "ps_partkey": "p_partkey", "p_mfgr": "p_mfgr"}, "rtol": 1e-6},
    3: {"map": {"o_orderkey": "l_orderkey", "revenue": "revenue", "o_shippriority": "o_shippriority"},
        "rtol": 1e-5},
    4: {"map": {"o_orderpriority": "o_orderpriority", "order_count": "order_count"}, "rtol": 0},
    5: {"map": {"n_name": "n_name", "revenue": "revenue"}, "rtol": 1e-5},
    6: {"map": {"revenue": "revenue"}, "rtol": 1e-5},
    7: {"map": {"supp_nation": "supp_nation", "cust_nation": "cust_nation", "l_year": "l_year",
                "revenue": "revenue"}, "rtol": 1e-5},
    8: {"map": {"o_year": "o_year", "mkt_share": "mkt_share"}, "rtol": 1e-4},
    9: {"map": {"n_name": "nation", "o_year": "o_year", "sum_profit": "sum_profit"}, "rtol": 1e-4},
    10: {"map": {"o_custkey": "c_custkey", "n_name": "n_name", "revenue": "revenue"}, "rtol": 1e-5},
    11: {"map": {"ps_partkey": "ps_partkey", "value": "value"}, "rtol": 1e-4},
    12: {"map": {"l_shipmode": "l_shipmode", "high_line_count": "high_line_count",
                 "low_line_count": "low_line_count"}, "rtol": 0},
    14: {"map": {"promo_revenue": "promo_revenue"}, "rtol": 1e-5},
    15: {"map": {"s_suppkey": "s_suppkey", "total_revenue": "total_revenue"}, "rtol": 1e-5},
    17: {"map": {"revenue": "avg_yearly"}, "rtol": 1e-4},
    18: {"map": {"o_custkey": "c_custkey", "l_orderkey": "o_orderkey", "sum_qty": "sum"},
         "rtol": 1e-6, "subset": True},
    19: {"map": {"revenue": "revenue"}, "rtol": 1e-5},
    20: {"map": {"s_name": "s_name", "s_address": "s_address"}, "rtol": 0},
    21: {"map": {"l_suppkey": "s_name", "numwait": "numwait"}, "rtol": 0, "suffix_int": "l_suppkey"},
    22: {"map": {"cntrycode": "cntrycode", "numcust": "numcust", "totacctbal": "totacctbal"}, "rtol": 1e-4},
}

_ANSWERS: dict[int, str] | None = None


def load_answers() -> dict[int, str]:
    """{query: '|'-separated text with a header line} for SF1 from DuckDB's tpch extension."""
    global _ANSWERS
    if _ANSWERS is None:
        import duckdb
        con = duckdb.connect()
        con.execute("INSTALL tpch")
        con.execute("LOAD tpch")
        _ANSWERS = dict(con.execute(
            "SELECT query_nr, answer FROM tpch_answers() WHERE scale_factor = 1").fetchall())
    return _ANSWERS


def _published(text: str, cfg: dict, sample: dict) -> list[tuple]:
    lines = text.rstrip("\n").split("\n")
    head = lines[0].split("|")
    out = []
    for ln in lines[1:]:
        cells = dict(zip(head, ln.split("|")))
        row = []
        for ours, theirs in cfg["map"].items():
            v = cells[theirs]
            if cfg.get("suffix_int") == ours:
                v = int(v.rsplit("#", 1)[1])
            elif isinstance(sample[ours], bool) or isinstance(sample[ours], int):
                v = int(float(v))
            elif isinstance(sample[ours], float):
                v = float(v)
            row.append(v)
        out.append(tuple(row))
    return out


def check_official(q: int, tbl: pa.Table, answers: dict[int, str] | None = None) -> str | None:
    """None if the query is not checked; else "ok" or a description of the first mismatch."""
    cfg = CHECKS.get(q)
    if cfg is None:
        return None
    answers = answers or load_answers()
    cols = list(cfg["map"])
    got = [tuple(r[c] for c in cols) for r in tbl.select(cols).to_pylist()]
    if not got:
        return "empty result"
    want = _published(answers[q], cfg, dict(zip(cols, got[0])))
    rtol = cfg["rtol"]
    isf = [isinstance(v, float) for v in got[0]]

    def key(r):
        return tuple(v for v, f in zip(r, isf) if not f) + tuple(round(v, 2) for v, f in zip(r, isf) if f)

    if cfg.get("subset"):
        want_keys = {tuple(v for v, f in zip(w, isf) if not f) for w in want}
        got = [g for g in got if tuple(v for v, f in zip(g, isf) if not f) in want_keys]
    if len(got) != len(want):
        return f"{len(got)} matching rows, the published answer has {len(want)}"
    for g, w in zip(sorted(got, key=key), sorted(want, key=key)):
        for gv, wv, f in zip(g, w, isf):
            bad = (abs(gv - wv) > max(rtol, 1e-12) * abs(wv)) if f else gv != wv
            if bad:
                return f"row {w}: got {g}"
    return "ok"
