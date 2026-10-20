"""F-044: seeded random queries over the TPC-H tables, each written once as a spec and rendered three
ways (omxcore LazyFrame, Polars LazyFrame, DuckDB SQL), so the engines run the same logical query.

Shapes: filter + group-by on one table, joins along the TPC-H foreign keys (1-3 tables, filters
before or after the joins), global aggregates, top-N rows, COUNT(DISTINCT) per key, HAVING, the
revenue expression. Filter constants come from the data's quantiles and distinct values (read once
from the SF1 files), so a query keeps a sensible selectivity at any scale.

Every ORDER BY is a total order (the aggregate, then every group key; top-N adds the table's primary
key), so results can be compared row by row; unordered results are compared as sorted rows.
"""
from __future__ import annotations

import json
import os
import random

import numpy as np

# measures, group keys (low / mid cardinality), primary keys, per table
MEASURES = {
    "lineitem": ["l_quantity", "l_extendedprice", "l_discount", "l_tax"],
    "orders": ["o_totalprice"],
    "customer": ["c_acctbal"],
    "part": ["p_size"],
    "partsupp": ["ps_availqty", "ps_supplycost"],
    "supplier": ["s_acctbal"],
    "nation": [],
}
GROUPS = {
    "lineitem": ["l_returnflag", "l_linestatus", "l_shipmode", "l_shipinstruct", "l_suppkey"],
    "orders": ["o_orderstatus", "o_orderpriority", "o_shippriority", "o_custkey"],
    "customer": ["c_mktsegment", "c_nationkey"],
    "part": ["p_brand", "p_mfgr", "p_container", "p_size", "p_type"],
    "partsupp": ["ps_suppkey"],
    "supplier": ["s_nationkey"],
    "nation": ["n_name"],
}
FILTERS = {
    "lineitem": ["l_quantity", "l_discount", "l_shipdate", "l_receiptdate", "l_returnflag", "l_shipmode",
                 "l_shipinstruct", "l_extendedprice"],
    "orders": ["o_orderdate", "o_totalprice", "o_orderpriority", "o_orderstatus"],
    "customer": ["c_acctbal", "c_mktsegment"],
    "part": ["p_size", "p_brand", "p_container", "p_type", "p_mfgr"],
    "partsupp": ["ps_availqty", "ps_supplycost"],
    "supplier": ["s_acctbal"],
    "nation": ["n_name"],
}
PK = {"lineitem": ["l_orderkey", "l_linenumber"], "orders": ["o_orderkey"], "customer": ["c_custkey"],
      "part": ["p_partkey"], "partsupp": ["ps_partkey", "ps_suppkey"], "supplier": ["s_suppkey"],
      "nation": ["n_nationkey"]}
# foreign-key edges: (table, column) -> (parent table, its key)
FK = {
    "lineitem": [("l_orderkey", "orders", "o_orderkey"), ("l_partkey", "part", "p_partkey"),
                 ("l_suppkey", "supplier", "s_suppkey")],
    "orders": [("o_custkey", "customer", "c_custkey")],
    "customer": [("c_nationkey", "nation", "n_nationkey")],
    "partsupp": [("ps_partkey", "part", "p_partkey"), ("ps_suppkey", "supplier", "s_suppkey")],
    "supplier": [("s_nationkey", "nation", "n_nationkey")],
    "part": [],
    "nation": [],
}
STRING_COLS = {"l_returnflag", "l_linestatus", "l_shipmode", "l_shipinstruct", "o_orderstatus", "o_orderpriority",
               "c_mktsegment", "p_brand", "p_mfgr", "p_container", "p_type", "n_name"}
BASES = ["lineitem", "lineitem", "lineitem", "orders", "orders", "partsupp", "customer", "part"]


def stats(data_dir: str) -> dict:
    """Quantiles of numeric filter columns and distinct values of string ones (SF1 files)."""
    import pyarrow.parquet as pq
    out = {}
    for t, cols in FILTERS.items():
        tab = pq.read_table(os.path.join(data_dir, f"{t}.parquet"), columns=cols)
        for c in cols:
            a = tab.column(c)
            if c in STRING_COLS:
                out[c] = sorted(set(a.to_pylist()))
            else:
                v = a.to_numpy()
                out[c] = [float(x) for x in np.quantile(v, np.linspace(0.02, 0.98, 25))]
    return out


def _filter(r: random.Random, c: str, st: dict) -> dict:
    if c in STRING_COLS:
        vals = st[c]
        kind = r.choice(["eq", "ne", "isin", "isin"] + (["prefix"] if c == "p_type" else []))
        if kind == "isin":
            return {"col": c, "op": "isin", "v": sorted(r.sample(vals, min(len(vals), r.randint(2, 4))))}
        if kind == "prefix":
            return {"col": c, "op": "prefix", "v": r.choice(sorted({x.split()[0] for x in vals}))}
        return {"col": c, "op": kind, "v": r.choice(vals)}
    q = st[c]
    if c.endswith("date") or c in ("p_size", "ps_availqty"):
        q = [int(round(x)) for x in q]
    else:                                  # 2 decimals (every engine compares float32 columns the same way)
        q = [round(x, 2) for x in q]
    kind = r.choice(["lt", "ge", "between"])
    if kind == "between":
        i = r.randrange(0, len(q) - 6)
        return {"col": c, "op": "between", "v": [q[i], q[i + r.randint(3, 5)]]}
    return {"col": c, "op": kind, "v": q[r.randrange(len(q))]}


def make(seed: int, st: dict) -> dict:
    r = random.Random(seed)
    shape = r.choice(["agg", "agg", "join", "join", "join", "global", "topn", "distinct", "select_distinct"])
    base = r.choice(BASES)
    tables = [base]
    joins = []
    if shape in ("join", "topn", "distinct", "select_distinct") and FK[base]:
        frontier = [base]
        for _ in range(r.randint(1, 3) if shape == "join" else r.randint(0, 1)):
            cand = [(t, e) for t in frontier for e in FK[t] if e[1] not in tables]
            if not cand:
                break
            t, (lk, parent, pk) = r.choice(cand)
            joins.append({"left": t, "lk": lk, "table": parent, "rk": pk})
            tables.append(parent)
            frontier.append(parent)
    filt_cols = [c for t in tables for c in FILTERS[t]]
    filters = [_filter(r, c, st) for c in r.sample(filt_cols, min(len(filt_cols), r.randint(0, 3)))]
    pre = shape == "join" and r.random() < 0.5            # filters on each table before the joins
    spec = {"seed": seed, "shape": shape, "base": base, "joins": joins, "filters": filters, "pre": pre,
            "group": [], "aggs": [], "having": None, "order": [], "limit": None, "distinct": None}
    meas = [m for t in tables for m in MEASURES[t]]
    groupable = [g for t in tables for g in GROUPS[t]
                 if not any(j["rk"] == g or j["lk"] == g for j in joins)]
    if shape == "select_distinct":
        spec["select"] = r.sample(groupable, min(len(groupable), r.randint(1, 3)))
        spec["unique"] = True
        if r.random() < 0.4:
            spec["order"] = [[c, False] for c in spec["select"]]
            spec["limit"] = r.choice([None, 20])
    elif shape == "topn":
        cols = r.sample(meas, min(len(meas), 2)) if meas else []
        spec["select"] = PK[base] + cols
        by = cols[0] if cols else PK[base][0]
        spec["order"] = [[by, r.random() < 0.6]] + [[k, False] for k in PK[base]]
        spec["limit"] = r.choice([10, 50, 200])
    elif shape == "distinct":
        keys = [g for g in groupable if g not in STRING_COLS] or groupable
        a = r.choice(keys + PK[base][:1])
        b = r.choice([c for c in (groupable + PK[base]) if c != a])
        spec["distinct"] = [a, b]
        spec["group"] = [a]
        spec["aggs"] = [["count", b, "n"]]
        if r.random() < 0.4:
            spec["having"] = ["n", r.randint(1, 3)]
        spec["order"] = [["n", True], [a, False]]
        spec["limit"] = r.choice([None, 20, 100])
    else:
        if shape != "global":
            spec["group"] = r.sample(groupable, min(len(groupable), r.randint(1, 3)))
        aggs = []
        for i in range(r.randint(1, 4)):
            fn = r.choice(["sum", "sum", "min", "max", "mean", "count"])
            src = r.choice(meas) if meas else PK[base][0]
            if fn != "count" and "l_extendedprice" in meas and "l_discount" in meas and r.random() < 0.25:
                src = "rev"                                   # l_extendedprice * (1 - l_discount)
            aggs.append([fn, src, f"a{i}"])
        spec["aggs"] = aggs
        if shape != "global" and any(a[0] == "count" for a in aggs) and r.random() < 0.3:
            spec["having"] = [next(a[2] for a in aggs if a[0] == "count"), r.randint(1, 20)]
        if shape != "global" and r.random() < 0.5:
            spec["order"] = [[aggs[0][2], r.random() < 0.7]] + [[g, False] for g in spec["group"]]
            spec["limit"] = r.choice([None, 10, 100])
    spec["tables"] = _columns(spec)
    return spec


def _columns(spec) -> dict:
    """The columns each table must load."""
    need = {t: set() for t in [spec["base"]] + [j["table"] for j in spec["joins"]]}
    owner = {c: t for t in need for c in (MEASURES[t] + GROUPS[t] + FILTERS[t] + PK[t] + [e[0] for e in FK[t]])}
    used = [f["col"] for f in spec["filters"]] + spec["group"] + [o[0] for o in spec["order"]]
    used += spec.get("select", []) + (spec["distinct"] or [])
    for fn, src, _ in spec["aggs"]:
        used += ["l_extendedprice", "l_discount"] if src == "rev" else [src]
    for j in spec["joins"]:
        need[j["left"]].add(j["lk"])
        need[j["table"]].add(j["rk"])
    for c in used:
        if c in owner:
            need[owner[c]].add(c)
    return {t: sorted(cs) for t, cs in need.items()}


# ── rendering ────────────────────────────────────────────────────────────────────────────────────

def _sql_filter(f):
    c, op, v = f["col"], f["op"], f["v"]
    q = lambda x: "'" + str(x).replace("'", "''") + "'" if isinstance(x, str) else repr(x)
    if op == "between":
        return f"{c} >= {q(v[0])} AND {c} <= {q(v[1])}"
    if op == "isin":
        return f"{c} IN ({', '.join(q(x) for x in v)})"
    if op == "prefix":
        return f"starts_with({c}, {q(v)})"
    return f"{c} {dict(eq='=', ne='<>', lt='<', ge='>=')[op]} {q(v)}"


def sql(spec) -> str:
    agg_sql = {"sum": "SUM", "min": "MIN", "max": "MAX", "mean": "AVG", "count": "COUNT"}
    src = lambda s: "l_extendedprice * (1 - l_discount)" if s == "rev" else s
    frm = spec["base"] + "".join(f" JOIN {j['table']} ON {j['lk']} = {j['rk']}" for j in spec["joins"])
    where = " AND ".join(f"({_sql_filter(f)})" for f in spec["filters"]) or "TRUE"
    if spec["shape"] in ("topn", "select_distinct"):
        sel = ", ".join(spec["select"])
        q = f"SELECT {'DISTINCT ' if spec.get('unique') else ''}{sel} FROM {frm} WHERE {where}"
    elif spec["distinct"]:
        a, b = spec["distinct"]
        q = f"SELECT {a}, COUNT(DISTINCT {b}) AS n FROM {frm} WHERE {where} GROUP BY {a}"
    else:
        sel = spec["group"] + [f"{agg_sql[fn]}({src(s)}) AS {al}" for fn, s, al in spec["aggs"]]
        q = f"SELECT {', '.join(sel)} FROM {frm} WHERE {where}"
        if spec["group"]:
            q += f" GROUP BY {', '.join(spec['group'])}"
    if spec["having"]:
        h = spec["having"]
        hexpr = {"n": "COUNT(DISTINCT " + spec["distinct"][1] + ")"} if spec["distinct"] else {}
        al = h[0]
        if not spec["distinct"]:
            fn, s, _ = next(a for a in spec["aggs"] if a[2] == al)
            hexpr[al] = f"{agg_sql[fn]}({src(s)})"
        q += f" HAVING {hexpr[al]} > {h[1]}"
    if spec["order"]:
        q += " ORDER BY " + ", ".join(f"{c} {'DESC' if d else 'ASC'}" for c, d in spec["order"])
    if spec["limit"]:
        q += f" LIMIT {spec['limit']}"
    return q


def _pre_split(spec):
    """Filters per table when they run before the joins, else all after."""
    own = {}
    for f in spec["filters"]:
        t = next(t for t in spec["tables"] if f["col"] in FILTERS.get(t, []))
        own.setdefault(t, []).append(f)
    return own


def run_mx(spec, t, device="cpu"):
    """omxcore: the spec on the LazyFrames `t`; returns the computed pyarrow.Table."""
    from omxcore import LazyFrame, col, lit

    def pred(f):
        c, op, v = col(f["col"]), f["op"], f["v"]
        return {"eq": lambda: c == lit(v), "ne": lambda: c != lit(v), "lt": lambda: c < lit(v),
                "ge": lambda: c >= lit(v), "between": lambda: (c >= lit(v[0])) & (c <= lit(v[1])),
                "isin": lambda: c.isin(v), "prefix": lambda: c.startswith(v)}[op]()

    def filt(lf, fs):
        for f in fs:
            lf = lf.filter(pred(f))
        return lf

    own = _pre_split(spec) if spec["pre"] else {}
    lf = filt(t[spec["base"]], own.get(spec["base"], []))
    for j in spec["joins"]:
        lf = lf.join(filt(t[j["table"]], own.get(j["table"], [])), left_on=j["lk"], right_on=j["rk"])
    if not spec["pre"]:
        lf = filt(lf, spec["filters"])
    src = lambda s: col("l_extendedprice") * (lit(1.0) - col("l_discount")) if s == "rev" else col(s)
    if spec["shape"] in ("topn", "select_distinct"):
        lf = lf.select(*spec["select"])
        if spec.get("unique"):
            lf = lf.distinct()
    elif spec["distinct"]:
        a, b = spec["distinct"]
        lf = lf.select(a, b).distinct().groupby(a).agg(col(b).count().alias("n"))
    else:
        aggs = [getattr(src(s), fn)().alias(al) for fn, s, al in spec["aggs"]]
        lf = lf.groupby(*spec["group"]).agg(*aggs)
    if spec["having"]:
        lf = lf.filter(col(spec["having"][0]) > lit(spec["having"][1]))
    if spec["order"]:
        lf = lf.sort(*[c for c, _ in spec["order"]], descending=[d for _, d in spec["order"]])
    if spec["limit"]:
        lf = lf.limit(spec["limit"])
    return lf.compute(device=device)


def run_polars(spec, t):
    import polars as pl

    def pred(f):
        c, op, v = pl.col(f["col"]), f["op"], f["v"]
        return {"eq": lambda: c == v, "ne": lambda: c != v, "lt": lambda: c < v, "ge": lambda: c >= v,
                "between": lambda: (c >= v[0]) & (c <= v[1]), "isin": lambda: c.is_in(v),
                "prefix": lambda: c.str.starts_with(v)}[op]()

    def filt(lf, fs):
        for f in fs:
            lf = lf.filter(pred(f))
        return lf

    own = _pre_split(spec) if spec["pre"] else {}
    lf = filt(t[spec["base"]], own.get(spec["base"], []))
    for j in spec["joins"]:
        lf = lf.join(filt(t[j["table"]], own.get(j["table"], [])), left_on=j["lk"], right_on=j["rk"])
    if not spec["pre"]:
        lf = filt(lf, spec["filters"])
    src = lambda s: pl.col("l_extendedprice") * (1.0 - pl.col("l_discount")) if s == "rev" else pl.col(s)
    if spec["shape"] in ("topn", "select_distinct"):
        lf = lf.select(spec["select"])
        if spec.get("unique"):
            lf = lf.unique()
    elif spec["distinct"]:
        a, b = spec["distinct"]
        lf = lf.group_by(a).agg(pl.col(b).n_unique().alias("n"))
    else:
        aggs = [getattr(src(s), fn)().alias(al) for fn, s, al in spec["aggs"]]
        lf = lf.group_by(spec["group"]).agg(aggs) if spec["group"] else lf.select(aggs)
    if spec["having"]:
        lf = lf.filter(pl.col(spec["having"][0]) > spec["having"][1])
    if spec["order"]:
        lf = lf.sort([c for c, _ in spec["order"]], descending=[d for _, d in spec["order"]])
    if spec["limit"]:
        lf = lf.head(spec["limit"])
    return lf.collect()            # a Polars DataFrame; to Arrow after the clock (as in the TPC-H runs)


def run_duckdb(spec, con):
    res = con.execute(sql(spec))
    out = res.fetch_arrow_table() if hasattr(res, "fetch_arrow_table") else res.arrow()
    return out.read_all() if hasattr(out, "read_all") else out


if __name__ == "__main__":
    import sys
    st = stats(sys.argv[1])
    for s in range(int(sys.argv[2])):
        sp = make(s, st)
        print(json.dumps({"seed": s, "shape": sp["shape"], "sql": sql(sp)}))
