"""F-157: FLOAT64 in the Query builder (Query, Frame, ParquetScan), equal to DuckDB.

mojo/tools/float_probe.mojo runs fixed plans of the builder over a Parquet file and prints the result as JSON (float64 values as their bits).
Every plan runs over a table, a file streamed by row groups (scan), a Frame after 300 edits (floats typed as text: exponents, nan, inf, -0.0,
subnormals; nulls; appended and removed rows; two formula columns) and a Frame over a scan, on random doubles with NaN, +-inf, -0.0,
subnormals and NULLs, and equals DuckDB's answer: exactly for counts, min, max, medians and quantiles, to 1e-12 relative for sums and means.
The float sum has one defined order: the same bits for any number of threads, any morsel setting and any block size.
"""
import math
import struct
import subprocess
from decimal import Decimal
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

from test_mojo_float_sql import make_table, same_value, ROW_GROUPS

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def fprobe(tmp_path_factory):
    out = tmp_path_factory.mktemp("floatprobe") / "float_probe"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/float_probe.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="session")
def fdata(tmp_path_factory):
    d = tmp_path_factory.mktemp("floatbuilder")
    n = 40_000
    t = make_table(n, 23)
    path = d / "t.parquet"
    pq.write_table(t, path, row_group_size=-(-n // ROW_GROUPS))
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return {"dir": d, "path": path, "con": con, "n": n}


def bits_to_float(h):
    return struct.unpack("<d", struct.pack("<Q", int(h, 16)))[0]


def run(fprobe, path, mode, plan, *extra):
    p = subprocess.run([str(fprobe), str(path), mode, plan, *map(str, extra)], capture_output=True, text=True)
    import json
    out = json.loads(p.stdout) if p.stdout.strip() else {"error": "no output: " + p.stderr[-300:]}
    assert "error" not in out, f"{plan} ({mode}): {out['error']}"
    return out


def agg_values(res):
    """Per aggregate, per group: the value (a float, an int, or None where the group has none)."""
    out = []
    for a in res["aggs"]:
        if "top" in a:
            out.append([[bits_to_float(x) if a["float"] else x for x in t] for t in a["top"]])
            continue
        vals = []
        for v, present in zip(a["vals"], a["present"]):
            vals.append(None if not present else bits_to_float(v) if a["float"] else v)
        out.append(vals)
    return out


# (plan, oracle SQL (one row per group g in 0..7, or one row), the aggregate columns compared to 1e-12 relative)
def grouped(sql_aggs, where=""):
    return f"SELECT {sql_aggs} FROM t WHERE g IS NOT NULL {where} GROUP BY g ORDER BY g"


PLANS = {
    "global": ("SELECT count(*), count(x), sum(x), avg(x), min(x), max(x), sum(y), avg(y), min(z), max(z), count(z), sum(z) FROM t", {2, 3, 6, 7, 11}),
    "grouped": (grouped("count(*), count(x), sum(x), avg(x), min(x), max(x), sum(y), avg(y), min(z), max(z), count(z), sum(z)"), {2, 3, 6, 7, 11}),
    "filtered": (grouped("count(*), sum(y), max(z), min(x)", "AND (z > 1e300 OR x < -990.5)"), {1}),
    "arith": (grouped("sum(x * 2.0 + y), sum(d * y), sum(i + y), sum(x / 4.0), sum(-x), max(x - y), sum(i / y), avg(i * 0.5), min(d + x), "
                      "sum(i * d), max(y / i)"), {0, 1, 2, 3, 4, 6, 7, 9}),
}
HOLISTIC = ("SELECT count(*), count(DISTINCT z), median(z), quantile_disc(z, 0.25), quantile_disc(z, 1.0), median(x), quantile_disc(x, 0.9), "
            "NULL, count(DISTINCT k), quantile_disc(k, 0.5) FROM t WHERE g IS NOT NULL GROUP BY g ORDER BY g")
CMPS = ["z = 0", "z <> 0", "z = z", "z > 1e300", "z >= 1e300", "z < 1e300", "z <= -1e300", "NOT (z > 1)", "z > k", "z = k",
        "(z <= k) OR (x > 500)", "z > i", "d < y"]


def check(got, want, rel, label, skip=()):
    """got: per aggregate a list over groups; want: DuckDB's rows (one per group)."""
    assert len(want) == len(got[0]) or all(len(g) == len(want) for g in got if g), f"{label}: {len(want)} rows against {[len(g) for g in got]}"
    for ci in range(len(got)):
        if ci in skip:
            continue
        for ri, row in enumerate(want):
            w = row[ci]
            g = got[ci][ri]
            if isinstance(w, Decimal):
                w = float(w)
            assert same_value(g, w, 1e-12 if ci in rel else 0.0), f"{label}: aggregate {ci} group {ri}: got {g!r}, DuckDB has {w!r}"


def groups_with_rows(res):
    return [g for g, c in enumerate(res["counts"]) if c > 0]


MODES = ["table", "scan"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("plan", sorted(PLANS))
def test_plans_equal_duckdb(fprobe, fdata, mode, plan):
    sql, rel = PLANS[plan]
    res = run(fprobe, fdata["path"], mode, plan)
    got = agg_values(res)
    want = fdata["con"].execute(sql).fetchall()
    if plan != "global":
        keep = groups_with_rows(res)
        got = [[col[g] for g in keep] for col in got]
    else:
        got = [[col[0]] for col in got]
    check(got, want, rel, f"{mode} {plan}")


@pytest.mark.parametrize("mode", MODES)
def test_holistic_equals_duckdb(fprobe, fdata, mode):
    res = run(fprobe, fdata["path"], mode, "holistic")
    got = agg_values(res)
    keep = groups_with_rows(res)
    got = [[col[g] for g in keep] for col in got]
    want = fdata["con"].execute(HOLISTIC).fetchall()
    check(got, want, set(), f"{mode} holistic", skip={7})
    # top_k: the 3 largest x of each group, largest first (NaN is above everything)
    top = agg_values(res)[7]
    for g, t in zip(keep, [top[g] for g in keep]):
        rows = fdata["con"].execute("SELECT x FROM t WHERE g = ? AND x IS NOT NULL ORDER BY x DESC LIMIT 3", [g]).fetchall()
        assert [x for (x,) in rows] == t, f"{mode} holistic top_k group {g}: got {t}, DuckDB has {rows}"


@pytest.mark.parametrize("mode", MODES)
def test_holistic_global_equals_duckdb(fprobe, fdata, mode):
    res = run(fprobe, fdata["path"], mode, "holistic_global")
    got = agg_values(res)
    want = fdata["con"].execute("SELECT count(DISTINCT z), median(z), quantile_disc(z, 0.5), median(y) FROM t").fetchall()
    check([[c[0]] for c in got[:4]], want, set(), f"{mode} holistic_global")


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("k", range(len(CMPS)))
def test_comparisons_count_like_duckdb(fprobe, fdata, mode, k):
    res = run(fprobe, fdata["path"], mode, f"cmp{k}")
    want = fdata["con"].execute(f"SELECT count(*) FROM t WHERE {CMPS[k]}").fetchone()[0]
    assert res["counts"][0] == want, f"{mode}: WHERE {CMPS[k]}: {res['counts'][0]} rows, DuckDB has {want}"


def test_a_32_bit_column_compares_with_a_literal_in_32_bits(fprobe, fdata):
    res = run(fprobe, fdata["path"], "table", "f32")
    want = fdata["con"].execute("SELECT count(*), sum(f::DOUBLE), max(f) FROM t WHERE f <= 7.07").fetchone()
    got = agg_values(res)
    assert got[0][0] == want[0]
    assert same_value(got[1][0], want[1], 1e-12) and got[2][0] == want[2]
    res64 = run(fprobe, fdata["path"], "table", "f64")
    want64 = fdata["con"].execute("SELECT count(*) FROM t WHERE f::DOUBLE <= 7.07").fetchone()[0]
    assert agg_values(res64)[0][0] == want64
    assert want64 != want[0]                          # the double 7.07 lies below the float 7.07: the 7.07 rows are left out


def test_a_float_group_key_is_refused_with_a_reason(fprobe, fdata):
    p = subprocess.run([str(fprobe), str(fdata["path"]), "table", "key"], capture_output=True, text=True)
    assert "a group key must be a whole number 0 .. groups-1: a FLOAT has no dense numbering" in p.stdout


@pytest.mark.parametrize("mode", ["frame", "framescan"])
@pytest.mark.parametrize("plan", ["global", "grouped", "filtered", "arith", "formula"])
def test_frames_after_300_edits_equal_duckdb(fprobe, fdata, mode, plan):
    baked = fdata["dir"] / f"baked-{mode}.parquet"
    res = run(fprobe, fdata["path"], mode, plan, "--baked", baked, "--seed", 5)
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{baked}')")
    con.execute("create view t2 as select *, x * 2.5 + y / 3 as fx, d * y - i as fd from t")
    sqls = {
        "global": ("SELECT count(*), count(x), sum(x), avg(x), min(x), max(x), sum(y), avg(y), min(z), max(z), count(z), sum(z) FROM t", {2, 3, 6, 7, 11}),
        "grouped": PLANS["grouped"], "filtered": PLANS["filtered"], "arith": PLANS["arith"],
        "formula": ("SELECT count(*), sum(fx), max(fx), sum(fd), min(fx) FROM t2 WHERE g IS NOT NULL GROUP BY g ORDER BY g", {1, 3}),
    }
    sql, rel = sqls[plan]
    got = agg_values(res)
    if plan == "global":
        got = [[col[0]] for col in got]
    else:
        keep = groups_with_rows(res)
        got = [[col[g] for g in keep] for col in got]
        sql = sql.replace(" FROM t WHERE", " FROM t2 WHERE") if plan == "formula" else sql
    want = con.execute(sql).fetchall()
    check(got, want, rel, f"{mode} {plan}")


def test_float_sums_are_the_same_bits_for_any_threads_morsels_and_block(fprobe, fdata):
    first = None
    for mode, extra in [("table", ["--threads", 1]), ("table", ["--threads", 2]), ("table", ["--threads", 7]), ("table", []),
                        ("scan", ["--threads", 1]), ("scan", ["--threads", 4]), ("scan", ["--morsels", "off"]), ("scan", ["--morsels", "on"]),
                        ("scan", ["--morsels", "on", "--block", 64]), ("scan", ["--block", 8192, "--threads", 3])]:
        res = run(fprobe, fdata["path"], mode, "grouped", *extra)
        sums = [a["vals"] for a in res["aggs"] if a["float"]]
        if mode == "table":
            ref_table = sums if first is None else ref_table
            assert sums == ref_table, f"table {extra}: float sums differ from the first run"
            first = True
        else:
            if "ref_scan" not in locals():
                ref_scan = sums
            assert sums == ref_scan, f"scan {extra}: float sums differ from the first scan run"
