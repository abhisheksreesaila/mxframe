"""F-157: FLOAT64 in SQL, continued (tests/test_mojo_float_sql.py): the path that ran, the same bits for any thread count, a 32-bit float column,
a Frame after edits, and the repo's own TPC-H SF1 Parquet, all against DuckDB."""
import json
import os
import random
import subprocess
from pathlib import Path

import duckdb
import pytest

from test_mojo_float_sql import MODES, assert_rows, data, run, probe  # noqa: F401  (fixtures)
from test_mojo_sql import run_probe, got_rows


def notes_of(res):
    return "\n".join(res["notes"])


def probe_json(probe, sql, path, mode="table", *flags):
    p = subprocess.run([str(probe), "--mode", mode, *map(str, flags), sql, f"t={path}"], capture_output=True)
    return json.loads(p.stdout.decode("utf-8"))


# ── the path that ran, not only the answer ────────────────────────────────────────────────────────────

@pytest.mark.parametrize("mode", MODES)
def test_float_keys_are_hashed_and_a_plain_float_query_has_no_group(probe, data, mode):
    res = run_probe(probe, "SELECT k, count(*) FROM t GROUP BY k", {"t": data["path"]}, mode=mode)
    assert "group: hashed on 2 key words" in notes_of(res) and "`k` is a FLOAT: floats are grouped by hashing, never by a dense range" in notes_of(res)
    res = run_probe(probe, "SELECT sum(x), min(z) FROM t", {"t": data["path"]}, mode=mode)
    assert "group: none (one result row)" in notes_of(res)
    res = run_probe(probe, "SELECT g, k, count(*) FROM t GROUP BY g, k", {"t": data["path"]}, mode=mode)
    assert "group: hashed" in notes_of(res)


def test_forcing_dense_groups_on_a_float_key_says_why(probe, data):
    res = run_probe(probe, "SELECT k, count(*) FROM t GROUP BY k", {"t": data["path"]}, group="dense", expect_error=True)
    assert "cannot force dense groups: `k` is a FLOAT: floats are grouped by hashing, never by a dense range" in res["error"]


def test_a_float_key_in_two_passes_with_holistic_aggregates(probe, data):
    sql = "SELECT k, count(DISTINCT z), median(y), quantile_disc(z, 0.5), sum(y) FROM t GROUP BY k ORDER BY k"
    res, got, want = run(probe, data, sql, "table", group="hashed")
    assert "holistic: a second pass maps each row to its group" in notes_of(res)
    assert_rows(got, want, {3}, True, sql)


@pytest.mark.parametrize("mode", MODES)
def test_sql_float_sums_are_the_same_bits_for_any_threads(probe, data, mode):
    for sql in ["SELECT sum(x), avg(y), sum(x * y) FROM t",
                "SELECT g, sum(x), avg(y) FROM t GROUP BY g ORDER BY g",
                "SELECT k, sum(x), avg(y) FROM t GROUP BY k ORDER BY k"]:
        seen = None
        for threads in (1, 2, 3, 7, 0):
            res = probe_json(probe, sql, data["path"], mode, *(["--threads", threads] if threads else []))
            seen = res["rows"] if seen is None else seen
            assert res["rows"] == seen, f"{mode} {sql}: threads {threads} gave other bits than threads 1"


# ── a 32-bit float column ─────────────────────────────────────────────────────────────────────────────

F32_QUERIES = [
    ("SELECT count(*) FROM t WHERE f <= 7.07", "SELECT count(*) FROM t WHERE f <= 7.07"),
    ("SELECT count(*) FROM t WHERE f >= 0.07", "SELECT count(*) FROM t WHERE f >= 0.07"),
    ("SELECT count(*) FROM t WHERE f = 0.1", "SELECT count(*) FROM t WHERE f = 0.1"),
    ("SELECT count(*) FROM t WHERE f BETWEEN 0.05 AND 0.07", "SELECT count(*) FROM t WHERE f BETWEEN 0.05 AND 0.07"),
    ("SELECT count(*) FROM t WHERE f IN (0.1, 0.25, 5)", "SELECT count(*) FROM t WHERE f IN (0.1, 0.25, 5)"),
    ("SELECT count(*) FROM t WHERE 0.3 < f", "SELECT count(*) FROM t WHERE 0.3 < f"),
    ("SELECT sum(f), avg(f), min(f), max(f) FROM t", "SELECT sum(f::DOUBLE), avg(f::DOUBLE), min(f), max(f) FROM t"),
    ("SELECT f, count(*), sum(y) FROM t GROUP BY f ORDER BY f", "SELECT f, count(*), sum(y) FROM t GROUP BY f ORDER BY f"),
    ("SELECT g, sum(f * y) FROM t GROUP BY g ORDER BY g", "SELECT g, sum(f::DOUBLE * y) FROM t GROUP BY g ORDER BY g"),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("q", range(len(F32_QUERIES)), ids=[q[0] for q in F32_QUERIES])
def test_a_32_bit_float_column_follows_duckdb(probe, data, mode, q):
    sql, duck = F32_QUERIES[q]
    res, got, want = run(probe, data, sql, mode, duck)
    rel = {0, 1} if sql.startswith("SELECT sum(f)") else {1, 2} if "sum(y)" in sql else {1} if "sum(f * y)" in sql else set()
    assert_rows(got, want, rel, "ORDER BY" in sql, f"{mode}: {sql}")


# ── a Frame after edits ───────────────────────────────────────────────────────────────────────────────

EDIT_TEXTS = ["1e3", "-2.5", "0", "-0.0", "nan", "inf", "-inf", "123456.789", "5e-324", "1.7976931348623157e308", "-7.25e-10", "42", ""]


def make_float_edits(con, n, seed):
    """An edit script (the probe's format) and the same edits applied to the DuckDB table `src` (rid = the row number)."""
    rng = random.Random(seed)
    lines = []
    alive = set(range(n))
    next_rid = n
    q = "'"
    for _ in range(250):
        what = rng.randint(0, 9)
        row = rng.randrange(n)
        while row not in alive:
            row = rng.randrange(n)
        if what <= 5:
            col = rng.choice(["k", "x", "y", "z"])
            text = rng.choice(EDIT_TEXTS)
            lines.append(f"set\t{col}\t{row}\t{text}")
            value = "NULL" if text == "" else f"CAST({q}{text}{q} AS DOUBLE)"
            con.execute(f"update src set {col} = {value} where rid = {row}")
        elif what == 6:
            text = rng.choice(["0.25", "1.5", "7.5", ""])
            lines.append(f"set\tf\t{row}\t{text}")
            value = "NULL" if text == "" else f"CAST({text} AS REAL)"
            con.execute(f"update src set f = {value} where rid = {row}")
        elif what == 7:
            vals = [str(next_rid), str(rng.randint(0, 7)), rng.choice(EDIT_TEXTS), "3.5", "11.25", rng.choice(EDIT_TEXTS), "12.50", str(rng.randint(-9, 9)), "0.5", "west"]
            lines.append("append\t" + "\t".join(vals))

            def lit(v, kind):
                return "NULL" if v == "" else f"CAST({q}{v}{q} AS {kind})"
            con.execute(f"insert into src values ({next_rid}, {vals[0]}, {vals[1]}, {lit(vals[2], 'DOUBLE')}, {lit(vals[3], 'DOUBLE')}, {lit(vals[4], 'DOUBLE')}, "
                        f"{lit(vals[5], 'DOUBLE')}, {vals[6]}, {vals[7]}, {lit(vals[8], 'REAL')}, {q}{vals[9]}{q})")
            alive.add(next_rid)
            next_rid += 1
        elif what == 8:
            lines.append(f"remove\t{row}")
            con.execute(f"delete from src where rid = {row}")
            alive.discard(row)
        else:
            v = rng.randint(-400, 400)
            lines.append(f"set\ti\t{row}\t{v}")
            con.execute(f"update src set i = {v} where rid = {row}")
    lines.append("formula\tfx = x * 2.5 + y / 3")
    return "\n".join(lines) + "\n"


FRAME_QUERIES = [
    ("SELECT count(*), count(x), sum(x), avg(y), min(z), max(z), count(DISTINCT z) FROM t", {2, 3}, False),
    ("SELECT g, sum(y), min(x), max(x), median(z) FROM t GROUP BY g ORDER BY g", {1}, True),
    ("SELECT k, count(*), sum(y) FROM t GROUP BY k ORDER BY k", {2}, True),
    ("SELECT count(*) FROM t WHERE z > 1e300 OR x < -990.5", set(), False),
    ("SELECT g, sum(fx), max(fx) FROM t2 GROUP BY g ORDER BY g", {1}, True),
    ("SELECT count(*) FROM t WHERE f <= 7.5 AND f >= 0.25", set(), False),
]


@pytest.fixture(scope="session")
def edited(tmp_path_factory, data):
    d = tmp_path_factory.mktemp("floatedits")
    con = duckdb.connect()
    con.execute(f"create table src as select row_number() over () - 1 as rid, * from read_parquet('{data['path']}')")
    n = con.execute("select count(*) from src").fetchone()[0]
    script = d / "edits.tsv"
    script.write_text(make_float_edits(con, n, 7))
    con.execute("create view t as select * exclude (rid) from src")
    con.execute("create view t2 as select *, x * 2.5 + y / 3 as fx from t")
    return {"con": con, "script": script}


@pytest.mark.parametrize("mode", ["frame", "framescan"])
@pytest.mark.parametrize("q", range(len(FRAME_QUERIES)), ids=[q[0] for q in FRAME_QUERIES])
def test_a_frame_after_250_float_edits_equals_duckdb(probe, data, edited, mode, q):
    sql, rel, ordered = FRAME_QUERIES[q]
    res = run_probe(probe, sql.replace("t2", "t"), {"t": data["path"]}, mode=mode, edits=edited["script"])
    got = got_rows(res)
    want = edited["con"].execute(sql).fetchall()
    assert_rows(got, want, rel, ordered, f"{mode}: {sql}")


# ── the repo's own TPC-H Parquet (REAL money columns, INT dates) ──────────────────────────────────────

TPCH = Path(os.environ.get("MXFRAME_TPCH_DATA", "/home/abhisheksreesaila/Projects/mxframe/data")) / "tpch_sf1" / "lineitem.parquet"

# Predicates stay native in DuckDB (a REAL column against a literal is compared as REAL, as the engine does); arithmetic is cast to DOUBLE,
# because the engine widens a 32-bit column to float64 when it reads it.
TPCH_QUERIES = [
    ("Q1",
     "SELECT l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice), sum(l_extendedprice * (1 - l_discount)), "
     "sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)), avg(l_quantity), avg(l_extendedprice), avg(l_discount), count(*) "
     "FROM lineitem WHERE l_shipdate <= 10471 GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
     "SELECT l_returnflag, l_linestatus, sum(l_quantity::DOUBLE), sum(l_extendedprice::DOUBLE), sum(l_extendedprice::DOUBLE * (1 - l_discount::DOUBLE)), "
     "sum(l_extendedprice::DOUBLE * (1 - l_discount::DOUBLE) * (1 + l_tax::DOUBLE)), avg(l_quantity::DOUBLE), avg(l_extendedprice::DOUBLE), "
     "avg(l_discount::DOUBLE), count(*) FROM lineitem WHERE l_shipdate <= 10471 GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
     {2, 3, 4, 5, 6, 7, 8}),
    ("Q6",
     "SELECT sum(l_extendedprice * l_discount) FROM lineitem WHERE l_shipdate >= 8766 AND l_shipdate < 9131 "
     "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24",
     "SELECT sum(l_extendedprice::DOUBLE * l_discount::DOUBLE) FROM lineitem WHERE l_shipdate >= 8766 AND l_shipdate < 9131 "
     "AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24", {0}),
    ("by discount",
     "SELECT l_discount, count(*), sum(l_extendedprice), min(l_quantity), max(l_tax) FROM lineitem GROUP BY l_discount ORDER BY l_discount",
     "SELECT l_discount, count(*), sum(l_extendedprice::DOUBLE), min(l_quantity), max(l_tax) FROM lineitem GROUP BY l_discount ORDER BY l_discount", {2}),
    ("float filter and key",
     "SELECT l_tax, count(*), avg(l_extendedprice / l_quantity) FROM lineitem WHERE l_quantity >= 10.5 AND l_extendedprice > 50000.5 "
     "GROUP BY l_tax ORDER BY l_tax",
     "SELECT l_tax, count(*), avg(l_extendedprice::DOUBLE / l_quantity::DOUBLE) FROM lineitem WHERE l_quantity >= 10.5 AND l_extendedprice > 50000.5 "
     "GROUP BY l_tax ORDER BY l_tax", {2}),
    ("median",
     "SELECT l_returnflag, median(l_extendedprice), quantile_disc(l_quantity, 0.9), count(DISTINCT l_extendedprice) FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag",
     "SELECT l_returnflag, median(l_extendedprice::DOUBLE), quantile_disc(l_quantity, 0.9), count(DISTINCT l_extendedprice) FROM lineitem GROUP BY l_returnflag ORDER BY l_returnflag",
     set()),
]


@pytest.fixture(scope="session")
def tpch_con():
    if not TPCH.exists():
        pytest.skip("TPC-H SF1 Parquet not found")
    con = duckdb.connect()
    con.execute(f"create view lineitem as select * from read_parquet('{TPCH}')")
    return con


@pytest.mark.slow
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("q", range(len(TPCH_QUERIES)), ids=[q[0] for q in TPCH_QUERIES])
def test_tpch_sf1_single_table_queries_equal_duckdb(probe, tpch_con, mode, q):
    name, sql, duck, rel = TPCH_QUERIES[q]
    res = run_probe(probe, sql, {"lineitem": TPCH}, mode=mode)
    got = got_rows(res)
    want = tpch_con.execute(duck).fetchall()
    assert_rows(got, want, rel, True, f"{mode} {name}")
