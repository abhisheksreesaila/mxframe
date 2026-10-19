"""F-157: FLOAT64 in SQL (the Mojo package's `sql`), equal to DuckDB.

The Mojo side is mojo/tools/sql_probe.mojo (as in tests/test_mojo_sql.py). Every query runs over a table read whole and over a Parquet file
streamed as 7 row groups, on random doubles with NaN (two payloads), +-inf, -0.0, subnormals and NULLs, and equals DuckDB's answer:
exactly for keys, counts, min, max, medians and quantiles, to 1e-12 relative for sums and averages. A 32-bit float column
(Parquet FLOAT) is read widened to float64, so the oracle casts it to DOUBLE inside arithmetic; a comparison with a literal stays in 32
bits on both sides, as DuckDB does.
"""
import math
import random
import struct
from decimal import Decimal

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f157_data import NEG_NAN, POS_NAN, PAYLOAD_NAN
from test_mojo_sql import probe, run_probe, got_rows  # noqa: F401  (probe is a fixture)

ROW_GROUPS = 7
KEYS = [None, 0.0, -0.0, 1.5, 2.0, POS_NAN, NEG_NAN, PAYLOAD_NAN, math.inf, -math.inf, 1e-310, 5e-324, 3.25, -7.5]
SPECIALS = [0.0, -0.0, 1.0, -1.0, 0.5, 1e-310, -5e-324, 5e-324, 2.2250738585072014e-308, 1.7976931348623157e308,
            -1.7976931348623157e308, math.inf, -math.inf, POS_NAN, NEG_NAN, PAYLOAD_NAN, 0.1, 0.2, 0.3, 1e16]


def make_table(n, seed):
    r = random.Random(seed)
    g, k, x, y, z, d, i, f, s = [], [], [], [], [], [], [], [], []
    words = ["west", "West", "east", "North", "south"]
    for row in range(n):
        g.append(None if r.random() < 0.03 else r.randint(0, 7))
        k.append(r.choice(KEYS))
        x.append(None if r.random() < 0.06 else r.uniform(-1000, 1000))
        y.append(None if r.random() < 0.03 else r.uniform(0.01, 100))
        u = r.random()
        z.append(None if u < 0.08 else r.choice(SPECIALS) if u < 0.30 else r.uniform(-50, 50) * 10 ** r.randint(-3, 3))
        d.append(None if r.random() < 0.05 else Decimal(r.randint(-100000, 100000)).scaleb(-2))
        i.append(None if r.random() < 0.05 else r.randint(-500, 500))
        f.append(None if r.random() < 0.04 else float(np.float32(r.randint(0, 1000) / 100)))
        s.append(r.choice(words))
    return pa.table({
        "id": pa.array(range(n), pa.int64()), "g": pa.array(g, pa.int64()), "k": pa.array(k, pa.float64()),
        "x": pa.array(x, pa.float64()), "y": pa.array(y, pa.float64()), "z": pa.array(z, pa.float64()),
        "d": pa.array(d, pa.decimal128(12, 2)), "i": pa.array(i, pa.int64()), "f": pa.array(f, pa.float32()),
        "s": pa.array(s, pa.string()),
    })


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    d = tmp_path_factory.mktemp("floatsql")
    n = 60_000
    t = make_table(n, 11)
    path = d / "t.parquet"
    pq.write_table(t, path, row_group_size=-(-n // ROW_GROUPS))
    con = duckdb.connect()
    con.execute(f"create table t as select * from read_parquet('{path}')")
    return {"path": path, "con": con}


# ── comparing ────────────────────────────────────────────────────────────────────────────────────────

def same_value(got, want, rel):
    if got is None or want is None:
        return got is None and want is None
    if isinstance(want, float) or isinstance(got, float):
        g, w = float(got), float(want)
        if math.isnan(g) or math.isnan(w):
            return math.isnan(g) and math.isnan(w)
        if math.isinf(g) or math.isinf(w):
            return g == w
        if rel:
            return abs(g - w) <= rel * max(abs(w), abs(g)) or (g == 0.0 and w == 0.0)
        return g == w
    return got == want


def order_key(row):
    out = []
    for v in row:
        if v is None:
            out.append((0, 0.0, ""))
        elif isinstance(v, float):
            out.append((1, 1e400 if math.isnan(v) else 0.0 + v, ""))
        else:
            out.append((1, 0.0, repr(v)))
    return tuple(out)


def assert_rows(got, want, rel_cols, ordered, label):
    """rel_cols: the output columns compared to 1e-12 relative (the rest exactly)."""
    assert len(got) == len(want), f"{label}: {len(got)} rows, DuckDB has {len(want)}"
    if not ordered:
        got = sorted(got, key=order_key)
        want = sorted(want, key=order_key)
    for ri, (a, b) in enumerate(zip(got, want)):
        assert len(a) == len(b), f"{label}: row {ri}: {a} against {b}"
        for ci, (x, y) in enumerate(zip(a, b)):
            ok = same_value(x, y, 1e-12 if ci in rel_cols else 0.0)
            assert ok, f"{label}: row {ri} column {ci}: got {x!r}, DuckDB has {y!r}\n  got  {a}\n  want {b}"


def run(probe, data, sql, mode, duck=None, group=None):
    res = run_probe(probe, sql, {"t": data["path"]}, mode=mode, group=group)
    got = got_rows(res)
    want = data["con"].execute(duck or sql).fetchall()
    return res, got, want


MODES = ["table", "scan"]

# (sql for mx, sql for DuckDB or None, columns compared to 1e-12 relative, ordered)
GENERAL = [
    ("SELECT count(*), count(x), sum(x), avg(x), min(x), max(x) FROM t", None, {2, 3}, False),
    ("SELECT sum(y), avg(y), min(y), max(y), sum(y * 2), sum(y * y) FROM t", None, {0, 1, 4, 5}, False),
    ("SELECT count(*), sum(x) FROM t WHERE x > 10.5", None, {1}, False),
    ("SELECT count(*), sum(y) FROM t WHERE y >= 1e1 AND y < 5e1", None, {1}, False),
    ("SELECT count(*) FROM t WHERE y BETWEEN 2 AND 7.5", None, set(), False),
    ("SELECT count(*), count(z) FROM t WHERE z = 0", None, set(), False),
    ("SELECT count(*) FROM t WHERE z <> 0", None, set(), False),
    ("SELECT count(*) FROM t WHERE z = z", None, set(), False),
    ("SELECT count(*) FROM t WHERE z > 1e300", None, set(), False),
    ("SELECT count(*) FROM t WHERE z >= 1e300", None, set(), False),
    ("SELECT count(*) FROM t WHERE z < 1e300", None, set(), False),
    ("SELECT count(*) FROM t WHERE z <= -1e300", None, set(), False),
    ("SELECT count(*) FROM t WHERE NOT (z > 1)", None, set(), False),
    ("SELECT count(*) FROM t WHERE z > k", None, set(), False),
    ("SELECT count(*) FROM t WHERE z = k", None, set(), False),
    ("SELECT count(*) FROM t WHERE z <= k OR x > 500", None, set(), False),
    ("SELECT count(*) FROM t WHERE z IS NULL", None, set(), False),
    ("SELECT min(z), max(z), count(z), count(DISTINCT z) FROM t", None, set(), False),
    ("SELECT min(k), max(k), count(k), count(DISTINCT k) FROM t", None, set(), False),
    ("SELECT g, count(*), sum(y), avg(y), min(x), max(x) FROM t GROUP BY g ORDER BY g", None, {2, 3}, True),
    ("SELECT g, min(z), max(z), count(DISTINCT z) FROM t GROUP BY g ORDER BY g", None, set(), True),
    ("SELECT k, count(*), sum(y), min(z), max(z) FROM t GROUP BY k ORDER BY k", None, {2}, True),
    ("SELECT k, g, count(*), sum(y) FROM t GROUP BY k, g ORDER BY k, g", None, {3}, True),
    ("SELECT k, count(*) FROM t GROUP BY k HAVING count(*) > 4000 ORDER BY k DESC", None, set(), True),
    ("SELECT g, sum(y) FROM t GROUP BY g HAVING sum(y) > 360000 ORDER BY 2 DESC", None, {1}, True),
    ("SELECT g, sum(y) / count(*), max(y) - min(y) FROM t GROUP BY g ORDER BY g", None, {1, 2}, True),
    ("SELECT g, sum(x * 2 + y), sum(-x), max(x - y), sum(x / 4) FROM t GROUP BY g ORDER BY g", None, {1, 2, 3, 4}, True),
    ("SELECT sum(d * y), sum(i + y), sum(i * 0.5), sum(d + x) FROM t", None, {0, 1, 2, 3}, False),
    ("SELECT sum(i / 4), sum(d / 8), avg(i / 3), sum(y / 0.5) FROM t", None, {0, 1, 2, 3}, False),
    ("SELECT count(*) FROM t WHERE d < y", None, set(), False),
    ("SELECT count(*) FROM t WHERE i >= y * 4 - 100", None, set(), False),
    ("SELECT k, count(DISTINCT z), count(DISTINCT g) FROM t GROUP BY k ORDER BY k", None, set(), True),
    ("SELECT g, median(x), quantile_disc(x, 0.25), quantile_disc(x, 1), quantile_disc(x, 0), count(DISTINCT x) FROM t GROUP BY g ORDER BY g", None, set(), True),
    ("SELECT median(z), quantile_disc(z, 0.5), quantile_disc(z, 0.9), quantile_disc(z, 1) FROM t", None, set(), False),
    ("SELECT k, median(y), quantile_disc(y, 0.75) FROM t GROUP BY k ORDER BY k", None, set(), True),
    ("SELECT count(*) FROM t WHERE k IN (0, 1.5, 2e0)", None, set(), False),
    ("SELECT count(*) FROM t WHERE k NOT IN (0, 1.5)", None, set(), False),
    ("SELECT s, sum(y), max(z) FROM t GROUP BY s ORDER BY s", "SELECT s, sum(y), max(z) FROM t GROUP BY s ORDER BY lower(s), s", {1}, True),
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("q", range(len(GENERAL)), ids=[g[0][:90] for g in GENERAL])
def test_float_queries_equal_duckdb(probe, data, mode, q):
    sql, duck, rel, ordered = GENERAL[q]
    res, got, want = run(probe, data, sql, mode, duck)
    assert_rows(got, want, rel, ordered, f"{mode}: {sql}")
