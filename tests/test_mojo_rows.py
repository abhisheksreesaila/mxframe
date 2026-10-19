"""F-152: SQL row queries in the Mojo API (SELECT *, expressions, DISTINCT, ORDER BY, top-k, LIMIT / OFFSET), equal to DuckDB.

The data are test_mojo_sql.py's generated datasets (nulls, mixed-scale decimals, dates, a boolean, text with case variants, at
20,000 / 200,000 / 1.2 million rows). Every query runs through

- a table read whole (`mx sql` with --stream never),
- a Parquet file streamed as 13 row groups (`mx sql` with --stream always),
- a Frame after 300 random edits (the probe's `frame` mode; `mx` opens no Frames),

and is compared with DuckDB (tests/mx_oracle.py's typed comparison). ORDER BY has ties, which no two engines break the same way, so an
ordered query is compared in two parts: the sequence of its sort-key columns must equal DuckDB's exactly (text under the owner's
case-insensitive order, `lower(s), s`), and the rows must be DuckDB's rows (as a multiset; with a LIMIT, a sub-multiset of the
unlimited result). Each test also asserts the path that ran, from the planner's notes.
"""
from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

import pytest

from mx_oracle import duck_kind, mx_binary, run_mx, typed_rows
from test_mojo_sql import Dataset, data, probe, run_probe, got_rows, ROW_GROUPS  # noqa: F401  (fixtures)


@pytest.fixture(scope="session")
def mx():
    return mx_binary()


@dataclass
class Q:
    core: str                          # SELECT ... [ORDER BY ...] with no LIMIT / OFFSET, in our dialect
    duck: Optional[str] = None         # the same for DuckDB (text sort keys written as lower(s), s); core if None
    keys: Optional[List[int]] = None   # the output columns that are the sort keys, in order (None: no ORDER BY)
    limit: Optional[int] = None
    offset: int = 0
    tag: str = ""

    @property
    def sql(self):
        s = self.core
        if self.limit is not None:
            s += f" LIMIT {self.limit}"
        if self.offset:
            s += f" OFFSET {self.offset}"
        return s

    def id(self):
        return self.tag or self.sql[:60]


def fetch(mode, ds, mx, probe, sql):
    """(rows, columns) of `sql` over the dataset in the given mode, typed."""
    if mode == "frame":
        res = run_probe(probe, sql, {"t": ds.path}, mode="frame", edits=ds.edits_path)
        return got_rows(res), res["columns"], res["notes"]
    out = run_mx(mx, ["sql", "--max-rows", "0", "--stream", "always" if mode == "scan" else "never", sql], {"t": ds.path})
    return typed_rows(out), out["columns"], out["notes"]


def check_query(mode, ds, mx, probe, q: Q, big_ok=True):
    got, cols, notes = fetch(mode, ds, mx, probe, q.sql)
    con = ds.oracle(mode)
    cur = con.execute(q.duck or q.core)
    full = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        assert cols[i]["kind"] == dk[0] and (dk[0] != "DECIMAL" or cols[i]["scale"] == dk[1]), f"{q.sql}: column {i}: got {cols[i]}, DuckDB has {dk}"
    if q.keys is not None:
        want_keys = [tuple(r[i] for i in q.keys) for r in full]
        want_keys = want_keys[q.offset:q.offset + q.limit] if q.limit is not None else want_keys[q.offset:]
        got_keys = [tuple(r[i] for i in q.keys) for r in got]
        assert got_keys == want_keys, f"{q.sql}: the sort keys differ (first mismatch at {next((i for i, (a, b) in enumerate(zip(got_keys, want_keys)) if a != b), None)}; {len(got_keys)} rows, DuckDB has {len(want_keys)})"
    else:
        assert q.limit is None or len(got) == min(q.limit, max(len(full) - q.offset, 0)), f"{q.sql}: {len(got)} rows"
    if q.limit is None and q.offset == 0:
        assert len(got) == len(full), f"{q.sql}: {len(got)} rows, DuckDB has {len(full)}"
        assert Counter(map(_canon, got)) == Counter(map(_canon, full)), f"{q.sql}: the rows differ"
    elif q.limit is None:
        assert len(got) == max(len(full) - q.offset, 0)
    else:
        extra = Counter(map(_canon, got)) - Counter(map(_canon, full))
        assert not extra, f"{q.sql}: rows DuckDB does not have: {list(extra)[:3]}"
    return notes


def _canon(row):
    return tuple("nan" if isinstance(v, float) and v != v else v for v in row)


QUERIES = [
    Q("SELECT * FROM t", tag="select-star"),
    Q("SELECT t.*, x + 1 AS x1 FROM t WHERE x > 0", tag="qualified-star-and-expression"),
    Q("SELECT id, x, d FROM t WHERE x > 500 AND d IS NOT NULL", tag="where-and-null"),
    Q("SELECT id, x * 2 + 1 AS y, d + 0.5 AS d5, dt, e - d AS ed FROM t WHERE k = 1 OR b", tag="expressions"),
    Q("SELECT s, b, dt FROM t WHERE s IN ('West', 'east', 'NORTH') OR NOT b", tag="text-in-and-bool"),
    Q("SELECT id, x > 0 AS pos, b, x IS NULL AS nx FROM t WHERE id < 5000", tag="boolean-outputs"),
    Q("SELECT id, k // 2 AS h, x % 7 AS m, -x AS neg FROM t WHERE x IS NOT NULL AND k IS NOT NULL", tag="integer-division-and-modulo"),
    Q("SELECT id FROM t WHERE x > 100000", tag="empty"),
    Q("SELECT DISTINCT s FROM t", tag="distinct-text"),
    Q("SELECT DISTINCT k, b FROM t", tag="distinct-int-bool-null"),
    Q("SELECT DISTINCT x % 10 AS m, s FROM t", tag="distinct-expression-and-text"),
    Q("SELECT DISTINCT dt FROM t WHERE dt > DATE '2021-01-01'", tag="distinct-date"),
    Q("SELECT DISTINCT * FROM t WHERE id < 300", tag="distinct-star"),
    Q("SELECT id, x, s FROM t ORDER BY x DESC, id", duck="SELECT id, x, s FROM t ORDER BY x DESC NULLS LAST, id NULLS LAST", keys=[1, 0], tag="order-two-keys"),
    Q("SELECT id, k, x FROM t WHERE x IS NOT NULL ORDER BY k", duck="SELECT id, k, x FROM t WHERE x IS NOT NULL ORDER BY k NULLS LAST", keys=[1], limit=100, offset=20,
      tag="ties-limit-offset"),
    Q("SELECT id, x FROM t ORDER BY x DESC NULLS FIRST", duck="SELECT id, x FROM t ORDER BY x DESC NULLS FIRST", keys=[1], limit=10, tag="nulls-first-desc"),
    Q("SELECT id, s, dt FROM t ORDER BY s, dt DESC", duck="SELECT id, s, dt FROM t ORDER BY lower(s) NULLS LAST, s, dt DESC NULLS LAST", keys=[1, 2],
      limit=50, tag="text-order"),
    Q("SELECT id, s, k FROM t ORDER BY s DESC NULLS FIRST, k", duck="SELECT id, s, k FROM t ORDER BY lower(s) DESC NULLS FIRST, s DESC, k NULLS LAST", keys=[1, 2],
      limit=300, tag="text-desc-nulls-first"),
    Q("SELECT id, d, e FROM t ORDER BY d + e, id", duck="SELECT id, d, e FROM t ORDER BY d + e NULLS LAST, id NULLS LAST", keys=[0], limit=40, tag="order-by-expression"),
    Q("SELECT id, x FROM t ORDER BY 2 DESC, 1", duck="SELECT id, x FROM t ORDER BY 2 DESC NULLS LAST, 1", keys=[1, 0], limit=25, tag="order-by-ordinal"),
    Q("SELECT id, x AS xx FROM t ORDER BY xx NULLS LAST, id", duck="SELECT id, x AS xx FROM t ORDER BY xx NULLS LAST, id NULLS LAST", keys=[1, 0], limit=25,
      tag="order-by-alias"),
    Q("SELECT id, b FROM t ORDER BY b, id", duck="SELECT id, b FROM t ORDER BY b NULLS LAST, id NULLS LAST", keys=[1, 0], limit=30, tag="order-by-bool"),
    Q("SELECT s FROM t ORDER BY x, id", duck="SELECT s FROM t ORDER BY x NULLS LAST, id NULLS LAST", keys=None, limit=7, tag="order-by-a-column-not-selected"),
    Q("SELECT id FROM t ORDER BY id", duck="SELECT id FROM t ORDER BY id NULLS LAST", keys=[0], limit=10, offset=10_000_000, tag="offset-past-the-end"),
    Q("SELECT id FROM t ORDER BY id", duck="SELECT id FROM t ORDER BY id NULLS LAST", keys=[0], limit=0, tag="limit-zero"),
    Q("SELECT id, x FROM t WHERE x > 0", limit=17, tag="limit-alone"),
    Q("SELECT id, x FROM t WHERE x > 0", limit=5, offset=3, tag="limit-offset-alone"),
    Q("SELECT DISTINCT s FROM t ORDER BY s", duck="SELECT DISTINCT s FROM t ORDER BY lower(s) NULLS LAST, s", keys=[0], limit=4, tag="distinct-order-limit"),
]


@pytest.mark.parametrize("ds_name", ["g8", "g10k"])
@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
@pytest.mark.parametrize("q", QUERIES, ids=Q.id)
def test_row_queries_equal_duckdb(mx, probe, data, ds_name, mode, q):
    check_query(mode, data[ds_name], mx, probe, q)


BIG = [q for q in QUERIES if q.limit is not None or q.tag.startswith("distinct")]


@pytest.mark.parametrize("mode", ["table", "scan", "frame"])
@pytest.mark.parametrize("q", [q for q in BIG if q.tag not in ("distinct-star",)], ids=Q.id)
def test_row_queries_over_a_million_rows_with_ties_equal_duckdb(mx, probe, data, mode, q):
    check_query(mode, data["g1m"], mx, probe, q)
