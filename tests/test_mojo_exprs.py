"""F-156: SQL expressions in the Mojo engine, equal to DuckDB: CASE, COALESCE / IFNULL / NULLIF, IS [NOT] DISTINCT FROM, IN lists of any length, CAST,
text functions (LIKE, ILIKE, lower, upper, trim, substr, length, ||, ...), dates (EXTRACT, date_trunc, date +- INTERVAL, differences), division by zero.

The Mojo side is mojo/tools/sql_probe.mojo (as in tests/test_mojo_sql.py). Every query runs over

- a table read whole (`table`),
- a Frame after 250 random edits (`frame`: sets, nulls, new words, appended and removed rows, a merge of words; the oracle applies the same edits to a
  DuckDB table),
- a Parquet file streamed as 5 row groups (`scan`) and the same edited Frame over the streamed file (`framescan`),

on data with NULLs, zeros (division), text with case and accents (UTF-8, empty words, % and _), a second dictionary (text = text), dates around leap years,
before 1970 and at month ends, and equals DuckDB's answer: kinds and decimal scales too. Text functions are asserted to have run once per dictionary
word (the notes say how many per-word tables the kernel program carries), not per row.
DuckDB's differences that are on purpose: `date + INTERVAL` and `date_trunc` give a TIMESTAMP in DuckDB, a DATE here (the oracle casts to DATE); text
order ignores case (the oracle orders by (lower(s), s)).
"""
import datetime
import random
from decimal import Decimal

import duckdb
import pyarrow.parquet as pq
import pytest

from mx_oracle import assert_equal, duck_kind
from test_mojo_sql import probe, run_probe, got_rows  # noqa: F401  (probe is a fixture)

N = 3000
ROW_GROUPS = 5
N_EDITS = 250
MODES = ["table", "frame", "scan", "framescan"]

WORDS = ["West", "east", "WEST", "Éa", "éa", "École", "école", "ÀB", "àb", "", "  padded ", "a_b", "100%", "x", "Straße", "STRASSE", "İstanbul", "日本語",
         "Ωmega", "ωmega", "naïve", "NAÏVE", "Westx", "aa", "AA"]
WORDS2 = ["west", "East", "École", "ÉCOLE", "zebra", "a_b", "x", "aa", "AA", "naïve", "ΩMEGA", "日本", "", "100%", "Éa"]
NUMS = ["12", "  7", "3.5", "-4", "+9", "1_000", ".5", "5.", "-0", "1.255", "-1.255", "100"]
NSB = ["12", "  7", "3.5", "-4", "+9", "1e2", ".5", "5.", "-0", "abc", "", "1.255", "-1.255", "99999999999999999999", "2024-02-29", "true", "NaN", "inf", " 2024-1-5 ", "No"]
SPECIAL_DATES = ["1900-02-28", "1900-03-01", "2000-02-29", "2004-02-29", "2100-02-28", "2100-03-01", "1969-12-31", "1970-01-01", "1960-02-29", "1999-12-31",
                 "2024-02-29", "2023-01-31", "2024-01-31", "2024-03-31", "2024-04-30", "2024-12-31", "1901-01-01", "1904-02-29", "1899-12-31", "2001-10-31",
                 "1996-02-29", "1800-12-31", "2400-02-29", "1582-10-15", "2023-03-31", "2023-05-31", "1971-01-31"]
FLOATS = ["0.0", "-0.0", "1.5", "-2.25", "'NaN'", "'Infinity'", "'-Infinity'", "1e300", "5e-324", "123456.789", "0.1", "-7.0", "2.5", "3.5", "-2.5"]
COLS = [("id", "int"), ("g", "int"), ("x", "int"), ("y", "int"), ("d", "dec2"), ("e", "dec4"), ("f", "float"), ("s", "text"), ("s2", "text"), ("nums", "text"),
        ("nsb", "text"), ("dt", "date"), ("dt2", "date"), ("b", "bool")]
EDITABLE = [c for c in COLS if c[0] not in ("id", "g", "f")]


def sql_list(words):
    return "[" + ", ".join("'" + w.replace("'", "''") + "'" for w in words) + "]"


def make_source(con):
    specials = sql_list(SPECIAL_DATES)
    con.execute(f"""
        create table src as select
          i::BIGINT as rid,
          i::BIGINT as id,
          case when i % 29 = 0 then null else (i % 5)::BIGINT end as g,
          case when i % 7 = 0 then null else ((i * 37) % 101 - 50)::BIGINT end as x,
          case when i % 11 = 0 then null else ((i % 6) - 1)::BIGINT end as y,
          case when i % 13 = 0 then null else ((((i * 53) % 2001) - 1000) * 0.01)::decimal(18,2) end as d,
          case when i % 17 = 0 then null else ((((i * 71) % 20001) - 10000) * 0.0001)::decimal(9,4) end as e,
          case when i % 19 = 0 then null else [{", ".join(FLOATS)}][1 + i % {len(FLOATS)}]::DOUBLE end as f,
          case when i % 23 = 0 then null else {sql_list(WORDS)}[1 + (i * 7) % {len(WORDS)}] end as s,
          case when i % 31 = 0 then null else {sql_list(WORDS2)}[1 + (i * 5) % {len(WORDS2)}] end as s2,
          case when i % 37 = 0 then null else {sql_list(NUMS)}[1 + (i * 3) % {len(NUMS)}] end as nums,
          case when i % 43 = 0 then null else {sql_list(NSB)}[1 + (i * 11) % {len(NSB)}] end as nsb,
          case when i % 41 = 0 then null when i < {len(SPECIAL_DATES)} then {specials}[1 + i]::DATE else date '1896-01-01' + ((i * 977) % 80000)::int end as dt,
          case when i % 47 = 0 then null else date '1950-01-01' + ((i * 613) % 30000)::int end as dt2,
          case when i % 14 = 0 then null else (i % 3 = 0) end as b
        from range({N}) r(i)""")


def make_edits(con):
    """A deterministic edit script (the probe's format) and the same edits applied to the DuckDB table `src`."""
    rng = random.Random(7)
    lines = []
    alive = set(range(N))
    next_rid = N
    pool = [w for w in WORDS + WORDS2 if w == w.strip() and w] + ["zzz-new", "Mid", "Ünïcode", "q"]       # (Frame.set_text trims the cell: no padded words)

    def value(name, kind):
        if kind == "int":
            return str(rng.randint(-60, 60))
        if kind == "dec2":
            return f"{rng.randint(-100000, 100000) / 100:.2f}"
        if kind == "dec4":
            return f"{rng.randint(-1000000, 1000000) / 10000:.4f}"
        if kind == "text":
            if name == "nums":
                return rng.choice([w for w in NUMS if w == w.strip()])
            if name == "nsb":
                return rng.choice([w for w in NSB if w == w.strip()])
            return rng.choice(pool)
        if kind == "date":
            return (datetime.date(1900, 1, 1) + datetime.timedelta(days=rng.randint(0, 80000))).isoformat()
        return rng.choice(["true", "false"])

    def lit(kind, v):
        if v == "":
            return "NULL"
        if kind in ("int", "dec2", "dec4", "bool"):
            return v
        if kind == "date":
            return f"DATE '{v}'"
        return "'" + v.replace("'", "''") + "'"

    for _ in range(N_EDITS):
        what = rng.randint(0, 11)
        if what <= 6 or what >= 10:
            row = rng.randrange(N)
            while row not in alive:
                row = rng.randrange(N)
            name, kind = rng.choice(EDITABLE)
            v = "" if what == 6 and kind != "text" else value(name, kind)
            if kind == "text" and what == 6:
                v = ""
            lines.append(f"set\t{name}\t{row}\t{v}")
            con.execute(f"update src set {name} = {lit(kind, v)} where rid = {row}")
        elif what in (7, 8):
            vals = []
            for name, kind in COLS:
                if name == "id":
                    vals.append(str(N + 10000 + next_rid))
                elif name in ("g", "f"):
                    vals.append("" if name == "f" else str(rng.randint(0, 4)))
                else:
                    vals.append("" if rng.random() < 0.12 else value(name, kind))
            lines.append("append\t" + "\t".join(vals))
            kinds = dict(COLS)
            con.execute(f"insert into src (rid, {', '.join(n for n, _ in COLS)}) values ({next_rid}, {', '.join(lit(kinds[n], v) for (n, _), v in zip(COLS, vals))})")
            alive.add(next_rid)
            next_rid += 1
        else:
            row = rng.randrange(N)
            if row in alive:
                lines.append(f"remove\t{row}")
                con.execute(f"delete from src where rid = {row}")
                alive.discard(row)
    lines.append("merge\ts\tWest\tzzz-new,Mid")                       # two words folded into one
    con.execute("update src set s = 'West' where s in ('zzz-new', 'Mid')")
    return "\n".join(lines) + "\n"


def define_macros(con):
    con.execute("create macro alower(s) as array_to_string(list_transform(string_split(s, ''), c -> case when c between 'A' and 'Z' then lower(c) else c end), '')")


class Data:
    def __init__(self, directory):
        self.path = directory / "t.parquet"
        self.edits_path = directory / "t.edits"
        con = duckdb.connect()
        make_source(con)
        table = con.execute("select * exclude (rid) from src order by rid").to_arrow_table()
        pq.write_table(table, self.path, row_group_size=-(-N // ROW_GROUPS))
        self.table = duckdb.connect()
        self.table.execute(f"create table t as select * from read_parquet('{self.path}')")
        self.table.execute(f"create table u as select * from read_parquet('{self.path}')")
        for c in (self.table,):
            define_macros(c)
        self.edited = duckdb.connect()
        define_macros(self.edited)
        self.edited.execute(f"create table src as select row_number() over () - 1 as rid, * from read_parquet('{self.path}')")
        self.edits_path.write_text(make_edits(self.edited))
        self.edited.execute("create view t as select * exclude (rid) from src")
        self.edited.execute(f"create table u as select * from read_parquet('{self.path}')")

    def oracle(self, mode):
        return self.edited if mode in ("frame", "framescan") else self.table


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    return Data(tmp_path_factory.mktemp("exprs"))


ALOWER = "alower"


def tcmp(a, op, b):
    """`a op b` under the owner's text order for DuckDB, which orders bytes: alphabetical ignoring ASCII case (`alower`: a macro lowering A-Z only), then bytes."""
    if op in ("=", "<>"):
        return f"({a} {op} {b})"
    strict = "<" if op in ("<", "<=") else ">"
    return f"({ALOWER}({a}) {strict} {ALOWER}({b}) OR ({ALOWER}({a}) = {ALOWER}({b}) AND {a} {op} {b}))"


def run(probe, data, sql, mode, duck=None, ordered=False, files=None, group=None):
    """`sql` through the probe over `mode` and through DuckDB (`duck` if the dialects differ): same kinds, scales and values. Returns the probe's JSON."""
    files = files or {"t": data.path}
    res = run_probe(probe, sql, files, mode=mode, edits=data.edits_path if mode in ("frame", "framescan") else None, group=group)
    cur = data.oracle(mode).execute(duck or sql)
    want = cur.fetchall()
    for i, desc in enumerate(cur.description):
        dk = duck_kind(desc[1])
        c = res["columns"][i]
        assert c["kind"] == dk[0] and (dk[0] != "DECIMAL" or c["scale"] == dk[1]), f"column {i} of `{sql}` ({mode}): got {c}, DuckDB has {dk}"
    assert_equal(got_rows(res), want, ordered, f"{mode}: {sql}")
    return res


def ran_per_word(res):
    notes = res["notes"]
    hit = [n for n in notes if n.startswith("text: ")]
    assert hit and "per-word" in hit[0] and "never per row" in hit[0], notes
    return hit[0]


# ── CASE, COALESCE, NULLIF, DISTINCT FROM, IS TRUE ───────────────────────────────────────────────────

ROW = "SELECT id, {} AS v FROM t ORDER BY id"

CASES = [
    "CASE WHEN x > 10 THEN 'hi' WHEN x < -10 THEN 'lo' END",
    "CASE x WHEN 0 THEN 'zero' WHEN 1 THEN 'one' WHEN NULL THEN 'null' ELSE 'many' END",
    "CASE WHEN y = 0 THEN NULL ELSE x // y END",
    "CASE WHEN x IS NULL THEN d WHEN x > 0 THEN d * 2 ELSE 0 END",
    "CASE WHEN x IS NULL THEN e WHEN x > 0 THEN d ELSE 7 END",
    "CASE WHEN b THEN x ELSE f END",
    "CASE WHEN b THEN CAST(x AS REAL) ELSE CAST(y AS REAL) END",
    "CASE WHEN b THEN dt ELSE DATE '2000-01-01' END",
    "CASE WHEN b THEN dt WHEN NOT b THEN dt2 END",
    "CASE WHEN b IS NULL THEN true WHEN x > 0 THEN b ELSE NOT b END",
    "CASE WHEN s = 'West' THEN s2 ELSE s END",
    "CASE WHEN s LIKE 'W%' THEN lower(s) WHEN s2 IS NULL THEN 'none' ELSE upper(s2) END",
    "CASE WHEN x > 0 THEN 'a' ELSE 'b' END",
    "CASE WHEN x > 0 THEN 1 WHEN x < 0 THEN 2.5 ELSE 3 END",
    "CASE WHEN x > 5 THEN x WHEN y > 2 THEN y END",
    "COALESCE(x, y, 99)",
    "COALESCE(s, s2, 'none')",
    "COALESCE(d, 0)",
    "COALESCE(d, e, 1)",
    "COALESCE(dt, DATE '1970-01-01')",
    "COALESCE(b, false)",
    "COALESCE(f, 0)",
    "COALESCE(x, f)",
    "IFNULL(s, 'n/a')",
    "IFNULL(y, x)",
    "NULLIF(x, 0)",
    "NULLIF(s, 'West')",
    "NULLIF(s, s2)",
    "NULLIF(d, 0)",
    "NULLIF(dt, DATE '2000-02-29')",
    "NULLIF(b, true)",
    "NULLIF(f, 0)",
    "x IS DISTINCT FROM y",
    "x IS NOT DISTINCT FROM y",
    "s IS DISTINCT FROM s2",
    "s IS NOT DISTINCT FROM 'West'",
    "f IS NOT DISTINCT FROM f",
    "f IS DISTINCT FROM 1.5",
    "dt IS DISTINCT FROM dt2",
    "d IS DISTINCT FROM e",
    "x IS DISTINCT FROM NULL",
    "b IS TRUE",
    "b IS NOT TRUE",
    "b IS FALSE",
    "b IS NOT FALSE",
    "(x > 5) IS TRUE",
    "(x > 5) IS NOT FALSE",
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expr", CASES)
def test_case_coalesce_nullif_distinct(probe, data, mode, expr):
    run(probe, data, ROW.format(expr), mode, ordered=True)


@pytest.mark.parametrize("mode", MODES)
def test_conditional_aggregates_keys_filters_and_the_stage_after_aggregation(probe, data, mode):
    for sql, duck, ordered in [
        ("SELECT g, sum(CASE WHEN x > 0 THEN d ELSE 0 END), count(CASE WHEN y = 0 THEN 1 END), max(COALESCE(x, -100)), min(NULLIF(y, 0)) FROM t GROUP BY g", None, False),
        ("SELECT CASE WHEN x > 0 THEN 'pos' WHEN x < 0 THEN 'neg' ELSE 'zero' END AS sign, count(*), sum(d) FROM t GROUP BY 1", None, False),
        ("SELECT COALESCE(g, -1) AS gg, count(*) FROM t GROUP BY COALESCE(g, -1)", None, False),
        ("SELECT CASE WHEN dt < DATE '1970-01-01' THEN 'old' ELSE 'new' END AS era, count(*), count(dt) FROM t GROUP BY 1", None, False),
        ("SELECT count(*) FROM t WHERE CASE WHEN y = 0 THEN x > 3 ELSE x < -3 END", None, False),
        ("SELECT count(*) FROM t WHERE COALESCE(x, 0) > 5 AND x IS DISTINCT FROM y", None, False),
        ("SELECT count(*) FROM t WHERE NULLIF(x, 3) IS NULL", None, False),
        ("SELECT g, CASE WHEN count(*) > 400 THEN 'big' ELSE 'small' END, COALESCE(sum(x), 0), NULLIF(max(y), 4) FROM t GROUP BY g", None, False),
        ("SELECT g, sum(x) FROM t GROUP BY g HAVING CASE WHEN g IS NULL THEN sum(x) > 0 ELSE count(*) > 100 END", None, False),
        ("SELECT s, count(*) FROM t WHERE CASE WHEN s IS NULL THEN true ELSE s <> 'West' END GROUP BY s", None, False),
        ("SELECT id, x FROM t ORDER BY CASE WHEN x IS NULL THEN 1 ELSE 0 END, x DESC, id LIMIT 40", None, True),
        ("SELECT 'a' AS k, count(*) FROM t", None, False),
        ("SELECT CASE WHEN x > 0 THEN 'p' END AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT NULLIF(x, 0) AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT COALESCE(s, s2) AS k, count(*), sum(d) FROM t GROUP BY 1", None, False),
        ("SELECT TRY_CAST(nsb AS INTEGER) AS k, count(*) FROM t WHERE nsb IN ('12', 'abc', '', '3.5', '-4') GROUP BY 1", None, False),
        ("SELECT s || s2 AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT EXTRACT(year FROM dt) AS k, count(*) FROM t GROUP BY 1", "SELECT CAST(EXTRACT(year FROM dt) AS BIGINT) AS k, count(*) FROM t GROUP BY 1", False),
        ("SELECT dt + INTERVAL 1 MONTH AS k, count(*) FROM t WHERE id < 400 GROUP BY 1", "SELECT CAST(dt + INTERVAL 1 MONTH AS DATE) AS k, count(*) FROM t WHERE id < 400 GROUP BY 1", False),
        ("SELECT b AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT id, 'tag' AS k FROM t WHERE id < 10 ORDER BY id", None, True),
    ]:
        run(probe, data, sql, mode, duck, ordered)


# ── IN lists of any length ──────────────────────────────────────────────────────────────────────────

BIG_INTS = ",".join(str((i * 7919) % 100003 - 50000) for i in range(400)) + ",-9223372036854775808,9223372036854775807,4611686018427387904"
BIG_DEC = ",".join(f"{((i * 31) % 4001 - 2000) / 100:.2f}" for i in range(120))
BIG_DATES = ",".join(f"DATE '{(datetime.date(1990, 1, 1) + datetime.timedelta(days=i * 61)).isoformat()}'" for i in range(150))
BIG_WORDS = ",".join("'" + w + "'" for w in WORDS + WORDS2) + ",'nothing'"
BIG_FLOATS = ",".join(f"{i * 0.25 - 10}" for i in range(80)) + ",1e300,5e-324"
IN_LISTS = [
    f"x IN ({BIG_INTS})", f"x NOT IN ({BIG_INTS})", f"id IN ({BIG_INTS},2999,1500,7)", f"d IN ({BIG_DEC})", f"d NOT IN ({BIG_DEC},NULL)", f"dt IN ({BIG_DATES})",
    f"s IN ({BIG_WORDS})", f"f IN ({BIG_FLOATS})", f"f IN ({BIG_FLOATS},'NaN'::DOUBLE)", f"x IN (1,2,3,4,5,6,7,8,9,10,11,12,13)", f"y IN ({BIG_INTS})",
    "x IN (1, NULL, 3)", "x IN (0)", f"x IN ({BIG_INTS},NULL)", "x IN (1,2,3,4,5,6,7,8,9,10,NULL)",
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("cond", IN_LISTS)
def test_in_lists_have_no_length_cap(probe, data, mode, cond):
    run(probe, data, f"SELECT id, {cond} AS v FROM t ORDER BY id", mode, ordered=True)
    run(probe, data, f"SELECT count(*) FROM t WHERE {cond}", mode)


def test_a_long_in_list_runs_through_the_kernels_hash_set(probe, data):
    res = run(probe, data, f"SELECT count(*) FROM t WHERE x IN ({BIG_INTS})", "table")
    assert any(n.startswith("kernel:") and " sets" in n for n in res["notes"]), res["notes"]


# ── arithmetic: division and modulo by zero, scales ──────────────────────────────────────────────────

ARITH = [
    "x / y", "x // y", "x % y", "d / y", "d % y", "d % e", "d + e", "d - e", "d * e", "d * y", "d / e", "x / 0", "x % 0", "x // 0", "7 / x", "7.5 / x", "7 // x", "7 % x",
    "-x // y", "-x % y", "x * y", "x + d", "d % 3", "d % 0", "e % d", "f / y", "f / 0", "f * x", "f + d", "y / y", "x // -3", "x % -3", "d % -3", "(x + y) % 4",
    "f // y", "f // 0", "f // 0.5", "d // y", "d // 3", "d // 0", "x // d", "7.5 // x", "x // 2.0", "d // e",
    "CASE WHEN y <> 0 THEN x / y END",
]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expr", ARITH)
def test_division_modulo_and_decimal_scales_as_duckdb(probe, data, mode, expr):
    run(probe, data, ROW.format(expr), mode, ordered=True)


# ── CAST ─────────────────────────────────────────────────────────────────────────────────────────────

CASTS = [
    "CAST(x AS DECIMAL(10,2))", "CAST(x AS DOUBLE)", "CAST(x AS BIGINT)", "CAST(x AS TINYINT)", "CAST(x AS BOOLEAN)", "CAST(x AS REAL)", "CAST(y AS DECIMAL(6,3))",
    "CAST(d AS INTEGER)", "CAST(d AS DECIMAL(10,1))", "CAST(d AS DECIMAL(18,4))", "CAST(d AS DECIMAL(18,0))", "CAST(d AS DOUBLE)", "CAST(d AS BOOLEAN)",
    "CAST(e AS DECIMAL(5,2))", "CAST(e AS DECIMAL(9,1))", "CAST(e AS INTEGER)", "CAST(e AS DOUBLE)", "CAST(e AS REAL)",
    "TRY_CAST(d AS DECIMAL(4,1))", "TRY_CAST(d AS TINYINT)", "TRY_CAST(x * 1000 AS TINYINT)", "TRY_CAST(x * 1000 AS SMALLINT)", "TRY_CAST(d * 1000000 AS DECIMAL(5,2))",
    "TRY_CAST(f AS INTEGER)", "TRY_CAST(f AS DECIMAL(10,2))", "TRY_CAST(f AS BIGINT)", "TRY_CAST(f AS SMALLINT)", "TRY_CAST(f AS REAL)", "CAST(f AS DOUBLE)", "CAST(f AS BOOLEAN)",
    "CAST(CASE WHEN f BETWEEN -1e30 AND 1e30 THEN f END AS REAL)", "CAST(CAST(CASE WHEN f BETWEEN -1e30 AND 1e30 THEN f END AS REAL) AS DOUBLE)", "CAST(b AS INTEGER)", "CAST(b AS DOUBLE)", "CAST(b AS DECIMAL(5,1))", "CAST(b AS VARCHAR)",
    "CAST(dt AS DATE)", "CAST(s AS VARCHAR)", "CAST(x AS VARCHAR)", "CAST(y AS VARCHAR)", "CAST(d AS VARCHAR)", "CAST(e AS VARCHAR)", "CAST(dt AS VARCHAR)", "dt::VARCHAR || '!'", "CAST(x AS VARCHAR) || ':' || s",
    "CAST(y AS VARCHAR) = '0'", "CAST(dt AS VARCHAR) LIKE '19%'", "length(CAST(d AS VARCHAR))", "CAST(x > 5 AS INTEGER)", "CAST(x > 5 AS VARCHAR)", "x::DECIMAL(8,2)", "d::INTEGER", "(x + 1)::DOUBLE", "x::BOOLEAN",
    "CAST(nums AS INTEGER)", "CAST(nums AS DECIMAL(10,2))", "CAST(nums AS DOUBLE)", "CAST(nums AS REAL)", "CAST(nums AS DECIMAL(7,1))",
    "TRY_CAST(nsb AS INTEGER)", "TRY_CAST(nsb AS BIGINT)", "TRY_CAST(CASE WHEN nsb <> '1e2' THEN nsb END AS DECIMAL(10,2))", "TRY_CAST(nsb AS DOUBLE)", "TRY_CAST(CASE WHEN nsb NOT IN ('NaN', 'inf', 'No', 'true') THEN nsb END AS DATE)", "TRY_CAST(nsb AS BOOLEAN)",
    "TRY_CAST(nsb AS TINYINT)", "TRY_CAST(s AS INTEGER)", "TRY_CAST(CASE WHEN s IN ('West', 'WEST', 'Éa') THEN s END AS DATE)", "TRY_CAST(s AS DOUBLE)",
    "TRY_CAST(lower(nsb) AS BOOLEAN)",
]


def duck_cast_sql(expr):
    return expr


def test_a_float_remainder_is_refused_by_name(probe, data):
    res = run_probe(probe, "SELECT f % 3 FROM t", {"t": data.path}, expect_error=True)
    assert "`%` on a FLOAT value is not supported" in res["error"], res


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expr", CASTS)
def test_casts_as_duckdb(probe, data, mode, expr):
    res = run(probe, data, ROW.format(expr), mode, ordered=True)
    if "nums" in expr or "nsb" in expr or "(s AS INTEGER" in expr or "s END AS DATE" in expr or "(s AS DOUBLE" in expr or "AS VARCHAR" in expr and "(b AS" not in expr and "(s AS" not in expr and "> 5 AS" not in expr or "::VARCHAR" in expr:
        ran_per_word(res)


@pytest.mark.parametrize("mode", MODES)
def test_a_cast_that_cannot_hold_the_value_raises(probe, data, mode):
    for sql, why in [("SELECT CAST(x * 1000 AS TINYINT) FROM t", "could not be converted"), ("SELECT CAST(nsb AS INTEGER) FROM t", "could not be converted"),
                     ("SELECT CAST(d * 1000000 AS DECIMAL(5,2)) FROM t", "could not be converted"), ("SELECT CAST(f AS INTEGER) FROM t", "could not be converted")]:
        res = run_probe(probe, sql, {"t": data.path}, mode=mode, edits=data.edits_path if mode in ("frame", "framescan") else None, expect_error=True)
        assert why in res["error"], (sql, res)
        with pytest.raises(duckdb.Error):
            data.oracle(mode).execute(sql)


@pytest.mark.parametrize("mode", MODES)
def test_a_word_with_a_quirk_in_duckdb_is_an_error_not_a_guess(probe, data, mode):
    """`inf` is a date in DuckDB (infinity), which MX frame has no value for: a CAST and a TRY_CAST both say so instead of answering NULL."""
    for kind in ("CAST", "TRY_CAST"):
        res = run_probe(probe, f"SELECT {kind}(nsb AS DATE) FROM t", {"t": data.path}, mode=mode, edits=data.edits_path if mode in ("frame", "framescan") else None, expect_error=True)
        assert "quirks that MX frame does not read" in res["error"], res


@pytest.mark.parametrize("mode", MODES)
def test_a_cast_in_a_branch_no_row_takes_does_not_raise(probe, data, mode):
    sql = "SELECT id, CASE WHEN nsb IN ('12', '  7', '3.5') THEN CAST(nsb AS INTEGER) WHEN x > 1000 THEN CAST(x * 1000 AS TINYINT) END FROM t ORDER BY id"
    run(probe, data, sql, mode, ordered=True)


# ── dates ────────────────────────────────────────────────────────────────────────────────────────────

PARTS = ["year", "month", "day", "dow", "isodow", "doy", "quarter", "epoch", "decade", "century", "millennium"]
DATE_EXPRS = (
    [f"EXTRACT({p} FROM dt)" for p in PARTS] + [f"date_part('{p}', dt)" for p in ("year", "dow")] + ["year(dt)", "month(dt)", "day(dt)", "quarter(dt)", "dayofweek(dt)", "dayofyear(dt)"]
    + [f"date_trunc('{p}', dt)" for p in ("year", "quarter", "month", "week", "day", "decade", "century", "millennium")]
    + [f"dt {op} INTERVAL {n} {u}" for op in ("+", "-") for n, u in [(1, "MONTH"), (3, "MONTH"), (12, "MONTH"), (1, "YEAR"), (30, "DAY"), (2, "WEEK"), (13, "MONTH"), (400, "MONTH")]]
    + ["dt + INTERVAL '1' MONTH", "dt - INTERVAL '2 months'", "dt + INTERVAL (y) DAY", "dt + INTERVAL (x) MONTH", "dt + INTERVAL (y + 1) YEAR", "INTERVAL 5 DAY + dt"]
    + ["dt - dt2", "dt - 100", "dt + 7", "dt2 - dt"] + [f"datediff('{p}', dt, dt2)" for p in ("day", "week", "month", "quarter", "year", "decade", "century")]
    + [f"date_diff('{p}', dt2, dt)" for p in ("day", "week", "month")] + ["EXTRACT(year FROM dt) * 100 + EXTRACT(month FROM dt)", "date_trunc('month', dt + INTERVAL 1 MONTH)"]
)
CAST_DATE = ("date_trunc(", "INTERVAL")


def date_duck(expr):
    if expr.startswith("date_trunc(") or "INTERVAL" in expr:
        return f"CAST({expr} AS DATE)"
    if expr.startswith("EXTRACT(") or expr.startswith("date_part("):
        return f"CAST({expr} AS BIGINT)"
    return expr


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expr", DATE_EXPRS)
def test_dates_as_duckdb(probe, data, mode, expr):
    run(probe, data, ROW.format(expr), mode, duck=f"SELECT id, {date_duck(expr)} AS v FROM t ORDER BY id", ordered=True)


@pytest.mark.parametrize("mode", MODES)
def test_dates_in_filters_groups_and_aggregates(probe, data, mode):
    for sql, duck in [
        ("SELECT EXTRACT(year FROM dt) AS y, count(*), min(dt), max(dt) FROM t GROUP BY 1", None),
        ("SELECT date_trunc('month', dt) AS m, count(*) FROM t GROUP BY 1", "SELECT CAST(date_trunc('month', dt) AS DATE) AS m, count(*) FROM t GROUP BY 1"),
        ("SELECT count(*) FROM t WHERE dt >= DATE '2000-01-01' - INTERVAL '90' DAY AND dt < DATE '2000-01-31' + INTERVAL 1 MONTH", None),
        ("SELECT count(*) FROM t WHERE dt + INTERVAL 1 MONTH > DATE '2024-03-01'", None),
        ("SELECT count(*) FROM t WHERE EXTRACT(dow FROM dt) = 0 AND EXTRACT(month FROM dt) IN (1, 2, 12)", None),
        ("SELECT sum(datediff('day', dt, dt2)), max(dt2 - dt), min(dt - dt2) FROM t", None),
        ("SELECT g, CASE WHEN EXTRACT(year FROM dt) < 1970 THEN 'old' ELSE 'new' END AS era, count(*) FROM t GROUP BY 1, 2", None),
    ]:
        run(probe, data, sql, mode, duck)


def test_date_constants_are_folded(probe, data):
    res = run(probe, data, "SELECT count(*) FROM t WHERE dt <= DATE '1998-12-01' - INTERVAL '90' DAY", "table",
              duck="SELECT count(*) FROM t WHERE dt <= CAST(DATE '1998-12-01' - INTERVAL '90' DAY AS DATE)")
    assert res["rows"][0][0] > 0


# ── text ─────────────────────────────────────────────────────────────────────────────────────────────

TEXT_VALUE = [
    "lower(s)", "upper(s)", "trim(s)", "ltrim(s)", "rtrim(s)", "trim(s, 'xX ')", "length(s)", "substr(s, 2)", "substr(s, 2, 3)", "substr(s, -3, 2)", "substr(s, 0, 2)",
    "substr(s, 3, -1)", "substr(s, -2, -1)", "substring(s, 2, 2)", "substring(s from 2 for 2)", "left(s, 2)", "right(s, 2)", "left(s, -1)", "reverse(s)", "replace(s, 'a', 'AA')",
    "strpos(s, 'a')", "s || '!'", "'>' || s", "s || s2", "s2 || s", "s || '-' || s2", "concat(s, s2)", "concat(s, '-', s2)", "concat(s, 1)", "s || 1",
    "lower(s) || upper(s2)", "length(lower(s)) + length(s2)", "upper(substr(s, 1, 1)) || lower(substr(s, 2))", "lower(upper(s))", "length(s || s2)",
    "s LIKE 'W%'", "s LIKE '%a%'", "s ILIKE 'é%'", "s ILIKE '%ÉA%'", "s NOT LIKE '_a'", "s LIKE '100!%' ESCAPE '!'", "s LIKE 'a!_b' ESCAPE '!'", "s LIKE '__'", "s LIKE ''", "s LIKE '%'",
    "s ILIKE 'STRASSE'", "s ILIKE 'straße'", "s LIKE '日%'", "s LIKE '_______'", "s2 ILIKE '%é%'", "lower(s) LIKE 'west%'",
    "starts_with(s, 'W')", "contains(s, 'a')", "ends_with(s, 'a')", "prefix(s, 'É')", "suffix(s, 'b')",
    "s = s2", "s <> s2", "s = lower(s)", "s = upper(s)", "s IN ('West', 'east', 'xx')", "lower(s) IN ('west', 'east')", "upper(s) NOT IN ('WEST', 'EAST', 'ÉA')",
]


def text_duck(expr):
    out = expr
    for a, op, b in [("s", "<", "s2"), ("s2", ">", "s")]:
        pass
    return out


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("expr", TEXT_VALUE)
def test_text_functions_once_per_word(probe, data, mode, expr):
    duck = f"SELECT id, {expr} AS v FROM t ORDER BY id"
    res = run(probe, data, ROW.format(expr), mode, duck=duck, ordered=True)
    if expr not in ("s = s2", "s <> s2", "s IN ('West', 'east', 'xx')", "s LIKE ''") and "NOT IN" not in expr and "IN (" not in expr:
        ran_per_word(res)


TEXT_ORDER = [("s", "<", "s2"), ("s", "<=", "s2"), ("s", ">", "s2"), ("s", ">=", "s2"), ("s2", "<", "s"), ("lower(s)", "<", "s2"), ("s", "<", "'Zebra'"), ("s", ">=", "'east'"),
              ("upper(s)", ">", "lower(s2)")]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("a,op,b", TEXT_ORDER)
def test_text_order_between_two_values_ignores_case(probe, data, mode, a, op, b):
    res = run(probe, data, f"SELECT id, {a} {op} {b} AS v FROM t ORDER BY id", mode, duck=f"SELECT id, {tcmp(a, op, b)} AS v FROM t ORDER BY id", ordered=True)
    assert any(n.startswith("kernel") for n in res["notes"])


@pytest.mark.parametrize("mode", MODES)
def test_text_in_filters_groups_aggregates_and_sorts(probe, data, mode):
    for sql, duck, ordered in [
        ("SELECT lower(s) AS k, count(*), sum(x) FROM t GROUP BY lower(s)", None, False),
        ("SELECT upper(s) AS k, count(DISTINCT y) FROM t GROUP BY 1", None, False),
        ("SELECT count(*) FROM t WHERE lower(s) = 'west'", None, False),
        ("SELECT count(*) FROM t WHERE s LIKE '%a%' AND length(s) > 2", None, False),
        ("SELECT count(*) FROM t WHERE s = s2", None, False),
        ("SELECT count(*) FROM t WHERE s < s2", f"SELECT count(*) FROM t WHERE {tcmp('s', '<', 's2')}", False),
        ("SELECT max(length(s)), min(length(s2)), sum(length(s || s2)), count(lower(s)) FROM t", None, False),
        ("SELECT length(s) AS n, count(*) FROM t GROUP BY n", None, False),
        ("SELECT left(s, 1) AS c, count(*), max(x) FROM t GROUP BY c", None, False),
        ("SELECT s || s2 AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT CASE WHEN s LIKE 'W%' THEN 'w' WHEN s LIKE 'é%' THEN 'e' ELSE 'other' END AS k, count(*) FROM t GROUP BY 1", None, False),
        ("SELECT id, lower(s) AS l FROM t ORDER BY lower(s), s, id LIMIT 30", "SELECT id, lower(s) AS l FROM t ORDER BY lower(s) NULLS LAST, s NULLS LAST, id LIMIT 30", True),
        ("SELECT id, s || s2 AS c FROM t ORDER BY 2, id LIMIT 30", "SELECT id, s || s2 AS c FROM t ORDER BY lower(s || s2) NULLS LAST, s || s2 NULLS LAST, id LIMIT 30", True),
        ("SELECT DISTINCT lower(s) FROM t", None, False),
        ("SELECT g, upper(s) AS u, count(*) FROM t GROUP BY g, u", None, False),
        ("SELECT s, upper(s) AS u, length(s) FROM t GROUP BY s", None, False),
        ("SELECT upper(s), count(*) FROM t GROUP BY s", None, False),
        ("SELECT g, min(lower(s)), max(upper(s)), count(DISTINCT lower(s)), count(lower(s)) FROM t GROUP BY g",
         "SELECT g, arg_min(lower(s), (alower(lower(s)), lower(s))), arg_max(upper(s), (alower(upper(s)), upper(s))), count(DISTINCT lower(s)), count(lower(s)) FROM t GROUP BY g", False),
        ("SELECT median(lower(s)), quantile_disc(upper(s), 0.3) FROM t",
         "SELECT (list(lower(s) ORDER BY alower(lower(s)), lower(s)) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.5 * count(s))::int, 1)], "
         "(list(upper(s) ORDER BY alower(upper(s)), upper(s)) FILTER (WHERE s IS NOT NULL))[greatest(ceil(0.3 * count(s))::int, 1)] FROM t", False),
        ("SELECT g, length(min(s)) FROM t GROUP BY g", "SELECT g, length(arg_min(s, (lower(s), s))) FROM t GROUP BY g", False),
    ]:
        res = run(probe, data, sql, mode, duck, ordered)


def test_text_equality_between_columns_of_two_tables_after_a_join(probe, data):
    files = {"t": data.path, "u": data.path}
    for sql in ["SELECT CASE WHEN a.x > b.x THEN a.s ELSE b.s2 END AS k, count(*) FROM t a JOIN u b ON a.id = b.id GROUP BY 1",
                "SELECT lower(b.s2) AS k, sum(a.d), count(*) FROM t a JOIN u b ON a.g = b.g AND a.y = b.y WHERE a.s LIKE '%a%' AND a.id < 500 GROUP BY 1",
                "SELECT count(*) FROM t JOIN u ON t.id = u.id WHERE t.s = u.s2",
                "SELECT count(*) FROM t a JOIN t b ON a.g = b.g AND a.y = b.y WHERE a.s <> b.s AND a.id < 300",
                "SELECT a.g, count(*) FROM t a JOIN t b ON a.g = b.g AND a.y = b.y WHERE a.s = b.s2 AND a.id < 400 GROUP BY a.g",
                "SELECT count(*) FROM t JOIN u ON t.id = u.id WHERE lower(t.s) = lower(u.s2)",
                "SELECT count(*) FROM t LEFT JOIN u ON t.id = u.id AND t.s2 = u.s WHERE u.id IS NULL",
                "SELECT count(*) FROM t JOIN u ON t.x = u.x WHERE t.s2 = u.s2 AND t.id < u.id"]:
        for mode in ["table", "scan"]:
            run(probe, data, sql, mode, files=files)


# ── the way a program is built ───────────────────────────────────────────────────────────────────────

def test_a_text_function_costs_one_table_not_a_pass_over_rows(probe, data):
    res = run(probe, data, "SELECT count(*) FROM t WHERE lower(s) = 'west' AND s LIKE '%a%'", "table")
    text = ran_per_word(res)
    assert text.startswith("text: 2 per-word tables over ") or text.startswith("text: 1 per-word table over "), text


def test_past_the_register_limit_the_query_says_so(probe, data):
    parts = ", ".join(f"sum(CASE WHEN x > {i} THEN d * {i + 1} + COALESCE(y, {i}) ELSE e END)" for i in range(40))
    res = run_probe(probe, f"SELECT {parts} FROM t", {"t": data.path}, expect_error=True)
    assert "the query needs more than 48 registers" in res["error"], res


def test_two_big_dictionaries_are_not_multiplied_silently(probe, data):
    res = run_probe(probe, "SELECT CAST(dt AS VARCHAR) || CAST(dt2 AS VARCHAR) FROM t", {"t": data.path}, expect_error=True)
    assert "combinations" in res["error"] and "never per row" in res["error"], res
