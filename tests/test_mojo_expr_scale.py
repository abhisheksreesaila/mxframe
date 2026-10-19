"""F-156: the date arithmetic over every day, and the text functions over a large UTF-8 corpus, equal to DuckDB value by value.

tests/test_mojo_exprs.py compares whole result sets on 3,000 rows. This file asks for more cases than a result set can hold: DuckDB computes each expression
once, the answers are written next to the inputs in a Parquet file, and the engine counts the rows where its own answer is DISTINCT FROM DuckDB's (it must be
0, and the test checks the file really holds the values it compares with). The date file has every day from 1800 to 2400 and every 5th day from year 1 to 9999
(leap years, month ends, the century rules, dates before 1970); the text file has 60,000 words of ASCII, accents, Greek, Cyrillic, CJK, emoji, combining marks, spaces,
% and _, many of them repeated (a dictionary is made of distinct words).
"""
import datetime
import random

import duckdb
import pytest

from test_mojo_sql import probe, run_probe, got_rows  # noqa: F401  (probe is a fixture)

DAY_PARTS = ["year", "month", "day", "dow", "isodow", "doy", "quarter", "epoch", "decade", "century", "millennium"]
DATE_EXPRS = (
    {f"x_{p}": (f"EXTRACT({p} FROM dt)", f"EXTRACT({p} FROM dt)" if p == "epoch" else f"CAST(EXTRACT({p} FROM dt) AS BIGINT)") for p in DAY_PARTS}
    | {f"t_{p}": (f"date_trunc('{p}', dt)", f"CAST(date_trunc('{p}', dt) AS DATE)") for p in ("year", "quarter", "month", "week", "day", "decade", "century", "millennium")}
    | {f"m_{n}_{u}".replace("-", "n"): (f"dt + INTERVAL '{n}' {u}", f"CAST(dt + INTERVAL '{n}' {u} AS DATE)") for n, u in [(1, "MONTH"), (-1, "MONTH"), (13, "MONTH"), (12, "MONTH"), (1, "YEAR"), (-4, "YEAR"),
                                                                                                   (30, "DAY"), (-366, "DAY"), (2, "WEEK"), (100, "YEAR")]}
    | {f"d_{p}": (f"datediff('{p}', dt, dt2)", f"datediff('{p}', dt, dt2)") for p in ("day", "week", "month", "quarter", "year", "decade", "century", "millennium")}
    | {"diff": ("dt2 - dt", "dt2 - dt")}
)


@pytest.fixture(scope="session")
def date_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("datescale")
    con = duckdb.connect()
    con.execute("""create table days as
        select (date '1800-01-01' + i::int) as dt from range(0, 219146) r(i)
        union all select (date '0001-01-01' + (i * 5)::int) from range(0, 730000) r(i) where date '0001-01-01' + (i * 5)::int <= date '9999-12-31'""")
    con.execute("pragma disable_optimizer")          # (DuckDB's common-subexpression elimination treats INTERVAL '30' DAY and INTERVAL '1' MONTH as equal)
    con.execute("create table t as select dt, case when i % 9 = 0 then null else date '1950-01-01' + ((i * 7919) % 90000)::int end as dt2 from (select dt, row_number() over () as i from days)")
    cols = ", ".join(f"{duck} AS {name}" for name, (_, duck) in DATE_EXPRS.items())
    path = d / "dates.parquet"
    con.execute(f"copy (select dt, dt2, {cols} from t) to '{path}' (format parquet, row_group_size 100000)")
    return path, con


@pytest.mark.parametrize("name", list(DATE_EXPRS))
def test_every_day_equals_duckdb(probe, date_file, name):
    path, con = date_file
    expr = DATE_EXPRS[name][0]
    res = run_probe(probe, f"SELECT count(*) FROM t WHERE ({expr}) IS DISTINCT FROM {name}", {"t": path}, mode="scan")
    assert got_rows(res) == [(0,)], (name, res["rows"])
    total = run_probe(probe, f"SELECT count(*), count({name}) FROM t", {"t": path}, mode="scan")
    assert got_rows(total)[0][0] > 900000 and got_rows(total)[0][1] > 800000, total["rows"]


# ── text ─────────────────────────────────────────────────────────────────────────────────────────────

ALPHABETS = ["abcxyzAB", "aeiouAEIOU", "éÉèêëÀàÂâÇçÑñÖöÜüßẞ", "ΑΒΓΔαβγδΩωΣσς", "АБВабв", "日本語中文", "😀🙂", "́̈", " \t", "%_\\!", "0123", "İıǅǆŉſ", "-.,'"]


def random_word(rng):
    n = rng.choice([0, 1, 1, 2, 3, 4, 5, 6, 8, 12])
    alpha = "".join(rng.sample(ALPHABETS, rng.randint(1, 4)))
    return "".join(rng.choice(alpha) for _ in range(n))


TEXT_EXPRS = {
    "lower": "lower(w)", "upper": "upper(w)", "trim": "trim(w)", "ltrim": "ltrim(w)", "rtrim": "rtrim(w)", "trimc": "trim(w, 'xaé ')", "length": "length(w)",
    "reverse": "CASE WHEN w NOT LIKE '%\u0301%' AND w NOT LIKE '%\u0308%' THEN reverse(w) END", "rep1": "replace(w, 'a', 'bb')", "rep2": "replace(w, 'é', 'e')", "rep3": "replace(w, 'ab', '')", "pos1": "strpos(w, 'a')", "pos2": "strpos(w, 'é')",
    "pos3": "strpos(w, '')", "bang": "w || '!'", "pre": "'>' || w", "cat2": "w || w2", "cat3": "concat(w, w2)", "cat4": "concat(w, '-', w2)",
    "like1": "w LIKE '%a%'", "like2": "w LIKE 'a%'", "like3": "w LIKE '%a'", "like4": "w LIKE '_a_'", "like5": "w LIKE '%é_'", "like6": "w LIKE 'a%b'", "like7": "w LIKE ''",
    "like8": "w LIKE '%'", "like9": "w LIKE '_'", "like10": "w LIKE '%_%'", "like11": "w LIKE '%!%%' ESCAPE '!'", "like12": "w LIKE 'a!_%' ESCAPE '!'", "like13": "w NOT LIKE '%b%'",
    "ilike1": "w ILIKE '%É%'", "ilike2": "w ILIKE 'ΑΒ%'", "ilike3": "w ILIKE '%ß'", "ilike4": "w ILIKE 'İ%'", "ilike5": "w ILIKE 'A_B'",
    "starts": "starts_with(w, 'a')", "contains": "contains(w, 'é')", "ends": "ends_with(w, 'b')", "starts2": "starts_with(w, '')",
    "eq": "w = w2", "ne": "w <> w2", "eqlow": "w = lower(w)", "ltrim_upper": "upper(trim(w))", "lowlen": "length(lower(w))",
} | {f"sub_{s}_{n}".replace("-", "n"): (f"substr(w, {s}, {n})" if n is not None else f"substr(w, {s})") for s in (-6, -3, -1, 0, 1, 2, 5) for n in (None, -4, -1, 0, 1, 3, 5)} \
  | {f"left{k}".replace("-", "n"): f"left(w, {k})" for k in (-3, 0, 1, 4)} | {f"right{k}".replace("-", "n"): f"right(w, {k})" for k in (-3, 0, 1, 4)}


@pytest.fixture(scope="session")
def text_file(tmp_path_factory):
    d = tmp_path_factory.mktemp("textscale")
    rng = random.Random(11)
    words = [random_word(rng) for _ in range(900)]
    words += ["", "A", "a", "É", "é", "İ", "ß", "ẞ", "SS", "ǅ", "Ω", "ω", "Σ", "ς", "a b", " a", "a ", "100%", "a_b", "日本語", "😀"]
    words2 = [random_word(rng) for _ in range(400)] + words[:200]
    con = duckdb.connect()
    con.execute("create table t(w varchar, w2 varchar)")
    rows = []
    for i in range(60000):
        w = None if i % 53 == 0 else rng.choice(words)
        w2 = None if i % 59 == 0 else rng.choice(words2)
        rows.append((w, w2))
    con.executemany("insert into t values (?, ?)", rows)
    cols = ", ".join(f"{expr} AS {name}" for name, expr in TEXT_EXPRS.items())
    path = d / "text.parquet"
    con.execute(f"copy (select w, w2, {cols} from t) to '{path}' (format parquet, row_group_size 20000)")
    return path, con


@pytest.mark.parametrize("name", list(TEXT_EXPRS))
def test_every_text_function_equals_duckdb_on_a_utf8_corpus(probe, text_file, name):
    path, con = text_file
    expr = TEXT_EXPRS[name]
    res = run_probe(probe, f"SELECT count(*) FROM t WHERE ({expr}) IS DISTINCT FROM {name}", {"t": path}, mode="scan")
    assert got_rows(res) == [(0,)], (name, res["rows"], "the engine disagrees with DuckDB on this many rows")
    assert any(n.startswith("text: ") for n in res["notes"]), res["notes"]
    seen = con.execute(f"select count(*), count(distinct {name}) from (select {name} from read_parquet('{path}'))").fetchone()
    assert seen[0] == 60000 and seen[1] > 0


def test_reverse_of_a_word_that_duckdb_reverses_by_grapheme_clusters_is_an_error(probe, tmp_path):
    con = duckdb.connect()
    path = tmp_path / "g.parquet"
    con.execute(f"copy (select * from (values ('abc'), ('e\u0301a')) t(w)) to '{path}' (format parquet)")
    res = run_probe(probe, "SELECT reverse(w) FROM t", {"t": path}, expect_error=True)
    assert "reverses by grapheme clusters" in res["error"], res
    ok = run_probe(probe, "SELECT reverse(w) FROM t WHERE w = 'abc'", {"t": path})
    assert got_rows(ok) == [("cba",)]


def test_a_footer_that_lies_about_a_date_range_cannot_change_an_answer(probe, tmp_path):
    """The per-day table is built from a file's footer min / max; a row outside it is the footer lying: the query runs again without the statistics (and says so)."""
    import struct
    import pyarrow as pa
    import pyarrow.parquet as pq
    n = 20000
    days = [datetime.date(2020, 1, 1) + datetime.timedelta(days=(i * 7) % 300) for i in range(n)]
    days[15500] = datetime.date(2022, 6, 15)                                   # one date far past the others, in the last row group
    tbl = pa.table({"dt": pa.array(days, pa.date32()), "x": pa.array(range(n), pa.int64())})
    honest = tmp_path / "honest.parquet"
    pq.write_table(tbl, honest, row_group_size=5000, write_page_index=False)
    raw = bytearray(honest.read_bytes())
    flen = struct.unpack("<I", raw[-8:-4])[0]
    start = len(raw) - 8 - flen
    foot = bytes(raw[start:-8])
    big = (datetime.date(2022, 6, 15) - datetime.date(1970, 1, 1)).days
    smaller = (datetime.date(2020, 10, 25) - datetime.date(1970, 1, 1)).days
    assert foot.count(struct.pack("<i", big)) >= 1
    lying = tmp_path / "lying.parquet"
    lying.write_bytes(bytes(raw[:start]) + foot.replace(struct.pack("<i", big), struct.pack("<i", smaller)) + bytes(raw[-8:]))
    con = duckdb.connect()
    for sql in ["SELECT EXTRACT(year FROM dt) AS y, count(*), sum(x) FROM t GROUP BY 1", "SELECT count(*), sum(datediff('day', dt, DATE '2023-01-01')) FROM t",
                "SELECT EXTRACT(month FROM dt + INTERVAL 1 YEAR) AS m, count(*) FROM t GROUP BY 1"]:
        want = con.execute(sql.replace("FROM t", f"FROM read_parquet('{honest}')")).fetchall()
        for file in (honest, lying):
            for mode in ("scan", "framescan"):
                res = run_probe(probe, sql, {"t": file}, mode=mode)
                assert sorted(got_rows(res)) == sorted(want), (file.name, mode, sql, got_rows(res), want)
                again = [n for n in res["notes"] if n.startswith("statistics:")]
                assert bool(again) == (file is lying), (file.name, mode, res["notes"])
                if file is lying:
                    assert "ran again without them" in again[0], again
                else:
                    assert any(n.startswith("dates:") for n in res["notes"]), res["notes"]
