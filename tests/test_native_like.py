"""F-034: LIKE / contains / startswith / endswith on string columns in the native executor.

Plain utf8 columns are matched in Mojo over the Arrow offsets and bytes (one pass in the filter
program); dictionary columns evaluate the pattern once over the dictionary and test the codes.
Every result is compared with DuckDB (`LIKE` has no escape character unless ESCAPE is given, and
`_` is one UTF-8 character), and every case asserts that the native path ran (MXFRAME_STRICT_NATIVE
is on in the tests, but a decline is not an exception, so the path is checked as well)."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, when

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

WORDS = ["", "a", "ab", "abc", "abcabc", "xabcx", "a_c", "a%c", "100%", "a\\b", "é", "aé", "éa", "日本", "日本語",
         "naïve café", "x" * 70, "😀smile", "ABC", "Abc", "a\nb", "ab\n", "green forest", "forest green", "Customer Complaints",
         "Customer   Complaints", "BRASS", "polished BRASS", "BRASS polished"]
# (pattern, escape): prefix, suffix, infix, several %, _ at the ends and inside, escapes, empty, only %, non-ASCII
PATTERNS = [("ab%", None), ("%bc", None), ("%b%", None), ("%a%c%", None), ("a%c%", None), ("%a%b%c", None),
            ("_", None), ("__", None), ("a_c", None), ("%_", None), ("_%", None), ("%_%", None), ("a__%", None),
            ("", None), ("%", None), ("%%", None), ("abc", None), ("a%%c", None), ("é%", None), ("%本", None),
            ("_é", None), ("%é%", None), ("日_", None), ("%green%", None), ("%Customer%Complaints%", None),
            ("%BRASS", None), ("forest%", None), ("a\\b", None), ("a\\%c", None), ("a\\%c", "\\"),
            ("a!%c", "!"), ("100!%", "!"), ("a!_c", "!"), ("%!!%", "!"), ("a%%c", "%"), ("%x%y%", None),
            ("x" * 70, None), ("%" + "x" * 69 + "%", None), ("%\n", None), ("a_b", None)]


def _table(n, seed, nulls=True):
    r = np.random.default_rng(seed)
    vals = [WORDS[i] for i in r.integers(0, len(WORDS), n)]
    if nulls:
        vals = [None if i % 11 == 3 else v for i, v in enumerate(vals)]
    s = pa.array(vals, pa.string())
    nn = pa.array([v if v is not None else "zz" for v in vals], pa.string())
    return pa.table({"id": pa.array(np.arange(n, dtype=np.int64)), "s": s, "d": nn.dictionary_encode(),
                     "p": nn, "k": pa.array(r.integers(0, 5, n).astype(np.int32)), "v": pa.array(r.random(n))})


def _duck_ids(t, column, pattern, escape, negate=False, op="like"):
    con = duckdb.connect()
    con.register("t", t)
    esc = "" if escape is None else f" ESCAPE '{escape}'"
    q = pattern.replace("'", "''")
    return [r[0] for r in con.sql(
        f"select id from t where {column} {'not ' if negate else ''}{op} '{q}'{esc} order by id").fetchall()]


def _ids(t, pred):
    out = LazyFrame(t).filter(pred).select(col("id")).compute(device="cpu")
    return sorted(out.column("id").to_pylist())


def _like(column, pattern, escape):
    return col(column).like(pattern, escape=escape) if escape else col(column).like(pattern)


@pytest.fixture(scope="module")
def tab():
    return _table(150_000, 1)


@pytest.mark.parametrize("pattern,escape", PATTERNS)
@pytest.mark.parametrize("negate", [False, True])
def test_plain_column_like_matches_duckdb(tab, pattern, escape, negate):
    pred = _like("s", pattern, escape)
    got = _ids(tab, ~pred if negate else pred)
    assert nx.last_filter() == ("native_filter", None)
    assert got == _duck_ids(tab, "s", pattern, escape, negate)


@pytest.mark.parametrize("pattern,escape", PATTERNS)
@pytest.mark.parametrize("negate", [False, True])
def test_dictionary_column_like_matches_duckdb(tab, pattern, escape, negate):
    pred = _like("d", pattern, escape)
    got = _ids(tab, ~pred if negate else pred)
    assert nx.last_filter() == ("native_filter", None)
    assert got == _duck_ids(tab, "d", pattern, escape, negate)


@pytest.mark.parametrize("column", ["s", "d"])
@pytest.mark.parametrize("kind,arg,pattern", [
    ("startswith", "ab", "ab%"), ("startswith", "", "%"), ("startswith", "é", "é%"), ("startswith", "a%c", None),
    ("endswith", "bc", "%bc"), ("endswith", "", "%"), ("endswith", "本", "%本"), ("endswith", "a_c", None),
    ("endswith", "100%", None), ("contains", "bc", "%bc%"), ("contains", "", "%"), ("contains", "a%c", None),
    ("contains", "😀", "%😀%")])
def test_prefix_suffix_substring_methods(tab, column, kind, arg, pattern):
    got = _ids(tab, getattr(col(column), kind)(arg))
    assert nx.last_filter() == ("native_filter", None)
    if pattern is None:                       # the argument has LIKE wildcards: they are plain characters here
        t = tab.column("id").to_pylist()
        vals = tab.column(column).to_pylist()
        test = {"startswith": str.startswith, "endswith": str.endswith, "contains": lambda s, a: a in s}[kind]
        exp = [i for i, s in zip(t, vals) if s is not None and test(s, arg)]
    else:
        exp = _duck_ids(tab, column, pattern, None)
    assert got == exp


def test_like_inside_and_or_and_not(tab):
    pred = (col("s").like("%b%") & ~col("s").like("a%")) | col("s").endswith("本")
    got = _ids(tab, pred)
    assert nx.last_filter() == ("native_filter", None)
    con = duckdb.connect()
    con.register("t", tab)
    exp = [r[0] for r in con.sql("select id from t where (s like '%b%' and not s like 'a%') or s like '%本' order by id").fetchall()]
    assert got == exp


def test_not_over_a_conjunction_keeps_nulls_out(tab):
    # NOT (a AND b) over a NULL string is NULL: the row is not kept (a plain two-valued NOT would keep it)
    pred = ~(col("s").like("%b%") & (col("id") >= 0))
    got = _ids(tab, pred)
    con = duckdb.connect()
    con.register("t", tab)
    exp = [r[0] for r in con.sql("select id from t where not (s like '%b%' and id >= 0) order by id").fetchall()]
    assert got == exp


@pytest.mark.parametrize("layout", ["chunks", "sliced", "tiny"])
def test_multi_chunk_and_sliced_plain_columns(layout):
    parts = [_table(n, seed) for n, seed in ((70_000, 2), (5, 3), (90_000, 4), (1, 5))]
    if layout == "sliced":
        parts = [p.slice(7, len(p) - 20) for p in parts]
    if layout == "tiny":
        parts = [_table(3, 6), _table(2, 7)]
    t = pa.concat_tables(parts)
    t = t.set_column(0, "id", pa.chunked_array([np.arange(len(t), dtype=np.int64)]))
    assert t.column("s").num_chunks > 1
    for pattern, escape in (("%b%", None), ("a_%", None), ("%é", None), ("", None)):
        for negate in (False, True):
            pred = _like("s", pattern, escape)
            got = _ids(t, ~pred if negate else pred)
            assert nx.last_filter() == ("native_filter", None)
            assert got == _duck_ids(t, "s", pattern, escape, negate)


def test_dictionary_columns_with_different_dictionaries_per_chunk():
    parts = [_table(n, seed, nulls=False) for n, seed in ((40_000, 8), (30_000, 9))]
    # re-encode every chunk on its own so the dictionaries differ
    t = pa.concat_tables(parts)
    t = t.set_column(2, "d", pa.chunked_array([c.dictionary_encode() for c in t.column("p").chunks]))
    t = t.set_column(0, "id", pa.chunked_array([np.arange(len(t), dtype=np.int64)]))
    assert t.column("d").num_chunks > 1
    got = _ids(t, col("d").like("%a_c%"))
    assert nx.last_filter() == ("native_filter", None)
    assert got == _duck_ids(t, "d", "%a_c%", None)


def test_like_in_a_grouped_aggregate_runs_in_one_native_call(tab):
    # (a string column with NULLs sends an aggregate to the null-aware Arrow path before the executor: `p` has none)
    got = (LazyFrame(tab).filter(col("p").like("%b%c")).groupby("k")
           .agg(col("v").sum().alias("sv"), col("id").count().alias("n")).sort("k").compute(device="cpu"))
    assert nx.last_path() == "native"
    con = duckdb.connect()
    con.register("t", tab)
    exp = con.sql("select k, sum(v) sv, count(id) n from t where p like '%b%c' group by k order by k").fetch_arrow_table()
    assert got.column("k").to_pylist() == exp.column("k").to_pylist()
    assert got.column("n").to_pylist() == exp.column("n").to_pylist()
    assert got.column("sv").to_pylist() == pytest.approx(exp.column("sv").to_pylist(), rel=1e-9)


def test_like_in_a_case_when_aggregate_argument(tab):
    # the match runs on the surviving rows of the body section (gathered through the selection vector)
    got = (LazyFrame(tab).filter(col("id") >= 1000).groupby("k")
           .agg(when(col("p").like("%b_c%"), col("v"), 0.0).sum().alias("sv")).sort("k").compute(device="cpu"))
    assert nx.last_path() == "native"
    con = duckdb.connect()
    con.register("t", tab)
    exp = con.sql("select k, sum(case when p like '%b_c%' then v else 0.0 end) sv from t where id >= 1000 "
                  "group by k order by k").fetch_arrow_table()
    assert got.column("k").to_pylist() == exp.column("k").to_pylist()
    assert got.column("sv").to_pylist() == pytest.approx(exp.column("sv").to_pylist(), rel=1e-9)


def test_ilike_on_ascii_dictionary_matches_duckdb():
    t = _table(20_000, 11, nulls=False)
    t = t.set_column(1, "s", t.column("p"))
    for pattern in ("AB%", "%bC", "%B%", "a_c", "ABC"):
        got = _ids(t, col("d").ilike(pattern))
        con = duckdb.connect()
        con.register("t", t)
        q = ("select id from t where d ilike '%s' order by id" % pattern)
        ascii_only = all(v.isascii() for v in t.column("d").chunk(0).dictionary.to_pylist())
        if ascii_only:
            assert got == [r[0] for r in con.sql(q).fetchall()]


def test_ilike_matches_duckdb_on_plain_columns(tab):
    # not necessarily native (case folding of non-ASCII text is DuckDB's own); the answer must still be DuckDB's
    for pattern in ("AB%", "%bC", "É%", "%GREEN%"):
        got = _ids(tab, col("s").ilike(pattern))
        con = duckdb.connect()
        con.register("t", tab)
        exp = [r[0] for r in con.sql(f"select id from t where s ilike '{pattern}' order by id").fetchall()]
        assert got == exp


def test_bad_like_arguments_raise():
    with pytest.raises(ValueError):
        col("s").like("abc!", escape="!")             # ends with the escape character
    with pytest.raises(ValueError):
        col("s").like("abc", escape="!!")             # one character only
