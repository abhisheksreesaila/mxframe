"""F-065: the SQL forms the standard TPC-H text needs besides subqueries: comma joins (FROM a, b WHERE
a.k = b.k), the same table twice under aliases, DATE literals and INTERVAL arithmetic, EXTRACT(YEAR ...),
SUBSTRING, LIKE '%x%', constant arithmetic. Each variant is checked against DuckDB; forms we cannot
translate raise instead of changing meaning.

Day-number dates: the TPC-H Parquet files store dates as int32 days since 1970-01-01, so ``d`` is such a
column here and DuckDB sees it as a DATE (``DATE '1970-01-01' + d``)."""
from datetime import date
from decimal import Decimal

import duckdb
import pyarrow as pa
import pytest

import omxcore as mx


def _days(y, m, d):
    return (date(y, m, d) - date(1970, 1, 1)).days


def _tables():
    n = pa.table({"nk": pa.array([0, 1, 2, 3], pa.int64()), "nm": pa.array(["FR", "DE", "BR", "US"]),
                  "rk": pa.array([0, 0, 1, 1], pa.int64())})
    r = pa.table({"rk": pa.array([0, 1], pa.int64()), "rn": pa.array(["EUROPE", "AMERICA"])})
    s = pa.table({"sk": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
                  "snk": pa.array([0, 1, 2, 3, 0, 9], pa.int64()),       # supplier 6: no such nation
                  "bal": pa.array([10.5, 20.0, 30.25, 40.0, 50.0, 60.0], pa.float64())})
    c = pa.table({"ck": pa.array([1, 2, 3, 4], pa.int64()), "cnk": pa.array([1, 0, 2, 3], pa.int64()),
                  "phone": pa.array(["13-555", "31-222", "23-111", "13-999"])})
    dates = [_days(1994, 1, 15), _days(1994, 12, 31), _days(1995, 1, 1), _days(1995, 3, 15), _days(1995, 3, 16),
             _days(1995, 9, 15), _days(1996, 2, 29), _days(1996, 12, 31), _days(1998, 9, 2), _days(1998, 12, 1)]
    l = pa.table({"lsk": pa.array([1, 1, 2, 3, 4, 5, 5, 6, 2, 3], pa.int64()),
                  "lck": pa.array([1, 2, 3, 4, 1, 2, 3, 4, 1, 2], pa.int64()),
                  "qty": pa.array([5.0, 12.0, 30.0, 8.0, 25.0, 3.0, 17.0, 22.0, 9.0, 40.0], pa.float64()),
                  "disc": pa.array([0.05, 0.06, 0.07, 0.05, 0.08, 0.06, 0.04, 0.07, 0.06, 0.05], pa.float64()),
                  "d": pa.array(dates, pa.int32()),
                  "note": pa.array(["a special pkg", "plain", "special requests", "xx green yy", "green", "none",
                                    "pending green", "special", "requests", "x"])})
    dd = pa.table({"k": pa.array([1, 2, 3], pa.int64()),
                   "dt": pa.array([date(1994, 5, 1), date(1995, 6, 7), date(1996, 7, 8)], pa.date32())})
    fe = pa.table({"f": pa.array([0.001, 0.0015, 0.003, 0.5], pa.float64())})
    return dict(n=n, r=r, s=s, c=c, l=l, dd=dd, fe=fe)


TABLES = _tables()


def _duck():
    con = duckdb.connect()
    for name, t in TABLES.items():
        con.register(f"_{name}", t)
        cols = ", ".join(f"DATE '1970-01-01' + {c} AS {c}" if (name == "l" and c == "d") else c
                         for c in t.column_names)
        con.execute(f"CREATE VIEW {name} AS SELECT {cols} FROM _{name}")
    return con


def _norm(x):
    if isinstance(x, (int, float, Decimal)) and not isinstance(x, bool):
        return round(float(x), 6)
    return x


def _result(con, q):
    t = con.execute(q).to_arrow_table()
    for i, typ in enumerate(t.schema.types):
        if pa.types.is_date(typ):
            t = t.set_column(i, t.schema.field(i).name, t.column(i).cast(pa.int32()))
    return t


def _check(q, nonempty=True):
    exp = _result(_duck(), q)
    got = mx.sql(q, **TABLES).compute(device="cpu")
    assert got.num_columns == exp.num_columns, q
    g = [tuple(_norm(got.column(i)[r].as_py()) for i in range(got.num_columns)) for r in range(got.num_rows)]
    e = [tuple(_norm(exp.column(i)[r].as_py()) for i in range(exp.num_columns)) for r in range(exp.num_rows)]
    if "ORDER BY" not in q:
        g, e = sorted(g, key=repr), sorted(e, key=repr)
    assert g == e, q
    if nonempty:
        assert len(e) > 0, q


COMMA = [
    "SELECT sk, nm FROM s, n WHERE snk = nk",
    "SELECT s.sk, n.nm FROM s, n WHERE s.snk = n.nk AND s.bal > 15",
    "SELECT sk, nm, rn FROM s, n, r WHERE snk = nk AND n.rk = r.rk",
    "SELECT sk, nm, rn FROM s, r, n WHERE n.rk = r.rk AND snk = nk",          # the first two are not linked
    "SELECT sk, nm, rn FROM r, n, s WHERE snk = nk AND n.rk = r.rk AND rn = 'EUROPE'",
    "SELECT nm, SUM(qty) AS q FROM l, s, n WHERE lsk = sk AND snk = nk GROUP BY nm ORDER BY nm",
    "SELECT nm, SUM(qty * (1 - disc)) AS rev FROM l, s, n WHERE lsk = sk AND snk = nk AND qty > 5 "
    "GROUP BY nm ORDER BY rev DESC",
    "SELECT ck, nm FROM c, n WHERE cnk = nk AND nk > 0",
    "SELECT COUNT(*) AS n FROM l, s WHERE lsk = sk",
    "SELECT nm, COUNT(*) AS n FROM l, s, c, n WHERE lsk = sk AND lck = ck AND snk = cnk AND cnk = nk GROUP BY nm",  # a cycle: one condition left over
    "SELECT sk, ck FROM s, c WHERE snk = cnk",
    "SELECT sk, nm FROM s, n WHERE nk = snk AND nm <> 'FR'",                  # keys written right to left
]


@pytest.mark.parametrize("q", COMMA)
def test_comma_join_takes_its_keys_from_where(q):
    _check(q)


ALIASES = [
    "SELECT s.sk, n1.nm AS sn, n2.nm AS cn FROM s, c, n n1, n n2 "
    "WHERE s.snk = n1.nk AND c.cnk = n2.nk AND s.sk = c.ck",
    "SELECT n1.nm AS sn, n2.nm AS cn, SUM(qty) AS q FROM l, s, c, n n1, n n2 "
    "WHERE lsk = sk AND lck = ck AND snk = n1.nk AND cnk = n2.nk AND "
    "((n1.nm = 'FR' AND n2.nm = 'DE') OR (n1.nm = 'DE' AND n2.nm = 'FR')) GROUP BY n1.nm, n2.nm ORDER BY sn, cn",
    "SELECT x.nm AS a, y.nm AS b FROM n x, n y WHERE x.rk = y.rk AND x.nk < y.nk",
]


@pytest.mark.parametrize("q", ALIASES)
def test_one_table_twice_under_aliases(q):
    _check(q)


def test_a_table_with_no_join_condition_is_a_cross_join():
    """F-096: nothing links the tables, so it is a cross product that the rest of WHERE filters (DuckDB does the same)."""
    _check("SELECT sk, nm FROM s, n WHERE bal > 3")
    _check("SELECT sk, rn FROM s, n, r WHERE snk = nk")
    _check("SELECT sk, nm, rn FROM s, n, r WHERE snk = nk AND bal > 20")


OR_COMMON = [         # TPC-H Q19: the join key sits in every branch of an OR
    "SELECT SUM(qty * (1 - disc)) AS rev FROM l, s WHERE (lsk = sk AND bal > 40 AND qty > 10) "
    "OR (lsk = sk AND bal < 15 AND qty < 10)",
    "SELECT COUNT(*) AS n FROM l, s WHERE (sk = lsk AND qty > 20) OR (lsk = sk AND qty < 5) OR (lsk = sk AND bal = 60)",
]


@pytest.mark.parametrize("q", OR_COMMON)
def test_a_join_key_shared_by_every_or_branch_joins(q):
    _check(q)


def test_an_or_without_a_shared_key_is_a_cross_join_with_a_filter():
    _check("SELECT COUNT(*) AS n FROM l, s WHERE (lsk = sk AND qty > 20) OR bal = 60")


DATES = [
    "SELECT lsk FROM l WHERE d <= DATE '1995-03-15'",
    "SELECT lsk FROM l WHERE d < DATE '1995-01-01' AND d >= DATE '1994-01-01'",
    "SELECT lsk FROM l WHERE d <= DATE '1998-12-01' - INTERVAL '90' DAY",
    "SELECT lsk FROM l WHERE d >= DATE '1995-01-01' AND d < DATE '1995-01-01' + INTERVAL '3' MONTH",
    "SELECT lsk FROM l WHERE d >= DATE '1994-01-01' AND d < DATE '1994-01-01' + INTERVAL '1' YEAR",
    "SELECT lsk FROM l WHERE d BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'",
    "SELECT lsk FROM l WHERE d < DATE '1996-01-31' + INTERVAL '1' MONTH",                      # month end: 29 Feb
    "SELECT lsk FROM l WHERE d > DATE '1996-03-31' - INTERVAL '1' MONTH",
    "SELECT lsk FROM l WHERE d >= DATE '1995-09-01' AND d < DATE '1995-09-01' + INTERVAL '1' MONTH",
    "SELECT lsk FROM l WHERE d <= DATE '1995-12-31' + INTERVAL '2' YEAR",
]


@pytest.mark.parametrize("q", DATES)
def test_date_literals_and_interval_arithmetic(q):
    _check(q)


def test_an_interval_unit_we_cannot_fold_raises():
    with pytest.raises(NotImplementedError, match="INTERVAL"):
        mx.sql("SELECT lsk FROM l WHERE d < DATE '1995-01-01' + INTERVAL '3' HOUR", **TABLES).compute(device="cpu")
    with pytest.raises(NotImplementedError):
        mx.sql("SELECT lsk FROM l WHERE d < d + INTERVAL '3' DAY", **TABLES).compute(device="cpu")


EXTRACT = [
    "SELECT EXTRACT(year FROM d) AS y, SUM(qty) AS q FROM l GROUP BY EXTRACT(year FROM d) ORDER BY y",
    "SELECT EXTRACT(year FROM d) AS y, COUNT(*) AS n FROM l WHERE d > DATE '1994-12-31' GROUP BY y",
    "SELECT y, SUM(q) AS t FROM (SELECT EXTRACT(year FROM d) AS y, qty AS q FROM l) AS x GROUP BY y ORDER BY y",
]


@pytest.mark.parametrize("q", EXTRACT)
def test_extract_year_of_a_day_number_date(q):
    _check(q)


def test_extract_year_of_a_date32_column():
    q = "SELECT k, EXTRACT(year FROM dt) AS y FROM dd"
    exp = duckdb.connect()
    exp.register("dd", TABLES["dd"])
    want = exp.execute(q).fetchall()
    got = mx.sql(q, **TABLES).compute(device="cpu")
    assert sorted(zip(got.column("k").to_pylist(), got.column("y").to_pylist())) == sorted(want)


def test_extract_of_another_field_raises():
    with pytest.raises(NotImplementedError, match="EXTRACT"):
        mx.sql("SELECT EXTRACT(month FROM d) AS m FROM l", **TABLES).compute(device="cpu")


SUBSTRING = [
    "SELECT ck, SUBSTRING(phone FROM 1 FOR 2) AS cc FROM c",
    "SELECT ck, SUBSTRING(phone, 1, 2) AS cc FROM c",
    "SELECT ck, SUBSTRING(phone, 4) AS rest FROM c",
    "SELECT ck FROM c WHERE SUBSTRING(phone FROM 1 FOR 2) IN ('13', '23')",
    "SELECT cc, COUNT(*) AS n FROM (SELECT SUBSTRING(phone FROM 1 FOR 2) AS cc FROM c) AS x GROUP BY cc",
]


@pytest.mark.parametrize("q", SUBSTRING)
def test_substring(q):
    _check(q)


LIKE = [
    "SELECT lsk FROM l WHERE note LIKE 'special%'",
    "SELECT lsk FROM l WHERE note LIKE '%green%'",
    "SELECT lsk FROM l WHERE note NOT LIKE '%green%'",
    "SELECT lsk FROM l WHERE note NOT LIKE 'special%'",
    # F-034: every other LIKE form
    "SELECT lsk FROM l WHERE note LIKE '%pkg'",
    "SELECT lsk FROM l WHERE note LIKE '%special%requests%'",
    "SELECT lsk FROM l WHERE note NOT LIKE '%special%requests%'",
    "SELECT lsk FROM l WHERE note LIKE 'sp_cial%'",
    "SELECT lsk FROM l WHERE note LIKE 'a%g'",
    "SELECT lsk FROM l WHERE note LIKE 'plain'",
    "SELECT lsk FROM l WHERE note LIKE '_%'",
    "SELECT lsk FROM l WHERE note ILIKE 'SPECIAL%'",
    "SELECT lsk FROM l WHERE note NOT ILIKE '%GREEN%'",
    "SELECT lsk FROM l WHERE note LIKE 'special!%' ESCAPE '!' OR note LIKE 'sp_cial_requests'",
    "SELECT lsk FROM l WHERE note NOT LIKE '%!%%' ESCAPE '!'",
    "SELECT lsk FROM l WHERE note LIKE '%green%' AND NOT note LIKE 'special%'",
]


@pytest.mark.parametrize("q", LIKE)
def test_like_forms(q):
    _check(q)


@pytest.mark.parametrize("q", ["SELECT lsk FROM l WHERE note LIKE ''",
                               "SELECT lsk FROM l WHERE note LIKE 'sp!_cial%' ESCAPE '!'"])
def test_like_forms_matching_nothing(q):
    _check(q, nonempty=False)


@pytest.mark.parametrize("sql,msg", [
    ("SELECT lsk FROM l WHERE note LIKE note", "string literal"),
    ("SELECT lsk FROM l WHERE note LIKE 'abc!' ESCAPE '!'", "escape character"),
    ("SELECT lsk FROM l WHERE note LIKE 'abc' ESCAPE '!!'", "single character"),
])
def test_unsupported_like_forms_raise(sql, msg):
    with pytest.raises((NotImplementedError, ValueError), match=msg):
        mx.sql(sql, **TABLES).compute(device="cpu")


CONSTANTS = [
    "SELECT lsk FROM l WHERE disc BETWEEN 0.06 - 0.01 AND 0.06 + 0.01",
    "SELECT lsk FROM l WHERE qty >= 1 AND qty <= 1 + 10",
    "SELECT lsk FROM l WHERE qty * 2 > 5 * 5",
    "SELECT lsk FROM l WHERE qty > -(2 + 3) * 4",
    "SELECT SUM(qty) * 0.5 AS h FROM l WHERE qty < 2 * 10 + 5",
    "SELECT lsk FROM l WHERE disc > 1e-3 * 2",
    "SELECT lsk FROM l WHERE qty > 1.5e1 * 1",
    "SELECT lsk FROM l WHERE qty < 2E1 + 3",
]


@pytest.mark.parametrize("q", CONSTANTS)
def test_constant_arithmetic_is_exact_like_sql_decimals(q):
    _check(q)


DATE32 = [
    "SELECT k FROM dd WHERE dt >= DATE '1995-01-01'",
    "SELECT k FROM dd WHERE dt < DATE '1995-06-07' + INTERVAL '1' DAY",
    "SELECT k FROM dd WHERE dt BETWEEN DATE '1994-01-01' AND DATE '1995-12-31'",
    "SELECT k FROM dd WHERE DATE '1996-01-01' <= dt",
    "SELECT k FROM dd WHERE dt = DATE '1996-07-08'",
]


@pytest.mark.parametrize("q", DATE32)
def test_date_literal_against_a_real_date32_column(q):
    _check(q)


def test_an_exponent_literal_is_not_an_integer_constant():
    """`1e-3 * 2` is 0.002 (it used to be folded to the integer 0, so every row passed)."""
    got = mx.sql("SELECT f FROM fe WHERE f > 1e-3 * 2", **TABLES).compute(device="cpu")
    assert sorted(got.column("f").to_pylist()) == [0.003, 0.5]
    _check("SELECT f FROM fe WHERE f > 1e-3 * 2")


def test_a_column_name_shared_by_two_tables_is_typed_through_its_own_table():
    """Both tables have `d`: p's is a date32, q's an int32 day number. The type comes from the table the
    query names, not from the first table that has the name."""
    p = pa.table({"k": pa.array([1, 2, 3], pa.int64()),
                  "d": pa.array([date(1994, 5, 1), date(1995, 6, 7), date(1996, 7, 8)], pa.date32())})
    q = pa.table({"k": pa.array([1, 2, 3], pa.int64()),
                  "d": pa.array([_days(1994, 5, 1), _days(1995, 6, 7), _days(1996, 7, 8)], pa.int32())})
    con = duckdb.connect()
    con.register("p", p)
    con.register("_q", q)
    con.execute("CREATE VIEW q AS SELECT k, DATE '1970-01-01' + d AS d FROM _q")
    for text in ("SELECT q.k FROM q, p WHERE q.k = p.k AND q.d >= DATE '1995-01-01'",
                 "SELECT q.k, EXTRACT(year FROM q.d) AS y FROM q",
                 "SELECT p.k FROM q, p WHERE q.k = p.k AND p.d >= DATE '1995-01-01'"):
        want = sorted(con.execute(text).fetchall())
        got = mx.sql(text, p=p, q=q).compute(device="cpu")
        assert sorted(zip(*[got.column(i).to_pylist() for i in range(got.num_columns)])) == want, text
