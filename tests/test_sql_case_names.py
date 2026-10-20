"""F-095: SQL over tables whose column names have uppercase letters, spaces or a case-only clash.

DuckDB's rules (probed, native and registered Arrow tables alike): identifiers are case-insensitive whether
quoted or not (`"customerid"` finds `CustomerID`); a plain column keeps the SOURCE column's spelling in the
result (`SELECT customerid` returns `CustomerID`, `SELECT *` too); an alias keeps the spelling written;
an aggregate without an alias is named from the text as written (`sum(AMOUNT)`); of two columns that differ
only by case the first one is found by either spelling, and `SELECT *` names the second `A_1`.
Every query is checked against DuckDB: column names and values."""
import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx


def _t():
    return pa.table({
        "CustomerID": pa.array([1, 2, 3, 4, 5, 6, 7, 8], pa.int64()),
        "Order Date": pa.array([10, 20, 20, 30, 30, 30, 40, 50], pa.int64()),
        "Region": pa.array(["n", "s", "n", "e", "w", "n", "s", "e"], pa.string()),
        "Amount": pa.array([5, 7, 9, 11, 13, 2, 4, 6], pa.int64()),
        "mIxEd Q": pa.array([1, 0, 1, 0, 1, 0, 1, 0], pa.int64()),
    })


def _u():
    return pa.table({
        "CUSTOMERID": pa.array([2, 3, 3, 5, 9], pa.int64()),
        "Segment Name": pa.array(["gold", "silver", "bronze", "gold", "none"], pa.string()),
        "Weight": pa.array([100, 200, 300, 400, 500], pa.int64()),
    })


def _f():                                    # keys of another type (F-090): float against int
    return pa.table({"customerid": pa.array([2.0, 3.0, 5.0, 9.0], pa.float64()),
                     "Score": pa.array([1, 2, 3, 4], pa.int64())})


def _w():                                    # columns that differ from t's only by case (amount / Amount)
    return pa.table({"customerid": pa.array([1, 2, 3, 9], pa.int64()), "amount": pa.array([5, 7, 7, 13], pa.int64()),
                     "weight": pa.array([1, 2, 3, 4], pa.int64())})


def _c():                                    # names that differ only by case
    return pa.table({"a": pa.array([1, 2, 3], pa.int64()), "A": pa.array([10, 20, 30], pa.int64()),
                     "B": pa.array([5, 6, 7], pa.int64())})


def _duck(q, t=None, u=None, f=None, c=None, w=None):
    """DuckDB's answer, tables found by name as local variables (a replacement scan), the way a user runs it:
    that is the form that renames a repeated output name to `x_1`, case-insensitively."""
    exp = duckdb.sql(q).arrow()                                 # noqa: F841 (t, u, f, c, w are read by DuckDB)
    return exp.read_all() if isinstance(exp, pa.RecordBatchReader) else exp


def _check(q, **tables):
    exp = _duck(q, **tables)
    got = mx.sql(q, **tables).compute()
    assert got.column_names == exp.column_names, q
    assert got.to_pydict() == exp.to_pydict(), q


def _tables():
    return dict(t=_t(), u=_u(), f=_f(), w=_w())


# --- SELECT, aliases, qualifiers -------------------------------------------------------------------------
SELECTS = [
    "SELECT customerid, \"Order Date\", REGION FROM t ORDER BY customerid",
    "SELECT \"customerid\", \"ORDER DATE\", \"region\" FROM t ORDER BY 1",
    "SELECT CUSTOMERID, \"mixed q\" FROM t ORDER BY customerid",
    "SELECT t.customerid, T.\"order date\", t.REGION FROM t ORDER BY 1",
    "SELECT x.customerid, X.\"ORDER DATE\" FROM t AS x ORDER BY 1",
    "SELECT CUSTOMERID AS MyId, amount * 2 AS Dbl, \"order date\" AS \"Ord Day\" FROM t ORDER BY MYID",
    "SELECT customerid AS region FROM t ORDER BY region",
    "SELECT customerid, CUSTOMERID FROM t ORDER BY 1",            # a name taken twice: the second is _1
    "SELECT amount, Amount * 2 AS amount FROM t ORDER BY 1",
    "SELECT customerid + amount AS s, -AMOUNT AS n FROM t ORDER BY customerid",
    "SELECT DISTINCT region FROM t ORDER BY REGION",
    "SELECT DISTINCT REGION, \"Mixed Q\" FROM t ORDER BY region, 2",
    "SELECT customerid, CASE WHEN AMOUNT > 6 THEN 1 ELSE 0 END AS size FROM t ORDER BY CUSTOMERID",
    "SELECT customerid FROM t ORDER BY customerid LIMIT 3 OFFSET 2",
]

STARS = [
    "SELECT * FROM t ORDER BY customerid",
    "SELECT * FROM t WHERE REGION = 'n' ORDER BY CUSTOMERID",
    "SELECT *, amount + 1 AS x FROM t ORDER BY customerid",       # F-088: raised KeyError 'Field "customerid"'
    "SELECT amount + 1 AS x, * FROM t ORDER BY CUSTOMERID",
    "SELECT t.* FROM t ORDER BY customerid",
    "SELECT T.*, AMOUNT * 2 AS d FROM t ORDER BY customerid",
    "SELECT x.*, 1 AS one FROM t x ORDER BY x.customerid",
    "SELECT *, REGION AS r2 FROM t ORDER BY \"ORDER DATE\" DESC, customerid",
    "SELECT * FROM t ORDER BY CUSTOMERID DESC LIMIT 3",
    "SELECT *, customerid AS \"Order Date\" FROM t ORDER BY 1",   # the alias repeats a name: _1
    "SELECT * FROM u ORDER BY customerid, WEIGHT",
    "SELECT u.*, t.amount AS ta FROM t JOIN u ON t.customerid = u.customerid ORDER BY u.weight",
    "SELECT t.*, u.\"segment name\" FROM t JOIN u ON T.CustomerID = U.customerid ORDER BY t.customerid, u.weight",
    "SELECT * FROM t JOIN u USING (customerid) ORDER BY customerid, u.weight",
    "SELECT * FROM t JOIN u USING (CUSTOMERID) ORDER BY customerid, \"Segment Name\"",
    "SELECT * FROM t LEFT JOIN u USING (customerid) ORDER BY customerid, \"Segment Name\"",
]

# --- WHERE -----------------------------------------------------------------------------------------------
WHERES = [
    "SELECT customerid FROM t WHERE REGION = 'n' AND \"order date\" > 10 ORDER BY customerid",
    "SELECT customerid FROM t WHERE \"MIXED Q\" = 1 OR AMOUNT >= 11 ORDER BY customerid",
    "SELECT customerid FROM t WHERE CUSTOMERID IN (1, 4, 6) ORDER BY customerid",
    "SELECT customerid FROM t WHERE AMOUNT BETWEEN 5 AND 9 ORDER BY customerid",
    "SELECT customerid FROM t WHERE region LIKE 'n%' ORDER BY customerid",
    "SELECT customerid FROM t WHERE NOT (REGION = 's') AND t.AMOUNT <> 5 ORDER BY customerid",
    "SELECT customerid FROM t WHERE amount * 2 > \"ORDER DATE\" ORDER BY customerid",
    "SELECT u.customerid FROM u WHERE \"SEGMENT NAME\" = 'gold' ORDER BY 1",
]

# --- GROUP BY, HAVING, aggregates ------------------------------------------------------------------------
GROUPS = [
    "SELECT region, SUM(amount) AS total FROM t GROUP BY REGION ORDER BY region",
    "SELECT REGION, sum(AMOUNT) AS Total, count(*) AS N FROM t GROUP BY region ORDER BY REGION",
    "SELECT region, SUM(amount) FROM t GROUP BY region ORDER BY region",                 # sum(amount) as written
    "SELECT Region, SUM(AMOUNT), MAX(\"ORDER DATE\") FROM t GROUP BY 1 ORDER BY 1",      # names as written
    "SELECT region, avg(t.amount), min(\"order date\") FROM t GROUP BY region ORDER BY region",
    "SELECT \"order date\", count(*) AS n FROM t GROUP BY \"ORDER DATE\" ORDER BY \"Order Date\"",
    "SELECT region AS R, sum(amount) AS S FROM t GROUP BY R ORDER BY r",
    "SELECT REGION, \"mixed q\", SUM(amount) AS s FROM t GROUP BY region, \"MIXED Q\" ORDER BY 1, 2",
    "SELECT region, SUM(amount) AS s FROM t GROUP BY ALL ORDER BY REGION",
    "SELECT region, SUM(AMOUNT) AS s FROM t GROUP BY region HAVING SUM(amount) > 15 ORDER BY REGION",
    "SELECT region, count(*) AS n FROM t GROUP BY region HAVING REGION <> 's' AND MAX(AMOUNT) > 5 ORDER BY 1",
    "SELECT region FROM t GROUP BY region HAVING SUM(Amount) > 15 ORDER BY REGION",
    "SELECT region, SUM(amount) * 2 AS d FROM t GROUP BY region ORDER BY SUM(AMOUNT) DESC",
    "SELECT region, SUM(amount) AS s FROM t GROUP BY region ORDER BY S DESC, REGION",
    "SELECT count(*), SUM(AMOUNT), MIN(\"Order DATE\") FROM t",
    "SELECT count(DISTINCT REGION) AS n FROM t",
    "SELECT region, count(DISTINCT \"ORDER DATE\") AS n FROM t GROUP BY region ORDER BY region",
    "SELECT SUM(amount) / count(*) AS m FROM t WHERE REGION = 'n'",
]

# --- ORDER BY, hidden sort keys (F-083) ------------------------------------------------------------------
ORDERS = [
    "SELECT region FROM t ORDER BY AMOUNT, CUSTOMERID",
    "SELECT region, customerid FROM t ORDER BY \"ORDER DATE\" DESC, CUSTOMERID",
    "SELECT region AS R FROM t ORDER BY amount DESC, customerid",
    "SELECT customerid FROM t ORDER BY amount + \"mixed q\", customerid",                 # an expression key
    "SELECT customerid AS Id FROM t ORDER BY ID DESC",
    "SELECT customerid AS Id, amount AS a FROM t ORDER BY 2, ID",
    "SELECT t.customerid FROM t ORDER BY T.AMOUNT DESC, T.CUSTOMERID",
    "SELECT * FROM t ORDER BY REGION, \"ORDER DATE\" DESC, CUSTOMERID",
    "SELECT *, amount + 1 AS x FROM t ORDER BY amount + \"MIXED Q\", customerid",
    "SELECT region FROM t ORDER BY AMOUNT DESC, customerid LIMIT 3",
]

# --- JOIN ON / USING, right-key copies (F-086, F-104), keys of their own type (F-090) --------------------
JOINS = [
    "SELECT t.customerid, u.\"segment name\" FROM t JOIN u ON t.customerid = u.customerid ORDER BY 1, 2",
    "SELECT t.customerid, u.\"SEGMENT NAME\" FROM t JOIN u ON u.CUSTOMERID = T.CUSTOMERID ORDER BY 1, 2",
    "SELECT customerid, \"segment name\", t.amount AS a FROM t JOIN u USING (customerid) ORDER BY 1, 2",
    "SELECT CUSTOMERID, u.weight FROM t JOIN u USING (CUSTOMERID) ORDER BY 1, 2",
    "SELECT t.customerid, u.weight FROM t JOIN u ON t.customerid = u.customerid ORDER BY 1, 2",
    "SELECT t.customerid, u.\"segment name\" FROM t LEFT JOIN u ON t.customerid = u.customerid ORDER BY 1, 2",
    "SELECT t.customerid, u.customerid AS uc FROM t LEFT JOIN u ON t.CUSTOMERID = u.customerid ORDER BY 1, 2",
    "SELECT T.CUSTOMERID, U.CUSTOMERID AS uc, \"Segment Name\" FROM t LEFT JOIN u ON u.customerid = t.customerid ORDER BY 1, 3",
    "SELECT customerid, \"SEGMENT NAME\" FROM t LEFT JOIN u USING (customerid) ORDER BY 1, 2",
    "SELECT t.customerid, u.customerid AS uc, f.\"score\" FROM t LEFT JOIN u ON t.customerid = u.customerid "
    "LEFT JOIN f ON u.CUSTOMERID = f.CUSTOMERID ORDER BY 1, 3",                              # F-104 chain
    "SELECT t.customerid, f.customerid AS fc, f.SCORE FROM t JOIN f ON t.customerid = f.customerid ORDER BY 1",   # F-090
    "SELECT t.customerid, f.customerid AS fc FROM t LEFT JOIN f ON T.CUSTOMERID = F.CUSTOMERID ORDER BY 1",
    "SELECT t.amount, u.weight AS ua FROM t JOIN u ON t.customerid = u.customerid ORDER BY 1, 2",
    "SELECT region, SUM(u.weight) AS s FROM t JOIN u ON t.customerid = u.customerid GROUP BY REGION ORDER BY region",
    "SELECT u.\"segment name\", count(*) AS n FROM t JOIN u ON t.customerid = u.customerid "
    "GROUP BY \"SEGMENT NAME\" ORDER BY 1",
    "SELECT t.customerid, u.weight FROM t, u WHERE T.CUSTOMERID = U.customerid ORDER BY 1, 2",         # a comma join
    "SELECT customerid FROM t SEMI JOIN u ON t.customerid = u.customerid ORDER BY 1",
]

# --- subqueries, CTEs, IN / EXISTS -----------------------------------------------------------------------
SUBQUERIES = [
    "SELECT customerid FROM t WHERE AMOUNT > (SELECT AVG(amount) FROM t) ORDER BY CUSTOMERID",
    "SELECT customerid FROM t WHERE customerid IN (SELECT CUSTOMERID FROM u) ORDER BY 1",
    "SELECT customerid FROM t WHERE CUSTOMERID NOT IN (SELECT customerid FROM u WHERE \"SEGMENT NAME\" = 'gold') ORDER BY 1",
    "SELECT customerid FROM t WHERE EXISTS (SELECT 1 FROM u WHERE U.CUSTOMERID = T.customerid) ORDER BY 1",
    "SELECT customerid FROM t WHERE NOT EXISTS (SELECT 1 FROM u WHERE u.customerid = t.CustomerID) ORDER BY 1",
    "SELECT customerid FROM t WHERE amount > (SELECT AVG(u.weight) / 50 FROM u WHERE U.customerid = T.CUSTOMERID) ORDER BY 1",
    "SELECT customerid, REGION FROM t WHERE \"order date\" = (SELECT MAX(\"ORDER DATE\") FROM t) ORDER BY 1",
    "WITH Big AS (SELECT CustomerID AS Cid, Amount FROM t WHERE amount > 5) SELECT cid, AMOUNT FROM big ORDER BY CID",
    "WITH Big AS (SELECT customerid, Amount FROM t WHERE amount > 5) SELECT CUSTOMERID, AMOUNT FROM BIG ORDER BY 1",
    "WITH Big AS (SELECT * FROM t WHERE amount > 5) SELECT * FROM big ORDER BY customerid",
    "WITH s AS (SELECT region, SUM(amount) AS Total FROM t GROUP BY region) SELECT REGION, TOTAL FROM s WHERE total > 10 ORDER BY 1",
    "SELECT d.region, d.TOTAL FROM (SELECT REGION, SUM(AMOUNT) AS Total FROM t GROUP BY region) AS d WHERE TOTAL > 10 ORDER BY 1",
    "SELECT * FROM (SELECT customerid, \"Order Date\" FROM t) AS d WHERE \"ORDER DATE\" > 20 ORDER BY CUSTOMERID",
    "SELECT d.customerid FROM (SELECT * FROM t) d JOIN u ON d.CUSTOMERID = u.customerid ORDER BY 1",
    "SELECT region, (SELECT MAX(AMOUNT) FROM t) AS mx FROM t ORDER BY customerid",
]

# --- names that differ only by case ----------------------------------------------------------------------
CLASHES = [
    "SELECT a FROM c ORDER BY 1",
    "SELECT \"A\" FROM c ORDER BY 1",
    "SELECT a, b FROM c WHERE A > 1 ORDER BY b",
    "SELECT * FROM c ORDER BY b",                                        # a, A_1, B
    "SELECT c.* FROM c ORDER BY b",
    "SELECT *, b + 1 AS z FROM c ORDER BY b",
    "SELECT b, SUM(a) AS s FROM c GROUP BY b ORDER BY b",
    "SELECT a FROM c ORDER BY A DESC",
]

# --- a SELECT alias that swaps names with a column: ORDER BY takes the alias, GROUP BY the column ----------
SWAPS = [
    "SELECT amount AS customerid, customerid AS amount FROM t ORDER BY amount",
    "SELECT amount AS customerid, customerid AS amount FROM t ORDER BY AMOUNT DESC",
    "SELECT customerid AS Amount, amount AS CustomerID FROM t ORDER BY CUSTOMERID, amount",
    "SELECT customerid AS Region, region AS CustomerID FROM t ORDER BY region DESC, CUSTOMERID",
    "SELECT region AS R, SUM(amount) AS Region FROM t GROUP BY region ORDER BY region",
    "SELECT region AS r, SUM(amount) AS s FROM t GROUP BY REGION HAVING SUM(AMOUNT) > 15 ORDER BY R",
    "SELECT region, SUM(amount) AS Amount FROM t GROUP BY region HAVING SUM(amount) > 15 ORDER BY AMOUNT DESC",
    "SELECT region, SUM(amount) AS Amount FROM t GROUP BY REGION ORDER BY AMOUNT, region",
    "SELECT SUM(amount) AS Region, region AS Amount FROM t GROUP BY region ORDER BY REGION",
    "SELECT customerid AS Amount FROM t WHERE AMOUNT > 5 ORDER BY amount",             # WHERE sees the column
]

# --- a correlated subquery whose inner and outer columns differ only by case -----------------------------
CORRELATED = [
    "SELECT customerid FROM t WHERE EXISTS (SELECT 1 FROM w WHERE W.AMOUNT = T.AMOUNT) ORDER BY 1",
    "SELECT customerid FROM t WHERE EXISTS (SELECT 1 FROM w WHERE AMOUNT = t.Amount) ORDER BY 1",
    "SELECT customerid FROM t WHERE NOT EXISTS (SELECT 1 FROM w WHERE amount = T.AMOUNT) ORDER BY 1",
    "SELECT customerid FROM t WHERE \"Order Date\" > (SELECT MIN(w.WEIGHT) * 10 FROM w WHERE w.amount = t.AMOUNT) ORDER BY 1",
    "SELECT customerid FROM t WHERE customerid IN (SELECT w.CUSTOMERID FROM w WHERE w.AMOUNT = t.amount) ORDER BY 1",
]

ALL = SWAPS + CORRELATED + SELECTS + STARS + WHERES + GROUPS + ORDERS + JOINS + SUBQUERIES


@pytest.mark.parametrize("q", ALL)
def test_matches_duckdb(q):
    _check(q, **_tables())


@pytest.mark.parametrize("q", CLASHES)
def test_case_only_clash(q):
    _check(q, c=_c())


def test_named_results():
    """The spelling rules, pinned with exact values (not only equality with DuckDB)."""
    t = _t()
    r = mx.sql("SELECT customerid, \"order date\", REGION AS Reg FROM t ORDER BY 1 LIMIT 2", t=t).compute()
    assert r.column_names == ["CustomerID", "Order Date", "Reg"]
    assert r.to_pydict() == {"CustomerID": [1, 2], "Order Date": [10, 20], "Reg": ["n", "s"]}
    r = mx.sql("SELECT region, SUM(AMOUNT) FROM t GROUP BY REGION ORDER BY 1", t=t).compute()
    assert r.column_names == ["Region", "sum(AMOUNT)"]
    assert r.to_pydict() == {"Region": ["e", "n", "s", "w"], "sum(AMOUNT)": [17, 16, 11, 13]}
    r = mx.sql("SELECT * FROM c", c=_c()).compute()
    assert r.column_names == ["a", "A_1", "B"]


def test_unknown_column_still_raises():
    with pytest.raises(KeyError, match="nope"):
        mx.sql("SELECT nope FROM t", t=_t()).compute()


def test_nested_subqueries_translate_each_query_once():
    """IN over a derived table, 8 deep: names are resolved by the translator that builds each query, not by
    an extra throwaway copy (which doubled the work per level: 0.4 s -> 6 s)."""
    from omxcore import sql_frontend as sf
    q = "SELECT CUSTOMERID FROM t"
    for k in range(8):
        q = f"SELECT customerid FROM t WHERE customerid IN (SELECT customerid FROM ({q}) AS d{k})"
    builds, orig = [0], sf._SQLTranslator._build_select

    def counted(self, node):
        builds[0] += 1
        return orig(self, node)
    sf._SQLTranslator._build_select = counted
    try:
        mx.sql(q, t=_t())
    finally:
        sf._SQLTranslator._build_select = orig
    assert builds[0] == 766        # the number of queries translated before F-095 too


# --- a Parquet file with such names (F-092) --------------------------------------------------------------
@pytest.fixture
def files(tmp_path):
    out = {}
    for name, tbl in (("t", _t()), ("u", _u())):
        p = str(tmp_path / f"{name}.parquet")
        pq.write_table(tbl, p, row_group_size=3)
        out[name] = p
    return out


PARQUET = [
    "SELECT customerid, \"order date\", REGION FROM t WHERE AMOUNT > 5 ORDER BY customerid",
    "SELECT * FROM t WHERE \"ORDER DATE\" >= 30 ORDER BY CUSTOMERID",
    "SELECT *, amount + 1 AS x FROM t WHERE region = 'n' ORDER BY customerid",
    "SELECT region, SUM(AMOUNT) AS total, count(*) AS n FROM t WHERE \"Order Date\" < 50 GROUP BY REGION ORDER BY region",
    "SELECT region, SUM(amount) FROM t GROUP BY region HAVING SUM(AMOUNT) > 15 ORDER BY 1",
    "SELECT region FROM t WHERE \"MIXED q\" = 1 ORDER BY AMOUNT, customerid",
    "SELECT t.customerid, u.\"segment name\" FROM t JOIN u ON t.customerid = u.CUSTOMERID WHERE T.AMOUNT > 4 ORDER BY 1, 2",
    "SELECT customerid, \"segment name\" FROM t LEFT JOIN u USING (customerid) ORDER BY 1, 2",
    "SELECT customerid FROM t WHERE customerid IN (SELECT CUSTOMERID FROM u) ORDER BY 1",
]


@pytest.mark.parametrize("q", PARQUET)
def test_parquet(q, files):
    exp = _duck(q, t=pq.read_table(files["t"]), u=pq.read_table(files["u"]))
    for make in (mx.scan_parquet, mx.read_parquet):
        got = mx.sql(q, t=make(files["t"]), u=make(files["u"])).compute()
        assert got.column_names == exp.column_names, q
        assert got.to_pydict() == exp.to_pydict(), q
