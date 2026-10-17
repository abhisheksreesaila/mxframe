"""SQL frontend for MXFrame: SQL string → LogicalPlan via sqlglot.

Supports a large subset of TPC-H-style SQL:
  SELECT / FROM / WHERE / GROUP BY / ORDER BY / LIMIT
  Arithmetic: + - * /
  Comparisons: = != < <= > >=
  Boolean: AND OR NOT
  BETWEEN, IN / NOT IN, LIKE / ILIKE [ESCAPE]
  Aggregates: SUM AVG MIN MAX COUNT(*) COUNT(DISTINCT x), arithmetic on aggregates, HAVING, GROUP BY ALL
  CASE WHEN ... THEN ... ELSE ... END
  INNER / LEFT JOIN ... ON ...
  WITH name AS (...), derived tables (FROM (SELECT ...) AS d)
  Subqueries (F-065): a scalar subquery in WHERE / HAVING / SELECT expressions; x [NOT] IN (SELECT ...) and
  [NOT] EXISTS (SELECT ...) as top-level AND conditions of WHERE (semi / anti joins); a correlated scalar
  comparison in WHERE (join + aggregate). Other forms raise NotImplementedError naming the form.
"""

__all__ = ["sql"]

import re
import calendar
import itertools
from datetime import date, timedelta
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal

import numpy as np
import pyarrow as pa
import sqlglot
import sqlglot.expressions as sge
from typing import Any, Dict, List, Optional, Union

from .lazy_expr import Expr, col, lit, when, coalesce, row_number
from .lazy_frame import (
    LazyFrame, LogicalPlan,
    Scan, ParquetScan, Filter, Project, Aggregate, Sort, Limit, Distinct, Join, join_output_names, column_types,
)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def sql(query: str, **tables) -> LazyFrame:
    """Run a SQL query against the provided tables.

    Args:
        query:   SQL string (DuckDB dialect).
        **tables: named ``pa.Table`` or :class:`LazyFrame` objects referenced in the query. A LazyFrame
            (``scan_parquet``, ``read_parquet`` or any plan) is used as that table's plan, so scan pushdown
            (columns, filters, row groups) still applies.

    Returns:
        A :class:`LazyFrame` whose ``.compute()`` executes the query.

    Example::

        import mxframe as mx
        result = mx.sql(
            "SELECT l_returnflag, SUM(l_quantity) AS sum_qty "
            "FROM lineitem WHERE l_shipdate <= 10471 "
            "GROUP BY l_returnflag ORDER BY l_returnflag",
            lineitem=lineitem_table,
        ).compute()
    """
    translator = _SQLTranslator({k.lower(): v for k, v in tables.items()})
    return translator.translate(query)


# ---------------------------------------------------------------------------
# internal translator
# ---------------------------------------------------------------------------



_CROSS_LIMIT = 4_000_000          # row pairs a join with no equality (a cross product) may build


def _rows_upper(plan: LogicalPlan) -> Optional[int]:
    """An upper bound on the rows ``plan`` yields (not its actual count), when that is cheap to know (tables, Parquet footers and what only
    keeps or reshapes their rows); else None."""
    if isinstance(plan, Scan):
        return plan.table.num_rows
    if isinstance(plan, ParquetScan):
        from .io_parquet import _metadata
        return _metadata(plan.path).num_rows
    if isinstance(plan, Limit):
        inner = _rows_upper(plan.input)
        if plan.n is None:                                   # OFFSET without LIMIT: what is left after the skipped rows
            return None if inner is None else max(inner - plan.offset, 0)
        return min(plan.n + plan.offset, inner) if inner is not None else plan.n + plan.offset
    if isinstance(plan, Join):
        l, r = _rows_upper(plan.left), _rows_upper(plan.right)
        return None if l is None or r is None else l if plan.how in ("semi", "anti") else l * max(r, 1)      # LEFT: an empty right side still keeps every left row
    inner = getattr(plan, "input", None)
    return _rows_upper(inner) if inner is not None else None


def _join_type(join) -> str:
    """The engine's join type for a sqlglot join: sqlglot keeps LEFT / RIGHT / FULL in `side` and
    SEMI / ANTI / OUTER / CROSS in `kind` (``LEFT ANTI JOIN``: side LEFT, kind ANTI)."""
    side = str(join.args.get("side") or "").lower()
    kind = str(join.args.get("kind") or "").lower()
    if kind in ("semi", "anti"):
        if side not in ("", "left"):
            raise ValueError(f"unsupported SQL join: {side.upper()} {kind.upper()} "
                             "(supported: [INNER] JOIN, LEFT [OUTER] JOIN, [LEFT] SEMI JOIN, [LEFT] ANTI JOIN)")
        return kind
    if kind in ("", "inner", "outer") and side == "left":
        return "left"
    if kind in ("", "inner") and side == "":
        return "inner"
    raise ValueError(f"unsupported SQL join: {' '.join(x for x in (side, kind) if x).upper() or 'JOIN'} "
                     "(supported: [INNER] JOIN, LEFT [OUTER] JOIN, [LEFT] SEMI JOIN, [LEFT] ANTI JOIN)")
_AGG_NAMES = {sge.Sum: "sum", sge.Avg: "avg", sge.Min: "min", sge.Max: "max", sge.Count: "count"}


def _aggs_in(e: sge.Expression) -> List[sge.Expression]:
    """The outermost aggregate calls in ``e`` (window functions are left to ``_expr``, which refuses them)."""
    return [n for n in e.walk(prune=lambda n: isinstance(n, (sge.AggFunc, sge.Window, sge.Subquery, sge.Exists)))
            if isinstance(n, sge.AggFunc)]


_COMPARE = (sge.EQ, sge.NEQ, sge.LT, sge.LTE, sge.GT, sge.GTE)


def _number(text, neg: bool = False) -> Union[int, float]:
    """A numeric literal by its SQL type, with its minus sign (`neg`) folded in before the range check, so
    -9223372036854775808 is an int64: digits only (`1_000` too) is an integer; a point, an exponent or both make
    it a float (DECIMAL and DOUBLE both read as a float64 here; `_is_decimal_literal` tells them apart). An
    integer past int64 (DuckDB's HUGEINT) is a float here; `_huge_int` finds it where it is compared with an
    integer column, and `_SQLTranslator._compare` answers that comparison exactly."""
    s = str(text)
    if re.fullmatch(r"\d[\d_]*", s):
        i = -int(s) if neg else int(s)
        if -(1 << 63) <= i < (1 << 63):
            return i
    return -float(s) if neg else float(s)


def _folded(x: Expr, value: bool) -> Expr:
    """A predicate decided at plan time (`i = 2.5`, `i < 99999999999999999999`): the constant where the integer
    expression ``x`` has a value, NULL where it is NULL, so NOT, IS NULL and SELECT see SQL's three values.
    ``x == x`` is true and ``x != x`` false for an integer, NULL for NULL."""
    return x == x if value else x != x


def _exact_decimal(n: sge.Expression) -> Optional[Decimal]:
    """The exact value of an integer or DECIMAL literal (digits, a point, no exponent), negated or not, else None."""
    n = _unparen(n)
    neg = isinstance(n, sge.Neg)
    n = _unparen(n.this) if neg else n
    if not (isinstance(n, sge.Literal) and not n.is_string and re.fullmatch(r"[\d_]*\.?[\d_]*", str(n.this))
            and re.search(r"\d", str(n.this))):
        return None
    d = Decimal(str(n.this).replace("_", ""))
    return -d if neg else d


def _is_double_literal(n: sge.Expression) -> bool:
    """`1e3`, `-1.5E-5`: DuckDB types a number with an exponent as DOUBLE, which widens a float32 column it is
    compared with or added to (a DECIMAL or INTEGER literal is cast to the column's float32 instead)."""
    n = _unparen(n)
    n = _unparen(n.this) if isinstance(n, sge.Neg) else n
    return isinstance(n, sge.Literal) and not n.is_string and re.fullmatch(r"\d[\d_]*", str(n.this)) is None \
        and "e" in str(n.this).lower()


def _is_decimal_literal(n: sge.Expression) -> bool:
    """``2.5`` or ``-2.5``: DuckDB types a number with a point and no exponent as DECIMAL (``1e3`` is DOUBLE)."""
    n = _unparen(n)
    n = _unparen(n.this) if isinstance(n, sge.Neg) else n
    return isinstance(n, sge.Literal) and not n.is_string and re.fullmatch(r"\d*\.\d*", str(n.this)) is not None


def _unparen(n: sge.Expression) -> sge.Expression:
    while isinstance(n, sge.Paren):
        n = n.this
    return n


def _conjuncts(n: sge.Expression) -> List[sge.Expression]:
    """The conditions joined by AND at the top of ``n``."""
    n = _unparen(n)
    return _conjuncts(n.this) + _conjuncts(n.expression) if isinstance(n, sge.And) else [n]


def _and_all(nodes: List[sge.Expression]) -> Optional[sge.Expression]:
    out = None
    for n in nodes:
        out = n if out is None else sge.And(this=out, expression=n)
    return out


_EPOCH = date(1970, 1, 1)


def _float_cast(node: sge.Expression) -> Optional[float]:
    """The value of ``CAST('nan' | 'inf' | '-inf' | '1.5' AS DOUBLE | FLOAT)``, else None: the way SQL writes
    a NaN or infinity constant."""
    if (isinstance(node, sge.Cast) and node.to.this in (sge.DataType.Type.DOUBLE, sge.DataType.Type.FLOAT)
            and isinstance(node.this, sge.Literal) and node.this.is_string):
        try:
            return float(node.this.this)
        except ValueError:
            raise ValueError(f"not a number: CAST('{node.this.this}' AS {node.to.sql(dialect='duckdb')})") from None
    return None


def _date_of(node: sge.Expression) -> Optional[date]:
    """The date of a constant ``DATE 'yyyy-mm-dd'`` [+ / - INTERVAL 'n' DAY | MONTH | YEAR ...], else None."""
    node = _unparen(node)
    if isinstance(node, sge.Cast) and node.to.this == sge.DataType.Type.DATE:
        if isinstance(node.this, sge.Literal) and node.this.is_string:
            try:
                return date.fromisoformat(node.this.this)
            except ValueError:
                raise ValueError(f"not a date: DATE '{node.this.this}'") from None
        return None
    if isinstance(node, (sge.Add, sge.Sub)):
        d = _date_of(node.this)
        iv = _unparen(node.expression)
        if d is None or not isinstance(iv, sge.Interval):
            return None
        n = int(iv.this.this) * (1 if isinstance(node, sge.Add) else -1)
        unit = iv.unit.name.upper() if iv.unit is not None else ""
        if unit == "DAY":
            return d + timedelta(days=n)
        if unit in ("MONTH", "YEAR"):
            m = d.year * 12 + d.month - 1 + n * (12 if unit == "YEAR" else 1)
            y, mo = divmod(m, 12)
            return date(y, mo + 1, min(d.day, calendar.monthrange(y, mo + 1)[1]))
        raise NotImplementedError(f"INTERVAL unit {unit or '?'} is not supported (DAY, MONTH, YEAR)")
    return None


_TS_UNITS = {sge.DataType.Type.TIMESTAMPNTZ: "us", sge.DataType.Type.TIMESTAMP: "us", sge.DataType.Type.TIMESTAMP_S: "s",
             sge.DataType.Type.TIMESTAMP_MS: "ms", sge.DataType.Type.TIMESTAMP_NS: "ns"}


def _timestamp_of(node: sge.Expression) -> Optional[np.datetime64]:
    """The value of ``TIMESTAMP '2024-01-01 10:00:00.123'`` / ``CAST('...' AS TIMESTAMP)`` (also ``TIMESTAMP_S`` /
    ``_MS`` / ``_NS``) as a numpy datetime64 of that type's unit (TIMESTAMP is microseconds, finer digits are cut
    as DuckDB cuts them), else None. A zone offset is not supported."""
    node = _unparen(node)
    if not (isinstance(node, sge.Cast) and isinstance(node.this, sge.Literal) and node.this.is_string):
        return None
    if node.to.this == sge.DataType.Type.TIMESTAMPTZ:
        raise NotImplementedError(f"TIMESTAMPTZ literals are not supported: {node.sql(dialect='duckdb')}")
    unit = _TS_UNITS.get(node.to.this)
    if unit is None:
        return None
    text = node.this.this.strip()
    if re.search(r"\d(\s*Z|\s*[+-]\d\d(:?\d\d)?)$", text, re.I) and re.search(r"\d:\d", text):
        raise NotImplementedError(f"a time zone offset in a TIMESTAMP literal is not supported: {text}")
    m = re.fullmatch(r"(-?\d{4,}-\d\d-\d\d)(?:[ T](\d\d:\d\d(?::\d\d)?)(?:\.(\d+))?)?", text)
    try:
        if m is None:
            raise ValueError(text)
        hms = m.group(2) or "00:00:00"
        seconds = int(np.datetime64(f"{m.group(1)}T{hms}{':00' if hms.count(':') == 1 else ''}", "s").astype("int64"))
    except ValueError:
        raise ValueError(f"not a timestamp: {node.to.sql(dialect='duckdb')} '{text}'") from None
    per_second = {"s": 1, "ms": 10 ** 3, "us": 10 ** 6, "ns": 10 ** 9}[unit]
    ticks = seconds * per_second + int((m.group(3) or "0").ljust(9, "0")[:9]) * per_second // 10 ** 9      # finer digits are cut
    if not -(1 << 63) <= ticks < (1 << 63):
        raise ValueError(f"out of range for {node.to.sql(dialect='duckdb')}: '{text}'")
    return np.datetime64(ticks, unit)


def _num_const(node: sge.Expression) -> Optional[Decimal]:
    """The exact value of a constant number expression (+ - * of literals), as SQL decimals compute it."""
    node = _unparen(node)
    if isinstance(node, sge.Literal):
        return None if node.is_string else Decimal(str(node.this))
    if isinstance(node, sge.Neg):
        v = _num_const(node.this)
        return None if v is None else -v
    if isinstance(node, (sge.Add, sge.Sub, sge.Mul)):
        a, b = _num_const(node.this), _num_const(node.expression)
        if a is None or b is None:
            return None
        return a + b if isinstance(node, sge.Add) else a - b if isinstance(node, sge.Sub) else a * b
    return None


def _keep_ints(item: sge.Expression) -> None:
    """Mark the integer literals of an IN (SELECT ...) item: they stay integers in its arithmetic."""
    for n in item.find_all(sge.Literal):
        if not n.is_string:
            n.meta["keep_int"] = True


def _type_in(src: sge.Expression, schemas: dict):
    """The Arrow type of a plain column, resolved through its table alias (or, unqualified, the one FROM item
    that has the name; None when unknown or ambiguous). ``schemas``: alias -> pa.Schema or None."""
    src = _unparen(src)
    if not isinstance(src, sge.Column):
        return None
    name = src.name.lower()
    if src.table:
        cands = [schemas.get(src.table.lower())]
    else:
        cands = list(schemas.values())
    types = []
    for sch in cands:
        if sch is None:
            if src.table:
                return None
            continue
        for n in sch.names:
            if n.lower() == name:
                types.append(sch.field(n).type)
    unknown = (not src.table) and any(sc is None for sc in cands)
    return types[0] if len(types) == 1 and not unknown else None


def _null_on_empty(n: sge.Expression) -> bool:
    """True when ``n`` is NULL over no rows: aggregates (not COUNT) under + - * / and numeric constants. A
    correlated subquery is decorrelated to an inner join only for such an item."""
    n = _unparen(n)
    if isinstance(n, sge.AggFunc):
        return not isinstance(n, sge.Count)
    if isinstance(n, sge.Literal):
        return not n.is_string
    if isinstance(n, sge.Neg):
        return _null_on_empty(n.this)
    if isinstance(n, (sge.Add, sge.Sub, sge.Mul, sge.Div, sge.Mod)):
        return _null_on_empty(n.this) and _null_on_empty(n.expression) and bool(_aggs_in(n))
    return False


def _is_comma_join(join: sge.Join) -> bool:
    """FROM a, b (or a CROSS JOIN b): the join keys, if any, are in WHERE."""
    return (not join.args.get("on") and not join.args.get("using") and not join.args.get("side")
            and str(join.args.get("kind") or "").lower() in ("", "cross") and not join.args.get("method"))


def _disjuncts(n: sge.Expression) -> List[sge.Expression]:
    n = _unparen(n)
    return _disjuncts(n.this) + _disjuncts(n.expression) if isinstance(n, sge.Or) else [n]


def _cond_key(c: sge.Expression):
    """A condition's identity for comparing OR branches: ``a = b`` and ``b = a`` are one."""
    if isinstance(c, sge.EQ):
        return ("eq", frozenset((_unparen(c.this).sql(dialect="duckdb"), _unparen(c.expression).sql(dialect="duckdb"))))
    return c.sql(dialect="duckdb")


def _or_common_equalities(node: sge.Or) -> List[sge.EQ]:
    """Column = column conditions that every branch of an OR has (TPC-H Q19's join key)."""
    branches = [{_cond_key(c): c for c in _conjuncts(b)} for b in _disjuncts(node)]
    common = set.intersection(*[set(b) for b in branches])
    return [branches[0][k] for k in common if isinstance(branches[0][k], sge.EQ)]


def _or_without(node: sge.Or, drop: set) -> Optional[sge.Expression]:
    """The OR with the conditions whose SQL text is in ``drop`` removed from every branch."""
    out = []
    for b in _disjuncts(node):
        rest = [c for c in _conjuncts(b) if _cond_key(c) not in drop]
        if not rest:
            return None                      # a branch that is only the dropped conditions: always true
        out.append(sge.Paren(this=_and_all(rest)))
    r = out[0]
    for o in out[1:]:
        r = sge.Or(this=r, expression=o)
    return sge.Paren(this=r)


_MAX_ROWS = 2 ** 62          # a LIMIT / OFFSET above this is clamped to it: more rows than any table, and n + offset stays an int64


def _alias(node: sge.Expression, name: str) -> sge.Alias:
    return sge.Alias(this=node.copy(), alias=sge.to_identifier(name))


def _unsupported_shape(sel: sge.Expression, what: str, keys=("group", "having", "limit", "distinct", "offset", "qualify")):
    """``what`` needs a subquery without these clauses (they change which rows a join would see)."""
    names = {"group": "GROUP BY", "having": "HAVING", "limit": "LIMIT", "distinct": "DISTINCT",
             "offset": "OFFSET", "qualify": "QUALIFY"}
    for k in keys:
        if sel.args.get(k):
            raise NotImplementedError(f"{what}: {names[k]} in the subquery is not supported")
    if _aggs_in(sge.Tuple(expressions=list(sel.expressions))):
        raise NotImplementedError(f"{what}: an aggregate in the subquery's SELECT is not supported")


def _default_agg_name(node: sge.Expression) -> str:
    """DuckDB's name for an aggregate without an alias: ``sum(v)``, ``count_star()``, ``count(DISTINCT v)``;
    the argument is written as in the query. Other arguments need an alias (we do not guess DuckDB's text)."""
    fn = _AGG_NAMES.get(type(node))
    if fn is None:
        raise NotImplementedError(f"Unsupported aggregate function: {node.sql_name()} (the engine has SUM, AVG, MIN, MAX, "
                                  f"COUNT and COUNT(DISTINCT x); no MEDIAN, quantile, STDDEV or VARIANCE)")
    arg = node.this
    distinct = isinstance(arg, sge.Distinct)
    if distinct:
        arg = arg.expressions[0]
    if isinstance(arg, sge.Star):
        return "count_star()"
    if isinstance(arg, (sge.Column, sge.Literal)):
        return f"{fn}({'DISTINCT ' if distinct else ''}{_written(arg)})"
    raise ValueError(f"give {node.sql(dialect='duckdb')} an alias (AS name): the output column has no name")


def _dedupe(names) -> List[Optional[str]]:
    """Output names as DuckDB returns them: a name already taken (names are case-insensitive) becomes
    ``x_1``, ``x_2`` ... (the first free one, in SELECT order). Unnamed (None) stay unnamed."""
    seen, out = set(), []
    for n in names:
        if n is not None and n.lower() in seen:
            i = 1
            while f"{n}_{i}".lower() in seen:
                i += 1
            n = f"{n}_{i}"
        if n is not None:
            seen.add(n.lower())
        out.append(n)
    return out


def _ci_find(names, name: str) -> Optional[str]:
    """The first of ``names`` equal to ``name`` ignoring case (DuckDB's rule for every identifier, quoted or
    not; of two columns that differ only by case the first one answers to both spellings), else None."""
    low = name.lower()
    return next((n for n in names if n.lower() == low), None)


def _written(col: sge.Expression) -> str:
    """A column or literal as the query wrote it (DuckDB names an unaliased aggregate ``sum(AMOUNT)`` from
    this text): name resolution rewrites a column to its source spelling and keeps the written one in meta."""
    if not isinstance(col, sge.Column):
        return col.sql(dialect="duckdb")
    name = col.meta.get("written", col.name)
    text = name if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) else '"' + name.replace('"', '""') + '"'
    return f"{col.table}.{text}" if col.table else text


def _ordinal(node: sge.Expression, n_items: int) -> Optional[int]:
    """0-based SELECT position when ``node`` is an integer literal (GROUP BY 1, ORDER BY 2)."""
    if isinstance(node, sge.Literal) and not node.is_string:
        i = int(node.this)
        if not 1 <= i <= n_items:
            raise ValueError(f"ORDER BY / GROUP BY position {i} is not in the SELECT list ({n_items} columns)")
        return i - 1
    return None


class _SQLTranslator:
    def __init__(self, tables: Dict[str, pa.Table]):
        self._tables = tables
        # column names visible in the current FROM scope (populated lazily)
        self._scope_columns: List[str] = []
        # columns accumulated on the left side as joins are built
        self._left_schema: set = set()
        # map right-join-key → left-join-key so post-join expressions use
        # the column that actually exists in the output (left side is kept,
        # right join key is dropped by the hash-join implementation)
        self._col_aliases: dict = {}
        # a LEFT JOIN drops its right key, but `u.k` is NULL for unmatched rows: when the query reads it, the
        # right side carries a copy of it (NULL where nothing matched). (alias, key) / bare key -> that copy
        self._marks: dict = {}
        self._marks_bare: dict = {}
        self._renamed: dict = {}
        self._copies: set = set()            # the copies: internal, never part of a result
        self._used_names: set = set()        # internal column names handed out by _fresh
        # (table name or alias, its columns) of every scanned table
        self._table_cols: List[tuple] = []
        # SELECT * output: (column of the joined plan, name it has in the result) in DuckDB's order, both key
        # columns of a JOIN ... ON included (the copies), one for USING / NATURAL
        self._star_items: List[tuple] = []
        self._any_star = False
        # aggregate query state (see _build_aggregate)
        self._post = False          # translating an expression above the Aggregate
        self._key_names: dict = {}  # translated group key signature -> output name
        self._out_src: dict = {}    # SELECT output name -> its column in the Aggregate output
        self._sort = None
        self._aggs: List[Expr] = []
        self._agg_names: dict = {}
        self._key_names_list: List[Expr] = []
        self._cd = None             # (argument, output name) of the one COUNT(DISTINCT x)
        # WITH names -> their plans; unique names for the columns a decorrelated subquery adds
        self._ctes: Dict[str, LogicalPlan] = {}
        self._ids = itertools.count()
        self._has_star = False
        self._date_lit = False      # translating a comparison with a date32 column: DATE constants are dates
        self._alias_schema: dict = {}   # FROM alias -> Arrow schema (None when unknown)
        # comma joins: WHERE conditions not yet used as join keys; (table alias, column) -> its column in
        # the joined plan; aliases and output columns joined so far
        self._conj: List[sge.Expression] = []
        self._qual: dict = {}
        self._joined: set = set()
        self._cur_cols: List[str] = []
        self._sources: dict = {}        # id(FROM item) -> (item, plan, columns)
        # name resolution (`_resolve_names`): the scopes of this query and then of the queries around it
        # (each [(alias, columns)]), and its SELECT aliases
        self._outer: list = []
        self._levels: list = []
        self._aliases: dict = {}

    def _child(self) -> "_SQLTranslator":
        """A translator for a subquery: its own state, the same tables and WITH names."""
        c = _SQLTranslator(self._tables)
        c._ctes = dict(self._ctes)
        c._outer = self._levels
        c._ids = self._ids
        return c

    # ------------------------------------------------------------------
    # top-level dispatch
    # ------------------------------------------------------------------

    def translate(self, query: str) -> LazyFrame:
        tree = sqlglot.parse_one(query, dialect="duckdb")
        if isinstance(tree, sge.Copy):
            return self._copy(tree)
        if not isinstance(tree, sge.Select):
            raise ValueError(f"Only SELECT statements are supported, got {type(tree).__name__}")
        plan = self._build_select(tree)
        return LazyFrame(plan)

    def _copy(self, tree: sge.Copy) -> LazyFrame:
        """``COPY (SELECT ...) TO 'file.parquet' [(FORMAT PARQUET)]``: runs the query on the CPU now and streams the result
        into the file (`LazyFrame.sink_parquet`, so a result bigger than memory finishes), then returns the one-row
        table ``Count`` of the rows written, as DuckDB does (F-113)."""
        body, files = tree.this, tree.args.get("files") or []
        if tree.args.get("kind") or not isinstance(body, sge.Subquery) or not isinstance(body.this, sge.Select) \
                or len(files) != 1 or not isinstance(files[0], sge.Literal) or not files[0].is_string:
            raise NotImplementedError("COPY is supported as COPY (SELECT ...) TO 'file.parquet'")
        path, explicit = files[0].name, False
        for prm in tree.args.get("params") or []:
            name = prm.this.name.upper() if hasattr(prm.this, "name") else str(prm.this).upper()
            expr = prm.args.get("expression")
            value = (expr.name if hasattr(expr, "name") else str(expr)).upper() if expr is not None else ""
            if (name, value) != ("FORMAT", "PARQUET"):
                shown = f"{name} {value}".strip()
                raise NotImplementedError(f"COPY option {shown} is not supported (only FORMAT PARQUET)")
            explicit = True
        if not explicit and not path.lower().endswith((".parquet", ".parq")):
            raise NotImplementedError(f"COPY writes Parquet only: name the file *.parquet or add (FORMAT PARQUET); got '{path}'")
        n = LazyFrame(self._build_select(body.this)).sink_parquet(path)
        return LazyFrame(Scan(pa.table({"Count": pa.array([n], pa.int64())})))

    def _build_select(self, node: sge.Select) -> LogicalPlan:
        if not isinstance(node, sge.Select):
            raise NotImplementedError(f"{type(node).__name__} (UNION / INTERSECT / EXCEPT) is not supported, only SELECT")
        self._register_ctes(node.args.get("with_"))
        self._resolve_names(node)
        self._has_star = any(isinstance(e, sge.Star) for e in node.expressions)
        self._any_star = any(isinstance(e, sge.Star) or (isinstance(e, sge.Column) and isinstance(e.this, sge.Star))
                             for e in node.expressions)

        where = node.args.get("where")
        if where and _aggs_in(where.this):
            raise ValueError("aggregate functions are not allowed in WHERE (use HAVING)")
        self._conj = _conjuncts(where.this) if where else []
        n_conj = len(self._conj)

        # 1. FROM + JOINs (a comma join takes its keys from WHERE)
        plan = self._build_from(node)

        # 2. WHERE (what the joins did not use)
        if where and self._conj:
            same = len(self._conj) == n_conj and all(a is b for a, b in zip(self._conj, _conjuncts(where.this)))
            plan = self._where(plan, where.this if same else _and_all(self._conj))

        self._expand_stars(node)
        plan = self._select_subqueries(plan, node)

        for o in (node.args.get("order").expressions if node.args.get("order") else []):
            if o.args.get("nulls_first"):
                raise NotImplementedError("ORDER BY ... NULLS FIRST is not supported (NULLs sort last, as in DuckDB's default)")

        # 3. GROUP BY / aggregates / HAVING
        group_node = node.args.get("group")
        having = node.args.get("having")
        has_aggs = any(_aggs_in(e) for e in node.expressions)

        distinct = node.args.get("distinct")
        if distinct is not None and distinct.args.get("on") is not None:
            raise NotImplementedError("SELECT DISTINCT ON (...) is not supported")
        if distinct is not None and (group_node or has_aggs):
            raise NotImplementedError("SELECT DISTINCT together with GROUP BY or aggregates is not supported yet")

        final_proj = None
        names = None
        alias_out: dict = {}
        sort_cols: dict = {}
        if group_node or has_aggs:
            plan, final_proj = self._build_aggregate(plan, node, group_node, having)
        else:
            if having:
                raise ValueError("HAVING needs GROUP BY or an aggregate function in SELECT")
            # pure projection (skip if SELECT *)
            non_star = [e for e in node.expressions if not isinstance(e, sge.Star)]
            star_cols = [c for c, _ in self._star_items]
            outs = _dedupe([n for _, n in self._star_items])
            if not non_star and (self._copies or outs != star_cols):           # SELECT *: the key columns the join
                star_proj = [col(c) if c == o else col(c).alias(o)             # kept, a repeated name renamed (k_1)
                             for c, o in zip(star_cols, outs)]
                if node.args.get("order"):             # ORDER BY u.k names the plan's columns: rename after it
                    final_proj = star_proj
                else:
                    plan = Project(plan, star_proj)
            if non_star:
                proj = [self._select_expr(e) for e in non_star]
                names = _dedupe([p._alias or (p.args[0] if p.op == "col" else None) for p in proj])
                for p, nm in zip(proj, names):
                    if nm is not None and nm != (p._alias or (p.args[0] if p.op == "col" else None)):
                        p.alias(nm)
                # ORDER BY a name two output columns share takes the aliased one (as DuckDB does)
                alias_out = {e.alias.lower(): nm for e, nm in zip(non_star, names)
                             if isinstance(e, sge.Alias) and e.alias != nm}
                hidden, sort_cols = self._hidden_sort_keys(node, non_star, names, distinct is not None) if names else ([], {})
                plan = Project(plan, proj + [ex.alias(nm) for nm, ex in hidden])
                if hidden:                             # carried to the sort, dropped after it
                    final_proj = [col(nm) for nm in names]
            if distinct is not None:
                plan = Distinct(plan, None)

        # 4. ORDER BY
        order_node = node.args.get("order")
        if order_node:
            if self._sort is not None:                 # an aggregate query translated it already
                by, desc = self._sort
            else:
                by = []
                for o in order_node.expressions:
                    i = _ordinal(o.this, len(node.expressions))
                    ref = o.this
                    if i is None and isinstance(ref, sge.Column) and not ref.table and ref.name.lower() in alias_out:
                        by.append(col(alias_out[ref.name.lower()]))
                    elif i is None and isinstance(ref, sge.Column) and not ref.table and _ci_find(names or [], ref.name):
                        by.append(col(_ci_find(names, ref.name)))          # a SELECT output name
                    elif i is None:
                        ex = self._expr(o.this)
                        by.append(col(sort_cols[ex.signature()]) if ex.signature() in sort_cols else ex)
                    elif names is None or names[i] is None:
                        raise ValueError("ORDER BY position needs a named SELECT column (give it an alias)")
                    else:
                        by.append(col(names[i]))
                desc = [bool(o.args.get("desc")) for o in order_node.expressions]
            plan = Sort(plan, by, desc)

        # 5. LIMIT / OFFSET (OFFSET alone keeps every row after the skipped ones)
        limit_node, offset_node = node.args.get("limit"), node.args.get("offset")
        if limit_node or offset_node:
            plan = Limit(plan, min(int(limit_node.expression.this), _MAX_ROWS) if limit_node else None,
                         min(int(offset_node.expression.this), _MAX_ROWS) if offset_node else 0)

        if final_proj:
            plan = Project(plan, final_proj)
        return plan

    def _expand_stars(self, node: sge.Select) -> None:
        """``*`` and ``t.*`` next to other SELECT items become the columns they stand for, in place and in
        order (ORDER BY positions count them). A lone ``*`` stays: the plan needs no projection for it."""
        items = list(node.expressions)

        def table_star(e):
            return isinstance(e, sge.Column) and isinstance(e.this, sge.Star)
        if not any(isinstance(e, sge.Star) or table_star(e) for e in items) \
                or (len(items) == 1 and isinstance(items[0], sge.Star)):
            return
        out = []
        for e in items:
            star = e if isinstance(e, sge.Star) else e.this if table_star(e) else None
            if star is None:
                out.append(e)
                continue
            if any(star.args.get(k) for k in ("except_", "replace", "rename")):
                raise NotImplementedError("* EXCLUDE / REPLACE / RENAME next to other SELECT items is not supported "
                                          "(list the columns)")
            if isinstance(e, sge.Star):
                out += [sge.Column(this=sge.to_identifier(c)) if c == n
                        else sge.Alias(this=sge.Column(this=sge.to_identifier(c)), alias=sge.to_identifier(n))
                        for c, n in self._star_items]
                continue
            table = e.table.lower()
            if table not in {a for a, _ in self._table_cols}:
                raise ValueError(f"{e.sql(dialect='duckdb')}: unknown table {table!r} in the FROM clause")
            if len(self._table_cols) == 1:
                cols = self._cur_cols
            elif self._alias_schema.get(table) is not None:
                cols = self._alias_schema[table].names
            else:
                raise NotImplementedError(f"{table}.*: the columns of this FROM item are not known, list them")
            out += [sge.Column(this=sge.to_identifier(c), table=sge.to_identifier(table)) for c in cols]
        node.set("expressions", out)

    def _hidden_sort_keys(self, node: sge.Select, items, names, distinct: bool):
        """ORDER BY keys of a plain SELECT that are not output columns: (the extra columns to carry
        through the projection as ``[(name, expr)]``, ``{signature: column}`` of every key that is a
        column already). SELECT DISTINCT may not sort by anything it does not output (as in DuckDB)."""
        order_node = node.args.get("order")
        have = {}
        for e, nm in zip(items, names):
            if nm is not None:
                have.setdefault(self._expr(e.this if isinstance(e, sge.Alias) else e).signature(), nm)
        hidden = []
        taken = set(names) | {c.lower() for c in self._scope_columns}
        for o in order_node.expressions if order_node else []:
            ref = o.this
            if _ordinal(ref, len(node.expressions)) is not None:
                continue
            if isinstance(ref, sge.Column) and not ref.table and _ci_find(names, ref.name):
                continue
            ex = self._expr(ref)
            if ex.signature() in have:
                continue
            if distinct:
                raise ValueError("for SELECT DISTINCT, ORDER BY expressions must appear in the select list")
            n = len(hidden)
            while f"__ord{n}" in taken:                # never a name the query or its tables use
                n += 1
            nm = f"__ord{n}"
            taken.add(nm)
            have[ex.signature()] = nm
            hidden.append((nm, ex))
        return hidden, have

    # ------------------------------------------------------------------
    # FROM / JOIN
    # ------------------------------------------------------------------

    def _build_from(self, node: sge.Select) -> LogicalPlan:
        from_clause = node.args.get("from_")
        if from_clause is None:
            raise ValueError("No FROM clause found")

        joins_ = node.args.get("joins") or []
        if any(_is_comma_join(j) for j in joins_) and not all(_is_comma_join(j) for j in joins_):
            raise NotImplementedError("comma joins (FROM a, b WHERE ...) mixed with JOIN ... ON are not supported: "
                                      "write every join as JOIN ... ON, or every one as a comma join")
        plan, main_cols = self._scan(from_clause.this)
        # Seed the accumulated left-side schema with the main FROM table
        self._left_schema = set(main_cols)
        self._joined = {from_clause.this.alias_or_name.lower()}
        self._cur_cols = list(main_cols)
        self._qual = {(next(iter(self._joined)), c): c for c in main_cols}
        self._star_items = [(c, c) for c in main_cols]
        pending = []

        for join in node.args.get("joins") or []:
            right_plan, right_cols = self._scan(join.this)
            right_schema = set(right_cols)
            if _is_comma_join(join):
                pending.append((join.this.alias_or_name.lower(), right_plan, right_cols))
                continue

            how = _join_type(join)
            alias = join.this.alias_or_name.lower()
            on = join.args.get("on")
            mixed, temp, rid, lcols = [], set(), None, None
            if on is not None:
                on, right_only = self._split_on(on, alias, right_schema)
                if right_only:                        # ON ... AND u.v IS NULL: the right rows that may match
                    pred = self._expr(right_only[0])
                    for c in right_only[1:]:
                        pred = pred & self._expr(c)
                    right_plan = Filter(right_plan, pred)
                keys, lonly, ronly, mixed = self._split_join_on(on, alias, right_schema)
                if (lonly or ronly or mixed) and how in ("semi", "anti"):
                    raise NotImplementedError("a condition besides column = column in the ON of a SEMI / ANTI JOIN "
                                              "is not supported")
                plan, right_plan, left_on, right_on, temp = self._join_keys(
                    plan, right_plan, right_cols, keys, lonly, ronly, how, alias)
                if mixed and how == "left":           # a residual that needs both tables: see _left_residual
                    lcols = self._columns_of(plan)
                    rid = self._fresh("__rid")
                    plan = Project(plan, [col(c) for c in lcols] + [row_number().alias(rid)])
                    lcols = lcols + [rid]
            using = join.args.get("using")
            if on is None and (using or join.args.get("method")):
                if using:
                    cols = using if isinstance(using, list) else using.expressions      # sqlglot versions differ
                    keys = [str(c.name) for c in cols]
                else:                                  # NATURAL: the columns both sides have
                    keys = [c for c in self._cur_cols if _ci_find(right_cols, c)]
                    if not keys:
                        raise ValueError("NATURAL JOIN: the two sides share no column name")
                left_on = [_ci_find(self._cur_cols, k) or k for k in keys]       # each side's own spelling
                right_on = [_ci_find(right_cols, k) or k for k in keys]
            elif on is None:
                raise ValueError("JOIN requires ON or USING clause")

            on_keys = right_on
            right_plan, right_on, own = self._own_typed_keys(plan, right_plan, right_cols, left_on, right_on, how)
            if how == "left" or (how == "inner" and on is not None and self._any_star):
                # SELECT * keeps both key columns of a JOIN ... ON: the right one is a copy (own-typed keys
                # stay ordinary right columns already)
                carried = list(right_cols) + [n for n in right_on if n not in right_cols]
                visible = [i for i, k in enumerate(on_keys) if k not in own and k not in temp]
                right_plan = self._copy_right_keys(node, join, right_plan, carried, [left_on[i] for i in visible],
                                                   [on_keys[i] for i in visible], all_keys=self._any_star and on is not None,
                                                   extra=mixed)
                self._left_schema |= self._copies            # a later ON may name a copy (u.k)
            plan = Join(plan, right_plan, left_on, right_on, how)
            # After this join, the accumulated left side grows to include right cols (a semi / anti
            # join outputs the left columns only)
            if how not in ("semi", "anti"):
                self._left_schema |= right_schema
                outs = join_output_names(self._cur_cols, right_cols, right_on)
                self._cur_cols += [o for _, o in outs]
                self._left_schema |= {o for _, o in outs}        # also the renamed names (v_right)
                out_of = dict(outs)
                for c in right_cols:                   # the right table's columns in its own order
                    key_copy = self._marks.get((alias, c))
                    if c in out_of:
                        self._star_items.append((out_of[c], c))
                    elif on is not None and key_copy in self._copies:
                        self._star_items.append((key_copy, c))
                for c, o in outs:
                    if c != o:                         # a clashing column is renamed (x.k -> k_right)
                        self._renamed[(alias, c)] = o
            self._joined.add(join.this.alias_or_name.lower())
            # Record that right join keys should be referred to by left join keys
            for lk, rk in zip(left_on, right_on):
                if lk in self._copies and how != "left" and rk not in own:
                    self._marks[(join.this.alias_or_name.lower(), rk)] = lk       # x.k equals the copy it matched
                elif lk != rk and how != "left" and rk not in own:
                    self._col_aliases[rk] = lk
                    self._marks.setdefault((alias, rk), lk)       # z.v equals the left key it matched
            if mixed:                                   # a condition on both tables, once the names are final
                res = self._expr(_and_all(mixed))
                plan = Filter(plan, res) if how == "inner" else self._left_residual(plan, res, rid, lcols)

        while pending:
            for i, (alias, right_plan, right_cols) in enumerate(pending):
                links = self._links(alias, set(right_cols))
                if links:
                    break
            else:                                      # nothing links them: a cross product the WHERE filters
                i, links = 0, None
                alias, right_plan, right_cols = pending[0]
            pending.pop(i)
            if links is None:
                self._check_cross(plan, right_plan, alias)
                const = self._fresh("__jc")
                plan = Project(plan, [col(c) for c in self._columns_of(plan)] + [lit(1).alias(const)])
                right_plan = Project(right_plan, [col(c) for c in right_cols] + [lit(1).alias(const)])
                self._copies.add(const)
                left_on = right_on = [const]
            else:
                left_on = [self._expr(l).args[0] for l, _ in links]
                right_on = [r.name for _, r in links]
            # a right key the query can see (SELECT *) or whose type differs from the left key's is a column
            # of its own: the join would drop it and read the left key
            lt, rt = column_types(plan), column_types(right_plan)
            need = [rk for lk, rk in zip(left_on, right_on)
                    if links is not None and (self._any_star or (lk in lt and rk in rt and lt[lk] != rt[rk]))]
            copied = {}
            if need:
                taken = set(lt) | set(rt) | set(right_cols) | set(self._cur_cols) | self._copies
                for rk in need:
                    name = f"__rk_{alias}_{rk}"
                    while name in taken:
                        name += "_"
                    taken.add(name)
                    copied[rk] = name
                right_plan = Project(right_plan, [col(c) for c in right_cols]
                                     + [col(rk).alias(n) for rk, n in copied.items()])
                self._copies |= set(copied.values())
            plan = Join(plan, right_plan, left_on, right_on, "inner")
            outs = join_output_names(self._cur_cols, right_cols, right_on)
            out_of = dict(outs)
            for c in right_cols:
                if c in out_of:
                    self._star_items.append((out_of[c], c))
                elif c in copied:
                    self._star_items.append((copied[c], c))
            for c, out in outs:
                self._qual[(alias, c)] = out
            for lk, rk in zip(left_on, right_on):
                self._qual[(alias, rk)] = copied.get(rk, lk)
                if rk in copied:
                    self._marks_bare[rk] = copied[rk]
                elif lk != rk:
                    self._col_aliases[rk] = lk
            self._cur_cols += [o for _, o in outs]
            self._left_schema |= set(right_cols)
            self._joined.add(alias)

        return plan

    def _split_on(self, on, alias, right_schema):
        """(``on`` without its conjuncts that test only the right table for NULL, those conjuncts): they
        decide which right rows may match, so they filter the right side before the join."""
        keep, mine = [], []
        for c in _conjuncts(on):
            c0 = _unparen(c)
            t = _unparen(c0.this if isinstance(c0, sge.Not) else c0)
            x = _unparen(t.this) if isinstance(t, sge.Is) else None
            right = isinstance(x, sge.Column) and (x.table.lower() == alias if x.table
                                                   else x.name in right_schema and x.name not in self._left_schema)
            (mine if right else keep).append(c)
        if mine and not keep:
            raise ValueError("JOIN ... ON needs a column = column condition besides IS [NOT] NULL")
        return (_and_all(keep) if keep else on), mine

    def _copy_right_keys(self, node, join, right_plan, right_cols, left_on, right_on, all_keys=False, extra=()):
        """The join keeps only the left key. When the query reads a right key (``u.k``; a bare ``k`` that is
        only the right table's), the right side carries a copy of it under a name no table has: NULL where
        nothing matched, the key where it did. ``u.k`` resolves to the copy."""
        alias = join.this.alias_or_name.lower()
        pairs = dict(zip(right_on, left_on))
        asked = set()
        joins = node.args["joins"]
        later_on = [j.args["on"] for j in joins[[j is join for j in joins].index(True) + 1:] if j.args.get("on")]
        for key, val in list(node.args.items()) + [("on", later_on), ("residual", list(extra))]:
            if key in ("joins", "from_", "with_") or val is None:
                continue
            for e in (val if isinstance(val, list) else [val]):
                for c in (e.find_all(sge.Column) if isinstance(e, sge.Expression) else ()):
                    name = c.name
                    if name in pairs and (c.table.lower() == alias if c.table
                                          else pairs[name] != name and name not in self._left_schema):
                        asked.add(name)
        if all_keys:
            asked |= set(pairs)
        if not asked:
            return right_plan
        taken = set(right_cols) | set(self._cur_cols) | {c for t in self._tables.values()
                                                          for c in getattr(t, "column_names", ())}
        added = []
        for k in sorted(asked):
            name = f"__rk_{alias}_{k}"
            while name in taken:
                name += "_"
            self._marks[(alias, k)] = name
            if pairs[k] != k:
                self._marks_bare[k] = name
            self._copies.add(name)
            added.append(col(k).alias(name))
        return Project(right_plan, [col(c) for c in right_cols] + added)

    @staticmethod
    def _own_typed_keys(plan, right_plan, right_cols, left_on, right_on, how):
        """An inner join's right key column is dropped and read as the left key. When the two key
        columns have different types (an int against a float), the right one must keep its own type
        and values: join on a copy of it, so the original stays an ordinary right column.
        Returns (right plan, right_on, names of the right keys kept as columns)."""
        if how != "inner":
            return right_plan, right_on, set()
        lt, rt = column_types(plan), column_types(right_plan)
        own = {rk for lk, rk in zip(left_on, right_on)
               if lk != rk and lk in lt and rk in rt and lt[lk] != rt[rk] and rk not in lt}
        if not own:
            return right_plan, right_on, own
        taken = set(lt) | set(rt) | set(right_cols)
        copies = {}
        for rk in sorted(own):
            name = f"__key_{rk}"
            while name in taken:
                name = "_" + name
            taken.add(name)
            copies[rk] = name
        exprs = [col(c) for c in right_cols] + [col(rk).alias(copies[rk]) for rk in copies]
        return Project(right_plan, exprs), [copies.get(rk, rk) for rk in right_on], own

    def _links(self, alias: str, cols: set):
        """[(column of the tables joined so far, column of table ``alias``)] from the WHERE conditions
        that equate the two; they are taken out of ``self._conj``. A condition every branch of an OR has
        counts (it is removed from the branches)."""
        def owner(c: sge.Column) -> Optional[str]:
            name = c.name
            if c.table:
                t = c.table.lower()
                return "R" if t == alias else "L" if t in self._joined else None
            in_r = name in cols
            in_l = any(name in cs for a, cs in self._table_cols if a in self._joined)
            return "R" if in_r and not in_l else "L" if in_l and not in_r else None

        links, keep = [], []
        for c in self._conj:
            c0 = _unparen(c)
            eqs = _or_common_equalities(c0) if isinstance(c0, sge.Or) else [c0] if isinstance(c0, sge.EQ) else []
            used = []
            for e in eqs:
                a, b = _unparen(e.this), _unparen(e.expression)
                if isinstance(a, sge.Column) and isinstance(b, sge.Column):
                    oa, ob = owner(a), owner(b)
                    if {oa, ob} == {"L", "R"}:
                        links.append((a, b) if oa == "L" else (b, a))
                        used.append(e)
            if not used:
                keep.append(c)
            elif isinstance(c0, sge.Or):
                rest = _or_without(c0, {_cond_key(e) for e in used})
                if rest is not None:
                    keep.append(rest)
            # a plain equality that was used is dropped
        if links:
            self._conj = keep
        return links

    def _resolve_names(self, sel: sge.Select) -> None:
        """The one place where a column's written spelling becomes its real one. DuckDB matches every
        identifier case-insensitively, quoted or not, and returns the source column's spelling: each column
        reference of ``sel`` is rewritten to the name of the column it finds (its FROM columns, then a SELECT
        alias, then the scopes of the queries around it), so everything after this works on exact names. A
        subquery's own columns are resolved when it is translated, by the translator that builds it (it is
        given the scopes around it); only its references to this query are resolved here (`_correlation`).
        The text as written stays in the node's meta (an aggregate is named from it)."""
        items = ([sel.args["from_"].this] if sel.args.get("from_") is not None else []) \
            + [j.this for j in sel.args.get("joins") or []]
        scope = [(it.alias_or_name.lower(), list(self._source(it)[1])) for it in items]
        joins = sel.args.get("joins") or []
        # names a USING / NATURAL join merges into one column, per join (a bare reference to one is not ambiguous)
        merged_at: List[set] = [set()]
        for j, join in enumerate(joins):
            using = join.args.get("using")
            if using:
                new = {str(c.name).lower() for c in (using if isinstance(using, list) else using.expressions)}
            elif join.args.get("method") and join.args.get("on") is None:
                new = {c.lower() for _, cs in scope[:j + 1] for c in cs} & {c.lower() for c in scope[j + 1][1]}
            else:
                new = set()
            for n in new - merged_at[-1]:
                self._check_unambiguous(n, scope[:j + 1], merged_at[-1])      # a USING key is one column on the left
            merged_at.append(merged_at[-1] | new)
        self._aliases = {e.alias.lower(): e.alias for e in sel.expressions if isinstance(e, sge.Alias)}
        semi = {scope[j + 1][0] for j, jn in enumerate(joins) if str(jn.args.get("kind") or "").lower() in ("semi", "anti")}
        self._levels = [(scope, merged_at[-1], semi), *self._outer]      # a SEMI / ANTI table is seen by its own ON only
        self._order_outputs = {(e.alias if isinstance(e, sge.Alias) else e.name).lower()
                               for e in sel.expressions if isinstance(e, (sge.Alias, sge.Column))}

        def visit(n: sge.Expression, upto=None, order=False) -> None:
            for x in n.walk(prune=lambda y: isinstance(y, sge.Select)):
                if isinstance(x, sge.Column):
                    self._resolve_col(x, upto, merged_at, order)

        for key, val in sel.args.items():
            if key in ("with_", "from_") or val is None:
                continue
            for v in val if isinstance(val, list) else [val]:
                if key == "joins":
                    j = joins.index(v)
                    for k2, v2 in v.args.items():
                        for w in v2 if isinstance(v2, list) else [v2]:
                            if k2 != "this" and isinstance(w, sge.Expression):
                                visit(w, j + 2)             # an ON sees the tables up to its own
                elif isinstance(v, sge.Expression):
                    visit(v, order=key == "order")

    @staticmethod
    def _check_unambiguous(name: str, tables: List[tuple], merged: set) -> None:
        """DuckDB's "Ambiguous reference to column name": a bare ``name`` that two or more of ``tables`` have
        (case-insensitively) and no USING / NATURAL join merged into one column."""
        owners = [a for a, cs in tables if _ci_find(cs, name)]
        if len(owners) > 1 and name.lower() not in merged:
            raise ValueError(f"ambiguous column reference: {name} (tables {', '.join(owners)}): "
                             "qualify it with its table")

    def _resolve_col(self, c: sge.Column, upto=None, merged_at=(), order=False) -> None:
        """Rewrite column ``c`` to the spelling of the column it names in this query's scopes. A bare name that
        two tables of one scope have is ambiguous (as in DuckDB), unless a USING / NATURAL join merged it or
        ORDER BY means the SELECT output of that name. ``upto``: the ON of a join sees only the tables so far."""
        if isinstance(c.this, sge.Star):
            return
        hit = None
        for i, (lvl, merged, semi) in enumerate(self._levels):
            if c.table:
                cols = next((cs for a, cs in lvl if a == c.table.lower()), None)
                if cols is None:
                    continue
                hit = _ci_find(cols, c.name)
                break
            vis = lvl[:upto] if i == 0 and upto else lvl
            vis = [t for k, t in enumerate(vis) if t[0] not in semi or (i == 0 and upto and k == upto - 1)]
            if i == 0 and upto:
                merged = merged_at[upto - 1]
            hit = _ci_find([x for _, cs in vis for x in cs], c.name)
            if hit is not None:
                if not (order and i == 0 and c.name.lower() in self._order_outputs):
                    self._check_unambiguous(c.name, vis, merged)
            else:
                hit = _ci_find([x for _, cs in lvl for x in cs], c.name)
                if hit is None and i == 0:
                    hit = self._aliases.get(c.name.lower())
            if hit is not None:
                break
        if hit is not None:
            c.meta.setdefault("written", c.name)
            c.this.set("this", hit)

    def _source(self, table_node: sge.Expression):
        """(plan, column names) of a FROM item, built once per translator."""
        hit = self._sources.get(id(table_node))
        if hit is None or hit[0] is not table_node:
            plan, cols = self._load_source(table_node)
            hit = self._sources[id(table_node)] = (table_node, plan, cols)
        return hit[1], list(hit[2])

    def _load_source(self, table_node: sge.Expression):
        """(plan, column names) of a FROM item: a WITH name, a table, or a derived table (SELECT ...)."""
        from .optimizer import _output_columns
        if isinstance(table_node, sge.Subquery):
            plan = self._child()._build_select(table_node.this)
            cols = _output_columns(plan)
            if cols is None:
                raise ValueError("every column of a derived table needs a name (give it an alias)")
            return plan, cols
        if not isinstance(table_node, sge.Table):
            raise ValueError(f"Unsupported FROM target: {type(table_node).__name__}")
        name = table_node.name.lower()
        if name in self._ctes:
            plan = self._ctes[name]
            cols = _output_columns(plan)
            if cols is None:
                raise ValueError(f"every column of WITH {name} needs a name (give it an alias)")
            return plan, cols
        if name not in self._tables:
            avail = list(self._tables.keys()) + list(self._ctes)
            raise ValueError(f"Unknown table {name!r}. Available: {avail}")
        tbl = self._tables[name]
        if isinstance(tbl, LazyFrame):
            return tbl.plan, tbl.schema.names
        return Scan(tbl), tbl.schema.names

    def _scan(self, table_node: sge.Expression) -> LogicalPlan:
        plan, cols = self._source(table_node)
        self._scope_columns.extend(cols)
        alias = table_node.alias_or_name.lower()
        self._table_cols.append((alias, set(cols)))
        self._alias_schema[alias] = self._schema_of(table_node, plan)
        return plan, cols

    def _schema_of(self, table_node: sge.Expression, plan: LogicalPlan):
        """The Arrow schema of a FROM item: a table's own, or (WITH name, derived table) a zero-row run; None if unknown."""
        if isinstance(table_node, sge.Table) and table_node.name.lower() not in self._ctes \
                and table_node.name.lower() in self._tables:
            return self._tables[table_node.name.lower()].schema
        try:
            return LazyFrame(plan).schema
        except Exception:                                    # noqa: BLE001 (unknown type: callers fall back)
            return None

    def _scope_schemas(self, sel: sge.Select) -> dict:
        """alias -> schema of every FROM item of a subquery."""
        items = []
        if sel.args.get("from_") is not None:
            items.append(sel.args["from_"].this)
        items += [j.this for j in sel.args.get("joins") or []]
        out = {}
        for it in items:
            plan, _ = self._child()._source(it)
            out[it.alias_or_name.lower()] = self._schema_of(it, plan)
        return out

    def _register_ctes(self, with_node) -> None:
        if with_node is None:
            return
        if with_node.args.get("recursive"):
            raise NotImplementedError("WITH RECURSIVE is not supported")
        for cte in with_node.expressions:
            alias = cte.args["alias"]
            if alias.args.get("columns"):
                raise NotImplementedError("a column list on a WITH name (WITH a(x, y) AS ...) is not supported: "
                                          "name the columns in the SELECT")
            self._ctes[alias.name.lower()] = self._child()._build_select(cte.this)

    def _on_side(self, c: sge.Column, alias: str, right_schema: set) -> str:
        """"L" when column ``c`` of a JOIN's ON belongs to the tables joined so far, "R" when it is a column of
        the table being joined (``alias``). A qualifier decides; a bare name must be in exactly one side."""
        if c.table:
            t = c.table.lower()
            if t == alias:
                return "R"
            if t in self._joined:
                return "L"
            raise ValueError(f"JOIN ON: unknown table {c.table!r} in {c.sql(dialect='duckdb')}")
        in_l, in_r = c.name in self._left_schema, c.name in right_schema
        if in_l and in_r:
            raise ValueError(f"JOIN ON: ambiguous reference to column {c.name!r} (in {alias!r} and an earlier table): "
                             "qualify it with its table")
        if not (in_l or in_r):
            raise ValueError(f"JOIN ON: unknown column {c.name!r}")
        return "L" if in_l else "R"

    def _split_join_on(self, on: sge.Expression, alias: str, right_schema: set):
        """The AND-ed conditions of a JOIN's ON, sorted by the tables they read: ``(keys, left only, right only,
        both)``. A key is ``a = b`` with all of ``a`` on one side and all of ``b`` on the other (either order, any
        expression: ``t.a + 1 = u.b``), as ``(left expression, right expression)``. The rest are residual
        conditions. ``AND TRUE`` is dropped."""
        keys, lonly, ronly, both = [], [], [], []

        def sides(n):
            return {self._on_side(x, alias, right_schema) for x in n.find_all(sge.Column)}
        for c in _conjuncts(on):
            if isinstance(c, sge.Boolean) and c.this:
                continue
            if c.find(sge.Subquery, sge.Exists):
                raise NotImplementedError(f"a subquery in a JOIN's ON ({c.sql(dialect='duckdb')}) is not supported")
            if isinstance(c, sge.EQ):
                a, b = _unparen(c.this), _unparen(c.expression)
                sa, sb = sides(a), sides(b)
                if sa == {"L"} and sb == {"R"}:
                    keys.append((a, b))
                    continue
                if sa == {"R"} and sb == {"L"}:
                    keys.append((b, a))
                    continue
            both_sides = sides(c)
            if not both_sides:
                raise NotImplementedError(f"the JOIN condition {c.sql(dialect='duckdb')} refers to no column of "
                                          "either table")
            (lonly if both_sides == {"L"} else ronly if both_sides == {"R"} else both).append(c)
        return keys, lonly, ronly, both

    def _fresh(self, base: str) -> str:
        """An internal column name no table, copy or earlier internal column has."""
        taken = set(self._scope_columns) | set(self._cur_cols) | self._copies | self._used_names
        name = base
        while name in taken:
            name += "_"
        self._used_names.add(name)
        return name

    @staticmethod
    def _columns_of(plan: LogicalPlan) -> List[str]:
        from .optimizer import _output_columns
        cols = _output_columns(plan)
        if cols is None:
            raise NotImplementedError("this JOIN needs the columns of its input, which are not known here "
                                      "(give the derived table's columns names)")
        return cols

    def _right_expr(self, node: sge.Expression, alias: str) -> Expr:
        """``node`` (a condition or key over the table being joined only) as an expression over that table's own
        columns: the names the earlier joins gave their columns do not apply to it."""
        node = node.copy()
        for c in node.find_all(sge.Column):
            if not c.table:
                c.set("table", sge.to_identifier(alias))
        saved = self._col_aliases, self._marks_bare
        self._col_aliases, self._marks_bare = {}, {}
        try:
            return self._expr(node)
        finally:
            self._col_aliases, self._marks_bare = saved

    def _join_keys(self, plan, right_plan, right_cols, keys, lonly, ronly, how, alias):
        """Everything an ON clause asks of the join itself, as plain key columns: ``(left plan, right plan,
        left_on, right_on, names of the internal key columns)``.

        - a key that is an expression (``t.a + 1 = u.b``) is computed into a column on its side first;
        - a condition on the right table alone filters the right side (the right rows that may match);
        - a condition on the left table alone filters the left side of an INNER join; in a LEFT join it decides
          whether a left row may match at all, so it becomes one more key: 1 on the right, and on the left 1 where
          it holds and 0 where it fails or is NULL;
        - no equality at all: the key is a constant, a cross product (small inputs only) that the residual
          conditions then filter."""
        right_cols = list(right_cols)
        if ronly:
            right_plan = Filter(right_plan, self._right_expr(_and_all(ronly), alias))
        left_on, right_on, names, left_add, right_add = [], [], [], [], []
        for l, r in keys:
            le = self._expr(l)
            plain_l = le.op == "col"
            plain_r = isinstance(r, sge.Column)
            if plain_l and plain_r:
                left_on.append(le.args[0])
                right_on.append(r.name)
                continue
            name = self._fresh("__jk")
            names.append(name)
            if plain_l:
                left_on.append(le.args[0])
            else:
                left_on.append(name)
                left_add.append(le.alias(name))
            if plain_r:
                right_on.append(r.name)
            else:
                right_on.append(name)
                right_add.append(self._right_expr(r, alias).alias(name))
        if lonly:
            cond = self._expr(_and_all(lonly))
            if how == "inner":
                plan = Filter(plan, cond)
            else:
                gate = self._fresh("__jg")
                names.append(gate)
                left_on.append(gate)
                right_on.append(gate)
                left_add.append(when(cond, 1, 0).alias(gate))
                right_add.append(lit(1).alias(gate))
        if not keys:
            self._check_cross(plan, right_plan, alias)
            if not left_on:
                const = self._fresh("__jc")
                names.append(const)
                left_on.append(const)
                right_on.append(const)
                left_add.append(lit(1).alias(const))
                right_add.append(lit(1).alias(const))
        if left_add:
            plan = Project(plan, [col(c) for c in self._columns_of(plan)] + left_add)
            self._copies |= {a._alias for a in left_add}
        if right_add:
            right_plan = Project(right_plan, [col(c) for c in right_cols] + right_add)
        return plan, right_plan, left_on, right_on, set(names)

    @staticmethod
    def _check_cross(plan: LogicalPlan, right_plan: LogicalPlan, alias: str) -> None:
        """A join with no equality is a cross product: refuse it unless both inputs are known to be small."""
        lrows, rrows = _rows_upper(plan), _rows_upper(right_plan)
        if lrows is None or rrows is None or lrows * rrows > _CROSS_LIMIT:
            size = "of an unknown size" if lrows is None or rrows is None else f"up to {lrows} x {rrows} rows"
            raise NotImplementedError(
                f"JOIN {alias}: no equality between the two tables, so it would run as a cross join "
                f"({size}; up to {_CROSS_LIMIT} row pairs are supported): add a column = column condition")

    def _left_residual(self, joined: LogicalPlan, res: Expr, rid: str, lcols: List[str]) -> LogicalPlan:
        """``joined`` is ``L LEFT JOIN R ON <keys>`` with a row id on L; ``res`` is the ON's remaining condition
        over both tables. In SQL it decides which right rows match: a left row none of whose partners satisfy
        it still comes back, once, with NULLs for the right columns. So: keep the rows that satisfy it; for a
        left row (``rid``) with no such row keep one row (its first), and blank its right columns."""
        cols = self._columns_of(joined)
        right = cols[len(lcols):]
        ok, anyok, nth = self._fresh("__ok"), self._fresh("__any"), self._fresh("__nth")
        plan = Project(joined, [col(c) for c in cols] + [when(res, 1, 0).alias(ok)])
        plan = Project(plan, [col(c) for c in cols + [ok]]
                       + [col(ok).max().over(rid).alias(anyok), row_number().over(rid).alias(nth)])
        plan = Filter(plan, (col(ok) == 1) | ((col(anyok) == 0) & (col(nth) == 1)))
        return Project(plan, [col(c) for c in lcols if c != rid]
                       + [when(col(ok) == 1, col(c)).alias(c) for c in right])

    # ------------------------------------------------------------------
    # aggregate building
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # subqueries (F-065): see the module docstring for the forms
    # ------------------------------------------------------------------

    def _scope_of(self, sel: sge.Select) -> List[tuple]:
        """(alias, column names) of every FROM item of a subquery."""
        items = []
        if sel.args.get("from_") is not None:
            items.append(sel.args["from_"].this)
        items += [j.this for j in sel.args.get("joins") or []]
        return [(it.alias_or_name.lower(), {c.lower() for c in self._child()._source(it)[1]}) for it in items]

    def _outer_refs(self, node: sge.Expression, scope: List[tuple]) -> List[sge.Column]:
        """The columns in ``node`` that the subquery's own FROM (``scope``) does not provide: they belong to
        this query (a correlated reference). Columns of subqueries nested in ``node`` are theirs to resolve."""
        mine = {a for a, _ in self._table_cols}
        aliases = {a for a, _ in scope}
        inner_cols = set().union(*[c for _, c in scope]) if scope else set()
        out = []
        for c in node.walk(prune=lambda n: isinstance(n, (sge.Subquery, sge.Exists))):
            if not isinstance(c, sge.Column):
                continue
            if c.table:
                if c.table.lower() in aliases:
                    continue
                known = c.table.lower() in mine
            else:
                if c.name.lower() in inner_cols:
                    continue
                known = c.name.lower() in {x.lower() for x in self._scope_columns}
            if not known:
                raise ValueError(f"unknown column {c.sql(dialect='duckdb')} in a subquery")
            out.append(c)
        return out

    def _correlation(self, sel: sge.Select, what: str):
        """Split a subquery's WHERE into ([(outer column, inner expression)], [inner-only conditions]): the
        equalities ``inner = outer`` become join keys. Any other reference to this query's columns raises."""
        scope = self._scope_of(sel)
        where = sel.args.get("where")
        body = sel.copy()
        body.set("where", None)
        if self._outer_refs(body, scope):
            raise NotImplementedError(f"{what}: the subquery uses a column of the outer query outside its WHERE "
                                      "(only inner.col = outer.col conditions in WHERE can be decorrelated)")
        keys, rest = [], []
        for c in _conjuncts(where.this) if where else []:
            if not self._outer_refs(c, scope):
                rest.append(c)
                continue
            if isinstance(c, sge.EQ):
                a, b = _unparen(c.this), _unparen(c.expression)
                oa, ob = bool(self._outer_refs(a, scope)), bool(self._outer_refs(b, scope))
                if oa != ob:
                    outer, inner = (a, b) if oa else (b, a)
                    if isinstance(outer, sge.Column):
                        self._resolve_col(outer)          # its spelling in this query; the subquery does not know it
                        keys.append((outer, inner))
                        continue
            raise NotImplementedError(f"{what}: correlated predicate {c.sql(dialect='duckdb')} cannot be decorrelated "
                                      "(only inner.col = outer.col equalities become a join)")
        return keys, rest

    @staticmethod
    def _keyed_select(sel: sge.Select, rest, items) -> sge.Select:
        """``sel`` with the given SELECT items and only its inner-only conditions."""
        new = sel.copy()
        new.set("expressions", items)
        new.set("where", sge.Where(this=_and_all(rest)) if rest else None)
        new.set("order", None)
        return new

    def _unalias_where(self, sel: sge.Select) -> sge.Select:
        """``sel`` with an inner SELECT alias used in its WHERE replaced by the expression it names (DuckDB lets
        a WHERE use a SELECT alias; a column of the subquery's FROM with that name wins). The decorrelation
        rebuilds the SELECT list, which would lose the alias."""
        where = sel.args.get("where")
        aliases = {e.alias.lower(): e.this for e in sel.expressions if isinstance(e, sge.Alias)}
        if where is None or not aliases:
            return sel
        inner = {c for _, cs in self._scope_of(sel) for c in cs}
        hits = [c for c in where.walk(prune=lambda n: isinstance(n, (sge.Subquery, sge.Exists)))
                if isinstance(c, sge.Column) and not c.table and c.name.lower() not in inner
                and c.name.lower() in aliases]
        if not hits:
            return sel
        sel = sel.copy()
        for c in [c for c in sel.args["where"].walk(prune=lambda n: isinstance(n, (sge.Subquery, sge.Exists)))
                  if isinstance(c, sge.Column) and not c.table and c.name.lower() not in inner
                  and c.name.lower() in aliases]:
            c.replace(sge.Paren(this=aliases[c.name.lower()].copy()))
        return sel

    def _select_subqueries(self, plan: LogicalPlan, node: sge.Select) -> LogicalPlan:
        """Correlated scalar subqueries in the SELECT list: ``SELECT a, (SELECT max(b) FROM u WHERE u.k = t.k)
        FROM t``. Each is the per-key aggregate LEFT-joined on the key, and the item is rewritten over the
        aggregate columns. A key without partners sees every aggregate on no rows: NULL, except COUNT, which is
        0 (so ``count(*) + 1`` is 1 and ``coalesce(sum(b), 0)`` is 0)."""
        found = []
        for e in node.expressions:
            for n in e.walk(prune=lambda x: isinstance(x, (sge.Subquery, sge.Exists))):
                if isinstance(n, sge.Subquery) and isinstance(n.this, sge.Select) \
                        and self._outer_refs(self._unalias_where(n.this), self._scope_of(n.this)):
                    found.append(n)
        if not found:
            return plan
        what = "correlated scalar subquery in the SELECT list"
        if node.args.get("group") or any(_aggs_in(e) for e in node.expressions):
            raise NotImplementedError(f"a {what} together with GROUP BY or aggregates in the outer query is not "
                                      "supported (compute the aggregate in a subquery first)")
        for sub in found:
            sel = self._unalias_where(sub.this)
            self._check_scalar(sel, what)
            keys, rest = self._correlation(sel, what)
            i = next(self._ids)
            item = sel.expressions[0]
            item = _unparen(item.this if isinstance(item, sge.Alias) else item).copy()
            holder = sge.Paren(this=item)
            aggs = _aggs_in(holder)
            items = [_alias(inner, f"__k{i}_{j}") for j, (_, inner) in enumerate(keys)] \
                + [_alias(a, f"__a{i}_{j}") for j, a in enumerate(aggs)]
            new = self._keyed_select(sel, rest, items)
            new.set("group", sge.Group(expressions=[inner.copy() for _, inner in keys]))
            inner_plan = self._child()._build_select(new)
            plan = Join(plan, inner_plan, [self._expr(o).args[0] for o, _ in keys],
                        [f"__k{i}_{j}" for j in range(len(keys))], "left")
            for j, a in enumerate(aggs):
                ref = sge.Column(this=sge.to_identifier(f"__a{i}_{j}"))
                a.replace(sge.Coalesce(this=ref, expressions=[sge.Literal.number(0)]) if isinstance(a, sge.Count)
                          else ref)
            value = holder
            if isinstance(sub.parent, sge.Select):                    # an unaliased item keeps a readable name
                value = sge.Alias(this=holder, alias=sge.to_identifier(sub.sql(dialect="duckdb")))
            sub.replace(value)
        return plan

    def _where(self, plan: LogicalPlan, cond: sge.Expression) -> LogicalPlan:
        """WHERE: plain conditions as one Filter; IN / EXISTS subqueries (top-level AND conditions) as semi /
        anti joins, a correlated scalar comparison as a join with the per-key aggregate."""
        forms, plain = [], []
        for c in _conjuncts(cond):
            f = self._subquery_form(c)
            if f:
                forms.append(f)
            else:
                plain.append(c)
        if not forms:
            return Filter(plan, self._expr(cond))
        before = None
        if self._has_star:
            from .optimizer import _output_columns
            before = _output_columns(plan)
        if plain:
            plan = Filter(plan, self._expr(_and_all(plain)))
        for kind, c, neg in forms:
            plan = getattr(self, f"_{kind}_form")(plan, c, neg)
        if before is not None:
            plan = Project(plan, [col(n) for n in before])      # drop the columns a decorrelation added
        return plan

    def _subquery_form(self, c: sge.Expression):
        """(kind, node, negated) when ``c`` is a condition that becomes a join, else None."""
        neg = False
        if isinstance(c, sge.Not):
            c, neg = _unparen(c.this), True
        if isinstance(c, sge.Exists):
            return "exists", c, neg
        if isinstance(c, sge.In) and c.args.get("query") is not None:
            return "in", c, neg
        if not neg and isinstance(c, _COMPARE):
            for side in (c.this, c.expression):
                side = _unparen(side)
                if isinstance(side, sge.Subquery) and isinstance(side.this, sge.Select) \
                        and self._outer_refs(self._unalias_where(side.this), self._scope_of(side.this)):
                    return "scalar", c, False
        return None

    def _exists_form(self, plan, node: sge.Exists, negated: bool):
        what = "NOT EXISTS" if negated else "EXISTS"
        sel = node.this
        if not isinstance(sel, sge.Select):
            raise NotImplementedError(f"{what} over a {type(sel).__name__} is not supported")
        sel = self._unalias_where(sel)
        _unsupported_shape(sel, what)
        keys, rest = self._correlation(sel, what)
        if not keys:         # no outer column: there are rows or there are none
            flat = sel.copy()
            flat.set("expressions", [sge.Star()])
            n = Expr("scalar", Aggregate(self._child()._build_select(flat), [], [lit(1).count().alias("__scalar")]))
            return Filter(plan, (n == lit(0)) if negated else (n > lit(0)))
        i = next(self._ids)
        items = [_alias(inner, f"__k{i}_{j}") for j, (_, inner) in enumerate(keys)]
        inner_plan = self._child()._build_select(self._keyed_select(sel, rest, items))
        return Join(plan, inner_plan, [self._expr(o).args[0] for o, _ in keys],
                    [f"__k{i}_{j}" for j in range(len(keys))], "anti" if negated else "semi")

    def _in_form(self, plan, node: sge.In, negated: bool):
        what = "NOT IN (SELECT ...)" if negated else "IN (SELECT ...)"
        sel = node.args["query"].this
        if not isinstance(sel, sge.Select):
            raise NotImplementedError(f"{what} over a {type(sel).__name__} (UNION / INTERSECT / EXCEPT) is not supported")
        x = _unparen(node.this)
        if not isinstance(x, sge.Column):
            raise NotImplementedError(f"{what}: the left side must be a plain column, got {x.sql(dialect='duckdb')}")
        if len(sel.expressions) != 1:
            raise ValueError(f"{what}: the subquery must return one column, it returns {len(sel.expressions)}")
        sel = self._unalias_where(sel)
        item = sel.expressions[0]
        if isinstance(item, sge.Star):
            raise NotImplementedError(f"{what}: SELECT * in the subquery is not supported (name the column)")
        item = item.this if isinstance(item, sge.Alias) else item
        left = self._expr(x).args[0]
        keys, rest = self._correlation(sel, what)
        i = next(self._ids)
        int_lits = self._in_item_ints(x, item, sel, what)
        if keys:
            if negated:
                raise NotImplementedError("a correlated NOT IN (SELECT ...) is not supported (its NULL rules differ "
                                          "per outer row); use NOT EXISTS")
            _unsupported_shape(sel, what)
            items = [_alias(item, "__in")] + [_alias(inner, f"__k{i}_{j}") for j, (_, inner) in enumerate(keys)]
            if int_lits:
                _keep_ints(items[0])
            inner_plan = self._child()._build_select(self._keyed_select(sel, rest, items))
            return Join(plan, inner_plan, [left] + [self._expr(o).args[0] for o, _ in keys],
                        ["__in"] + [f"__k{i}_{j}" for j in range(len(keys))], "semi")
        new = sel.copy()
        new.set("expressions", [_alias(item, "__in")])
        if not (new.args.get("limit") or new.args.get("offset")):
            new.set("order", None)                     # the order only matters to pick the LIMIT rows
        if int_lits:
            _keep_ints(new.expressions[0])
        inner_plan = self._child()._build_select(new)
        if not negated:
            return Join(plan, inner_plan, [left], ["__in"], "semi")
        # NOT IN keeps a row with no partner when the subquery is empty, or when it has no NULL and the
        # row's value is not NULL. m: 1 = empty, 2 = rows and no NULL, 0 = some NULL (nothing is kept)
        stats = Aggregate(inner_plan, [], [lit(1).count().alias("n"), col("__in").count().alias("nn")])
        m = Expr("scalar", Project(stats, [when(col("n") == lit(0), lit(1),
                                                when(col("nn") == col("n"), lit(2), lit(0))).alias("__scalar")]))
        anti = Join(plan, inner_plan, [left], ["__in"], "anti")
        return Filter(anti, (m == lit(1)) | ((m == lit(2)) & (col(left) == col(left))))

    def _in_item_ints(self, x: sge.Column, item: sge.Expression, sel: sge.Select, what: str) -> bool:
        """Arithmetic in an IN subquery's item makes a float (integer literals are floats there), which cannot
        join an integer key. When the key and every column of the item are integers (and there is no division)
        the item is computed in integers; any other arithmetic against an integer key raises."""
        ops = (sge.Add, sge.Sub, sge.Mul, sge.Div)
        if not any(isinstance(n, ops) for n in item.walk()):
            return False
        kt = self._col_type(x)
        if kt is None or not pa.types.is_integer(kt):
            return False
        inner = self._scope_schemas(sel)
        cols = list(item.find_all(sge.Column))
        ints = all((t := _type_in(c, inner)) is not None and pa.types.is_integer(t) for c in cols)
        lits = all(str(n.this).isdigit() for n in item.find_all(sge.Literal) if not n.is_string)
        if ints and lits and not any(isinstance(n, sge.Div) for n in item.walk()):
            return True
        raise NotImplementedError(f"{what}: the subquery computes {item.sql(dialect='duckdb')}, a float, against the "
                                  f"integer column {x.sql(dialect='duckdb')}; keys of different types cannot join "
                                  "(use integer columns and constants, without division)")

    def _scalar_form(self, plan, c: sge.Expression, _neg: bool):
        """``expr <op> (SELECT agg ... WHERE inner.k = outer.k)``: the aggregate per key, joined on the key."""
        sub = next(_unparen(s) for s in (c.this, c.expression)
                   if isinstance(_unparen(s), sge.Subquery) and isinstance(_unparen(s).this, sge.Select)
                   and self._outer_refs(_unparen(s).this, self._scope_of(_unparen(s).this)))
        sel = sub.this
        what = "correlated scalar subquery"
        self._check_scalar(sel, what)
        item0 = _unparen(sel.expressions[0].this if isinstance(sel.expressions[0], sge.Alias) else sel.expressions[0])
        if not _null_on_empty(item0):
            raise NotImplementedError(
                f"{what}: only aggregates other than COUNT combined by + - * / and constants are supported "
                f"(got {item0.sql(dialect='duckdb')}): an outer row without partners must see the item's value on "
                "no rows (NULL for these, but 0 for COUNT, or whatever CASE / COALESCE give), which a join would drop")
        keys, rest = self._correlation(sel, what)
        i = next(self._ids)
        item = sel.expressions[0]
        item = item.this if isinstance(item, sge.Alias) else item
        items = [_alias(inner, f"__k{i}_{j}") for j, (_, inner) in enumerate(keys)] + [_alias(item, f"__s{i}")]
        new = self._keyed_select(sel, rest, items)
        new.set("group", sge.Group(expressions=[inner.copy() for _, inner in keys]))
        inner_plan = self._child()._build_select(new)
        plan = Join(plan, inner_plan, [self._expr(o).args[0] for o, _ in keys],
                    [f"__k{i}_{j}" for j in range(len(keys))], "inner")
        cmp = c.copy()
        for side in ("this", "expression"):
            if isinstance(_unparen(cmp.args[side]), sge.Subquery):
                cmp.set(side, sge.Column(this=sge.to_identifier(f"__s{i}")))
        return Filter(plan, self._expr(cmp))

    @staticmethod
    def _check_scalar(sel: sge.Expression, what: str) -> None:
        if not isinstance(sel, sge.Select):
            raise NotImplementedError(f"{what} over a {type(sel).__name__} (UNION / INTERSECT / EXCEPT) is not supported")
        if len(sel.expressions) != 1:
            raise ValueError(f"{what}: the subquery must return one column, it returns {len(sel.expressions)}")
        if sel.args.get("group") or sel.args.get("having") or sel.args.get("limit") or sel.args.get("offset") \
                or sel.args.get("distinct") or not _aggs_in(sel.expressions[0]):
            raise NotImplementedError(f"{what}: only a single aggregate over an ungrouped query is supported "
                                      "(SELECT AVG(x) FROM ...), not a grouped, DISTINCT or LIMIT subquery or a plain column")

    def _scalar_subquery(self, node: sge.Subquery) -> Expr:
        """An uncorrelated ``(SELECT agg FROM ...)`` inside an expression: the engine runs it first
        (``LazyFrame.scalar``) and uses its value as a literal."""
        sel = node.this
        self._check_scalar(sel, "scalar subquery")
        if self._outer_refs(sel, self._scope_of(sel)):
            raise NotImplementedError("a correlated scalar subquery is supported only as a comparison that is a "
                                      "top-level AND condition of WHERE (decorrelated to a join), not here")
        item = sel.expressions[0]
        new = sel.copy()
        new.set("expressions", [item if isinstance(item, sge.Alias) else _alias(item, "__scalar")])
        return Expr("scalar", self._child()._build_select(new))

    def _agg_parts(self, node: sge.Expression):
        """(kind, expression) of an aggregate call: kind is sum / avg / min / max / count, or "cd" for
        COUNT(DISTINCT x) (the expression is then x). Anything else raises."""
        kind = _AGG_NAMES.get(type(node))
        if kind is None:
            raise NotImplementedError(f"Unsupported aggregate function: {node.sql_name()}")
        arg = node.this
        post, self._post = self._post, False
        try:
            if isinstance(arg, sge.Distinct):
                if kind != "count":
                    raise NotImplementedError(f"{kind.upper()}(DISTINCT ...) is not supported (only COUNT(DISTINCT x))")
                if len(arg.expressions) != 1:
                    raise NotImplementedError("COUNT(DISTINCT a, b) over several columns is not supported")
                return "cd", self._expr(arg.expressions[0])
            return kind, self._expr(node)
        finally:
            self._post = post

    def _slot(self, node: sge.Expression, out: Optional[str] = None) -> str:
        """The Aggregate output column that holds aggregate call ``node`` (registered on first use;
        the same call is computed once)."""
        kind, ex = self._agg_parts(node)
        sig = (kind, ex.signature())
        name = self._agg_names.get(sig)
        if name is None:
            if kind == "cd":
                if self._cd is not None:
                    raise NotImplementedError("more than one COUNT(DISTINCT ...) in a query is not supported")
                name = out or "__agg_cd"
                self._cd = (ex, name)
            else:
                taken = {k._alias for k in self._key_names_list} | set(self._agg_names.values())
                name = out if out and out not in taken else f"__agg{len(self._aggs)}"
                self._aggs.append(ex.alias(name))
            self._agg_names[sig] = name
        return name

    def _post_ref(self, node: sge.Expression) -> Optional[Expr]:
        """Above the Aggregate (HAVING, ORDER BY, arithmetic on aggregates): an aggregate call or a
        group key is the column the Aggregate produced; a bare column that is neither is an error."""
        if isinstance(node, sge.Literal):
            return None
        if isinstance(node, sge.AggFunc):
            return col(self._slot(node))
        if _aggs_in(node):
            return None                                   # its parts are resolved one by one
        self._post = False
        try:
            ex = self._expr(node)
        finally:
            self._post = True
        name = self._key_names.get(ex.signature())
        if name is not None:
            return col(name)
        if isinstance(node, sge.Column):
            src = None if node.table else self._out_src.get(_ci_find(self._out_src, node.name))
            if src is None:
                raise ValueError(f"column {node.sql(dialect='duckdb')} must appear in the GROUP BY clause "
                                 "or inside an aggregate function")
            return col(src)
        return None

    def _build_aggregate(self, plan: LogicalPlan, node: sge.Select, group_node, having):
        """The plan for a grouped SELECT: Aggregate, then (arithmetic on aggregates) a Project, then
        HAVING as a Filter. Returns it with the final projection (result columns in SELECT order,
        applied after ORDER BY / LIMIT, which may use a key or aggregate the SELECT list leaves out;
        None when the columns are already in place). ORDER BY is translated here too (``self._sort``)
        because it may need aggregates the Aggregate has to compute."""
        items = list(node.expressions)
        if any(isinstance(e, sge.Star) for e in items):
            raise ValueError("SELECT * cannot be combined with GROUP BY or aggregates")

        def split(e):
            return (e.this, e.alias) if isinstance(e, sge.Alias) else (e, None)

        # Group-by keys. A SELECT alias names its key: GROUP BY may name the alias (when no input
        # column has that name, as in DuckDB) or repeat the aliased expression (compared after
        # translation, so t.s and s are the same key). GROUP BY ALL groups by every SELECT item
        # without an aggregate.
        plain = [split(e) for e in items if not _aggs_in(e)]
        aliased = [(a, self._expr(i).signature()) for i, a in plain if a]
        alias_expr = {a.lower(): i for i, a in plain if a}
        scope = {c.lower() for c in self._scope_columns}
        if group_node is not None and group_node.args.get("all"):
            group_exprs = [i for i, _ in plain]
        else:
            group_exprs = list(group_node.expressions) if group_node else []
        group_by: List[Expr] = []
        for n, gb in enumerate(group_exprs):
            inner = gb
            i = _ordinal(gb, len(items))
            if i is not None:
                if _aggs_in(items[i]):
                    raise ValueError("GROUP BY position refers to an aggregate")
                inner = split(items[i])[0]
            elif (isinstance(gb, sge.Column) and not gb.table and gb.name.lower() in alias_expr
                    and gb.name.lower() not in scope):
                inner = alias_expr[gb.name.lower()]
            key = self._expr(inner)
            if key.signature() in {k.signature() for k in group_by}:
                continue
            name = next((a for a, sg in aliased if sg == key.signature()), None)
            if name is None:
                name = key.args[0] if key.op == "col" else f"__key{n}"
            group_by.append(key.alias(name))
        # output name of every key, by its translated expression (ORDER BY, HAVING and the final
        # projection refer to a key through it, whatever name or qualification the query used)
        self._key_names = {k.signature(): k._alias for k in group_by}
        self._key_names_list = group_by

        # SELECT items: (output name, column of the Aggregate output). Plain aggregates first, so
        # they keep their own names; then arithmetic on aggregates and group keys.
        outs: List[Optional[tuple]] = [None] * len(items)
        calcs: List[tuple] = []
        for n, e in enumerate(items):
            inner, alias = split(e)
            if isinstance(inner, sge.AggFunc):
                out = alias or _default_agg_name(inner)
                outs[n] = (out, self._slot(inner, out))
        for n, e in enumerate(items):
            if outs[n] is not None:
                continue
            inner, alias = split(e)
            if _aggs_in(inner):
                if alias is None:
                    raise ValueError(f"give {inner.sql(dialect='duckdb')} an alias (AS name): the output column has no name")
                self._post = True
                try:
                    ex = self._expr(inner)
                finally:
                    self._post = False
                calcs.append((f"__calc{n}", ex))
                outs[n] = (alias, f"__calc{n}")
                continue
            src = self._key_names.get(self._expr(inner).signature())
            if src is None:
                raise ValueError(f"column {inner.sql(dialect='duckdb')} must appear in the GROUP BY clause "
                                 "or inside an aggregate function")
            if alias is None and not isinstance(inner, sge.Column):
                raise ValueError(f"give {inner.sql(dialect='duckdb')} an alias (AS name): the output column has no name")
            outs[n] = (alias or inner.name, src)
        outs = list(zip(_dedupe([o for o, _ in outs]), [src for _, src in outs]))
        self._out_src = {}
        for o, src in outs:
            self._out_src.setdefault(o, src)

        # HAVING and ORDER BY, above the Aggregate (they may need aggregates the SELECT list lacks)
        self._post = True
        try:
            having_expr = self._expr(having.this) if having else None
            order_node = node.args.get("order")
            if order_node:
                by = []
                for o in order_node.expressions:
                    ref = o.this
                    i = _ordinal(ref, len(items))
                    if i is not None:
                        by.append(col(outs[i][1]))
                    elif isinstance(ref, sge.Column) and not ref.table and _ci_find(self._out_src, ref.name):
                        by.append(col(self._out_src[_ci_find(self._out_src, ref.name)]))   # a SELECT output name
                    else:
                        ex = self._expr(ref)
                        if ex.op != "col":              # Sort takes columns: compute the expression first
                            sig = ex.signature()
                            nm = next((c for c, e in calcs if e.signature() == sig), None)
                            if nm is None:
                                nm = f"__ord{len(by)}"
                                calcs.append((nm, ex))
                            ex = col(nm)
                        by.append(ex)
                self._sort = (by, [bool(o.args.get("desc")) for o in order_node.expressions])
        finally:
            self._post = False

        # the Aggregate
        if self._cd is not None:
            x, cd_name = self._cd
            names = [k._alias for k in group_by]
            if not self._aggs:
                # distinct (keys, x) rows, then a count per key: the engine's count-distinct path
                proj = [k for k in group_by] + [x.alias("__cd")]
                agg_plan = Aggregate(Distinct(Project(plan, proj), None), [col(nm) for nm in names],
                                     [col("__cd").count().alias(cd_name)])
                frame = names + [cd_name]
            else:
                # aggregate per (keys, x) first, then combine the partials per key (sum of sums, min of
                # mins, ...) and count the x values: no join, so null keys stay a group
                inner, outer, avg_calcs = [], [], []
                for a in self._aggs:
                    kind, arg, nm = a.op, a.args[0], a._alias
                    if kind == "mean":
                        inner += [arg.sum().alias(f"__ps_{nm}"), arg.count().alias(f"__pc_{nm}")]
                        outer += [col(f"__ps_{nm}").sum().alias(f"__s_{nm}"), col(f"__pc_{nm}").sum().alias(f"__c_{nm}")]
                        avg_calcs.append((nm, col(f"__s_{nm}") / col(f"__c_{nm}")))
                    else:
                        inner.append(getattr(arg, kind)().alias(f"__p_{nm}"))
                        outer.append(getattr(col(f"__p_{nm}"), "sum" if kind == "count" else kind)().alias(nm))
                outer.append(col("__cdx").count().alias(cd_name))
                agg_plan = Aggregate(Aggregate(plan, group_by + [x.alias("__cdx")], inner),
                                     [col(nm) for nm in names], outer)
                frame = names + [o._alias for o in outer]
                calcs = avg_calcs + calcs
        else:
            if not self._aggs:
                raise ValueError("GROUP BY without an aggregate function is not supported (use SELECT DISTINCT)")
            agg_plan = Aggregate(plan, group_by, self._aggs)
            frame = [k._alias for k in group_by] + [a._alias for a in self._aggs]
        plan = agg_plan
        if calcs:
            plan = Project(plan, [col(c) for c in frame] + [ex.alias(nm) for nm, ex in calcs])
            frame += [nm for nm, _ in calcs]
        if having_expr is not None:
            plan = Filter(plan, having_expr)

        if [src for _, src in outs] == frame and [o for o, _ in outs] == frame:
            return plan, None
        return plan, [col(src) if out == src else col(src).alias(out) for out, src in outs]

    # ------------------------------------------------------------------
    # SELECT-column translation (handles Alias wrapper)
    # ------------------------------------------------------------------

    def _select_expr(self, e: sge.Expression) -> Optional[Expr]:
        if isinstance(e, sge.Star):
            return None
        alias_name: Optional[str] = None
        if isinstance(e, sge.Alias):
            alias_name = e.alias
            inner = e.this
        else:
            inner = e
        mx = self._value(inner)             # a DATE constant that is an output is a date32 (F-115)
        if not alias_name and isinstance(inner, sge.Column) and mx.op == "col":
            alias_name = inner.name if inner.name != mx.args[0] else None   # joined key: as written
        if alias_name:
            mx = mx.alias(alias_name)
        return mx

    # ------------------------------------------------------------------
    # expression translation
    # ------------------------------------------------------------------

    def _like(self, node, escape_node):
        """``x [I]LIKE 'pattern' [ESCAPE 'c']``; ``%x``-free shapes keep the cheaper prefix / substring ops."""
        from .str_patterns import parse_like
        def literal(n, what):
            if not (isinstance(n, sge.Literal) and n.is_string):
                raise NotImplementedError(f"LIKE: the {what} must be a string literal")
            return n.this
        pattern = literal(node.expression, "pattern")
        escape = None if escape_node is None else literal(escape_node, "escape character")
        c = self._expr(node.this)
        ignore = isinstance(node, sge.ILike)
        try:
            kinds = [k for k, _ in parse_like(pattern, escape)]
            text = "".join(t for k, t in parse_like(pattern, escape) if k == "lit")
        except ValueError as e:
            raise ValueError(f"LIKE {pattern!r}: {e}") from None
        if not ignore and text:
            if kinds == ["lit", "run"]:
                return c.startswith(text)
            if kinds == ["run", "lit", "run"]:
                return c.contains(text)
        return c.like(pattern, escape, ignore)

    def _expr(self, node: sge.Expression) -> Expr:
        """Translate a sqlglot expression to an MXFrame Expr."""
        if isinstance(node, sge.Subquery):
            return self._scalar_subquery(node)
        if isinstance(node, sge.Exists) or (isinstance(node, sge.In) and node.args.get("query") is not None):
            what = "EXISTS (SELECT ...)" if isinstance(node, sge.Exists) else "IN (SELECT ...)"
            raise NotImplementedError(f"{what} is supported only as a condition joined by AND at the top of WHERE "
                                      "(it becomes a semi / anti join); not inside OR, NOT (...), CASE, SELECT or HAVING")
        d = _date_of(node)
        if d is not None:                                   # next to a date32 column a date, else a day number
            return lit(d) if self._date_lit else lit((d - _EPOCH).days)
        t = _timestamp_of(node)
        if t is not None:
            return lit(t)
        if not self._date_lit and isinstance(node, _COMPARE + (sge.Between,)):
            kids = [node.this, node.args["low"], node.args["high"]] if isinstance(node, sge.Between) \
                else [node.this, node.expression]
            if any(self._is_date_col(k) for k in kids):
                self._date_lit = True
                try:
                    return self._expr(node)
                finally:
                    self._date_lit = False
        if isinstance(node, (sge.Add, sge.Sub, sge.Mul)):
            v = _num_const(node)
            if v is not None:
                is_int = v == v.to_integral_value() and all(str(x.this).isdigit() for x in node.find_all(sge.Literal))
                if is_int:
                    return lit(int(v))
                return lit(np.float64(v)) if any(_is_double_literal(x) for x in node.find_all(sge.Literal)) else lit(float(v))
        if isinstance(node, sge.Extract):
            return self._extract(node)
        if isinstance(node, sge.Substring):
            return self._substring(node)
        if self._post:
            ref = self._post_ref(node)
            if ref is not None:
                return ref

        # ---- column reference ----
        if isinstance(node, sge.Column):
            name = node.name
            copy = self._marks.get((node.table.lower(), name)) if node.table else self._marks_bare.get(name)
            if copy:                                    # a LEFT JOIN's right key: NULL where unmatched
                return col(copy)
            if node.table and (node.table.lower(), name) in self._qual:
                return col(self._qual[(node.table.lower(), name)])        # a comma-joined table's own copy
            if node.table and (node.table.lower(), name) in self._renamed:
                return col(self._renamed[(node.table.lower(), name)])     # a clashing column the join renamed
            # Remap right join keys to their left-side equivalents (hash join
            # drops the right join key and keeps the left one in the output)
            name = self._col_aliases.get(name, name)
            return col(name)

        # ---- literals ----
        if isinstance(node, sge.Literal):
            if node.is_string:
                return lit(node.this)
            v = _number(node.this)
            return lit(np.float64(v)) if _is_double_literal(node) else lit(v)      # np.float64: a DOUBLE, kept at 64 bits
        f = _float_cast(node)
        if f is not None:
            return lit(f)

        if isinstance(node, sge.Cast):
            return self._cast(node)
        if isinstance(node, sge.Coalesce):
            return self._coalesce(node)
        if isinstance(node, sge.Nullif):
            return self._nullif(node)
        if isinstance(node, sge.Abs):
            self._no_constants(node, node.this)
            return self._expr(node.this).abs()

        # ---- parentheses ----
        if isinstance(node, sge.Paren):
            return self._expr(node.this)

        # ---- negation ----
        if isinstance(node, sge.Neg):
            n = _unparen(node.this)
            if isinstance(n, sge.Literal) and not n.is_string and not _is_double_literal(n):
                return lit(_number(n.this, neg=True))             # the sign belongs to the literal: -2^63 is an int64
            inner = self._expr(node.this)
            if inner.op == "lit" and isinstance(inner.args[0], (int, float)) and not isinstance(inner.args[0], bool):
                return lit(-inner.args[0])                      # a constant folds (`-1`, `-(-1)`), not 0 - 1 (a float32 scalar)
            return lit(0) - inner

        # ---- arithmetic ----
        if isinstance(node, (sge.Add, sge.Sub)) and isinstance(_unparen(node.expression), sge.Interval):
            return self._interval(node)
        if isinstance(node, sge.Add):
            a, b = self._widened(node.this, node.expression, self._arith_operand, node)
            return a + b
        if isinstance(node, sge.Sub):
            a, b = self._widened(node.this, node.expression, self._arith_operand, node)
            return a - b
        if isinstance(node, sge.Mul):
            a, b = self._widened(node.this, node.expression, self._arith_operand, node)
            return a * b
        if isinstance(node, sge.Div):
            a, b = self._widened(node.this, node.expression, lambda _, n: self._float_expr(n))
            return a / b
        if isinstance(node, sge.Mod):                    # `a % b` and MOD(a, b): DuckDB's fmod, NULL on zero
            l, r = self._mod_operand(node.this), self._mod_operand(node.expression)
            if l.op == "lit" and r.op == "lit" and all(type(v.args[0]) is int for v in (l, r)):
                return lit(None if r.args[0] == 0 else (abs(l.args[0]) % abs(r.args[0])) * (-1 if l.args[0] < 0 else 1))   # two constants
            dec = any(_is_decimal_literal(x) for x in (node.this, node.expression))
            return Expr("mod", l, r, True) if dec else l % r      # a literal with a point (no exponent) is DECIMAL

        # ---- comparisons ----
        if isinstance(node, _COMPARE):
            return self._compare(type(node), node.this, node.expression)

        # ---- boolean ----
        if isinstance(node, sge.And):
            return self._expr(node.this) & self._expr(node.expression)
        if isinstance(node, sge.Or):
            return self._expr(node.this) | self._expr(node.expression)
        if isinstance(node, sge.Not):
            return ~self._expr(node.this)

        # ---- IS [NOT] NULL / TRUE / FALSE, IS [NOT] DISTINCT FROM ----
        if isinstance(node, sge.Is):
            return self._is(node)
        if isinstance(node, (sge.NullSafeEQ, sge.NullSafeNEQ)):
            self._no_constants(node, node.this, node.expression)
            if isinstance(node.expression, sge.Null) or isinstance(node.this, sge.Null):
                x = self._expr(node.this if isinstance(node.expression, sge.Null) else node.expression)
                return x.is_null() if isinstance(node, sge.NullSafeEQ) else ~x.is_null()
            a, b = self._expr(node.this), self._expr(node.expression)
            an, bn = a.is_null(), b.is_null()
            unequal = a != b
            if not any(isinstance(o, sge.Literal) for o in (node.this, node.expression)):
                unequal = unequal & ~((a != a) & (b != b))          # two NaNs are not distinct (DuckDB)
            differ = (an & ~bn) | (~an & bn) | (~an & ~bn & unequal)
            return ~differ if isinstance(node, sge.NullSafeEQ) else differ

        # ---- BETWEEN ----
        if isinstance(node, sge.Between):
            return self._compare(sge.GTE, node.this, node.args["low"]) & self._compare(sge.LTE, node.this, node.args["high"])

        # ---- IN / NOT IN ----
        if isinstance(node, sge.In):
            c = self._expr(node.this)
            if self._ftype(node.this) == "f32" and any(_is_double_literal(v) for v in node.expressions):
                c = c * lit(np.float64(1.0))
            values = []
            for v in node.expressions:
                d = _exact_decimal(v) if self._is_int_expr(node.this) else None
                if d is None:
                    values.append(self._scalar(v))
                elif d == d.to_integral_value() and -(1 << 63) <= d < (1 << 63):     # exactly: a fraction or a value past int64
                    values.append(int(d))                                              # never equals an integer
            return c.isin(values) if values else _folded(c, False)

        # ---- LIKE / ILIKE [ESCAPE c] ----
        if isinstance(node, sge.Escape) and isinstance(node.this, (sge.Like, sge.ILike)):
            return self._like(node.this, node.expression)
        if isinstance(node, (sge.Like, sge.ILike)):
            return self._like(node, None)

        # ---- aggregates ----
        if isinstance(node, sge.Sum):
            return self._expr(node.this).sum()
        if isinstance(node, sge.Avg):
            return self._expr(node.this).mean()
        if isinstance(node, sge.Min):
            return self._expr(node.this).min()
        if isinstance(node, sge.Max):
            return self._expr(node.this).max()
        if isinstance(node, sge.Count):
            if isinstance(node.this, sge.Distinct):
                raise NotImplementedError("COUNT(DISTINCT x) is supported as a SELECT, HAVING or ORDER BY aggregate only")
            if isinstance(node.this, sge.Star):
                return lit(1).count()          # rows, not the non-null values of some column
            return self._expr(node.this).count()

        # ---- CASE WHEN ----
        if isinstance(node, sge.Case):
            return self._translate_case(node)

        raise NotImplementedError(
            f"Unsupported SQL expression type: {type(node).__name__}: {node}"
        )

    # ------------------------------------------------------------------
    # CASE WHEN helpers
    # ------------------------------------------------------------------

    def _col_type(self, src: sge.Expression):
        """The Arrow type of a plain column of this query's FROM items, through its alias; else None."""
        return _type_in(src, self._alias_schema)

    @staticmethod
    def _case_results(node: sge.Case) -> list:
        """The result expressions of a CASE: every THEN and the ELSE."""
        return [i.args["true"] for i in node.args.get("ifs") or []] + ([node.args["default"]] if node.args.get("default") else [])

    _INTERVAL_US = {"DAY": 86_400_000_000, "HOUR": 3_600_000_000, "MINUTE": 60_000_000, "SECOND": 1_000_000,
                    "MILLISECOND": 1_000, "MICROSECOND": 1}

    def _interval(self, node) -> Expr:
        """``x + INTERVAL n DAY | HOUR | MINUTE | SECOND ...`` (also ``-``) on a DATE or timestamp expression: a
        TIMESTAMP, as in DuckDB. MONTH and YEAR depend on the calendar and are only supported on a constant date."""
        iv = _unparen(node.expression)
        unit = iv.unit.name.upper() if iv.unit is not None else ""
        if unit not in self._INTERVAL_US or not isinstance(iv.this, sge.Literal):
            raise NotImplementedError(f"INTERVAL {iv.sql(dialect='duckdb')} on a column is not supported "
                                      f"({', '.join(self._INTERVAL_US)}); MONTH and YEAR only on a constant DATE")
        n = int(iv.this.this) * (1 if isinstance(node, sge.Add) else -1)
        self._no_constants(node, node.this)
        t = self._col_type(node.this)
        if t is not None and not (pa.types.is_date(t) or pa.types.is_timestamp(t)):      # (an int day number column: TPC-H files)
            raise NotImplementedError(f"INTERVAL can be added to a DATE or TIMESTAMP column, not to {t}: {node.sql(dialect='duckdb')}")
        return self._expr(node.this).add_us(n * self._INTERVAL_US[unit])

    def _is_date_col(self, src: sge.Expression) -> bool:
        if isinstance(src, (sge.Min, sge.Max)):             # F-093: MIN / MAX of a date column is a date
            src = src.this
        if isinstance(src, sge.Case):                       # a CASE with a date result is a date
            return any(_date_of(r) is not None or self._is_date_col(r) for r in self._case_results(src))
        if isinstance(src, (sge.Add, sge.Sub)) and isinstance(_unparen(src.expression), sge.Interval):
            return self._is_date_col(src.this)              # a DATE / timestamp column + INTERVAL is a timestamp
        if isinstance(src, sge.Coalesce):
            return any(_date_of(a) is not None or self._is_date_col(a) for a in [src.this] + list(src.expressions))
        if isinstance(src, sge.Nullif):
            return self._is_date_col(src.this)
        if _timestamp_of(src) is not None:
            return True
        t = self._col_type(src)
        return t is not None and (pa.types.is_date(t) or pa.types.is_timestamp(t))

    def _extract(self, node: sge.Extract) -> Expr:
        """EXTRACT(YEAR FROM x): a date32 column through ``year()``; any other (integer day numbers since
        1970-01-01, as the TPC-H Parquet files store dates) through ``year_of_days()``."""
        if node.this.name.upper() != "YEAR":
            raise NotImplementedError(f"EXTRACT({node.this.name.upper()} FROM ...) is not supported, only YEAR")
        arg = self._expr(node.expression)
        t = self._col_type(node.expression)
        if t is not None and (pa.types.is_date(t) or pa.types.is_timestamp(t)):
            return arg.year()
        return arg.year_of_days()

    def _substring(self, node: sge.Substring) -> Expr:
        start, length = node.args.get("start"), node.args.get("length")
        ok = isinstance(start, sge.Literal) and not start.is_string and int(start.this) >= 1 \
            and (length is None or (isinstance(length, sge.Literal) and not length.is_string))
        if not ok:
            raise NotImplementedError("SUBSTRING needs a constant start >= 1 and a constant length: "
                                      f"{node.sql(dialect='duckdb')}")
        return self._expr(node.this).str_slice(int(start.this) - 1, None if length is None else int(length.this))

    def _is(self, node: sge.Is) -> Expr:
        """``x IS NULL`` / ``IS TRUE`` / ``IS FALSE``: never NULL themselves (``IS NOT ...`` is a NOT above)."""
        what = node.expression
        self._no_constants(node, node.this)
        x = self._expr(node.this)
        if isinstance(what, sge.Null):
            return x.is_null()
        if isinstance(what, sge.Boolean):
            return (x if what.this else ~x) & ~x.is_null()     # a NULL x gives NULL & False = False
        raise NotImplementedError(f"IS {what.sql(dialect='duckdb')} is not supported: IS [NOT] NULL, TRUE, FALSE only")

    @staticmethod
    def _no_constants(node, *operands):
        if all(isinstance(o, (sge.Literal, sge.Null, sge.Boolean)) or _date_of(o) is not None for o in operands):
            raise NotImplementedError(f"{node.sql(dialect='duckdb')}: a test of constants only has no column to "
                                      "filter or compute on; put a column in it")

    def _value(self, n: sge.Expression) -> Expr:
        """A CASE result or COALESCE / NULLIF argument: NULL, TRUE / FALSE and DATE constants are allowed here
        (they are not elsewhere); the rest as usual."""
        d = _date_of(n)
        if d is not None:
            if isinstance(_unparen(n), (sge.Add, sge.Sub)):          # DATE +/- INTERVAL is a TIMESTAMP (microseconds)
                return lit(np.datetime64(d, "us"))
            return lit(d)
        if isinstance(n, (sge.Null, sge.Boolean)):
            return lit(None if isinstance(n, sge.Null) else bool(n.this))
        return self._expr(n)

    def _coalesce(self, node: sge.Coalesce) -> Expr:
        """``COALESCE(a, b, ...)`` / ``IFNULL``: the first non-NULL argument, of their common type."""
        args = [node.this] + list(node.expressions)
        self._no_constants(node, *args)
        return coalesce(*(self._value(a) for a in args))

    def _nullif(self, node: sge.Nullif) -> Expr:
        """``NULLIF(a, b)``: NULL where ``a = b``, else ``a`` (the type of ``a``)."""
        if isinstance(node.expression, sge.Null):
            return self._value(node.this)
        self._no_constants(node, node.this)
        same = self._expr(sge.EQ(this=node.this.copy(), expression=node.expression.copy()))    # as `a = b` is written
        return when(same, lit(None), self._value(node.this))

    _CAST_NAMES = {sge.DataType.Type.FLOAT: "float32", sge.DataType.Type.DOUBLE: "float64",
                   sge.DataType.Type.TINYINT: "int8", sge.DataType.Type.SMALLINT: "int16",
                   sge.DataType.Type.INT: "int32", sge.DataType.Type.BIGINT: "int64"}

    def _cast(self, node: sge.Cast) -> Expr:
        """``CAST(x AS FLOAT | DOUBLE | TINYINT | SMALLINT | INT | BIGINT)``; other targets are unsupported."""
        name = self._CAST_NAMES.get(node.to.this)
        if name is None:
            raise NotImplementedError(f"Unsupported SQL expression type: Cast: {node}")
        self._no_constants(node, node.this)
        t = self._col_type(node.this)
        if t is not None and not (pa.types.is_integer(t) or pa.types.is_floating(t)):
            raise NotImplementedError(f"CAST of a {t} column to {node.to.sql(dialect='duckdb')} is not supported (numbers only)")
        return self._expr(node.this).cast(name, decimal=name.startswith("int") and self._is_decimal_expr(node.this))

    def _is_decimal_expr(self, node: sge.Expression) -> bool:
        """True when DuckDB types ``node`` as DECIMAL: a DECIMAL literal (a point, no exponent) or + - * of one with
        integers or other DECIMALs (`i * 0.5`); a float or double operand makes it a float."""
        node = _unparen(node)
        if _is_decimal_literal(node):
            return True
        if isinstance(node, sge.Neg):
            return self._is_decimal_expr(node.this)
        if isinstance(node, (sge.Add, sge.Sub, sge.Mul)):
            a, b = node.this, node.expression
            da, db = self._is_decimal_expr(a), self._is_decimal_expr(b)
            return (da and (db or self._is_int_expr(b))) or (db and self._is_int_expr(a))
        return False

    def _translate_case(self, node: sge.Case) -> Expr:
        ifs = node.args.get("ifs") or []
        default_node = node.args.get("default")
        if not ifs:
            raise ValueError("CASE requires at least one WHEN clause")
        branch = self._value
        otherwise: Expr = branch(default_node) if default_node else lit(None)         # no ELSE is ELSE NULL
        # Build right-to-left so the first WHEN is outermost
        for if_node in reversed(ifs):
            cond = (self._expr(node.this) == self._expr(if_node.this)) if node.this is not None \
                else self._expr(if_node.this)               # CASE x WHEN v: x = v
            then = branch(if_node.args["true"])
            otherwise = when(cond, then, otherwise)
        return otherwise

    # ------------------------------------------------------------------
    # helper: extract Python scalar from a sqlglot Literal
    # ------------------------------------------------------------------

    def _scalar(self, node: sge.Expression) -> Any:
        if isinstance(node, sge.Null):
            return None                                 # IN (.., NULL): np_arrow.is_in's three-valued logic
        if isinstance(node, sge.Literal):
            if node.is_string:
                return node.this
            return _number(node.this)
        if isinstance(node, sge.Neg):
            n = _unparen(node.this)
            if isinstance(n, sge.Literal) and not n.is_string:
                return _number(n.this, neg=True)
            return -self._scalar(node.this)
        f = _float_cast(node)
        if f is not None:
            return f
        t = _timestamp_of(node)
        if t is not None:
            return t
        raise ValueError(f"Expected scalar literal in IN list, got {type(node).__name__}")

    def _ftype(self, node: sge.Expression) -> Optional[str]:
        """"f64" / "f32" when ``node`` is certainly a DOUBLE / FLOAT expression, as DuckDB types it, else None:
        a DOUBLE (exponent) literal or float64 column; a float32 column; and + - * / and negation, where any
        DOUBLE operand makes DOUBLE, else any FLOAT operand FLOAT (an integer or DECIMAL operand is cast to it)."""
        node = _unparen(node)
        if isinstance(node, sge.Literal):
            return "f64" if _is_double_literal(node) else None
        if isinstance(node, sge.Neg):
            return self._ftype(node.this)
        if isinstance(node, (sge.Add, sge.Sub, sge.Mul, sge.Div)):
            kinds = {self._ftype(node.this), self._ftype(node.expression)}
            return "f64" if "f64" in kinds else "f32" if "f32" in kinds else None
        if isinstance(node, (sge.Abs, sge.Nullif)):
            return self._ftype(node.this)
        if isinstance(node, sge.Coalesce):                  # DOUBLE if any argument is, else FLOAT if any is
            kinds = {self._ftype(a) for a in [node.this] + list(node.expressions)}
            return "f64" if "f64" in kinds else "f32" if "f32" in kinds else None
        if isinstance(node, sge.Cast):
            return {"float32": "f32", "float64": "f64"}.get(self._CAST_NAMES.get(node.to.this))
        t = self._col_type(node)
        return "f32" if t is not None and pa.types.is_float32(t) else "f64" if t is not None and pa.types.is_float64(t) else None

    def _widened(self, a: sge.Expression, b: sge.Expression, translate=None, parent=None):
        """The two operands of a comparison or + - * /. A FLOAT expression against a DOUBLE one (`1e-30`, `1e-30 * 1`,
        a float64 column) is widened to double, as DuckDB does (a DECIMAL / INTEGER operand is cast to float32
        instead): `g * np.float64(1.0)` is the expression in double on every path (the bytecode computes in
        double; the Arrow path widens a float32 array next to an np.float64 literal, see ``_arith_operands``),
        so the DOUBLE operand is never narrowed."""
        tr = (lambda n: self._expr(n)) if translate is None else (lambda n: translate(parent, n))
        x, y = tr(a), tr(b)
        kinds = (self._ftype(a), self._ftype(b))
        if kinds == ("f32", "f64"):
            x = x * lit(np.float64(1.0))
        elif kinds == ("f64", "f32"):
            y = y * lit(np.float64(1.0))
        return x, y

    def _compare(self, cls, a: sge.Expression, b: sge.Expression) -> Expr:
        """``a <op> b``. Against an integer column or expression, an integer literal past int64 or a DECIMAL
        literal (a point, no exponent) is compared exactly, as DuckDB does: the DECIMAL becomes the integer bound
        with the same answer (`< 2.5` is `< 3`, `> 2.5` is `> 2`, `= 2.5` never holds), and a bound past int64 is
        a constant (above every value `<` `<=` `<>` hold, below every value `>` `>=`); NULL stays NULL. A DOUBLE
        (exponent) literal compares in double instead, so it is left to the engine."""
        kind = {sge.LT: "lt", sge.LTE: "le", sge.GT: "gt", sge.GTE: "ge", sge.EQ: "eq", sge.NEQ: "ne"}[cls]
        for lit_node, x_node, flip in ((b, a, False), (a, b, True)):
            d = _exact_decimal(lit_node)
            if d is None or not self._is_int_expr(x_node):
                continue
            if flip:
                kind = {"lt": "gt", "le": "ge", "gt": "lt", "ge": "le"}.get(kind, kind)
            x = self._expr(x_node)
            whole = d == d.to_integral_value()
            if kind in ("eq", "ne") and not whole:
                return _folded(x, kind == "ne")
            n = int(d.to_integral_value(ROUND_CEILING if kind in ("lt", "ge") else ROUND_FLOOR)) if not whole else int(d)
            if -(1 << 63) <= n < (1 << 63):
                return {"lt": x < lit(n), "le": x <= lit(n), "gt": x > lit(n), "ge": x >= lit(n), "eq": x == lit(n),
                        "ne": x != lit(n)}[kind]
            keep = kind == "ne" or (kind in ("lt", "le")) == (n > 0) and kind != "eq"
            return _folded(x, keep)
        x, y = self._widened(a, b)
        return {sge.EQ: x == y, sge.NEQ: x != y, sge.LT: x < y, sge.LTE: x <= y, sge.GT: x > y, sge.GTE: x >= y}[cls]

    def _is_int_expr(self, node: sge.Expression) -> bool:
        """True when ``node`` is certainly an integer: an integer literal, an integer column of a known table,
        or + - * % and negation of those, COUNT, or an integer MIN / MAX. Unknown (unresolved names) is False."""
        node = _unparen(node)
        if isinstance(node, sge.Literal):
            return not node.is_string and "." not in str(node.this) and "e" not in str(node.this).lower()
        if isinstance(node, sge.Neg):
            return self._is_int_expr(node.this)
        if isinstance(node, (sge.Add, sge.Sub, sge.Mul, sge.Mod)):
            return self._is_int_expr(node.this) and self._is_int_expr(node.expression)
        if isinstance(node, sge.Count):
            return True
        if isinstance(node, (sge.Min, sge.Max)):               # not SUM: the engine returns it as a double until F-060
            return self._is_int_expr(node.this)
        if isinstance(node, (sge.Abs, sge.Nullif)):
            return self._is_int_expr(node.this)
        if isinstance(node, sge.Coalesce):
            return all(self._is_int_expr(a) for a in [node.this] + list(node.expressions) if not isinstance(a, sge.Null))
        if isinstance(node, sge.Case):                          # integer results (a NULL branch takes the other's type)
            res = [r for r in self._case_results(node) if not isinstance(r, sge.Null)]
            return bool(res) and all(self._is_int_expr(r) for r in res)
        t = self._col_type(node)
        return t is not None and pa.types.is_integer(t)

    def _arith_operand(self, parent: sge.Expression, node: sge.Expression) -> "Expr":
        """An operand of + - *: integer arithmetic stays integer (exact, checked overflow, like DuckDB) when both
        operands are certainly integers; anything else promotes an integer literal to float (``_float_expr``)."""
        if self._is_int_expr(parent.this) and self._is_int_expr(parent.expression):
            return self._expr(node)
        return self._float_expr(node)

    def _mod_operand(self, node: sge.Expression) -> "Expr":
        """An operand of ``%``: a numeric literal (negated or not) stays a plain constant, an integer one an integer, anything else is evaluated
        as it is, so ``np_arrow.modulo`` picks integer or float arithmetic from the real arrays."""
        n = _unparen(node)
        neg = isinstance(n, sge.Neg)
        n = _unparen(n.this) if neg else n
        if isinstance(n, sge.Literal) and not n.is_string:
            return lit(_number(n.this, neg))
        return self._expr(node)

    def _float_expr(self, node: sge.Expression) -> "Expr":
        """Like _expr but promotes integer literals to float for arithmetic use.

        In SQL, ``1 - l_discount`` is floating-point arithmetic even though 1
        is written without a decimal point.  Calling this instead of _expr for
        operands of Add/Sub/Mul/Div avoids int64 vs float32 dtype mismatches in
        the MAX graph.
        """
        if isinstance(node, sge.Literal) and node.meta.get("keep_int"):
            return self._expr(node)
        if isinstance(node, sge.Literal) and not node.is_string:
            return lit(float(node.this))
        if isinstance(node, sge.Neg) and isinstance(node.this, sge.Literal) and not node.this.is_string:
            return lit(-float(node.this.this))
        return self._expr(node)
