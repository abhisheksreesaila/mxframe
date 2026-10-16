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

import calendar
import itertools
from datetime import date, timedelta
from decimal import Decimal

import pyarrow as pa
import sqlglot
import sqlglot.expressions as sge
from typing import Any, Dict, List, Optional

from .lazy_expr import Expr, col, lit, when
from .lazy_frame import (
    LazyFrame, LogicalPlan,
    Scan, Filter, Project, Aggregate, Sort, Limit, Distinct, Join, join_output_names,
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
    """FROM a, b: the join keys are in WHERE. (An explicit CROSS JOIN stays unsupported.)"""
    return (not join.args.get("on") and not join.args.get("using") and not join.args.get("side")
            and not join.args.get("kind"))


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


_ALL_ROWS = 2 ** 62          # LIMIT of an OFFSET without LIMIT


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
    fn = _AGG_NAMES[type(node)]
    arg = node.this
    distinct = isinstance(arg, sge.Distinct)
    if distinct:
        arg = arg.expressions[0]
    if isinstance(arg, sge.Star):
        return "count_star()"
    if isinstance(arg, (sge.Column, sge.Literal)):
        return f"{fn}({'DISTINCT ' if distinct else ''}{arg.sql(dialect='duckdb')})"
    raise ValueError(f"give {node.sql(dialect='duckdb')} an alias (AS name): the output column has no name")


def _dedupe(names) -> List[Optional[str]]:
    """Output names as DuckDB returns them: a name already taken becomes ``x_1``, ``x_2`` ...
    (the first free one, in SELECT order). Unnamed (None) stay unnamed."""
    seen, out = set(), []
    for n in names:
        if n is not None and n in seen:
            i = 1
            while f"{n}_{i}" in seen:
                i += 1
            n = f"{n}_{i}"
        if n is not None:
            seen.add(n)
        out.append(n)
    return out


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
        # right join keys of a LEFT JOIN: the join drops them, and they are NULL for unmatched rows,
        # so a left key cannot stand in for them
        self._dropped_right: dict = {}
        # (table name or alias, its columns) of every scanned table, and join keys that have the
        # same name on both sides (one copy survives the join, so t.k and u.k are the same column)
        self._table_cols: List[tuple] = []
        self._same_keys: set = set()
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

    def _child(self) -> "_SQLTranslator":
        """A translator for a subquery: its own state, the same tables and WITH names."""
        c = _SQLTranslator(self._tables)
        c._ctes = dict(self._ctes)
        c._ids = self._ids
        return c

    # ------------------------------------------------------------------
    # top-level dispatch
    # ------------------------------------------------------------------

    def translate(self, query: str) -> LazyFrame:
        tree = sqlglot.parse_one(query, dialect="duckdb")
        if not isinstance(tree, sge.Select):
            raise ValueError(f"Only SELECT statements are supported, got {type(tree).__name__}")
        plan = self._build_select(tree)
        return LazyFrame(plan)

    def _build_select(self, node: sge.Select) -> LogicalPlan:
        if not isinstance(node, sge.Select):
            raise NotImplementedError(f"{type(node).__name__} (UNION / INTERSECT / EXCEPT) is not supported, only SELECT")
        self._register_ctes(node.args.get("with_"))
        self._has_star = any(isinstance(e, sge.Star) for e in node.expressions)

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
            if non_star:
                proj = [self._select_expr(e) for e in non_star]
                names = _dedupe([p._alias or (p.args[0] if p.op == "col" else None) for p in proj])
                for p, nm in zip(proj, names):
                    if nm is not None and nm != (p._alias or (p.args[0] if p.op == "col" else None)):
                        p.alias(nm)
                # ORDER BY a name two output columns share takes the aliased one (as DuckDB does)
                alias_out = {e.alias.lower(): nm for e, nm in zip(non_star, names)
                             if isinstance(e, sge.Alias) and e.alias.lower() != nm}
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
                    elif i is None and isinstance(ref, sge.Column) and ref.name.lower() in (names or []):
                        by.append(col(ref.name.lower()))          # a SELECT output name
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
            plan = Limit(plan, int(limit_node.expression.this) if limit_node else _ALL_ROWS,
                         int(offset_node.expression.this) if offset_node else 0)

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
                cols, table = self._cur_cols, None
            else:
                table = e.table.lower()
                if table not in {a for a, _ in self._table_cols}:
                    raise ValueError(f"{e.sql(dialect='duckdb')}: unknown table {table!r} in the FROM clause")
                if len(self._table_cols) == 1:
                    cols = self._cur_cols
                elif self._alias_schema.get(table) is not None:
                    cols = self._alias_schema[table].names
                else:
                    raise NotImplementedError(f"{table}.*: the columns of this FROM item are not known, list them")
            out += [sge.Column(this=sge.to_identifier(c), table=sge.to_identifier(table) if table else None)
                    for c in cols]
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
            if isinstance(ref, sge.Column) and not ref.table and ref.name.lower() in names:
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
        self._qual = {(next(iter(self._joined)), c.lower()): c for c in main_cols}
        pending = []

        for join in node.args.get("joins") or []:
            right_plan, right_cols = self._scan(join.this)
            right_schema = set(right_cols)
            if _is_comma_join(join):
                pending.append((join.this.alias_or_name.lower(), right_plan, right_cols))
                continue

            how = _join_type(join)
            on = join.args.get("on")
            using = join.args.get("using")
            if on is not None:
                left_on, right_on = self._parse_join_on(on, self._left_schema, right_schema)
            elif using:
                cols = using if isinstance(using, list) else using.expressions      # sqlglot versions differ
                keys = [str(c.name) for c in cols]
                left_on = right_on = keys
            else:
                raise ValueError("JOIN requires ON or USING clause")

            plan = Join(plan, right_plan, left_on, right_on, how)
            # After this join, the accumulated left side grows to include right cols (a semi / anti
            # join outputs the left columns only)
            if how not in ("semi", "anti"):
                self._left_schema |= right_schema
                self._cur_cols += [o for _, o in join_output_names(self._cur_cols, right_cols, right_on)]
            self._joined.add(join.this.alias_or_name.lower())
            # Record that right join keys should be referred to by left join keys
            for lk, rk in zip(left_on, right_on):
                if lk == rk:
                    self._same_keys.add(lk)
                if lk != rk:
                    (self._dropped_right if how == "left" else self._col_aliases)[rk] = lk

        while pending:
            for i, (alias, right_plan, right_cols) in enumerate(pending):
                links = self._links(alias, {c.lower() for c in right_cols})
                if links:
                    break
            else:
                names = ", ".join(a for a, _, _ in pending)
                raise NotImplementedError(f"no join condition links {names} to the other tables: FROM a, b needs "
                                          "column = column conditions in WHERE (a cross product is not supported)")
            pending.pop(i)
            left_on = [self._expr(l).args[0] for l, _ in links]
            right_on = [r.name.lower() for _, r in links]
            plan = Join(plan, right_plan, left_on, right_on, "inner")
            outs = join_output_names(self._cur_cols, right_cols, right_on)
            for c, out in outs:
                self._qual[(alias, c.lower())] = out
            for lk, rk in zip(left_on, right_on):
                self._qual[(alias, rk)] = lk
                if lk == rk:
                    self._same_keys.add(lk)
                else:
                    self._col_aliases[rk] = lk
            self._cur_cols += [o for _, o in outs]
            self._left_schema |= set(right_cols)
            self._joined.add(alias)

        return plan

    def _links(self, alias: str, cols: set):
        """[(column of the tables joined so far, column of table ``alias``)] from the WHERE conditions
        that equate the two; they are taken out of ``self._conj``. A condition every branch of an OR has
        counts (it is removed from the branches)."""
        def owner(c: sge.Column) -> Optional[str]:
            name = c.name.lower()
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

    def _copy_dropped(self, qualifier: str, name: str) -> bool:
        """True when ``qualifier.name`` is the copy of a column the join dropped (an earlier joined
        table has a column of that name; the join keeps the first one)."""
        for i, (alias, _) in enumerate(self._table_cols):
            if alias == qualifier:
                return any(name in cols for _, cols in self._table_cols[:i])
        return False

    def _source(self, table_node: sge.Expression):
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
        self._table_cols.append((alias, {c.lower() for c in cols}))
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

    def _parse_join_on(self, on_expr: sge.Expression,
                       left_schema: set = None, right_schema: set = None):
        """Return ([left_cols], [right_cols]) from an ON expression.

        When *left_schema* and *right_schema* are provided we detect which
        column belongs to which side and assign accordingly, handling cases
        where the SQL author wrote them in right→left order.
        """
        if isinstance(on_expr, sge.EQ):
            a = on_expr.this.name.lower()
            b = on_expr.expression.name.lower()
            if left_schema is not None and right_schema is not None:
                a_in_left  = a in left_schema
                a_in_right = a in right_schema
                b_in_left  = b in left_schema
                b_in_right = b in right_schema
                # If a belongs to right and b belongs to left → swap
                if a_in_right and b_in_left and not a_in_left:
                    return [b], [a]
            return [a], [b]
        if isinstance(on_expr, sge.And):
            ll, rl = self._parse_join_on(on_expr.this, left_schema, right_schema)
            lr, rr = self._parse_join_on(on_expr.expression, left_schema, right_schema)
            return ll + lr, rl + rr
        raise ValueError(f"Cannot parse JOIN ON: {on_expr}")

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
                        and self._outer_refs(side.this, self._scope_of(side.this)):
                    return "scalar", c, False
        return None

    def _exists_form(self, plan, node: sge.Exists, negated: bool):
        what = "NOT EXISTS" if negated else "EXISTS"
        sel = node.this
        if not isinstance(sel, sge.Select):
            raise NotImplementedError(f"{what} over a {type(sel).__name__} is not supported")
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
            src = None if node.table else self._out_src.get(node.name.lower())
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
            return (e.this, e.alias.lower()) if isinstance(e, sge.Alias) else (e, None)

        # Group-by keys. A SELECT alias names its key: GROUP BY may name the alias (when no input
        # column has that name, as in DuckDB) or repeat the aliased expression (compared after
        # translation, so t.s and s are the same key). GROUP BY ALL groups by every SELECT item
        # without an aggregate.
        plain = [split(e) for e in items if not _aggs_in(e)]
        aliased = [(a, self._expr(i).signature()) for i, a in plain if a]
        alias_expr = {a: i for i, a in plain if a}
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
            outs[n] = (alias or inner.name.lower(), src)
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
                    elif isinstance(ref, sge.Column) and not ref.table and ref.name.lower() in self._out_src:
                        by.append(col(self._out_src[ref.name.lower()]))   # a SELECT output name
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
            alias_name = e.alias.lower()
            inner = e.this
        else:
            inner = e
        mx = self._expr(inner)
        if not alias_name and isinstance(inner, sge.Column) and mx.op == "col":
            alias_name = inner.name.lower() if inner.name.lower() != mx.args[0] else None   # joined key: as written
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
                return lit(int(v)) if is_int else lit(float(v))
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
            name = node.name.lower()
            if node.table and (node.table.lower(), name) in self._qual:
                return col(self._qual[(node.table.lower(), name)])        # a comma-joined table's own copy
            if node.table and name not in self._same_keys and self._copy_dropped(node.table.lower(), name):
                raise ValueError(f"column {node.sql(dialect='duckdb')}: the name exists in more than one joined "
                                 "table and the join keeps the first copy, so this one is not supported yet")
            if name in self._dropped_right:
                raise ValueError(f"column {name!r} is the right key of a LEFT JOIN, which the join drops "
                                 f"(it is NULL where nothing matched); use the left key {self._dropped_right[name]!r}")
            # Remap right join keys to their left-side equivalents (hash join
            # drops the right join key and keeps the left one in the output)
            name = self._col_aliases.get(name, name)
            return col(name)

        # ---- literals ----
        if isinstance(node, sge.Literal):
            if node.is_string:
                return lit(node.this)
            v = node.this
            return lit(int(v)) if "." not in str(v) else lit(float(v))
        f = _float_cast(node)
        if f is not None:
            return lit(f)

        # ---- parentheses ----
        if isinstance(node, sge.Paren):
            return self._expr(node.this)

        # ---- negation ----
        if isinstance(node, sge.Neg):
            return lit(0) - self._expr(node.this)

        # ---- arithmetic ----
        if isinstance(node, sge.Add):
            return self._float_expr(node.this) + self._float_expr(node.expression)
        if isinstance(node, sge.Sub):
            return self._float_expr(node.this) - self._float_expr(node.expression)
        if isinstance(node, sge.Mul):
            return self._float_expr(node.this) * self._float_expr(node.expression)
        if isinstance(node, sge.Div):
            return self._float_expr(node.this) / self._float_expr(node.expression)
        if isinstance(node, sge.Mod):                    # `a % b` and MOD(a, b): DuckDB's fmod, NULL on zero
            l, r = self._mod_operand(node.this), self._mod_operand(node.expression)
            if l.op == "lit" and r.op == "lit" and all(type(v.args[0]) is int for v in (l, r)):
                return lit(None if r.args[0] == 0 else (abs(l.args[0]) % abs(r.args[0])) * (-1 if l.args[0] < 0 else 1))   # two constants
            return l % r

        # ---- comparisons ----
        if isinstance(node, sge.EQ):
            return self._expr(node.this) == self._expr(node.expression)
        if isinstance(node, sge.NEQ):
            return self._expr(node.this) != self._expr(node.expression)
        if isinstance(node, sge.LT):
            return self._expr(node.this) < self._expr(node.expression)
        if isinstance(node, sge.LTE):
            return self._expr(node.this) <= self._expr(node.expression)
        if isinstance(node, sge.GT):
            return self._expr(node.this) > self._expr(node.expression)
        if isinstance(node, sge.GTE):
            return self._expr(node.this) >= self._expr(node.expression)

        # ---- boolean ----
        if isinstance(node, sge.And):
            return self._expr(node.this) & self._expr(node.expression)
        if isinstance(node, sge.Or):
            return self._expr(node.this) | self._expr(node.expression)
        if isinstance(node, sge.Not):
            return ~self._expr(node.this)

        # ---- BETWEEN ----
        if isinstance(node, sge.Between):
            c = self._expr(node.this)
            lo = self._expr(node.args["low"])
            hi = self._expr(node.args["high"])
            return (c >= lo) & (c <= hi)

        # ---- IN / NOT IN ----
        if isinstance(node, sge.In):
            c = self._expr(node.this)
            values = [self._scalar(v) for v in node.expressions]
            result = c.isin(values)
            return result

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

    def _is_date_col(self, src: sge.Expression) -> bool:
        t = self._col_type(src)
        return t is not None and pa.types.is_date(t)

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

    def _translate_case(self, node: sge.Case) -> Expr:
        ifs = node.args.get("ifs") or []
        default_node = node.args.get("default")
        if not ifs:
            raise ValueError("CASE requires at least one WHEN clause")
        otherwise: Expr = self._expr(default_node) if default_node else lit(0)
        # Build right-to-left so the first WHEN is outermost
        for if_node in reversed(ifs):
            cond = self._expr(if_node.this)
            then = self._expr(if_node.args["true"])
            otherwise = when(cond, then, otherwise)
        return otherwise

    # ------------------------------------------------------------------
    # helper: extract Python scalar from a sqlglot Literal
    # ------------------------------------------------------------------

    def _scalar(self, node: sge.Expression) -> Any:
        if isinstance(node, sge.Literal):
            if node.is_string:
                return node.this
            v = node.this
            return int(v) if "." not in str(v) else float(v)
        if isinstance(node, sge.Neg):
            return -self._scalar(node.this)
        f = _float_cast(node)
        if f is not None:
            return f
        raise ValueError(f"Expected scalar literal in IN list, got {type(node).__name__}")

    def _mod_operand(self, node: sge.Expression) -> "Expr":
        """An operand of ``%``: a numeric literal (negated or not) stays a plain constant, an integer one an integer, anything else is evaluated
        as it is, so ``np_arrow.modulo`` picks integer or float arithmetic from the real arrays."""
        n = _unparen(node)
        neg = isinstance(n, sge.Neg)
        n = _unparen(n.this) if neg else n
        if isinstance(n, sge.Literal) and not n.is_string:
            v = int(n.this) if str(n.this).isdigit() else float(n.this)
            return lit(-v if neg else v)
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
