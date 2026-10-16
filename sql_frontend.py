"""SQL frontend for MXFrame: SQL string → LogicalPlan via sqlglot.

Supports a large subset of TPC-H-style SQL:
  SELECT / FROM / WHERE / GROUP BY / ORDER BY / LIMIT
  Arithmetic: + - * /
  Comparisons: = != < <= > >=
  Boolean: AND OR NOT
  BETWEEN, IN / NOT IN, LIKE (prefix% only)
  Aggregates: SUM AVG MIN MAX COUNT(*) COUNT(DISTINCT x), arithmetic on aggregates, HAVING, GROUP BY ALL
  CASE WHEN ... THEN ... ELSE ... END
  INNER / LEFT JOIN ... ON ...
  WITH name AS (...), derived tables (FROM (SELECT ...) AS d)
  Subqueries (F-065): a scalar subquery in WHERE / HAVING / SELECT expressions; x [NOT] IN (SELECT ...) and
  [NOT] EXISTS (SELECT ...) as top-level AND conditions of WHERE (semi / anti joins); a correlated scalar
  comparison in WHERE (join + aggregate). Other forms raise NotImplementedError naming the form.
"""

__all__ = ["sql"]

import itertools

import pyarrow as pa
import sqlglot
import sqlglot.expressions as sge
from typing import Any, Dict, List, Optional

from .lazy_expr import Expr, col, lit, when
from .lazy_frame import (
    LazyFrame, LogicalPlan,
    Scan, Filter, Project, Aggregate, Sort, Limit, Distinct, Join,
)


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def sql(query: str, **tables: pa.Table) -> LazyFrame:
    """Run a SQL query against the provided Arrow tables.

    Args:
        query:   SQL string (DuckDB dialect).
        **tables: named ``pa.Table`` objects referenced in the query.

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

        # 1. FROM + JOINs
        plan = self._build_from(node)

        # 2. WHERE
        where = node.args.get("where")
        if where:
            if _aggs_in(where.this):
                raise ValueError("aggregate functions are not allowed in WHERE (use HAVING)")
            plan = self._where(plan, where.this)

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
                if len(non_star) < len(node.expressions):
                    names = None                       # a * shifts the positions
                plan = Project(plan, proj)
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
                    if i is None and isinstance(ref, sge.Column) and ref.name.lower() in (names or []):
                        by.append(col(ref.name.lower()))          # a SELECT output name
                    elif i is None:
                        by.append(self._expr(o.this))
                    elif names is None or names[i] is None:
                        raise ValueError("ORDER BY position needs a named SELECT column (give it an alias)")
                    else:
                        by.append(col(names[i]))
                desc = [bool(o.args.get("desc")) for o in order_node.expressions]
            plan = Sort(plan, by, desc)

        # 5. LIMIT
        limit_node = node.args.get("limit")
        if limit_node:
            plan = Limit(plan, int(limit_node.expression.this))

        if final_proj:
            plan = Project(plan, final_proj)
        return plan

    # ------------------------------------------------------------------
    # FROM / JOIN
    # ------------------------------------------------------------------

    def _build_from(self, node: sge.Select) -> LogicalPlan:
        from_clause = node.args.get("from_")
        if from_clause is None:
            raise ValueError("No FROM clause found")

        plan, main_cols = self._scan(from_clause.this)
        # Seed the accumulated left-side schema with the main FROM table
        self._left_schema = set(main_cols)

        for join in node.args.get("joins") or []:
            right_plan, right_cols = self._scan(join.this)
            right_schema = set(right_cols)

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
            # Record that right join keys should be referred to by left join keys
            for lk, rk in zip(left_on, right_on):
                if lk == rk:
                    self._same_keys.add(lk)
                if lk != rk:
                    (self._dropped_right if how == "left" else self._col_aliases)[rk] = lk

        return plan

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
        return Scan(tbl), tbl.schema.names

    def _scan(self, table_node: sge.Expression) -> LogicalPlan:
        plan, cols = self._source(table_node)
        self._scope_columns.extend(cols)
        alias = table_node.alias_or_name.lower()
        self._table_cols.append((alias, {c.lower() for c in cols}))
        return plan, cols

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
        if keys:
            if negated:
                raise NotImplementedError("a correlated NOT IN (SELECT ...) is not supported (its NULL rules differ "
                                          "per outer row); use NOT EXISTS")
            _unsupported_shape(sel, what)
            items = [_alias(item, "__in")] + [_alias(inner, f"__k{i}_{j}") for j, (_, inner) in enumerate(keys)]
            inner_plan = self._child()._build_select(self._keyed_select(sel, rest, items))
            return Join(plan, inner_plan, [left] + [self._expr(o).args[0] for o, _ in keys],
                        ["__in"] + [f"__k{i}_{j}" for j in range(len(keys))], "semi")
        new = sel.copy()
        new.set("expressions", [_alias(item, "__in")])
        new.set("order", None)
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

    def _scalar_form(self, plan, c: sge.Expression, _neg: bool):
        """``expr <op> (SELECT agg ... WHERE inner.k = outer.k)``: the aggregate per key, joined on the key."""
        sub = next(_unparen(s) for s in (c.this, c.expression)
                   if isinstance(_unparen(s), sge.Subquery) and isinstance(_unparen(s).this, sge.Select)
                   and self._outer_refs(_unparen(s).this, self._scope_of(_unparen(s).this)))
        sel = sub.this
        what = "correlated scalar subquery"
        self._check_scalar(sel, what)
        if any(isinstance(a, sge.Count) for a in sel.walk(prune=lambda n: isinstance(n, (sge.Subquery, sge.Exists)))):
            raise NotImplementedError(f"{what} with COUNT is not supported: an outer row without partners must see 0, "
                                      "which a join would drop")
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
        if sel.args.get("group") or sel.args.get("having") or sel.args.get("limit") or sel.args.get("distinct") \
                or not _aggs_in(sel.expressions[0]):
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

    def _expr(self, node: sge.Expression) -> Expr:
        """Translate a sqlglot expression to an MXFrame Expr."""
        if isinstance(node, sge.Subquery):
            return self._scalar_subquery(node)
        if isinstance(node, sge.Exists) or (isinstance(node, sge.In) and node.args.get("query") is not None):
            what = "EXISTS (SELECT ...)" if isinstance(node, sge.Exists) else "IN (SELECT ...)"
            raise NotImplementedError(f"{what} is supported only as a condition joined by AND at the top of WHERE "
                                      "(it becomes a semi / anti join); not inside OR, NOT (...), CASE, SELECT or HAVING")
        if self._post:
            ref = self._post_ref(node)
            if ref is not None:
                return ref

        # ---- column reference ----
        if isinstance(node, sge.Column):
            name = node.name.lower()
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

        # ---- LIKE (prefix% patterns only) ----
        if isinstance(node, sge.Like):
            c = self._expr(node.this)
            pattern = node.expression.this
            if pattern.endswith("%") and "%" not in pattern[:-1] and "_" not in pattern:
                return c.startswith(pattern[:-1])
            raise NotImplementedError(
                f"Only 'prefix%' LIKE patterns are supported, got {pattern!r}"
            )

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
        raise ValueError(f"Expected scalar literal in IN list, got {type(node).__name__}")

    def _float_expr(self, node: sge.Expression) -> "Expr":
        """Like _expr but promotes integer literals to float for arithmetic use.

        In SQL, ``1 - l_discount`` is floating-point arithmetic even though 1
        is written without a decimal point.  Calling this instead of _expr for
        operands of Add/Sub/Mul/Div avoids int64 vs float32 dtype mismatches in
        the MAX graph.
        """
        if isinstance(node, sge.Literal) and not node.is_string:
            return lit(float(node.this))
        if isinstance(node, sge.Neg) and isinstance(node.this, sge.Literal) and not node.this.is_string:
            return lit(-float(node.this.this))
        return self._expr(node)
