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
"""

__all__ = ["sql"]

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
    return [n for n in e.walk(prune=lambda n: isinstance(n, (sge.AggFunc, sge.Window)))
            if isinstance(n, sge.AggFunc)]


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
        # 1. FROM + JOINs
        plan = self._build_from(node)

        # 2. WHERE
        where = node.args.get("where")
        if where:
            if _aggs_in(where.this):
                raise ValueError("aggregate functions are not allowed in WHERE (use HAVING)")
            plan = Filter(plan, self._expr(where.this))

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

    def _scan(self, table_node: sge.Expression) -> LogicalPlan:
        if not isinstance(table_node, sge.Table):
            raise ValueError(f"Unsupported FROM target: {type(table_node).__name__}")
        name = table_node.name.lower()
        if name not in self._tables:
            avail = list(self._tables.keys())
            raise ValueError(f"Unknown table {name!r}. Available: {avail}")
        tbl = self._tables[name]
        self._scope_columns.extend(tbl.schema.names)
        alias = table_node.alias_or_name.lower()
        self._table_cols.append((alias, {c.lower() for c in tbl.schema.names}))
        return Scan(tbl), tbl.schema.names

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
        if node.find(sge.AggFunc) is not None:
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
        if self._post:
            ref = self._post_ref(node)
            if ref is not None:
                return ref

        # ---- column reference ----
        if isinstance(node, sge.Column):
            name = node.name.lower()
            if node.table and name not in self._same_keys and \
                    sum(name in cols for _, cols in self._table_cols) > 1:
                raise ValueError(f"column {node.sql(dialect='duckdb')}: the name exists in more than one joined "
                                 "table and the join keeps one copy, so this is not supported yet")
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
                # COUNT(*) — count any in-scope column
                c_name = self._scope_columns[0] if self._scope_columns else None
                if c_name is None:
                    raise ValueError("COUNT(*) with no tables in scope")
                return col(c_name).count()
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
