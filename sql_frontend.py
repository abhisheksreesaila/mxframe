"""SQL frontend for MXFrame: SQL string → LogicalPlan via sqlglot.

Supports a large subset of TPC-H-style SQL:
  SELECT / FROM / WHERE / GROUP BY / ORDER BY / LIMIT
  Arithmetic: + - * /
  Comparisons: = != < <= > >=
  Boolean: AND OR NOT
  BETWEEN, IN / NOT IN, LIKE (prefix% only)
  Aggregates: SUM AVG MIN MAX COUNT(*)
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
            plan = Filter(plan, self._expr(where.this))

        # 3. GROUP BY / aggregates
        group_node = node.args.get("group")
        has_aggs = any(self._contains_agg(e) for e in node.expressions)

        if group_node or has_aggs:
            plan = self._build_aggregate(plan, node, group_node)
        else:
            # pure projection (skip if SELECT *)
            non_star = [e for e in node.expressions if not isinstance(e, sge.Star)]
            if non_star:
                proj = [self._select_expr(e) for e in non_star if self._select_expr(e) is not None]
                if proj:
                    plan = Project(plan, proj)

        # 4. ORDER BY
        order_node = node.args.get("order")
        if order_node:
            by, desc = [], []
            for o in order_node.expressions:
                by.append(self._expr(o.this))
                desc.append(bool(o.args.get("desc")))
            plan = Sort(plan, by, desc)

        # 5. LIMIT
        limit_node = node.args.get("limit")
        if limit_node:
            plan = Limit(plan, int(limit_node.expression.this))

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

            on = join.args.get("on")
            using = join.args.get("using")
            if on is not None:
                left_on, right_on = self._parse_join_on(on, self._left_schema, right_schema)
            elif using:
                keys = [str(c.name) for c in using.expressions]
                left_on = right_on = keys
            else:
                raise ValueError("JOIN requires ON or USING clause")

            how = "inner"
            kind = join.args.get("kind")
            if kind:
                ks = str(kind).lower()
                if "left" in ks:
                    how = "left"
                elif "right" in ks:
                    how = "right"

            plan = Join(plan, right_plan, left_on, right_on, how)
            # After this join, the accumulated left side grows to include right cols
            self._left_schema |= right_schema
            # Record that right join keys should be referred to by left join keys
            for lk, rk in zip(left_on, right_on):
                if lk != rk:
                    self._col_aliases[rk] = lk

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

    def _contains_agg(self, e: sge.Expression) -> bool:
        inner = e.this if isinstance(e, sge.Alias) else e
        return isinstance(inner, (sge.Sum, sge.Avg, sge.Min, sge.Max, sge.Count))

    def _build_aggregate(self, plan: LogicalPlan, node: sge.Select, group_node) -> LogicalPlan:
        # Group-by keys
        group_by: List[Expr] = []
        if group_node:
            for gb in group_node.expressions:
                group_by.append(self._expr(gb))

        # Aggregate expressions (only the aggregate parts of SELECT)
        agg_exprs: List[Expr] = []
        for e in node.expressions:
            if isinstance(e, sge.Star):
                continue
            if not self._contains_agg(e):
                continue  # group key — already in group_by
            mx = self._select_expr(e)
            if mx is not None:
                agg_exprs.append(mx)

        return Aggregate(plan, group_by, agg_exprs)

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
        if alias_name:
            mx = mx.alias(alias_name)
        return mx

    # ------------------------------------------------------------------
    # expression translation
    # ------------------------------------------------------------------

    def _expr(self, node: sge.Expression) -> Expr:
        """Translate a sqlglot expression to an MXFrame Expr."""

        # ---- column reference ----
        if isinstance(node, sge.Column):
            name = node.name.lower()
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
