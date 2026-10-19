"""Names and types for SQL: the Catalog of sources, the typed expression tree and the binder (F-150).

The binder turns a parsed statement (`sql.mojo`) into a `BoundQuery`:
- row-level expressions (WHERE, aggregate arguments, GROUP BY keys) bound against the source's columns, in `rexprs`;
- the aggregates the query computes (`aggs`), each once, and the group keys (`keys`);
- the expressions that run after aggregation (select items, HAVING, ORDER BY), in `pexprs`, whose leaves are
  group-key and aggregate references.
It checks types the way DuckDB does for the subset (decimal scales aligned, a string literal read as a DATE next to a
DATE) and raises with the position of the offending text. Nothing here reads data: the planner and the executor do.
"""
from .table import SharedTable, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT, F32_COLUMN
from .parquet import ParquetScan
from .frame import Frame, Overlay, Formula
from .expr import N_COL, N_LIT, N_LITF, N_DIV, N_I2F, N_ADD, N_SUB, N_MUL, N_FLOORDIV, N_MOD, N_NEG, N_LEAST, N_GREATEST, N_SCALE_TO
from .csv import days_from_civil, _days_in_month, parse_float_text, F_OK, F_RANGE
from .sql import (SqlStatement, parse_sql, sql_error, closest, lower_ascii, X_INT, X_DEC, X_STR, X_NULL, X_TRUE, X_FALSE,
                  X_DATE, X_COL, X_NEG, X_BIN, X_NOT, X_ISNULL, X_BETWEEN, X_IN, X_CALL, X_FLT, EXPLAIN_NONE, SqlExpr, SelectItem)

comptime K_NULL = 6          # the type of a bare NULL literal

# ── the catalog ──────────────────────────────────────────────────────────────────────────────────────

comptime SRC_TABLE = 0
comptime SRC_FRAME = 1       # a Frame over a table: its edits as a mask over the shared rows plus the changed rows
comptime SRC_SCAN = 2        # a Parquet file read one row group at a time (also a Frame over one)


struct SqlSource(Copyable, Movable):
    var name: String
    var kind: Int
    var table: SharedTable           # the data (a table, a frame's base); a scan's columns without rows
    var scan: Optional[ParquetScan]
    var overlay: Overlay             # a frame's edits as of registration (empty otherwise)
    var framed: Bool
    var version: Int                 # a frame's edit counter when it was registered: stats are cached per version
    var range_cols: List[Int]        # the planner's cache of integer-like column ranges: column, lowest, highest, has values
    var range_lo: List[Int64]
    var range_hi: List[Int64]
    var range_has: List[Bool]

    def __init__(out self, name: String, table: SharedTable):
        self.name = name
        self.kind = SRC_TABLE
        self.table = table
        self.scan = None
        self.overlay = Overlay(table)
        self.framed = False
        self.version = 0
        self.range_cols = List[Int]()
        self.range_lo = List[Int64]()
        self.range_hi = List[Int64]()
        self.range_has = List[Bool]()

    def columns(self) -> Int:
        return self.table.columns()

    def rows(self) -> Int:
        """Rows in the source as registered (a frame: its base rows plus appended, removed rows included)."""
        if self.scan:
            return self.scan.value().rows()
        return self.table.rows()


struct Catalog(Movable):
    """The named sources a query may read. `add_table`, `add_frame` and `add_scan` register one under a name
    (case does not matter); a frame is snapshotted when added: add it again after more edits."""
    var sources: List[SqlSource]

    def __init__(out self):
        self.sources = List[SqlSource]()

    def _put(mut self, var s: SqlSource):
        var key = lower_ascii(s.name)
        for i in range(len(self.sources)):
            if lower_ascii(self.sources[i].name) == key:
                self.sources[i] = s^
                return
        self.sources.append(s^)

    def add_table(mut self, name: String, table: SharedTable):
        self._put(SqlSource(name, table))

    def add_scan(mut self, name: String, scan: ParquetScan):
        var s = SqlSource(name, scan.schema)
        s.kind = SRC_SCAN
        s.scan = scan.copy()
        self._put(s^)

    def add_frame(mut self, name: String, frame: Frame) raises:
        var s = SqlSource(name, frame.base)
        s.kind = SRC_SCAN if frame.scan else SRC_FRAME
        s.scan = frame.scan.copy()
        s.overlay = frame.overlay()
        s.framed = True
        s.version = frame.edits
        self._put(s^)

    def find(self, name: String) -> Int:
        var key = lower_ascii(name)
        for i in range(len(self.sources)):
            if lower_ascii(self.sources[i].name) == key:
                return i
        return -1

    def names(self) -> List[String]:
        var out = List[String]()
        for s in self.sources:
            out.append(s.name)
        return out^


# ── the typed expression tree ────────────────────────────────────────────────────────────────────────

comptime B_COL = 0           # value: the source column
comptime B_LIT = 1           # value: the stored value (kind INT, DECIMAL, DATE or BOOL)
comptime B_STR = 2           # text: a string literal, still to be matched against a dictionary
comptime B_NULL = 3
comptime B_ADD = 10
comptime B_SUB = 11
comptime B_MUL = 12
comptime B_IDIV = 13         # integer division, truncating toward zero; x // 0 is NULL
comptime B_IMOD = 14         # remainder with the dividend's sign; x % 0 is NULL
comptime B_NEG = 15
comptime B_DIV = 16          # `/`: a FLOAT (the operands are FLOATs before aggregation)
comptime B_I2F = 28          # an INT or DECIMAL as a FLOAT (a = the operand, value = its decimal places), as DuckDB casts: float(v) / 10^scale
comptime B_RESCALE = 17      # a times 10^value: the same number with `value` more decimal places
comptime B_FDIV = 18         # a frame formula's `//`: floor division, x // 0 is 0 (the builder's rule, not SQL's)
comptime B_FMOD = 19         # a frame formula's `%`: the divisor's sign, x % 0 is 0
comptime B_LEAST = 26        # a frame formula's min(a, b)
comptime B_GREATEST = 27     # a frame formula's max(a, b)
comptime B_CMP = 20          # value: 20 <, 21 <=, 22 >, 23 >=, 24 =, 25 <> (the kernel's own codes)
comptime B_AND = 30
comptime B_OR = 31
comptime B_NOT = 32
comptime B_ISNULL = 33
comptime B_IN = 34           # a IN (items), the items literals; value 1: the list holds a NULL
comptime B_AGG = 40          # value: the aggregate's number (after aggregation)
comptime B_KEY = 41          # value: the group key's number (after aggregation)

comptime C_LT = 20
comptime C_LE = 21
comptime C_GT = 22
comptime C_GE = 23
comptime C_EQ = 24
comptime C_NE = 25


struct BExpr(Copyable, Movable):
    var op: Int
    var a: Int
    var b: Int
    var items: List[Int]
    var kind: Int                    # ColumnKind value of the result, or K_NULL
    var scale: Int
    var value: Int64
    var text: String
    var src: Int                     # a TEXT value: the source column whose dictionary its codes belong to
    var line: Int
    var col: Int

    def __init__(out self, op: Int, kind: Int, scale: Int, a: Int, b: Int, value: Int64, text: String, line: Int, col: Int):
        self.op = op
        self.src = -1
        self.kind = kind
        self.scale = scale
        self.a = a
        self.b = b
        self.items = List[Int]()
        self.value = value
        self.text = text
        self.line = line
        self.col = col


comptime F_COUNT_STAR = 0
comptime F_COUNT = 1
comptime F_SUM = 2
comptime F_MIN = 3
comptime F_MAX = 4
comptime F_AVG = 5
comptime F_COUNT_DISTINCT = 6
comptime F_MEDIAN = 7
comptime F_QUANTILE = 8


struct AggSpec(Copyable, Movable):
    var func: Int
    var arg: Int                     # a row-level expression of `rexprs`, -1 for COUNT(*)
    var p: Float64
    var kind: Int                    # the result type
    var scale: Int
    var name: String                 # as the output names it: sum(revenue), count_star()
    var canon: String
    var hol: Bool                    # needs every value of its group: count(DISTINCT), median, quantile_disc, min / max of TEXT

    def __init__(out self, func: Int, arg: Int, p: Float64, kind: Int, scale: Int, name: String, canon: String, hol: Bool = False):
        self.hol = hol
        self.func = func
        self.arg = arg
        self.p = p
        self.kind = kind
        self.scale = scale
        self.name = name
        self.canon = canon


struct KeySpec(Copyable, Movable):
    var expr: Int                    # a row-level expression
    var name: String
    var canon: String

    def __init__(out self, expr: Int, name: String, canon: String):
        self.expr = expr
        self.name = name
        self.canon = canon


struct SortKey(Copyable, Movable):
    var expr: Int                    # a post-aggregate expression
    var desc: Bool
    var nulls_first: Bool

    def __init__(out self, expr: Int, desc: Bool, nulls_first: Bool):
        self.expr = expr
        self.desc = desc
        self.nulls_first = nulls_first


struct BoundQuery(Movable):
    var source: Int                  # the catalog's source
    var rexprs: List[BExpr]
    var pexprs: List[BExpr]
    var where: Int                   # a row-level boolean, -1 for none
    var keys: List[KeySpec]
    var aggs: List[AggSpec]
    var outputs: List[Int]           # per select item: a post-aggregate expression
    var out_names: List[String]
    var having: Int                  # a post-aggregate boolean, -1 for none
    var order: List[SortKey]
    var limit: Int
    var offset: Int
    var explain: Int
    var text: String                 # the SQL
    var line: Int                    # where its select list starts (for errors that belong to the whole query)
    var col: Int
    var where_text: String
    var having_text: String
    var order_text: List[String]
    var out_text: List[String]
    # F-152: row queries (no aggregate, no GROUP BY) and SELECT DISTINCT
    var rows: Bool                   # a row query: `routs` and `rorder` hold its select list and ORDER BY (expressions of `rexprs`)
    var distinct: Bool               # SELECT DISTINCT (bound as a group-by on the select list)
    var routs: List[Int]
    var rorder: List[SortKey]

    def __init__(out self):
        self.rows = False
        self.distinct = False
        self.routs = List[Int]()
        self.rorder = List[SortKey]()
        self.source = -1
        self.rexprs = List[BExpr]()
        self.pexprs = List[BExpr]()
        self.where = -1
        self.keys = List[KeySpec]()
        self.aggs = List[AggSpec]()
        self.outputs = List[Int]()
        self.out_names = List[String]()
        self.having = -1
        self.order = List[SortKey]()
        self.limit = -1
        self.offset = 0
        self.explain = EXPLAIN_NONE
        self.text = String()
        self.line = 1
        self.col = 1
        self.where_text = String()
        self.having_text = String()
        self.order_text = List[String]()
        self.out_text = List[String]()


# ── helpers ──────────────────────────────────────────────────────────────────────────────────────────

def pow10_checked(k: Int) raises -> Int64:
    if k < 0 or k > 18:
        raise Error("a power of ten outside 10^0 .. 10^18")
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def kind_name(kind: Int, scale: Int = 0) -> String:
    if kind == K_NULL:
        return "NULL"
    if kind == 1:
        return "DECIMAL(" + String(scale) + ")"
    var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL", "FLOAT"]
    return names[kind]


def _is_numeric(kind: Int) -> Bool:
    return kind == INT.value or kind == DECIMAL.value


def _is_num(kind: Int) -> Bool:
    """A number the arithmetic takes: INT, DECIMAL or FLOAT."""
    return kind == INT.value or kind == DECIMAL.value or kind == FLOAT.value


def decimal_to_float(v: Int64, scale: Int) -> Float64:
    """A stored decimal as a float64, as DuckDB casts DECIMAL to DOUBLE: the integer as a double, divided by the power of ten."""
    if scale == 0:
        return Float64(v)
    var p = Float64(1.0)
    for _ in range(scale):
        p *= 10.0                     # exact up to 10^22
    return Float64(v) / p


def decimal_to_f32(v: Int64, scale: Int) -> Float64:
    """A stored decimal as a 32-bit float (widened), as DuckDB casts DECIMAL to REAL: the integer as a float32 divided by the power
    of ten in float32 (exact for up to ten places and 24 bits of integer), else the float64 value rounded to float32."""
    if scale <= 10 and v > -16777216 and v < 16777216:
        var p = Float32(1.0)
        for _ in range(scale):
            p *= 10.0
        return Float64(Float32(Float64(v)) / p)
    return Float64(Float32(decimal_to_float(v, scale)))


def date_problem(s: String) -> String:
    """Why 'YYYY-MM-DD' text is not a date, "" when it is one."""
    var b = s.as_bytes()
    if len(b) != 10 or b[4] != UInt8(45) or b[7] != UInt8(45):
        return "write it as YYYY-MM-DD"
    var nums = List[Int]()
    for rng in [(0, 4), (5, 7), (8, 10)]:
        var v = 0
        for i in range(rng[0], rng[1]):
            if not (UInt8(48) <= b[i] <= UInt8(57)):
                return "write it as YYYY-MM-DD"
            v = v * 10 + Int(b[i] - 48)
        nums.append(v)
    if nums[1] < 1 or nums[1] > 12:
        return "there is no month " + String(nums[1])
    var days = _days_in_month(nums[0], nums[1])
    if nums[2] < 1 or nums[2] > days:
        var names: List[String] = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        return "the day " + String(nums[2]) + " does not exist: " + names[nums[1] - 1] + " " + String(s[byte=0:4]) + " has " + String(days) + " days"
    return ""


def parse_date_literal(s: String) -> Int:
    """Days since 1970-01-01 of 'YYYY-MM-DD', or Int.MIN when the text is not such a date."""
    var b = s.as_bytes()
    if len(b) != 10 or b[4] != UInt8(45) or b[7] != UInt8(45):
        return Int.MIN
    var nums = List[Int]()
    for rng in [(0, 4), (5, 7), (8, 10)]:
        var v = 0
        for i in range(rng[0], rng[1]):
            if not (UInt8(48) <= b[i] <= UInt8(57)):
                return Int.MIN
            v = v * 10 + Int(b[i] - 48)
        nums.append(v)
    if nums[1] < 1 or nums[1] > 12 or nums[2] < 1 or nums[2] > _days_in_month(nums[0], nums[1]):
        return Int.MIN
    return days_from_civil(nums[0], nums[1], nums[2])


struct _Num(Movable):
    var value: Int64
    var scale: Int
    var ok: Bool

    def __init__(out self):
        self.value = 0
        self.scale = 0
        self.ok = False


def parse_number(text: String) -> _Num:
    """The stored value and scale of a literal such as `-12.50`; ok is False past int64 or 18 decimal places."""
    var out = _Num()
    var b = text.as_bytes()
    var neg = False
    var i = 0
    if len(b) > 0 and b[0] == UInt8(45):
        neg = True
        i = 1
    var v = UInt64(0)
    var scale = 0
    var seen_dot = False
    var limit = UInt64(9223372036854775807) + (UInt64(1) if neg else UInt64(0))
    while i < len(b):
        if b[i] == UInt8(46):
            seen_dot = True
            i += 1
            continue
        var d = UInt64(b[i] - 48)
        if v > (limit - d) // 10:
            return out^
        v = v * 10 + d
        if seen_dot:
            scale += 1
        i += 1
    if scale > 18:
        return out^
    out.value = Int64(0) - Int64(v) if neg else Int64(v)
    out.scale = scale
    out.ok = True
    return out^


# ── the binder ───────────────────────────────────────────────────────────────────────────────────────

struct _Binder(Movable):
    var st: SqlStatement
    var cat_names: List[String]
    var src_names: List[String]       # the source's columns
    var src_kinds: List[Int]
    var src_scales: List[Int]
    var table_name: String
    var table_alias: String
    var q: BoundQuery
    var in_post: Bool
    var key_canons: List[String]
    var alias_expr: List[Int]         # post-aggregate expression of each select item, by item (-1 while unbound)
    var n_stored: Int                 # the source's stored columns; the frame's formula columns follow them
    var formulas: List[Formula]
    var distinct: Bool

    def __init__(out self, var st: SqlStatement):
        self.distinct = False
        self.st = st^
        self.cat_names = List[String]()
        self.src_names = List[String]()
        self.src_kinds = List[Int]()
        self.src_scales = List[Int]()
        self.table_name = String()
        self.table_alias = String()
        self.q = BoundQuery()
        self.in_post = False
        self.key_canons = List[String]()
        self.alias_expr = List[Int]()
        self.n_stored = 0
        self.formulas = List[Formula]()

    def take(deinit self) -> BoundQuery:
        return self.q^

    # -- tree building
    def mk(mut self, op: Int, kind: Int, scale: Int, a: Int, b: Int, value: Int64, text: String, ast: Int) -> Int:
        var line = self.st.exprs[ast].line if ast >= 0 else 0
        var col = self.st.exprs[ast].col if ast >= 0 else 0
        if self.in_post:
            self.q.pexprs.append(BExpr(op, kind, scale, a, b, value, text, line, col))
            return len(self.q.pexprs) - 1
        self.q.rexprs.append(BExpr(op, kind, scale, a, b, value, text, line, col))
        return len(self.q.rexprs) - 1

    def at(self, e: Int) -> BExpr:
        return self.q.pexprs[e].copy() if self.in_post else self.q.rexprs[e].copy()

    def err(self, ast: Int, msg: String) -> Error:
        return sql_error(self.st.exprs[ast].line, self.st.exprs[ast].col, msg)

    # -- names
    def find_column(self, ast: Int) raises -> Int:
        ref x = self.st.exprs[ast]
        if x.text2.byte_length() > 0:
            var q = lower_ascii(x.text2)
            if q != lower_ascii(self.table_name) and (self.table_alias.byte_length() == 0 or q != lower_ascii(self.table_alias)):
                raise self.err(ast, "unknown table or alias `" + x.text2 + "` in `" + x.text2 + "." + x.text + "`")
        var key = lower_ascii(x.text)
        var found = -1
        for c in range(len(self.src_names)):
            if lower_ascii(self.src_names[c]) == key:
                if found >= 0:
                    raise self.err(ast, "column `" + x.text + "` is ambiguous: `" + self.src_names[found] + "` and `" + self.src_names[c] + "`")
                found = c
        if found < 0:
            var hint = closest(x.text, self.src_names)
            raise self.err(ast, "column `" + x.text + "` not found in table `" + self.table_name + "`" +
                           ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else ""))
        return found

    def canon(self, e: Int) -> String:
        """The expression with names in lower case, to tell whether a select item is a GROUP BY key."""
        ref x = self.st.exprs[e]
        if x.kind == X_COL:
            var nm = lower_ascii(x.text)
            return nm
        if x.kind == X_CALL:
            var s = lower_ascii(x.text) + "("
            if x.a == 1:
                s += "*"
            elif x.flag:
                s += "distinct "
            for k in range(len(x.items)):
                s += ("," if k > 0 else "") + self.canon(x.items[k])
            return s + ")"
        if x.kind == X_BIN:
            return "(" + self.canon(x.a) + x.text + self.canon(x.b) + ")"
        if x.kind == X_NEG or x.kind == X_NOT:
            return x.text + "(" + self.canon(x.a) + ")"
        if x.kind == X_ISNULL:
            return "isnull" + ("not" if x.flag else "") + "(" + self.canon(x.a) + ")"
        if x.kind == X_BETWEEN:
            return "between" + ("not" if x.flag else "") + "(" + self.canon(x.a) + "," + self.canon(x.b) + "," + self.canon(x.c) + ")"
        if x.kind == X_IN:
            var s = "in" + ("not" if x.flag else "") + "(" + self.canon(x.a)
            for k in range(len(x.items)):
                s += "," + self.canon(x.items[k])
            return s + ")"
        if x.kind == X_STR:
            return "'" + x.text + "'"
        return x.text

    # -- a frame's formula column: its expression over stored columns, bound as the builder evaluates it
    def formula_column(mut self, fi: Int, ast: Int, depth: Int) raises -> Int:
        if depth > 32:
            raise self.err(ast, "formulas refer to each other in a cycle")
        var stack = List[Int]()
        var nodes = self.formulas[fi].expr.nodes.copy()
        var fname = self.formulas[fi].name
        var fscale = self.formulas[fi].scale
        for n in nodes:
            if n.op == N_COL:
                var found = -1
                for c in range(len(self.src_names)):
                    if self.src_names[c] == n.name:
                        found = c
                        break
                if found < 0:
                    raise self.err(ast, "formula `" + fname + "` reads column `" + n.name + "`, which is not there")
                if found >= self.n_stored:
                    stack.append(self.formula_column(found - self.n_stored, ast, depth + 1))
                else:
                    var node = self.mk(B_COL, self.src_kinds[found], self.src_scales[found], -1, -1, Int64(found), String(self.src_names[found]), ast)
                    if self.src_kinds[found] == TEXT.value:
                        self.q.rexprs[node].src = found
                    stack.append(node)
            elif n.op == N_LIT:
                stack.append(self.mk(B_LIT, INT.value, 0, -1, -1, n.value, "", ast))
            elif n.op == N_LITF:
                stack.append(self.mk(B_LIT, FLOAT.value, 0, -1, -1, n.value, "", ast))
            elif n.op == N_I2F:
                var a = stack.pop()
                stack.append(self.mk(B_I2F, FLOAT.value, 0, a, -1, n.value, "", ast))
            elif n.op == N_SCALE_TO:
                pass                                              # the child already holds the scaled number
            elif n.op == N_NEG:
                var a = stack.pop()
                stack.append(self.mk(B_NEG, self.at(a).kind, 0, a, -1, 0, "", ast))
            else:
                var b = stack.pop()
                var a = stack.pop()
                var op = B_ADD if n.op == N_ADD else B_SUB if n.op == N_SUB else B_MUL if n.op == N_MUL else B_FDIV if n.op == N_FLOORDIV else \
                         B_FMOD if n.op == N_MOD else B_LEAST if n.op == N_LEAST else B_GREATEST if n.op == N_GREATEST else B_DIV if n.op == N_DIV else -1
                if op < 0:
                    raise self.err(ast, "formula `" + fname + "` uses an operator SQL cannot read yet")
                var float_op = self.at(a).kind == FLOAT.value or self.at(b).kind == FLOAT.value or n.op == N_DIV
                stack.append(self.mk(op, FLOAT.value if float_op else INT.value, 0, a, b, 0, "", ast))
        var top = stack.pop()
        self.q.rexprs[top].kind = FLOAT.value if fscale < 0 else DECIMAL.value if fscale > 0 else INT.value
        self.q.rexprs[top].scale = max(fscale, 0)
        return top

    # -- literals
    def literal(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if x.kind == X_INT or x.kind == X_DEC:
            var n = parse_number(x.text)
            if not n.ok:
                raise self.err(ast, "the number " + x.text + " does not fit: integers must fit 64 bits and decimals 18 places")
            return self.mk(B_LIT, INT.value if x.kind == X_INT else DECIMAL.value, n.scale, -1, -1, n.value, String(x.text), ast)
        if x.kind == X_FLT:
            var f = Float64(0.0)
            var st = parse_float_text(x.text, f)
            if st == F_RANGE:
                raise self.err(ast, "the number " + x.text + " is past the largest float64 (about 1.8e308)")
            return self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(f.to_bits()), String(x.text), ast)
        if x.kind == X_STR:
            return self.mk(B_STR, TEXT.value, 0, -1, -1, 0, String(x.text), ast)
        if x.kind == X_NULL:
            return self.mk(B_NULL, K_NULL, 0, -1, -1, 0, "", ast)
        if x.kind == X_TRUE or x.kind == X_FALSE:
            return self.mk(B_LIT, BOOL.value, 0, -1, -1, Int64(1 if x.kind == X_TRUE else 0), String(x.text), ast)
        var days = parse_date_literal(x.text)
        if days == Int.MIN:
            raise self.err(ast, "`" + x.text + "` is not a date: " + date_problem(x.text))
        return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(days), String(x.text), ast)

    # -- coercions between the two sides of an operator
    def rescale(mut self, e: Int, to: Int, ast: Int) raises -> Int:
        var x = self.at(e)
        if x.scale == to:
            return e
        var k = to - x.scale
        if k < 0:
            raise self.err(ast, "internal: a value cannot lose decimal places")
        if x.op == B_LIT:
            var m = pow10_checked(k)
            var v = x.value * m
            if x.value != 0 and v // x.value != m:
                raise self.err(ast, "the number " + x.text + " does not fit at " + String(to) + " decimal places")
            return self.mk(B_LIT, DECIMAL.value if to > 0 else INT.value, to, -1, -1, v, x.text, ast)
        return self.mk(B_RESCALE, DECIMAL.value, to, e, -1, Int64(k), "", ast)

    def string_as_date(mut self, e: Int, ast: Int) raises -> Int:
        var x = self.at(e)
        var days = parse_date_literal(x.text)
        if days == Int.MIN:
            raise self.err(ast, "cannot read '" + x.text + "' as a DATE: " + date_problem(x.text))
        return self.mk(B_LIT, DATE.value, 0, -1, -1, Int64(days), String(x.text), ast)

    def align(mut self, mut a: Int, mut b: Int, ast: Int) raises:
        """Gives two numeric operands the same decimal places (the shorter one is scaled up), as DuckDB does."""
        var x = self.at(a)
        var y = self.at(b)
        if x.kind == FLOAT.value or y.kind == FLOAT.value:
            return
        if x.scale < y.scale:
            a = self.rescale(a, y.scale, ast)
        elif y.scale < x.scale:
            b = self.rescale(b, x.scale, ast)


    def to_float(mut self, e: Int, ast: Int, real: Bool = False) raises -> Int:
        """The operand as a FLOAT. `real`: in a 32-bit context (DuckDB's REAL: the other side is a 32-bit float), so an INT or DECIMAL is
        cast to REAL: a literal is folded to a 32-bit float, anything else is converted and rounded to 32 bits. A FLOAT stays as it is
        (an exponent literal is a DOUBLE and is never re-rounded). After aggregation the evaluator reads INT and DECIMAL as doubles
        itself, so only a literal is folded there and a 32-bit context with anything else raises."""
        var x = self.at(e)
        if x.kind == FLOAT.value:
            return e
        if x.op == B_LIT:
            var f = decimal_to_f32(x.value, x.scale) if real else decimal_to_float(x.value, x.scale)
            return self.mk(B_LIT, FLOAT.value, 0, -1, -1, Int64(f.to_bits()), x.text, ast)
        if self.in_post:
            if real:
                raise self.err(ast, "a 32-bit float (a REAL column, or min / max of one) meeting an INT or DECIMAL value that is not a literal, after aggregation, is not supported: DuckDB does it in 32 bits")
            return e
        if real and x.scale > 10:
            raise self.err(ast, "a DECIMAL with more than 10 places meeting a 32-bit float is not supported")
        return self.mk(B_I2F, FLOAT.value, F32_COLUMN if real else 0, e, -1, Int64(x.scale), "", ast)

    def float_pair(mut self, mut a: Int, mut b: Int, ast: Int) raises -> Bool:
        """Both operands as FLOATs, in DuckDB's type for the pair: a 32-bit float (REAL: a Parquet FLOAT column, min / max of one, its
        negation or arithmetic) next to REAL, INT or DECIMAL is REAL (`l_discount <= 0.07` keeps the rows holding 0.07; `f = d` casts the
        DECIMAL to REAL); next to a DOUBLE (a double column, an exponent literal such as 7e-2) it is DOUBLE. Returns True for REAL."""
        var x = self.at(a)
        var y = self.at(b)
        var xr = x.kind == FLOAT.value and x.scale == F32_COLUMN
        var yr = y.kind == FLOAT.value and y.scale == F32_COLUMN
        var xd = x.kind == FLOAT.value and x.scale != F32_COLUMN
        var yd = y.kind == FLOAT.value and y.scale != F32_COLUMN
        var real = (xr or yr) and not xd and not yd
        a = self.to_float(a, ast, real)
        b = self.to_float(b, ast, real)
        return real

    def text_node(mut self, e: Int) -> Bool:
        var op = self.at(e).op
        return op == B_COL or op == B_KEY or op == B_AGG

    def compare(mut self, code: Int, left: Int, right: Int, ast: Int) raises -> Int:
        var a = left
        var b = right
        var op = code
        var x = self.at(a)
        var y = self.at(b)
        var opname = self.st.exprs[ast].text
        if x.op == B_NULL or y.op == B_NULL:                 # anything compared with NULL is unknown
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        # a string next to a DATE is that date
        if x.kind == DATE.value and y.op == B_STR:
            b = self.string_as_date(b, ast)
            y = self.at(b)
        elif y.kind == DATE.value and x.op == B_STR:
            a = self.string_as_date(a, ast)
            x = self.at(a)
        if x.kind == TEXT.value or y.kind == TEXT.value:
            if x.kind != TEXT.value or y.kind != TEXT.value:
                raise self.err(ast, "cannot compare " + kind_name(x.kind, x.scale) + " with " + kind_name(y.kind, y.scale) + " (`" + opname + "`)")
            if x.op == B_STR and y.op == B_STR:
                raise self.err(ast, "comparing two string literals is not supported (it is a constant)")
            if x.op == B_STR:                                  # the column goes first: `'a' < c` is `c > 'a'`
                var t = a
                a = b
                b = t
                op = C_GT if code == C_LT else C_GE if code == C_LE else C_LT if code == C_GT else C_LE if code == C_GE else code
            if not self.text_node(a) or self.at(b).op != B_STR:
                raise self.err(ast, "TEXT can only be compared with a string literal (a column, a group key or MIN / MAX of one on the other side)")
        elif _is_numeric(x.kind) and _is_numeric(y.kind):
            self.align(a, b, ast)
        elif (x.kind == FLOAT.value and _is_num(y.kind)) or (y.kind == FLOAT.value and _is_num(x.kind)):
            _ = self.float_pair(a, b, ast)
        elif x.kind == y.kind and (x.kind == DATE.value or x.kind == BOOL.value):
            pass
        else:
            raise self.err(ast, "cannot compare " + kind_name(x.kind, x.scale) + " with " + kind_name(y.kind, y.scale) + " (`" + opname + "`)")
        return self.mk(B_CMP, BOOL.value, 0, a, b, Int64(op), "", ast)

    def numeric_result(self, x: BExpr, y: BExpr, ast: Int) raises -> Bool:
        """Both operands are numbers (INT, DECIMAL or, after aggregation, FLOAT)."""
        var ok = _is_num(x.kind) and _is_num(y.kind)
        if not ok and x.kind != K_NULL and y.kind != K_NULL:
            raise self.err(ast, "cannot apply `" + self.st.exprs[ast].text + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        return ok

    def arith(mut self, op: String, left: Int, right: Int, ast: Int) raises -> Int:
        var a = left
        var b = right
        var x = self.at(a)
        var y = self.at(b)
        if x.op == B_NULL or y.op == B_NULL:
            var other = y.copy() if x.op == B_NULL else x.copy()
            return self.mk(B_NULL, other.kind, other.scale, -1, -1, 0, "", ast)
        # DATE arithmetic: date +- days, date - date
        if x.kind == DATE.value or y.kind == DATE.value:
            if op == "-" and x.kind == DATE.value and y.kind == DATE.value:
                return self.mk(B_SUB, INT.value, 0, a, b, 0, "", ast)
            if (op == "+" and ((x.kind == DATE.value and y.kind == INT.value) or (x.kind == INT.value and y.kind == DATE.value))) or \
               (op == "-" and x.kind == DATE.value and y.kind == INT.value):
                return self.mk(B_ADD if op == "+" else B_SUB, DATE.value, 0, a, b, 0, "", ast)
            raise self.err(ast, "cannot apply `" + op + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        if not _is_num(x.kind) or not _is_num(y.kind):
            raise self.err(ast, "cannot apply `" + op + "` to " + kind_name(x.kind, x.scale) + " and " + kind_name(y.kind, y.scale))
        if op == "/":                                          # DuckDB's `/` is a division of doubles (of 32-bit floats next to REAL)
            var real_div = self.float_pair(a, b, ast)
            if real_div and self.in_post:
                raise self.err(ast, "arithmetic on 32-bit floats after aggregation is not supported: DuckDB does it in 32 bits")
            return self.mk(B_DIV, FLOAT.value, F32_COLUMN if real_div else 0, a, b, 0, "", ast)
        if x.kind == FLOAT.value or y.kind == FLOAT.value:
            if op == "//" or op == "%":
                raise self.err(ast, "`" + op + "` on a FLOAT value is not supported")
            var real_op = self.float_pair(a, b, ast)
            if real_op and self.in_post:
                raise self.err(ast, "arithmetic on 32-bit floats after aggregation is not supported: DuckDB does it in 32 bits")
            return self.mk(B_ADD if op == "+" else B_SUB if op == "-" else B_MUL, FLOAT.value, F32_COLUMN if real_op else 0, a, b, 0, "", ast)
        if op == "//":
            if x.kind != INT.value or y.kind != INT.value:
                raise self.err(ast, "`//` on a DECIMAL gives a FLOAT in DuckDB (a floor of doubles), which MX frame does not do yet: use it on integers")
            return self.mk(B_IDIV, INT.value, 0, a, b, 0, "", ast)
        if op == "*":
            var scale = x.scale + y.scale
            if scale > 18:
                raise self.err(ast, "the product has " + String(scale) + " decimal places; the most a value can hold is 18")
            if x.op == B_LIT and y.op == B_LIT:
                var v = x.value * y.value
                if (x.value != 0 and v // x.value != y.value) or (x.value == -1 and y.value == Int64.MIN):
                    raise self.err(ast, "the product passes the int64 range")
                return self.mk(B_LIT, DECIMAL.value if scale > 0 else INT.value, scale, -1, -1, v, "", ast)
            return self.mk(B_MUL, DECIMAL.value if scale > 0 else INT.value, scale, a, b, 0, "", ast)
        self.align(a, b, ast)
        x = self.at(a)
        y = self.at(b)
        var kind = DECIMAL.value if x.scale > 0 else INT.value
        if op == "%":
            return self.mk(B_IMOD, kind, x.scale, a, b, 0, "", ast)
        if x.op == B_LIT and y.op == B_LIT:
            var v = x.value + y.value if op == "+" else x.value - y.value
            var over = ((x.value ^ v) & (y.value ^ v)) < 0 if op == "+" else ((x.value ^ y.value) & (x.value ^ v)) < 0
            if over:
                raise self.err(ast, "the result passes the int64 range")
            return self.mk(B_LIT, kind, x.scale, -1, -1, v, "", ast)
        return self.mk(B_ADD if op == "+" else B_SUB, kind, x.scale, a, b, 0, "", ast)

    def boolean(mut self, e: Int, ast: Int, what: String) raises -> Int:
        var x = self.at(e)
        if x.op == B_NULL and x.kind == K_NULL:
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        if x.kind != BOOL.value:
            raise self.err(ast, what + " needs a true / false value, found " + kind_name(x.kind, x.scale))
        return e

    def in_list(mut self, ast: Int) raises -> Int:
        var xi = self.st.exprs[ast].copy()
        var a = self.value(xi.a)
        var xa = self.at(a)
        if xa.op == B_NULL:
            return self.mk(B_NULL, BOOL.value, 0, -1, -1, 0, "", ast)
        var lits = List[Int]()
        var has_null = False
        var scale = xa.scale
        var any_double = False                                    # one DOUBLE in the list makes the whole list DOUBLE, as DuckDB's common type
        for it in xi.items:
            any_double = any_double or self.st.exprs[it].kind == X_FLT
        for it in xi.items:
            var ik = self.st.exprs[it].kind
            if ik == X_COL or ik == X_CALL:
                raise self.err(it, "IN lists must hold literals, not column references or functions")
            var lit = self.value(it)
            var l = self.at(lit)
            if l.op == B_NULL:
                has_null = True
                continue
            if l.op != B_LIT and l.op != B_STR:
                raise self.err(it, "IN lists must hold literals")
            if xa.kind == DATE.value and l.op == B_STR:
                lit = self.string_as_date(lit, it)
                l = self.at(lit)
            if xa.kind == TEXT.value:
                if l.op != B_STR:
                    raise self.err(it, "cannot compare TEXT with " + kind_name(l.kind, l.scale) + " in an IN list")
                if not self.text_node(a):
                    raise self.err(ast, "TEXT can only be tested against a list when it is a plain column")
            elif xa.kind == FLOAT.value and _is_num(l.kind):
                lit = self.to_float(lit, it, xa.scale == F32_COLUMN and not any_double)
            elif _is_numeric(xa.kind) and l.kind == FLOAT.value:
                raise self.err(it, "an IN list with a FLOAT value needs a FLOAT on the left: " + kind_name(xa.kind, xa.scale) + " IN (a float) compares as doubles")
            elif _is_numeric(xa.kind) and _is_numeric(l.kind):
                scale = max(scale, l.scale)
            elif xa.kind == l.kind and (xa.kind == DATE.value or xa.kind == BOOL.value):
                pass
            else:
                raise self.err(it, "cannot compare " + kind_name(xa.kind, xa.scale) + " with " + kind_name(l.kind, l.scale) + " in an IN list")
            lits.append(lit)
        if xa.kind != TEXT.value and len(lits) > 40:
            raise self.err(ast, "an IN list of " + String(len(lits)) + " numbers is not supported yet: the most is 40")
        if _is_numeric(xa.kind) and scale > xa.scale:
            a = self.rescale(a, scale, ast)
        if _is_numeric(xa.kind):
            for k in range(len(lits)):
                lits[k] = self.rescale(lits[k], scale, ast)
        var node = self.mk(B_IN, BOOL.value, 0, a, -1, Int64(1 if has_null else 0), "", ast)
        if self.in_post:
            self.q.pexprs[node].items = lits^
        else:
            self.q.rexprs[node].items = lits^
        return node

    def value(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        if self.in_post:
            var cn = self.canon(ast)
            for k in range(len(self.key_canons)):
                if self.key_canons[k] == cn:
                    var key = self.q.keys[k].copy()
                    var ke = self.q.rexprs[key.expr].copy()
                    var n = self.mk(B_KEY, ke.kind, ke.scale, -1, -1, Int64(k), key.name, ast)
                    self.q.pexprs[n].src = ke.src
                    return n
        var kind = x.kind
        if kind == X_INT or kind == X_DEC or kind == X_FLT or kind == X_STR or kind == X_NULL or kind == X_TRUE or kind == X_FALSE or kind == X_DATE:
            return self.literal(ast)
        if kind == X_COL:
            if self.in_post:
                # a select item's alias (HAVING, ORDER BY); a plain column is only allowed as a group key
                for k in range(len(self.st.items)):
                    if self.st.items[k].has_alias and lower_ascii(self.st.items[k].alias) == lower_ascii(x.text) and x.text2.byte_length() == 0 \
                       and self.alias_expr[k] >= 0:
                        return self.alias_expr[k]
                var c = self.find_column(ast)
                if self.distinct:
                    raise self.err(ast, "for SELECT DISTINCT, ORDER BY expressions must appear in the select list: `" + self.src_names[c] + "` does not")
                raise self.err(ast, "column `" + self.src_names[c] + "` must appear in the GROUP BY clause or be used in an aggregate function")
            var c = self.find_column(ast)
            if c >= self.n_stored:
                return self.formula_column(c - self.n_stored, ast, 0)
            var n = self.mk(B_COL, self.src_kinds[c], self.src_scales[c], -1, -1, Int64(c), String(self.src_names[c]), ast)
            if self.src_kinds[c] == TEXT.value:
                self.q.rexprs[n].src = c
            return n
        if kind == X_NEG:
            var a = self.value(x.a)
            var xa = self.at(a)
            if xa.op == B_NULL:
                return a
            if not _is_num(xa.kind):
                raise self.err(ast, "cannot negate " + kind_name(xa.kind, xa.scale))
            if xa.op == B_LIT and xa.kind == FLOAT.value:
                return self.mk(B_LIT, FLOAT.value, xa.scale, -1, -1, Int64((-Float64(from_bits=UInt64(xa.value))).to_bits()), "-" + xa.text, ast)
            if xa.op == B_LIT:
                if xa.value == Int64.MIN:
                    raise self.err(ast, "the number passes the int64 range")
                return self.mk(B_LIT, xa.kind, xa.scale, -1, -1, -xa.value, "", ast)
            return self.mk(B_NEG, xa.kind, xa.scale, a, -1, 0, "", ast)
        if kind == X_BIN:
            var op = x.text
            if op == "AND" or op == "OR":
                var l = self.boolean(self.value(x.a), x.a, "`" + op + "`")
                var r = self.boolean(self.value(x.b), x.b, "`" + op + "`")
                return self.mk(B_AND if op == "AND" else B_OR, BOOL.value, 0, l, r, 0, "", ast)
            var a = self.value(x.a)
            var b = self.value(x.b)
            if op == "=" or op == "<>" or op == "<" or op == "<=" or op == ">" or op == ">=":
                return self.compare(C_EQ if op == "=" else C_NE if op == "<>" else C_LT if op == "<" else C_LE if op == "<=" else C_GT if op == ">" else C_GE,
                                    a, b, ast)
            return self.arith(op, a, b, ast)
        if kind == X_NOT:
            var a = self.boolean(self.value(x.a), x.a, "NOT")
            if self.at(a).op == B_NULL:
                return a
            return self.mk(B_NOT, BOOL.value, 0, a, -1, 0, "", ast)
        if kind == X_ISNULL:
            var a = self.value(x.a)
            var n = self.mk(B_ISNULL, BOOL.value, 0, a, -1, 0, "", ast)
            if x.flag:
                return self.mk(B_NOT, BOOL.value, 0, n, -1, 0, "", ast)
            return n
        if kind == X_BETWEEN:
            var v = self.value(x.a)
            var lo = self.value(x.b)
            var hi = self.value(x.c)
            var c1 = self.compare(C_GE, v, lo, ast)
            var c2 = self.compare(C_LE, v, hi, ast)
            var both = self.mk(B_AND, BOOL.value, 0, c1, c2, 0, "", ast)
            if x.flag:
                return self.mk(B_NOT, BOOL.value, 0, both, -1, 0, "", ast)
            return both
        if kind == X_IN:
            var n = self.in_list(ast)
            if x.flag and self.at(n).op != B_NULL:
                return self.mk(B_NOT, BOOL.value, 0, n, -1, 0, "", ast)
            return n
        # a function call
        if not _is_aggregate_name(x.text):
            raise self.unknown_function(ast)
        if not self.in_post:
            raise self.err(ast, "aggregate function " + x.text.lower() + "() is not allowed here: aggregates belong in the select list, HAVING and ORDER BY")
        return self.aggregate(ast)

    def unknown_function(self, ast: Int) -> Error:
        var name = self.st.exprs[ast].text
        var names = _aggregate_names()
        var hint = closest(name, names)
        if hint.byte_length() > 0:
            return self.err(ast, "function `" + name.lower() + "` is not supported (did you mean `" + hint + "`?)")
        return self.err(ast, "function `" + name.lower() + "` is not supported: SQL here has the aggregates " + ", ".join(names) +
                        "; scalar functions arrive with F-156")

    def aggregate(mut self, ast: Int) raises -> Int:
        var x = self.st.exprs[ast].copy()
        var name = x.text
        var func = -1
        var arg = -1
        var p = 0.0
        var nargs = len(x.items)
        if name == "COUNT":
            if x.a == 1:
                if x.flag:
                    raise self.err(ast, "COUNT(DISTINCT *) is not valid")
                func = F_COUNT_STAR
            else:
                if nargs != 1:
                    raise self.err(ast, "count() takes one argument or *, found " + String(nargs))
                func = F_COUNT_DISTINCT if x.flag else F_COUNT
        else:
            if x.a == 1:
                raise self.err(ast, name.lower() + "(*) is not valid: only COUNT takes *")
            if x.flag and name != "MIN" and name != "MAX":
                raise self.err(ast, name.lower() + "(DISTINCT ...) is not supported: only COUNT(DISTINCT ...) is")
            if name == "SUM":
                func = F_SUM
            elif name == "MIN":
                func = F_MIN
            elif name == "MAX":
                func = F_MAX
            elif name == "AVG" or name == "MEAN":
                func = F_AVG
            elif name == "MEDIAN":
                func = F_MEDIAN
            else:
                func = F_QUANTILE
            var want = 2 if func == F_QUANTILE else 1
            if nargs != want:
                raise self.err(ast, name.lower() + "() takes " + String(want) + (" arguments" if want != 1 else " argument") + ", found " + String(nargs))
        if func != F_COUNT_STAR:
            if _has_aggregate(self.st, x.items[0]):
                raise self.err(x.items[0], "aggregate functions cannot be nested")
            var was = self.in_post
            self.in_post = False
            arg = self.value(x.items[0])
            self.in_post = was
        if func == F_QUANTILE:
            var pe = x.items[1]
            if self.st.exprs[pe].kind != X_INT and self.st.exprs[pe].kind != X_DEC:
                raise self.err(pe, "the quantile of quantile_disc(x, p) must be a number between 0 and 1")
            var pn = parse_number(self.st.exprs[pe].text)
            if not pn.ok:
                raise self.err(pe, "the quantile must be a number between 0 and 1")
            p = Float64(Int(pn.value)) / Float64(Int(pow10_checked(pn.scale)))
            if p < 0.0 or p > 1.0:
                raise self.err(pe, "the quantile must be between 0 and 1, found " + self.st.exprs[pe].text)
        var rkind = INT.value
        var rscale = 0
        var src = -1
        if arg >= 0:
            var a = self.q.rexprs[arg].copy()
            if a.op == B_NULL:
                raise self.err(x.items[0], name.lower() + "(NULL) cannot be typed: write a column or a value")
            if a.kind == BOOL.value and a.op != B_COL and a.op != B_LIT:
                raise self.err(x.items[0], name.lower() + "() needs a value, found a comparison")
            if func == F_COUNT or func == F_COUNT_DISTINCT:
                if a.kind == TEXT.value and a.op != B_COL:
                    raise self.err(x.items[0], name.lower() + "() of TEXT needs the column itself")
            elif func == F_SUM:
                if a.kind == BOOL.value:                              # DuckDB sums a BOOLEAN as 0 / 1
                    rkind = INT.value
                elif not _is_num(a.kind):
                    raise self.err(x.items[0], "sum() needs numbers, found " + kind_name(a.kind, a.scale))
                else:
                    rkind = a.kind
                    rscale = 0 if a.kind == FLOAT.value else a.scale
            elif func == F_AVG:
                if not _is_num(a.kind):
                    raise self.err(x.items[0], "avg() needs numbers, found " + kind_name(a.kind, a.scale))
                rkind = FLOAT.value
            elif func == F_MIN or func == F_MAX or func == F_QUANTILE:
                rkind = a.kind
                rscale = a.scale
                src = a.src
                if a.kind == TEXT.value and a.op != B_COL:
                    raise self.err(x.items[0], name.lower() + "() of TEXT needs the column itself")
            elif func == F_MEDIAN and a.kind == TEXT.value:             # words have no mean: the lower middle word, in the text order, as DuckDB
                if a.op != B_COL:
                    raise self.err(x.items[0], "median() of TEXT needs the column itself")
                rkind = TEXT.value
                src = a.src
            elif func == F_MEDIAN and a.kind == DATE.value:
                raise self.err(x.items[0], "median() of a DATE is a TIMESTAMP in DuckDB (the middle of two dates can be half a day), which MX frame has no kind for yet (F-157): use quantile_disc(" +
                               self.st.render(x.items[0]) + ", 0.5)")
            elif func == F_MEDIAN:
                if not _is_num(a.kind):
                    raise self.err(x.items[0], "median() needs numbers, found " + kind_name(a.kind, a.scale))
                if a.kind == INT.value or a.kind == FLOAT.value:
                    rkind = FLOAT.value
                    rscale = a.scale if a.kind == FLOAT.value else 0       # a median of REAL is REAL
                else:
                    if a.scale + 1 > 18:
                        raise self.err(x.items[0], "median() of a DECIMAL with " + String(a.scale) + " places needs " + String(a.scale + 1) + ", past the 18 a value can hold")
                    rkind = DECIMAL.value
                    rscale = a.scale + 1
            # count(1), count('x'): a count of rows
            if func == F_COUNT and (a.op == B_LIT or a.op == B_STR):
                func = F_COUNT_STAR
                arg = -1
        var hol = func == F_COUNT_DISTINCT or func == F_MEDIAN or func == F_QUANTILE
        if (func == F_MIN or func == F_MAX) and rkind == TEXT.value:        # TEXT has no running min: the lowest / highest of its ranked values
            hol = True
            p = 0.0 if func == F_MIN else 1.0
        # dedupe: the same aggregate asked twice is computed once
        var canon = String(func) + ":" + ("*" if arg < 0 else self.canon(x.items[0])) + ":" + String(p)
        var shown: String
        if func == F_COUNT_STAR:
            shown = "count_star()"
        elif func == F_COUNT_DISTINCT:
            shown = "count(DISTINCT " + self.st.render(x.items[0]) + ")"
        elif func == F_QUANTILE:
            shown = "quantile_disc(" + self.st.render(x.items[0]) + ", " + self.st.render(x.items[1]) + ")"
        else:
            shown = (name.lower() if name != "MEAN" else "avg") + "(" + self.st.render(x.items[0]) + ")"
        var idx = -1
        for k in range(len(self.q.aggs)):
            if self.q.aggs[k].canon == canon:
                idx = k
                break
        if idx < 0:
            self.q.aggs.append(AggSpec(func, arg, p, rkind, rscale, shown, canon, hol))
            idx = len(self.q.aggs) - 1
        var n = self.mk(B_AGG, rkind, rscale, -1, -1, Int64(idx), shown, ast)
        self.q.pexprs[n].src = src
        return n


def _aggregate_names() -> List[String]:
    return ["sum", "count", "min", "max", "avg", "median", "quantile_disc"]


def _is_aggregate_name(up: String) -> Bool:
    return up == "SUM" or up == "COUNT" or up == "MIN" or up == "MAX" or up == "AVG" or up == "MEAN" or up == "MEDIAN" or up == "QUANTILE_DISC"


def _has_aggregate(st: SqlStatement, e: Int) -> Bool:
    ref x = st.exprs[e]
    if x.kind == X_CALL and _is_aggregate_name(x.text):
        return True
    if x.kind == X_CALL or x.kind == X_IN:
        for it in x.items:
            if _has_aggregate(st, it):
                return True
    if x.kind == X_IN or x.kind == X_BIN or x.kind == X_NEG or x.kind == X_NOT or x.kind == X_ISNULL or x.kind == X_BETWEEN:
        if x.a >= 0 and _has_aggregate(st, x.a):
            return True
        if x.b >= 0 and _has_aggregate(st, x.b):
            return True
        if x.c >= 0 and _has_aggregate(st, x.c):
            return True
    return False


def bind_sql(catalog: Catalog, text: String) raises -> BoundQuery:
    """Parses and binds one statement against the catalog."""
    var st = parse_sql(text)
    var src = catalog.find(st.table)
    if src < 0:
        var hint = closest(st.table, catalog.names())
        if st.table_is_file:
            raise sql_error(st.table_line, st.table_col, "the file '" + st.table + "' is not open: the `mx` command opens a file named in FROM; with the library, add it to the Catalog under that name")
        raise sql_error(st.table_line, st.table_col, "table `" + st.table + "` not found" +
                        ((" (did you mean `" + hint + "`?)") if hint.byte_length() > 0 else
                         (" (the catalog has " + ", ".join(catalog.names()) + ")") if len(catalog.sources) > 0 else " (the catalog is empty)"))
    var b = _Binder(st^)
    b.table_name = catalog.sources[src].name
    b.table_alias = b.st.table_alias
    for c in range(catalog.sources[src].table.columns()):
        b.src_names.append(catalog.sources[src].table.name(c))
        b.src_kinds.append(catalog.sources[src].table.kind(c).value)
        b.src_scales.append(catalog.sources[src].table.scale(c))
    b.n_stored = len(b.src_names)
    for f in catalog.sources[src].overlay.formulas:           # a frame's formula columns are columns too
        b.formulas.append(f.copy())
        b.src_names.append(f.name)
        b.src_kinds.append(FLOAT.value if f.scale < 0 else DECIMAL.value if f.scale > 0 else INT.value)
        b.src_scales.append(max(f.scale, 0))
    b.q.source = src
    b.q.text = text
    b.q.line = b.st.items[0].line
    b.q.col = b.st.items[0].col
    b.q.explain = b.st.explain
    b.q.limit = b.st.limit
    b.q.offset = b.st.offset
    for i in range(len(b.st.exprs)):                    # an unknown function is reported as that, before anything else
        if b.st.exprs[i].kind == X_CALL and not _is_aggregate_name(b.st.exprs[i].text):
            raise b.unknown_function(i)
    # F-152: `*` and `t.*` become the source's columns (a frame's formula columns too)
    var items = List[SelectItem]()
    for it in b.st.items:
        if not it.star:
            items.append(it.copy())
            continue
        if it.alias.byte_length() > 0:
            var qual = lower_ascii(it.alias)
            if qual != lower_ascii(b.table_name) and (b.table_alias.byte_length() == 0 or qual != lower_ascii(b.table_alias)):
                raise sql_error(it.line, it.col, "unknown table or alias `" + it.alias + "` in `" + it.alias + ".*`")
        for nm in b.src_names:
            b.st.exprs.append(SqlExpr(X_COL, nm, String(), -1, -1, -1, List[Int](), False, it.line, it.col))
            items.append(SelectItem(len(b.st.exprs) - 1, String(), False, it.line, it.col))
    b.st.items = items^
    if len(b.st.items) == 0:
        raise sql_error(1, 1, "the select list is empty")
    # an aggregate query: GROUP BY, or an aggregate somewhere
    var aggregating = len(b.st.group_by) > 0
    for it in b.st.items:
        if _has_aggregate(b.st, it.expr):
            aggregating = True
    if b.st.having >= 0:
        aggregating = True
    for o in b.st.order_by:
        if _has_aggregate(b.st, o.expr):
            aggregating = True
    if b.st.distinct:
        if aggregating:
            raise sql_error(b.st.items[0].line, b.st.items[0].col, "`SELECT DISTINCT` together with aggregates, GROUP BY or HAVING is not supported yet: " +
                            "use GROUP BY alone (it returns the same distinct rows), or DISTINCT without aggregates")
        # SELECT DISTINCT a, b is the group-by of the select list: the same machinery runs it
        for it in b.st.items:
            b.st.group_by.append(it.expr)
        aggregating = True
        b.q.distinct = True
        b.distinct = True
    if not aggregating:
        return bind_rows(b^)
    # WHERE
    if b.st.where >= 0:
        if _has_aggregate(b.st, b.st.where):
            raise b.err(b.st.where, "aggregate functions are not allowed in WHERE: filter the groups with HAVING")
        var w = b.value(b.st.where)
        b.q.where = b.boolean(w, b.st.where, "WHERE")
        b.q.where_text = b.st.render(b.st.where)
    # GROUP BY: ordinals and aliases name select items
    for g in range(len(b.st.group_by)):
        var ast = b.st.group_by[g]
        var gx = b.st.exprs[ast].copy()
        var target = ast
        if gx.kind == X_INT and not b.distinct:
            var n = Int(parse_number(gx.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(ast, "GROUP BY " + gx.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            target = b.st.items[n - 1].expr
            if _has_aggregate(b.st, target):
                raise b.err(ast, "GROUP BY " + gx.text + " refers to an aggregate")
        elif gx.kind == X_COL and gx.text2.byte_length() == 0 and not b.distinct:
            var is_column = False
            for nm in b.src_names:
                if lower_ascii(nm) == lower_ascii(gx.text):
                    is_column = True
            if not is_column:
                for it in b.st.items:
                    if it.has_alias and lower_ascii(it.alias) == lower_ascii(gx.text):
                        target = it.expr
                        break
        if _has_aggregate(b.st, target):
            raise b.err(ast, "aggregate functions are not allowed in GROUP BY")
        var cn = b.canon(target)
        var dup = False
        for k in b.key_canons:
            if k == cn:
                dup = True
        if dup:
            continue
        var e = b.value(target)
        var ke = b.q.rexprs[e].copy()
        if ke.op == B_NULL:
            raise b.err(target, "GROUP BY NULL cannot be typed")
        if ke.kind == BOOL.value and ke.op != B_COL:
            raise b.err(target, "grouping by a comparison is not supported yet: group by a column")
        b.q.keys.append(KeySpec(e, b.src_names[Int(ke.value)] if ke.op == B_COL else b.st.render(target), cn))
        b.key_canons.append(cn)
    # select items, HAVING, ORDER BY: after aggregation
    b.in_post = True
    for _ in range(len(b.st.items)):
        b.alias_expr.append(-1)
    for k in range(len(b.st.items)):
        var it = b.st.items[k].copy()
        var e = b.value(it.expr)
        var pe = b.q.pexprs[e].copy()
        if pe.op == B_NULL and pe.kind == K_NULL:
            raise b.err(it.expr, "a bare NULL in the select list cannot be typed")
        if pe.op == B_STR:
            raise b.err(it.expr, "a string constant in the select list is not supported yet")
        b.q.outputs.append(e)
        var nm: String
        if it.has_alias:
            nm = it.alias
        elif pe.op == B_KEY or pe.op == B_AGG:
            nm = pe.text
        else:
            nm = b.st.render(it.expr)
        b.q.out_names.append(nm)
        b.q.out_text.append(b.st.render(it.expr))
        if it.has_alias:
            b.alias_expr[k] = e
    if b.st.having >= 0:
        var h = b.value(b.st.having)
        b.q.having = b.boolean(h, b.st.having, "HAVING")
        b.q.having_text = b.st.render(b.st.having)
    for oi in b.st.order_by:
        var ox = b.st.exprs[oi.expr].copy()
        var e = -1
        if ox.kind == X_INT:
            var n = Int(parse_number(ox.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(oi.expr, "ORDER BY " + ox.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            e = b.q.outputs[n - 1]
        elif ox.kind == X_COL and ox.text2.byte_length() == 0:
            for k in range(len(b.st.items)):
                if b.st.items[k].has_alias and lower_ascii(b.st.items[k].alias) == lower_ascii(ox.text):
                    e = b.q.outputs[k]
                    break
        if e < 0:
            e = b.value(oi.expr)
        var pe = b.q.pexprs[e].copy()
        if pe.op == B_NULL:
            raise b.err(oi.expr, "ORDER BY a constant NULL does nothing: remove it")
        b.q.order.append(SortKey(e, oi.desc, oi.nulls == 1))
        b.q.order_text.append(b.st.render(oi.expr) + (" DESC" if oi.desc else "") + (" NULLS FIRST" if oi.nulls == 1 else ""))
    return b^.take()


def bind_rows(var b: _Binder) raises -> BoundQuery:
    """A query with no aggregate and no GROUP BY: it returns rows. The select list and ORDER BY are row-level expressions
    (`routs`, `rorder`); WHERE is bound as before."""
    b.q.rows = True
    if b.st.where >= 0:
        var w = b.value(b.st.where)
        b.q.where = b.boolean(w, b.st.where, "WHERE")
        b.q.where_text = b.st.render(b.st.where)
    for k in range(len(b.st.items)):
        var it = b.st.items[k].copy()
        var e = b.value(it.expr)
        var x = b.q.rexprs[e].copy()
        if x.op == B_NULL and x.kind == K_NULL:
            raise b.err(it.expr, "a bare NULL in the select list cannot be typed")
        if x.op == B_STR:
            raise b.err(it.expr, "a string constant in the select list is not supported yet")
        b.q.routs.append(e)
        var nm: String
        if it.has_alias:
            nm = it.alias
        elif x.op == B_COL:
            nm = x.text
        else:
            nm = b.st.render(it.expr)
        b.q.out_names.append(nm)
        b.q.out_text.append(b.st.render(it.expr))
    for oi in b.st.order_by:
        var ox = b.st.exprs[oi.expr].copy()
        var e = -1
        if ox.kind == X_INT:
            var n = Int(parse_number(ox.text).value)
            if n < 1 or n > len(b.st.items):
                raise b.err(oi.expr, "ORDER BY " + ox.text + " is out of range: the select list has " + String(len(b.st.items)) + " items")
            e = b.q.routs[n - 1]
        elif ox.kind == X_COL and ox.text2.byte_length() == 0:
            for k in range(len(b.st.items)):
                if b.st.items[k].has_alias and lower_ascii(b.st.items[k].alias) == lower_ascii(ox.text):
                    e = b.q.routs[k]
                    break
        if e < 0:
            e = b.value(oi.expr)
        var x = b.q.rexprs[e].copy()
        if x.op == B_NULL:
            raise b.err(oi.expr, "ORDER BY a constant NULL does nothing: remove it")
        b.q.rorder.append(SortKey(e, oi.desc, oi.nulls == 1))
        b.q.order_text.append(b.st.render(oi.expr) + (" DESC" if oi.desc else "") + (" NULLS FIRST" if oi.nulls == 1 else ""))
    return b^.take()
