"""SQL lowering mode (F-150): a bound query as one kernel program, with SQL's rules.

The builder's `_lower` (query.mojo) keeps the Query API's rules: a filter that reads a null cell is false as a whole,
a group is a dense number 0 .. groups-1. SQL needs more, so this module emits the same kernel instructions with its own
rules, on the same `Program` type:

- Three-valued logic. A boolean is a float register holding 1.0 (true), 0.0 (false) or 0.5 (unknown), so AND is the
  minimum, OR the maximum and NOT is 1 - x (Kleene), and WHERE keeps the rows that read exactly 1.0. A comparison is
  unknown only when an operand can be NULL: a column with no nulls compiles to plain 0.0 / 1.0 with no extra work.
- A value is an int64 register plus, when it can be NULL, a float register that is 1.0 where it is (`nul`). Arithmetic
  ORs the `nul` of its operands; `x // 0` and `x % 0` are NULL; `//` and `%` truncate toward zero as DuckDB does.
- A group key is a number in a mixed-radix layout with a NULL slot per nullable key (dense), or a list of key words
  (hashed): NULL is a group of its own.
- Every aggregate skips NULL: it is fed `SELECT(nul, identity, value)` and a count of the NULLs it skipped, so that
  SUM / MIN / MAX / AVG of nothing is NULL, which the finishing code reads off `rows - nulls`.
"""
from .table import TEXT, BOOL, FLOAT, SharedTable
from .parquet import ParquetScan
from .query import (Program, OP_LOAD_CONST, OP_LOADI, OP_IADD, OP_ISUB, OP_IMUL, OP_F2I, OP_AND, OP_OR, OP_NOT, OP_ISNULL,
                    OP_IDIV, OP_IMOD, OP_IMIN, OP_IMAX, OP_ICMP, OP_AGG_ISUM, OP_AGG_IMIN, OP_AGG_IMAX, MAX_REGS,
                    CTL_PLACEHOLDER, A_SUM, A_MIN, A_MAX, A_FSUM, A_FMIN, A_FMAX, _frame_mask, _null_mask,
                    OP_LOAD_COL, OP_ADD, OP_SUB, OP_MUL, OP_DIV, OP_AGG_SUM, OP_AGG_MIN, OP_AGG_MAX, NAN_AWARE)
from .frame import _text_less
from .bind import (BoundQuery, BExpr, B_DIV, B_I2F, B_FDIV, B_FMOD, B_LEAST, B_GREATEST, B_COL, B_LIT, B_NULL, B_ADD, B_SUB, B_MUL,
                   B_IDIV, B_IMOD, B_NEG, B_RESCALE, B_CMP, B_AND, B_OR, B_NOT, B_ISNULL, B_IN, C_LT, C_GE, C_LE, C_GT, C_EQ,
                   C_NE, F_SUM, F_MIN, F_MAX, F_AVG, pow10_checked)
from .planner import PhysicalPlan, G_DENSE, G_HASHED

comptime OP_SUBF = 11        # float x - y
comptime OP_I2F = 19
comptime OP_SELECT = 35
comptime OP_IN_SET = 37


struct DictAccess(Movable):
    """The words of a source's TEXT columns: a table's or a scan's dictionary, then the words a frame's session added."""
    var table: SharedTable
    var scan: Optional[ParquetScan]
    var extra: List[List[String]]

    def __init__(out self, table: SharedTable, scan: Optional[ParquetScan], extra: List[List[String]]):
        self.table = table
        self.scan = scan.copy()
        self.extra = extra.copy()

    def base_count(self, c: Int) raises -> Int:
        if self.scan:
            return self.scan.value().word_count(c)
        return len(self.table.dictionary(c))

    def count(self, c: Int) raises -> Int:
        return self.base_count(c) + (len(self.extra[c]) if c < len(self.extra) else 0)

    def word(self, c: Int, code: Int) raises -> String:
        var b = self.base_count(c)
        if code < b:
            return self.scan.value().text_of(c, code) if self.scan else self.table.text_of(c, code)
        return self.extra[c][code - b]

    def code_of(self, c: Int, w: String) raises -> Int:
        var k = self.scan.value().code_of(c, w) if self.scan else self.table.code_of(c, w)
        if k >= 0:
            return k
        if c < len(self.extra):
            for i in range(len(self.extra[c])):
                if self.extra[c][i] == w:
                    return self.base_count(c) + i
        return -1


struct V(Copyable, Movable):
    """An emitted expression: its register, the float register that is 1.0 where it is NULL (-1: never), and for a
    boolean whether it can hold the unknown 0.5; `fl`: the register holds a float64 (F-157), not int64 bits."""
    var reg: Int
    var nul: Int
    var unk: Bool
    var fl: Bool

    def __init__(out self, reg: Int, nul: Int, unk: Bool, fl: Bool = False):
        self.reg = reg
        self.nul = nul
        self.unk = unk
        self.fl = fl


struct SqlProgram(Movable):
    var p: Program
    var set_bytes: List[UInt8]
    var set_meta: List[Int64]
    var agg_val_slot: List[Int]      # per aggregate: its kernel slot, -1 for none
    var agg_nul_slot: List[Int]      # per aggregate: the slot counting the NULLs it skipped, -1 when its argument has none
    var agg_nan_slot: List[Int]      # per float min / max: the slot counting the NaNs it left out (NaN is above every number), else -1
    var agg_src: List[Int]           # per aggregate: its holistic value source, -1 for a plain one
    var slot_ops: List[Int]          # per slot: how partial results merge (A_SUM, A_MIN or A_MAX)
    var key_regs: Int                # registers 0 .. key_regs-1 hold the group key (dense: 1)
    var n_src: Int
    var guard_slot: Int              # dense with a footer-stated range: the extra group that rows outside it land in, else -1
    var instr: Int                   # instructions a row runs (predicate, body, aggregates)

    def __init__(out self, var p: Program):
        self.p = p^
        self.set_bytes = List[UInt8]()
        self.set_meta = List[Int64]()
        self.agg_val_slot = List[Int]()
        self.agg_nul_slot = List[Int]()
        self.agg_nan_slot = List[Int]()
        self.agg_src = List[Int]()
        self.slot_ops = List[Int]()
        self.key_regs = 1
        self.n_src = 0
        self.guard_slot = -1
        self.instr = 0


def bkey(rexprs: List[BExpr], e: Int) -> String:
    """A string that is equal for equal row-level expressions: to tell which aggregates read the same argument."""
    ref x = rexprs[e]
    var s = String(x.op) + ":" + String(x.value) + ":" + x.text
    if x.a >= 0:
        s += "(" + bkey(rexprs, x.a)
        if x.b >= 0:
            s += "," + bkey(rexprs, x.b)
        s += ")"
    for it in x.items:
        s += "," + bkey(rexprs, it)
    return s


def collect_columns(q: BoundQuery) -> List[Int]:
    """The source columns the query reads, ascending."""
    var seen = List[Int]()
    for e in q.rexprs:
        if e.op == B_COL:
            var c = Int(e.value)
            var at = len(seen)
            var dup = False
            for s in seen:
                if s == c:
                    dup = True
            if not dup:
                seen.append(c)
    # insertion sort (a handful of columns)
    for i in range(1, len(seen)):
        var j = i
        while j > 0 and seen[j - 1] > seen[j]:
            var t = seen[j]
            seen[j] = seen[j - 1]
            seen[j - 1] = t
            j -= 1
    return seen^


struct _Emitter(Movable):
    var p: Program
    var rx: List[BExpr]
    var nullable: List[Bool]
    var pos: List[Int]
    var aliases: List[List[Int]]     # per source column: from, to, from, to ... (a frame's merged TEXT codes)
    var live: List[Bool]
    var kept_next: Int
    var dicts: DictAccess
    var set_bytes: List[UInt8]
    var set_meta: List[Int64]

    def __init__(out self, var p: Program, var rx: List[BExpr], var nullable: List[Bool], var pos: List[Int],
                 var aliases: List[List[Int]], var dicts: DictAccess):
        self.p = p^
        self.rx = rx^
        self.nullable = nullable^
        self.pos = pos^
        self.aliases = aliases^
        self.live = List[Bool](length=MAX_REGS, fill=False)
        self.kept_next = 0
        self.dicts = dicts^
        self.set_bytes = List[UInt8]()
        self.set_meta = List[Int64]()

    def take_program(deinit self) -> Program:
        return self.p^

    # -- registers
    def tmp(mut self) raises -> Int:
        var r = MAX_REGS - 1
        while r >= self.kept_next:
            if not self.live[r]:
                self.live[r] = True
                return r
            r -= 1
        raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")

    def free(mut self, r: Int):
        if r >= self.kept_next and r < MAX_REGS:
            self.live[r] = False

    def reserve(mut self, n: Int):
        """Registers 0 .. n-1 are for the group key (and the control block): kept, never temporaries."""
        for i in range(n):
            self.live[i] = True
        self.kept_next = n

    def keep(mut self, r: Int) raises -> Int:
        """Copies register r into one that lives to the end of the program."""
        if self.kept_next >= MAX_REGS or self.live[self.kept_next]:
            raise Error("the query is too big for one program (more than " + String(MAX_REGS) + " registers)")
        var k = self.kept_next
        self.kept_next += 1
        self.live[k] = True
        self.p.emit(OP_IMAX, k, r, r)
        return k

    def ins(mut self, op: Int, d: Int, a: Int = 0, b: Int = 0, c: Int = 0):
        self.p.emit(op, d, a, b, c)

    def const_i(mut self, v: Int64) raises -> Int:
        var r = self.tmp()
        self.p.emit(OP_LOAD_CONST, r, self.p.const(v))
        return r

    def const_f(mut self, x: Float64) raises -> Int:
        var r = self.tmp()
        self.p.emit(OP_LOAD_CONST, r, self.p.const(Int64(x.to_bits())))
        return r

    def or_nul(mut self, a: Int, b: Int) -> Int:
        if a < 0:
            return b
        if b < 0:
            return a
        self.p.emit(OP_OR, a, a, b)
        self.free(b)
        return a

    # -- values
    def value(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        if x.op == B_COL:
            var c = Int(x.value)
            var r = self.tmp()
            if x.kind == FLOAT.value:
                self.p.has_float = True
                self.ins(OP_LOAD_COL, r, self.p.column_f64(self.pos[c]))
            else:
                self.ins(OP_LOADI, r, self.p.column(self.pos[c]))
            if x.kind == TEXT.value and len(self.aliases[c]) > 0:
                var k = 0
                while k < len(self.aliases[c]):
                    var from_c = self.const_i(Int64(self.aliases[c][k]))
                    var hit = self.tmp()
                    self.ins(OP_ICMP, hit, r, from_c, C_EQ)
                    var to_c = self.const_i(Int64(self.aliases[c][k + 1]))
                    self.ins(OP_SELECT, r, hit, to_c, r)
                    self.free(from_c)
                    self.free(hit)
                    self.free(to_c)
                    k += 2
            var nul = -1
            if self.nullable[c]:
                nul = self.tmp()
                self.ins(OP_ISNULL, nul, _null_mask(self.p, self.pos[c]))
            return V(r, nul, False, x.kind == FLOAT.value)
        if x.op == B_LIT:
            return V(self.const_i(x.value), -1, False, x.kind == FLOAT.value)       # (a float literal's value is its bits: the constant pool holds bits)
        if x.op == B_NULL:
            return V(self.const_i(0), self.const_f(1.0), False)
        if x.op == B_RESCALE:
            var a = self.value(x.a)
            var k = self.const_i(pow10_checked(Int(x.value)))
            self.ins(OP_IMUL, a.reg, a.reg, k)
            self.free(k)
            return a^
        if x.op == B_I2F:                                      # an INT or DECIMAL read as a float: float(v) / 10^scale, as DuckDB casts it
            var a = self.value(x.a)
            self.p.has_float = True
            self.ins(OP_I2F, a.reg, a.reg)
            if x.value > 0:
                var k = self.const_f(Float64(Int(pow10_checked(Int(x.value)))))
                self.ins(OP_DIV, a.reg, a.reg, k)
                self.free(k)
            return V(a.reg, a.nul, False, True)
        if x.op == B_NEG:
            var a = self.value(x.a)
            if a.fl:                                           # times -1.0: the sign of zero, inf and NaN flips too
                var m1 = self.const_f(-1.0)
                self.ins(OP_MUL, a.reg, a.reg, m1)
                self.free(m1)
                return a^
            var z = self.const_i(0)
            self.ins(OP_ISUB, a.reg, z, a.reg)
            self.free(z)
            return a^
        if x.op == B_ADD or x.op == B_SUB or x.op == B_MUL or x.op == B_DIV:
            var a = self.value(x.a)
            var b = self.value(x.b)
            if x.kind == FLOAT.value:
                self.ins(OP_ADD if x.op == B_ADD else OP_SUB if x.op == B_SUB else OP_MUL if x.op == B_MUL else OP_DIV, a.reg, a.reg, b.reg)
                self.free(b.reg)
                return V(a.reg, self.or_nul(a.nul, b.nul), False, True)
            self.ins(OP_IADD if x.op == B_ADD else OP_ISUB if x.op == B_SUB else OP_IMUL, a.reg, a.reg, b.reg)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False)
        if x.op == B_IDIV or x.op == B_IMOD:
            return self.divmod(x)
        if (x.op == B_LEAST or x.op == B_GREATEST) and x.kind == FLOAT.value:         # NaN is above every number: least skips it, greatest is it
            var a = self.value(x.a)
            var b = self.value(x.b)
            var pick = self.tmp()
            self.ins(C_LT if x.op == B_LEAST else C_GT, pick, a.reg, b.reg, NAN_AWARE)
            self.ins(OP_SELECT, a.reg, pick, a.reg, b.reg)
            self.free(pick)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False, True)
        if x.op == B_FDIV or x.op == B_FMOD or x.op == B_LEAST or x.op == B_GREATEST:      # a frame formula's operators: the kernel's own
            var a = self.value(x.a)
            var b = self.value(x.b)
            self.ins(OP_IDIV if x.op == B_FDIV else OP_IMOD if x.op == B_FMOD else OP_IMIN if x.op == B_LEAST else OP_IMAX, a.reg, a.reg, b.reg)
            self.free(b.reg)
            return V(a.reg, self.or_nul(a.nul, b.nul), False)
        raise Error("internal: expression kind " + String(x.op) + " is not a row-level value")

    def divmod(mut self, x: BExpr) raises -> V:
        """a // b or a % b truncating toward zero (DuckDB's), from the kernel's floor division: the quotient is one more
        than the floored one when the remainder is not zero and the operands' signs differ. b = 0 gives NULL."""
        var a = self.value(x.a)
        var b = self.value(x.b)
        var zero = self.const_i(0)
        var q = self.tmp()
        self.ins(OP_IDIV, q, a.reg, b.reg)
        var t = self.tmp()
        self.ins(OP_IMUL, t, q, b.reg)
        self.ins(OP_ISUB, t, a.reg, t)                            # t: the floored remainder
        var rz = self.tmp()
        self.ins(OP_ICMP, rz, t, zero, C_NE)
        var na = self.tmp()
        self.ins(OP_ICMP, na, a.reg, zero, C_LT)
        var nb = self.tmp()
        self.ins(OP_ICMP, nb, b.reg, zero, C_LT)
        self.ins(C_NE, na, na, nb)                                # float compare: the signs differ
        self.ins(OP_AND, rz, rz, na)
        self.ins(OP_F2I, rz, rz)
        self.ins(OP_IADD, q, q, rz)                               # the truncated quotient
        var res = q
        if x.op == B_IMOD:
            self.ins(OP_IMUL, t, q, b.reg)
            self.ins(OP_ISUB, q, a.reg, t)
        var dz = self.tmp()
        self.ins(OP_ICMP, dz, b.reg, zero, C_EQ)
        var nul = self.or_nul(a.nul, b.nul)
        nul = self.or_nul(nul, dz)
        self.free(a.reg)
        self.free(b.reg)
        self.free(zero)
        self.free(t)
        self.free(rz)
        self.free(na)
        self.free(nb)
        return V(res, nul, False)

    # -- booleans (Kleene: 1.0 true, 0.0 false, 0.5 unknown)
    def with_unknown(mut self, r: Int, nul: Int) raises -> V:
        """r (a definite boolean) where nul is 0, unknown where it is 1."""
        if nul < 0:
            return V(r, -1, False)
        var h = self.const_f(0.5)
        self.ins(OP_SELECT, r, nul, h, r)
        self.free(h)
        self.free(nul)
        return V(r, -1, True)

    def boolean(mut self, e: Int) raises -> V:
        var x = self.rx[e].copy()
        if x.op == B_CMP:
            return self.compare(x)
        if x.op == B_AND or x.op == B_OR:
            var a = self.boolean(x.a)
            var b = self.boolean(x.b)
            if not a.unk and not b.unk:
                self.ins(OP_AND if x.op == B_AND else OP_OR, a.reg, a.reg, b.reg)
            else:                                                 # the minimum / the maximum
                var t = self.tmp()
                self.ins(C_LT if x.op == B_AND else C_GT, t, a.reg, b.reg)
                self.ins(OP_SELECT, a.reg, t, a.reg, b.reg)
                self.free(t)
            self.free(b.reg)
            return V(a.reg, -1, a.unk or b.unk)
        if x.op == B_NOT:
            var a = self.boolean(x.a)
            if not a.unk:
                self.ins(OP_NOT, a.reg, a.reg)
            else:
                var one = self.const_f(1.0)
                self.ins(OP_SUBF, a.reg, one, a.reg)
                self.free(one)
            return a^
        if x.op == B_ISNULL:
            var inner = self.rx[x.a].copy()
            if inner.kind == BOOL.value and inner.op != B_COL and inner.op != B_LIT:       # a boolean expression: unknown is its NULL
                var a = self.boolean(x.a)
                var h = self.const_f(0.5)
                self.ins(C_EQ, a.reg, a.reg, h)
                self.free(h)
                return V(a.reg, -1, False)
            var v = self.value(x.a)
            self.free(v.reg)
            if v.nul >= 0:
                return V(v.nul, -1, False)
            return V(self.const_f(0.0), -1, False)
        if x.op == B_IN:
            return self.in_list(x)
        if x.op == B_NULL:
            return V(self.const_f(0.5), -1, True)
        if x.op == B_LIT:
            return V(self.const_f(1.0 if x.value != 0 else 0.0), -1, False)
        if x.op == B_COL:                                          # a BOOL column used as a condition
            var v = self.value(e)
            var r = self.tmp()
            self.ins(OP_I2F, r, v.reg)
            self.free(v.reg)
            return self.with_unknown(r, v.nul)
        raise Error("internal: expression kind " + String(x.op) + " is not a condition")

    def compare(mut self, x: BExpr) raises -> V:
        var lhs = self.rx[x.a].copy()
        if lhs.kind == TEXT.value:
            return self.text_compare(x)
        var a = self.value(x.a)
        var b = self.value(x.b)
        var r = a.reg
        if a.fl:                                               # DuckDB's order: NaN equals itself and is above every number; -0.0 equals 0.0
            self.ins(Int(x.value), r, a.reg, b.reg, NAN_AWARE)
        else:
            self.ins(OP_ICMP, r, a.reg, b.reg, Int(x.value))
        self.free(b.reg)
        return self.with_unknown(r, self.or_nul(a.nul, b.nul))

    def text_compare(mut self, x: BExpr) raises -> V:
        """A TEXT column against a string literal: the codes of the words that satisfy it, in the dictionary, as a set."""
        var col = self.rx[x.a].copy()
        var lit = self.rx[x.b].text
        var op = Int(x.value)
        var v = self.value(x.a)
        var c = Int(col.value)
        var r = v.reg
        if op == C_EQ or op == C_NE:
            var code = self.dicts.code_of(c, lit)
            var k = self.const_i(Int64(code))                       # -1: no row holds the word
            self.ins(OP_ICMP, r, r, k, op)
            self.free(k)
        else:
            var n = self.dicts.count(c)
            var bytes = List[UInt8](length=max(n, 1), fill=0)
            for code in range(n):
                var w = self.dicts.word(c, code)
                var hit: Bool
                if op == C_LT:
                    hit = _text_less(w, lit)
                elif op == C_GT:
                    hit = _text_less(lit, w)
                elif op == C_LE:
                    hit = not _text_less(lit, w)
                else:
                    hit = not _text_less(w, lit)
                bytes[code] = 1 if hit else 0
            var f = self.tmp()
            self.ins(OP_I2F, f, r)
            self.free(r)
            r = f
            var set_at = self.add_set(bytes, 0)
            self.ins(OP_IN_SET, r, r, set_at)
        return self.with_unknown(r, v.nul)

    def add_set(mut self, bytes: List[UInt8], lo: Int) -> Int:
        var off = len(self.set_bytes)
        self.set_bytes.extend(bytes.copy())
        self.set_meta.append(Int64(off))
        self.set_meta.append(Int64(len(bytes)))
        self.set_meta.append(Int64(lo))
        return len(self.set_meta) // 3 - 1

    def in_list(mut self, x: BExpr) raises -> V:
        var lhs = self.rx[x.a].copy()
        var v = self.value(x.a)
        var r: Int
        if lhs.kind == TEXT.value:
            var c = Int(lhs.value)
            var n = self.dicts.count(c)
            var bytes = List[UInt8](length=max(n, 1), fill=0)
            for it in x.items:
                var code = self.dicts.code_of(c, self.rx[it].text)
                if code >= 0:
                    bytes[code] = 1
            r = self.tmp()
            self.ins(OP_I2F, r, v.reg)
            self.free(v.reg)
            var set_at = self.add_set(bytes, 0)
            self.ins(OP_IN_SET, r, r, set_at)
        else:
            if len(x.items) > 40:
                self.free(v.reg)
                raise Error("an IN list of more than 40 numbers is not supported yet")
            r = self.tmp()
            if len(x.items) == 0:
                self.ins(OP_LOAD_CONST, r, self.p.const(Int64(0)))
            for i in range(len(x.items)):
                var k = self.const_i(self.rx[x.items[i]].value)
                if i == 0:
                    if v.fl:
                        self.ins(C_EQ, r, v.reg, k, NAN_AWARE)
                    else:
                        self.ins(OP_ICMP, r, v.reg, k, C_EQ)
                else:
                    var t = self.tmp()
                    if v.fl:
                        self.ins(C_EQ, t, v.reg, k, NAN_AWARE)
                    else:
                        self.ins(OP_ICMP, t, v.reg, k, C_EQ)
                    self.ins(OP_OR, r, r, t)
                    self.free(t)
                self.free(k)
            self.free(v.reg)
        if x.value != 0:                                           # the list holds a NULL: a miss is unknown, not false
            var one = self.const_f(1.0)
            var h = self.const_f(0.5)
            self.ins(OP_SELECT, r, r, one, h)
            self.free(one)
            self.free(h)
            if v.nul >= 0:
                var hh = self.const_f(0.5)
                self.ins(OP_SELECT, r, v.nul, hh, r)
                self.free(hh)
                self.free(v.nul)
            return V(r, -1, True)
        return self.with_unknown(r, v.nul)


comptime PHASE_ALL = 0       # the plain aggregates and the holistic ones in one program
comptime PHASE_PLAIN = 1     # hashed groups, pass 1: the plain aggregates, the key words out
comptime PHASE_HOL = 2       # hashed groups, pass 2: only the holistic sources, each row mapped to its group by the hook


def lower_sql(q: BoundQuery, plan: PhysicalPlan, var nullable: List[Bool], var pos: List[Int], nchunk_cols: Int,
              var dicts: DictAccess, var aliases: List[List[Int]], masked: Bool, phase: Int = PHASE_ALL) raises -> SqlProgram:
    """The program for `q` under `plan`. `nullable[c]`: source column c may hold NULLs; `pos[c]`: its place in the chunk
    tables the program runs on; `aliases[c]`: a frame's merged TEXT codes; `masked`: a frame mask hides changed rows."""
    var e = _Emitter(Program(nchunk_cols), q.rexprs.copy(), nullable^, pos^, aliases^, dicts^)
    var sp_aggs = q.aggs.copy()
    # which aggregates are holistic: their values go to the kernel hook, one source per distinct argument
    var holistic = False
    for a in sp_aggs:
        if a.hol:
            holistic = True
    if phase == PHASE_PLAIN:
        holistic = False
    var nkw = 1 if plan.group != G_HASHED else plan.key_words()
    var key_regs = nkw
    var key_base = 0                                              # the first register of the key
    var ctl_reg = 1                                               # the register holding the control block of a holistic query
    if phase == PHASE_HOL:                                        # hashed, pass 2: the control block first, then the key words
        key_base = 1
        ctl_reg = 0
    var out = SqlProgram(Program(0))
    e.p.mat = holistic
    var guarded = False                                           # a dense key whose range a file's footer stated: one extra slot catches rows outside it
    if plan.group == G_DENSE:
        for lay in plan.keys:
            guarded = guarded or lay.guarded
    e.p.groups = (plan.domain + (1 if guarded else 0)) if plan.group == G_DENSE else 1
    out.guard_slot = plan.domain if guarded else -1
    # registers: the key (dense: register 0, and register 1 the control block of a holistic query) or the key words
    e.reserve(key_regs + (1 if holistic else 0))
    e.p.section = 1
    if holistic:
        e.p.ctl_const = len(e.p.consts)
        e.p.consts.append(CTL_PLACEHOLDER)
        var cr = e.tmp()
        e.ins(OP_LOAD_CONST, cr, e.p.ctl_const)
        e.ins(OP_IMAX, ctl_reg, cr, cr)
        e.free(cr)
    # the predicate: the WHERE (unknown is not true), then the frame mask
    e.p.section = 0
    var pred = -1
    if q.where >= 0:
        var w = e.boolean(q.where)
        if w.unk:
            var one = e.const_f(1.0)
            e.ins(24, w.reg, w.reg, one)                         # == 1.0: unknown and false both drop the row
            e.free(one)
        pred = w.reg
    if masked:
        var isnull = e.tmp()
        e.ins(OP_ISNULL, isnull, _frame_mask(e.p))
        e.ins(OP_NOT, isnull, isnull)
        if pred >= 0:
            e.ins(OP_AND, pred, pred, isnull)
            e.free(isnull)
        else:
            pred = isnull
    if pred >= 0:
        var k = e.keep(pred)
        e.free(pred)
        pred = k
    e.p.pred_reg = pred
    # the group key
    e.p.section = 1
    if plan.group == G_DENSE:
        e.p.keyed = True
        var acc = -1
        var anybad = -1
        for k in range(len(plan.keys)):
            ref lay = plan.keys[k]
            var v = e.value(lay.expr)
            if lay.kind == TEXT.value or lay.kind == BOOL.value:
                if lay.nullable:
                    var one = e.const_i(1)
                    e.ins(OP_IADD, v.reg, v.reg, one)
                    e.free(one)
            else:
                var lo = e.const_i(lay.lo)
                e.ins(OP_ISUB, v.reg, v.reg, lo)
                e.free(lo)
                if lay.guarded:                                   # a value below the range or past it is flagged (NULL is not a value)
                    var zero = e.const_i(0)
                    var bad = e.tmp()
                    e.ins(OP_ICMP, bad, v.reg, zero, C_LT)
                    e.free(zero)
                    var span = e.const_i(Int64(lay.radix - (1 if lay.nullable else 0)))
                    var past = e.tmp()
                    e.ins(OP_ICMP, past, v.reg, span, C_GE)
                    e.free(span)
                    e.ins(OP_OR, bad, bad, past)
                    e.free(past)
                    if lay.nullable:
                        var f0 = e.const_f(0.0)
                        e.ins(OP_SELECT, bad, v.nul, f0, bad)
                        e.free(f0)
                    if anybad < 0:
                        anybad = bad
                    else:
                        e.ins(OP_OR, anybad, anybad, bad)
                        e.free(bad)
                if lay.nullable:
                    var one = e.const_i(1)
                    e.ins(OP_IADD, v.reg, v.reg, one)
                    e.free(one)
            if lay.nullable:
                var z = e.const_i(0)
                e.ins(OP_SELECT, v.reg, v.nul, z, v.reg)         # NULL is slot 0
                e.free(z)
                e.free(v.nul)
            if acc < 0:
                acc = v.reg
            else:
                var rd = e.const_i(Int64(lay.radix))
                e.ins(OP_IMUL, acc, acc, rd)
                e.ins(OP_IADD, acc, acc, v.reg)
                e.free(rd)
                e.free(v.reg)
        if anybad >= 0:
            var dom = e.const_i(Int64(plan.domain))
            e.ins(OP_SELECT, acc, anybad, dom, acc)
            e.free(dom)
            e.free(anybad)
        e.ins(OP_IMAX, 0, acc, acc)
        e.free(acc)
    elif plan.group == G_HASHED:
        e.p.keyed = True
        var at = key_base
        for k in range(len(plan.keys)):
            ref lay = plan.keys[k]
            var v = e.value(lay.expr)
            if v.fl:                                                  # -0.0 is 0.0 and every NaN is one value (F-077): one key word per value
                var zero = e.const_f(0.0)
                e.ins(OP_ADD, v.reg, v.reg, zero)                     # -0.0 + 0.0 is +0.0
                e.free(zero)
                var nanf = e.tmp()
                e.ins(25, nanf, v.reg, v.reg, 0)
                var canon = e.const_f(Float64(from_bits=UInt64(0x7FF8000000000000)))
                e.ins(OP_SELECT, v.reg, nanf, canon, v.reg)
                e.free(canon)
                e.free(nanf)
            if lay.nullable:
                var z = e.const_i(0)
                var ni = e.tmp()
                e.ins(OP_F2I, ni, v.nul)
                e.ins(OP_SELECT, v.reg, v.nul, z, v.reg)         # NULL's value word is 0, its flag word 1
                e.free(z)
                e.free(v.nul)
                e.ins(OP_IMAX, at, v.reg, v.reg)
                e.ins(OP_IMAX, at + 1, ni, ni)
                e.free(ni)
                at += 2
            else:
                e.ins(OP_IMAX, at, v.reg, v.reg)
                at += 1
            e.free(v.reg)
    elif holistic:
        e.p.keyed = True
        e.ins(OP_LOAD_CONST, 0, e.p.const(0))
    # the aggregates, one pass per distinct argument
    var n_aggs = len(sp_aggs)
    for _ in range(n_aggs):
        out.agg_val_slot.append(-1)
        out.agg_nul_slot.append(-1)
        out.agg_nan_slot.append(-1)
        out.agg_src.append(-1)
    var done = List[Bool](length=n_aggs, fill=False)
    var n_slots = 0
    var n_src = 0
    var src_vreg = List[Int]()
    var src_valid = List[Int]()
    var src_float = List[Bool]()
    for i in range(n_aggs):
        if (phase == PHASE_PLAIN and sp_aggs[i].hol) or (phase == PHASE_HOL and not sp_aggs[i].hol):
            done[i] = True                                        # the other pass computes it
        if done[i] or sp_aggs[i].arg < 0:
            continue
        var key = bkey(e.rx, sp_aggs[i].arg)
        var members = List[Int]()
        for j in range(i, n_aggs):
            var other_pass = (phase == PHASE_PLAIN and sp_aggs[j].hol) or (phase == PHASE_HOL and not sp_aggs[j].hol)
            if not done[j] and not other_pass and sp_aggs[j].arg >= 0 and bkey(e.rx, sp_aggs[j].arg) == key:
                members.append(j)
                done[j] = True
        var need_sum = False
        var need_min = False
        var need_max = False
        var need_hol = False
        for j in members:
            var f = sp_aggs[j].func
            var h = sp_aggs[j].hol
            need_sum = need_sum or ((f == F_SUM or f == F_AVG) and not h)
            need_min = need_min or (f == F_MIN and not h)
            need_max = need_max or (f == F_MAX and not h)
            need_hol = need_hol or h
        var v: V
        if not (need_sum or need_min or need_max or need_hol) and e.rx[sp_aggs[i].arg].op == B_COL:       # count(col): its NULL flags only
            var c = Int(e.rx[sp_aggs[i].arg].value)
            var nul = -1
            if e.nullable[c]:
                nul = e.tmp()
                e.ins(OP_ISNULL, nul, _null_mask(e.p, e.pos[c]))
            v = V(-1, nul, False)
        else:
            v = e.value(sp_aggs[i].arg)
        var raw_kept = -1
        var sum_slot = -1
        var min_slot = -1
        var max_slot = -1
        var nul_slot = -1
        if need_sum:
            var m = v.reg
            if v.nul >= 0:
                var z = e.const_i(0)
                m = e.tmp()
                e.ins(OP_SELECT, m, v.nul, z, v.reg)
                e.free(z)
            var kept = e.keep(m)
            if v.nul < 0:
                raw_kept = kept
            else:
                e.free(m)
            sum_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_SUM if v.fl else OP_AGG_ISUM, sum_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FSUM if v.fl else A_SUM)
            if v.fl:
                e.p.repro = True                                      # a float sum has one defined order (F-064)
            n_slots += 1
        var isn = -1                                                  # float min / max: 1.0 where the value is NaN
        if v.fl and (need_min or need_max):
            isn = e.tmp()
            e.ins(25, isn, v.reg, v.reg, 0)                           # != (IEEE): true only for NaN
        if need_min and v.fl:
            var ident = e.const_f(Float64(from_bits=UInt64(0x7FF0000000000000)))
            var m = e.tmp()
            e.ins(OP_SELECT, m, isn, ident, v.reg)                    # a NaN is left out: it is above every number
            if v.nul >= 0:
                e.ins(OP_SELECT, m, v.nul, ident, m)
            var kept = e.keep(m)
            e.free(m)
            e.free(ident)
            min_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_MIN, min_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FMIN)
            n_slots += 1
        elif need_min:
            var kept: Int
            if v.nul < 0 and raw_kept >= 0:
                kept = raw_kept
            else:
                var m = v.reg
                if v.nul >= 0:
                    var id = e.const_i(Int64.MAX)
                    m = e.tmp()
                    e.ins(OP_SELECT, m, v.nul, id, v.reg)
                    e.free(id)
                kept = e.keep(m)
                if v.nul >= 0:
                    e.free(m)
                else:
                    raw_kept = kept
            min_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_IMIN, min_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_MIN)
            n_slots += 1
        if need_max and v.fl:
            var ident = e.const_f(Float64(from_bits=UInt64(0xFFF0000000000000)))
            var m = e.tmp()
            e.ins(OP_SELECT, m, isn, ident, v.reg)
            if v.nul >= 0:
                e.ins(OP_SELECT, m, v.nul, ident, m)
            var kept = e.keep(m)
            e.free(m)
            e.free(ident)
            max_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_MAX, max_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_FMAX)
            n_slots += 1
        elif need_max:
            var kept: Int
            if v.nul < 0 and raw_kept >= 0:
                kept = raw_kept
            else:
                var m = v.reg
                if v.nul >= 0:
                    var id = e.const_i(Int64.MIN)
                    m = e.tmp()
                    e.ins(OP_SELECT, m, v.nul, id, v.reg)
                    e.free(id)
                kept = e.keep(m)
                if v.nul >= 0:
                    e.free(m)
                else:
                    raw_kept = kept
            max_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_IMAX, max_slot, kept)
            e.p.section = 1
            out.slot_ops.append(A_MAX)
            n_slots += 1
        var nan_slot = -1
        if isn >= 0 and phase != PHASE_HOL:                           # the NaNs a float min / max left out (a NaN is a value: MAX is NaN when there is one)
            var nf = isn
            if v.nul >= 0:
                nf = e.tmp()
                var zf = e.const_i(0)
                e.ins(OP_SELECT, nf, v.nul, zf, isn)                  # a NULL row is not a NaN
                e.free(zf)
            var ni2 = e.tmp()
            e.ins(OP_F2I, ni2, nf)
            var kept_nan = e.keep(ni2)
            e.free(ni2)
            if nf != isn:
                e.free(nf)
            nan_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_ISUM, nan_slot, kept_nan)
            e.p.section = 1
            out.slot_ops.append(A_SUM)
            n_slots += 1
        e.free(isn)
        if v.nul >= 0 and phase != PHASE_HOL:                     # the NULLs this argument skipped
            var ni = e.tmp()
            e.ins(OP_F2I, ni, v.nul)
            var kept_n = e.keep(ni)
            e.free(ni)
            nul_slot = n_slots
            e.p.section = 2
            e.ins(OP_AGG_ISUM, nul_slot, kept_n)
            e.p.section = 1
            out.slot_ops.append(A_SUM)
            n_slots += 1
        var src = -1
        if need_hol:
            src = n_src
            n_src += 1
            var vr = v.reg
            var kept_v = raw_kept if (v.nul < 0 and raw_kept >= 0) else e.keep(vr)
            var kept_valid = -1
            if v.nul >= 0:
                var nv = e.tmp()
                e.ins(OP_NOT, nv, v.nul)
                e.ins(OP_F2I, nv, nv)
                kept_valid = e.keep(nv)
                e.free(nv)
            src_vreg.append(kept_v)
            src_valid.append(kept_valid)
            src_float.append(v.fl)
        for j in members:
            var f = sp_aggs[j].func
            out.agg_nul_slot[j] = nul_slot
            out.agg_nan_slot[j] = nan_slot
            if sp_aggs[j].hol:
                out.agg_src[j] = src
            elif f == F_SUM or f == F_AVG:
                out.agg_val_slot[j] = sum_slot
            elif f == F_MIN:
                out.agg_val_slot[j] = min_slot
            elif f == F_MAX:
                out.agg_val_slot[j] = max_slot
            elif sp_aggs[j].hol:
                out.agg_src[j] = src
        e.free(v.reg)
        e.free(v.nul)
    if n_slots == 0:                                              # only count(*): one dummy sum keeps the kernel happy
        var z = e.const_i(0)
        var kz = e.keep(z)
        e.free(z)
        e.p.section = 2
        e.ins(OP_AGG_ISUM, 0, kz)
        out.slot_ops.append(A_SUM)
        n_slots = 1
    e.p.n_slots = n_slots
    e.p.n_src = n_src
    e.p.src_vreg = src_vreg^
    e.p.src_valid = src_valid^
    e.p.src_float = src_float^
    out.n_src = n_src
    out.key_regs = key_regs
    out.set_bytes = e.set_bytes.copy()
    out.set_meta = e.set_meta.copy()
    out.instr = (len(e.p.pred) + len(e.p.body) + len(e.p.aggs)) // 5
    out.p = e^.take_program()
    out.p.set_bytes = out.set_bytes.copy()
    out.p.set_meta = out.set_meta.copy()
    return out^
