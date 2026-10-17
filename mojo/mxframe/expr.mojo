"""Column expressions (docs/mxframe-mojo-api.md sections 4 and 5).

An `Expr` is a tree kept flat in post-order (operands before their operator), the order a stack machine and the
kernels' register programs both want. Values are exact Int64 (every column stores Int64, see table.mojo);
comparisons and `&` / `|` / `~` give booleans, used by filters.

    col("revenue") - col("cost")            # arithmetic: + - * // %, unary -, least / greatest
    col("day") // 7                          # a bin
    (col("region") == 2) & col("day").between(0, 364)
"""

comptime N_COL = 1
comptime N_LIT = 2
comptime N_ADD = 10
comptime N_SUB = 11
comptime N_MUL = 12
comptime N_FLOORDIV = 13
comptime N_MOD = 14
comptime N_NEG = 15
comptime N_LEAST = 16
comptime N_GREATEST = 17
comptime N_LT = 20
comptime N_LE = 21
comptime N_GT = 22
comptime N_GE = 23
comptime N_EQ = 24
comptime N_NE = 25
comptime N_AND = 30
comptime N_OR = 31
comptime N_NOT = 32


@fieldwise_init
struct Node(Copyable, Movable):
    var op: Int
    var value: Int64        # N_LIT: the literal
    var name: String        # N_COL: the column's name

    def arity(self) -> Int:
        if self.op == N_COL or self.op == N_LIT:
            return 0
        if self.op == N_NEG or self.op == N_NOT:
            return 1
        return 2

    def is_bool(self) -> Bool:
        return self.op >= N_LT


struct Expr(Copyable, Movable):
    var nodes: List[Node]

    def __init__(out self, var nodes: List[Node]):
        self.nodes = nodes^

    @implicit
    def __init__(out self, value: Int):
        """An integer literal (so `col("a") + 1` works)."""
        self.nodes = [Node(N_LIT, Int64(value), "")]

    def _bin(self, op: Int, other: Expr) -> Expr:
        var nodes = self.nodes.copy()
        for n in other.nodes:
            nodes.append(n.copy())
        nodes.append(Node(op, 0, ""))
        return Expr(nodes^)

    def _un(self, op: Int) -> Expr:
        var nodes = self.nodes.copy()
        nodes.append(Node(op, 0, ""))
        return Expr(nodes^)

    def __add__(self, other: Expr) -> Expr:
        return self._bin(N_ADD, other)

    def __sub__(self, other: Expr) -> Expr:
        return self._bin(N_SUB, other)

    def __mul__(self, other: Expr) -> Expr:
        return self._bin(N_MUL, other)

    def __floordiv__(self, other: Expr) -> Expr:
        """Floor division, as Mojo's Int64 (x // 0 is 0, as in the app's formulas)."""
        return self._bin(N_FLOORDIV, other)

    def __mod__(self, other: Expr) -> Expr:
        """The remainder with the divisor's sign, as Mojo's Int64 (x % 0 is 0)."""
        return self._bin(N_MOD, other)

    def __neg__(self) -> Expr:
        return self._un(N_NEG)

    def __lt__(self, other: Expr) -> Expr:
        return self._bin(N_LT, other)

    def __le__(self, other: Expr) -> Expr:
        return self._bin(N_LE, other)

    def __gt__(self, other: Expr) -> Expr:
        return self._bin(N_GT, other)

    def __ge__(self, other: Expr) -> Expr:
        return self._bin(N_GE, other)

    def __eq__(self, other: Expr) -> Expr:
        return self._bin(N_EQ, other)

    def __ne__(self, other: Expr) -> Expr:
        return self._bin(N_NE, other)

    def __and__(self, other: Expr) -> Expr:
        return self._bin(N_AND, other)

    def __or__(self, other: Expr) -> Expr:
        return self._bin(N_OR, other)

    def __invert__(self) -> Expr:
        return self._un(N_NOT)

    def between(self, lo: Int, hi: Int) -> Expr:
        """lo <= self <= hi (inclusive, the app's range filter)."""
        return (self >= Expr(lo)) & (self <= Expr(hi))

    def is_bool(self) -> Bool:
        return self.nodes[len(self.nodes) - 1].is_bool()

    def columns(self) -> List[String]:
        """The column names it reads, each once, in first-use order."""
        var out = List[String]()
        for n in self.nodes:
            if n.op == N_COL and n.name not in out:
                out.append(n.name)
        return out^


def col(name: String) -> Expr:
    return Expr([Node(N_COL, 0, name)])


def lit(value: Int) -> Expr:
    return Expr(value)


def least(a: Expr, b: Expr) -> Expr:
    """The smaller of two values per row (a formula's `min`)."""
    return a._bin(N_LEAST, b)


def greatest(a: Expr, b: Expr) -> Expr:
    """The larger of two values per row (a formula's `max`)."""
    return a._bin(N_GREATEST, b)
