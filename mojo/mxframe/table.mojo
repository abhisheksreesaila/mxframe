"""Tables and columns (docs/mxframe-mojo-api.md section 1).

Every value is an Int64 underneath, column-major and contiguous, so the app borrows a column as `Span[Int64]` and
the kernels read it as an int64 column with no conversion:
  INT      the integer
  DECIMAL  the value times 10^scale (scale 2: cents), so sums stay exact
  TEXT     a dictionary code (0, 1, ... in first-seen order); the words are the column's dictionary
  DATE     days since 1970-01-01
  BOOL     0 or 1
A null is a cleared bit in the column's validity bitmap (Arrow's layout: bit i of byte i // 8, least significant
first); its value slot holds 0. A column with no nulls has no bitmap.

A `Table` is built by one owner (`add_column`, `append*`), then `share()`d: a `SharedTable` is a read-only handle
that any number of sessions copy without copying the columns.
"""
from std.collections import Dict
from std.memory import ArcPointer


struct ColumnKind(Equatable, ImplicitlyCopyable, Writable):
    var value: Int

    def __init__(out self, value: Int):
        self.value = value

    def __eq__(self, other: Self) -> Bool:
        return self.value == other.value

    def __ne__(self, other: Self) -> Bool:
        return self.value != other.value

    def write_to(self, mut writer: Some[Writer]):
        var names: List[String] = ["INT", "DECIMAL", "TEXT", "DATE", "BOOL"]
        writer.write(names[self.value] if 0 <= self.value < len(names) else "kind " + String(self.value))


comptime INT = ColumnKind(0)
comptime DECIMAL = ColumnKind(1)
comptime TEXT = ColumnKind(2)
comptime DATE = ColumnKind(3)
comptime BOOL = ColumnKind(4)


struct Column(Movable):
    var name: String
    var kind: ColumnKind
    var scale: Int
    var values: List[Int64]
    var validity: List[UInt8]          # empty: no nulls
    var nulls: Int
    var words: List[String]            # TEXT: the dictionary, by code
    var codes: Dict[String, Int]       # TEXT: word -> code

    def __init__(out self, name: String, kind: ColumnKind, scale: Int):
        self.name = name
        self.kind = kind
        self.scale = scale
        self.values = List[Int64]()
        self.validity = List[UInt8]()
        self.nulls = 0
        self.words = List[String]()
        self.codes = Dict[String, Int]()

    def push(mut self, value: Int64, valid: Bool):
        var i = len(self.values)
        self.values.append(value if valid else 0)
        if not valid and len(self.validity) == 0:      # first null: every earlier row was valid
            for _ in range((i + 8) // 8):
                self.validity.append(0xFF)
        if len(self.validity) > 0:
            if i // 8 >= len(self.validity):
                self.validity.append(0xFF)
            if not valid:
                self.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                self.nulls += 1

    def is_null(self, row: Int) -> Bool:
        if len(self.validity) == 0:
            return False
        return (self.validity[row // 8] >> UInt8(row % 8)) & 1 == 0


struct Table(Movable):
    var cols: List[Column]
    var by_name: Dict[String, Int]

    def __init__(out self):
        self.cols = List[Column]()
        self.by_name = Dict[String, Int]()

    @staticmethod
    def empty() -> Table:
        """A table with no columns ("New blank sheet")."""
        return Table()

    # ── shape and metadata ──────────────────────────────────────────────────────────────────────

    def add_column(mut self, name: String, kind: ColumnKind, scale: Int = 0) raises -> Int:
        """Adds an empty column; returns its index. A DECIMAL's scale is its number of decimal places."""
        if name in self.by_name:
            raise Error("column `" + name + "` already exists")
        if scale < 0 or scale > 18:
            raise Error("column `" + name + "`: scale " + String(scale) + " is outside 0..18")
        if kind != DECIMAL and scale != 0:
            raise Error("column `" + name + "`: only a DECIMAL has a scale")
        self.by_name[name] = len(self.cols)
        self.cols.append(Column(name, kind, scale))
        return len(self.cols) - 1

    def rows(self) -> Int:
        return len(self.cols[0].values) if len(self.cols) > 0 else 0

    def columns(self) -> Int:
        return len(self.cols)

    def name(self, c: Int) -> String:
        return self.cols[c].name

    def kind(self, c: Int) -> ColumnKind:
        return self.cols[c].kind

    def scale(self, c: Int) -> Int:
        return self.cols[c].scale

    def index(self, name: String) -> Int:
        """The column's index, or -1 when there is none."""
        return self.by_name.get(name, -1)

    def column(self, name: String) raises -> Int:
        """The column's index; raises "column `x` not found" when there is none."""
        var c = self.index(name)
        if c < 0:
            raise Error("column `" + name + "` not found")
        return c

    def check(self) raises:
        """Raises when the columns do not all have the same number of rows."""
        for c in range(1, len(self.cols)):
            if len(self.cols[c].values) != self.rows():
                raise Error("column `" + self.cols[c].name + "` has " + String(len(self.cols[c].values)) +
                            " rows, column `" + self.cols[0].name + "` has " + String(self.rows()))

    # ── building ────────────────────────────────────────────────────────────────────────────────

    def append(mut self, c: Int, value: Int64):
        """Appends a stored value (scaled for a DECIMAL, a code for TEXT, days for DATE, 0 / 1 for BOOL)."""
        self.cols[c].push(value, True)

    def append_null(mut self, c: Int):
        self.cols[c].push(0, False)

    def append_text(mut self, c: Int, word: String) raises:
        """Appends a TEXT value: an existing word keeps its code, a new word gets the next one."""
        if self.cols[c].kind != TEXT:
            raise Error("column `" + self.cols[c].name + "` is " + String(self.cols[c].kind) + ", not TEXT")
        var code = self.cols[c].codes.get(word, -1)
        if code < 0:
            code = len(self.cols[c].words)
            self.cols[c].words.append(word)
            self.cols[c].codes[word] = code
        self.cols[c].push(Int64(code), True)

    # ── reading ─────────────────────────────────────────────────────────────────────────────────

    def get(self, c: Int, row: Int) -> Int64:
        """The stored value (0 for a null: ask `is_null`)."""
        return self.cols[c].values[row]

    def is_null(self, c: Int, row: Int) -> Bool:
        return self.cols[c].is_null(row)

    def null_count(self, c: Int) -> Int:
        return self.cols[c].nulls

    def values(ref self, c: Int) -> Span[Int64, origin_of(self)]:
        """The column's values without a copy, valid until the column next changes."""
        var p = self.cols[c].values.unsafe_ptr().unsafe_origin_cast[origin_of(self)]()
        return Span[Int64, origin_of(self)](unsafe_ptr=p, length=len(self.cols[c].values))

    def dictionary(ref self, c: Int) -> Span[String, origin_of(self)]:
        """A TEXT column's words, by code."""
        var p = self.cols[c].words.unsafe_ptr().unsafe_origin_cast[origin_of(self)]()
        return Span[String, origin_of(self)](unsafe_ptr=p, length=len(self.cols[c].words))

    def text_of(self, c: Int, code: Int) -> String:
        return self.cols[c].words[code]

    def code_of(self, c: Int, word: String) -> Int:
        """The word's code, or -1 when the column has never held it."""
        return self.cols[c].codes.get(word, -1)

    def share(deinit self) -> SharedTable:
        """Hands the table over as a read-only table every session can hold (no copy)."""
        return SharedTable(self^)


struct SharedTable(ImplicitlyCopyable):
    """A read-only handle on a shared table. Copies share the columns; the table lives while any handle does."""
    var table: ArcPointer[Table]

    def __init__(out self, var table: Table):
        self.table = ArcPointer(table^)

    def rows(self) -> Int:
        return self.table[].rows()

    def columns(self) -> Int:
        return self.table[].columns()

    def name(self, c: Int) -> String:
        return self.table[].name(c)

    def kind(self, c: Int) -> ColumnKind:
        return self.table[].kind(c)

    def scale(self, c: Int) -> Int:
        return self.table[].scale(c)

    def index(self, name: String) -> Int:
        return self.table[].index(name)

    def column(self, name: String) raises -> Int:
        return self.table[].column(name)

    def get(self, c: Int, row: Int) -> Int64:
        return self.table[].get(c, row)

    def is_null(self, c: Int, row: Int) -> Bool:
        return self.table[].is_null(c, row)

    def null_count(self, c: Int) -> Int:
        return self.table[].null_count(c)

    def values(self, c: Int) -> Span[Int64, ImmutAnyOrigin]:
        """The column's values, read-only (the table is shared), without a copy."""
        var p = Pointer[Int64, ImmutAnyOrigin](unsafe_from_address=Int(self.table[].cols[c].values.unsafe_ptr()))
        return Span[Int64, ImmutAnyOrigin](unsafe_ptr=p, length=len(self.table[].cols[c].values))

    def text_of(self, c: Int, code: Int) -> String:
        return self.table[].text_of(c, code)

    def code_of(self, c: Int, word: String) -> Int:
        return self.table[].code_of(c, word)
