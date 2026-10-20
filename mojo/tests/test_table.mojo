# F-132 (M1): Table and columns, the API in docs/omxcore-mojo-api.md section 1.
from omxcore import Table, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def expect_raises_with(mut t: Table, name: String, needle: String) raises:
    try:
        _ = t.column(name)
    except e:
        expect(needle in String(e), "error '" + String(e) + "' mentions '" + needle + "'")
        return
    raise Error("FAILED: column(" + name + ") did not raise")


def test_kinds_and_metadata() raises:
    var t = Table.empty()
    expect(t.rows() == 0 and t.columns() == 0, "a blank sheet is empty")
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var g = t.add_column("region", TEXT)
    var d = t.add_column("day", DATE)
    var b = t.add_column("paid", BOOL)
    var q = t.add_column("qty", INT)
    expect(t.columns() == 5, "five columns")
    expect(t.name(r) == "revenue" and t.kind(r) == DECIMAL and t.scale(r) == 2, "revenue is DECIMAL(2)")
    expect(t.kind(g) == TEXT and t.kind(d) == DATE and t.kind(b) == BOOL and t.kind(q) == INT, "kinds")
    expect(t.scale(q) == 0, "an INT has scale 0")
    expect(t.index("region") == g and t.index("nope") == -1, "index by name, -1 if absent")
    expect_raises_with(t, "cost", "column `cost` not found")


def test_values_nulls_and_text() raises:
    var t = Table.empty()
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var g = t.add_column("region", TEXT)
    t.append(r, 1250)                    # 12.50
    t.append_text(g, "West")
    t.append_null(r)
    t.append_text(g, "East")
    t.append(r, -99)
    t.append_text(g, "West")             # an existing word keeps its code
    t.append(r, Int64(9_007_199_254_740_993))   # past 2^53: stored exactly
    t.append_null(g)
    expect(t.rows() == 4, "four rows")
    expect(t.get(r, 0) == 1250 and t.get(r, 2) == -99 and t.get(r, 3) == 9_007_199_254_740_993, "int64 values")
    expect(t.is_null(r, 1) and not t.is_null(r, 0) and not t.is_null(r, 2), "nulls per column")
    expect(t.is_null(g, 3) and not t.is_null(g, 0), "a null text cell")
    expect(t.get(g, 0) == t.get(g, 2), "same word, same code")
    expect(t.text_of(g, Int(t.get(g, 1))) == "East", "text of a code")
    expect(t.code_of(g, "West") == Int(t.get(g, 0)) and t.code_of(g, "North") == -1, "code of a word, -1 if new")
    var words = t.dictionary(g)
    expect(len(words) == 2 and words[0] == "West" and words[1] == "East", "dictionary in first-seen order")
    var span = t.values(r)                # the app borrows a column without a copy
    expect(len(span) == 4 and span[0] == 1250 and span[3] == 9_007_199_254_740_993, "borrowed span")
    expect(t.null_count(r) == 1 and t.null_count(g) == 1, "null counts")


def test_validity_bitmap_edges() raises:
    """Every pattern up to 69 rows: no nulls, one null after k valid rows, nulls at multiples of 8, alternating, all."""
    for n in range(1, 70):
        for pattern in range(5):
            for k in range(n if pattern == 1 else 1):
                var t = Table.empty()
                var a = t.add_column("a", INT)
                var want = 0
                for i in range(n):
                    var null = (pattern == 1 and i == k) or (pattern == 2 and i % 8 == 0) or \
                               (pattern == 3 and i % 2 == 1) or pattern == 4
                    if null:
                        t.append_null(a)
                        want += 1
                    else:
                        t.append(a, Int64(i + 1))
                expect(t.null_count(a) == want, "null count n=" + String(n) + " pattern " + String(pattern))
                for i in range(n):
                    var null = (pattern == 1 and i == k) or (pattern == 2 and i % 8 == 0) or \
                               (pattern == 3 and i % 2 == 1) or pattern == 4
                    expect(t.is_null(a, i) == null and t.get(a, i) == (0 if null else Int64(i + 1)),
                           "row " + String(i) + " of " + String(n) + " pattern " + String(pattern))


def test_bad_values_and_kinds_raise() raises:
    var t = Table.empty()
    var g = t.add_column("g", TEXT)
    var b = t.add_column("b", BOOL)
    var q = t.add_column("q", INT)
    var cases = List[String]()
    try:
        t.append(g, 0)                    # no words yet
    except e:
        cases.append(String(e))
    try:
        t.append(b, 5)
    except e:
        cases.append(String(e))
    try:
        t.append_text(q, "x")
    except e:
        cases.append(String(e))
    try:
        _ = t.add_column("d", DECIMAL, scale=19)
    except e:
        cases.append(String(e))
    try:
        _ = t.add_column("i", INT, scale=2)
    except e:
        cases.append(String(e))
    try:
        _ = t.add_column("k", ColumnKind(42))
    except e:
        cases.append(String(e))
    expect(len(cases) == 6, "six errors, got " + String(len(cases)))
    expect("column `g`: code 0 is not one of its 0 words" in cases[0], cases[0])
    expect("column `b` is BOOL: 5 is not 0 or 1" in cases[1], cases[1])
    expect("column `q` is INT, not TEXT" in cases[2], cases[2])
    expect("scale 19 is outside 0..18" in cases[3], cases[3])
    expect("only a DECIMAL has a scale" in cases[4], cases[4])
    expect("unknown kind 42" in cases[5], cases[5])


def test_a_dictionary_loads_without_rows() raises:
    var t = Table.empty()
    var g = t.add_column("g", TEXT)
    expect(t.add_word(g, "West") == 0 and t.add_word(g, "East") == 1 and t.add_word(g, "West") == 0, "codes")
    expect(t.rows() == 0, "no rows added")
    t.append(g, 1)
    t.append(g, 0)
    expect(t.text_of(g, Int(t.get(g, 0))) == "East" and t.null_count(g) == 0, "codes appended")


def test_any_number_of_columns() raises:
    var t = Table.empty()
    for c in range(100):                  # the stand-in stopped at 16
        _ = t.add_column("c" + String(c), INT)
    for c in range(100):
        t.append(c, Int64(c * 3))
    expect(t.columns() == 100 and t.rows() == 1 and t.get(99, 0) == 297, "100 columns")


def test_ragged_columns_raise() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    var b = t.add_column("b", INT)
    t.append(a, 1)
    t.append(a, 2)
    t.append(b, 1)
    try:
        t.check()
    except e:
        expect("column `b` has 1 row, column `a` has 2 rows" in String(e), "ragged message: " + String(e))
        return
    raise Error("FAILED: check() accepted ragged columns")


def test_duplicate_column_name_raises() raises:
    var t = Table.empty()
    _ = t.add_column("a", INT)
    try:
        _ = t.add_column("a", TEXT)
    except e:
        expect("column `a` already exists" in String(e), "duplicate message: " + String(e))
        return
    raise Error("FAILED: a duplicate column name was accepted")


def test_shared_table_is_read_only() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    t.append(a, 41)
    var shared = t^.share()                # one read-only table for every session
    var again = shared                     # another session's handle: the same columns, no copy
    expect(again.get(a, 0) == 41 and shared.rows() == 1, "shared reads")
    expect(Int(again.values(a).unsafe_ptr()) == Int(shared.values(a).unsafe_ptr()), "no copy between handles")


def test_shared_text_reads() raises:
    var t = Table.empty()
    var g = t.add_column("g", TEXT)
    t.append_text(g, "West")
    t.append_null(g)
    var shared = t^.share()
    var words = shared.dictionary(g)
    expect(len(words) == 1 and words[0] == "West", "dictionary through a shared handle")
    expect(shared.text_of(g, 0) == "West" and shared.null_count(g) == 1 and shared.is_null(g, 1), "shared reads")


def main() raises:
    test_kinds_and_metadata()
    test_values_nulls_and_text()
    test_any_number_of_columns()
    test_ragged_columns_raise()
    test_duplicate_column_name_raises()
    test_shared_table_is_read_only()
    test_validity_bitmap_edges()
    test_bad_values_and_kinds_raise()
    test_a_dictionary_loads_without_rows()
    test_shared_text_reads()
