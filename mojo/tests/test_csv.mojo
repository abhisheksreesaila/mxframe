# F-134 (M3): the CSV reader, docs/mxframe-mojo-api.md section 2.
from std.os import makedirs
from mxframe import (read_csv, CsvOptions, CsvResult, FailedRow, Table, INT, DECIMAL, TEXT, DATE, BOOL, write_parquet,
                     read_parquet)

comptime DIR = "/tmp/mxframe_csv_test/"


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def write_file(name: String, content: String) raises -> String:
    """Writes the text to a temp file; returns its path."""
    makedirs(DIR, exist_ok=True)
    var f = open(DIR + name, "w")
    f.write(content)
    f.close()
    return DIR + name


def write_with_bom(name: String, content: String) raises -> String:
    makedirs(DIR, exist_ok=True)
    var bytes: List[UInt8] = [0xEF, 0xBB, 0xBF]
    bytes.extend(content.as_bytes())
    var f = open(DIR + name, "w")
    f.write_bytes(bytes)
    f.close()
    return DIR + name


def read(name: String, content: String) raises -> CsvResult:
    return read_csv(write_file(name, content))


def cell(r: CsvResult, c: Int, row: Int) -> String:
    """A cell as a user would read it back: null, a word, or the stored number."""
    if r.table.is_null(c, row):
        return "null"
    if r.table.kind(c) == TEXT:
        return r.table.text_of(c, Int(r.table.get(c, row)))
    return String(r.table.get(c, row))


def check_row(r: CsvResult, row: Int, expected: String) raises:
    var got = String()
    for c in range(r.table.columns()):
        got += (";" if c > 0 else "") + cell(r, c, row)
    expect(got == expected, "row " + String(row) + " reads `" + got + "`, expected `" + expected + "`")


# ── type inference ──────────────────────────────────────────────────────────────────────────────


def test_integers() raises:
    var r = read("ints.csv", "n\n1\n-20\n+3\n9007199254740993\n-9223372036854775808\n9223372036854775807\n")
    expect(r.table.kind(0) == INT and r.table.scale(0) == 0, "whole numbers are INT")
    expect(r.rows == 6, "six rows")
    expect(r.table.get(0, 1) == -20 and r.table.get(0, 2) == 3, "signs")
    expect(r.table.get(0, 3) == 9_007_199_254_740_993, "exact past 2^53")
    expect(r.table.get(0, 4) == Int64.MIN and r.table.get(0, 5) == Int64.MAX, "the int64 limits")
    var big = read("bigint.csv", "n\n1\n9223372036854775808\n")
    expect(big.table.kind(0) == TEXT, "a number past int64 fits no numeric kind: text")


def test_decimals() raises:
    var r = read("dec.csv", "a,b,c,d\n1.5,1.234,12345678901234.56,5\n2,2.5,-0.07,.5\n-3.25,0.00005,1.50,6.\n")
    expect(r.table.kind(0) == DECIMAL and r.table.scale(0) == 2, "at most 2 places: scale 2")
    expect(r.table.kind(1) == DECIMAL and r.table.scale(1) == 4, "3 or 4 places: scale 4")
    expect(r.table.scale(2) == 2, "trailing zeros do not count as places")
    check_row(r, 0, "150;12340;1234567890123456;500")
    check_row(r, 1, "200;25000;-7;50")
    check_row(r, 2, "-325;1;150;600")       # 0.00005 rounds half away from zero at 4 places
    var wide = read("dec2.csv", "x\n0.123456\n-0.123456\n")
    expect(wide.table.scale(0) == 4 and wide.table.get(0, 0) == 1235 and wide.table.get(0, 1) == -1235,
           "more than 4 places: scale 4, rounded from the digits")
    var exact = read("dec3.csv", "x\n0.1\n0.2\n0.30\n1234567890123456.78\n")
    expect(exact.table.get(0, 3) == 123_456_789_012_345_678, "a decimal is exact (no float): 17 digits kept")


def test_dates() raises:
    var r = read("dates.csv", "d,t\n1970-01-01,1970-01-02T10:30:00Z\n2024-02-29,2024-03-01 00:00:00.123\n" +
                 "1969-12-31,2000-01-01T23:59:59+05:30\n")
    expect(r.table.kind(0) == DATE and r.table.kind(1) == DATE, "ISO dates and timestamps are DATE")
    check_row(r, 0, "0;1")
    check_row(r, 1, "19782;19783")
    check_row(r, 2, "-1;10957")
    var bad = read("baddates.csv", "d\n2023-02-29\n2024-13-01\n")
    expect(bad.table.kind(0) == TEXT, "an impossible date is text")
    var bad2 = read("baddates2.csv", "d\n2024-01-01\n2024-1-2\n")
    expect(bad2.table.kind(0) == TEXT, "a date must be YYYY-MM-DD")


def test_bools_and_text() raises:
    var r = read("bool.csv", "p,w,e\ntrue,hello,\nFALSE,World,\nTrue,hello,\n")
    expect(r.table.kind(0) == BOOL, "true / false in any case is BOOL")
    check_row(r, 0, "1;hello;null")
    check_row(r, 1, "0;World;null")
    expect(r.table.kind(1) == TEXT and r.columns[1].distinct == 2, "text, two distinct words")
    expect(r.table.kind(2) == TEXT and r.columns[2].empty == 3, "a column with nothing but empty cells is TEXT")
    var mixed = read("mixed.csv", "m\n1\ntrue\n2\n")
    expect(mixed.table.kind(0) == TEXT, "numbers and booleans mixed are text")
    var words = read("words.csv", "m\n12abc\n1e5\n0x10\n-\n1.2.3\n")
    expect(words.table.kind(0) == TEXT, "not numbers: text")


def test_int_and_decimal_mix() raises:
    var r = read("mix.csv", "m\n1\n2.5\n3\n")
    expect(r.table.kind(0) == DECIMAL and r.table.scale(0) == 2, "whole numbers and decimals: DECIMAL")
    check_row(r, 0, "100")
    check_row(r, 2, "300")


# ── nulls, the report and failed rows ───────────────────────────────────────────────────────────


def test_nulls_and_report() raises:
    var r = read("nulls.csv", "id,price,city\n1,2.50,Oslo\n2,,\"\"\n,3.00,Oslo\n4,1.25,Rome\n")
    expect(r.rows == 4 and len(r.columns) == 3, "shape")
    check_row(r, 0, "1;250;Oslo")
    check_row(r, 1, "2;null;null")
    check_row(r, 2, "null;300;Oslo")
    expect(r.columns[0].empty == 1 and r.columns[1].empty == 1 and r.columns[2].empty == 1, "empty cells per column")
    expect(r.table.null_count(0) == 1 and r.table.null_count(2) == 1, "nulls in the table")
    expect(r.columns[0].name == "id" and r.columns[1].kind == DECIMAL and r.columns[1].scale == 2, "name, kind, scale")
    expect(r.columns[2].distinct == 2, "distinct text values: Oslo and Rome")
    expect(r.columns[0].distinct == 0, "distinct is only for text")
    expect(len(r.columns[2].sample) == 4 and r.columns[2].sample[0] == "Oslo" and r.columns[2].sample[1] == "",
           "sample: the first values as written")
    expect(r.seconds >= 0.0, "seconds")


def test_sample_is_the_first_eight() raises:
    var text = String("n\n")
    for i in range(20):
        text += String(i * 10) + "\n"
    var r = read("sample.csv", text)
    expect(len(r.columns[0].sample) == 8 and r.columns[0].sample[7] == "70", "eight sample values")


def test_failed_rows_after_the_window() raises:
    var r = read_csv(write_file("fail.csv", "price,when,ok,n\n1.5,2024-01-01,true,1\n2.5,2024-01-02,false,2\n" +
                                "3.5,2024-01-03,true,3\n12.x,2024-01-04,yes,4.5\n5,2024-13-45,FALSE,5\n"),
                     CsvOptions(infer_rows=3))
    expect(r.rows == 5, "all rows are kept")
    expect(r.table.kind(0) == DECIMAL and r.table.kind(1) == DATE and r.table.kind(2) == BOOL and r.table.kind(3) == INT,
           "kinds come from the first 3 rows only")
    check_row(r, 3, "null;19726;null;null")
    check_row(r, 4, "500;null;0;5")
    expect(r.columns[0].failed_count == 1 and len(r.columns[0].failed) == 1, "one failed price")
    expect(r.columns[0].failed[0].row == 4 and r.columns[0].failed[0].text == "12.x", "row number and text")
    expect(r.columns[0].failed[0].message() == "row 4: `12.x` isn't a number", r.columns[0].failed[0].message())
    expect(r.columns[1].failed[0].message() == "row 5: `2024-13-45` isn't a date (YYYY-MM-DD)",
           r.columns[1].failed[0].message())
    expect(r.columns[2].failed[0].message() == "row 4: `yes` isn't true or false", r.columns[2].failed[0].message())
    expect(r.columns[3].failed[0].message() == "row 4: `4.5` isn't a whole number", r.columns[3].failed[0].message())
    expect(r.columns[0].empty == 0, "a failed cell is not counted as empty")
    expect(FailedRow(41207, "12.x", "isn't a number").message() == "row 41,207: `12.x` isn't a number", "thousands")
    expect(r.table.null_count(0) == 1 and r.table.null_count(1) == 1, "failed cells are null")


# ── file format ─────────────────────────────────────────────────────────────────────────────────


def test_quoting() raises:
    var r = read("quotes.csv", "a,b,c\n\"x,y\",\"say \"\"hi\"\"\",plain\n\"line1\nline2\",\"\",\"\"\"\"\n\"a\r\nb\",z,w\n")
    expect(r.rows == 3, "three rows (an embedded newline does not end a row)")
    check_row(r, 0, "x,y;say \"hi\";plain")
    check_row(r, 1, "line1\nline2;null;\"")
    check_row(r, 2, "a\r\nb;z;w")


def test_bom_and_line_endings() raises:
    var bom = read_csv(write_with_bom("bom.csv", "id,name\n1,a\n2,b\n"))
    expect(bom.table.name(0) == "id", "the byte order mark is not part of the first name, got `" + bom.table.name(0) + "`")
    expect(bom.rows == 2, "rows after a BOM")
    var crlf = read("crlf.csv", "id,name\r\n1,a\r\n2,b\r\n")
    expect(crlf.rows == 2 and crlf.table.kind(0) == INT, "CRLF: no carriage return in the last cell")
    check_row(crlf, 1, "2;b")
    var no_nl = read("nonl.csv", "id,name\n1,a\n2,b")
    expect(no_nl.rows == 2, "no trailing newline")
    check_row(no_nl, 1, "2;b")
    var no_nl_crlf = read("nonl2.csv", "id,name\r\n1,a\r\n2,b\r")
    check_row(no_nl_crlf, 1, "2;b")
    var blanks = read("blank.csv", "id,name\n\n1,a\n\r\n2,b\n\n")
    expect(blanks.rows == 2, "blank lines are skipped")


def test_tsv_and_delimiters() raises:
    var tsv = read_csv(write_file("data.tsv", "a\tb\n1\t\"x\ty\"\n2\tz\n"))
    expect(tsv.table.columns() == 2 and tsv.rows == 2, "a .tsv file splits on tabs")
    check_row(tsv, 0, "1;x\ty")
    var semi = read_csv(write_file("semi.txt", "a;b\n1;2\n"), CsvOptions(delimiter=";"))
    expect(semi.table.columns() == 2 and semi.table.get(1, 0) == 2, "an explicit delimiter")
    var tabbed = read_csv(write_file("tabbed.txt", "a\tb\n1\t2\n"), CsvOptions(delimiter="\t"))
    expect(tabbed.table.columns() == 2, "an explicit tab")
    try:
        _ = read_csv(write_file("x.csv", "a\n1\n"), CsvOptions(delimiter="ab"))
        raise Error("FAILED: a two-character delimiter did not raise")
    except e:
        expect("one character" in String(e), String(e))


def test_header_options() raises:
    var no_header = read_csv(write_file("nohead.csv", "1,x\n2,y\n"), CsvOptions(has_header=False))
    expect(no_header.rows == 2 and no_header.table.name(0) == "column_1" and no_header.table.name(1) == "column_2",
           "no header: generated names, the first row is data")
    var dup = read("dup.csv", "a,a,,a\n1,2,3,4\n")
    expect(dup.table.name(0) == "a" and dup.table.name(1) == "a_2" and dup.table.name(2) == "column_3" and
           dup.table.name(3) == "a_3", "repeated and empty names are made unique")
    var wide = read_csv(write_file("wide.csv", "a,b,c,d\n1,2,3,4\n"), CsvOptions(max_columns=2))
    expect(wide.table.columns() == 2 and len(wide.skipped) == 2 and wide.skipped[0] == "c" and wide.skipped[1] == "d",
           "columns past max_columns are skipped and named")
    check_row(wide, 0, "1;2")


def test_ragged_rows() raises:
    var r = read("ragged.csv", "a,b,c\n1,2,3\n4,5\n6\n7,8,9,10,11\n12,13,14\n")
    expect(r.rows == 5, "five rows")
    check_row(r, 1, "4;5;null")
    check_row(r, 2, "6;null;null")
    check_row(r, 3, "7;8;9")
    check_row(r, 4, "12;13;14")
    expect(r.columns[2].empty == 2, "short rows count as empty cells")
    expect(r.ragged_count == 1 and len(r.ragged) == 1 and r.ragged[0].row == 4 and r.ragged[0].text == "7,8,9,10,11",
           "a row with too many cells is reported with its text")
    expect(r.ragged[0].message() == "row 4: `7,8,9,10,11` has 5 cells, the header has 3; the extra cells were dropped",
           r.ragged[0].message())


def test_empty_and_odd_files() raises:
    var empty = read("empty.csv", "")
    expect(empty.rows == 0 and empty.table.columns() == 0 and len(empty.columns) == 0, "an empty file")
    var blank = read("blankonly.csv", "\n\n")
    expect(blank.rows == 0 and blank.table.columns() == 0, "only blank lines")
    var header_only = read("headonly.csv", "a,b\n")
    expect(header_only.rows == 0 and header_only.table.columns() == 2 and header_only.table.kind(0) == TEXT,
           "a header and no rows: empty text columns")
    var one_cell = read("one.csv", "n\n5")
    expect(one_cell.rows == 1 and one_cell.table.get(0, 0) == 5, "one cell, no newline")
    try:
        _ = read_csv("/tmp/mxframe_csv_test/does-not-exist.csv")
        raise Error("FAILED: a missing file did not raise")
    except e:
        expect("couldn't open" in String(e) and "does-not-exist.csv" in String(e), String(e))


def test_utf8_text() raises:
    var r = read("utf8.csv", "name\nZürich\n東京\nZürich\n")
    expect(r.table.kind(0) == TEXT and r.columns[0].distinct == 2, "two distinct words")
    check_row(r, 1, "東京")
    check_row(r, 2, "Zürich")


# ── the parallel path equals the single-threaded one ───────────────────────────────────────────


def tables_equal(a: CsvResult, b: CsvResult) raises:
    expect(a.rows == b.rows and a.table.columns() == b.table.columns(), "same shape")
    for c in range(a.table.columns()):
        var what = "column `" + a.table.name(c) + "`: "
        expect(a.table.name(c) == b.table.name(c) and a.table.kind(c) == b.table.kind(c) and
               a.table.scale(c) == b.table.scale(c), what + "same name, kind, scale")
        expect(a.table.null_count(c) == b.table.null_count(c), what + "same null count")
        expect(a.columns[c].empty == b.columns[c].empty and a.columns[c].distinct == b.columns[c].distinct, what + "report")
        expect(a.columns[c].failed_count == b.columns[c].failed_count, what + "failed count")
        for k in range(len(a.columns[c].failed)):
            expect(a.columns[c].failed[k].row == b.columns[c].failed[k].row and
                   a.columns[c].failed[k].text == b.columns[c].failed[k].text, what + "failed rows")
        for row in range(a.rows):
            if cell(a, c, row) != cell(b, c, row):
                raise Error("FAILED: " + what + "row " + String(row) + " is `" + cell(a, c, row) + "` vs `" + cell(b, c, row) + "`")
    expect(a.ragged_count == b.ragged_count, "same ragged rows")


def test_quote_boundaries_in_chunks() raises:
    """Tiny chunks make most boundaries land inside quoted fields; the result must not change."""
    var text = String("id,note,amount\n")
    for i in range(400):
        var note = "plain" + String(i % 7)
        if i % 3 == 0:
            note = "\"multi\nline, \"\"quoted\"\" " + String(i) + "\nend\""
        elif i % 5 == 0:
            note = "\"\""
        text += String(i) + "," + note + "," + String(i) + "." + String(i % 100 // 10) + String(i % 10) + "\r\n"
    var path = write_file("boundaries.csv", text)
    var one = read_csv(path, CsvOptions(threads=1))
    expect(one.rows == 400, "400 rows, got " + String(one.rows))
    for size in [7, 31, 64, 100, 513]:
        var many = read_csv(path, CsvOptions(threads=4, chunk_bytes=size))
        tables_equal(one, many)
    check_row(one, 3, "3;multi\nline, \"quoted\" 3\nend;303")


def test_generated_file_in_parallel() raises:
    """A few hundred thousand rows: the multi-threaded parse equals the single-threaded one, and the values are the
    ones generated."""
    var rows = 300_000
    var names: List[String] = ["Oslo", "Rome", "Lima", "Kyiv", "Doha", "Quito, EC", "Bern \"city\""]
    var text = String("id,price,when,paid,city,rare\n")
    var price_total = Int64(0)
    var bad_rows = 0
    for i in range(rows):
        var cents = (i * 37) % 100_000
        price_total += Int64(cents)
        var price = String(cents // 100) + "." + ("0" if cents % 100 < 10 else "") + String(cents % 100)
        var city = names[i % 7]
        if city.startswith("Quito"):
            city = "\"Quito, EC\""
        elif city.startswith("Bern"):
            city = "\"Bern \"\"city\"\"\""
        var day = "2024-" + ("0" if (i % 12) + 1 < 10 else "") + String((i % 12) + 1) + "-15"
        var rare = String()
        if i == 150_000:
            rare = "oops"
            bad_rows += 1
        elif i % 1000 == 0:
            rare = String(i)
        text += String(i) + "," + price + "," + day + "," + ("true" if i % 2 == 0 else "false") + "," + city + "," + rare + "\n"
    var path = write_file("big.csv", text)
    var one = read_csv(path, CsvOptions(threads=1, infer_rows=1000))
    var par = read_csv(path, CsvOptions(threads=4, chunk_bytes=100_000, infer_rows=1000))
    expect(one.rows == rows and par.rows == rows, "300,000 rows both ways")
    tables_equal(one, par)
    var sum = Int64(0)
    for v in par.table.values(1):
        sum += v
    expect(sum == price_total, "the prices add up exactly: " + String(sum) + " vs " + String(price_total))
    expect(par.table.kind(1) == DECIMAL and par.table.kind(2) == DATE and par.table.kind(3) == BOOL, "kinds")
    expect(par.columns[4].distinct == 7, "seven cities")
    expect(par.table.text_of(4, Int(par.table.get(4, 5))) == "Quito, EC", "quoted comma survives the chunks")
    expect(par.table.text_of(4, Int(par.table.get(4, 6))) == "Bern \"city\"", "doubled quotes survive the chunks")
    expect(par.table.kind(5) == INT and par.columns[5].failed_count == 1 and par.columns[5].failed[0].row == 150_001 and
           par.columns[5].failed[0].text == "oops", "the one bad cell past the window is row 150,001")
    expect(par.table.get(0, 299_999) == 299_999, "last row")


# ── review fixes ────────────────────────────────────────────────────────────────────────────────


def test_max_columns_is_not_ragged() raises:
    var r = read_csv(write_file("capped.csv", "a,b,c,d\n1,2,3,4\n5,6,7,8\n"), CsvOptions(max_columns=2))
    expect(r.ragged_count == 0 and len(r.ragged) == 0, "dropping columns past max_columns is not a ragged row")
    expect(r.rows == 2 and r.table.columns() == 2, "two columns kept")
    var wide = read_csv(write_file("capped2.csv", "a,b,c\n1,2,3\n5,6,7,8\n"), CsvOptions(max_columns=2))
    expect(wide.ragged_count == 1 and wide.ragged[0].row == 2, "a row longer than the header is still reported")


def test_stray_quote_is_literal() raises:
    var path = write_file("stray.csv", "a,b\n5\" pipe,x\n6,y\n7,z\n")
    for threads in [1, 4]:
        var r = read_csv(path, CsvOptions(threads=threads, chunk_bytes=3))
        expect(r.rows == 3, "a quote inside an unquoted field is a character: 3 rows, got " + String(r.rows))
        check_row(r, 0, "5\" pipe;x")
        check_row(r, 2, "7;z")
        expect(r.ragged_count == 0, "nothing ragged")
    var mid = read("stray2.csv", "a,b\n\"quoted\"tail,x\nab\"cd,\"e\"\"f\"\n")
    check_row(mid, 0, "quotedtail;x")
    check_row(mid, 1, "ab\"cd;e\"f")


def test_unclosed_quote_is_reported() raises:
    var path = write_file("open.csv", "a,b\n1,\"open\n2,y\n")
    for threads in [1, 4]:
        var r = read_csv(path, CsvOptions(threads=threads, chunk_bytes=3))
        expect(r.rows == 1, "the rest of the file is one cell: 1 row, got " + String(r.rows))
        check_row(r, 0, "1;open\n2,y\n")
        expect(r.ragged_count == 1 and r.ragged[0].row == 1, "reported")
        expect(r.ragged[0].message() == "row 1: `1,\"open\n2,y\n` has an unmatched quote: the rest of the file was read as one cell",
               r.ragged[0].message())


def test_timestamp_zones_convert_to_utc() raises:
    var r = read("zones.csv", "t\n2023-01-01T23:30:00-05:00\n2023-01-01T00:30:00+05:00\n2023-01-01T12:00:00Z\n" +
                 "2023-01-01 23:30:00-0500\n2023-01-01T00:00:00+00:00\n2023-01-01T23:59:59.999-01\n")
    expect(r.table.kind(0) == DATE, "timestamps with zones are DATE")
    check_row(r, 0, "19359")   # 04:30 UTC on the 2nd
    check_row(r, 1, "19357")   # 19:30 UTC on 31 December
    check_row(r, 2, "19358")
    check_row(r, 3, "19359")
    check_row(r, 4, "19358")
    check_row(r, 5, "19359")


def test_decimal_widens_to_scale_4() raises:
    var path = write_file("widen.csv", "x\n1.5\n2.25\n1.239\n0.005\n")
    for threads in [1, 4]:
        var r = read_csv(path, CsvOptions(infer_rows=2, threads=threads, chunk_bytes=4))
        expect(r.table.scale(0) == 4 and r.columns[0].scale == 4, "scale 4 for the whole column")
        expect(r.table.get(0, 0) == 15000 and r.table.get(0, 1) == 22500 and r.table.get(0, 2) == 12390 and
               r.table.get(0, 3) == 50, "earlier values are multiplied by 100")
        expect(r.columns[0].rounded == 0, "nothing rounded")
    var round5 = read_csv(write_file("round.csv", "x\n1.5\n1.23456\n-1.23454\n"), CsvOptions(infer_rows=1))
    expect(round5.table.scale(0) == 4 and round5.table.get(0, 1) == 12346 and round5.table.get(0, 2) == -12345,
           "more than 4 places rounds half away from zero")
    expect(round5.columns[0].rounded == 2, "and is counted: " + String(round5.columns[0].rounded))
    var big = read_csv(write_file("widenbig.csv", "x\n92233720368547758.07\n1.239\n"), CsvOptions(infer_rows=1))
    expect(big.table.scale(0) == 4 and big.table.is_null(0, 0) and big.table.get(0, 1) == 12390,
           "a value too big for scale 4 becomes a failed cell")
    expect(big.columns[0].failed_count == 1 and big.columns[0].failed[0].row == 1 and
           big.columns[0].failed[0].text == "92233720368547758.07", "reported with its row and text")
    var mixed = read_csv(write_file("widenorder.csv", "x\n92233720368547758.07\n1.5\n1.x\n1.239\n"),
                         CsvOptions(infer_rows=2, threads=1))
    ref fails = mixed.columns[0].failed
    expect(len(fails) == 2 and fails[0].row == 1 and fails[1].row == 3,
           "failed cells come in row order whichever pass found them")
    var low = read_csv(write_file("widenmin.csv", "x\n-92233720368547758.08\n1.239\n"), CsvOptions(infer_rows=1))
    expect(low.columns[0].failed_count == 0 or not low.columns[0].failed[0].text.startswith("--"),
           "a widened INT64_MIN is written with one minus sign")


def test_overflow_inside_the_window_is_text() raises:
    var r = read("overflow.csv", "x\n92233720368547758.08\n1.5\n")
    expect(r.table.kind(0) == TEXT and r.columns[0].failed_count == 0, "a column that cannot hold its own values is text")
    check_row(r, 0, "92233720368547758.08")
    var fits = read("fits.csv", "x\n92233720368547758.07\n1.5\n")
    expect(fits.table.kind(0) == DECIMAL and fits.table.get(0, 0) == Int64.MAX, "the largest value that fits")


def test_stray_quotes_across_chunk_boundaries() raises:
    var text = String("id,note\n")
    for i in range(300):
        var note = "5\" pipe " + String(i)
        if i % 4 == 0:
            note = "\"a\nb\"\"" + String(i) + "\"tail\"x"
        elif i % 7 == 0:
            note = "\"closed, \nnote\""
        text += String(i) + "," + note + "\n"
    var path = write_file("strayboundaries.csv", text)
    var one = read_csv(path, CsvOptions(threads=1))
    expect(one.rows == 300, "300 rows, got " + String(one.rows))
    for size in [5, 17, 50, 211]:
        tables_equal(one, read_csv(path, CsvOptions(threads=4, chunk_bytes=size)))
    var broken = write_file("brokenboundaries.csv", text + "300,\"never closed\n301,x\n")
    var b1 = read_csv(broken, CsvOptions(threads=1))
    tables_equal(b1, read_csv(broken, CsvOptions(threads=4, chunk_bytes=40)))
    expect(b1.rows == 301 and b1.ragged_count == 1, "the unclosed quote at the end is reported once")


def test_take_table_leaves_the_report_readable() raises:
    var r = read_csv(write_file("take.csv", "a,b\n1,x\n2,y\n"))
    var shared = r.take_table().share()
    expect(shared.rows() == 2 and r.rows == 2 and r.columns[1].name == "b" and r.table.rows() == 0,
           "the table moves out, the report stays")
    var path = write_file("take.parquet", "")
    write_parquet(path, shared, metadata="{\"omx\":2}")
    var p = read_parquet(path)
    var back = p.take_table().share()
    expect(back.rows() == 2 and p.metadata == "{\"omx\":2}" and p.table.rows() == 0,
           "and a Parquet result's table moves out with its metadata still there")


def main() raises:
    test_take_table_leaves_the_report_readable()
    test_integers()
    test_decimals()
    test_dates()
    test_bools_and_text()
    test_int_and_decimal_mix()
    test_nulls_and_report()
    test_sample_is_the_first_eight()
    test_failed_rows_after_the_window()
    test_quoting()
    test_bom_and_line_endings()
    test_tsv_and_delimiters()
    test_header_options()
    test_ragged_rows()
    test_empty_and_odd_files()
    test_utf8_text()
    test_quote_boundaries_in_chunks()
    test_generated_file_in_parallel()
    test_max_columns_is_not_ragged()
    test_stray_quote_is_literal()
    test_unclosed_quote_is_reported()
    test_timestamp_zones_convert_to_utc()
    test_decimal_widens_to_scale_4()
    test_overflow_inside_the_window_is_text()
    test_stray_quotes_across_chunk_boundaries()
    print("ok")
