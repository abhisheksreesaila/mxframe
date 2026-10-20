# Writes a sample table to the Parquet file named on the command line (used by tests/test_mojo_parquet.py).
from std.sys import argv
from omxcore import Table, INT, DECIMAL, TEXT, DATE, BOOL, write_parquet, UNCOMPRESSED, SNAPPY, ZSTD


def main() raises:
    var args = argv()
    var path = String(args[1])
    var codec = ZSTD if String(args[2]) == "zstd" else SNAPPY if String(args[2]) == "snappy" else UNCOMPRESSED
    var n = Int(String(args[3]))
    var group = Int(String(args[4]))
    var t = Table.empty()
    var q = t.add_column("qty", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var g = t.add_column("region", TEXT)
    var d = t.add_column("day", DATE)
    var b = t.add_column("paid", BOOL)
    var words: List[String] = ["West", "East", "Nörth", "", "South-East"]
    for w in words:
        _ = t.add_word(g, w)
    for i in range(n):
        t.append(q, Int64(i * 7 - 3000))
        if i % 13 == 4:
            t.append_null(r)
        else:
            t.append(r, Int64(i) * 1_000_003 + Int64(1 << 53) if i % 97 == 0 else Int64(i * 101 - 50_000))
        if i % 17 == 2:
            t.append_null(g)
        else:
            t.append(g, Int64(i % 5))
        t.append(d, Int64(18_000 + i % 400))
        if i % 11 == 6:
            t.append_null(b)
        else:
            t.append(b, Int64(i % 3 == 0))
    write_parquet(path, t, metadata='{"formats": {"revenue": "money"}}', compression=codec, row_group_rows=group)
