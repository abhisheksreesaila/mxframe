# Reads the Parquet file named on the command line and prints its table as JSON lines (used by tests/test_mojo_parquet.py):
# a first line {"columns": [[name, kind, scale], ...], "metadata": "..."}, then one line per row with the stored
# Int64 values (TEXT as the word), null as null.
from std.sys import argv
from omxcore import read_parquet, TEXT


def main() raises:
    var r = read_parquet(String(argv()[1]))
    ref t = r.table
    var head = String('{"columns": [')
    for c in range(t.columns()):
        if c > 0:
            head += ", "
        head += '["' + t.name(c) + '", "' + String(t.kind(c)) + '", ' + String(t.scale(c)) + "]"
    print(head + '], "metadata_bytes": ' + String(r.metadata.byte_length()) + "}")
    for row in range(t.rows()):
        var line = String("[")
        for c in range(t.columns()):
            if c > 0:
                line += ", "
            if t.is_null(c, row):
                line += "null"
            elif t.kind(c) == TEXT:
                line += "{\"word\": " + String(Int(t.get(c, row))) + "}"
            else:
                line += String(t.get(c, row))
        print(line + "]")
    for c in range(t.columns()):
        if t.kind(c) == TEXT:
            var words = String("")
            for k in range(len(t.dictionary(c))):
                words += t.dictionary(c)[k] + "\x1f"
            print("WORDS " + String(c) + " " + words)
