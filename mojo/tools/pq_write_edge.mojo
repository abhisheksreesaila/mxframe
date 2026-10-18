# Writes the awkward tables the Parquet writer must handle or refuse (used by tests/test_mojo_parquet.py):
#   pq_write_edge <case> <path>
# cases: decimal_overflow, no_columns, null_text (13 rows in groups of 8, a TEXT column of nulls with no words),
#        atomic (a good file, then a refused write over it).
from std.sys import argv
from mxframe import Table, INT, DECIMAL, TEXT, write_parquet


def main() raises:
    var args = argv()
    var case = String(args[1])
    var path = String(args[2])
    var t = Table.empty()
    if case == "decimal_overflow" or case == "atomic":
        var ok = Table.empty()
        var a = ok.add_column("qty", INT)
        ok.append(a, 41)
        ok.append(a, 42)
        if case == "atomic":
            write_parquet(path, ok)
        var m = t.add_column("money", DECIMAL, scale=2)
        t.append(m, 5)
        t.append(m, 1_000_000_000_000_000_000)          # 10^18: one digit too many for DECIMAL(18, 2)
        if case == "atomic":
            try:
                write_parquet(path, t)
            except e:
                print("refused: " + String(e))
            return
    elif case == "null_text":
        var id = t.add_column("id", INT)
        var s = t.add_column("name", TEXT)
        for i in range(13):
            t.append(id, Int64(i))
            t.append_null(s)
    write_parquet(path, t, row_group_rows=8 if case == "null_text" else 1 << 20)
