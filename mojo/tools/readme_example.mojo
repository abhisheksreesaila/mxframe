# The README's Mojo example (kept compiling: scripts/mojo_test.sh runs it)
from std.sys import argv
from omxcore import read_csv, CsvOptions, Query, Frame, col, lit, sum, count, AUTO


def main() raises:
    var csv = read_csv(String(argv()[1]), CsvOptions())  # parallel, typed, with an import report
    var orders = csv.take_table().share()
    var region = orders.column("region")
    var groups = len(orders.dictionary(region))

    var out = (Query(orders)
               .where(col("qty") > 2)
               .group_by(col("region"), groups=groups)
               .agg(sum(col("revenue")), count())
               .compute(AUTO))                          # CPU or GPU; out.device says which
    for g in range(groups):
        if out.present(1, g):
            print(orders.text_of(region, g), out.values(0)[g], out.values(1)[g])

    var f = Frame(orders)                              # edits on top of shared data, no copy
    f.set_text(f.column("revenue"), 0, "99.00")
    f.formula("net = revenue - qty")
    var total = Query(f).agg(sum(col("net"))).compute()
    print("net", total.values(0)[0], "edits", f.edits)
