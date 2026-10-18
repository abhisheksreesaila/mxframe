# F-141: times Parquet decoding. Usage: pq_decode_bench <path> <scan|read> <numeric|text>
# scan: Query(scan_parquet(path)) end to end.  read: read_parquet(path) (decode only), then the same query in memory.
# Prints one JSON line: seconds for the decode+query (scan) or decode (read), the in-memory query's milliseconds, and a
# checksum of the answer, so a before / after run can be compared.
from std.sys import argv
from std.time import perf_counter_ns
from mxframe import Query, col, sum, count, min_of, max_of, read_parquet, scan_parquet, AUTO
from mxframe.kernels._fpenv import core_count


def plan(var q: Query, which: String) raises -> Query:
    if which == "numeric":
        return q^.group_by(col("carrier"), groups=10).agg(sum(col("cost")), sum(col("revenue")), count(), max_of(col("day")))
    return q^.group_by(col("region"), groups=8).agg(count(), sum(col("cost")))


def main() raises:
    var args = argv()
    var path = String(args[1])
    var mode = String(args[2])
    var which = String(args[3])
    var t0 = perf_counter_ns()
    var check = 0
    var qms = 0.0
    var rows = 0
    if mode == "scan":
        var scan = scan_parquet(path)
        rows = scan.rows()
        var r = plan(Query(scan), which).compute(AUTO)
        for s in range(len(r.slots)):
            for v in r.slots[s]:
                check = check * 31 + v
        for c in r.counts:
            check = check * 31 + c
    else:
        var rp = read_parquet(path)
        var t = rp.take_table().share()
        rows = t.rows()
        var t1 = perf_counter_ns()
        var r = plan(Query(t), which).compute(AUTO)
        qms = Float64(perf_counter_ns() - t1) / 1e6
        for s in range(len(r.slots)):
            for v in r.slots[s]:
                check = check * 31 + v
        for c in r.counts:
            check = check * 31 + c
        # words and a few cells: the table itself must be the same too
        for c in range(t.columns()):
            check = check * 31 + Int(t.get(c, rows - 1)) + Int(t.get(c, rows // 2)) + t.null_count(c)
        print('{"mode": "read", "rows": ' + String(rows) + ', "decode_s": ' + String(Float64(t1 - t0) / 1e9) +
              ', "kernel_ms": ' + String(qms) + ', "threads": ' + String(core_count()) + ', "check": ' + String(check) + "}")
        return
    print('{"mode": "scan", "rows": ' + String(rows) + ', "total_s": ' + String(Float64(perf_counter_ns() - t0) / 1e9) +
          ', "threads": ' + String(core_count()) + ', "check": ' + String(check) + "}")
