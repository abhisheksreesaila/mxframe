# F-138b: a small program over the whole package, compiled as an object for arm64-apple-macosx14.0 by
# scripts/check_mac_link.sh and linked the way the omxplorer Mac build links. It is never run: it exists so the
# linker sees every path an app uses (CSV import, queries on the CPU / AUTO / GPU, a frame's edits, Parquet save
# and open). `mojo_entry` is the entry point the app's Mac build links with (`-e _mojo_entry`).
from std.ffi import c_int
from mxframe import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, Query, col, sum, count, CPU, GPU, AUTO,
                     DeviceTable, gpu_available, read_csv, write_parquet, read_parquet, scan_parquet, UNCOMPRESSED, ZSTD,
                     count_distinct, median, percentile, top_k)


def run() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", DECIMAL, scale=2)
    for i in range(10):
        t.append(k, Int64(i % 3))
        t.append(v, Int64(i * 100))
    var shared = t^.share()
    var f = Frame(shared)
    f.set(1, 2, Int64(7))
    var q = Query(f).group_by(col("k"), 3).agg(sum(col("v")), count())
    var r = q.compute(AUTO)
    print(r.counts[0], r.device, gpu_available())
    var h = Query(f).group_by(col("k"), 3).agg(count_distinct(col("v")), median(col("v")), percentile(col("v"), 0.9), top_k(col("v"), 2)).compute(AUTO)
    print(h.values(0)[0], h.values(1)[0], h.top(3, 0).__len__())
    try:
        var d = DeviceTable(shared)                  # raises "no GPU on this build" off Linux
        _ = Query(shared).on_device(d).agg(count()).compute(GPU)
    except e:
        print(e)
    write_parquet("probe.parquet", shared, "{}", UNCOMPRESSED)
    try:
        write_parquet("probe-z.parquet", shared, "{}", ZSTD)     # saved UNCOMPRESSED on a no-codec build
    except e:
        print(e)
    var back = read_parquet("probe.parquet")
    print(back.table.rows())
    print(Query(scan_parquet("probe.parquet")).group_by(col("k"), 3).agg(sum(col("v")), count()).compute(AUTO).counts[0])
    var sf = Frame(scan_parquet("probe.parquet"))          # F-147: edits over a streamed file, queried, sorted, saved
    sf.set(1, 2, Int64(9))
    sf.remove(0)
    print(Query(sf).group_by(col("k"), 3).agg(sum(col("v")), count()).compute(AUTO).counts[0], sf.permutation(1).__len__(), sf.get(1, 2))
    sf.save("probe-edited.parquet", "{}", ZSTD)
    var c = read_csv("probe.csv")
    print(c.table.rows())


@export("mojo_entry")
def entry(argc: c_int, argv: Int) abi("C") -> c_int:
    try:
        run()
        return 0
    except e:
        print(e)
        return 1
