# F-138b: the package built for an app with no GPU, no thread pool and no codecs. scripts/mojo_test.sh builds this one
# file with `-D MXFRAME_NO_GPU -D MXFRAME_SERIAL -D MXFRAME_NO_CODECS` (the switches a Mac build has by default) and
# runs it with two Parquet files the normal build wrote: argv[1] zstd, argv[2] snappy (from tools/pq_write_sample).
# Everything an app does must still give the right answer; the three things that are gone must say so readably.
from std.sys import argv
from std.os import getenv
from mxframe import (Table, SharedTable, Query, QueryResult, INT, DECIMAL, col, sum, count, CPU, GPU, AUTO, DeviceTable,
                     gpu_available, read_csv, CsvOptions, write_parquet, read_parquet, scan_parquet, min_of, max_of, ZSTD, SNAPPY)
from mxframe.query import AUTO_GPU_MIN_ROWS
from mxframe.kernels._fpenv import core_count


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


def refusal(path: String) -> String:
    try:
        _ = read_parquet(path)
    except e:
        return String(e)
    return String("")


def main() raises:
    var args = argv()
    # no GPU
    expect(not gpu_available(), "gpu_available() is False")
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    for i in range(AUTO_GPU_MIN_ROWS + 5):                    # over AUTO's row threshold
        t.append(k, Int64(i % 3))
        t.append(v, Int64(i))
    var shared = t^.share()
    var msg = String("")
    try:
        _ = DeviceTable(shared)
    except e:
        msg = String(e)
    expect(msg == "no GPU on this build", "DeviceTable raises: `" + msg + "`")
    try:
        _ = Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count()).compute(GPU)
    except e:
        msg = String(e)
    expect("no GPU on this build" in msg, "compute(GPU) raises: `" + msg + "`")
    var auto = Query(shared).group_by(col("k"), groups=3).agg(sum(col("v")), count()).compute(AUTO)
    expect(auto.device == CPU and auto.fallback_reason == "", "AUTO runs on the CPU, not as a fallback")
    var n = AUTO_GPU_MIN_ROWS + 5
    expect(auto.counts[0] == (n + 2) // 3 and auto.counts[1] == (n + 1) // 3 and auto.counts[2] == n // 3,
           "the serial kernel counts every row: " + String(auto.counts[0]) + " " + String(auto.counts[1]) + " " + String(auto.counts[2]))
    var want0 = 0
    for i in range(0, n, 3):
        want0 += i
    expect(auto.sums[0] == want0, "and sums them exactly: " + String(auto.sums[0]) + " vs " + String(want0))

    # no thread pool: the parallel CSV reader runs its chunks one after the other
    var csv = String("/tmp/mxframe-switches.csv")
    var text = String("a,b\n")
    for i in range(50_000):
        text += String(i) + "," + String(i % 7) + "\n"
    with open(csv, "w") as f:
        f.write(text)
    var r = read_csv(csv)
    expect(r.table.rows() == 50_000, "read_csv rows: " + String(r.table.rows()))
    var total = 0
    for i in range(r.table.rows()):
        total += Int(r.table.get(1, i))
    expect(total == 149_997, "read_csv values: " + String(total))      # 7142 full cycles of 0..6 (21 each) plus 0+1+2+3+4+5 = 15

    # the serial `par` / `par_n` really run every chunk: many small chunks (quoted newlines across their edges)
    # read cell for cell as one chunk does
    expect(core_count() == 1, "core_count() is 1 on a serial build: " + String(core_count()))
    var qcsv = String("/tmp/mxframe-switches-quoted.csv")
    var qtext = String("id,note,n\n")
    for i in range(20_000):
        qtext += String(i) + ",\"line " + String(i) + "\nnext, " + String(i % 13) + "\"," + String(i % 5) + "\n"
    with open(qcsv, "w") as f:
        f.write(qtext)
    var one = read_csv(qcsv, CsvOptions(threads=1))
    var many = read_csv(qcsv, CsvOptions(threads=4, chunk_bytes=4096))
    expect(one.table.rows() == 20_000 and many.table.rows() == 20_000,
           "chunked rows: " + String(one.table.rows()) + " / " + String(many.table.rows()))
    for i in range(20_000):
        expect(many.table.get(0, i) == one.table.get(0, i) and many.table.get(2, i) == one.table.get(2, i) and
               many.table.text_of(1, Int(many.table.get(1, i))) == one.table.text_of(1, Int(one.table.get(1, i))), "chunked read equals one chunk at row " + String(i))

    # no codecs: saves UNCOMPRESSED, whatever is asked for, and the file reads back
    var path = String("/tmp/mxframe-switches.parquet")
    var big = Table.empty()
    var x = big.add_column("x", INT)
    for i in range(1000):
        big.append(x, Int64(i) * 0x9E3779B97F4A7C15)
    write_parquet(path, big, "", ZSTD)
    var size = len(open(path, "r").read_bytes())
    expect(size >= 8000, "the 1,000 values are stored uncompressed (" + String(size) + " bytes)")
    var back = read_parquet(path)
    expect(back.table.rows() == 1000 and back.table.get(0, 999) == Int64(999) * 0x9E3779B97F4A7C15, "and read back exactly")

    # a streamed query (F-140) over it: the same numbers, on the CPU
    var streamed = Query(scan_parquet(path)).agg(count(), min_of(col("x")), max_of(col("x"))).compute(AUTO)
    var whole = Query(back.take_table().share()).agg(count(), min_of(col("x")), max_of(col("x"))).compute(AUTO)
    expect(streamed.device == CPU and streamed.counts[0] == 1000 and streamed.values(1)[0] == whole.values(1)[0] and
           streamed.values(2)[0] == whole.values(2)[0], "a streamed query gives the whole-file answer")

    # ... and refuses a compressed file, naming the column
    expect(len(args) >= 3, "run with a zstd file and a snappy file")
    var z = refusal(String(args[1]))
    expect("`qty`" in z and "zstd" in z and "no zstd or snappy" in z, "a zstd file is refused: `" + z + "`")
    var s = refusal(String(args[2]))
    expect("`qty`" in s and "snappy" in s and "no zstd or snappy" in s, "a snappy file is refused: `" + s + "`")
    var scanned = String("")
    try:
        var zs = scan_parquet(String(args[1]))        # opens (the footer); its text is read lazily
        _ = zs.word_count(zs.index("region"))
    except e:
        scanned = String(e)
    expect("zstd" in scanned and "no zstd or snappy" in scanned, "a zstd file's text cannot be scanned: `" + scanned + "`")
    print("ok")
