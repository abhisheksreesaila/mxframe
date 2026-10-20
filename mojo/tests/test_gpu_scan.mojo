# F-146: a streamed Parquet file on the GPU. The oracle is the CPU scan of the same file (itself equal to read_parquet +
# Query, test_scan.mojo): every slot, count and presence flag identical, nulls, an all-null column in a row group, a
# TEXT key, a frame's edits over the scan, a plan the GPU cannot take. Skips cleanly with no GPU; allocations are modest
# (GPU 0 is shared): 8 row groups of 16,000 rows.
from omxcore import (Table, SharedTable, Frame, Query, QueryResult, col, lit, sum, count, mean, min_of, max_of, count_distinct,
                     INT, DECIMAL, TEXT, write_parquet, scan_parquet, ZSTD, CPU, GPU, AUTO, MORSELS_OFF, gpu_available)


struct Rng(Movable):
    var s: UInt64

    def __init__(out self, seed: UInt64):
        self.s = seed * 0x9E3779B97F4A7C15 + 1

    def next(mut self) -> UInt64:
        self.s ^= self.s << 13
        self.s ^= self.s >> 7
        self.s ^= self.s << 17
        return self.s

    def int(mut self, lo: Int, hi: Int) -> Int:       # [lo, hi]
        return lo + Int(self.next() % UInt64(hi - lo + 1))


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


comptime N = 128_000
comptime GROUP = 16_000


def make_table() raises -> SharedTable:
    """region TEXT (8 words), carrier INT -1..7 (null for the whole of row group 5), revenue DECIMAL(2) (null for the whole
    of row group 2, otherwise some nulls; every 50th past 2^53), cost INT."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var c = t.add_column("carrier", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", INT)
    var words: List[String] = ["West", "East", "North", "South", "Central", "Nörth-East", "", "Far"]
    var rng = Rng(11)
    for i in range(N):
        var rg = i // GROUP
        t.append_text(g, words[(rg + rng.int(0, 2)) % 8])
        if rg == 5 or i % 29 == 3:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if rg == 2 or i % 13 == 1:
            t.append_null(r)
        elif i % 50 == 0:
            var big = Int64(1 << 53) + Int64(rng.int(0, 1000))
            t.append(r, big if (i // 50) % 2 == 0 else -big + 7)
        else:
            t.append(r, Int64(rng.int(-1_000_000, 1_000_000)))
        t.append(k, Int64(rng.int(-5000, 5000)))
    return t^.share()


def same(a: QueryResult, b: QueryResult, label: String) raises:
    expect(len(a.kinds) == len(b.kinds) and len(a.slots) == len(b.slots), label + ": same layout")
    for i in range(len(a.kinds)):
        expect(a.slot_of[i] == b.slot_of[i] and a.count_slot_of[i] == b.count_slot_of[i], label + ": same slots")
    for s in range(len(a.slots)):
        for g in range(len(a.slots[s])):
            expect(a.slots[s][g] == b.slots[s][g], label + ": slot " + String(s) + " group " + String(g) + ": gpu " +
                   String(a.slots[s][g]) + ", cpu " + String(b.slots[s][g]))
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g))
        for i in range(len(a.kinds)):
            expect(a.present(i, g) == b.present(i, g), label + ": present(" + String(i) + ", " + String(g) + ")")
    expect(len(a.sums) == len(b.sums), label + ": sums")


def test_the_gpu_scan_equals_the_cpu_scan() raises:
    write_parquet("f146_scan.parquet", make_table(), "", ZSTD, GROUP)
    var scan = scan_parquet("f146_scan.parquet")
    var nreg = scan.word_count(scan.index("region"))
    var far = scan.code_of(scan.index("region"), "Far")
    # (CPU: the group-at-a-time loop and the morsels; GPU: the same file on the device)
    var a = Query(scan).group_by(col("region"), groups=nreg).agg(sum(col("revenue")), count(), count(col("revenue")),
                                                                 min_of(col("cost")), max_of(col("cost")), mean(col("revenue")),
                                                                 min_of(col("revenue")), max_of(col("revenue"))).compute(GPU)
    expect(a.device == GPU and a.fallback_reason == "", "the scan ran on the GPU")
    same(a, Query(scan).morsels(MORSELS_OFF).group_by(col("region"), groups=nreg).agg(
        sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")), mean(col("revenue")),
        min_of(col("revenue")), max_of(col("revenue"))).compute(CPU), "text key")
    same(Query(scan).where(col("region") == lit(far)).group_by(col("carrier"), groups=7).agg(
            sum(col("revenue")), count(), min_of(col("revenue"))).compute(GPU),
         Query(scan).where(col("region") == lit(far)).group_by(col("carrier"), groups=7).agg(
            sum(col("revenue")), count(), min_of(col("revenue"))).compute(CPU), "filter on a word")
    same(Query(scan).group_by(col("region") * 7 + col("carrier"), groups=nreg * 7).agg(sum(col("cost")), count(col("revenue"))).compute(GPU),
         Query(scan).group_by(col("region") * 7 + col("carrier"), groups=nreg * 7).agg(sum(col("cost")), count(col("revenue"))).compute(CPU), "pivot")
    same(Query(scan).where(col("cost").between(0, 100)).agg(sum(col("cost")), count()).compute(GPU),
         Query(scan).where(col("cost").between(0, 100)).agg(sum(col("cost")), count()).compute(CPU), "global")
    same(Query(scan).agg(count()).compute(GPU), Query(scan).agg(count()).compute(CPU), "only count()")
    same(Query(scan).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(GPU),
         Query(scan).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(CPU), "nothing kept")


def test_auto_says_why_it_chose_the_cpu_and_can_choose_the_gpu() raises:
    var scan = scan_parquet("f146_scan.parquet")
    var small = Query(scan).group_by(col("carrier"), groups=7).agg(sum(col("cost")), count()).compute(AUTO)
    expect(small.device == CPU and "under AUTO's" in small.fallback_reason, "a small file: the CPU, and why: `" + small.fallback_reason + "`")
    var cheap = Query(scan).auto_gpu_ops(8).group_by(col("carrier"), groups=7).agg(sum(col("cost")), count()).compute(AUTO)
    expect(cheap.device == CPU and cheap.fallback_reason != "", "still a small file")
    var forced = Query(scan).auto_gpu_ops(0).group_by(col("carrier"), groups=7).agg(sum(col("cost")), count()).compute(AUTO)
    expect(forced.device == GPU and forced.fallback_reason == "", "auto_gpu_ops(0): every file to the GPU")
    same(forced, Query(scan).group_by(col("carrier"), groups=7).agg(sum(col("cost")), count()).compute(CPU), "AUTO on the GPU")


def test_holistic_aggregates_stay_on_the_cpu_with_a_reason() raises:
    var scan = scan_parquet("f146_scan.parquet")
    var r = Query(scan).auto_gpu_ops(0).group_by(col("carrier"), groups=7).agg(count_distinct(col("cost")), sum(col("cost"))).compute(AUTO)
    expect(r.device == CPU and "count_distinct" in r.fallback_reason, "AUTO: holistic on the CPU, and why: `" + r.fallback_reason + "`")
    var msg = String("")
    try:
        _ = Query(scan).group_by(col("carrier"), groups=7).agg(count_distinct(col("cost"))).compute(GPU)
    except e:
        msg = String(e)
    expect("count_distinct" in msg, "compute(GPU) says so: `" + msg + "`")


def test_a_frame_over_the_scan_on_the_gpu() raises:
    """The frame's mask is uploaded with each row group; its changed and appended rows are added on the CPU after."""
    var scan = scan_parquet("f146_scan.parquet")
    var f = Frame(scan)
    var cost = f.column("cost")
    var region = f.column("region")
    var rng = Rng(5)
    for _ in range(300):
        f.set(cost, rng.int(0, N - 1), Int64(rng.int(-9000, 9000)))
    for _ in range(40):
        f.set_text(region, rng.int(0, N - 1), "Fresh")
    for _ in range(60):
        f.remove(rng.int(0, N - 1))
    for _ in range(5):
        _ = f.append(["Appended", "3", "1.50", "77"])
    var n = f.word_count(region)
    same(Query(f).group_by(col("region"), groups=n).agg(sum(col("cost")), count(), min_of(col("cost")), mean(col("revenue"))).compute(GPU),
         Query(f).morsels(MORSELS_OFF).group_by(col("region"), groups=n).agg(sum(col("cost")), count(), min_of(col("cost")), mean(col("revenue"))).compute(CPU),
         "frame over a scan")


def test_overflow_raises_the_same_error() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    for _ in range(64):
        t.append(v, Int64.MAX // 8)
    write_parquet("f146_ovf.parquet", t, "", ZSTD, 8)
    var msg = String("")
    try:
        _ = Query(scan_parquet("f146_ovf.parquet")).agg(sum(col("v"))).compute(GPU)
    except e:
        msg = String(e)
    var want = String("")
    try:
        _ = Query(scan_parquet("f146_ovf.parquet")).agg(sum(col("v"))).compute(CPU)
    except e:
        want = String(e)
    expect(want != "" and msg == want, "overflow on the GPU: `" + msg + "` vs `" + want + "`")


def main() raises:
    if not gpu_available():
        print("no GPU: skipped")
        return
    test_the_gpu_scan_equals_the_cpu_scan()
    test_auto_says_why_it_chose_the_cpu_and_can_choose_the_gpu()
    test_holistic_aggregates_stay_on_the_cpu_with_a_reason()
    test_a_frame_over_the_scan_on_the_gpu()
    test_overflow_raises_the_same_error()
    print("all gpu scan tests passed")
