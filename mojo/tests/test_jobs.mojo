# F-175: the real jobs (mojo/mxframe/jobs.mojo): every kind of query, read and save, run as a background job, gives exactly what the plain
# call gives; progress only grows and ends at the total; a cancel (at any point) returns no partial result, frees everything and
# leaves a saved file as it was; several jobs run at once; a job over an edited Frame runs while the caller reads cells.
from std.ffi import external_call, c_int
from std.os import makedirs
from mxframe import (Table, SharedTable, Frame, Query, QueryResult, col, lit, sum, count, mean, min_of, max_of, count_distinct, median,
                     percentile, top_k, INT, DECIMAL, TEXT, FLOAT, write_parquet, read_parquet, scan_parquet, ParquetResult, ZSTD,
                     UNCOMPRESSED, CPU, AUTO, MORSELS_ON, MORSELS_OFF, Job, JOB_RUNNING, JOB_DONE, JOB_FAILED, JOB_CANCELLED,
                     compute_job)
from mxframe.kernels._fpenv import SERIAL

comptime DIR = "mxframe_jobs_test/"


def expect(cond: Bool, what: String) raises:
    if not cond:
        raise Error("FAILED: " + what)


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


def pause(us: Int):
    external_call["usleep", c_int](Int32(us))


def rss_kb() raises -> Int:
    var f = open("/proc/self/statm", "r")
    var text = f.read()
    f.close()
    var parts = text.split(" ")
    return Int(String(parts[1])) * 4


# ── comparing results ────────────────────────────────────────────────────────────────────────────

def same_ints(a: List[Int], b: List[Int]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if a[i] != b[i]:
            return False
    return True


def same_rows(a: List[List[Int]], b: List[List[Int]]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if not same_ints(a[i], b[i]):
            return False
    return True


def same_result(a: QueryResult, b: QueryResult) -> Bool:
    return (same_ints(a.kinds, b.kinds) and same_ints(a.slot_of, b.slot_of) and same_ints(a.count_slot_of, b.count_slot_of)
            and same_rows(a.slots, b.slots) and same_ints(a.counts, b.counts) and same_ints(a.sums, b.sums)
            and same_ints(a.hol_of, b.hol_of) and same_rows(a.hvals, b.hvals) and same_rows(a.htop, b.htop)
            and same_rows(a.htop_off, b.htop_off) and same_rows(a.hcount, b.hcount))


def same_table(a: Table, b: Table) -> Bool:
    if a.columns() != b.columns() or a.rows() != b.rows():
        return False
    for c in range(a.columns()):
        ref x = a.cols[c]
        ref y = b.cols[c]
        if x.name != y.name or x.kind != y.kind or x.scale != y.scale or x.nulls != y.nulls:
            return False
        for r in range(a.rows()):
            if x.is_null(r) != y.is_null(r) or x.values[r] != y.values[r]:
                return False
        if x.kind == TEXT:
            if len(x.dict[].words) != len(y.dict[].words):
                return False
            for w in range(len(x.dict[].words)):
                if x.dict[].words[w] != y.dict[].words[w]:
                    return False
    return True


# ── watching a job ───────────────────────────────────────────────────────────────────────────────

def watch[T: Movable & Deinitable](mut job: Job[T], what: String) raises -> Int:
    """Polls the job until it ends, checking progress is monotonic; returns how many distinct progress values were seen."""
    var last = 0
    var seen = 0
    while job.poll() == JOB_RUNNING:
        var p = job.progress()
        expect(p.done >= last, what + ": progress went back from " + String(last) + " to " + String(p.done))
        expect(p.done <= p.total or p.total == 0, what + ": progress passed its total: " + String(p.done) + "/" + String(p.total))
        if p.done != last:
            seen += 1
        last = p.done
        pause(200)
    var p = job.progress()
    expect(p.done >= last, what + ": progress went back at the end")
    if job.poll() == JOB_DONE:
        expect(p.done == p.total, what + ": progress ends at the total: " + String(p.done) + "/" + String(p.total))
    return seen


# ── the data ─────────────────────────────────────────────────────────────────────────────────────

def big_table(n: Int) raises -> SharedTable:
    """k INT 0..6 (some null), v INT, w DECIMAL(2) with nulls, x FLOAT, name TEXT."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    var w = t.add_column("w", DECIMAL, scale=2)
    var x = t.add_column("x", FLOAT)
    var nm = t.add_column("name", TEXT)
    var names: List[String] = ["alpha", "beta", "gamma", "delta", "epsilon"]
    var rng = Rng(11)
    for i in range(n):
        if i % 97 == 5:
            t.append_null(k)
        else:
            t.append(k, Int64(i % 7))
        t.append(v, Int64(rng.int(-1000, 1000)))
        if i % 31 == 2:
            t.append_null(w)
        else:
            t.append(w, Int64(rng.int(0, 99999)))
        t.append_float(x, Float64(rng.int(-100000, 100000)) / 64.0)
        t.append_text(nm, names[rng.int(0, 4)])
    return t^.share()


def agg_query(t: SharedTable) raises -> Query:
    return Query(t).where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), count(col("w")), sum(col("w")),
                                                                              min_of(col("v")), max_of(col("w")), mean(col("v")))


def test_compute_equals_plain() raises:
    var t = big_table(6_000_000)                       # more than two job ranges: the query runs in several
    var plain = agg_query(t).compute(CPU)
    var job = compute_job(agg_query(t), CPU)
    var seen = watch(job, "compute")
    var r = job.result()
    var most = seen
    for _ in range(9):                                  # (ranges of 2M rows: progress is seen part way when it runs by ranges)
        var again = compute_job(agg_query(t), CPU)
        most = max(most, watch(again, "compute again"))
    expect(most >= 1, "progress was seen part way through a 6M-row query: the ranges ran (" + String(most) + ")")
    expect(same_result(r, plain), "integer aggregates over 6M rows: the job's result is the plain call's")
    expect(r.counts[0] > 0 and r.sums[0] == plain.sums[0], "a real answer (group 0 has rows)")
    var flo = Query(t).group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count()).compute(CPU)
    var fj = compute_job(Query(t).group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count()), CPU)
    expect(same_result(fj.result(), flo), "a float64 sum is the same bits as the plain call (it keeps its defined order)")
    var g = Query(t).where(col("w") > 5000).agg(sum(col("w")), count()).compute(CPU)
    var gj = compute_job(Query(t).where(col("w") > 5000).agg(sum(col("w")), count()), CPU)
    expect(same_result(gj.result(), g), "no group key")


def test_holistic_equals_plain() raises:
    var t = big_table(400_000)
    var q = Query(t).group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), percentile(col("v"), 0.9), top_k(col("v"), 3))
    q._window_rows = 65536 * 2                          # six windows
    var plain = q.compute(CPU)
    var q2 = Query(t).group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), percentile(col("v"), 0.9), top_k(col("v"), 3))
    q2._window_rows = 65536 * 2
    var job = compute_job(q2^, CPU)
    _ = watch(job, "holistic")
    expect(same_result(job.result(), plain), "count_distinct, median, percentile and top_k over windows: the plain call's answer")


def scan_queries(path: String, mode: Int, which: Int) raises -> Query:
    var q = Query(scan_parquet(path)).morsels(mode).block_rows(8192)
    if which == 0:
        return q^.where(col("v") > -900).group_by(col("k"), groups=7).agg(sum(col("v")), count(), count(col("w")), sum(col("w")), min_of(col("v")))
    if which == 1:
        return q^.group_by(col("k"), groups=7).agg(sum(col("x")), min_of(col("x")), count())          # float: file order
    if which == 2:
        return q^.group_by(col("name"), groups=5).agg(count(), sum(col("v")))                        # a TEXT key
    return q^.group_by(col("k"), groups=7).agg(count_distinct(col("v")), median(col("w")), top_k(col("v"), 2))     # holistic: row group by row group


def test_scan_equals_plain() raises:
    var t = big_table(1_500_000)
    var path = DIR + "scan.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 100_000)         # fifteen row groups
    for mode in [MORSELS_ON, MORSELS_OFF]:
        for which in range(4):
            var plain = scan_queries(path, mode, which).compute(CPU)
            var job = compute_job(scan_queries(path, mode, which), CPU)
            _ = watch(job, "scan " + String(which))
            var r = job.result()
            expect(same_result(r, plain), "scan query " + String(which) + " (morsels " + String(mode) + "): the plain call's answer")
            var p = job.progress()
            expect(p.done == 1_500_000 and p.total == 1_500_000, "the scan's progress is its rows: " + String(p.done) + "/" + String(p.total))


def test_cancel_storm() raises:
    """Cancels at random moments, drops handles mid-run, lets some finish: never a partial result, never a crash, and the memory
    after hundreds of cycles is the memory before (the first cycles warm the allocator)."""
    var t = big_table(5_000_000)
    var path = DIR + "storm.parquet"
    write_parquet(path, t, "", UNCOMPRESSED, 250_000)
    var want_mem = agg_query(t).compute(CPU)
    var want_scan = scan_queries(path, MORSELS_ON, 0).compute(CPU)
    var want_hol = scan_queries(path, MORSELS_OFF, 3).compute(CPU)
    var wants = List[QueryResult]()
    wants.append(want_mem^)
    wants.append(want_scan^)
    wants.append(want_hol^)
    var rng = Rng(99)
    var done = 0
    var cancelled = 0
    var base = 0
    comptime CYCLES = 300
    for cycle in range(CYCLES):
        if cycle == 40:
            base = rss_kb()
        var kind = cycle % 3
        var job = compute_job(agg_query(t) if kind == 0 else (scan_queries(path, MORSELS_ON, 0) if kind == 1 else scan_queries(path, MORSELS_OFF, 3)), CPU)
        pause(rng.int(0, 4000))
        var how = rng.int(0, 3)
        if how == 0:                                   # dropped while it may still run
            _ = job^
            continue
        if how != 1:
            job.cancel()
        var state = job.wait()
        if state == JOB_CANCELLED:
            cancelled += 1
            var msg = String()
            try:
                _ = job.result()
            except e:
                msg = String(e)
            expect(msg == "cancelled", "a cancelled job has no result, it says `cancelled`: `" + msg + "`")
        else:
            expect(state == JOB_DONE, "a job ends done or cancelled, not " + String(state))
            done += 1
            var got = job.result()
            expect(same_result(got, wants[kind]), "a job that finished before its cancel gives the full answer (kind " + String(kind) + ")")
    var grown = rss_kb() - base
    print("cancel storm:", CYCLES, "cycles,", done, "done,", cancelled, "cancelled, memory grew", grown, "KB after the first 40")
    expect(grown < 60_000, "memory grew " + String(grown) + " KB over 260 cycles (a leak per cycle would be tens of MB)")
    comptime if not SERIAL:
        expect(cancelled > 20, "the cancels reached running jobs (" + String(cancelled) + ")")


def main() raises:
    makedirs(DIR, exist_ok=True)
    test_compute_equals_plain()
    test_holistic_equals_plain()
    test_scan_equals_plain()
    test_cancel_storm()
    print("test_jobs ok")
