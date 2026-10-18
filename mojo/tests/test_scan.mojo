# F-140: a query over a Parquet file one row group at a time (`Query(scan_parquet(path))`) gives exactly the answer of
# `read_parquet(path)` + the same query: every slot, count, sum, min / max, mean and `present`, TEXT keys and filters
# through the file-wide dictionary, nulls, a row group where a column is all null, overflow.
from std.ffi import external_call
from mxframe import (Table, SharedTable, Query, QueryResult, col, lit, least, sum, count, mean, min_of, max_of,
                     INT, DECIMAL, TEXT, write_parquet, read_parquet, scan_parquet, ParquetScan, ZSTD, SNAPPY,
                     UNCOMPRESSED, CPU, GPU, AUTO, MORSELS_ON, MORSELS_OFF, gpu_available)


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


comptime N = 120_000
comptime GROUP = 16_000          # 7.5 row groups: the last holds 8,000 rows


def make_table() raises -> SharedTable:
    """region TEXT (8 words, later row groups bring new ones and drop old ones), note TEXT with nulls, carrier INT
    0..6 (with out-of-range keys; null for the whole of row group 5), revenue DECIMAL(2) (null for the whole of row
    group 2, otherwise some nulls), cost INT."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var nt = t.add_column("note", TEXT)
    var c = t.add_column("carrier", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", INT)
    var words: List[String] = ["West", "East", "North", "South", "Central", "Nörth-East", "", "Far"]
    var rng = Rng(7)
    for i in range(N):
        var rg = i // GROUP
        t.append_text(g, words[(rg + rng.int(0, 2)) % 8])
        if i % 5 == 0:
            t.append_null(nt)
        else:
            t.append_text(nt, "n" + String(rng.int(0, 3 + rg)))
        if rg == 5 or i % 29 == 3:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if rg == 2 or i % 13 == 1:
            t.append_null(r)
        elif i % 50 == 0:                                    # past 2^53, sparse enough that no total passes int64
            var big = Int64(1 << 53) + Int64(rng.int(0, 1000))
            t.append(r, big if (i // 50) % 2 == 0 else -big + 7)
        else:
            t.append(r, Int64(rng.int(-1_000_000, 1_000_000)))
        t.append(k, Int64(rng.int(-5000, 5000)))
    return t^.share()


def same(a: QueryResult, b: QueryResult, label: String) raises:
    """Every field a caller can read."""
    expect(len(a.kinds) == len(b.kinds), label + ": same aggregates")
    for i in range(len(a.kinds)):
        expect(a.kinds[i] == b.kinds[i] and a.slot_of[i] == b.slot_of[i] and a.count_slot_of[i] == b.count_slot_of[i],
               label + ": aggregate " + String(i) + " has the same slots")
    expect(len(a.slots) == len(b.slots), label + ": same slot count " + String(len(a.slots)) + " vs " + String(len(b.slots)))
    for s in range(len(a.slots)):
        expect(len(a.slots[s]) == len(b.slots[s]), label + ": slot " + String(s) + " has the same groups")
        for g in range(len(a.slots[s])):
            expect(a.slots[s][g] == b.slots[s][g], label + ": slot " + String(s) + " group " + String(g) + " is " +
                   String(a.slots[s][g]) + ", wanted " + String(b.slots[s][g]))
    expect(len(a.counts) == len(b.counts), label + ": same groups")
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g) + " is " + String(a.counts[g]) +
               ", wanted " + String(b.counts[g]))
        expect(a.is_empty(g) == b.is_empty(g), label + ": empty group " + String(g))
        for i in range(len(a.kinds)):
            expect(a.present(i, g) == b.present(i, g), label + ": present(" + String(i) + ", " + String(g) + ")")
    expect(len(a.sums) == len(b.sums), label + ": sums")
    for g in range(len(a.sums)):
        expect(a.sums[g] == b.sums[g], label + ": sums group " + String(g))


def check_all(path: String, mem: SharedTable, mode: Int = MORSELS_OFF) raises:
    var scan = scan_parquet(path)
    expect(scan.rows() == N and scan.row_groups() == 8, "the scan knows the file's shape: " + String(scan.rows()) +
           " rows in " + String(scan.row_groups()) + " row groups")
    var region = mem.index("region")
    var note = mem.index("note")
    expect(scan.index("region") == region and scan.columns() == mem.columns(), "same columns")
    for c in [region, note]:
        expect(scan.word_count(c) == len(mem.dictionary(c)), "the file-wide dictionary of `" + mem.name(c) + "` has every word")
        for w in range(len(mem.dictionary(c))):
            expect(scan.text_of(c, w) == mem.text_of(c, w) and scan.code_of(c, mem.text_of(c, w)) == w,
                   "a word keeps its code, as in the in-memory read")
    var nreg = scan.word_count(region)
    var nnote = scan.word_count(note)
    var far = mem.code_of(region, "Far")
    expect(far > 0, "Far is a word the file has")

    # 1. a TEXT key, every kind of aggregate, nulls, an all-null row group
    same(Query(scan).morsels(mode).block_rows(1500).group_by(col("region"), groups=nreg)
         .agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")),
              min_of(col("revenue")), max_of(col("revenue")), mean(col("revenue")), mean(col("cost"))).compute(),
         Query(mem).group_by(col("region"), groups=nreg)
         .agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")),
              min_of(col("revenue")), max_of(col("revenue")), mean(col("revenue")), mean(col("cost"))).compute(), "text key")
    # 2. a filter on a TEXT code and a column that is null in a whole row group
    same(Query(scan).morsels(mode).block_rows(1500).where(col("region") == lit(far)).group_by(col("carrier"), groups=7)
         .agg(sum(col("revenue")), count(), min_of(col("revenue"))).compute(),
         Query(mem).where(col("region") == lit(far)).group_by(col("carrier"), groups=7)
         .agg(sum(col("revenue")), count(), min_of(col("revenue"))).compute(), "filter on a word")
    # 3. a pivot, a bin, a nullable TEXT key
    same(Query(scan).morsels(mode).block_rows(1500).group_by(col("region") * 7 + col("carrier"), groups=nreg * 7).agg(sum(col("cost")), count(col("revenue"))).compute(),
         Query(mem).group_by(col("region") * 7 + col("carrier"), groups=nreg * 7).agg(sum(col("cost")), count(col("revenue"))).compute(), "pivot")
    same(Query(scan).morsels(mode).block_rows(1500).group_by(col("note"), groups=nnote).agg(count(), sum(col("cost"))).compute(),
         Query(mem).group_by(col("note"), groups=nnote).agg(count(), sum(col("cost"))).compute(), "nullable text key")
    # 4. one group, only count(), a filter that keeps nothing
    same(Query(scan).morsels(mode).block_rows(1500).where(col("cost").between(0, 100)).agg(sum(col("cost")), count()).compute(),
         Query(mem).where(col("cost").between(0, 100)).agg(sum(col("cost")), count()).compute(), "global")
    same(Query(scan).morsels(mode).block_rows(1500).agg(count()).compute(), Query(mem).agg(count()).compute(), "only count()")
    same(Query(scan).morsels(mode).block_rows(1500).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(),
         Query(mem).where(col("cost") > 99999).group_by(col("carrier"), groups=7).agg(sum(col("revenue")), max_of(col("cost"))).compute(), "nothing kept")
    # the result says where it ran (a GPU, if there is one, is not worth this file: AUTO says so)
    var r = Query(scan).morsels(mode).block_rows(1500).agg(count()).compute(AUTO)
    expect(r.device == CPU and (r.fallback_reason == "" or "under AUTO's" in r.fallback_reason), "AUTO on a small scan runs on the CPU")
    if not gpu_available():
        var raised = False
        try:
            _ = Query(scan).morsels(mode).block_rows(1500).agg(count()).compute(GPU)
        except e:
            raised = "no GPU" in String(e)
        expect(raised, "compute(GPU) on a scan says there is no GPU")


def test_equals_read_parquet_for_every_codec() raises:
    var mem = make_table()
    var codecs: List[Int] = [ZSTD, SNAPPY, UNCOMPRESSED]
    for c in codecs:
        var path = String("/tmp/f140_scan_") + String(c) + ".parquet"
        write_parquet(path, mem, "{}", c, GROUP)
        var rp = read_parquet(path)
        var read = rp.take_table().share()
        check_all(path, read)
        # and the table that was written, not only the table that was read back
        check_all(path, mem)
        # by row-range morsels in blocks of 1,500 rows (row groups of 16,000 rows cut into three ranges of whole blocks)
        check_all(path, read, MORSELS_ON)


def test_errors_are_readable() raises:
    var mem = make_table()
    write_parquet("/tmp/f140_scan_err.parquet", mem, "", ZSTD, GROUP)
    var scan = scan_parquet("/tmp/f140_scan_err.parquet")
    var msg = String("")
    try:
        _ = Query(scan).group_by(col("nope"), groups=3).agg(count()).compute()
    except e:
        msg = String(e)
    expect(msg == "column `nope` not found", "unknown column: " + msg)
    msg = ""
    try:
        _ = scan_parquet("/tmp/f140_does_not_exist.parquet")
    except e:
        msg = String(e)
    expect("f140_does_not_exist" in msg, "a missing file names the file: " + msg)


def test_overflow_raises_like_the_in_memory_run() raises:
    var t = Table.empty()
    var v = t.add_column("v", INT)
    for _ in range(64):
        t.append(v, Int64.MAX // 8)
    var mem = t^.share()
    write_parquet("/tmp/f140_scan_ovf.parquet", mem, "", ZSTD, 8)       # 8 row groups: each fits, the total does not
    var want = String("")
    try:
        _ = Query(mem).agg(sum(col("v"))).compute()
    except e:
        want = String(e)
    var got = String("")
    try:
        _ = Query(scan_parquet("/tmp/f140_scan_ovf.parquet")).agg(sum(col("v"))).compute()
    except e:
        got = String(e)
    expect(want != "" and got == want, "overflow: " + got + " vs " + want)


def test_a_file_that_changes_after_it_was_opened_is_refused() raises:
    """A truncation or an in-place rewrite changes the size: refused before any row group is mapped (a mapping of
    pages that vanish would be a bus error). A file replaced by rename leaves the open file as it was: the answer is
    the one for the file that was opened."""
    var mem = make_table()
    var path = String("/tmp/f140_scan_chg.parquet")
    write_parquet(path, mem, "", ZSTD, GROUP)
    var want = Query(mem).agg(sum(col("cost")), count()).compute()
    var scan = scan_parquet(path)
    write_parquet("/tmp/f140_scan_chg_other.parquet", make_table(), "", ZSTD, GROUP // 2)
    _ = external_call["rename", Int32]("/tmp/f140_scan_chg_other.parquet".as_c_string_span().ptr(), path.as_c_string_span().ptr())
    var got = Query(scan).agg(sum(col("cost")), count()).compute()
    expect(got.sums[0] == want.sums[0] and got.counts[0] == want.counts[0], "a renamed-over file: the opened file is read")
    var scan2 = scan_parquet(path)
    with open(path, "w") as f:                                  # rewritten in place: truncated, then shorter
        f.write("PAR1 and not much more")
    var msg = String("")
    try:
        _ = Query(scan2).agg(sum(col("cost")), count()).compute()
    except e:
        msg = String(e)
    expect(msg == "the file changed since it was opened (`" + path + "`)", "truncated in place: `" + msg + "`")


def test_a_text_column_nobody_reads_is_never_decoded() raises:
    var mem = make_table()
    write_parquet("/tmp/f140_scan_lazy.parquet", mem, "", ZSTD, GROUP)
    var scan = scan_parquet("/tmp/f140_scan_lazy.parquet")
    var region = scan.index("region")
    var note = scan.index("note")
    expect(not scan.dictionary_loaded(region) and not scan.dictionary_loaded(note), "nothing is built when the file is opened")
    _ = Query(scan).agg(sum(col("cost"))).compute()
    expect(not scan.dictionary_loaded(region) and not scan.dictionary_loaded(note), "a sum over an int reads no dictionary")
    _ = Query(scan).group_by(col("region"), groups=8).agg(count()).compute()
    expect(scan.dictionary_loaded(region) and not scan.dictionary_loaded(note), "a query on region builds region's only")
    expect(scan.word_count(note) > 0 and scan.dictionary_loaded(note), "asking for a column's words builds them")


def main() raises:
    test_a_file_that_changes_after_it_was_opened_is_refused()
    test_a_text_column_nobody_reads_is_never_decoded()
    test_equals_read_parquet_for_every_codec()
    test_errors_are_readable()
    test_overflow_raises_like_the_in_memory_run()
    print("all scan tests passed")
