# F-137 (M6): the GPU path of Query. The oracle is the CPU path of the same query (itself checked against a plain
# loop in test_query.mojo): every number, count and presence flag must be identical, not close. Skips cleanly when
# there is no GPU. Allocations are modest: GPU 0 is shared with other processes.
from mxframe import (Table, SharedTable, Frame, INT, DECIMAL, TEXT, Query, QueryResult, col, lit, least, greatest, sum,
                     count, mean, min_of, max_of, CPU, GPU, AUTO, DeviceTable, gpu_available)
from max.gpu.host import DeviceContext


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


def same(a: QueryResult, b: QueryResult, label: String) raises:
    """Every slot, count, sum and presence flag equal (the GPU is not allowed to be merely close)."""
    expect(len(a.slots) == len(b.slots), label + ": same number of slots")
    for s in range(len(a.slots)):
        for g in range(len(a.counts)):
            expect(a.slots[s][g] == b.slots[s][g], label + ": slot " + String(s) + " group " + String(g) + ": gpu " +
                   String(a.slots[s][g]) + " vs cpu " + String(b.slots[s][g]))
    for g in range(len(a.counts)):
        expect(a.counts[g] == b.counts[g], label + ": count group " + String(g))
    expect(len(a.sums) == len(b.sums), label + ": sums")
    for i in range(len(a.kinds)):
        for g in range(len(a.counts)):
            expect(a.present(i, g) == b.present(i, g), label + ": present " + String(i) + "/" + String(g))


comptime N = 300_000


def make_table(seed: UInt64, null_every: Int, n: Int = N) raises -> SharedTable:
    """test_query's table: region TEXT, carrier INT -1..7 (some out of range), day INT, revenue DECIMAL(2) (every 50th
    value past 2^53, many past int32), cost INT; every `null_every`-th cell of carrier / day / revenue is null."""
    var t = Table.empty()
    var g = t.add_column("region", TEXT)
    var c = t.add_column("carrier", INT)
    var d = t.add_column("day", INT)
    var r = t.add_column("revenue", DECIMAL, scale=2)
    var k = t.add_column("cost", INT)
    var words: List[String] = ["West", "East", "North", "South", "Central"]
    for w in words:
        _ = t.add_word(g, w)
    var rng = Rng(seed)
    for i in range(n):
        t.append(g, Int64(rng.int(0, 4)))
        if null_every > 0 and i % null_every == 3:
            t.append_null(c)
        else:
            t.append(c, Int64(rng.int(-1, 7)))
        if null_every > 0 and i % null_every == 5:
            t.append_null(d)
        else:
            t.append(d, Int64(rng.int(0, 400)))
        if null_every > 0 and i % null_every == 1:
            t.append_null(r)
        else:
            if i % 50 == 0:
                var big = Int64(1 << 53) + Int64(rng.int(0, 1000))
                t.append(r, big if (i // 50) % 2 == 0 else -big + 7)
            else:
                t.append(r, Int64(rng.int(-100_000_00, 100_000_00)))
        t.append(k, Int64(rng.int(-5000, 5000)))
    return t^.share()


def shape(t: SharedTable, key_kind: Int, groups: Int, flt: Int) raises -> Query:
    var q = Query(t)
    if flt == 1:
        q = q^.where(col("region") == 1).where(col("day").between(10, 300))
    if key_kind == 0:
        q = q^.group_by(col("carrier"), groups=groups)
    elif key_kind == 1:
        q = q^.group_by(col("region") * 7 + col("carrier"), groups=groups)
    elif key_kind == 2:
        q = q^.group_by(least(col("day") // 7, lit(51)), groups=groups)
    return q^.agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("cost")), max_of(col("cost")),
                  min_of(col("revenue")), max_of(col("revenue")), mean(col("cost")))


def check_shape(t: SharedTable, dt: DeviceTable, key_kind: Int, groups: Int, flt: Int, label: String) raises:
    var cpu = shape(t, key_kind, groups, flt).compute(CPU)
    var gpu = shape(t, key_kind, groups, flt).on_device(dt).compute(GPU)
    expect(cpu.device == CPU and gpu.device == GPU, label + ": ran where asked")
    same(gpu, cpu, label)


def test_every_app_shape_equals_the_cpu() raises:
    for null_every in [0, 11]:
        var t = make_table(UInt64(7 + null_every), null_every)
        var dt = DeviceTable(t)
        var tag = " (nulls every " + String(null_every) + ")"
        check_shape(t, dt, 0, 7, 0, "chart" + tag)
        check_shape(t, dt, 1, 35, 0, "pivot" + tag)
        check_shape(t, dt, 2, 52, 0, "bin" + tag)
        check_shape(t, dt, 3, 1, 1, "filtered tile" + tag)
        check_shape(t, dt, 0, 7, 1, "filtered chart" + tag)
        check_shape(t, dt, 1, 35, 1, "filtered pivot" + tag)
        check_shape(t, dt, 2, 52, 1, "filtered bin" + tag)
        check_shape(t, dt, 3, 1, 0, "global" + tag)
        check_shape(t, dt, 2, 2000, 0, "many groups (global atomics)" + tag)
        dt.release()


def small_table() raises -> SharedTable:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    var b = t.add_column("b", INT)
    var g = t.add_column("g", INT)
    var av: List[Int] = [-7, 7, -7, 7, 0, 5, 3, -9, 9, 1]
    var bv: List[Int] = [2, -2, -2, 2, 3, 0, 0, 1, 2, 1]
    for i in range(10):
        t.append(a, Int64(av[i]))
        t.append(b, Int64(bv[i]))
        if i == 2 or i == 5:
            t.append_null(g)
        else:
            t.append(g, Int64(i % 3))
    return t^.share()


def test_negative_floor_division_and_modulo() raises:
    """// and % on negative and zero divisors, row by row (rows are distinct (a, b) pairs), against Mojo's own Int64."""
    var t = small_table()
    var dt = DeviceTable(t)
    for i in range(t.rows()):
        var a = Int(t.get(0, i))
        var b = Int(t.get(1, i))
        var one = Query(t).where(col("a") == a).where(col("b") == b).on_device(dt)
        var r = one^.agg(sum(col("a") // col("b")), sum(col("a") % col("b")), sum(-col("a") + least(col("a"), col("b")) * 2 - greatest(col("a"), col("b")))).compute(GPU)
        var want_fd = a // b if b != 0 else 0
        var want_md = a % b if b != 0 else 0
        expect(r.slots[0][0] == want_fd, "a // b row " + String(i) + ": " + String(r.slots[0][0]) + " vs " + String(want_fd))
        expect(r.slots[1][0] == want_md, "a % b row " + String(i) + ": " + String(r.slots[1][0]) + " vs " + String(want_md))
        expect(r.slots[2][0] == -a + min(a, b) * 2 - max(a, b), "neg / least / greatest row " + String(i))
    var r = Query(t).on_device(dt).where((col("a") > 0) | (col("g") == 1)).agg(count(), sum(col("a"))).compute(GPU)
    same(r, Query(t).where((col("a") > 0) | (col("g") == 1)).agg(count(), sum(col("a"))).compute(CPU), "| with the null rule")
    var n = Query(t).on_device(dt).where(~(col("a") != 7)).agg(count()).compute(GPU)
    expect(n.counts[0] == 2, "~(a != 7) keeps the two 7s")
    dt.release()


def test_keys_past_int32_and_negative_are_left_out() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var keys: List[Int] = [(1 << 32) + 1, -1, 1, Int(Int64.MIN), 1 << 31, 3]
    for key in keys:
        t.append(k, Int64(key))
    var s = t^.share()
    var dt = DeviceTable(s)
    var r = Query(s).on_device(dt).group_by(col("k"), groups=4).agg(count()).compute(GPU)
    expect(r.counts[0] == 0 and r.counts[1] == 1 and r.counts[2] == 0 and r.counts[3] == 1,
           "2^32+1 must not land in group 1; -1, INT64_MIN and 2^31 are left out")
    dt.release()


def test_values_past_int32_and_2_53() raises:
    """Cents over $21.4M and sums that a float64 would round: exact on the device."""
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", DECIMAL, scale=2)
    var want = List[Int](length=2, fill=0)
    for i in range(1000):
        t.append(k, Int64(i % 2))
        var x = Int64(1 << 53) + 1 if i % 4 < 2 else -(Int64(1 << 53))     # cancelling, past 2^53
        if i % 7 == 0:
            x = 3_000_000_000 + Int64(i)                                  # $30M in cents: past int32
        t.append(v, x)
        want[i % 2] += Int(x)
    var s = t^.share()
    var dt = DeviceTable(s)
    var r = Query(s).on_device(dt).group_by(col("k"), groups=2).agg(sum(col("v")), max_of(col("v")), min_of(col("v"))).compute(GPU)
    expect(r.sums[0] == want[0] and r.sums[1] == want[1], "exact sums: " + String(r.sums[0]) + " " + String(r.sums[1]))
    same(r, Query(s).group_by(col("k"), groups=2).agg(sum(col("v")), max_of(col("v")), min_of(col("v"))).compute(CPU), "past 2^53")
    expect(r.values(1)[0] == (1 << 53) + 1 and r.values(2)[0] == -(1 << 53), "min / max keep all 64 bits")
    dt.release()


def test_two_million_rows_use_the_multi_block_modes() raises:
    var t = Table.empty()
    var k = t.add_column("k", INT)
    var v = t.add_column("v", INT)
    var rng = Rng(11)
    var n = 2_500_000
    for i in range(n):
        t.append(k, Int64(rng.int(-1, 40)))
        t.append(v, Int64(rng.int(-9_000_000_00, 9_000_000_00)))
    var s = t^.share()
    var dt = DeviceTable(s)
    # global (mode 0), 41 groups (shared tables), 5000 groups (atomics straight into the output), a filtered one
    var g1 = Query(s).on_device(dt).agg(sum(col("v")), count(), min_of(col("v")), max_of(col("v"))).compute(GPU)
    same(g1, Query(s).agg(sum(col("v")), count(), min_of(col("v")), max_of(col("v"))).compute(CPU), "2.5M global")
    var g2 = Query(s).on_device(dt).group_by(col("k"), groups=41).agg(sum(col("v")), count(col("v")), max_of(col("v"))).compute(GPU)
    same(g2, Query(s).group_by(col("k"), groups=41).agg(sum(col("v")), count(col("v")), max_of(col("v"))).compute(CPU), "2.5M 41 groups")
    var g3 = Query(s).on_device(dt).group_by((col("v") // 1_000_000_00) + 100, groups=201).agg(sum(col("v")), count()).compute(GPU)
    same(g3, Query(s).group_by((col("v") // 1_000_000_00) + 100, groups=201).agg(sum(col("v")), count()).compute(CPU), "2.5M 201 groups")
    var g4 = Query(s).on_device(dt).where(col("v") > 0).group_by(col("k") * 120 + col("v") % 120, groups=5000).agg(sum(col("v")), count()).compute(GPU)
    same(g4, Query(s).where(col("v") > 0).group_by(col("k") * 120 + col("v") % 120, groups=5000).agg(sum(col("v")), count()).compute(CPU), "2.5M 5000 groups, filtered")
    dt.release()


def test_overflow_raises_the_same_error() raises:
    var t = Table.empty()
    var a = t.add_column("a", INT)
    for _ in range(10):
        t.append(a, Int64(1 << 62))
    var s = t^.share()
    var dt = DeviceTable(s)
    var cpu_msg = String("")
    var gpu_msg = String("")
    try:
        _ = Query(s).agg(sum(col("a"))).compute(CPU)
    except e:
        cpu_msg = String(e)
    try:
        _ = Query(s).on_device(dt).agg(sum(col("a"))).compute(GPU)
    except e:
        gpu_msg = String(e)
    expect(cpu_msg.byte_length() > 0 and gpu_msg == cpu_msg, "gpu: `" + gpu_msg + "` cpu: `" + cpu_msg + "`")
    var m = Table.empty()
    var b = m.add_column("b", INT)
    m.append(b, Int64.MIN)
    var ms = m^.share()
    var dm = DeviceTable(ms)
    var raised = False
    try:
        _ = Query(ms).on_device(dm).agg(sum(col("b") // -1)).compute(GPU)
    except e:
        raised = "int64" in String(e)
    expect(raised, "INT64_MIN // -1 raised on the GPU")
    expect(Query(ms).on_device(dm).agg(sum(col("b") % -1)).compute(GPU).sums[0] == 0, "INT64_MIN % -1 is 0")
    dt.release()
    dm.release()


def test_a_frames_edits_without_re_uploading() raises:
    var t = make_table(21, 13, 60_000)
    var dt = DeviceTable(t)
    var uploaded = dt.bytes()
    var f = Frame(t)
    var rng = Rng(99)
    for step in range(300):
        var row = rng.int(0, t.rows() - 1)
        var what = rng.int(0, 5)
        if not f.alive(row):
            continue
        if what <= 2:
            f.set(f.column("revenue"), row, Int64(rng.int(-1_000_00, 3_000_00)))
        elif what == 3:
            f.set_null(f.column("revenue"), row)
        elif what == 4:
            f.remove(row)
        else:
            _ = f.append(["East", "3", "11", "12.50", "4"])
    _ = f.merge(f.column("region"), [Int64(2)], Int64(0))
    var q_gpu = Query(f).on_device(dt).where(col("day").between(10, 300)).group_by(col("region") * 7 + col("carrier"), groups=35)
    var gpu = q_gpu^.agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("revenue")), max_of(col("cost"))).compute(GPU)
    var cpu = Query(f).where(col("day").between(10, 300)).group_by(col("region") * 7 + col("carrier"), groups=35).agg(sum(col("revenue")), count(), count(col("revenue")), min_of(col("revenue")), max_of(col("cost"))).compute(CPU)
    same(gpu, cpu, "frame with edits")
    var base = Query(t).group_by(col("region") * 7 + col("carrier"), groups=35).agg(count()).compute(CPU)
    var differs = False
    for g in range(35):
        differs = differs or base.counts[g] != Query(f).group_by(col("region") * 7 + col("carrier"), groups=35).agg(count()).compute(CPU).counts[g]
    expect(differs, "the edits change the answer (or this test proves nothing)")
    expect(dt.bytes() == uploaded, "the device copy was not re-uploaded")
    dt.release()


def test_release_frees_the_device_memory() raises:
    var t = make_table(5, 0, 50_000)
    var ctx = DeviceContext()
    var free0 = ctx.get_memory_info()[0]
    var dt = DeviceTable(t, ctx)
    var copy = dt                                            # a session's handle: shares the one device copy
    expect(dt.bytes() >= 5 * 8 * 50_000 and copy.bytes() == dt.bytes(), "uploaded: " + String(dt.bytes()))
    var free1 = ctx.get_memory_info()[0]
    expect(free0 - free1 >= 5 * 8 * 50_000 - (1 << 20), "the device lost the table's bytes")
    dt.release()
    expect(copy.bytes() == 0 and not copy.is_live(), "release frees for every handle")
    var free2 = ctx.get_memory_info()[0]
    expect(free2 - free1 >= 5 * 8 * 50_000 - (1 << 20), "the device got the memory back: " + String(free2 - free1))
    var msg = String("")
    try:
        _ = Query(t).on_device(copy).agg(count()).compute(GPU)
    except e:
        msg = String(e)
    expect("released" in msg, "GPU on a released table: " + msg)
    var auto = Query(t).on_device(copy).agg(count()).compute(AUTO)
    expect(auto.device == CPU and auto.counts[0] == 50_000, "AUTO falls back to the CPU")


def test_auto_picks_the_gpu_for_big_tables_only() raises:
    var small = make_table(1, 0, 20_000)
    var ds = DeviceTable(small)
    var r = Query(small).on_device(ds).agg(count()).compute(AUTO)
    expect(r.device == CPU and r.counts[0] == 20_000, "a small table stays on the CPU")
    var big = Table.empty()
    var v = big.add_column("v", INT)
    for i in range(1_100_000):
        big.append(v, Int64(i % 10))
    var bs = big^.share()
    var db = DeviceTable(bs)
    var rb = Query(bs).on_device(db).group_by(col("v"), groups=10).agg(count()).compute(AUTO)
    expect(rb.device == GPU and rb.counts[3] == 110_000, "1.1M rows with a device copy go to the GPU")
    expect(Query(bs).group_by(col("v"), groups=10).agg(count()).compute(AUTO).device == CPU, "no device copy: CPU")
    var wrong = False
    try:
        _ = Query(small).on_device(db)
    except e:
        wrong = "not made from" in String(e)
    expect(wrong, "a DeviceTable of another table is refused")
    ds.release()
    db.release()


def test_the_apps_device_context_is_shared() raises:
    """The app's context allocates its own buffer; MX frame's table and kernels run on the same context."""
    var ctx = DeviceContext()
    var mine = ctx.enqueue_create_buffer[DType.int64](1000)
    ctx.enqueue_memset(mine, 7)
    var t = make_table(2, 0, 40_000)
    var dt = DeviceTable(t, ctx)
    var r = Query(t).on_device(dt).group_by(col("carrier"), groups=7).agg(sum(col("cost"))).compute(GPU)
    same(r, Query(t).group_by(col("carrier"), groups=7).agg(sum(col("cost"))).compute(CPU), "on the app's context")
    var host = List[Int64](length=1, fill=0)
    ctx.enqueue_copy(dst_ptr=host.unsafe_ptr(), src_buf=mine)
    ctx.synchronize()
    expect(host[0] == 7, "the app's buffer is intact beside the table")
    dt.release()


def main() raises:
    if not gpu_available():
        print("skipped: no GPU")
        return
    test_every_app_shape_equals_the_cpu()
    test_negative_floor_division_and_modulo()
    test_keys_past_int32_and_negative_are_left_out()
    test_values_past_int32_and_2_53()
    test_two_million_rows_use_the_multi_block_modes()
    test_overflow_raises_the_same_error()
    test_a_frames_edits_without_re_uploading()
    test_release_frees_the_device_memory()
    test_auto_picks_the_gpu_for_big_tables_only()
    test_the_apps_device_context_is_shared()
