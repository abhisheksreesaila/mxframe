# F-176: how long a 50-row x 40-column block read takes from a Frame with many edited cells (the grid's per-frame cost).
from std.time import perf_counter_ns
from omxcore import Table, Frame, INT, DECIMAL, read_block, range_aggregate


def main() raises:
    var wide = Table.empty()
    for c in range(40):
        _ = wide.add_column("c" + String(c), INT if c % 3 == 0 else DECIMAL, scale=0 if c % 3 == 0 else 2)
    for r in range(200_000):
        for c in range(40):
            wide.append(c, Int64(r * 7 + c))
    var ws = wide^.share()
    var order = List[Int]()
    for i in range(200_000):
        order.append((i * 7919) % 200_000)
    var cols = List[Int]()
    for c in range(40):
        cols.append(c)
    var values = List[Int64](length=40 * 50, fill=0)
    var nulls = List[UInt8](length=40 * 50, fill=0)
    for edits in [0, 300, 3_000, 30_000, 300_000]:
        var f = Frame(ws)
        for j in range(edits):
            f.set((j * 3) % 40, (j * 7) % 200_000, Int64(j))
        var best = 1e9
        for i in range(30):
            var t0 = perf_counter_ns()
            read_block(f, order, cols, 1000 + i, 1050 + i, values, nulls)
            best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        print(String(edits) + " edited cells: block read 50 x 40 best " + String(best) + " ms")
    # a long range over a frame: what the edits cost on top of the table's 0.6 ms a column
    var big = Table.empty()
    for c in range(3):
        _ = big.add_column("c" + String(c), INT if c % 3 == 0 else DECIMAL, scale=0 if c % 3 == 0 else 2)
    for r in range(1_000_000):
        for c in range(3):
            big.append(c, Int64(r * 7 + c))
    var bs = big^.share()
    var shuffled = List[Int]()
    for i in range(1_000_000):
        shuffled.append((i * 7919) % 1_000_000)
    var three: List[Int] = [0, 1, 2]
    for edits in [0, 300, 30_000, 300_000]:
        var f = Frame(bs)
        for j in range(edits):
            f.set((j * 3) % 3, (j * 7) % 1_000_000, Int64(j))
        var best = 1e9
        for _ in range(5):
            var t0 = perf_counter_ns()
            _ = range_aggregate(f, shuffled, three, 0, 1_000_000)
            best = min(best, Float64(perf_counter_ns() - t0) / 1e6)
        print(String(edits) + " edited cells: range aggregate of 3 columns over 1M rows (random order) best " + String(best) + " ms")
