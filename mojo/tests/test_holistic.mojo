# F-148: the holistic states (count distinct, percentiles, top-k) on (group, value) pairs, against a plain oracle:
# per group, collect the values, sort them (a merge sort written here), and read the answers off the sorted list.
from mxframe.holistic import HolStates, HolSpec, HolResult, A_DISTINCT, A_PCT, A_TOPK


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


def msort(mut v: List[Int64]):
    var n = len(v)
    var tmp = List[Int64](length=n, fill=0)
    var width = 1
    while width < n:
        var lo = 0
        while lo < n:
            var mid = min(lo + width, n)
            var hi = min(lo + 2 * width, n)
            var i = lo
            var j = mid
            var k = lo
            while i < mid and j < hi:
                if v[j] < v[i]:
                    tmp[k] = v[j]
                    j += 1
                else:
                    tmp[k] = v[i]
                    i += 1
                k += 1
            while i < mid:
                tmp[k] = v[i]
                i += 1
                k += 1
            while j < hi:
                tmp[k] = v[j]
                j += 1
                k += 1
            lo += 2 * width
        swap(v, tmp)
        width *= 2


def ceil_index(p: Float64, n: Int) -> Int:
    # ceil(p * n) - 1 by hand (the oracle must not share the code under test)
    var x = p * Float64(n)
    var c = Int(x)
    if Float64(c) < x:
        c += 1
    return max(c - 1, 0)


struct Case(Movable):
    var gids: List[Int32]
    var vals: List[Int64]

    def __init__(out self):
        self.gids = List[Int32]()
        self.vals = List[Int64]()


def make(seed: UInt64, n: Int, groups: Int, lo: Int, hi: Int, wide: Bool) -> Case:
    var c = Case()
    var rng = Rng(seed)
    for _ in range(n):
        c.gids.append(Int32(rng.int(0, groups - 1)))
        var v = Int64(rng.int(lo, hi))
        if wide and rng.int(0, 99) == 0:
            v = Int64.MAX - Int64(rng.int(0, 3)) if rng.int(0, 1) == 0 else Int64.MIN + Int64(rng.int(0, 3))
        c.vals.append(v)
    return c^


def check(name: String, c: Case, groups: Int, windows: Int, k: Int, reverse_rank: Int) raises:
    """Run the states over `windows` windows of the case and compare with the oracle. reverse_rank > 0: values are
    codes 0 .. reverse_rank-1 whose order is reversed (rank = reverse_rank - 1 - code)."""
    var specs = List[HolSpec]()
    var ps: List[Float64] = [0.0, 0.1, 0.5, 0.9, 0.99, 1.0, 1.0 / 3.0, 0.7]
    specs.append(HolSpec(0, A_DISTINCT, 0.0, 0, 0))
    for i in range(len(ps)):
        specs.append(HolSpec(1 + i, A_PCT, ps[i], 0, 0))
    specs.append(HolSpec(1 + len(ps), A_TOPK, 0.0, k, 0))
    var ranks = List[List[Int64]]()
    var rk = List[Int64]()
    for code in range(reverse_rank):
        rk.append(Int64(reverse_rank - 1 - code))
    ranks.append(rk^)
    var st = HolStates(groups, specs^, ranks^)
    var n = len(c.vals)
    for w in range(windows):
        var lo = n * w // windows
        var hi = n * (w + 1) // windows
        st.absorb_rows(0, List[Int32](c.gids[lo:hi]), List[Int64](c.vals[lo:hi]))
    var res = st.finish()
    # the oracle
    var per = List[List[Int64]]()
    for _ in range(groups):
        per.append(List[Int64]())
    for i in range(n):
        per[Int(c.gids[i])].append(c.vals[i])
    for g in range(groups):
        var key = List[Int64]()                  # the values in the order being tested (ranks for text)
        for v in per[g]:
            key.append(Int64(reverse_rank - 1) - v if reverse_rank > 0 else v)
        msort(key)
        var m = len(key)
        # distinct
        var d = 0
        for i in range(m):
            if i == 0 or key[i] != key[i - 1]:
                d += 1
        expect(res.vals[0][g] == d, name + ": distinct of group " + String(g) + " is " + String(d) + " got " + String(res.vals[0][g]))
        for q in range(len(ps)):
            var want = 0
            if m > 0:
                var v = key[ceil_index(ps[q], m)]
                want = Int(Int64(reverse_rank - 1) - v if reverse_rank > 0 else v)
            expect(res.vals[1 + q][g] == want, name + ": p" + String(ps[q]) + " of group " + String(g) + " is " + String(want) + " got " + String(res.vals[1 + q][g]))
            expect(res.counts[1 + q][g] == m, name + ": count of group " + String(g))
        var t = 1 + len(ps)
        var off = res.top_off[t][g]
        var cnt = res.top_off[t][g + 1] - off
        expect(cnt == min(k, m), name + ": top-k size of group " + String(g) + " is " + String(min(k, m)) + " got " + String(cnt))
        for i in range(cnt):
            var v = key[m - 1 - i]
            var want = Int(Int64(reverse_rank - 1) - v if reverse_rank > 0 else v)
            expect(res.top[t][off + i] == want, name + ": top " + String(i) + " of group " + String(g))


def main() raises:
    check("one group", make(1, 200_000, 1, -50, 50, False), 1, 3, 5, 0)
    check("seven groups, narrow", make(2, 300_000, 7, 0, 20, False), 7, 4, 3, 0)
    check("seven groups, wide", make(3, 300_000, 7, -1_000_000_000, 1_000_000_000, True), 7, 2, 10, 0)
    check("1000 groups", make(4, 400_000, 1000, -100_000, 100_000, False), 1000, 5, 4, 0)
    check("300k groups (many)", make(5, 600_000, 300_000, 0, 5, False), 300_000, 2, 2, 0)
    check("sparse groups", make(6, 1_000, 5_000, 0, 1_000_000, False), 5_000, 1, 3, 0)
    check("two big groups (radix select)", make(7, 1_500_000, 2, -9_000_000_000_000, 9_000_000_000_000, True), 2, 2, 7, 0)
    check("two big groups, few values", make(8, 1_500_000, 2, 0, 3, False), 2, 1, 2, 0)
    check("text order", make(9, 100_000, 6, 0, 49, False), 6, 3, 4, 50)
    check("empty", make(10, 0, 4, 0, 1, False), 4, 1, 3, 0)
    check("k larger than a group", make(11, 50, 10, 0, 100, False), 10, 1, 20, 0)
    print("ok")
