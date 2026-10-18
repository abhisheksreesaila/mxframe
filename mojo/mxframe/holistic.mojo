"""Holistic aggregates (F-148): count distinct, percentiles and top-k per group, exact, on all cores.

A sum, min or max folds one row at a time into a number per group; these three need more than that, so the query
kernel does not hold them. Instead its per-block hook (query.mojo `_mat_hook`) writes, for each distinct value
expression ("source"), the (group, value) pair of every row that is selected, whose group is in range and whose value
is not null, packed block by block (a `Window`). This file consumes those pairs:

- count distinct: a hash set of (group, value) pairs, split into P partitions by the pair's hash. A window's pairs are
  first scattered to their partitions (parallel, two passes), then each partition inserts its own pairs (parallel, no
  locks). The set persists between windows, so a streamed scan merges row group after row group exactly. Memory:
  12 bytes a slot, 2 to 4 slots per distinct (group, value) pair.
- top-k: the k largest values of a group, duplicates kept (SQL `ORDER BY e DESC LIMIT k`): a bounded min-heap of k
  values per group (O(rows log k), almost all rows rejected by one compare with the heap's smallest). Few groups: every
  task keeps private heaps over its blocks and they are merged; many groups: the pairs are scattered by group so that
  one task owns each group's heap. The state is k values a group, and merging windows is just offering their values.
- percentile: the value at index max(ceil(p * n) - 1, 0) of the group's sorted non-null values, which is DuckDB's
  `quantile_disc` (an actual value, exact in the stored integers; `median` is p = 0.5, the lower middle value for an
  even n, so it differs from DuckDB's `median`, which is `quantile_cont`). Exact percentiles need every value: the
  pairs are kept (12 bytes a non-null row, plus 8 bytes a row while the answer is computed), grouped with a parallel
  counting sort, then each group is selected, not sorted: quickselect (three-way, O(n)) for small groups, and for a
  group of 262,144 values or more a parallel most-significant-byte radix select, which reads the values
  without moving them (about two passes for values in a small range, at most eight).
- text: percentile and top-k order words alphabetically (the column's sort order, `Frame.permutation`'s) not by code:
  the values are mapped to their rank before ordering and back to the code after (`ranks`, `unranks`).
"""
from std.memory import Pointer
from std.atomic import Atomic
from std.bit import count_leading_zeros
from std.math import ceil
from .kernels._fpenv import core_count, par_n

comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime NP = Pointer[Int, MutUntrackedOrigin]

# aggregate kinds (the plain ones are in query.mojo)
comptime A_DISTINCT = 6
comptime A_PCT = 7
comptime A_TOPK = 8

comptime BLOCK = 65536               # entries in a block made from a flat array
comptime RADIX_MIN = 262_144         # a group at least this big is selected by the parallel radix select
comptime HIST_BUDGET = 4_000_000     # counters in the per-task histograms of a counting sort
comptime PRIVATE_HEAPS = 1 << 22     # heap words all tasks may hold privately (top-k with few groups)
comptime MAX_TOPK_WORDS = 1 << 28    # groups x k a top-k state may hold (2 GiB)
comptime SIGN = UInt64(1) << 63


def _lp(a: Int) -> LP:
    return LP(unsafe_from_address=a)


def _ip(a: Int) -> IP:
    return IP(unsafe_from_address=a)


def _np(a: Int) -> NP:
    return NP(unsafe_from_address=a)


@always_inline
def _cut(n: Int, parts: Int, i: Int) -> Int:
    return n * i // parts


@always_inline
def _mix(v: Int64, g: Int32) -> UInt64:
    var x = UInt64(v) ^ (UInt64(Int(g)) * 0x9E3779B97F4A7C15)
    x = (x ^ (x >> 32)) * 0xD6E8FEB86659FD93
    x = (x ^ (x >> 32)) * 0xD6E8FEB86659FD93
    return x ^ (x >> 32)


def _tasks(total: Int, blocks: Int, threads: Int) -> Int:
    """How many tasks to split `blocks` blocks over: one for a small input or a serial build."""
    if total < 32768 or threads <= 1:
        return 1
    return max(1, min(threads, blocks))


# ── the pairs ────────────────────────────────────────────────────────────────────────────────────────

struct Blocks(Movable):
    """A list of runs of (group, value) pairs: block i has `n[i]` Int32 groups at address `g[i]` and as many Int64
    values at `v[i]`. Every group is in range and every value is non-null (the kernel hook and `HolStates.absorb_rows`
    leave the others out)."""
    var g: List[Int]
    var v: List[Int]
    var n: List[Int]
    var total: Int

    def __init__(out self):
        self.g = List[Int]()
        self.v = List[Int]()
        self.n = List[Int]()
        self.total = 0

    def add(mut self, g: Int, v: Int, n: Int):
        if n > 0:
            self.g.append(g)
            self.v.append(v)
            self.n.append(n)
            self.total += n

    def add_flat(mut self, g: Int, v: Int, n: Int):
        """`n` consecutive pairs, cut into blocks that tasks can share out."""
        var at = 0
        while at < n:
            var m = min(BLOCK, n - at)
            self.add(g + at * 4, v + at * 8, m)
            at += m


def window_blocks(gids: Int, vals: Int, blk_n: Int, nblocks: Int, blk: Int) -> Blocks:
    """The blocks of a kernel window: block b holds `blk_n[b]` pairs at entry b * blk."""
    var out = Blocks()
    var bn = _ip(blk_n)
    for b in range(nblocks):
        out.add(gids + b * blk * 4, vals + b * blk * 8, Int(bn[b]))
    return out^


struct Scattered(Movable):
    """The pairs of some blocks regrouped by part: part p is entries seg[p] .. seg[p+1] of `vals` and `gids`."""
    var seg: List[Int]
    var vals: List[Int64]
    var gids: List[Int32]

    def __init__(out self):
        self.seg = List[Int]()
        self.vals = List[Int64]()
        self.gids = List[Int32]()


@always_inline
def _owner(v: Int64, g: Int32, nparts: Int, by_hash: Bool) -> Int:
    if by_hash:
        return Int((_mix(v, g) >> 40) & UInt64(nparts - 1))
    return Int(g) % nparts


def _scatter(b: Blocks, nparts: Int, by_hash: Bool, rank: Int, threads: Int) -> Scattered:
    """Regroup the pairs into `nparts` parts: by the hash of (group, value) (`nparts` a power of two) or by group
    modulo `nparts`; values are mapped through `rank` (an Int64 table, 0: none). Two parallel passes over the blocks
    (count, then place); the order inside a part follows the blocks."""
    var nb = len(b.n)
    var tp = _tasks(b.total, nb, threads)
    var cnt = List[Int](length=tp * nparts, fill=0)
    var cnt_at = Int(cnt.unsafe_ptr())
    var bg = Int(b.g.unsafe_ptr())
    var bv = Int(b.v.unsafe_ptr())
    var bn = Int(b.n.unsafe_ptr())

    def count(t: Int) {imm}:
        var c = _np(cnt_at) + t * nparts
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            for j in range(_np(bn)[k]):
                c[_owner(vp[j], gp[j], nparts, by_hash)] += 1

    par_n(count, tp, tp)
    var out = Scattered()
    out.seg = List[Int](length=nparts + 1, fill=0)
    var run = 0
    for p in range(nparts):
        out.seg[p] = run
        for t in range(tp):
            var c = cnt[t * nparts + p]
            cnt[t * nparts + p] = run
            run += c
    out.seg[nparts] = run
    out.vals = List[Int64](unsafe_uninit_length=max(run, 1))
    out.gids = List[Int32](unsafe_uninit_length=max(run, 1))
    var ov = Int(out.vals.unsafe_ptr())
    var og = Int(out.gids.unsafe_ptr())

    def place(t: Int) {imm}:
        var pos = _np(cnt_at) + t * nparts
        var dv = _lp(ov)
        var dg = _ip(og)
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            for j in range(_np(bn)[k]):
                var p = _owner(vp[j], gp[j], nparts, by_hash)
                var v = vp[j]
                if rank != 0:
                    v = _lp(rank)[Int(v)]
                dv[pos[p]] = v
                dg[pos[p]] = gp[j]
                pos[p] += 1

    par_n(place, tp, tp)
    _ = cnt^
    return out^


struct Grouped(Movable):
    """The values of the blocks sorted into groups by a counting sort: group g's values are
    `vals[off[g] : off[g + 1]]`, in no particular order."""
    var off: List[Int]
    var vals: List[Int64]

    def __init__(out self):
        self.off = List[Int]()
        self.vals = List[Int64]()


def _group(b: Blocks, groups: Int, rank: Int, threads: Int) -> Grouped:
    """Counting sort of the pairs by group, in parallel: each task counts the groups of its blocks, the counters give
    every (group, task) its place, and each task writes its values there. The counters are `tasks x groups`, so the
    tasks are fewer when the groups are many (down to one for 4 million groups and more)."""
    var nb = len(b.n)
    var tp = min(_tasks(b.total, nb, threads), max(1, HIST_BUDGET // max(groups, 1)))
    var hist = List[Int](length=tp * groups, fill=0)
    var hist_at = Int(hist.unsafe_ptr())
    var bg = Int(b.g.unsafe_ptr())
    var bv = Int(b.v.unsafe_ptr())
    var bn = Int(b.n.unsafe_ptr())

    def count(t: Int) {imm}:
        var h = _np(hist_at) + t * groups
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            for j in range(_np(bn)[k]):
                h[Int(gp[j])] += 1

    par_n(count, tp, tp)
    var out = Grouped()
    out.off = List[Int](length=groups + 1, fill=0)
    var run = 0
    for g in range(groups):
        out.off[g] = run
        for t in range(tp):
            var c = hist[t * groups + g]
            hist[t * groups + g] = run
            run += c
    out.off[groups] = run
    out.vals = List[Int64](unsafe_uninit_length=max(run, 1))
    var ov = Int(out.vals.unsafe_ptr())

    def place(t: Int) {imm}:
        var pos = _np(hist_at) + t * groups
        var dv = _lp(ov)
        for k in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
            var gp = _ip(_np(bg)[k])
            var vp = _lp(_np(bv)[k])
            for j in range(_np(bn)[k]):
                var g = Int(gp[j])
                var v = vp[j]
                if rank != 0:
                    v = _lp(rank)[Int(v)]
                dv[pos[g]] = v
                pos[g] += 1

    par_n(place, tp, tp)
    _ = hist^
    return out^


# ── selection ────────────────────────────────────────────────────────────────────────────────────────

def _select_nth(p: LP, n: Int, k: Int) -> Int64:
    """The value of rank k (0-based) among the n values at p, by quickselect with a three-way partition (equal values
    cost nothing); the values are left permuted. Pivots are the median of three pseudo-random elements, so no
    input order is a worst case."""
    var lo = 0
    var hi = n - 1
    var rng = UInt64(0x2545F4914F6CDD1D) ^ UInt64(n)
    while True:
        if hi - lo < 24:                              # insertion sort the small remainder
            for i in range(lo + 1, hi + 1):
                var x = p[i]
                var j = i - 1
                while j >= lo and p[j] > x:
                    p[j + 1] = p[j]
                    j -= 1
                p[j + 1] = x
            return p[k]
        var span = UInt64(hi - lo + 1)
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var a = p[lo + Int(rng % span)]
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var b = p[lo + Int(rng % span)]
        rng ^= rng << 13
        rng ^= rng >> 7
        rng ^= rng << 17
        var c = p[lo + Int(rng % span)]
        var pivot = max(min(a, b), min(max(a, b), c))
        var lt = lo
        var i = lo
        var gt = hi
        while i <= gt:
            var x = p[i]
            if x < pivot:
                p[i] = p[lt]
                p[lt] = x
                lt += 1
                i += 1
            elif x > pivot:
                p[i] = p[gt]
                p[gt] = x
                gt -= 1
            else:
                i += 1
        if k < lt:
            hi = lt - 1
        elif k > gt:
            lo = gt + 1
        else:
            return pivot


def _radix_select(addr: Int, n: Int, k: Int, threads: Int) -> Int64:
    """The value of rank k (0-based) among the n values at `addr`, read-only and in parallel: one pass finds the
    smallest and largest key (the value with its sign bit flipped, so keys order as unsigned), the bytes they share
    are skipped, then one pass per remaining byte counts how many values of the prefix fall in each of its 256
    digits and narrows the prefix to the digit holding rank k. At most eight passes; values in a small range need
    two."""
    var tp = max(1, min(threads, n // 65536))
    var lo_k = List[UInt64](length=tp, fill=UInt64.MAX)
    var hi_k = List[UInt64](length=tp, fill=0)
    var lo_at = Int(lo_k.unsafe_ptr())
    var hi_at = Int(hi_k.unsafe_ptr())

    def extent(t: Int) {imm}:
        var p = _lp(addr)
        var lo = UInt64.MAX
        var hi = UInt64(0)
        for i in range(_cut(n, tp, t), _cut(n, tp, t + 1)):
            var key = UInt64(p[i]) ^ SIGN
            lo = min(lo, key)
            hi = max(hi, key)
        Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=lo_at)[t] = lo
        Pointer[UInt64, MutUntrackedOrigin](unsafe_from_address=hi_at)[t] = hi

    par_n(extent, tp, tp)
    var lo = UInt64.MAX
    var hi = UInt64(0)
    for t in range(tp):
        lo = min(lo, lo_k[t])
        hi = max(hi, hi_k[t])
    if lo == hi:
        return Int64(lo ^ SIGN)
    var lead = Int(count_leading_zeros(lo ^ hi)) // 8          # bytes both ends share
    var prefix = (lo >> UInt64(64 - 8 * lead)) if lead > 0 else UInt64(0)
    var rem = k
    var hist = List[Int](length=tp * 256, fill=0)
    var hist_at = Int(hist.unsafe_ptr())
    for pass_ in range(lead, 8):
        var shift = 56 - 8 * pass_
        var want = prefix
        for i in range(tp * 256):
            hist[i] = 0

        def digits(t: Int) {imm}:
            var p = _lp(addr)
            var h = _np(hist_at) + t * 256
            for i in range(_cut(n, tp, t), _cut(n, tp, t + 1)):
                var key = UInt64(p[i]) ^ SIGN
                if shift == 56 or (key >> UInt64(shift + 8)) == want:
                    h[Int((key >> UInt64(shift)) & 255)] += 1

        par_n(digits, tp, tp)
        var d = 0
        while True:
            var c = 0
            for t in range(tp):
                c += hist[t * 256 + d]
            if rem < c or d == 255:
                break
            rem -= c
            d += 1
        prefix = (prefix << 8) | UInt64(d)
    _ = lo_k^
    _ = hi_k^
    _ = hist^
    return Int64(prefix ^ SIGN)


@always_inline
def _percentile_index(p: Float64, n: Int) -> Int:
    """DuckDB's `quantile_disc` index in n sorted values: ceil(p * n) - 1, at least 0."""
    var i = Int(ceil(p * Float64(n))) - 1
    return min(max(i, 0), n - 1)


struct Percentiles(Movable):
    var vals: List[List[Int]]       # per requested p: per group, the value (0 for a group with no values)
    var counts: List[Int]           # per group: its non-null values

    def __init__(out self):
        self.vals = List[List[Int]]()
        self.counts = List[Int]()


def _percentiles(b: Blocks, groups: Int, ps: List[Float64], rank: Int, unrank: Int, threads: Int) -> Percentiles:
    """Every p in `ps` for every group over the pairs: group them, then select (see the file comment)."""
    var gr = _group(b, groups, rank, threads)
    var np = len(ps)
    var out_at = List[Int](length=max(np * groups, 1), fill=0)
    var out_p = Int(out_at.unsafe_ptr())
    var ps_p = Int(ps.unsafe_ptr())
    var off_at = Int(gr.off.unsafe_ptr())
    var vals_at = Int(gr.vals.unsafe_ptr())
    # small groups: balanced chunks of groups, one task each; big groups are left for the parallel select below
    var tp = _tasks(b.total, groups, threads)
    var chunks = max(1, tp * 4) if tp > 1 else 1
    var cuts = List[Int](length=chunks + 1, fill=groups)
    cuts[0] = 0
    if chunks > 1:
        var c = 1
        var g = 0
        while g < groups and c < chunks:
            if gr.off[g] >= b.total * c // chunks:
                cuts[c] = g
                c += 1
            else:
                g += 1
        for k in range(c, chunks):
            cuts[k] = groups
    var cuts_at = Int(cuts.unsafe_ptr())
    var big_ok = threads > 1

    def small(ci: Int) {imm}:
        var off = _np(off_at)
        var vals = _lp(vals_at)
        var out = _np(out_p)
        var pp = Pointer[Float64, MutUntrackedOrigin](unsafe_from_address=ps_p)
        for g in range(_np(cuts_at)[ci], _np(cuts_at)[ci + 1]):
            var m = off[g + 1] - off[g]
            if m == 0 or (big_ok and m >= RADIX_MIN):
                continue
            for q in range(np):
                var v = _select_nth(vals + off[g], m, _percentile_index(pp[q], m))
                if unrank != 0:
                    v = _lp(unrank)[Int(v)]
                out[q * groups + g] = Int(v)

    par_n(small, chunks, threads)
    if big_ok:
        for g in range(groups):
            var m = gr.off[g + 1] - gr.off[g]
            if m >= RADIX_MIN:
                for q in range(np):
                    var v = _radix_select(vals_at + gr.off[g] * 8, m, _percentile_index(ps[q], m), threads)
                    if unrank != 0:
                        v = _lp(unrank)[Int(v)]
                    out_at[q * groups + g] = Int(v)
    var res = Percentiles()
    for q in range(np):
        var row = List[Int](length=groups, fill=0)
        for g in range(groups):
            row[g] = out_at[q * groups + g]
        res.vals.append(row^)
    for g in range(groups):
        res.counts.append(gr.off[g + 1] - gr.off[g])
    _ = gr^
    _ = cuts^
    return res^


# ── count distinct ───────────────────────────────────────────────────────────────────────────────────

def _dist_grow(mut keys: List[Int64], mut g1: List[Int32], cap: Int):
    """Rehash one partition's table into `cap` slots (a power of two)."""
    var old_keys = keys^
    var old_g1 = g1^
    keys = List[Int64](length=cap, fill=0)
    g1 = List[Int32](length=cap, fill=0)
    var mask = cap - 1
    var okp = _lp(Int(old_keys.unsafe_ptr()))
    var ogp = _ip(Int(old_g1.unsafe_ptr()))
    var nkp = _lp(Int(keys.unsafe_ptr()))
    var ngp = _ip(Int(g1.unsafe_ptr()))
    for i in range(len(old_keys)):
        var e = ogp[i]
        if e != 0:
            var s = Int(_mix(okp[i], e - 1) & UInt64(mask))
            while ngp[s] != 0:
                s = (s + 1) & mask
            nkp[s] = okp[i]
            ngp[s] = e
    _ = old_keys^
    _ = old_g1^


def _dist_add(mut keys: List[Int64], mut g1: List[Int32], used: Int, sv: LP, sg: IP, n: Int) -> Int:
    """Insert n (value, group) pairs into one partition's open-addressing table (linear probing, load at most 1/2);
    returns the number of distinct pairs it holds."""
    var u = used
    var cap = len(keys)
    var kp = _lp(Int(keys.unsafe_ptr()))
    var gp = _ip(Int(g1.unsafe_ptr()))
    for i in range(n):
        if (u + 1) * 2 > cap:
            _dist_grow(keys, g1, max(cap * 2, 1024))
            cap = len(keys)
            kp = _lp(Int(keys.unsafe_ptr()))
            gp = _ip(Int(g1.unsafe_ptr()))
        var v = sv[i]
        var g = sg[i]
        var mask = cap - 1
        var s = Int(_mix(v, g) & UInt64(mask))
        var tag = g + 1
        while True:
            var e = gp[s]
            if e == 0:
                kp[s] = v
                gp[s] = tag
                u += 1
                break
            if e == tag and kp[s] == v:
                break
            s = (s + 1) & mask
    return u


struct DistinctState(Movable):
    """Every distinct (group, value) pair seen so far, in `parts` hash partitions (a power of two)."""
    var parts: Int
    var keys: List[List[Int64]]
    var g1: List[List[Int32]]          # group + 1 of each slot, 0 for an empty slot
    var used: List[Int]

    def __init__(out self, parts: Int):
        self.parts = parts
        self.keys = List[List[Int64]]()
        self.g1 = List[List[Int32]]()
        self.used = List[Int]()
        for _ in range(parts):
            self.keys.append(List[Int64]())
            self.g1.append(List[Int32]())
            self.used.append(0)

    def absorb(mut self, b: Blocks, threads: Int):
        var sc = _scatter(b, self.parts, True, 0, threads)
        var keys_at = Int(self.keys.unsafe_ptr())
        var g1_at = Int(self.g1.unsafe_ptr())
        var used_at = Int(self.used.unsafe_ptr())
        var seg_at = Int(sc.seg.unsafe_ptr())
        var sv = Int(sc.vals.unsafe_ptr())
        var sg = Int(sc.gids.unsafe_ptr())

        def insert(p: Int) {imm}:
            var lo = _np(seg_at)[p]
            var n = _np(seg_at)[p + 1] - lo
            if n == 0:
                return
            var kp = Pointer[List[Int64], MutUntrackedOrigin](unsafe_from_address=keys_at) + p
            var gp = Pointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=g1_at) + p
            var up = _np(used_at) + p
            up[0] = _dist_add(kp[], gp[], up[0], _lp(sv) + lo, _ip(sg) + lo, n)

        par_n(insert, self.parts, threads)
        _ = sc^

    def counts(self, groups: Int, threads: Int) -> List[Int]:
        """The number of distinct values of each group."""
        var P = self.parts
        var out = List[Int](length=groups, fill=0)
        var keys_at = Int(self.keys.unsafe_ptr())
        var g1_at = Int(self.g1.unsafe_ptr())
        if P * groups <= (1 << 23):
            # a counter per (partition, group), summed after: no shared writes
            var loc = List[Int32](length=P * groups, fill=0)
            var loc_at = Int(loc.unsafe_ptr())

            def tally(p: Int) {imm}:
                ref gl = (Pointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=g1_at) + p)[]
                var gp = _ip(Int(gl.unsafe_ptr()))
                var c = _ip(loc_at) + p * groups
                for i in range(len(gl)):
                    if gp[i] != 0:
                        c[Int(gp[i]) - 1] += 1

            par_n(tally, P, threads)
            var out_at = Int(out.unsafe_ptr())
            var tp = _tasks(groups, groups, threads)

            def total(t: Int) {imm}:
                var o = _np(out_at)
                var c = _ip(loc_at)
                for g in range(_cut(groups, tp, t), _cut(groups, tp, t + 1)):
                    var s = 0
                    for p in range(P):
                        s += Int(c[p * groups + g])
                    o[g] = s

            par_n(total, tp, tp)
            _ = loc^
        else:
            # many groups: atomic adds into one counter per group (contention is low when groups are many)
            var acc = List[Int64](length=groups, fill=0)
            var acc_at = Int(acc.unsafe_ptr())

            def tally_atomic(p: Int) {imm}:
                ref gl = (Pointer[List[Int32], MutUntrackedOrigin](unsafe_from_address=g1_at) + p)[]
                var gp = _ip(Int(gl.unsafe_ptr()))
                var c = _lp(acc_at)
                for i in range(len(gl)):
                    if gp[i] != 0:
                        _ = Atomic[Scalar[DType.int64]].fetch_add(c + (Int(gp[i]) - 1), 1)

            par_n(tally_atomic, P, threads)
            for g in range(groups):
                out[g] = Int(acc[g])
            _ = acc^
        return out^


# ── top-k ────────────────────────────────────────────────────────────────────────────────────────────

@always_inline
def _offer(h: LP, n: Int, k: Int, v: Int64) -> Int:
    """Offer v to the min-heap of the n (at most k) largest values seen at h; returns the heap's new size."""
    if n < k:
        var i = n
        while i > 0:
            var par = (i - 1) // 2
            if h[par] <= v:
                break
            h[i] = h[par]
            i = par
        h[i] = v
        return n + 1
    if v <= h[0]:
        return n
    var i = 0
    while True:
        var c = 2 * i + 1
        if c >= k:
            break
        if c + 1 < k and h[c + 1] < h[c]:
            c += 1
        if h[c] >= v:
            break
        h[i] = h[c]
        i = c
    h[i] = v
    return n


def _offer_blocks(b: Blocks, k0: Int, k1: Int, heap: LP, size: NP, k: Int, rank: Int):
    """Offer the values of blocks k0 .. k1-1 to their groups' heaps (group g's heap is k words at g * k)."""
    for q in range(k0, k1):
        var gp = _ip(b.g[q])
        var vp = _lp(b.v[q])
        for j in range(b.n[q]):
            var g = Int(gp[j])
            var v = vp[j]
            if rank != 0:
                v = _lp(rank)[Int(v)]
            size[g] = _offer(heap + g * k, size[g], k, v)


def _heap_sort_desc(h: LP, n: Int):
    """Sort the min-heap at h of n values into descending order in place (heapsort: pop the minimum to the end)."""
    var m = n
    while m > 1:
        var top = h[0]
        m -= 1
        var x = h[m]
        h[m] = top
        var i = 0
        while True:
            var c = 2 * i + 1
            if c >= m:
                break
            if c + 1 < m and h[c + 1] < h[c]:
                c += 1
            if h[c] >= x:
                break
            h[i] = h[c]
            i = c
        h[i] = x


struct TopKState(Movable):
    """The k largest values of each group so far, a min-heap of up to k values per group."""
    var k: Int
    var groups: Int
    var heap: List[Int64]
    var size: List[Int]

    def __init__(out self, k: Int, groups: Int):
        self.k = k
        self.groups = groups
        self.heap = List[Int64](length=max(k * groups, 1), fill=0)
        self.size = List[Int](length=max(groups, 1), fill=0)

    def absorb(mut self, b: Blocks, rank: Int, threads: Int):
        var k = self.k
        var G = self.groups
        var nb = len(b.n)
        var tp = _tasks(b.total, nb, threads)
        var heap = _lp(Int(self.heap.unsafe_ptr()))
        var size = _np(Int(self.size.unsafe_ptr()))
        if tp == 1:
            _offer_blocks(b, 0, nb, heap, size, k, rank)
        elif tp * G * k <= PRIVATE_HEAPS:
            # few groups: every task keeps its own heaps over its blocks; they are merged into the state after
            var ph = List[Int64](length=tp * G * k, fill=0)
            var ps = List[Int](length=tp * G, fill=0)
            var ph_at = Int(ph.unsafe_ptr())
            var ps_at = Int(ps.unsafe_ptr())

            def mine(t: Int) {imm}:
                _offer_blocks(b, _cut(nb, tp, t), _cut(nb, tp, t + 1), _lp(ph_at) + t * G * k, _np(ps_at) + t * G, k, rank)

            par_n(mine, tp, tp)
            for t in range(tp):
                for g in range(G):
                    for i in range(ps[t * G + g]):
                        size[g] = _offer(heap + g * k, size[g], k, ph[(t * G + g) * k + i])
            _ = ph^
            _ = ps^
        else:
            # many groups: scatter by group so one task owns each group's heap
            var sc = _scatter(b, tp, False, rank, threads)
            var seg_at = Int(sc.seg.unsafe_ptr())
            var sv = Int(sc.vals.unsafe_ptr())
            var sg = Int(sc.gids.unsafe_ptr())
            var heap_at = Int(heap)
            var size_at = Int(size)

            def own(p: Int) {imm}:
                var vp = _lp(sv)
                var gp = _ip(sg)
                var h = _lp(heap_at)
                var s = _np(size_at)
                for i in range(_np(seg_at)[p], _np(seg_at)[p + 1]):
                    var g = Int(gp[i])
                    s[g] = _offer(h + g * k, s[g], k, vp[i])

            par_n(own, tp, tp)
            _ = sc^

    def finish(mut self, unrank: Int) -> Tuple[List[Int], List[Int]]:
        """(values, offsets): group g's values, largest first, are `values[offsets[g] : offsets[g + 1]]`."""
        var vals = List[Int]()
        var off = List[Int]()
        var heap = _lp(Int(self.heap.unsafe_ptr()))
        for g in range(self.groups):
            off.append(len(vals))
            _heap_sort_desc(heap + g * self.k, self.size[g])
            for i in range(self.size[g]):
                var v = heap[g * self.k + i]
                if unrank != 0:
                    v = _lp(unrank)[Int(v)]
                vals.append(Int(v))
        off.append(len(vals))
        return (vals^, off^)


# ── the states of one query ──────────────────────────────────────────────────────────────────────────

@fieldwise_init
struct HolSpec(Copyable, Movable):
    var agg: Int            # the aggregate's index in the query
    var kind: Int
    var p: Float64
    var k: Int
    var src: Int            # the value source (a distinct value expression) it reads


struct RawState(Movable):
    """Every non-null (group, value) pair of a source, copied from the windows (percentiles need them all)."""
    var cg: List[List[Int32]]
    var cv: List[List[Int64]]

    def __init__(out self):
        self.cg = List[List[Int32]]()
        self.cv = List[List[Int64]]()

    def absorb(mut self, b: Blocks, threads: Int):
        var n = b.total
        if n == 0:
            return
        var ng = List[Int32](unsafe_uninit_length=n)
        var nv = List[Int64](unsafe_uninit_length=n)
        var nb = len(b.n)
        var at = List[Int](length=nb + 1, fill=0)
        for q in range(nb):
            at[q + 1] = at[q] + b.n[q]
        var tp = _tasks(n, nb, threads)
        var ng_at = Int(ng.unsafe_ptr())
        var nv_at = Int(nv.unsafe_ptr())
        var at_at = Int(at.unsafe_ptr())
        var bg = Int(b.g.unsafe_ptr())
        var bv = Int(b.v.unsafe_ptr())
        var bn = Int(b.n.unsafe_ptr())

        def copy(t: Int) {imm}:
            for q in range(_cut(nb, tp, t), _cut(nb, tp, t + 1)):
                var m = _np(bn)[q]
                var o = _np(at_at)[q]
                var sg = _ip(_np(bg)[q])
                var sv = _lp(_np(bv)[q])
                var dg = _ip(ng_at) + o
                var dv = _lp(nv_at) + o
                for j in range(m):
                    dg[j] = sg[j]
                    dv[j] = sv[j]

        par_n(copy, tp, tp)
        _ = at^
        self.cg.append(ng^)
        self.cv.append(nv^)

    def blocks(self) -> Blocks:
        var out = Blocks()
        for c in range(len(self.cg)):
            out.add_flat(Int(self.cg[c].unsafe_ptr()), Int(self.cv[c].unsafe_ptr()), len(self.cg[c]))
        return out^


struct HolResult(Movable):
    """Per holistic aggregate (in `specs` order): its value per group (distinct count, percentile) or its top-k
    values (flat, with offsets); and whether a group has a value at all."""
    var vals: List[List[Int]]
    var top: List[List[Int]]
    var top_off: List[List[Int]]
    var counts: List[List[Int]]      # per aggregate: the group's non-null values (distinct and percentile; top-k: its size)

    def __init__(out self):
        self.vals = List[List[Int]]()
        self.top = List[List[Int]]()
        self.top_off = List[List[Int]]()
        self.counts = List[List[Int]]()


struct HolStates(Movable):
    var groups: Int
    var threads: Int
    var specs: List[HolSpec]
    var ranks: List[List[Int64]]         # per source: code -> its place in the text order (empty: not text)
    var unranks: List[List[Int64]]       # per source: the place -> code
    var has_distinct: List[Bool]
    var has_raw: List[Bool]
    var distinct: List[DistinctState]    # per source
    var raw: List[RawState]              # per source
    var topk: List[TopKState]            # per spec (k = 0 for the others)

    def __init__(out self, groups: Int, var specs: List[HolSpec], var ranks: List[List[Int64]]) raises:
        self.groups = groups
        self.threads = core_count()
        var nsrc = len(ranks)
        self.has_distinct = List[Bool](length=nsrc, fill=False)
        self.has_raw = List[Bool](length=nsrc, fill=False)
        self.distinct = List[DistinctState]()
        self.raw = List[RawState]()
        self.topk = List[TopKState]()
        self.unranks = List[List[Int64]]()
        var parts = 4
        while parts < 4 * self.threads:
            parts *= 2
        for s in range(nsrc):
            var un = List[Int64](length=len(ranks[s]), fill=0)
            for c in range(len(ranks[s])):
                un[Int(ranks[s][c])] = Int64(c)
            self.unranks.append(un^)
        for spec in specs:
            if spec.kind == A_PCT:
                if not (spec.p >= 0.0 and spec.p <= 1.0):
                    raise Error("percentile: p must be between 0 and 1, got " + String(spec.p))
                self.has_raw[spec.src] = True
            elif spec.kind == A_DISTINCT:
                self.has_distinct[spec.src] = True
            elif spec.kind == A_TOPK:
                if spec.k < 1:
                    raise Error("top_k: k must be at least 1, got " + String(spec.k))
                if spec.k * groups > MAX_TOPK_WORDS:
                    raise Error("top_k: k=" + String(spec.k) + " for " + String(groups) + " groups would hold more than "
                                + String(MAX_TOPK_WORDS) + " values")
        for s in range(nsrc):
            self.distinct.append(DistinctState(parts if self.has_distinct[s] else 1))
            self.raw.append(RawState())
        for spec in specs:
            self.topk.append(TopKState(spec.k if spec.kind == A_TOPK else 1, groups if spec.kind == A_TOPK else 1))
        self.specs = specs^
        self.ranks = ranks^

    def _rank(self, s: Int) -> Int:
        return Int(self.ranks[s].unsafe_ptr()) if len(self.ranks[s]) > 0 else 0

    def _unrank(self, s: Int) -> Int:
        return Int(self.unranks[s].unsafe_ptr()) if len(self.unranks[s]) > 0 else 0

    def absorb(mut self, s: Int, b: Blocks):
        """Take one window's pairs of source s into every state that reads it."""
        if b.total == 0:
            return
        if self.has_distinct[s]:
            self.distinct[s].absorb(b, self.threads)
        if self.has_raw[s]:
            self.raw[s].absorb(b, self.threads)
        for h in range(len(self.specs)):
            if self.specs[h].kind == A_TOPK and self.specs[h].src == s:
                self.topk[h].absorb(b, self._rank(s), self.threads)

    def absorb_rows(mut self, s: Int, gids: List[Int32], vals: List[Int64]):
        """The same for a few single pairs (a frame's changed and appended rows)."""
        var b = Blocks()
        b.add_flat(Int(gids.unsafe_ptr()), Int(vals.unsafe_ptr()), len(gids))
        self.absorb(s, b)
        _ = gids
        _ = vals

    def finish(mut self) -> HolResult:
        var res = HolResult()
        var G = self.groups
        var distinct_of = List[List[Int]]()
        for s in range(len(self.distinct)):
            distinct_of.append(self.distinct[s].counts(G, self.threads) if self.has_distinct[s] else List[Int]())
        var pct_of = List[Percentiles]()           # per source
        for s in range(len(self.distinct)):
            var ps = List[Float64]()
            for spec in self.specs:
                if spec.kind == A_PCT and spec.src == s:
                    ps.append(spec.p)
            if self.has_raw[s]:
                pct_of.append(_percentiles(self.raw[s].blocks(), G, ps, self._rank(s), self._unrank(s), self.threads))
            else:
                pct_of.append(Percentiles())
        for h in range(len(self.specs)):
            ref spec = self.specs[h]
            if spec.kind == A_DISTINCT:
                res.vals.append(distinct_of[spec.src].copy())
                res.counts.append(distinct_of[spec.src].copy())
                res.top.append(List[Int]())
                res.top_off.append(List[Int]())
            elif spec.kind == A_PCT:
                var q = 0                          # this p's place among its source's percentiles
                for h2 in range(h):
                    if self.specs[h2].kind == A_PCT and self.specs[h2].src == spec.src:
                        q += 1
                res.vals.append(pct_of[spec.src].vals[q].copy())
                res.counts.append(pct_of[spec.src].counts.copy())
                res.top.append(List[Int]())
                res.top_off.append(List[Int]())
            else:
                var t = self.topk[h].finish(self._unrank(spec.src))
                res.counts.append(List[Int](self.topk[h].size))
                res.vals.append(List[Int](length=G, fill=0))
                res.top.append(t[0].copy())
                res.top_off.append(t[1].copy())
        return res^
