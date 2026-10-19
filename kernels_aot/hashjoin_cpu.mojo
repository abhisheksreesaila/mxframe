"""
F-153: the hash join kernel of the MX frame engine (CPU).

A generic engine kernel: it knows int64 keys and row numbers, nothing about tables, SQL or any application. Every entry point takes
addresses (Int) of plain int64 / int32 / byte arrays, so the Mojo API, the SQL layer, the Python engine (ctypes, through the C entry
points at the end) and anything built on the engine call the same code.

    keys  Int64[n]   the normalized key of each row (the caller makes every kind of column one int64: integers as they are, scaled
                     decimals aligned, dates, bools, dictionary codes translated to one dictionary, floats by their ordered bits,
                     composite keys packed or hashed)
    ok    UInt8[n]   1 where the row may match, 0 where it never does (a NULL key, an unmatchable value); address 0: all rows may

`hj_build(keys, ok, n, threads)` builds a `JoinMap` over the build side. The layout follows the keys:

  direct   the keys span at most max(8 x rows, 2^20) values and at most 2^28: an array indexed by key - lowest. Unique keys: `u`, one
           Int32 row per key (-1 none). Repeated keys: `s`, the end offset of each key's rows in `r` (the rows of one key are
           consecutive, in build order; a key starts where the previous one ends).
  hash     any other keys: radix partitions by the hash's top bits, sized so a partition's table (open addressing, linear probing, load
           at most one half, 16-byte entries: key, count, first row or row-list start) stays in the L2 cache; the same `r` lists rows
           in build order for repeated keys. A presence bitmap over the key range (when it is at most 2^26 values) turns most misses
           into one load from a cache-resident bitmap.

The build is parallel and uses no atomics: rows are scattered into partitions by a stable counting pass (a histogram per slice, a prefix
over (partition, slice)), and each partition is built by one task into memory only it writes. Probes are read-only, run in parallel over
slices of the probe rows, and prefetch the table line of the next group of keys before resolving the current one.

Probing: `hj_probe_unique` (a map with unique keys: the build row of each probe row or -1), `hj_probe_pairs` (any map: every
(probe row, build row) pair in probe-row order, counted first so a result past the caller's budget is refused before any memory is
taken), and `hj_probe_block` (the fused form: a block of selected rows looked up in place, for a pipeline's front stage).

The serial build (no Mojo async runtime, the Apple link) runs every task in a loop on the calling thread; tasks never wait on each other.
"""

from std.memory.alloc import unsafe_alloc
from std.memory import Pointer, unsafe_memset_zero
from std.ffi import external_call
from std.sys import prefetch, PrefetchOptions, CompilationTarget
from _fpenv import par_n, core_count, start_pool

comptime IP = Pointer[Int32, MutUntrackedOrigin]
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime UP = Pointer[UInt64, MutUntrackedOrigin]
comptime BP = Pointer[UInt8, MutUntrackedOrigin]

comptime MODE_EMPTY = 0
comptime MODE_DIRECT = 1
comptime MODE_HASH = 2

comptime DIRECT_MAX = 1 << 28           # the most key values a direct array covers
comptime DIRECT_MIN_RANGE = 1 << 20     # a range this small is always direct (the array is small whatever the row count)
comptime DIRECT_FACTOR = 8              # past this many key values a build row the direct array is too sparse
comptime PART_ROWS = 8192               # rows per hash partition: a 16-byte-entry table of 2x that many slots is 256 KB (L2)
comptime MAX_PBITS = 12
comptime BITMAP_MAX_RANGE = 1 << 26     # a hashed map's presence bitmap covers at most 8 MB
comptime BITMAP_MIN_KEYS = 4096         # a smaller map is cache-resident anyway
comptime GROUP = 16                     # keys looked up per prefetch group
comptime HUGE = 1 << 21
comptime E_TOO_MANY_ROWS = -31          # 2^31 rows or more on a side
comptime E_BUDGET = -32                 # the pairs would pass the caller's budget


@always_inline
def _mix(h: UInt64) -> UInt64:
    """splitmix64's finalizer."""
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _alloc(n: Int) -> Int:
    """n bytes, 2 MiB aligned and marked for huge pages when large (first-touch faults of many threads serialize otherwise)."""
    if n < HUGE:
        return Int(unsafe_alloc[UInt8](max(n, 1), alignment=64))
    var size = (n + HUGE - 1) // HUGE * HUGE
    var b = unsafe_alloc[UInt8](size, alignment=HUGE)
    comptime if CompilationTarget.is_linux():
        _ = external_call["madvise", Int32](Int(b), size, Int32(14))
    return Int(b)


def _free(a: Int):
    if a != 0:
        BP(unsafe_from_address=a).free()


@always_inline
def _prefetch(addr: Int):
    prefetch[PrefetchOptions().for_read().high_locality().to_data_cache()](LP(unsafe_from_address=addr))


def _bits(n: Int) -> Int:
    var b = 0
    while (1 << b) < n:
        b += 1
    return b


struct JoinMap(Movable):
    var mode: Int
    var n: Int                       # rows the build was given
    var nvalid: Int                  # rows with a usable key
    var lo: Int64                    # the lowest and highest usable key
    var hi: Int64
    var span: Int                    # direct: hi - lo + 1
    var unique: Bool                 # no key repeats
    var pbits: Int                   # hash: 1 << pbits partitions
    var u: Int                       # direct, unique: Int32[span]  build row or -1
    var s: Int                       # direct, repeated: Int32[span]  end offset in r of each key's rows
    var r: Int                       # repeated: Int32[nvalid]  build rows, grouped by key, in build order
    var ent: Int                     # hash: entries, 16 bytes: key Int64, count Int32, start Int32 (the row when unique, else the offset in r)
    var eoff: Int                    # hash: Int64[partitions + 1]: first entry of each partition's table
    var bitmap: Int                  # hash: presence bits over lo .. hi, or 0
    var bytes: Int                   # memory the map holds

    def __init__(out self):
        self.mode = MODE_EMPTY
        self.n = 0
        self.nvalid = 0
        self.lo = 0
        self.hi = 0
        self.span = 0
        self.unique = True
        self.pbits = 0
        self.u = 0
        self.s = 0
        self.r = 0
        self.ent = 0
        self.eoff = 0
        self.bitmap = 0
        self.bytes = 0

    def __del__(deinit self):
        _free(self.u)
        _free(self.s)
        _free(self.r)
        _free(self.ent)
        _free(self.eoff)
        _free(self.bitmap)

    def kind(self) -> String:
        if self.mode == MODE_EMPTY:
            return "empty"
        var base = "direct array" if self.mode == MODE_DIRECT else "hash table (" + String(1 << self.pbits) + " partitions)"
        return base + (", unique keys" if self.unique else ", repeated keys (CSR rows)")


struct MapView(ImplicitlyCopyable, Movable):
    """The addresses and numbers a lookup needs, as plain values: a copy a closure can capture (a map keeps its memory; the view must
    not outlive it)."""
    var mode: Int
    var unique: Bool
    var lo: Int64
    var hi: Int64
    var u: Int
    var s: Int
    var r: Int
    var ent: Int
    var eoff: Int
    var pbits: Int
    var bitmap: Int

    def __init__(out self, m: JoinMap):
        self.mode = m.mode
        self.unique = m.unique
        self.lo = m.lo
        self.hi = m.hi
        self.u = m.u
        self.s = m.s
        self.r = m.r
        self.ent = m.ent
        self.eoff = m.eoff
        self.pbits = m.pbits
        self.bitmap = m.bitmap


@always_inline
def _slice(n: Int, t: Int, T: Int) -> Int:
    return n * t // T


@always_inline
def _part_of(k: Int64, direct: Bool, base: Int64, pshift: Int, pbits: Int) -> Int:
    """The partition of key k: a run of consecutive keys (direct) or the hash's top bits."""
    if direct:
        return Int(UInt64(k - base) >> UInt64(pshift))
    if pbits == 0:
        return 0
    return Int(_mix(UInt64(k)) >> UInt64(64 - pbits))


# ── build ────────────────────────────────────────────────────────────────────────────────────────────

def hj_build(keys_a: Int, ok_a: Int, n: Int, threads: Int) -> JoinMap:
    """The map of the build side's keys. See the module doc for the layout it chooses."""
    var m = JoinMap()
    m.n = n
    if n <= 0:
        return m^
    var T = max(1, min(threads if threads > 0 else core_count(), (n + 16383) // 16384))
    var keys = LP(unsafe_from_address=keys_a)
    var has_ok = ok_a != 0
    var oks = BP(unsafe_from_address=ok_a)
    # ── pass 0: the lowest and highest usable key and how many there are
    var st = LP(unsafe_from_address=_alloc(8 * 3 * T))

    def stats(t: Int) {imm}:
        var lo = Int64.MAX
        var hi = Int64.MIN
        var c = 0
        for i in range(_slice(n, t, T), _slice(n, t + 1, T)):
            if has_ok and oks[i] == 0:
                continue
            var k = keys[i]
            lo = min(lo, k)
            hi = max(hi, k)
            c += 1
        st[3 * t] = lo
        st[3 * t + 1] = hi
        st[3 * t + 2] = Int64(c)

    par_n(stats, T, T)
    var lo = Int64.MAX
    var hi = Int64.MIN
    var nvalid = 0
    for t in range(T):
        lo = min(lo, st[3 * t])
        hi = max(hi, st[3 * t + 1])
        nvalid += Int(st[3 * t + 2])
    _free(Int(st))
    m.nvalid = nvalid
    if nvalid == 0:
        return m^
    m.lo = lo
    m.hi = hi
    var span_u = UInt64(hi) - UInt64(lo)
    var direct = span_u < UInt64(DIRECT_MAX) and Int(span_u) + 1 <= max(DIRECT_FACTOR * nvalid, DIRECT_MIN_RANGE)
    var span = Int(span_u) + 1 if direct else 0
    # ── the partitions: direct ones are runs of consecutive keys (a partition owns a stretch of the array), hash ones the hash's top bits
    var pshift = 0
    var P = 1
    var pbits = 0
    if direct:
        var target = max(1, 4 * T)
        while ((span + (1 << pshift) - 1) >> pshift) > target:
            pshift += 1
        P = (span + (1 << pshift) - 1) >> pshift
    else:
        while (nvalid >> pbits) > PART_ROWS and pbits < MAX_PBITS:
            pbits += 1
        P = 1 << pbits
    m.mode = MODE_DIRECT if direct else MODE_HASH
    m.span = span
    m.pbits = pbits
    var hist = LP(unsafe_from_address=_alloc(8 * T * P))
    var poff = LP(unsafe_from_address=_alloc(8 * (P + 1)))
    var prow = IP(unsafe_from_address=_alloc(4 * nvalid))
    var pkey = LP(unsafe_from_address=_alloc(8 * nvalid))
    var base_key = lo

    def histo(t: Int) {imm}:
        var h = hist + t * P
        for p in range(P):
            h[p] = 0
        for i in range(_slice(n, t, T), _slice(n, t + 1, T)):
            if has_ok and oks[i] == 0:
                continue
            h[_part_of(keys[i], direct, base_key, pshift, pbits)] += 1

    par_n(histo, T, T)
    var run = 0
    for p in range(P):
        poff[p] = Int64(run)
        for t in range(T):
            var c = Int(hist[t * P + p])
            hist[t * P + p] = Int64(run)
            run += c
    poff[P] = Int64(run)

    def scatter(t: Int) {imm}:
        var h = hist + t * P
        for i in range(_slice(n, t, T), _slice(n, t + 1, T)):
            if has_ok and oks[i] == 0:
                continue
            var k = keys[i]
            var p = _part_of(k, direct, base_key, pshift, pbits)
            var at = Int(h[p])
            h[p] = Int64(at + 1)
            prow[at] = Int32(i)
            pkey[at] = k

    par_n(scatter, T, T)
    _free(Int(hist))
    var pmax = LP(unsafe_from_address=_alloc(8 * P))
    var nt = max(1, min(T, P))
    if direct:
        _build_direct(m, prow, pkey, poff, pmax, P, pshift, nt, nvalid)
    else:
        _build_hash(m, prow, pkey, poff, pmax, P, pbits, nt, nvalid)
    _free(Int(pmax))
    _free(Int(prow))
    _free(Int(pkey))
    _free(Int(poff))
    if (not direct) and span_u < UInt64(BITMAP_MAX_RANGE) and nvalid >= BITMAP_MIN_KEYS:
        var nbytes = (Int(span_u) + 1 + 7) // 8
        var bm = BP(unsafe_from_address=_alloc(nbytes))
        unsafe_memset_zero(bm, nbytes)
        for i in range(n):
            if has_ok and oks[i] == 0:
                continue
            var d = Int(keys[i] - lo)
            bm[d >> 3] = bm[d >> 3] | (UInt8(1) << UInt8(d & 7))
        m.bitmap = Int(bm)
        m.bytes += nbytes
    return m^


def _build_direct(mut m: JoinMap, prow: IP, pkey: LP, poff: LP, pmax: LP, P: Int, pshift: Int, nt: Int, nvalid: Int):
    var span = m.span
    var lo = m.lo
    var S = IP(unsafe_from_address=_alloc(4 * span))                 # first the counts of each key's rows
    # pass A: count (each partition owns a stretch of S)
    def count(t: Int) {imm}:
        var p = t
        while p < P:
            var k0 = p << pshift
            var k1 = min(span, (p + 1) << pshift)
            for i in range(k0, k1):
                S[i] = 0
            var mx = Int32(0)
            for idx in range(Int(poff[p]), Int(poff[p + 1])):
                var d = Int(pkey[idx] - lo)
                var c = S[d] + 1
                S[d] = c
                mx = max(mx, c)
            pmax[p] = Int64(mx)
            p += nt

    par_n(count, nt, nt)
    var most = Int64(0)
    for p in range(P):
        most = max(most, pmax[p])
    if most <= 1:
        m.unique = True
        _free(Int(S))
        var U = IP(unsafe_from_address=_alloc(4 * span))

        def fill_u(t: Int) {imm}:
            var p = t
            while p < P:
                var k0 = p << pshift
                var k1 = min(span, (p + 1) << pshift)
                for i in range(k0, k1):
                    U[i] = -1
                for idx in range(Int(poff[p]), Int(poff[p + 1])):
                    U[Int(pkey[idx] - lo)] = prow[idx]
                p += nt

        par_n(fill_u, nt, nt)
        m.u = Int(U)
        m.bytes += 4 * span
        return
    m.unique = False
    var R = IP(unsafe_from_address=_alloc(4 * nvalid))

    def fill_s(t: Int) {imm}:
        var p = t
        while p < P:
            var k0 = p << pshift
            var k1 = min(span, (p + 1) << pshift)
            var run = Int32(Int(poff[p]))
            for i in range(k0, k1):                                   # the counts become start offsets ...
                var c = S[i]
                S[i] = run
                run += c
            for idx in range(Int(poff[p]), Int(poff[p + 1])):         # ... the rows are placed in build order, the offsets walk to the ends
                var d = Int(pkey[idx] - lo)
                var at = S[d]
                R[Int(at)] = prow[idx]
                S[d] = at + 1
            p += nt

    par_n(fill_s, nt, nt)
    m.s = Int(S)
    m.r = Int(R)
    m.bytes += 4 * span + 4 * nvalid


def _build_hash(mut m: JoinMap, prow: IP, pkey: LP, poff: LP, pmax: LP, P: Int, pbits: Int, nt: Int, nvalid: Int):
    var eoff = LP(unsafe_from_address=_alloc(8 * (P + 1)))
    var total = 0
    for p in range(P):
        var rows = Int(poff[p + 1] - poff[p])
        var cap = 4
        while cap < 2 * rows:
            cap *= 2
        eoff[p] = Int64(total)
        total += cap
    eoff[P] = Int64(total)
    var ent = Int(_alloc(16 * total))
    var shift = UInt64(64 - pbits)

    def count(t: Int) {imm}:
        var p = t
        while p < P:
            var e0 = Int(eoff[p])
            var cap = Int(eoff[p + 1]) - e0
            var mask = UInt64(cap - 1)
            var tab = ent + 16 * e0
            unsafe_memset_zero(BP(unsafe_from_address=tab), 16 * cap)
            var mx = Int32(0)
            for idx in range(Int(poff[p]), Int(poff[p + 1])):
                var k = pkey[idx]
                var i = Int(_mix(UInt64(k)) & mask)
                while True:
                    var e = tab + 16 * i
                    var cnt = IP(unsafe_from_address=e + 8)
                    if cnt[0] == 0:
                        LP(unsafe_from_address=e)[0] = k
                        cnt[0] = 1
                        mx = max(mx, Int32(1))
                        break
                    if LP(unsafe_from_address=e)[0] == k:
                        cnt[0] += 1
                        mx = max(mx, cnt[0])
                        break
                    i = Int(UInt64(i + 1) & mask)
            pmax[p] = Int64(mx)
            p += nt

    par_n(count, nt, nt)
    var most = Int64(0)
    for p in range(P):
        most = max(most, pmax[p])
    m.ent = ent
    m.eoff = Int(eoff)
    m.bytes += 16 * total + 8 * (P + 1)
    if most <= 1:
        m.unique = True

        def put_rows(t: Int) {imm}:
            var p = t
            while p < P:
                var e0 = Int(eoff[p])
                var cap = Int(eoff[p + 1]) - e0
                var mask = UInt64(cap - 1)
                var tab = ent + 16 * e0
                for idx in range(Int(poff[p]), Int(poff[p + 1])):
                    var k = pkey[idx]
                    var i = Int(_mix(UInt64(k)) & mask)
                    while LP(unsafe_from_address=tab + 16 * i)[0] != k or IP(unsafe_from_address=tab + 16 * i + 8)[0] == 0:
                        i = Int(UInt64(i + 1) & mask)
                    IP(unsafe_from_address=tab + 16 * i + 12)[0] = prow[idx]
                p += nt

        par_n(put_rows, nt, nt)
        return
    m.unique = False
    var R = IP(unsafe_from_address=_alloc(4 * nvalid))

    def place(t: Int) {imm}:
        var p = t
        while p < P:
            var e0 = Int(eoff[p])
            var cap = Int(eoff[p + 1]) - e0
            var mask = UInt64(cap - 1)
            var tab = ent + 16 * e0
            var run = Int32(Int(poff[p]))
            for i in range(cap):                                      # row lists laid out in slot order
                var c = IP(unsafe_from_address=tab + 16 * i + 8)[0]
                if c > 0:
                    IP(unsafe_from_address=tab + 16 * i + 12)[0] = run
                    run += c
            for idx in range(Int(poff[p]), Int(poff[p + 1])):
                var k = pkey[idx]
                var i = Int(_mix(UInt64(k)) & mask)
                while LP(unsafe_from_address=tab + 16 * i)[0] != k or IP(unsafe_from_address=tab + 16 * i + 8)[0] == 0:
                    i = Int(UInt64(i + 1) & mask)
                var sp = IP(unsafe_from_address=tab + 16 * i + 12)
                R[Int(sp[0])] = prow[idx]
                sp[0] += 1
            for i in range(cap):                                      # the cursors walked to the ends: back to the starts
                var c = IP(unsafe_from_address=tab + 16 * i + 8)[0]
                if c > 0:
                    IP(unsafe_from_address=tab + 16 * i + 12)[0] -= c
            p += nt

    par_n(place, nt, nt)
    m.r = Int(R)
    m.bytes += 4 * nvalid


# ── probe ────────────────────────────────────────────────────────────────────────────────────────────

@always_inline
def _hash_entry(ent: Int, eoff: LP, pbits: Int, k: Int64) -> Int:
    """The address of key k's entry, 0 if there is none."""
    var h = _mix(UInt64(k))
    var p = Int(h >> UInt64(64 - pbits)) if pbits > 0 else 0
    var e0 = Int(eoff[p])
    var mask = UInt64(Int(eoff[p + 1]) - e0 - 1)
    var tab = ent + 16 * e0
    var i = Int(h & mask)
    while True:
        var e = tab + 16 * i
        if IP(unsafe_from_address=e + 8)[0] == 0:
            return 0
        if LP(unsafe_from_address=e)[0] == k:
            return e
        i = Int(UInt64(i + 1) & mask)


@always_inline
def _has_bit(bm: BP, d: Int) -> Bool:
    return ((bm[d >> 3] >> UInt8(d & 7)) & 1) != 0


@always_inline
def _prefetch_key(m_mode: Int, lo: Int64, hi: Int64, span: Int, unique: Bool, u: Int, s: Int, ent: Int, eoff: LP, pbits: Int, bitmap: Int, k: Int64):
    """Brings the table line of key k toward the cache."""
    if k < lo or k > hi:
        return
    if m_mode == MODE_DIRECT:
        _prefetch((u if unique else s) + 4 * Int(k - lo))
        return
    if bitmap != 0 and not _has_bit(BP(unsafe_from_address=bitmap), Int(k - lo)):
        return
    var h = _mix(UInt64(k))
    var p = Int(h >> UInt64(64 - pbits)) if pbits > 0 else 0
    var e0 = Int(eoff[p])
    var mask = UInt64(Int(eoff[p + 1]) - e0 - 1)
    _prefetch(ent + 16 * (e0 + Int(h & mask)))


def hj_probe_unique(m: JoinMap, keys_a: Int, ok_a: Int, n: Int, out_a: Int, threads: Int):
    """out[i] = the build row whose key is keys[i], or -1 (a map built with unique keys)."""
    var out = IP(unsafe_from_address=out_a)
    if m.mode == MODE_EMPTY:
        for i in range(n):
            out[i] = -1
        return
    var T = max(1, min(threads if threads > 0 else core_count(), (n + 16383) // 16384))
    var keys = LP(unsafe_from_address=keys_a)
    var has_ok = ok_a != 0
    var oks = BP(unsafe_from_address=ok_a)
    var mode = m.mode
    var lo = m.lo
    var hi = m.hi
    var span = m.span
    var u = m.u
    var ent = m.ent
    var eoff = LP(unsafe_from_address=m.eoff)
    var pbits = m.pbits
    var bitmap = m.bitmap
    var uniq = m.unique

    def work(t: Int) {imm}:
        var i = _slice(n, t, T)
        var end = _slice(n, t + 1, T)
        while i < end:
            var g = min(GROUP, end - i)
            for j in range(g):
                var k = keys[i + j]
                if (not has_ok) or oks[i + j] != 0:
                    _prefetch_key(mode, lo, hi, span, uniq, u, 0, ent, eoff, pbits, bitmap, k)
            for j in range(g):
                var r = Int32(-1)
                if (not has_ok) or oks[i + j] != 0:
                    var k = keys[i + j]
                    if k >= lo and k <= hi:
                        if mode == MODE_DIRECT:
                            r = IP(unsafe_from_address=u)[Int(k - lo)]
                        elif bitmap == 0 or _has_bit(BP(unsafe_from_address=bitmap), Int(k - lo)):
                            var e = _hash_entry(ent, eoff, pbits, k)
                            if e != 0:
                                r = IP(unsafe_from_address=e + 12)[0]
                out[i + j] = r
            i += g

    par_n(work, T, T)


@always_inline
def _match(mode: Int, unique: Bool, lo: Int64, hi: Int64, u: Int, s: Int, ent: Int, eoff: LP, pbits: Int, bitmap: Int, k: Int64,
           mut start: Int, mut count: Int):
    """Where key k's build rows are: `count` of them (0: none), the one row `start` of a unique map, or the first in `r` of a repeated one."""
    count = 0
    start = 0
    if k < lo or k > hi:
        return
    if mode == MODE_DIRECT:
        var d = Int(k - lo)
        if unique:
            var b = IP(unsafe_from_address=u)[d]
            if b >= 0:
                start = Int(b)
                count = 1
            return
        var e = Int(IP(unsafe_from_address=s)[d])
        var b0 = Int(IP(unsafe_from_address=s)[d - 1]) if d > 0 else 0
        start = b0
        count = e - b0
        return
    if bitmap != 0 and not _has_bit(BP(unsafe_from_address=bitmap), Int(k - lo)):
        return
    var e = _hash_entry(ent, eoff, pbits, k)
    if e != 0:
        start = Int(IP(unsafe_from_address=e + 12)[0])
        count = 1 if unique else Int(IP(unsafe_from_address=e + 8)[0])


def hj_probe_pairs(m: JoinMap, keys_a: Int, ok_a: Int, n: Int, threads: Int, budget_rows: Int, out_a: Int) -> Int:
    """Every (probe row, build row) pair, in probe-row order. `out` (4 words): [0] the number of pairs, [1] address of Int32 probe rows,
    [2] address of Int32 build rows (both from `_alloc`: release with `hj_release`). Returns 0, or E_BUDGET when the pairs would pass
    `budget_rows` (nothing is allocated, out[0] says how many there would be)."""
    var out = LP(unsafe_from_address=out_a)
    out[0] = 0
    out[1] = 0
    out[2] = 0
    if m.mode == MODE_EMPTY or n <= 0:
        return 0
    var T = max(1, min(threads if threads > 0 else core_count(), (n + 16383) // 16384))
    var keys = LP(unsafe_from_address=keys_a)
    var has_ok = ok_a != 0
    var oks = BP(unsafe_from_address=ok_a)
    var mode = m.mode
    var lo = m.lo
    var hi = m.hi
    var span = m.span
    var u = m.u
    var s = m.s
    var r = IP(unsafe_from_address=m.r)
    var ent = m.ent
    var eoff = LP(unsafe_from_address=m.eoff)
    var pbits = m.pbits
    var bitmap = m.bitmap
    var uniq = m.unique
    var totals = LP(unsafe_from_address=_alloc(8 * (T + 1)))

    def count_pass(t: Int) {imm}:
        var total = 0
        var i = _slice(n, t, T)
        var end = _slice(n, t + 1, T)
        while i < end:
            var g = min(GROUP, end - i)
            for j in range(g):
                if (not has_ok) or oks[i + j] != 0:
                    _prefetch_key(mode, lo, hi, span, uniq, u, s, ent, eoff, pbits, bitmap, keys[i + j])
            for j in range(g):
                if (not has_ok) or oks[i + j] != 0:
                    var st = 0
                    var c = 0
                    _match(mode, uniq, lo, hi, u, s, ent, eoff, pbits, bitmap, keys[i + j], st, c)
                    total += c
            i += g
        totals[t] = Int64(total)

    par_n(count_pass, T, T)
    var all = 0
    for t in range(T):
        var c = Int(totals[t])
        totals[t] = Int64(all)
        all += c
    out[0] = Int64(all)
    if all > budget_rows:
        _free(Int(totals))
        return E_BUDGET
    var pr = IP(unsafe_from_address=_alloc(4 * max(all, 1)))
    var br = IP(unsafe_from_address=_alloc(4 * max(all, 1)))

    def fill_pass(t: Int) {imm}:
        var at = Int(totals[t])
        var i = _slice(n, t, T)
        var end = _slice(n, t + 1, T)
        while i < end:
            var g = min(GROUP, end - i)
            for j in range(g):
                if (not has_ok) or oks[i + j] != 0:
                    _prefetch_key(mode, lo, hi, span, uniq, u, s, ent, eoff, pbits, bitmap, keys[i + j])
            for j in range(g):
                if (not has_ok) or oks[i + j] != 0:
                    var st = 0
                    var c = 0
                    _match(mode, uniq, lo, hi, u, s, ent, eoff, pbits, bitmap, keys[i + j], st, c)
                    if uniq:
                        if c == 1:
                            pr[at] = Int32(i + j)
                            br[at] = Int32(st)
                            at += 1
                    else:
                        for q in range(c):
                            pr[at] = Int32(i + j)
                            br[at] = r[st + q]
                            at += 1
            i += g

    par_n(fill_pass, T, T)
    _free(Int(totals))
    out[1] = Int64(Int(pr))
    out[2] = Int64(Int(br))
    return 0


def hj_release(addr: Int):
    """Frees an array a probe returned."""
    _free(addr)


def hj_probe_block(v: MapView, keys: LP, ok: BP, has_ok: Bool, i0: Int, sel: IP, nsel: Int, prow: IP) -> Int:
    """The fused form (a map with unique keys): for the block rows `sel[0..nsel)` (row = i0 + sel[j]) look the key up; rows with a partner stay
    in `sel` (compacted in place, in order) and `prow[q]` is the partner's build row of the q-th kept row. Returns how many are kept."""
    if v.mode == MODE_EMPTY:
        return 0
    var mode = v.mode
    var lo = v.lo
    var hi = v.hi
    var u = v.u
    var ent = v.ent
    var eoff = LP(unsafe_from_address=v.eoff)
    var pbits = v.pbits
    var bitmap = v.bitmap
    var kept = 0
    var i = 0
    while i < nsel:
        var g = min(GROUP, nsel - i)
        for j in range(g):
            var row = i0 + Int(sel[i + j])
            if (not has_ok) or ok[row] != 0:
                _prefetch_key(mode, lo, hi, 0, True, u, 0, ent, eoff, pbits, bitmap, keys[row])
        for j in range(g):
            var s0 = sel[i + j]
            var row = i0 + Int(s0)
            if has_ok and ok[row] == 0:
                continue
            var k = keys[row]
            if k < lo or k > hi:
                continue
            var b = Int32(-1)
            if mode == MODE_DIRECT:
                b = IP(unsafe_from_address=u)[Int(k - lo)]
            elif bitmap == 0 or _has_bit(BP(unsafe_from_address=bitmap), Int(k - lo)):
                var e = _hash_entry(ent, eoff, pbits, k)
                if e != 0:
                    b = IP(unsafe_from_address=e + 12)[0]
            if b >= 0:
                sel[kept] = s0
                prow[kept] = b
                kept += 1
        i += g
    return kept
