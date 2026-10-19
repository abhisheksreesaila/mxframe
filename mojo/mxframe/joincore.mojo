"""The inner equi-join of two tables, as an engine feature (F-153): key normalization, the hash join kernel's calls, the pairs, the gather.

Everything here works on `SharedTable`s and names no query language and no application: the SQL layer (`join.mojo`, `sqlexec.mojo`), the
Query builder (`Query.join`) and any other caller turn their input into two tables, the columns that are the keys and the columns to carry, and
call these.

- `normalize_keys` turns each key column into one int64 per row: integers as they are, DECIMAL aligned to the larger scale (INT is scale 0; a value
  that overflows when aligned cannot equal anything on the other side, so it never matches), DATE and BOOL as stored, TEXT through a code
  translation from the probe side's dictionary to the build side's (byte-exact: join equality is byte equality; a word the build side lacks never
  matches), FLOAT by `float_key` (-0.0 equals 0.0 and every NaN equals every NaN, DuckDB's rule), NULL never matches; several keys are packed into one
  int64 when their ranges multiply to fewer than 2^62, otherwise hashed to one and the candidate pairs are verified against the key columns. A single
  key that needs no change is the column's own values: nothing is copied.
- `join_pairs` probes the kernel's map (`kernels/hashjoin_cpu.mojo`) and returns every matching (probe row, build row) pair in probe-row order, or
  refuses past a budget before allocating.
- `gather_join` makes the joined table from the pairs: the carried columns of each side at its rows, gathered in parallel.
"""
from .table import Table, SharedTable, Column, INT, DECIMAL, TEXT, DATE, BOOL, FLOAT
from .holistic import float_key
from .kernels._fpenv import par_n, core_count
from .kernels.hashjoin_cpu import JoinMap, hj_build, hj_probe_pairs, hj_release, E_BUDGET

comptime JOIN_BUDGET_BYTES = 3 << 30        # the joined rows a join may materialize (8 bytes a value); past it a readable error (spill is later)


def _pow10(k: Int) raises -> Int64:
    if k < 0 or k > 18:
        raise Error("a power of ten outside 10^0 .. 10^18")
    var v: Int64 = 1
    for _ in range(k):
        v *= 10
    return v


def commas(n: Int) -> String:
    var s = String(n if n >= 0 else -n)
    var out = String()
    var k = s.byte_length()
    for i in range(k):
        if i > 0 and (k - i) % 3 == 0:
            out += ","
        out += String(s[byte=i:i + 1])
    return ("-" if n < 0 else "") + out


def _unique_name(t: Table, name: String) -> String:
    if t.index(name) < 0:
        return name
    var i = 1
    while t.index(name + "_" + String(i)) >= 0:
        i += 1
    return name + "_" + String(i)


# ── keys ─────────────────────────────────────────────────────────────────────────────────────────────

struct JoinKeys(Movable):
    var bkeys: List[Int64]           # the build side's key of each row, the probe side's ...
    var bok: List[UInt8]             # ... and whether the row can match at all
    var pkeys: List[Int64]
    var pok: List[UInt8]
    var verify: Bool                 # the keys are hashes of tuples: candidate pairs are checked against the columns
    var bcomp: List[List[Int64]]     # verify: each key column's normalized values
    var pcomp: List[List[Int64]]
    var layout: String               # for the notes: how several keys became one
    var bk_addr: Int                 # the arrays the kernel reads: the lists above, or a column's own values when no key had to change
    var bo_addr: Int                 # (a `ok` address of 0: every row may match)
    var pk_addr: Int
    var po_addr: Int

    def __init__(out self):
        self.bk_addr = 0
        self.bo_addr = 0
        self.pk_addr = 0
        self.po_addr = 0
        self.bkeys = List[Int64]()
        self.bok = List[UInt8]()
        self.pkeys = List[Int64]()
        self.pok = List[UInt8]()
        self.verify = False
        self.bcomp = List[List[Int64]]()
        self.pcomp = List[List[Int64]]()
        self.layout = String()


def _mix(h: UInt64) -> UInt64:
    var z = h
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB
    return z ^ (z >> 31)


def _valid_bits(t: SharedTable, c: Int) -> Int:
    return t._table[].cols[c].validity_address()


def _column_keys(t: SharedTable, c: Int, kind: Int, mul: Int64, translate: List[Int32], translating: Bool, mut vals: List[Int64], mut ok: List[UInt8]):
    """One key column as int64 values and `ok` flags (see the module doc)."""
    var n = t.rows()
    vals = List[Int64](length=n, fill=0)
    ok = List[UInt8](length=n, fill=1)
    var src = t.values(c).unsafe_ptr()
    var valid = _valid_bits(t, c)
    var vp = vals.unsafe_ptr()
    var op = ok.unsafe_ptr()
    for i in range(n):
        if valid != 0 and (Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)[i >> 3] >> UInt8(i & 7)) & 1 == 0:
            op[i] = 0
            continue
        var v = src[i]
        if kind == FLOAT.value:
            vp[i] = float_key(v)
        elif kind == TEXT.value:
            if translating:
                var tr = Int(translate[Int(v)])
                if tr < 0:
                    op[i] = 0
                    continue
                vp[i] = Int64(tr)
            else:
                vp[i] = v
        elif mul != 1:
            var r = v * mul
            if (v != 0 and r // v != mul) or (v == Int64.MIN):
                op[i] = 0
                continue
            vp[i] = r
        else:
            vp[i] = v


def _ok_bytes(t: SharedTable, c: Int, mut ok: List[UInt8]) -> Int:
    """The `ok` bytes of a column's validity (0 when it has no NULL: every row may match)."""
    if t.null_count(c) == 0:
        return 0
    var n = t.rows()
    ok = List[UInt8](unsafe_uninit_length=n)
    var valid = _valid_bits(t, c)
    var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
    for i in range(n):
        ok[i] = (bits[i >> 3] >> UInt8(i & 7)) & 1
    return Int(ok.unsafe_ptr())


def _pack_plain(bt: SharedTable, pt: SharedTable, bcols: List[Int], pcols: List[Int], mut keys: JoinKeys, threads: Int) raises -> Bool:
    """Several key columns that need no change (integers or decimals of one scale, dates, bools) and have no NULL: packed into one int64 straight
    from the columns, in parallel slices (the build side's ranges, then both sides' keys). False when the columns are not that plain or the ranges
    do not fit 2^62 values (the general path takes over)."""
    var nk = len(bcols)
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var plain = kind == DATE.value or kind == BOOL.value or ((kind == INT.value or kind == DECIMAL.value) and bt.scale(bcols[j]) == pt.scale(pcols[j]))
        if not plain or bt.null_count(bcols[j]) != 0 or pt.null_count(pcols[j]) != 0:
            return False
    var nb = bt.rows()
    var np = pt.rows()
    var bsrc = List[Int](length=nk, fill=0)
    var psrc = List[Int](length=nk, fill=0)
    for j in range(nk):
        bsrc[j] = Int(bt.values(bcols[j]).unsafe_ptr())
        psrc[j] = Int(pt.values(pcols[j]).unsafe_ptr())
    var T = max(1, min(threads if threads > 0 else core_count(), nb // 65536 + 1))
    var mins = List[Int64](length=T * nk, fill=Int64.MAX)
    var maxs = List[Int64](length=T * nk, fill=Int64.MIN)
    var mnp = mins.unsafe_ptr()
    var mxp = maxs.unsafe_ptr()

    def ranges(t: Int) {imm}:
        var a = nb * t // T
        var b = nb * (t + 1) // T
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=bsrc[j])
            var lo = Int64.MAX
            var hi = Int64.MIN
            for i in range(a, b):
                var v = src[i]
                lo = min(lo, v)
                hi = max(hi, v)
            mnp[t * nk + j] = lo
            mxp[t * nk + j] = hi

    par_n(ranges, T, T)
    var los = List[Int64]()
    var spans = List[UInt64]()
    var strides = List[UInt64]()
    var product = UInt64(1)
    for j in range(nk):
        var lo = Int64.MAX
        var hi = Int64.MIN
        for t in range(T):
            lo = min(lo, mins[t * nk + j])
            hi = max(hi, maxs[t * nk + j])
        if lo > hi:
            lo = 0
            hi = 0
        var span = UInt64(hi) - UInt64(lo) + 1
        if span == 0 or product > (UInt64(1) << 62) // span:
            return False
        los.append(lo)
        spans.append(span)
        strides.append(product)
        product *= span
    keys.bkeys = List[Int64](unsafe_uninit_length=nb)
    keys.pkeys = List[Int64](unsafe_uninit_length=np)
    keys.pok = List[UInt8](unsafe_uninit_length=np)
    var bk = keys.bkeys.unsafe_ptr()
    var pk = keys.pkeys.unsafe_ptr()
    var po = keys.pok.unsafe_ptr()
    var TB = max(1, min(threads if threads > 0 else core_count(), nb // 65536 + 1))
    var TP = max(1, min(threads if threads > 0 else core_count(), np // 65536 + 1))

    def build_keys(t: Int) {imm}:
        var a = nb * t // TB
        var b = nb * (t + 1) // TB
        for i in range(a, b):
            bk[i] = 0
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=bsrc[j])
            var lo = los[j]
            var st = strides[j]
            for i in range(a, b):
                bk[i] = Int64(UInt64(bk[i]) + UInt64(src[i] - lo) * st)

    def probe_keys(t: Int) {imm}:
        var a = np * t // TP
        var b = np * (t + 1) // TP
        for i in range(a, b):
            pk[i] = 0
            po[i] = 1
        for j in range(nk):
            var src = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=psrc[j])
            var lo = los[j]
            var sp = spans[j]
            var st = strides[j]
            for i in range(a, b):
                var d = UInt64(src[i] - lo)
                if d >= sp:
                    po[i] = 0                                         # outside the build side's range: it cannot match
                else:
                    pk[i] = Int64(UInt64(pk[i]) + d * st)

    par_n(build_keys, TB, TB)
    par_n(probe_keys, TP, TP)
    keys.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.bo_addr = 0
    keys.po_addr = Int(keys.pok.unsafe_ptr())
    return True


def normalize_keys(bt: SharedTable, pt: SharedTable, bcols: List[Int], pcols: List[Int], mut keys: JoinKeys, threads: Int = 0) raises:
    """The keys of the build side `bt` and the probe side `pt`, columns `bcols[j]` and `pcols[j]` being compared pairwise."""
    var nk = len(bcols)
    if nk == 1:
        var kind = bt.kind(bcols[0]).value
        var bs = bt.scale(bcols[0])
        var ps = pt.scale(pcols[0])
        var numeric = kind == INT.value or kind == DECIMAL.value
        if (numeric and bs == ps) or kind == DATE.value or kind == BOOL.value or (kind == TEXT.value and False):
            # the key is the column itself: no pass over the rows, only the NULL flags (when there are any)
            keys.bk_addr = Int(bt.values(bcols[0]).unsafe_ptr())
            keys.pk_addr = Int(pt.values(pcols[0]).unsafe_ptr())
            keys.bo_addr = _ok_bytes(bt, bcols[0], keys.bok)
            keys.po_addr = _ok_bytes(pt, pcols[0], keys.pok)
            keys.layout = "one key (the column itself)"
            return
    if nk > 1 and _pack_plain(bt, pt, bcols, pcols, keys, threads):
        return
    var bvals = List[List[Int64]]()
    var boks = List[List[UInt8]]()
    var pvals = List[List[Int64]]()
    var poks = List[List[UInt8]]()
    for j in range(nk):
        var kind = bt.kind(bcols[j]).value
        var bs = bt.scale(bcols[j])
        var ps = pt.scale(pcols[j])
        var bmul = Int64(1)
        var pmul = Int64(1)
        var numeric = kind == INT.value or kind == DECIMAL.value
        if numeric and bs != ps:
            if bs < ps:
                bmul = _pow10(ps - bs)
            else:
                pmul = _pow10(bs - ps)
        var translate = List[Int32]()
        if kind == TEXT.value:                                       # the probe side's code -> the build side's, once per word
            var words = pt.dictionary(pcols[j])
            for w in range(len(words)):
                translate.append(Int32(bt.code_of(bcols[j], String(words[w]))))
        var bv = List[Int64]()
        var bo = List[UInt8]()
        var pv = List[Int64]()
        var po = List[UInt8]()
        _column_keys(bt, bcols[j], kind, bmul, List[Int32](), False, bv, bo)
        _column_keys(pt, pcols[j], kind, pmul, translate, kind == TEXT.value, pv, po)
        bvals.append(bv^)
        boks.append(bo^)
        pvals.append(pv^)
        poks.append(po^)
    var nb = bt.rows()
    var np = pt.rows()
    keys.bok = List[UInt8](length=nb, fill=1)
    keys.pok = List[UInt8](length=np, fill=1)
    for j in range(nk):
        for i in range(nb):
            if boks[j][i] == 0:
                keys.bok[i] = 0
        for i in range(np):
            if poks[j][i] == 0:
                keys.pok[i] = 0
    if nk == 1:
        keys.bkeys = bvals[0].copy()
        keys.pkeys = pvals[0].copy()
        keys.layout = "one key"
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    # several keys: packed into one int64 when the build side's ranges multiply to fewer than 2^62 values
    var los = List[Int64]()
    var spans = List[UInt64]()
    var fits = True
    var product = UInt64(1)
    for j in range(nk):
        var lo = Int64.MAX
        var hi = Int64.MIN
        for i in range(nb):
            if keys.bok[i] != 0:
                lo = min(lo, bvals[j][i])
                hi = max(hi, bvals[j][i])
        if lo > hi:
            lo = 0
            hi = 0
        var span = UInt64(hi) - UInt64(lo) + 1
        los.append(lo)
        spans.append(span)
        if span == 0 or product > (UInt64(1) << 62) // span:
            fits = False
        else:
            product *= span
    keys.bkeys = List[Int64](length=nb, fill=0)
    keys.pkeys = List[Int64](length=np, fill=0)
    if fits:
        keys.layout = String(nk) + " keys packed into one int64 (" + String(product) + " combinations)"
        var stride = UInt64(1)
        for j in range(nk):
            for i in range(nb):
                keys.bkeys[i] = Int64(UInt64(keys.bkeys[i]) + UInt64(bvals[j][i] - los[j]) * stride)
            for i in range(np):
                if keys.pok[i] != 0 and (pvals[j][i] < los[j] or UInt64(pvals[j][i] - los[j]) >= spans[j]):
                    keys.pok[i] = 0                                  # outside the build side's range: it cannot match
                else:
                    keys.pkeys[i] = Int64(UInt64(keys.pkeys[i]) + UInt64(pvals[j][i] - los[j]) * stride)
            stride *= spans[j]
        keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
        keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
        keys.bo_addr = Int(keys.bok.unsafe_ptr())
        keys.po_addr = Int(keys.pok.unsafe_ptr())
        return
    keys.verify = True
    keys.layout = String(nk) + " keys hashed to one int64 (their ranges are too wide to pack), pairs verified against the columns"
    for i in range(nb):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(bvals[j][i]))
        keys.bkeys[i] = Int64(h)
    for i in range(np):
        var h = UInt64(0x9E3779B97F4A7C15)
        for j in range(nk):
            h = _mix(h ^ UInt64(pvals[j][i]))
        keys.pkeys[i] = Int64(h)
    keys.bcomp = bvals^
    keys.pcomp = pvals^
    keys.bk_addr = Int(keys.bkeys.unsafe_ptr())
    keys.pk_addr = Int(keys.pkeys.unsafe_ptr())
    keys.bo_addr = Int(keys.bok.unsafe_ptr())
    keys.po_addr = Int(keys.pok.unsafe_ptr())


# ── the pairs ────────────────────────────────────────────────────────────────────────────────────────

struct JoinPairs(Movable):
    """The matching (probe row, build row) pairs, in probe-row order: two Int32 arrays the kernel allocated (freed with the pairs)."""
    var n: Int
    var probe_a: Int
    var build_a: Int

    def __init__(out self):
        self.n = 0
        self.probe_a = 0
        self.build_a = 0

    def __del__(deinit self):
        hj_release(self.probe_a)
        hj_release(self.build_a)


def join_pairs(m: JoinMap, keys: JoinKeys, n_probe: Int, budget_rows: Int, threads: Int) raises -> JoinPairs:
    var out = JoinPairs()
    if n_probe == 0:
        return out^
    var info = List[Int64](length=4, fill=0)
    var rc = hj_probe_pairs(m, keys.pk_addr, keys.po_addr, n_probe, threads, budget_rows, Int(info.unsafe_ptr()))
    if rc == E_BUDGET:
        raise Error("the join produces " + commas(Int(info[0])) + " rows, past the budget of " + commas(budget_rows) + " (the joined columns would take over " +
                    String(JOIN_BUDGET_BYTES >> 30) + " GiB): filter either side first, or join on a unique key (a join that spills to disk is not built yet)")
    if rc != 0:
        raise Error("the hash join failed with code " + String(rc))
    var n = Int(info[0])
    var pr = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[1]))
    var br = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(info[2]))
    if keys.verify:                                                  # hashed tuples: keep the pairs whose key columns are really equal
        var kept = 0
        for i in range(n):
            var same = True
            for j in range(len(keys.bcomp)):
                if keys.bcomp[j][Int(br[i])] != keys.pcomp[j][Int(pr[i])]:
                    same = False
                    break
            if same:
                pr[kept] = pr[i]
                br[kept] = br[i]
                kept += 1
        n = kept
    out.n = n
    out.probe_a = Int(info[1])
    out.build_a = Int(info[2])
    return out^


struct JoinSide(Movable):
    """One side of the join: its table, the table's columns the joined rows carry (the needed ones, in order) and the table's column of each
    join key. A side with no filter over a plain table is the source table itself (nothing is copied); any other side is the result of a row
    query over its source."""
    var table: SharedTable
    var cols: List[Int]
    var key_cols: List[Int]
    var zero_copy: Bool

    def __init__(out self, table: SharedTable, var cols: List[Int], var key_cols: List[Int], zero_copy: Bool):
        self.table = table
        self.cols = cols^
        self.key_cols = key_cols^
        self.zero_copy = zero_copy


def _gather_column(src: Int, valid: Int, rows_a: Int, n: Int, dst_a: Int, nul_a: Int, threads: Int):
    """dst[i] = src[rows[i]] (and the NULL byte from the validity bits), over slices of the rows in parallel."""
    var tasks = max(1, min(threads if threads > 0 else core_count(), n // 32768 + 1))

    def work(t: Int) {imm}:
        var s = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=src)
        var rows = Pointer[Int32, MutUntrackedOrigin](unsafe_from_address=rows_a)
        var dst = Pointer[Int64, MutUntrackedOrigin](unsafe_from_address=dst_a)
        var a = n * t // tasks
        var b = n * (t + 1) // tasks
        for i in range(a, b):
            dst[i] = s[Int(rows[i])]
        if nul_a != 0 and valid != 0:
            var nd = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=nul_a)
            var bits = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=valid)
            for i in range(a, b):
                var r = Int(rows[i])
                nd[i] = 1 - ((bits[r >> 3] >> UInt8(r & 7)) & 1)

    par_n(work, tasks, tasks)


def gather_join(left: JoinSide, right: JoinSide, pairs: JoinPairs, left_builds: Bool, threads: Int) raises -> Table:
    """The joined table: the carried columns of the left side at its rows of the pairs, then the right side's (the build side's rows are the
    pairs' build rows, the probe side's the pairs' probe rows)."""
    var n = pairs.n
    var t = Table.empty()
    for side in range(2):
        ref sd = left if side == 0 else right
        var rows_a = (pairs.build_a if left_builds else pairs.probe_a) if side == 0 else (pairs.probe_a if left_builds else pairs.build_a)
        for c in sd.cols:
            var kind = sd.table.kind(c)
            var col = t.add_column(_unique_name(t, sd.table.name(c)), kind, scale=sd.table.scale(c))
            var vals = List[Int64](unsafe_uninit_length=n)
            var has_nulls = sd.table.null_count(c) > 0
            var nul = List[UInt8](length=n if has_nulls else 0, fill=0)
            _gather_column(Int(sd.table.values(c).unsafe_ptr()), _valid_bits(sd.table, c), rows_a, n, Int(vals.unsafe_ptr()),
                           Int(nul.unsafe_ptr()) if has_nulls else 0, threads)
            if kind == TEXT:                                          # the side's whole dictionary, codes unchanged
                var words = sd.table.dictionary(c)
                for w in range(len(words)):
                    _ = t.add_word(col, String(words[w]))
            _fill_column_owned(t.cols[col], vals^, nul^)
    return t^


def _fill_column_owned(mut col: Column, var vals: List[Int64], var nul: List[UInt8]):
    col.values = vals^
    if len(nul) > 0:
        var n = len(col.values)
        col.validity = List[UInt8](length=(n + 7) // 8, fill=0xFF)
        var nulls = 0
        for i in range(n):
            if nul[i] != 0:
                col.validity[i // 8] &= ~(UInt8(1) << UInt8(i % 8))
                nulls += 1
        col.nulls = nulls
        if nulls == 0:
            col.validity = List[UInt8]()


