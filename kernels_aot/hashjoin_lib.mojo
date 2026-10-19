"""
F-153: the hash join kernel as a shared library (`libmxhashjoin_cpu.so`), C ABI over plain addresses, for the Python engine and any other
caller that is not Mojo. The kernel itself is `hashjoin_cpu.mojo`; this file only boxes a `JoinMap` behind an integer handle.

    h = mx_hj_build(keys, ok, n, threads)              keys Int64[n], ok UInt8[n] or 0; a handle (> 0), or a negative error
    mx_hj_build_presence(keys, ok, n, threads)         the same, for a map only mx_hj_probe_select will read (a semi or anti join): cheaper to build
    mx_hj_info(h, out)                                 out Int64[4]: mode (0 empty, 1 direct array, 2 hash table), usable rows, 1 when keys are unique, bytes held
    mx_hj_probe_unique(h, keys, ok, n, out, threads)   unique-key maps: out Int32[n] gets each probe row's build row or -1
    mx_hj_probe_pairs(h, keys, ok, n, threads, budget_rows, out)
                                                       out Int64[3]: pair count, address of Int32 probe rows, address of Int32 build rows
                                                       (release each with mx_hj_release_array; both are 0 when the count is 0); -32 when the pairs pass the budget
    mx_hj_probe_pairs_outer(h, keys, ok, n, threads, budget_rows, out)
                                                       as mx_hj_probe_pairs, and a probe row with no partner (or no usable key) is one pair whose build row is -1
    mx_hj_probe_select(h, keys, ok, n, threads, mode, out)
                                                       mode 2 semi (rows whose key is present), 3 anti (rows whose key is absent, a NULL key too), 4 null-aware
                                                       anti (absent and not NULL), 5 none; out Int64[2]: how many, address of the Int32 rows (0 when none;
                                                       release with mx_hj_release_array); the map may hold repeated keys
    mx_hj_release_array(addr), mx_hj_free(h)

A handle has one owner: `mx_hj_free` once, after the last probe (a second free, or a probe after the free, is the caller's error). A handle that is 0 or negative
(`mx_hj_build`'s error code passed on by mistake) is refused with -33 by the other calls and ignored by `mx_hj_free`.
"""
from std.memory.alloc import unsafe_alloc
from std.memory import Pointer
from std.ffi import external_call
from hashjoin_cpu import JoinMap, hj_build, hj_probe_unique, hj_probe_pairs, hj_probe_select, hj_release

comptime ABI_VERSION = 1
comptime LP = Pointer[Int64, MutUntrackedOrigin]
comptime HEADER = 16
comptime E_BAD_HANDLE = -33          # a handle that is not a map (0, or an error code from mx_hj_build)


def _box(var m: JoinMap) -> Int:
    var h = LP(unsafe_from_address=Int(unsafe_alloc[Int64](HEADER)))
    h[0] = Int64(m.mode)
    h[1] = Int64(m.n)
    h[2] = Int64(m.nvalid)
    h[3] = m.lo
    h[4] = m.hi
    h[5] = Int64(m.span)
    h[6] = 1 if m.unique else 0
    h[15] = 1 if m.presence else 0
    h[7] = Int64(m.pbits)
    h[8] = Int64(m.u)
    h[9] = Int64(m.s)
    h[10] = Int64(m.r)
    h[11] = Int64(m.ent)
    h[12] = Int64(m.eoff)
    h[13] = Int64(m.bitmap)
    h[14] = Int64(m.bytes)
    _detach(m)
    return Int(h)


def _detach(mut m: JoinMap):
    """The map's memory now belongs to the box: the map must not free it."""
    m.u = 0
    m.s = 0
    m.r = 0
    m.ent = 0
    m.eoff = 0
    m.bitmap = 0


def _unbox(handle: Int) -> JoinMap:
    var h = LP(unsafe_from_address=handle)
    var m = JoinMap()
    m.mode = Int(h[0])
    m.n = Int(h[1])
    m.nvalid = Int(h[2])
    m.lo = h[3]
    m.hi = h[4]
    m.span = Int(h[5])
    m.unique = h[6] != 0
    m.presence = h[15] != 0
    m.pbits = Int(h[7])
    m.u = Int(h[8])
    m.s = Int(h[9])
    m.r = Int(h[10])
    m.ent = Int(h[11])
    m.eoff = Int(h[12])
    m.bitmap = Int(h[13])
    m.bytes = Int(h[14])
    return m^


@export
def mx_hj_abi_version() abi("C") -> Int:
    return ABI_VERSION


@export
def mx_hj_init_runtime() abi("C") -> Int:
    """Pin the async runtime for the life of the process (see kernels_aot.mojo mx_init_runtime)."""
    return external_call["KGEN_CompilerRT_AsyncRT_GetOrCreateCPUDevice", Int]()


@export
def mx_hj_build(keys_a: Int, ok_a: Int, n: Int, threads: Int) abi("C") -> Int:
    if n < 0 or n >= (1 << 31):
        return -31
    return _box(hj_build(keys_a, ok_a, n, threads))


@export
def mx_hj_build_presence(keys_a: Int, ok_a: Int, n: Int, threads: Int) abi("C") -> Int:
    """A map for `mx_hj_probe_select` only (semi and anti joins): cheaper to build, its probes by build row are meaningless."""
    if n < 0 or n >= (1 << 31):
        return -31
    return _box(hj_build(keys_a, ok_a, n, threads, True))


@export
def mx_hj_info(handle: Int, out_a: Int) abi("C") -> Int:
    if handle <= 0:
        return E_BAD_HANDLE
    var h = LP(unsafe_from_address=handle)
    var out = LP(unsafe_from_address=out_a)
    out[0] = h[0]
    out[1] = h[2]
    out[2] = h[6]
    out[3] = h[14]
    return 0


@export
def mx_hj_probe_unique(handle: Int, keys_a: Int, ok_a: Int, n: Int, out_a: Int, threads: Int) abi("C") -> Int:
    if handle <= 0:
        return E_BAD_HANDLE
    var m = _unbox(handle)
    var unique = m.unique
    if unique:
        hj_probe_unique(m, keys_a, ok_a, n, out_a, threads)
    _detach(m)
    return 0 if unique else -1


@export
def mx_hj_probe_pairs(handle: Int, keys_a: Int, ok_a: Int, n: Int, threads: Int, budget_rows: Int, out_a: Int) abi("C") -> Int:
    if handle <= 0:
        return E_BAD_HANDLE
    var m = _unbox(handle)
    var rc = hj_probe_pairs(m, keys_a, ok_a, n, threads, budget_rows, out_a)
    _detach(m)
    return rc


@export
def mx_hj_probe_pairs_outer(handle: Int, keys_a: Int, ok_a: Int, n: Int, threads: Int, budget_rows: Int, out_a: Int) abi("C") -> Int:
    if handle <= 0:
        return E_BAD_HANDLE
    var m = _unbox(handle)
    var rc = hj_probe_pairs(m, keys_a, ok_a, n, threads, budget_rows, out_a, True)
    _detach(m)
    return rc


@export
def mx_hj_probe_select(handle: Int, keys_a: Int, ok_a: Int, n: Int, threads: Int, mode: Int, out_a: Int) abi("C") -> Int:
    if handle <= 0:
        return E_BAD_HANDLE
    if mode < 2 or mode > 5:
        return -1
    var m = _unbox(handle)
    var rc = hj_probe_select(m, keys_a, ok_a, n, threads, mode, out_a)
    _detach(m)
    return rc


@export
def mx_hj_release_array(addr: Int) abi("C"):
    hj_release(addr)


@export
def mx_hj_free(handle: Int) abi("C"):
    if handle <= 0:                                  # like free(NULL): nothing to do
        return
    var m = _unbox(handle)      # the map frees its memory when it goes out of scope here
    LP(unsafe_from_address=handle).free()
