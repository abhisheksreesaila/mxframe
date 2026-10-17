# F-131: zstd and snappy (C libraries) called from Mojo; a page round-trips.
from std.ffi import external_call
from std.memory.alloc import unsafe_alloc


def main() raises:
    var n = 1 << 20
    var src = unsafe_alloc[UInt8](n)
    for i in range(n):
        src[i] = UInt8((i * 31 + i // 1000) % 251)
    # zstd
    var cap = external_call["ZSTD_compressBound", Int](n)
    var comp = unsafe_alloc[UInt8](cap)
    var csize = external_call["ZSTD_compress", Int](comp, cap, src, n, Int32(3))
    var back = unsafe_alloc[UInt8](n)
    var dsize = external_call["ZSTD_decompress", Int](back, n, comp, csize)
    var same = dsize == n
    for i in range(n):
        if back[i] != src[i]:
            same = False
            break
    print("zstd", n, "->", csize, "->", dsize, "same", same)
    # snappy (C API: 0 = OK)
    var scap = external_call["snappy_max_compressed_length", Int](n)
    var scomp = unsafe_alloc[UInt8](scap)
    var slen = scap
    var rc = external_call["snappy_compress", Int32](src, n, scomp, UnsafePointer(to=slen))
    var sback = unsafe_alloc[UInt8](n)
    var blen = n
    var rc2 = external_call["snappy_uncompress", Int32](scomp, slen, sback, UnsafePointer(to=blen))
    var same2 = rc == 0 and rc2 == 0 and blen == n
    for i in range(n):
        if sback[i] != src[i]:
            same2 = False
            break
    print("snappy", n, "->", slen, "->", blen, "same", same2)
    if not (same and same2):
        raise Error("codec round trip failed")

