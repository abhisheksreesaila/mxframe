# F-131: the native kernels run from the package, with no Python: an exact int64 SUM and COUNT under a filter,
# and the overflow flag when a total passes int64.
from std.memory.alloc import unsafe_alloc
from omxcore.kernels.pipeline_cpu import pipeline_plain, E_OVERFLOW


def sum_where_ge3(w: Int, v: Int, n: Int) raises -> Tuple[Int, Int64, Int64]:
    # SUM(v) WHERE w >= 3: CMPC r0 = col0 >= consts[0]; LOADI r0 = col1; AGG_ISUM slot0 += r0
    var code: List[Int32] = [0x4D585031, 1, 0, 1, 1, 1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0,
                             38, 0, 0, 0, 23, 3, 0, 1, 0, 0, 53, 0, 0, 0, 0]
    var consts: List[Float64] = [3.0]
    var set_bytes: List[UInt8] = [0]
    var set_meta: List[Int64] = [0, 0, 0]
    var tbl: List[Int64] = [0, Int64(n), Int64(w), Int64(v)]
    var dtypes: List[Int32] = [1, 1]
    var out_vals: List[Float64] = [0.0]
    var out_cnt: List[Int64] = [0]
    var rc = pipeline_plain(Int(code.unsafe_ptr()), Int(consts.unsafe_ptr()), Int(set_bytes.unsafe_ptr()),
                            Int(set_meta.unsafe_ptr()), Int(tbl.unsafe_ptr()), 1, Int(dtypes.unsafe_ptr()),
                            0, 1, 0, n, Int(out_vals.unsafe_ptr()), Int(out_cnt.unsafe_ptr()), 8, False)
    _ = code^, consts^, set_bytes^, set_meta^, tbl^, dtypes^      # Mojo frees at last use: keep them past the call
    return (rc, out_vals.unsafe_ptr().bitcast[Int64]()[0], out_cnt[0])


def main() raises:
    var n = 2_000_000
    var w = unsafe_alloc[Int64](n)
    var v = unsafe_alloc[Int64](n)
    var want_sum: Int64 = 0
    var want_cnt: Int64 = 0
    for i in range(n):
        w[i] = Int64(i % 7)
        # past 2^53 and cancelling: the exact total keeps every +1, a float64 sum would drop them
        v[i] = Int64((1 << 53) + 1) if i % 2 == 0 else Int64(-(1 << 53))
        if w[i] >= 3:
            want_sum += v[i]
            want_cnt += 1
    var r = sum_where_ge3(Int(w), Int(v), n)
    if r[0] != 0 or r[1] != want_sum or r[2] != want_cnt:
        raise Error("got rc " + String(r[0]) + " sum " + String(r[1]) + " count " + String(r[2]) +
                    ", want sum " + String(want_sum) + " count " + String(want_cnt))
    for i in range(n):
        v[i] = Int64(1 << 62)                                     # the total passes int64
    if sum_where_ge3(Int(w), Int(v), n)[0] != E_OVERFLOW:         # never a wrapped total
        raise Error("an int64 overflow was not flagged")
