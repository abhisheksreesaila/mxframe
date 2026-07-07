from std.gpu.host import DeviceContext
from math import ceildiv
from gpu import thread_idx, block_idx, block_dim
from os.atomic import Atomic

comptime BLOCK: Int = 256
comptime COARSE: Int = 4

fn _f32_dev(addr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)

fn _i32_dev(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)

@export
fn group_sum_f32_gpu_spike(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
):
    var out = _f32_dev(out_addr)
    var val = _f32_dev(val_addr)
    var lab = _i32_dev(lab_addr)

    @parameter
    fn zero_out(ng: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < ng:
            out[tid] = 0.0

    @parameter
    fn sum_global(n: Int, ng: Int):
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(lab[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.fetch_add(out + gid, val[i])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](
            n_groups, grid_dim=ceildiv(n_groups, BLOCK), block_dim=BLOCK
        )
        if n_rows > 0:
            ctx.enqueue_function_experimental[sum_global](
                n_rows, n_groups,
                grid_dim=ceildiv(n_rows, BLOCK * COARSE),
                block_dim=BLOCK,
            )
        ctx.synchronize()
    except:
        pass
