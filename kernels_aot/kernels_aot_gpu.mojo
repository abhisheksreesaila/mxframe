from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.os.atomic import Atomic

comptime BLOCK: Int = 256
comptime COARSE: Int = 4
comptime MAX_GROUPS: Int = 8192
comptime F32_MAX: Float32 = 3.4028235e38
comptime F32_MIN: Float32 = -3.4028235e38

@always_inline
fn _f32(addr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
fn _i32(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
fn _i64(addr: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=addr)

# ── group_sum ──────────────────────────────────────────────────────────────
@export
fn group_sum_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
):
    @parameter
    fn zero_out(ng: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < ng: _f32(out_addr)[tid] = 0.0

    @parameter
    fn sum_shared(n: Int, ng: Int):
        var shmem = stack_allocation[MAX_GROUPS, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var t = Int(thread_idx.x)
        while t < MAX_GROUPS:
            shmem[t] = 0.0; t += BLOCK
        barrier()
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(_i32(lab_addr)[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.fetch_add(shmem + gid, _f32(val_addr)[i])
        barrier()
        t = Int(thread_idx.x)
        while t < ng:
            var sv = shmem[t]
            if sv != 0.0: _ = Atomic.fetch_add(_f32(out_addr) + t, sv)
            t += BLOCK

    @parameter
    fn sum_global(n: Int, ng: Int):
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(_i32(lab_addr)[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.fetch_add(_f32(out_addr) + gid, _f32(val_addr)[i])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](
            n_groups, grid_dim=ceildiv(n_groups, BLOCK), block_dim=BLOCK)
        if n_rows > 0:
            var nb = ceildiv(n_rows, BLOCK * COARSE)
            if n_groups <= MAX_GROUPS:
                ctx.enqueue_function_experimental[sum_shared](
                    n_rows, n_groups, grid_dim=nb, block_dim=BLOCK)
            else:
                ctx.enqueue_function_experimental[sum_global](
                    n_rows, n_groups, grid_dim=nb, block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── group_min ──────────────────────────────────────────────────────────────
@export
fn group_min_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
):
    @parameter
    fn init_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = F32_MAX

    @parameter
    fn min_kernel(n: Int, ng: Int):
        var shmem = stack_allocation[MAX_GROUPS, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var t = Int(thread_idx.x)
        while t < MAX_GROUPS:
            shmem[t] = F32_MAX; t += BLOCK
        barrier()
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(_i32(lab_addr)[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.min(shmem + gid, _f32(val_addr)[i])
        barrier()
        t = Int(thread_idx.x)
        while t < ng:
            _ = Atomic.min(_f32(out_addr) + t, shmem[t])
            t += BLOCK

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_out](
            n_groups, grid_dim=ceildiv(n_groups, BLOCK), block_dim=BLOCK)
        if n_rows > 0:
            ctx.enqueue_function_experimental[min_kernel](
                n_rows, n_groups,
                grid_dim=ceildiv(n_rows, BLOCK * COARSE), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── group_max ──────────────────────────────────────────────────────────────
@export
fn group_max_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
):
    @parameter
    fn init_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = F32_MIN

    @parameter
    fn max_kernel(n: Int, ng: Int):
        var shmem = stack_allocation[MAX_GROUPS, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var t = Int(thread_idx.x)
        while t < MAX_GROUPS:
            shmem[t] = F32_MIN; t += BLOCK
        barrier()
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(_i32(lab_addr)[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.max(shmem + gid, _f32(val_addr)[i])
        barrier()
        t = Int(thread_idx.x)
        while t < ng:
            _ = Atomic.max(_f32(out_addr) + t, shmem[t])
            t += BLOCK

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_out](
            n_groups, grid_dim=ceildiv(n_groups, BLOCK), block_dim=BLOCK)
        if n_rows > 0:
            ctx.enqueue_function_experimental[max_kernel](
                n_rows, n_groups,
                grid_dim=ceildiv(n_rows, BLOCK * COARSE), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── group_count ────────────────────────────────────────────────────────────
@export
fn group_count_f32_gpu(
    out_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
):
    @parameter
    fn zero_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = 0.0

    @parameter
    fn count_kernel(n: Int, ng: Int):
        var base = Int(block_idx.x) * BLOCK * COARSE + Int(thread_idx.x)
        for c in range(COARSE):
            var i = base + c * BLOCK
            if i < n:
                var gid = Int(_i32(lab_addr)[i])
                if gid >= 0 and gid < ng:
                    _ = Atomic.fetch_add(_f32(out_addr) + gid, Float32(1.0))

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](
            n_groups, grid_dim=ceildiv(n_groups, BLOCK), block_dim=BLOCK)
        if n_rows > 0:
            ctx.enqueue_function_experimental[count_kernel](
                n_rows, n_groups,
                grid_dim=ceildiv(n_rows, BLOCK * COARSE), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── masked_global_sum ──────────────────────────────────────────────────────
@export
fn masked_global_sum_f32_gpu(
    out_addr: Int, val_addr: Int, mask_addr: Int, n: Int
):
    @parameter
    fn zero_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = 0.0

    @parameter
    fn sum_kernel(n_: Int):
        var shmem = stack_allocation[BLOCK, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK + tid
        var v: Float32 = 0.0
        if gid < n_ and _i32(mask_addr)[gid] != 0: v = _f32(val_addr)[gid]
        shmem[tid] = v
        barrier()
        var s = BLOCK >> 1
        while s > 0:
            if tid < s: shmem[tid] += shmem[tid + s]
            barrier(); s = s >> 1
        if tid == 0: _ = Atomic.fetch_add(_f32(out_addr), shmem[0])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](
            1, grid_dim=1, block_dim=BLOCK)
        if n > 0:
            ctx.enqueue_function_experimental[sum_kernel](
                n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── masked_global_sum_product ──────────────────────────────────────────────
@export
fn masked_global_sum_product_f32_gpu(
    out_addr: Int, a_addr: Int, b_addr: Int, mask_addr: Int, n: Int
):
    @parameter
    fn zero_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = 0.0

    @parameter
    fn sum_kernel(n_: Int):
        var shmem = stack_allocation[BLOCK, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK + tid
        var v: Float32 = 0.0
        if gid < n_ and _i32(mask_addr)[gid] != 0:
            v = _f32(a_addr)[gid] * _f32(b_addr)[gid]
        shmem[tid] = v
        barrier()
        var s = BLOCK >> 1
        while s > 0:
            if tid < s: shmem[tid] += shmem[tid + s]
            barrier(); s = s >> 1
        if tid == 0: _ = Atomic.fetch_add(_f32(out_addr), shmem[0])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](
            1, grid_dim=1, block_dim=BLOCK)
        if n > 0:
            ctx.enqueue_function_experimental[sum_kernel](
                n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── gather ─────────────────────────────────────────────────────────────────
@export
fn gather_f32_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _f32(out_addr)[i] = _f32(src_addr)[Int(_i32(idx_addr)[i])]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
fn gather_i32_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _i32(out_addr)[i] = _i32(src_addr)[Int(_i32(idx_addr)[i])]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
fn gather_i64_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _i64(out_addr)[i] = _i64(src_addr)[Int(_i32(idx_addr)[i])]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── filter_gather ──────────────────────────────────────────────────────────
@export
fn filter_gather_f32_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_ and _i32(mask_addr)[i] != 0:
            _f32(out_addr)[Int(_i32(offsets_addr)[i])] = _f32(src_addr)[i]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
fn filter_gather_i32_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_ and _i32(mask_addr)[i] != 0:
            _i32(out_addr)[Int(_i32(offsets_addr)[i])] = _i32(src_addr)[i]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
fn filter_gather_i64_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_ and _i32(mask_addr)[i] != 0:
            _i64(out_addr)[Int(_i32(offsets_addr)[i])] = _i64(src_addr)[i]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── unique_mask ────────────────────────────────────────────────────────────
# Mark out[i]=1 where sorted keys[i] != keys[i-1] (first in run).
@export
fn unique_mask_gpu(out_addr: Int, keys_addr: Int, n: Int):
    @parameter
    fn kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_:
            if i == 0:
                _i32(out_addr)[0] = 1
            else:
                _i32(out_addr)[i] = Int32(1) if _i32(keys_addr)[i] != _i32(keys_addr)[i-1] else Int32(0)
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass
