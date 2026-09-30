from std.gpu.host import DeviceContext, DeviceAttribute
from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, grid_dim, barrier
from std.gpu.memory import AddressSpace, external_memory
from std.memory import stack_allocation
from _atomic import Atomic

comptime BLOCK: Int = 256
comptime COARSE: Int = 4
comptime MAX_GROUPS: Int = 8192
comptime F32_MAX: Float32 = 3.4028235e38
comptime F32_MIN: Float32 = -3.4028235e38

@always_inline
def _f32(addr: Int) -> UnsafePointer[Float32, MutAnyOrigin]:
    return UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
def _i32(addr: Int) -> UnsafePointer[Int32, MutAnyOrigin]:
    return UnsafePointer[Int32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
def _i64(addr: Int) -> UnsafePointer[Int64, MutAnyOrigin]:
    return UnsafePointer[Int64, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
def _u8(addr: Int) -> UnsafePointer[UInt8, MutAnyOrigin]:
    return UnsafePointer[UInt8, MutAnyOrigin](unsafe_from_address=addr)

# ── UTF-8 predicates ──────────────────────────────────────────────────────
# Arrow string layout: offsets[n+1] point into one contiguous UTF-8 byte
# buffer. One GPU thread compares one row against the prefix bytes.
@export
def utf8_startswith_mask_gpu(
    out_addr: Int,
    offsets_addr: Int,
    data_addr: Int,
    valid_addr: Int,
    prefix_addr: Int,
    n: Int,
    prefix_len: Int,
) abi("C"):
    @parameter
    def kernel(n_: Int, prefix_len_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_:
            return
        if _u8(valid_addr)[row] == UInt8(0):
            _i32(out_addr)[row] = Int32(0)
            return
        var start = Int(_i32(offsets_addr)[row])
        var length = Int(_i32(offsets_addr)[row + 1]) - start
        if length < prefix_len_:
            _i32(out_addr)[row] = Int32(0)
            return
        var matches = True
        for byte_idx in range(prefix_len_):
            if _u8(data_addr)[start + byte_idx] != _u8(prefix_addr)[byte_idx]:
                matches = False
                break
        _i32(out_addr)[row] = Int32(1) if matches else Int32(0)

    try:
        var ctx = DeviceContext()
        if n > 0:
            ctx.enqueue_function_experimental[kernel](
                n, prefix_len, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
def utf8_contains_mask_gpu(
    out_addr: Int,
    offsets_addr: Int,
    data_addr: Int,
    valid_addr: Int,
    pattern_addr: Int,
    n: Int,
    pattern_len: Int,
) abi("C"):
    @parameter
    def kernel(n_: Int, pattern_len_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_:
            return
        if _u8(valid_addr)[row] == UInt8(0):
            _i32(out_addr)[row] = Int32(0)
            return
        var start = Int(_i32(offsets_addr)[row])
        var length = Int(_i32(offsets_addr)[row + 1]) - start
        var found = pattern_len_ == 0
        var candidate = 0
        while not found and candidate + pattern_len_ <= length:
            var matches = True
            for byte_idx in range(pattern_len_):
                if _u8(data_addr)[start + candidate + byte_idx] != _u8(pattern_addr)[byte_idx]:
                    matches = False
                    break
            if matches:
                found = True
            candidate += 1
        _i32(out_addr)[row] = Int32(1) if found else Int32(0)

    try:
        var ctx = DeviceContext()
        if n > 0:
            ctx.enqueue_function_experimental[kernel](
                n, pattern_len, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
def utf8_equal_mask_gpu(
    out_addr: Int,
    offsets_addr: Int,
    data_addr: Int,
    valid_addr: Int,
    pattern_addr: Int,
    n: Int,
    pattern_len: Int,
) abi("C"):
    @parameter
    def kernel(n_: Int, pattern_len_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_:
            return
        if _u8(valid_addr)[row] == UInt8(0):
            _i32(out_addr)[row] = Int32(0)
            return
        var start = Int(_i32(offsets_addr)[row])
        var length = Int(_i32(offsets_addr)[row + 1]) - start
        var matches = length == pattern_len_
        for byte_idx in range(pattern_len_):
            if not matches or _u8(data_addr)[start + byte_idx] != _u8(pattern_addr)[byte_idx]:
                matches = False
                break
        _i32(out_addr)[row] = Int32(1) if matches else Int32(0)

    try:
        var ctx = DeviceContext()
        if n > 0:
            ctx.enqueue_function_experimental[kernel](
                n, pattern_len, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
def utf8_isin_mask_gpu(
    out_addr: Int,
    offsets_addr: Int,
    data_addr: Int,
    valid_addr: Int,
    pattern_offsets_addr: Int,
    pattern_data_addr: Int,
    n: Int,
    n_patterns: Int,
) abi("C"):
    @parameter
    def kernel(n_: Int, n_patterns_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_:
            return
        if _u8(valid_addr)[row] == UInt8(0):
            _i32(out_addr)[row] = Int32(0)
            return
        var start = Int(_i32(offsets_addr)[row])
        var length = Int(_i32(offsets_addr)[row + 1]) - start
        var found = False
        for pattern_idx in range(n_patterns_):
            var pattern_start = Int(_i32(pattern_offsets_addr)[pattern_idx])
            var pattern_length = Int(_i32(pattern_offsets_addr)[pattern_idx + 1]) - pattern_start
            if pattern_length != length:
                continue
            var matches = True
            for byte_idx in range(pattern_length):
                if _u8(data_addr)[start + byte_idx] != _u8(pattern_data_addr)[pattern_start + byte_idx]:
                    matches = False
                    break
            if matches:
                found = True
                break
        _i32(out_addr)[row] = Int32(1) if found else Int32(0)

    try:
        var ctx = DeviceContext()
        if n > 0:
            ctx.enqueue_function_experimental[kernel](
                n, n_patterns, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── grouped f32 aggregation (sum / count / min / max) ───────────────────────
# One templated kernel family, two strategies picked on the host by group
# count.  Both are grid-stride loops over a grid sized to the GPU (a few
# resident blocks per SM), so a block covers many rows before it flushes.
# (The old design launched one 1024-row block per slice, zeroed all 8192
# shared slots in each, flushed every slot with a global atomic, and faulted
# for > 8192 groups.)
#
#   SHARED ng <= 12288  block-private table in *dynamic* shared memory sized to
#                       exactly ng slots, zeroed / flushed once per block
#   GLOBAL otherwise    atomics straight into the output (float min/max are the
#                       LLVM atomicrmw fmin/fmax, a CAS loop on NVPTX)
#
# The big win is in _atomic.mojo: atomics are `monotonic` (CUDA atomicAdd
# semantics).  The old seq_cst ordering made NVPTX emit a fence.sc.gpu around
# every atomic (~20x slower).  A register-accumulator path for <= 32 groups
# with warp-shuffle reduction was tried and was slower than shared atomics for
# float sums, so it was dropped.
#
# Rows with a label outside [0, ng) are ignored.
comptime OP_SUM: Int = 0
comptime OP_COUNT: Int = 1
comptime OP_MIN: Int = 2
comptime OP_MAX: Int = 3
comptime SHARED_MAX_GROUPS: Int = 12288       # 48 KB of Float32 slots
comptime SM_SHARED_BYTES: Int = 96 * 1024     # usable per SM on sm_80/86
comptime MAX_BLOCKS_PER_SM: Int = 6           # 1536 threads / 256
comptime GA_UNROLL: Int = 4

@always_inline
def _ident[op: Int]() -> Float32:
    comptime if op == OP_MIN:
        return F32_MAX
    elif op == OP_MAX:
        return F32_MIN
    else:
        return 0.0

@always_inline
def _fold[op: Int](a: Float32, b: Float32) -> Float32:
    comptime if op == OP_MIN:
        return b if b < a else a
    elif op == OP_MAX:
        return b if b > a else a
    else:
        return a + b

@always_inline
def _atomic_fold[op: Int, space: AddressSpace](
    ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space], v: Float32
):
    comptime if op == OP_MIN:
        _ = Atomic.min[space](ptr, v)
    elif op == OP_MAX:
        _ = Atomic.max[space](ptr, v)
    else:
        _ = Atomic.fetch_add[space](ptr, v)

@always_inline
def _row_value[op: Int](val_addr: Int, i: Int) -> Float32:
    comptime if op == OP_COUNT:
        return 1.0
    else:
        return _f32(val_addr)[i]

def group_init_kernel[op: Int](out_addr: Int, ng: Int):
    var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    if t < ng:
        _f32(out_addr)[t] = _ident[op]()

def group_global_kernel[op: Int](
    out_addr: Int, val_addr: Int, lab_addr: Int, n: Int, ng: Int, stride: Int
):
    var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
    while i + (GA_UNROLL - 1) * stride < n:
        var g = InlineArray[Int, GA_UNROLL](fill=0)
        var v = InlineArray[Float32, GA_UNROLL](fill=0.0)
        comptime for u in range(GA_UNROLL):
            g[u] = Int(_i32(lab_addr)[i + u * stride])
            v[u] = _row_value[op](val_addr, i + u * stride)
        comptime for u in range(GA_UNROLL):
            if g[u] >= 0 and g[u] < ng:
                _atomic_fold[op, AddressSpace.GENERIC](_f32(out_addr) + g[u], v[u])
        i += GA_UNROLL * stride
    while i < n:
        var gid = Int(_i32(lab_addr)[i])
        if gid >= 0 and gid < ng:
            _atomic_fold[op, AddressSpace.GENERIC](
                _f32(out_addr) + gid, _row_value[op](val_addr, i))
        i += stride

def group_shared_kernel[op: Int](
    out_addr: Int, val_addr: Int, lab_addr: Int, n: Int, ng: Int, stride: Int
):
    var shmem = rebind[
        UnsafePointer[Scalar[DType.float32], MutAnyOrigin,
                      address_space=AddressSpace.SHARED]
    ](external_memory[
        Scalar[DType.float32],
        address_space=AddressSpace.SHARED,
        alignment=16,
        name="shared_dynamic_memory",
    ]())
    var tid = Int(thread_idx.x)
    var t = tid
    while t < ng:
        shmem[t] = _ident[op]()
        t += BLOCK
    barrier()
    var i = Int(block_idx.x) * BLOCK + tid
    while i + (GA_UNROLL - 1) * stride < n:
        var g = InlineArray[Int, GA_UNROLL](fill=0)
        var v = InlineArray[Float32, GA_UNROLL](fill=0.0)
        comptime for u in range(GA_UNROLL):
            g[u] = Int(_i32(lab_addr)[i + u * stride])
            v[u] = _row_value[op](val_addr, i + u * stride)
        comptime for u in range(GA_UNROLL):
            if g[u] >= 0 and g[u] < ng:
                _atomic_fold[op, AddressSpace.SHARED](shmem + g[u], v[u])
        i += GA_UNROLL * stride
    while i < n:
        var gid = Int(_i32(lab_addr)[i])
        if gid >= 0 and gid < ng:
            _atomic_fold[op, AddressSpace.SHARED](
                shmem + gid, _row_value[op](val_addr, i))
        i += stride
    barrier()
    t = tid
    while t < ng:
        var sv = shmem[t]
        if sv != _ident[op]():          # untouched slot: nothing to flush
            _atomic_fold[op, AddressSpace.GENERIC](_f32(out_addr) + t, sv)
        t += BLOCK

def run_group[op: Int](
    out_addr: Int, val_addr: Int, lab_addr: Int, n: Int, ng: Int
) raises:
    if ng <= 0:
        return
    var ctx = DeviceContext()
    ctx.enqueue_function_experimental[group_init_kernel[op]](
        out_addr, ng, grid_dim=ceildiv(ng, BLOCK), block_dim=BLOCK)
    if n > 0:
        var sms = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
        if ng <= SHARED_MAX_GROUPS and n >= 4 * ng:
            var smem = ng * 4
            var per_sm = max(1, min(MAX_BLOCKS_PER_SM, SM_SHARED_BYTES // (smem + 1024)))
            # each block must cover >= ~2 rows per slot it flushes
            var blocks = max(1, min(sms * per_sm, n // (2 * ng)))
            blocks = min(blocks, ceildiv(n, BLOCK * GA_UNROLL))
            ctx.enqueue_function_experimental[group_shared_kernel[op]](
                out_addr, val_addr, lab_addr, n, ng, blocks * BLOCK,
                grid_dim=blocks, block_dim=BLOCK, shared_mem_bytes=smem)
        else:
            var blocks = min(sms * MAX_BLOCKS_PER_SM, ceildiv(n, BLOCK * GA_UNROLL))
            ctx.enqueue_function_experimental[group_global_kernel[op]](
                out_addr, val_addr, lab_addr, n, ng, blocks * BLOCK,
                grid_dim=blocks, block_dim=BLOCK)
    ctx.synchronize()

@export
def group_sum_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    try: run_group[OP_SUM](out_addr, val_addr, lab_addr, n_rows, n_groups)
    except: pass

@export
def group_min_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    try: run_group[OP_MIN](out_addr, val_addr, lab_addr, n_rows, n_groups)
    except: pass

@export
def group_max_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    try: run_group[OP_MAX](out_addr, val_addr, lab_addr, n_rows, n_groups)
    except: pass

@export
def group_count_f32_gpu(
    out_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    try: run_group[OP_COUNT](out_addr, 0, lab_addr, n_rows, n_groups)
    except: pass

# ── masked_global_sum ──────────────────────────────────────────────────────
@export
def masked_global_sum_f32_gpu(
    out_addr: Int, val_addr: Int, mask_addr: Int, n: Int
) abi("C"):
    @parameter
    def zero_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = 0.0

    @parameter
    def sum_kernel(n_: Int):
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
def masked_global_sum_product_f32_gpu(
    out_addr: Int, a_addr: Int, b_addr: Int, mask_addr: Int, n: Int
) abi("C"):
    @parameter
    def zero_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = 0.0

    @parameter
    def sum_kernel(n_: Int):
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

# Fused range predicate + product reduction. Each thread evaluates five
# comparisons and contributes a[i] * b[i] only when every predicate passes.
@export
def range5_sum_product_f32_gpu(
    out_addr: Int,
    a_addr: Int,
    b_addr: Int,
    int_col_addr: Int,
    float_col_addr: Int,
    upper_col_addr: Int,
    n: Int,
    int_lo: Int,
    int_hi: Int,
    float_lo: Float32,
    float_hi: Float32,
    upper_hi: Float32,
) abi("C"):
    @parameter
    def zero_out():
        if Int(thread_idx.x) == 0:
            _f32(out_addr)[0] = 0.0

    @parameter
    def sum_kernel(n_: Int, ilo: Int, ihi: Int, flo: Float32, fhi: Float32, uhi: Float32):
        var shmem = stack_allocation[BLOCK, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var row = Int(block_idx.x) * BLOCK + tid
        var value: Float32 = 0.0
        if row < n_:
            var int_value = Int(_i32(int_col_addr)[row])
            var float_value = _f32(float_col_addr)[row]
            var upper_value = _f32(upper_col_addr)[row]
            if (int_value >= ilo and int_value < ihi and
                    float_value >= flo and float_value <= fhi and
                    upper_value < uhi):
                value = _f32(a_addr)[row] * _f32(b_addr)[row]
        shmem[tid] = value
        barrier()
        var stride = BLOCK >> 1
        while stride > 0:
            if tid < stride:
                shmem[tid] += shmem[tid + stride]
            barrier()
            stride = stride >> 1
        if tid == 0:
            _ = Atomic.fetch_add(_f32(out_addr), shmem[0])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[zero_out](grid_dim=1, block_dim=BLOCK)
        if n > 0:
            ctx.enqueue_function_experimental[sum_kernel](
                n, int_lo, int_hi, float_lo, float_hi, upper_hi,
                grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# Mark right-side keys satisfying right_a < right_b, then count left rows by
# group when their key is marked and left_lo <= left_range < left_hi.
@export
def semi_range_group_count_i32_gpu(
    out_addr: Int,
    marks_addr: Int,
    right_keys_addr: Int,
    right_a_addr: Int,
    right_b_addr: Int,
    left_keys_addr: Int,
    left_range_addr: Int,
    left_groups_addr: Int,
    n_right: Int,
    n_left: Int,
    table_size: Int,
    n_groups: Int,
    left_lo: Int,
    left_hi: Int,
) abi("C"):
    @parameter
    def init_buffers(table_size_: Int, n_groups_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < table_size_:
            _i32(marks_addr)[row] = Int32(0)
        if row < n_groups_:
            _i32(out_addr)[row] = Int32(0)

    @parameter
    def mark_right(n_right_: Int, table_size_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < n_right_ and _i32(right_a_addr)[row] < _i32(right_b_addr)[row]:
            var key = Int(_i32(right_keys_addr)[row])
            if key >= 0 and key < table_size_:
                _i32(marks_addr)[key] = Int32(1)

    @parameter
    def count_left(n_left_: Int, table_size_: Int, n_groups_: Int, lo: Int, hi: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_left_:
            return
        var range_value = Int(_i32(left_range_addr)[row])
        var key = Int(_i32(left_keys_addr)[row])
        var group = Int(_i32(left_groups_addr)[row])
        if (range_value >= lo and range_value < hi and key >= 0 and
                key < table_size_ and group >= 0 and group < n_groups_ and
                _i32(marks_addr)[key] != 0):
            _ = Atomic.fetch_add(_i32(out_addr) + group, Int32(1))

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_buffers](
            table_size, n_groups,
            grid_dim=ceildiv(max(table_size, n_groups), BLOCK), block_dim=BLOCK)
        if n_right > 0:
            ctx.enqueue_function_experimental[mark_right](
                n_right, table_size,
                grid_dim=ceildiv(n_right, BLOCK), block_dim=BLOCK)
        if n_left > 0:
            ctx.enqueue_function_experimental[count_left](
                n_left, table_size, n_groups, left_lo, left_hi,
                grid_dim=ceildiv(n_left, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# Count right-key occurrences, then emit one count for every left key. This is
# the compact form of a left join followed by COUNT(right_key) per left row.
@export
def key_match_counts_i32_gpu(
    out_addr: Int,
    counts_addr: Int,
    left_keys_addr: Int,
    right_keys_addr: Int,
    n_left: Int,
    n_right: Int,
    table_size: Int,
) abi("C"):
    @parameter
    def init_counts(table_size_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < table_size_:
            _i32(counts_addr)[row] = Int32(0)

    @parameter
    def count_right(n_right_: Int, table_size_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < n_right_:
            var key = Int(_i32(right_keys_addr)[row])
            if key >= 0 and key < table_size_:
                _ = Atomic.fetch_add(_i32(counts_addr) + key, Int32(1))

    @parameter
    def gather_left(n_left_: Int, table_size_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < n_left_:
            var key = Int(_i32(left_keys_addr)[row])
            _i32(out_addr)[row] = _i32(counts_addr)[key] if key >= 0 and key < table_size_ else Int32(0)

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_counts](
            table_size, grid_dim=ceildiv(table_size, BLOCK), block_dim=BLOCK)
        if n_right > 0:
            ctx.enqueue_function_experimental[count_right](
                n_right, table_size,
                grid_dim=ceildiv(n_right, BLOCK), block_dim=BLOCK)
        if n_left > 0:
            ctx.enqueue_function_experimental[gather_left](
                n_left, table_size,
                grid_dim=ceildiv(n_left, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# Exact distinct (order, supplier) counting for Q21. Hash slots use a state
# machine: 0=empty, -1=writer owns slot, 1=ready. Full key comparison resolves
# hash collisions without relying on fingerprints.
@export
def single_late_supplier_counts_i32_gpu(
    out_addr: Int,
    order_counts_addr: Int,
    states_addr: Int,
    hash_orders_addr: Int,
    hash_suppliers_addr: Int,
    order_keys_addr: Int,
    supplier_keys_addr: Int,
    commit_addr: Int,
    receipt_addr: Int,
    finished_orders_addr: Int,
    target_suppliers_addr: Int,
    n_rows: Int,
    n_orders: Int,
    n_suppliers: Int,
    cap: Int,
) abi("C"):
    @parameter
    def init_buffers(n_orders_: Int, n_suppliers_: Int, cap_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row < cap_:
            _i32(states_addr)[row] = Int32(0)
        if row < n_orders_:
            _i32(order_counts_addr)[row] = Int32(0)
        if row < n_suppliers_:
            _i32(out_addr)[row] = Int32(0)

    @parameter
    def insert_pairs(n_rows_: Int, n_orders_: Int, n_suppliers_: Int, cap_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_rows_ or _i32(receipt_addr)[row] <= _i32(commit_addr)[row]:
            return
        var order_key = _i32(order_keys_addr)[row]
        var supplier_key = _i32(supplier_keys_addr)[row]
        if order_key < 0 or Int(order_key) >= n_orders_ or supplier_key < 0 or Int(supplier_key) >= n_suppliers_:
            return
        var mixed = UInt32(Int(order_key)) * UInt32(0x9e3779b1)
        mixed ^= UInt32(Int(supplier_key)) * UInt32(0x85ebca6b)
        mixed ^= mixed >> 16
        var slot = Int(mixed) % cap_
        while True:
            # states / keys are written by other blocks while we read them: use L2
            # acquire loads and a release store (plain loads may hit a stale L1 line)
            var state = Atomic.load(_i32(states_addr) + slot)
            if state == Int32(0):
                var expected = Int32(0)
                if Atomic.compare_exchange(_i32(states_addr) + slot, expected, Int32(-1)):
                    _i32(hash_orders_addr)[slot] = order_key
                    _i32(hash_suppliers_addr)[slot] = supplier_key
                    Atomic.store(_i32(states_addr) + slot, Int32(1))
                    _ = Atomic.fetch_add(_i32(order_counts_addr) + Int(order_key), Int32(1))
                    return
                state = expected
            while state == Int32(-1):
                state = Atomic.load(_i32(states_addr) + slot)
            if (state == Int32(1) and Atomic.load(_i32(hash_orders_addr) + slot) == order_key and
                    Atomic.load(_i32(hash_suppliers_addr) + slot) == supplier_key):
                return
            slot = (slot + 1) % cap_

    @parameter
    def count_suppliers(n_rows_: Int, n_orders_: Int, n_suppliers_: Int):
        var row = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if row >= n_rows_ or _i32(receipt_addr)[row] <= _i32(commit_addr)[row]:
            return
        var order_key = Int(_i32(order_keys_addr)[row])
        var supplier_key = Int(_i32(supplier_keys_addr)[row])
        if (order_key >= 0 and order_key < n_orders_ and supplier_key >= 0 and
                supplier_key < n_suppliers_ and _i32(order_counts_addr)[order_key] == Int32(1) and
                _i32(finished_orders_addr)[order_key] != 0 and
                _i32(target_suppliers_addr)[supplier_key] != 0):
            _ = Atomic.fetch_add(_i32(out_addr) + supplier_key, Int32(1))

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_buffers](
            n_orders, n_suppliers, cap,
            grid_dim=ceildiv(max(max(n_orders, n_suppliers), cap), BLOCK), block_dim=BLOCK)
        if n_rows > 0:
            ctx.enqueue_function_experimental[insert_pairs](
                n_rows, n_orders, n_suppliers, cap,
                grid_dim=ceildiv(n_rows, BLOCK), block_dim=BLOCK)
            ctx.enqueue_function_experimental[count_suppliers](
                n_rows, n_orders, n_suppliers,
                grid_dim=ceildiv(n_rows, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── gather ─────────────────────────────────────────────────────────────────
@export
def gather_f32_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    def kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _f32(out_addr)[i] = _f32(src_addr)[Int(_i32(idx_addr)[i])]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
def gather_i32_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    def kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _i32(out_addr)[i] = _i32(src_addr)[Int(_i32(idx_addr)[i])]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

@export
def gather_i64_gpu(out_addr: Int, src_addr: Int, idx_addr: Int, n: Int):
    @parameter
    def kernel(n_: Int):
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
def filter_gather_f32_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
) abi("C"):
    @parameter
    def kernel(n_: Int):
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
def filter_gather_i32_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
) abi("C"):
    @parameter
    def kernel(n_: Int):
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
def filter_gather_i64_gpu(
    out_addr: Int, src_addr: Int, mask_addr: Int, offsets_addr: Int, n: Int
) abi("C"):
    @parameter
    def kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_ and _i32(mask_addr)[i] != 0:
            _i64(out_addr)[Int(_i32(offsets_addr)[i])] = _i64(src_addr)[i]
    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── masked_global_min ─────────────────────────────────────────────────────
@export
def masked_global_min_f32_gpu(
    out_addr: Int, val_addr: Int, mask_addr: Int, n: Int
) abi("C"):
    @parameter
    def init_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = F32_MAX

    @parameter
    def min_kernel(n_: Int):
        var shmem = stack_allocation[BLOCK, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK + tid
        var v: Float32 = F32_MAX
        if gid < n_ and _i32(mask_addr)[gid] != 0: v = _f32(val_addr)[gid]
        shmem[tid] = v
        barrier()
        var s = BLOCK >> 1
        while s > 0:
            if tid < s:
                if shmem[tid + s] < shmem[tid]: shmem[tid] = shmem[tid + s]
            barrier(); s = s >> 1
        if tid == 0: _ = Atomic.min(_f32(out_addr), shmem[0])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_out](
            1, grid_dim=1, block_dim=BLOCK)
        if n > 0:
            ctx.enqueue_function_experimental[min_kernel](
                n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── masked_global_max ─────────────────────────────────────────────────────
@export
def masked_global_max_f32_gpu(
    out_addr: Int, val_addr: Int, mask_addr: Int, n: Int
) abi("C"):
    @parameter
    def init_out(nout: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < nout: _f32(out_addr)[t] = F32_MIN

    @parameter
    def max_kernel(n_: Int):
        var shmem = stack_allocation[BLOCK, Scalar[DType.float32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var gid = Int(block_idx.x) * BLOCK + tid
        var v: Float32 = F32_MIN
        if gid < n_ and _i32(mask_addr)[gid] != 0: v = _f32(val_addr)[gid]
        shmem[tid] = v
        barrier()
        var s = BLOCK >> 1
        while s > 0:
            if tid < s:
                if shmem[tid + s] > shmem[tid]: shmem[tid] = shmem[tid + s]
            barrier(); s = s >> 1
        if tid == 0: _ = Atomic.max(_f32(out_addr), shmem[0])

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_out](
            1, grid_dim=1, block_dim=BLOCK)
        if n > 0:
            ctx.enqueue_function_experimental[max_kernel](
                n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── group_encode_i32 ──────────────────────────────────────────────────────
# Open-addressing linear-probing hash table: assign dense int32 group IDs to an
# int32 key array.  Every int32 value is a legal key (INT32_MIN included);
# slot state lives in htid, not in a key sentinel:
#
#   htid[h] == -1   empty
#   htid[h] == -2   claimed, key being written
#   htid[h] >= 0    ready: htab[h] holds the key, htid[h] its dense id
#
# Per thread i (key k):
#   h = hash(k) % cap
#   loop:
#     st = acquire_load(htid[h])                 # L2, never a stale L1 line
#     if st == EMPTY and CAS(htid[h], EMPTY -> CLAIMED) wins:
#         htab[h] = k; id = atomicAdd(out_ng, 1); release_store(htid[h], id)
#         out_ids[i] = id; return
#     while st == CLAIMED: st = acquire_load(htid[h])   # writer is 2 stores away
#     if acquire_load(htab[h]) == k: out_ids[i] = st; return
#     h = (h + 1) % cap                           # different key: probe on
#
# The previous version spun on plain loads of htid, which the SM may serve from
# L1 forever (the hang), and used INT32_MIN as the empty key so a real
# INT32_MIN key could win the same slot several times.
#
# Inputs:  keys[n] int32; cap > (number of distinct keys), e.g. >= 2*n
# Outputs: out_ids[n] dense group ids in claim order (not first-occurrence
#          order; the numbering can differ between runs), out_ng[1] n_groups
#          (ids are 0..n_groups-1).  out_ids[i] = -1 if the table was full.
# Scratch: htab[cap], htid[cap]  (the kernel initialises them)
@export
def group_encode_i32_gpu(
    keys_addr: Int,
    out_ids_addr: Int,
    out_ng_addr: Int,
    htab_addr: Int,
    htid_addr: Int,
    n: Int,
    cap: Int,
) abi("C"):
    @parameter
    def init_tables(cap_: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < cap_:
            _i32(htab_addr)[t] = Int32(0)
            _i32(htid_addr)[t] = Int32(-1)
        if t == 0:
            _i32(out_ng_addr)[0] = Int32(0)

    @parameter
    def insert(n_: Int, cap_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i >= n_: return

        var k = _i32(keys_addr)[i]

        # 32-bit avalanche hash of the key's bit pattern
        var kk = k.cast[DType.uint32]()
        kk ^= kk >> 16
        kk *= UInt32(0x7feb352d)
        kk ^= kk >> 15
        kk *= UInt32(0x846ca68b)
        kk ^= kk >> 16
        var h = Int(kk) % cap_

        var probes = 0
        while probes < cap_:
            var st = Atomic.load(_i32(htid_addr) + h)
            if st == Int32(-1):
                var expected = Int32(-1)
                if Atomic.compare_exchange(_i32(htid_addr) + h, expected, Int32(-2)):
                    _i32(htab_addr)[h] = k
                    var new_id = Atomic.fetch_add(_i32(out_ng_addr), Int32(1))
                    Atomic.store(_i32(htid_addr) + h, new_id)
                    _i32(out_ids_addr)[i] = new_id
                    return
                st = expected
            while st == Int32(-2):
                st = Atomic.load(_i32(htid_addr) + h)
            if Atomic.load(_i32(htab_addr) + h) == k:
                _i32(out_ids_addr)[i] = st
                return
            h += 1
            if h == cap_: h = 0
            probes += 1
        _i32(out_ids_addr)[i] = Int32(-1)

    try:
        var ctx = DeviceContext()
        if cap > 0:
            ctx.enqueue_function_experimental[init_tables](
                cap, grid_dim=ceildiv(cap, BLOCK), block_dim=BLOCK)
        if n > 0 and cap > 0:
            ctx.enqueue_function_experimental[insert](
                n, cap, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── unique_mask ────────────────────────────────────────────────────────────
# Mark out[i]=1 where sorted keys[i] != keys[i-1] (first in run).
@export
def unique_mask_gpu(out_addr: Int, keys_addr: Int, n: Int):
    @parameter
    def kernel(n_: Int):
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

# ── sort_topk_i32 ─────────────────────────────────────────────────────────
# Stable GPU sort of an int32 key array into a row-index permutation.
#
# Each row becomes one 64-bit word  (order-preserving key bits << 32) | row,
# so a plain 64-bit compare orders by key and breaks ties by row: the sort is
# stable, ascending or descending, and there is no indirect key load per
# compare.  The words are sorted with a bitonic network over a power-of-two
# scratch buffer; the padding words (rows >= n) carry the maximum key and sort
# to the end, so any n works.  Up to SORT_TILE (2048) words the whole sort is one
# shared-memory kernel; above that, tiles are sorted in shared memory and merged
# with global steps for distance >= SORT_TILE plus one shared-memory kernel for the
# remaining steps of each merge stage (about a third of the launches and memory
# passes of the plain network).  (The old version sorted the index array in place
# and skipped out-of-range partners, which is wrong for n that is not a power
# of two, and it was not stable.)
#
# out_addr  int32[n]   output: row indices in sorted order (all n)
# keys_addr int32[n]   input:  sort keys
# desc_flag            0 = ascending, 1 = descending (ties keep row order)
@always_inline
def _u64(addr: Int) -> UnsafePointer[UInt64, MutAnyOrigin]:
    return UnsafePointer[UInt64, MutAnyOrigin](unsafe_from_address=addr)

comptime SORT_TILE: Int = 2048        # words sorted in shared memory (16 KB)

@always_inline
def _sort_word(keys_addr: Int, i: Int, n: Int, desc: Int) -> UInt64:
    """(order-preserving key bits << 32) | row; rows >= n are +infinity padding."""
    var hi = UInt64(0xFFFFFFFF)
    if i < n:
        var k = _i32(keys_addr)[i]
        # flag bit 1: negative keys sort after every non-negative one (join build side)
        if not ((desc & 2) != 0 and k < Int32(0)):
            var bits = k.cast[DType.uint32]() ^ UInt32(0x80000000)
            if (desc & 1) != 0: bits = ~bits
            hi = UInt64(bits)
    return (hi << 32) | UInt64(i)

@always_inline
def _pair_index(p: Int, j: Int) -> Int:
    """Lower element of the p-th compare-exchange pair at distance j (a power of two)."""
    var lo = p & (j - 1)
    return ((p - lo) << 1) + lo

@always_inline
def _cmpx(
    sh: UnsafePointer[Scalar[DType.uint64], MutAnyOrigin, address_space=AddressSpace.SHARED],
    i: Int, j: Int, asc: Bool,
):
    var a = sh[i]
    var b = sh[i + j]
    if (asc and a > b) or (not asc and a < b):
        sh[i] = b
        sh[i + j] = a

@always_inline
def _sort_stages(
    sh: UnsafePointer[Scalar[DType.uint64], MutAnyOrigin, address_space=AddressSpace.SHARED],
    base: Int, tile: Int, tid: Int,
):
    """Bitonic stages k = 2..tile inside a shared-memory tile; the direction of each block
    comes from its *global* index, so tiles alternate ascending / descending like the
    global network."""
    var k = 2
    while k <= tile:
        var j = k >> 1
        while j > 0:
            var p = tid
            while p < (tile >> 1):
                var i = _pair_index(p, j)
                _cmpx(sh, i, j, ((base + i) & k) == 0)
                p += BLOCK
            barrier()
            j >>= 1
        k <<= 1

@always_inline
def _merge_steps(
    sh: UnsafePointer[Scalar[DType.uint64], MutAnyOrigin, address_space=AddressSpace.SHARED],
    base: Int, tile: Int, k_val: Int, tid: Int,
):
    """The steps j = tile/2..1 of one bitonic merge stage of size k_val >= 2 * tile."""
    var j = tile >> 1
    while j > 0:
        var p = tid
        while p < (tile >> 1):
            var i = _pair_index(p, j)
            _cmpx(sh, i, j, ((base + i) & k_val) == 0)
            p += BLOCK
        barrier()
        j >>= 1

# out_addr  int32[n]   output: row indices in sorted order (all n)
# keys_addr int32[n]   input:  sort keys
# desc_flag            bit 0: descending (ties keep row order);
#                      bit 1: negative keys go last, in row order (the hash join's
#                             build side, where a negative key means "no key")
@export
def sort_topk_i32_gpu(
    out_addr:  Int,
    keys_addr: Int,
    n:         Int,
    desc_flag: Int,
) abi("C"):
    @parameter
    def small_sort(n_: Int, padded_: Int, desc_: Int):
        # whole sort in one block: keys -> words -> shared sort -> rows
        var sh = stack_allocation[SORT_TILE, Scalar[DType.uint64],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var t = tid
        while t < padded_:
            sh[t] = _sort_word(keys_addr, t, n_, desc_)
            t += BLOCK
        barrier()
        _sort_stages(sh, 0, padded_, tid)
        t = tid
        while t < n_:
            _i32(out_addr)[t] = Int32(Int(sh[t] & UInt64(0xFFFFFFFF)))
            t += BLOCK

    @parameter
    def init_words(n_: Int, padded_: Int, desc_: Int, words_addr: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < padded_:
            _u64(words_addr)[i] = _sort_word(keys_addr, i, n_, desc_)

    @parameter
    def tile_sort(words_addr: Int):
        var sh = stack_allocation[SORT_TILE, Scalar[DType.uint64],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var base = Int(block_idx.x) * SORT_TILE
        var t = tid
        while t < SORT_TILE:
            sh[t] = _u64(words_addr)[base + t]
            t += BLOCK
        barrier()
        _sort_stages(sh, base, SORT_TILE, tid)
        t = tid
        while t < SORT_TILE:
            _u64(words_addr)[base + t] = sh[t]
            t += BLOCK

    @parameter
    def tile_merge(k_val: Int, words_addr: Int):
        var sh = stack_allocation[SORT_TILE, Scalar[DType.uint64],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var base = Int(block_idx.x) * SORT_TILE
        var t = tid
        while t < SORT_TILE:
            sh[t] = _u64(words_addr)[base + t]
            t += BLOCK
        barrier()
        _merge_steps(sh, base, SORT_TILE, k_val, tid)
        t = tid
        while t < SORT_TILE:
            _u64(words_addr)[base + t] = sh[t]
            t += BLOCK

    @parameter
    def global_step(padded_: Int, k_val: Int, j_val: Int, words_addr: Int):
        var p = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if p >= (padded_ >> 1): return
        var i = _pair_index(p, j_val)
        var a = _u64(words_addr)[i]
        var b = _u64(words_addr)[i + j_val]
        var asc = (i & k_val) == 0
        if (asc and a > b) or (not asc and a < b):
            _u64(words_addr)[i] = b
            _u64(words_addr)[i + j_val] = a

    @parameter
    def extract(n_: Int, words_addr: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_:
            _i32(out_addr)[i] = Int32(Int(_u64(words_addr)[i] & UInt64(0xFFFFFFFF)))

    try:
        if n > 0:
            var ctx = DeviceContext()
            var padded = 2
            while padded < n:
                padded *= 2
            if padded <= SORT_TILE:
                ctx.enqueue_function_experimental[small_sort](
                    n, padded, desc_flag, grid_dim=1, block_dim=BLOCK)
            else:
                var words = ctx.enqueue_create_buffer[DType.uint64](padded)
                var w = Int(words.unsafe_ptr())
                ctx.enqueue_function_experimental[init_words](
                    n, padded, desc_flag, w,
                    grid_dim=ceildiv(padded, BLOCK), block_dim=BLOCK)
                ctx.enqueue_function_experimental[tile_sort](
                    w, grid_dim=padded // SORT_TILE, block_dim=BLOCK)
                var k = 2 * SORT_TILE
                while k <= padded:
                    var j = k >> 1
                    while j >= SORT_TILE:
                        ctx.enqueue_function_experimental[global_step](
                            padded, k, j, w,
                            grid_dim=ceildiv(padded >> 1, BLOCK), block_dim=BLOCK)
                        j >>= 1
                    ctx.enqueue_function_experimental[tile_merge](
                        k, w, grid_dim=padded // SORT_TILE, block_dim=BLOCK)
                    k <<= 1
                ctx.enqueue_function_experimental[extract](
                    n, w, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
            ctx.synchronize()
    except: pass

# ── small device utilities used by the join / chaining wrappers ────────────
@export
def fill_i32_gpu(ptr_addr: Int, n: Int, value: Int) abi("C"):
    @parameter
    def kernel(n_: Int, v: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_: _i32(ptr_addr)[i] = Int32(v)
    try:
        if n > 0:
            var ctx = DeviceContext()
            ctx.enqueue_function_experimental[kernel](
                n, value, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
            ctx.synchronize()
    except: pass

# out[0] = max(out[0], max(in[0..n))): the caller seeds out[0].
@export
def max_i32_gpu(out_addr: Int, in_addr: Int, n: Int) abi("C"):
    @parameter
    def kernel(n_: Int):
        var shmem = stack_allocation[BLOCK, Scalar[DType.int32],
            address_space=AddressSpace.SHARED]()
        var tid = Int(thread_idx.x)
        var m = Int32(-2147483648)
        var i = Int(block_idx.x) * BLOCK + tid
        var stride = Int(grid_dim.x) * BLOCK
        while i < n_:
            var v = _i32(in_addr)[i]
            if v > m: m = v
            i += stride
        shmem[tid] = m
        barrier()
        var s = BLOCK >> 1
        while s > 0:
            if tid < s and shmem[tid + s] > shmem[tid]: shmem[tid] = shmem[tid + s]
            barrier(); s = s >> 1
        if tid == 0: _ = Atomic.max(_i32(out_addr), shmem[0])
    try:
        if n > 0:
            var ctx = DeviceContext()
            ctx.enqueue_function_experimental[kernel](
                n, grid_dim=min(ceildiv(n, BLOCK), 1024), block_dim=BLOCK)
            ctx.synchronize()
    except: pass

# out[i] = counts[i] > 0 ? sums[i] / counts[i] : 0
@export
def group_mean_finish_f32_gpu(out_addr: Int, sums_addr: Int, cnts_addr: Int, n: Int) abi("C"):
    @parameter
    def kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_:
            var c = _f32(cnts_addr)[i]
            _f32(out_addr)[i] = _f32(sums_addr)[i] / c if c > 0.0 else Float32(0.0)
    try:
        if n > 0:
            var ctx = DeviceContext()
            ctx.enqueue_function_experimental[kernel](
                n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
            ctx.synchronize()
    except: pass

# ── scan_i32: exclusive prefix sum ─────────────────────────────────────────
# out[i] = sum(in[0..i)),  total_addr[0] = sum(in[0..n)).
# Tiles of 1024 (256 threads x 4 items); the tile sums are scanned recursively.
comptime SCAN_TILE: Int = 1024

def _scan_tiles(
    ctx: DeviceContext, out_addr: Int, in_addr: Int, sums_addr: Int, n: Int
) raises:
    @parameter
    def kernel(n_: Int):
        var sh = stack_allocation[BLOCK, Scalar[DType.int32],
            address_space=AddressSpace.SHARED]()
        var t = Int(thread_idx.x)
        var base = Int(block_idx.x) * SCAN_TILE + t * 4
        var x0 = Int32(0); var x1 = Int32(0); var x2 = Int32(0); var x3 = Int32(0)
        if base + 0 < n_: x0 = _i32(in_addr)[base + 0]
        if base + 1 < n_: x1 = _i32(in_addr)[base + 1]
        if base + 2 < n_: x2 = _i32(in_addr)[base + 2]
        if base + 3 < n_: x3 = _i32(in_addr)[base + 3]
        var mine = x0 + x1 + x2 + x3
        sh[t] = mine
        barrier()
        var off = 1
        while off < BLOCK:
            var add = Int32(0)
            if t >= off: add = sh[t - off]
            barrier()
            sh[t] += add
            barrier()
            off <<= 1
        var e = sh[t] - mine
        if base + 0 < n_: _i32(out_addr)[base + 0] = e
        if base + 1 < n_: _i32(out_addr)[base + 1] = e + x0
        if base + 2 < n_: _i32(out_addr)[base + 2] = e + x0 + x1
        if base + 3 < n_: _i32(out_addr)[base + 3] = e + x0 + x1 + x2
        if t == BLOCK - 1: _i32(sums_addr)[Int(block_idx.x)] = sh[t]
    ctx.enqueue_function_experimental[kernel](
        n, grid_dim=ceildiv(n, SCAN_TILE), block_dim=BLOCK)

def _scan_add_offsets(ctx: DeviceContext, out_addr: Int, offs_addr: Int, n: Int) raises:
    @parameter
    def kernel(n_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_:
            _i32(out_addr)[i] += _i32(offs_addr)[i // SCAN_TILE]
    ctx.enqueue_function_experimental[kernel](
        n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)

def _scan_level(
    ctx: DeviceContext, out_addr: Int, in_addr: Int, total_addr: Int, n: Int
) raises:
    var nb = ceildiv(n, SCAN_TILE)
    if nb == 1:
        _scan_tiles(ctx, out_addr, in_addr, total_addr, n)
        return
    var sums = ctx.enqueue_create_buffer[DType.int32](nb)
    var offs = ctx.enqueue_create_buffer[DType.int32](nb)
    var sums_addr = Int(sums.unsafe_ptr())
    var offs_addr = Int(offs.unsafe_ptr())
    _scan_tiles(ctx, out_addr, in_addr, sums_addr, n)
    _scan_level(ctx, offs_addr, sums_addr, total_addr, nb)
    _scan_add_offsets(ctx, out_addr, offs_addr, n)

@export
def scan_i32_gpu(out_addr: Int, in_addr: Int, total_addr: Int, n: Int) abi("C"):
    try:
        if n > 0:
            var ctx = DeviceContext()
            _scan_level(ctx, out_addr, in_addr, total_addr, n)
            ctx.synchronize()
    except: pass

# ── join_count_i32 ─────────────────────────────────────────────────────────# Two-pass GPU hash join — Phase 1: count right-key matches per left row.
#
# Algorithm:
#  1. Atomically increment table[right_key[j]] for each j in 0..n_right.
#  2. For each i in 0..n_left: counts[i] = table[left_key[i]].
#
# Caller must pre-zero table_addr (size = table_size * 4 bytes).
@export
def join_count_i32_gpu(
    counts_addr: Int,   # out: int32[n_left]  per-row match counts
    table_addr:  Int,   # work: int32[table_size]  must be pre-zeroed
    left_addr:   Int,   # in: int32[n_left]
    right_addr:  Int,   # in: int32[n_right]
    n_left:      Int,
    n_right:     Int,
    table_size:  Int,
) abi("C"):
    @parameter
    def count_right(n_r: Int, ts: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < n_r:
            var k = Int(_i32(right_addr)[tid])
            if k >= 0 and k < ts:
                _ = Atomic.fetch_add(_i32(table_addr) + k, Int32(1))

    @parameter
    def probe_left(n_l: Int, ts: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < n_l:
            var k = Int(_i32(left_addr)[tid])
            _i32(counts_addr)[tid] = _i32(table_addr)[k] if k >= 0 and k < ts else Int32(0)

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[count_right](
            n_right, table_size,
            grid_dim=ceildiv(n_right, BLOCK), block_dim=BLOCK)
        ctx.enqueue_function_experimental[probe_left](
            n_left, table_size,
            grid_dim=ceildiv(n_left, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass

# ── join_scatter_i32 ───────────────────────────────────────────────────────
# Two-pass GPU hash join — Phase 2: scatter (left_idx, right_idx) pairs.
#
# Inputs (all device pointers):
#   left_addr       int32[n_left]       left key array
#   offsets_addr    int32[n_left]       prefix-sum of match counts (from Phase 1)
#   right_order_addr int32[n_right]     argsort of right_keys
#   right_starts_addr int32[table_size] per-key start index in right_order
#   right_count_addr  int32[table_size] per-key match count
#
# Outputs:
#   left_out_addr   int32[total]        left row indices
#   right_out_addr  int32[total]        right row indices
@export
def join_scatter_i32_gpu(
    left_out_addr:    Int,  # out: int32[total]
    right_out_addr:   Int,  # out: int32[total]
    left_addr:        Int,  # in: int32[n_left]
    offsets_addr:     Int,  # in: int32[n_left]
    right_order_addr: Int,  # in: int32[n_right]
    right_starts_addr:Int,  # in: int32[table_size]
    right_count_addr: Int,  # in: int32[table_size]
    n_left:           Int,
    table_size:       Int,
) abi("C"):
    @parameter
    def scatter(n_l: Int, ts: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i < n_l:
            var k = Int(_i32(left_addr)[i])
            if k >= 0 and k < ts:
                var cnt     = Int(_i32(right_count_addr)[k])
                var rs      = Int(_i32(right_starts_addr)[k])
                var out_pos = Int(_i32(offsets_addr)[i])
                for j in range(cnt):
                    _i32(left_out_addr) [out_pos + j] = Int32(i)
                    _i32(right_out_addr)[out_pos + j] = _i32(right_order_addr)[rs + j]

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[scatter](
            n_left, table_size,
            grid_dim=ceildiv(n_left, BLOCK), block_dim=BLOCK)
        ctx.synchronize()
    except: pass
