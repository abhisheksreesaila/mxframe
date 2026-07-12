from std.gpu.host import DeviceContext
from std.math import ceildiv
from std.gpu import thread_idx, block_idx, block_dim, barrier
from std.gpu.memory import AddressSpace
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

# ── group_sum ──────────────────────────────────────────────────────────────
@export
def group_sum_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    @parameter
    def zero_out(ng: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < ng: _f32(out_addr)[tid] = 0.0

    @parameter
    def sum_shared(n: Int, ng: Int):
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
    def sum_global(n: Int, ng: Int):
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
def group_min_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    @parameter
    def init_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = F32_MAX

    @parameter
    def min_kernel(n: Int, ng: Int):
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
def group_max_f32_gpu(
    out_addr: Int, val_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    @parameter
    def init_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = F32_MIN

    @parameter
    def max_kernel(n: Int, ng: Int):
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
def group_count_f32_gpu(
    out_addr: Int, lab_addr: Int, n_rows: Int, n_groups: Int
) abi("C"):
    @parameter
    def zero_out(ng: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < ng: _f32(out_addr)[t] = 0.0

    # Fast path: shared-memory privatisation (same pattern as group_sum)
    @parameter
    def count_shared(n: Int, ng: Int):
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
                    _ = Atomic.fetch_add(shmem + gid, Float32(1.0))
        barrier()
        t = Int(thread_idx.x)
        while t < ng:
            var sv = shmem[t]
            if sv != 0.0: _ = Atomic.fetch_add(_f32(out_addr) + t, sv)
            t += BLOCK

    # Slow path: direct global atomics (used when n_groups > MAX_GROUPS)
    @parameter
    def count_global(n: Int, ng: Int):
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
            var nb = ceildiv(n_rows, BLOCK * COARSE)
            if n_groups <= MAX_GROUPS:
                ctx.enqueue_function_experimental[count_shared](
                    n_rows, n_groups, grid_dim=nb, block_dim=BLOCK)
            else:
                ctx.enqueue_function_experimental[count_global](
                    n_rows, n_groups, grid_dim=nb, block_dim=BLOCK)
        ctx.synchronize()
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
            var state = _i32(states_addr)[slot]
            if state == Int32(0):
                var expected = Int32(0)
                if Atomic.compare_exchange(_i32(states_addr) + slot, expected, Int32(-1)):
                    _i32(hash_orders_addr)[slot] = order_key
                    _i32(hash_suppliers_addr)[slot] = supplier_key
                    _i32(states_addr)[slot] = Int32(1)
                    _ = Atomic.fetch_add(_i32(order_counts_addr) + Int(order_key), Int32(1))
                    return
                state = expected
            while state == Int32(-1):
                state = _i32(states_addr)[slot]
            if (state == Int32(1) and _i32(hash_orders_addr)[slot] == order_key and
                    _i32(hash_suppliers_addr)[slot] == supplier_key):
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
# Open-addressing linear-probing hash table: assign dense int32 group IDs
# to an int32 key array.  INT32_MIN (-2147483648) is the empty-slot sentinel;
# callers must ensure no key equals that value.
#
# Algorithm per thread i:
#   h = murmur_hash(keys[i]) % cap
#   expected = EMPTY
#   loop:
#     won = CAS(htab[h], expected, keys[i])   — expected is updated on miss
#     if won:
#       new_id = atomicAdd(out_ng, 1)          — monotone counter
#       htid[h] = new_id; out_ids[i] = new_id; return
#     if expected == keys[i]:
#       spin until htid[h] != -1; out_ids[i] = htid[h]; return
#     h = (h+1)%cap; expected = EMPTY          — linear probe
#
# Inputs:  keys[n] int32, cap (power-of-2 >= 2*n)
# Outputs: out_ids[n] dense group IDs, out_ng[1] n_groups found
# Scratch: htab[cap], htid[cap]  (kernel initialises them internally)
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
    # Pass 1: initialise hash table slots and counter atomically
    @parameter
    def init_tables(cap_: Int):
        var t = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if t < cap_:
            _i32(htab_addr)[t] = Int32(-2147483648)  # EMPTY sentinel
            _i32(htid_addr)[t] = Int32(-1)
        if t == 0:
            _i32(out_ng_addr)[0] = Int32(0)

    # Pass 2: each thread claims its key slot, assigns a dense ID
    @parameter
    def insert(n_: Int, cap_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i >= n_: return

        var k = _i32(keys_addr)[i]
        var EMPTY = Int32(-2147483648)

        # Murmur-inspired 32-bit avalanche hash
        var kk = UInt32(Int(k))
        kk ^= kk >> 16
        kk ^= UInt32(0x45d9f3b)
        kk ^= kk >> 16
        kk ^= kk >> 4
        kk ^= UInt32(0x27d4eb2f)
        kk ^= kk >> 15
        var h = Int(kk) % cap_

        var expected = EMPTY
        while True:
            # Atomic CAS: try to install our key in slot h.
            # On miss, `expected` is updated to the current slot value.
            var won = Atomic.compare_exchange(
                _i32(htab_addr) + h,
                expected,
                k,
            )
            if won:
                # Slot claimed: get a fresh dense ID (atomicAdd returns old value = new ID)
                var new_id = Atomic.fetch_add(_i32(out_ng_addr), Int32(1))
                _i32(htid_addr)[h] = new_id
                _i32(out_ids_addr)[i] = new_id
                return
            if expected == k:
                # Another thread already claimed this key; spin until it writes the ID
                var gid = _i32(htid_addr)[h]
                while gid == Int32(-1):
                    gid = _i32(htid_addr)[h]
                _i32(out_ids_addr)[i] = gid
                return
            # Collision with a different key: linear probe to next slot
            h = (h + 1) % cap_
            expected = EMPTY  # reset for the next CAS attempt

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_tables](
            cap, grid_dim=ceildiv(cap, BLOCK), block_dim=BLOCK)
        if n > 0:
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
# Full GPU bitonic sort producing a sorted index permutation.
# Caller takes out_addr[0..k-1] for top-k, or the full array for full sort.
#
# out_addr  int32[n]   output: sorted row indices
# keys_addr int32[n]   input:  sort keys
# n                    number of elements
# desc_flag            0 = ascending, 1 = descending
@export
def sort_topk_i32_gpu(
    out_addr:  Int,
    keys_addr: Int,
    n:         Int,
    desc_flag: Int,
) abi("C"):
    @parameter
    def init_kernel(n_: Int):
        var tid = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if tid < n_:
            _i32(out_addr)[tid] = Int32(tid)

    @parameter
    def bitonic_step(n_: Int, k_val: Int, j_val: Int, desc_: Int):
        var i = Int(block_idx.x) * BLOCK + Int(thread_idx.x)
        if i >= n_:
            return
        var partner = i ^ j_val
        if partner <= i or partner >= n_:
            return
        # Ascending in first half of each k-block; flip if descending requested
        var want_asc = ((i & k_val) == 0)
        if desc_ != 0:
            want_asc = not want_asc
        var idx_i = Int(_i32(out_addr)[i])
        var idx_p = Int(_i32(out_addr)[partner])
        var ki = _i32(keys_addr)[idx_i]
        var kp = _i32(keys_addr)[idx_p]
        if (want_asc and ki > kp) or (not want_asc and ki < kp):
            _i32(out_addr)[i]       = Int32(idx_p)
            _i32(out_addr)[partner] = Int32(idx_i)

    try:
        var ctx = DeviceContext()
        ctx.enqueue_function_experimental[init_kernel](
            n, grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
        # Find next power of 2 >= n
        var padded = 1
        while padded < n:
            padded *= 2
        var k = 2
        while k <= padded:
            var j = k >> 1
            while j > 0:
                ctx.enqueue_function_experimental[bitonic_step](
                    n, k, j, desc_flag,
                    grid_dim=ceildiv(n, BLOCK), block_dim=BLOCK)
                j >>= 1
            k <<= 1
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
