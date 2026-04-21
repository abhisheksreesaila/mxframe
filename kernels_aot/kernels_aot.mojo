"""
MXFrame AOT CPU kernels — compiled once to libmxkernels_aot.so at build time.

All kernels accept raw Int addresses (pointer-as-int) and cast to
UnsafePointer[T, MutAnyOrigin] inside. This satisfies @export's C-ABI requirement
while giving fully typed pointer access. N is always a runtime parameter so one
compiled binary handles any row count — no MAX Graph JIT cold starts.

Build:
    mojo build --emit shared-lib kernels_aot/kernels_aot.mojo -o kernels_aot/libmxkernels_aot.so

Pattern confirmed working in Mojo 0.26.2:
    UnsafePointer[Float32, MutAnyOrigin](unsafe_from_address=addr)
"""

from std.memory import UnsafePointer

comptime F32 = Float32
comptime I32 = Int32
comptime I64 = Int64
comptime F32_MAX: Float32 = 3.4028234663852886e+38
comptime F32_MIN: Float32 = -3.4028234663852886e+38


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: typed pointers from raw Int addresses
# ─────────────────────────────────────────────────────────────────────────────

@always_inline
fn _f32(addr: Int) -> UnsafePointer[F32, MutAnyOrigin]:
    return UnsafePointer[F32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
fn _i32(addr: Int) -> UnsafePointer[I32, MutAnyOrigin]:
    return UnsafePointer[I32, MutAnyOrigin](unsafe_from_address=addr)

@always_inline
fn _i64(addr: Int) -> UnsafePointer[I64, MutAnyOrigin]:
    return UnsafePointer[I64, MutAnyOrigin](unsafe_from_address=addr)


# ─────────────────────────────────────────────────────────────────────────────
# group_sum_f32
# out[labels[i]] += values[i]  for i in 0..n_rows
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_sum_f32(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = F32(0.0)
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            out[g] += values[i]


# ─────────────────────────────────────────────────────────────────────────────
# group_sum_i64
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_sum_i64(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = _i64(out_addr)
    var values = _i64(values_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = I64(0)
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            out[g] += values[i]


# ─────────────────────────────────────────────────────────────────────────────
# group_min_f32
# out[labels[i]] = min(out[labels[i]], values[i])
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_min_f32(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = F32_MAX
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            var v = values[i]
            if v < out[g]:
                out[g] = v


# ─────────────────────────────────────────────────────────────────────────────
# group_max_f32
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_max_f32(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = F32_MIN
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            var v = values[i]
            if v > out[g]:
                out[g] = v


# ─────────────────────────────────────────────────────────────────────────────
# group_mean_f32
# Compute sum and count in one pass, divide at end.
# out_sum[g] receives the mean; out_count (scratch) is int-sized counts.
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_mean_f32(
    out_addr:      Int,
    count_addr:    Int,   # scratch: Int32 counts, length n_groups
    values_addr:   Int,
    labels_addr:   Int,
    n_rows:        Int,
    n_groups:      Int,
):
    var out    = _f32(out_addr)
    var cnt    = _i32(count_addr)
    var values = _f32(values_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = F32(0.0)
        cnt[g] = I32(0)
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            out[g] += values[i]
            cnt[g] += I32(1)
    for g in range(n_groups):
        if cnt[g] > 0:
            out[g] = out[g] / F32(Int(cnt[g]))


# ─────────────────────────────────────────────────────────────────────────────
# group_count_f32  (output is float32 counts for consistency with MAX Graph)
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_count_f32(
    out_addr:    Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = _f32(out_addr)
    var labels = _i32(labels_addr)
    for g in range(n_groups):
        out[g] = F32(0.0)
    for i in range(n_rows):
        var g = Int(labels[i])
        if g >= 0 and g < n_groups:
            out[g] += F32(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# group_composite
# out[i] = k0[i]*s0 + k1[i]*s1 + k2[i]*s2 + k3[i]*s3  (int64)
# ─────────────────────────────────────────────────────────────────────────────
@export
fn group_composite(
    out_addr:     Int,
    k0_addr:      Int,
    k1_addr:      Int,
    k2_addr:      Int,
    k3_addr:      Int,
    strides_addr: Int,   # int64[4]: s0,s1,s2,s3
    n_rows:       Int,
):
    var out     = _i64(out_addr)
    var k0      = _i32(k0_addr)
    var k1      = _i32(k1_addr)
    var k2      = _i32(k2_addr)
    var k3      = _i32(k3_addr)
    var strides = _i64(strides_addr)
    var s0 = strides[0]
    var s1 = strides[1]
    var s2 = strides[2]
    var s3 = strides[3]
    for i in range(n_rows):
        out[i] = I64(k0[i])*s0 + I64(k1[i])*s1 + I64(k2[i])*s2 + I64(k3[i])*s3


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_sum_f32
# result[0] = sum(values[i] where mask[i] != 0)
# ─────────────────────────────────────────────────────────────────────────────
@export
fn masked_global_sum_f32(
    out_addr:    Int,
    values_addr: Int,
    mask_addr:   Int,
    n:           Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var mask   = _i32(mask_addr)
    var acc: F32 = 0.0
    for i in range(n):
        if mask[i] != I32(0):
            acc += values[i]
    out[0] = acc


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_min_f32
# ─────────────────────────────────────────────────────────────────────────────
@export
fn masked_global_min_f32(
    out_addr:    Int,
    values_addr: Int,
    mask_addr:   Int,
    n:           Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var mask   = _i32(mask_addr)
    var acc: F32 = F32_MAX
    for i in range(n):
        if mask[i] != I32(0):
            var v = values[i]
            if v < acc:
                acc = v
    out[0] = acc


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_max_f32
# ─────────────────────────────────────────────────────────────────────────────
@export
fn masked_global_max_f32(
    out_addr:    Int,
    values_addr: Int,
    mask_addr:   Int,
    n:           Int,
):
    var out    = _f32(out_addr)
    var values = _f32(values_addr)
    var mask   = _i32(mask_addr)
    var acc: F32 = F32_MIN
    for i in range(n):
        if mask[i] != I32(0):
            var v = values[i]
            if v > acc:
                acc = v
    out[0] = acc


# ─────────────────────────────────────────────────────────────────────────────
# masked_global_sum_product_f32
# result[0] = sum(values_a[i] * values_b[i] where mask[i] != 0)
# ─────────────────────────────────────────────────────────────────────────────
@export
fn masked_global_sum_product_f32(
    out_addr:      Int,
    values_a_addr: Int,
    values_b_addr: Int,
    mask_addr:     Int,
    n:             Int,
):
    var out      = _f32(out_addr)
    var values_a = _f32(values_a_addr)
    var values_b = _f32(values_b_addr)
    var mask     = _i32(mask_addr)
    var acc: F32 = 0.0
    for i in range(n):
        if mask[i] != I32(0):
            acc += values_a[i] * values_b[i]
    out[0] = acc


# ─────────────────────────────────────────────────────────────────────────────
# gather_f32   output[i] = values[indices[i]]
# ─────────────────────────────────────────────────────────────────────────────
@export
fn gather_f32(
    out_addr:     Int,
    values_addr:  Int,
    indices_addr: Int,
    m:            Int,   # length of output / indices
):
    var out     = _f32(out_addr)
    var values  = _f32(values_addr)
    var indices = _i32(indices_addr)
    for i in range(m):
        out[i] = values[Int(indices[i])]


# ─────────────────────────────────────────────────────────────────────────────
# gather_i32
# ─────────────────────────────────────────────────────────────────────────────
@export
fn gather_i32(
    out_addr:     Int,
    values_addr:  Int,
    indices_addr: Int,
    m:            Int,
):
    var out     = _i32(out_addr)
    var values  = _i32(values_addr)
    var indices = _i32(indices_addr)
    for i in range(m):
        out[i] = values[Int(indices[i])]


# ─────────────────────────────────────────────────────────────────────────────
# gather_i64
# ─────────────────────────────────────────────────────────────────────────────
@export
fn gather_i64(
    out_addr:     Int,
    values_addr:  Int,
    indices_addr: Int,
    m:            Int,
):
    var out     = _i64(out_addr)
    var values  = _i64(values_addr)
    var indices = _i32(indices_addr)
    for i in range(m):
        out[i] = values[Int(indices[i])]


# ─────────────────────────────────────────────────────────────────────────────
# sort_indices
# output[i] = original index at sorted position i.  Iterative bottom-up merge sort.
# descending: 0 = ascending, 1 = descending
# ─────────────────────────────────────────────────────────────────────────────

fn _merge_sort(idx: UnsafePointer[I32, MutAnyOrigin],
               keys: UnsafePointer[I32, MutAnyOrigin],
               left: Int, mid: Int, right: Int, desc: Bool):
    """In-place stable merge of idx[left..mid) and idx[mid..right)."""
    var i = left
    var j = mid
    while i < j and j < right:
        var ki = keys[Int(idx[i])]
        var kj = keys[Int(idx[j])]
        var should_swap: Bool
        if desc:
            should_swap = kj > ki
        else:
            should_swap = kj < ki
        if should_swap:
            # Rotate: bring idx[j] into position i by shift
            var tmp = idx[j]
            var k = j
            while k > i:
                idx[k] = idx[k - 1]
                k -= 1
            idx[i] = tmp
            j += 1
        i += 1

@export
fn sort_indices(
    out_addr:   Int,
    keys_addr:  Int,
    n:          Int,
    descending: Int,   # 0=asc, 1=desc
):
    var idx  = _i32(out_addr)
    var keys = _i32(keys_addr)
    var desc = descending > 0
    for i in range(n):
        idx[i] = I32(i)
    var width = 1
    while width < n:
        var left = 0
        while left < n:
            var mid = left + width
            if mid > n:
                mid = n
            var right = mid + width
            if right > n:
                right = n
            _merge_sort(idx, keys, left, mid, right, desc)
            left += 2 * width
        width *= 2


# ─────────────────────────────────────────────────────────────────────────────
# unique_mask
# output[i] = 1 if sorted_keys[i] is the first occurrence of that value, else 0
# ─────────────────────────────────────────────────────────────────────────────
@export
fn unique_mask(
    out_addr:  Int,
    keys_addr: Int,
    n:         Int,
):
    var out  = _i32(out_addr)
    var keys = _i32(keys_addr)
    if n == 0:
        return
    out[0] = I32(1)
    for i in range(1, n):
        if keys[i] != keys[i - 1]:
            out[i] = I32(1)
        else:
            out[i] = I32(0)


# ─────────────────────────────────────────────────────────────────────────────
# prefix_sum_count
# Exclusive prefix sum of a binary mask into offsets[0..n].
# offsets[n] = total count.
# ─────────────────────────────────────────────────────────────────────────────
@export
fn prefix_sum_count(
    offsets_addr: Int,   # int32[n+1]
    mask_addr:    Int,   # int32[n]
    n:            Int,
):
    var offsets = _i32(offsets_addr)
    var mask    = _i32(mask_addr)
    offsets[0] = I32(0)
    for i in range(n):
        offsets[i + 1] = offsets[i] + mask[i]


# ─────────────────────────────────────────────────────────────────────────────
# filter_gather_f32
# output[offsets[i]] = values[i]  where mask[i] == 1
# ─────────────────────────────────────────────────────────────────────────────
@export
fn filter_gather_f32(
    out_addr:     Int,
    values_addr:  Int,
    mask_addr:    Int,
    offsets_addr: Int,
    n:            Int,
):
    var out     = _f32(out_addr)
    var values  = _f32(values_addr)
    var mask    = _i32(mask_addr)
    var offsets = _i32(offsets_addr)
    for i in range(n):
        if mask[i] == I32(1):
            out[Int(offsets[i])] = values[i]


# ─────────────────────────────────────────────────────────────────────────────
# filter_gather_i32
# ─────────────────────────────────────────────────────────────────────────────
@export
fn filter_gather_i32(
    out_addr:     Int,
    values_addr:  Int,
    mask_addr:    Int,
    offsets_addr: Int,
    n:            Int,
):
    var out     = _i32(out_addr)
    var values  = _i32(values_addr)
    var mask    = _i32(mask_addr)
    var offsets = _i32(offsets_addr)
    for i in range(n):
        if mask[i] == I32(1):
            out[Int(offsets[i])] = values[i]


# ─────────────────────────────────────────────────────────────────────────────
# filter_gather_i64
# ─────────────────────────────────────────────────────────────────────────────
@export
fn filter_gather_i64(
    out_addr:     Int,
    values_addr:  Int,
    mask_addr:    Int,
    offsets_addr: Int,
    n:            Int,
):
    var out     = _i64(out_addr)
    var values  = _i64(values_addr)
    var mask    = _i32(mask_addr)
    var offsets = _i32(offsets_addr)
    for i in range(n):
        if mask[i] == I32(1):
            out[Int(offsets[i])] = values[i]


# ─────────────────────────────────────────────────────────────────────────────
# join_count (inner join)
# For each left row: match_counts[i] = number of right rows with same key
# ─────────────────────────────────────────────────────────────────────────────
@export
fn join_count(
    match_counts_addr: Int,   # int32[n_left]
    left_keys_addr:    Int,   # int32[n_left]
    right_keys_addr:   Int,   # int32[n_right]
    n_left:            Int,
    n_right:           Int,
    max_key:           Int,   # caller provides max(max(left), max(right))
):
    var match_counts = _i32(match_counts_addr)
    var left_keys    = _i32(left_keys_addr)
    var right_keys   = _i32(right_keys_addr)
    var table_size   = max_key + 1
    var right_count  = List[I32](capacity=table_size)
    for _ in range(table_size):
        right_count.append(I32(0))
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_count[k] += I32(1)
    for i in range(n_left):
        var k = Int(left_keys[i])
        if k >= 0 and k < table_size:
            match_counts[i] = right_count[k]
        else:
            match_counts[i] = I32(0)


# ─────────────────────────────────────────────────────────────────────────────
# join_scatter (inner join)
# Emit (left_index, right_index) pairs given prefix-sum offsets
# ─────────────────────────────────────────────────────────────────────────────
@export
fn join_scatter(
    left_out_addr:   Int,   # int32[total_matches]
    right_out_addr:  Int,   # int32[total_matches]
    left_keys_addr:  Int,   # int32[n_left]
    right_keys_addr: Int,   # int32[n_right]
    offsets_addr:    Int,   # int32[n_left] exclusive prefix sum of match_counts
    n_left:          Int,
    n_right:         Int,
    max_key:         Int,
):
    var left_out   = _i32(left_out_addr)
    var right_out  = _i32(right_out_addr)
    var left_keys  = _i32(left_keys_addr)
    var right_keys = _i32(right_keys_addr)
    var offsets    = _i32(offsets_addr)
    var table_size = max_key + 1

    var right_count = List[I32](capacity=table_size)
    for _ in range(table_size):
        right_count.append(I32(0))
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_count[k] += I32(1)

    var right_start = List[I32](capacity=table_size)
    right_start.append(I32(0))
    for i in range(1, table_size):
        right_start.append(right_start[i-1] + right_count[i-1])

    var right_positions = List[I32](capacity=n_right)
    for _ in range(n_right):
        right_positions.append(I32(0))
    var cursor = List[I32](capacity=table_size)
    for i in range(table_size):
        cursor.append(right_start[i])
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_positions[Int(cursor[k])] = I32(i)
            cursor[k] += I32(1)

    for i in range(n_left):
        var k = Int(left_keys[i])
        if k < 0 or k >= table_size:
            continue
        var rc = Int(right_count[k])
        var base = Int(offsets[i])
        var rs   = Int(right_start[k])
        for j in range(rc):
            left_out[base + j]  = I32(i)
            right_out[base + j] = right_positions[rs + j]


# ─────────────────────────────────────────────────────────────────────────────
# join_count_left (left join)
# match_counts[i] = max(1, number of right matches) — unmatched left rows get 1
# ─────────────────────────────────────────────────────────────────────────────
@export
fn join_count_left(
    match_counts_addr: Int,
    left_keys_addr:    Int,
    right_keys_addr:   Int,
    n_left:            Int,
    n_right:           Int,
    max_key:           Int,
):
    var match_counts = _i32(match_counts_addr)
    var left_keys    = _i32(left_keys_addr)
    var right_keys   = _i32(right_keys_addr)
    var table_size   = max_key + 1
    var right_count  = List[I32](capacity=table_size)
    for _ in range(table_size):
        right_count.append(I32(0))
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_count[k] += I32(1)
    for i in range(n_left):
        var k = Int(left_keys[i])
        var cnt: Int = 0
        if k >= 0 and k < table_size:
            cnt = Int(right_count[k])
        match_counts[i] = I32(cnt if cnt > 0 else 1)


# ─────────────────────────────────────────────────────────────────────────────
# join_scatter_left (left join)
# Like join_scatter but unmatched left rows emit right_index = -1
# ─────────────────────────────────────────────────────────────────────────────
@export
fn join_scatter_left(
    left_out_addr:   Int,
    right_out_addr:  Int,
    left_keys_addr:  Int,
    right_keys_addr: Int,
    offsets_addr:    Int,
    n_left:          Int,
    n_right:         Int,
    max_key:         Int,
):
    var left_out   = _i32(left_out_addr)
    var right_out  = _i32(right_out_addr)
    var left_keys  = _i32(left_keys_addr)
    var right_keys = _i32(right_keys_addr)
    var offsets    = _i32(offsets_addr)
    var table_size = max_key + 1

    var right_count = List[I32](capacity=table_size)
    for _ in range(table_size):
        right_count.append(I32(0))
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_count[k] += I32(1)

    var right_start = List[I32](capacity=table_size)
    right_start.append(I32(0))
    for i in range(1, table_size):
        right_start.append(right_start[i-1] + right_count[i-1])

    var right_positions = List[I32](capacity=n_right)
    for _ in range(n_right):
        right_positions.append(I32(0))
    var cursor = List[I32](capacity=table_size)
    for i in range(table_size):
        cursor.append(right_start[i])
    for i in range(n_right):
        var k = Int(right_keys[i])
        if k >= 0 and k < table_size:
            right_positions[Int(cursor[k])] = I32(i)
            cursor[k] += I32(1)

    for i in range(n_left):
        var k = Int(left_keys[i])
        var rc: Int = 0
        if k >= 0 and k < table_size:
            rc = Int(right_count[k])
        var base = Int(offsets[i])
        if rc == 0:
            left_out[base]  = I32(i)
            right_out[base] = I32(-1)
        else:
            var rs = Int(right_start[k])
            for j in range(rc):
                left_out[base + j]  = I32(i)
                right_out[base + j] = right_positions[rs + j]
