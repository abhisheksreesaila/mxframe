import compiler
from math import ceildiv
from gpu import WARP_SIZE, block_dim, block_idx, thread_idx
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, ManagedTensorSlice, OutputTensor

comptime key_dtype = DType.int32
comptime idx_dtype = DType.int32


# ── CPU: merge sort producing sorted index permutation ───────────────────────

fn _sort_indices_cpu(
    output: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    descending_flag: ManagedTensorSlice[dtype=DType.int32, rank=1],
):
    """Produce sorted index permutation of `keys`.

    output[i] = original index at position i in the sorted order.
    descending_flag[0] > 0 → sort descending; else ascending.
    """
    var n = keys.dim_size(0)
    var desc = Int(descending_flag[0]) > 0

    # Initialise indices 0..n-1
    for i in range(n):
        output[i] = Int32(i)

    # Iterative bottom-up merge sort (stable, O(n log n))
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
            _merge(output, keys, left, mid, right, desc)
            left += 2 * width
        width *= 2


fn _merge(
    idx: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    left: Int, mid: Int, right: Int,
    desc: Bool,
):
    """In-place merge of idx[left..mid) and idx[mid..right) using keys for ordering."""
    # Simple merge using a temporary section of the output itself
    # We use an O(n) auxiliary buffer approach with swaps
    var i = left
    var j = mid
    while i < j and j < right:
        var ki = keys[Int(idx[i])]
        var kj = keys[Int(idx[j])]
        var should_swap: Bool
        if desc:
            should_swap = kj > ki   # descending: larger values first
        else:
            should_swap = kj < ki   # ascending: smaller values first
        if should_swap:
            # Rotate idx[j] into position i
            var tmp = idx[j]
            var k = j
            while k > i:
                idx[k] = idx[k - 1]
                k -= 1
            idx[i] = tmp
            j += 1
        i += 1


# ── GPU: bitonic sort producing sorted index permutation ─────────────────────

fn _sort_indices_gpu(
    output: ManagedTensorSlice[mut=True, dtype=idx_dtype, rank=1],
    keys: ManagedTensorSlice[dtype=key_dtype, rank=1],
    descending_flag: ManagedTensorSlice[dtype=DType.int32, rank=1],
    ctx: DeviceContextPtr,
) raises:
    comptime BLOCK_SIZE = 256
    var n = keys.dim_size(0)

    if n == 0:
        return

    # 1. Initialise output indices
    @parameter
    fn init_kernel(n: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if tid < UInt(n):
            output[Int(tid)] = Int32(Int(tid))

    var init_blocks = ceildiv(n, BLOCK_SIZE)
    ctx.get_device_context().enqueue_function_experimental[init_kernel](
        n,
        grid_dim=init_blocks,
        block_dim=BLOCK_SIZE,
    )

    # 2. Bitonic sort: repeatedly compare-and-swap pairs
    #    - Outer loop k: subsequence size (2, 4, 8, ..., up to next power of 2 >= n)
    #    - Inner loop j: stride for compare-swap (k/2, k/4, ..., 1)
    var padded = 1
    while padded < n:
        padded *= 2

    var k = 2
    while k <= padded:
        var j = k >> 1
        while j > 0:
            @parameter
            fn bitonic_step_kernel(n: Int, k_val: Int, j_val: Int):
                var tid = block_dim.x * block_idx.x + thread_idx.x
                var i = Int(tid)
                if i >= n:
                    return

                var partner = i ^ j_val
                if partner <= i or partner >= n:
                    return

                # Determine direction: ascending in first half of k-block, descending in second
                var asc_block = ((i & k_val) == 0)
                var desc_flag = Int(descending_flag[0]) > 0
                # If global descending is requested, flip the direction
                if desc_flag:
                    asc_block = not asc_block

                var idx_i = Int(output[i])
                var idx_p = Int(output[partner])
                var key_i = keys[idx_i]
                var key_p = keys[idx_p]

                var should_swap: Bool
                if asc_block:
                    should_swap = key_i > key_p
                else:
                    should_swap = key_i < key_p

                if should_swap:
                    output[i] = Int32(idx_p)
                    output[partner] = Int32(idx_i)

            var step_blocks = ceildiv(n, BLOCK_SIZE)
            ctx.get_device_context().enqueue_function_experimental[bitonic_step_kernel](
                n,
                k,
                j,
                grid_dim=step_blocks,
                block_dim=BLOCK_SIZE,
            )
            j >>= 1
        k <<= 1


@compiler.register("sort_indices")
struct SortIndices:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=idx_dtype, rank=1],
        keys: InputTensor[dtype=key_dtype, rank=1],
        descending_flag: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        @parameter
        if target == "cpu":
            _sort_indices_cpu(output, keys, descending_flag)
        elif target == "gpu":
            _sort_indices_gpu(output, keys, descending_flag, ctx)
        else:
            raise Error("No known target:", target)
