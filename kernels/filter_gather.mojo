"""filter_gather: boolean-mask gather for GPU-side filter execution.

Replaces the PyArrow CPU filter path when running on GPU.  Two-kernel design:

  1. prefix_sum_count  -- each thread checks mask[i] and writes an exclusive
                          prefix sum into offsets[i], returning total count
                          in offsets[N] (N+1 elements).

  2. filter_gather_f32 -- scatters values[i] to output[offsets[i]] wherever
                          mask[i] == 1.  Same pattern works for any rank-1
                          float32 / int32 tensor.

Usage in the MAX Graph compiler (Python side):
    mask_node   = ... (int32 rank-1, 0 or 1)
    values_node = ... (float32 rank-1)
    offsets_node, count_node = prefix_sum_count(mask_node)
    gathered = filter_gather_f32(values_node, mask_node, offsets_node, count_node)

Both kernels are dispatched via ops.custom().
"""

import compiler
from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx, barrier
from std.gpu.memory import AddressSpace
from std.memory import stack_allocation
from std.gpu.host import DeviceContext
from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

comptime val_dtype = DType.float32
comptime idx_dtype = DType.int32


# ─────────────────────────────────────────────────────────────────────────────
# 1. prefix_sum_count
#    Input:  mask   [N]  int32, values in {0, 1}
#    Output: offsets[N+1] int32, exclusive prefix sum; offsets[N] = total count
# ─────────────────────────────────────────────────────────────────────────────

def _prefix_sum_count_cpu(
    offsets: OutputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
):
    var n = mask.dim_size(0)
    offsets[0] = 0
    for i in range(n):
        offsets[i + 1] = offsets[i] + mask[i]


def _prefix_sum_count_gpu(
    offsets: OutputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    ctx: DeviceContext,
) raises:
    """Single-pass exclusive prefix sum using a block-level scan + global fixup.

    Strategy:
      Pass 1 (block_scan): each block computes its local prefix sums into
                           offsets[], storing each block's total in block_sums[].
      Pass 2 (fixup):      add the preceding blocks' totals to each position.

    This is a two-pass approach suitable for arrays up to ~1 billion elements.
    """
    comptime BLOCK_SIZE = 256
    var n = mask.dim_size(0)
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    # Shared scratch for block totals -- reuse offsets as temp storage via
    # a dedicated global_block_sums array; here we use the tail of offsets[]
    # (offsets has n+1 elements, [0..n]).  We store block totals in a separate
    # pass using atomic approach for simplicity.

    # Pass 1: each block writes local exclusive prefix sum into offsets[1..n].
    @parameter
    def block_scan_kernel(n: Int, num_blocks: Int):
        comptime BSIZE = 256
        var shared = stack_allocation[BSIZE, Scalar[idx_dtype], address_space=AddressSpace.SHARED]()

        var tid = Int(thread_idx.x)
        var bid = Int(block_idx.x)
        var gid = bid * BSIZE + tid

        # Load mask (0 for out-of-bounds)
        var val: Int32 = 0
        if gid < n:
            val = mask[gid]
        shared[tid] = val
        barrier()

        # Kogge-Stone in-block inclusive scan
        var stride = 1
        while stride < BSIZE:
            var tmp: Int32 = 0
            if tid >= stride:
                tmp = shared[tid - stride]
            barrier()
            shared[tid] = shared[tid] + tmp
            barrier()
            stride *= 2

        # Convert to exclusive: offsets[gid+1] = shared[tid] (inclusive), offsets[0]=0
        if gid < n:
            offsets[gid + 1] = shared[tid]

        # Block total (last thread in block stores inclusive sum for this block)
        # We need to broadcast the total; store it at position n temporarily.
        # Only the last active thread of the last block writes final count.
        if bid == num_blocks - 1:
            var last_active = n - bid * BSIZE - 1
            if last_active < 0:
                last_active = 0
            if tid == last_active:
                offsets[0] = 0  # will be fixed below

    ctx.enqueue_function[block_scan_kernel](
        n,
        num_blocks,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # Pass 2: add inter-block offsets so each block's values are globally correct.
    # After pass 1, offsets[bid*BLOCK+1 .. (bid+1)*BLOCK] contain block-local
    # exclusive prefix sums.  We need to add the sum of all prior blocks.
    @parameter
    def fixup_kernel(n: Int, num_blocks: Int):
        comptime BSIZE = 256
        var bid = Int(block_idx.x)
        var tid = Int(thread_idx.x)
        var gid = bid * BSIZE + tid
        if gid >= n or bid == 0:
            return
        # Compute sum of all blocks before this one by reading the last value of
        # each prior block (the inclusive prefix sum at its last element).
        var prior_sum: Int32 = 0
        for b in range(bid):
            var last_idx = (b + 1) * BSIZE  # offsets[(b+1)*BSIZE] = inclusive sum of block b
            if last_idx <= n:
                prior_sum = offsets[last_idx]
        offsets[gid + 1] = offsets[gid + 1] + prior_sum

    ctx.enqueue_function[fixup_kernel](
        n,
        num_blocks,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )

    # offsets[0] is always 0 (guaranteed by exclusive-scan semantics).
    @parameter
    def zero_first(_n: Int):
        if Int(block_idx.x) == 0 and Int(thread_idx.x) == 0:
            offsets[0] = 0

    ctx.enqueue_function[zero_first](
        n,
        grid_dim=1,
        block_dim=1,
    )


@compiler.register("prefix_sum_count")
struct PrefixSumCount:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        offsets: OutputTensor[dtype=idx_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime
        if target == "cpu":
            _prefix_sum_count_cpu(offsets, mask)
        elif target == "gpu":
            _prefix_sum_count_gpu(offsets, mask, ctx)
        else:
            raise Error("No known target:", target)


# ─────────────────────────────────────────────────────────────────────────────
# 2. filter_gather_f32
#    Input:  values  [N]   float32
#            mask    [N]   int32 {0,1}
#            offsets [N+1] int32  (from prefix_sum_count)
#    Output: output  [N]   float32  (output[offsets[N]] is the last valid element)
# ─────────────────────────────────────────────────────────────────────────────

def _filter_gather_f32_cpu(
    output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
    values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
    mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    offsets: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
):
    var n = values.dim_size(0)
    for i in range(n):
        if mask[i] == 1:
            output[Int(offsets[i])] = values[i]


def _filter_gather_f32_gpu(
    output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
    values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
    mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    offsets: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
    ctx: DeviceContext,
) raises:
    comptime BLOCK_SIZE = 256
    var n = values.dim_size(0)
    var num_blocks = ceildiv(n, BLOCK_SIZE)

    @parameter
    def gather_kernel(n: Int):
        var gid = Int(block_dim.x) * Int(block_idx.x) + Int(thread_idx.x)
        if gid >= n:
            return
        if mask[gid] == 1:
            output[Int(offsets[gid])] = values[gid]

    ctx.enqueue_function[gather_kernel](
        n,
        grid_dim=num_blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("filter_gather_f32")
struct FilterGatherF32:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=val_dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=val_dtype, rank=1, static_spec=_],
        mask: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
        offsets: InputTensor[dtype=idx_dtype, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime
        if target == "cpu":
            _filter_gather_f32_cpu(output, values, mask, offsets)
        elif target == "gpu":
            _filter_gather_f32_gpu(output, values, mask, offsets, ctx)
        else:
            raise Error("No known target:", target)
