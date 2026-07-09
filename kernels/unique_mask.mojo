import compiler
from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

comptime key_dtype = DType.int32
comptime mask_dtype = DType.int32


# ── CPU: mark first occurrence of each unique value in a SORTED array ────────

def _unique_mask_cpu(
    output: OutputTensor[dtype=mask_dtype, rank=1, static_spec=_],
    sorted_keys: InputTensor[dtype=key_dtype, rank=1, static_spec=_],
):
    """Given a sorted key array, output[i] = 1 if sorted_keys[i] is the first
    occurrence of that value, else 0.

    output has same length as sorted_keys.
    """
    var n = sorted_keys.dim_size(0)
    if n == 0:
        return

    # First element is always unique
    output[0] = 1

    for i in range(1, n):
        if sorted_keys[i] != sorted_keys[i - 1]:
            output[i] = 1
        else:
            output[i] = 0


# ── GPU: parallel adjacent-difference to detect unique boundaries ────────────

def _unique_mask_gpu(
    output: OutputTensor[dtype=mask_dtype, rank=1, static_spec=_],
    sorted_keys: InputTensor[dtype=key_dtype, rank=1, static_spec=_],
    ctx: DeviceContext,
) raises:
    comptime BLOCK_SIZE = 256
    var n = sorted_keys.dim_size(0)

    if n == 0:
        return

    @parameter
    def unique_kernel(n: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if Int(tid) >= n:
            return
        var i = Int(tid)
        if i == 0:
            output[0] = 1
        else:
            if sorted_keys[i] != sorted_keys[i - 1]:
                output[i] = 1
            else:
                output[i] = 0

    var blocks = ceildiv(n, BLOCK_SIZE)
    ctx.enqueue_function[unique_kernel](
        n,
        grid_dim=blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("unique_mask")
struct UniqueMask:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=mask_dtype, rank=1, static_spec=_],
        sorted_keys: InputTensor[dtype=key_dtype, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime
        if target == "cpu":
            _unique_mask_cpu(output, sorted_keys)
        elif target == "gpu":
            _unique_mask_gpu(output, sorted_keys, ctx)
        else:
            raise Error("No known target:", target)
