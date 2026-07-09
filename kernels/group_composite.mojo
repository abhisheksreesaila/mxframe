import compiler
from std.math import ceildiv
from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from extensibility import InputTensor, ManagedTensorSlice, OutputTensor

# ── Fused composite group-key encoding ───────────────────────────────────────
#
# Computes: out[i] = k0[i]*s0 + k1[i]*s1 + k2[i]*s2 + k3[i]*s3
#
# where s0..s3 are passed in the `strides` tensor (length 4).
# Unused key slots should be filled with zeros and have stride = 0.
# Output is int64 to avoid overflow for large cartesian-product group spaces.
#
# This kernel replaces the Python NumPy loop:
#   composite = np.zeros(n, dtype=np.int32)
#   for i, enc in enumerate(encodings):
#       composite += enc.indices.to_numpy().astype(np.int32) * strides[i]
# with a single fused pass over all 4 key arrays simultaneously,
# achieving 4x better memory bandwidth utilisation.


def _group_composite_cpu(
    output: OutputTensor[dtype=DType.int64, rank=1, static_spec=_],
    k0: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k1: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k2: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k3: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    strides: InputTensor[dtype=DType.int64, rank=1, static_spec=_],
):
    var n = k0.dim_size(0)
    var s0 = strides[0]
    var s1 = strides[1]
    var s2 = strides[2]
    var s3 = strides[3]

    for i in range(n):
        var v = Int64(k0[i]) * s0 + Int64(k1[i]) * s1 + Int64(k2[i]) * s2 + Int64(k3[i]) * s3
        output[i] = v


def _group_composite_gpu(
    output: OutputTensor[dtype=DType.int64, rank=1, static_spec=_],
    k0: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k1: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k2: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    k3: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
    strides: InputTensor[dtype=DType.int64, rank=1, static_spec=_],
    ctx: DeviceContext,
) raises:
    comptime BLOCK_SIZE = 256
    var n = k0.dim_size(0)
    var s0 = strides[0]
    var s1 = strides[1]
    var s2 = strides[2]
    var s3 = strides[3]

    @parameter
    def composite_kernel(n: Int):
        var tid = block_dim.x * block_idx.x + thread_idx.x
        if Int(tid) >= n:
            return
        var i = Int(tid)
        output[i] = Int64(k0[i]) * s0 + Int64(k1[i]) * s1 + Int64(k2[i]) * s2 + Int64(k3[i]) * s3

    var blocks = ceildiv(n, BLOCK_SIZE)
    ctx.enqueue_function[composite_kernel](
        n,
        grid_dim=blocks,
        block_dim=BLOCK_SIZE,
    )


@compiler.register("group_composite")
struct GroupComposite:
    @staticmethod
    def execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=DType.int64, rank=1, static_spec=_],
        k0: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        k1: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        k2: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        k3: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        strides: InputTensor[dtype=DType.int64, rank=1, static_spec=_],
        ctx: DeviceContext,
    ) raises:
        comptime
        if target == "cpu":
            _group_composite_cpu(output, k0, k1, k2, k3, strides)
        elif target == "gpu":
            _group_composite_gpu(output, k0, k1, k2, k3, strides, ctx)
        else:
            raise Error("No known target:", target)
