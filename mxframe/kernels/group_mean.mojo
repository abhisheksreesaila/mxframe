# Group Mean Kernel
# Single-pass fused scatter that accumulates per-group sum and count, then divides.
# For each group g: output[g] = mean(values[i]) for all i where group_ids[i] == g
# Uses InlineArray for per-group counts (same MAX_GROUPS constraint as group_sum).
# CPU-only for Phase 1. GPU fused warp-reduction path deferred to Phase 2.

import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

alias dtype = DType.float32
alias MAX_GROUPS = 64   # Maximum groups supported — same constraint as group_sum


@compiler.register("group_mean")
struct GroupMean:
    @staticmethod
    fn execute[target: StaticString](
        output: OutputTensor[dtype=dtype, rank=1],
        values: InputTensor[dtype=dtype, rank=1],
        group_ids: InputTensor[dtype=DType.int32, rank=1],
        ctx: DeviceContextPtr,
    ) raises:
        var size    = Int(values.spec().shape[0])
        var ng      = Int(output.spec().shape[0])   # n_groups
        @parameter
        if target == "gpu":
            raise Error("group_mean GPU path not yet implemented — deferred to Phase 2.")
        else:
            var out_ptr = output.unsafe_ptr()
            var val_ptr = values.unsafe_ptr()
            var gid_ptr = group_ids.unsafe_ptr()
            # Stack-allocated per-group count buffer (compile-time MAX_GROUPS cap).
            var counts = InlineArray[Scalar[dtype], MAX_GROUPS](0)
            for i in range(ng):
                out_ptr[i] = 0.0
            # Single pass: accumulate sum and count simultaneously.
            for i in range(size):
                var g = Int(gid_ptr[i])
                if g >= 0 and g < ng:
                    out_ptr[g] += val_ptr[i]
                    counts[g]  += 1.0
            # Divide sums by counts to produce means.
            for i in range(ng):
                if counts[i] > 0.0:
                    out_ptr[i] /= counts[i]
