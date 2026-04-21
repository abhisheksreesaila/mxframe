import compiler
from runtime.asyncrt import DeviceContextPtr
from tensor import InputTensor, OutputTensor

alias dtype = DType.float32
alias MAX_GROUPS = 8192


@compiler.register("group_mean")
struct GroupMean:
    @staticmethod
    fn execute[
        target: StaticString,
    ](
        output: OutputTensor[dtype=dtype, rank=1, static_spec=_],
        values: InputTensor[dtype=dtype, rank=1, static_spec=_],
        group_ids: InputTensor[dtype=DType.int32, rank=1, static_spec=_],
        ctx: DeviceContextPtr,
    ) raises:
        var size = Int(values.spec().shape[0])
        var ng = Int(output.spec().shape[0])

        @parameter
        if target == "gpu":
            raise Error("group_mean GPU path not used; GPU mean is composed from group_sum/group_count")
        else:
            var out_ptr = output.unsafe_ptr()
            var val_ptr = values.unsafe_ptr()
            var gid_ptr = group_ids.unsafe_ptr()
            var counts = InlineArray[Scalar[dtype], MAX_GROUPS](fill=0)

            for i in range(ng):
                out_ptr[i] = 0.0

            for i in range(size):
                var g = Int(gid_ptr[i])
                if g >= 0 and g < ng:
                    out_ptr[g] += val_ptr[i]
                    counts[g] += 1.0

            for i in range(ng):
                if counts[i] > 0.0:
                    out_ptr[i] /= counts[i]
