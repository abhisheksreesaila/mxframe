# AOT spike: group_sum exported as a shared library.
# N is a runtime parameter — compiles once, works for any row count.
# Pointers are passed as Int (raw address) to satisfy @export's C-ABI requirement.
from memory import UnsafePointer


@export
fn group_sum_f32(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    """Scatter-add float32 values into group accumulators.

    out_addr    : address of pre-zeroed float32 array, length n_groups
    values_addr : address of float32 input array, length n_rows
    labels_addr : address of int32 group label array, length n_rows
    """
    var out    = UnsafePointer[Float32](unsafe_from_address=out_addr)
    var values = UnsafePointer[Float32](unsafe_from_address=values_addr)
    var labels = UnsafePointer[Int32](unsafe_from_address=labels_addr)
    for i in range(n_rows):
        out[Int(labels[i])] += values[i]


@export
fn group_sum_i64(
    out_addr:    Int,
    values_addr: Int,
    labels_addr: Int,
    n_rows:      Int,
    n_groups:    Int,
):
    var out    = UnsafePointer[Int64](unsafe_from_address=out_addr)
    var values = UnsafePointer[Int64](unsafe_from_address=values_addr)
    var labels = UnsafePointer[Int32](unsafe_from_address=labels_addr)
    for i in range(n_rows):
        out[Int(labels[i])] += values[i]
