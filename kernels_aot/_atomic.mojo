"""
_atomic.mojo — Mojo 26.4 drop-in for `from os.atomic import Atomic`.
Uses MLIR llvm.atomicrmw / llvm.cmpxchg — works in both mojo package
and mojo build --emit shared-lib (including GPU shared memory via addrspacecast).

AtomicBinOp: add=1(int), fadd=11, fmax=13, fmin=14
AtomicOrdering: monotonic=2, seq_cst=7
"""

from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer


@always_inline
def _to_llvm_ptr[space: AddressSpace](ptr_addr: __mlir_type.`!kgen.pointer<scalar<f32>>`) -> __mlir_type.`!llvm.ptr`:
    """Convert kgen.pointer → !llvm.ptr, handling shared memory via addrspacecast."""
    comptime if space == AddressSpace.SHARED:
        var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr_addr)
        return __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
    else:
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr_addr)


struct Atomic:

    @staticmethod
    @always_inline
    def fetch_add[space: AddressSpace = AddressSpace.GENERIC](
        ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float add via MLIR llvm.atomicrmw (fadd=11)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr.address)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`11 : i64`, ordering = __mlir_attr.`7 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    @always_inline
    def fetch_add[space: AddressSpace = AddressSpace.GENERIC](
        ptr: UnsafePointer[Int32, MutAnyOrigin, address_space=space],
        val: Int32,
    ) -> Int32:
        """Atomic int32 add via MLIR llvm.atomicrmw (add=1)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr.address)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`1 : i64`, ordering = __mlir_attr.`7 : i64`, _type = __mlir_type.i32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](r)

    @staticmethod
    @always_inline
    def min[space: AddressSpace = AddressSpace.GENERIC](
        ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float min via MLIR llvm.atomicrmw (fmin=14)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr.address)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`14 : i64`, ordering = __mlir_attr.`7 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    @always_inline
    def max[space: AddressSpace = AddressSpace.GENERIC](
        ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic float max via MLIR llvm.atomicrmw (fmax=13)."""
        var p: __mlir_type.`!llvm.ptr`
        comptime if space == AddressSpace.SHARED:
            var p3 = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr<3>`](ptr.address)
            p = __mlir_op.`llvm.addrspacecast`[_type = __mlir_type.`!llvm.ptr`](p3)
        else:
            p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
        var v = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.f32](val)
        var r = __mlir_op.`llvm.atomicrmw`[bin_op = __mlir_attr.`13 : i64`, ordering = __mlir_attr.`7 : i64`, _type = __mlir_type.f32](p, v)
        return __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Float32]](r)

    @staticmethod
    @always_inline
    def compare_exchange(
        ptr: UnsafePointer[Int32, MutAnyOrigin],
        mut expected: Int32,
        desired: Int32,
    ) -> Bool:
        """True CAS on global int32 memory via MLIR llvm.cmpxchg."""
        var p = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`!llvm.ptr`](ptr.address)
        var cmp = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](expected)
        var new = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.i32](desired)
        var result = __mlir_op.`llvm.cmpxchg`[
            success_ordering = __mlir_attr.`7 : i64`,
            failure_ordering = __mlir_attr.`2 : i64`,
            _type = __mlir_type.`!llvm.struct<(i32, i1)>`,
        ](p, cmp, new)
        var old_i32 = __mlir_op.`llvm.extractvalue`[position = __mlir_attr.`array<i64: 0>`, _type = __mlir_type.i32](result)
        var suc_i1  = __mlir_op.`llvm.extractvalue`[position = __mlir_attr.`array<i64: 1>`, _type = __mlir_type.`i1`](result)
        var ok = Bool(__mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type.`i1`](suc_i1))
        if not ok:
            expected = __mlir_op.`builtin.unrealized_conversion_cast`[_type = __mlir_type[Int32]](old_i32)
        return ok
