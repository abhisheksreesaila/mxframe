"""
_atomic.mojo — Mojo 26.4 drop-in for `from os.atomic import Atomic`.

std.os.atomic is absent from the MAX 26.4 consumer build.
This shim uses vendor-specific LLVM/PTX intrinsics dispatched at compile time.
"""

from std.sys import is_nvidia_gpu, is_amd_gpu
from std.gpu.memory import AddressSpace
from std.memory import UnsafePointer
from std.sys.intrinsics import llvm_intrinsic


struct Atomic:

    @staticmethod
    @always_inline
    def fetch_add[
        T: DType,
        space: AddressSpace = AddressSpace.GENERIC,
    ](
        ptr: UnsafePointer[Scalar[T], MutAnyOrigin, address_space=space],
        val: Scalar[T],
    ) -> Scalar[T]:
        """Atomic add: *ptr += val, returns the old value."""
        comptime if T == DType.float32:
            comptime if space == AddressSpace.SHARED:
                comptime if is_nvidia_gpu():
                    return llvm_intrinsic["llvm.nvvm.atomic.load.add.f32.p3f32", Scalar[T], has_side_effect=True](ptr, val)
                elif is_amd_gpu():
                    return llvm_intrinsic["llvm.amdgcn.ds.fadd", Scalar[T], has_side_effect=True](ptr, val)
                else:
                    var old = ptr[]
                    ptr[] = old + val
                    return old
            else:
                comptime if is_nvidia_gpu():
                    return llvm_intrinsic["llvm.nvvm.atomic.load.add.f32.p0f32", Scalar[T], has_side_effect=True](ptr, val)
                elif is_amd_gpu():
                    return llvm_intrinsic["llvm.amdgcn.flat.atomic.fadd.f32.p0.f32", Scalar[T], has_side_effect=True](ptr, val)
                else:
                    var old = ptr[]
                    ptr[] = old + val
                    return old
        elif T == DType.int32:
            comptime if space == AddressSpace.SHARED:
                comptime if is_nvidia_gpu():
                    return llvm_intrinsic["llvm.nvvm.atomic.load.add.i32.p3i32", Scalar[T], has_side_effect=True](ptr, val)
                elif is_amd_gpu():
                    return llvm_intrinsic["llvm.amdgcn.ds.add.i32", Scalar[T], has_side_effect=True](ptr, val)
                else:
                    var old = ptr[]
                    ptr[] = old + val
                    return old
            else:
                comptime if is_nvidia_gpu():
                    return llvm_intrinsic["llvm.nvvm.atomic.load.add.i32.p0i32", Scalar[T], has_side_effect=True](ptr, val)
                elif is_amd_gpu():
                    return llvm_intrinsic["llvm.amdgcn.flat.atomic.add.i32.p0.i32", Scalar[T], has_side_effect=True](ptr, val)
                else:
                    var old = ptr[]
                    ptr[] = old + val
                    return old
        else:
            return val

    @staticmethod
    @always_inline
    def min[
        space: AddressSpace = AddressSpace.GENERIC,
    ](
        ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic min: *ptr = min(*ptr, val), returns the old value."""
        comptime if space == AddressSpace.SHARED:
            comptime if is_nvidia_gpu():
                return llvm_intrinsic["llvm.nvvm.atomic.min.f32.p3f32", Float32, has_side_effect=True](ptr, val)
            elif is_amd_gpu():
                return llvm_intrinsic["llvm.amdgcn.ds.fmin.f32", Float32, has_side_effect=True](ptr, val)
            else:
                var old = ptr[]
                if val < old:
                    ptr[] = val
                return old
        else:
            comptime if is_nvidia_gpu():
                return llvm_intrinsic["llvm.nvvm.atomic.min.f32.p0f32", Float32, has_side_effect=True](ptr, val)
            elif is_amd_gpu():
                return llvm_intrinsic["llvm.amdgcn.flat.atomic.fmin.f32.p0.f32", Float32, has_side_effect=True](ptr, val)
            else:
                var old = ptr[]
                if val < old:
                    ptr[] = val
                return old

    @staticmethod
    @always_inline
    def max[
        space: AddressSpace = AddressSpace.GENERIC,
    ](
        ptr: UnsafePointer[Float32, MutAnyOrigin, address_space=space],
        val: Float32,
    ) -> Float32:
        """Atomic max: *ptr = max(*ptr, val), returns the old value."""
        comptime if space == AddressSpace.SHARED:
            comptime if is_nvidia_gpu():
                return llvm_intrinsic["llvm.nvvm.atomic.max.f32.p3f32", Float32, has_side_effect=True](ptr, val)
            elif is_amd_gpu():
                return llvm_intrinsic["llvm.amdgcn.ds.fmax.f32", Float32, has_side_effect=True](ptr, val)
            else:
                var old = ptr[]
                if val > old:
                    ptr[] = val
                return old
        else:
            comptime if is_nvidia_gpu():
                return llvm_intrinsic["llvm.nvvm.atomic.max.f32.p0f32", Float32, has_side_effect=True](ptr, val)
            elif is_amd_gpu():
                return llvm_intrinsic["llvm.amdgcn.flat.atomic.fmax.f32.p0.f32", Float32, has_side_effect=True](ptr, val)
            else:
                var old = ptr[]
                if val > old:
                    ptr[] = val
                return old

    @staticmethod
    @always_inline
    def compare_exchange(
        ptr: UnsafePointer[Int32, MutAnyOrigin],
        mut expected: Int32,
        desired: Int32,
    ) -> Bool:
        """CAS on global (generic, p0) int32 memory.
        Returns True on success; updates expected with current value on failure.
        """
        var old: Int32
        comptime if is_nvidia_gpu():
            old = llvm_intrinsic["llvm.nvvm.atomic.cas.i32.p0", Int32, has_side_effect=True](ptr, expected, desired)
        elif is_amd_gpu():
            old = llvm_intrinsic["llvm.amdgcn.atomic.cmpxchg.i32.p0", Int32, has_side_effect=True](ptr, expected, desired)
        else:
            old = ptr[]
            if old == expected:
                ptr[] = desired
                return True
            expected = old
            return False
        var ok = old == expected
        if not ok:
            expected = old
        return ok
