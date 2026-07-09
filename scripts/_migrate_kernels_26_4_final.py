"""
Comprehensive single-pass migration: kernels/*.mojo from Mojo 26.2 → 26.4.

Changes applied (in order):
1. Import replacements (incl. tensor→extensibility, runtime.asyncrt→std.gpu.host,
   os.atomic→std.os, enqueue_function_experimental→enqueue_function)
2. ManagedTensorSlice[mut=True/io_spec=_] → OutputTensor/InputTensor aliases
3. ALL fn → def (no exceptions — in 26.4 fn is fully removed)
4. DeviceContextPtr → DeviceContext
5. ctx.get_device_context(). → ctx.
6. UInt comparison fixes (UInt vs Int mismatch in GPU kernels)
7. @parameter if → comptime if (deprecation warning fix)
8. alias → comptime (deprecation warning fix)
"""

import re
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"

# Simple string replacements applied in order
REPLACEMENTS = [
    # Imports
    ("from tensor import InputTensor, ManagedTensorSlice, OutputTensor",
     "from extensibility import InputTensor, ManagedTensorSlice, OutputTensor"),
    ("from runtime.asyncrt import DeviceContextPtr",
     "from std.gpu.host import DeviceContext"),
    ("from math import ceildiv",              "from std.math import ceildiv"),
    ("from gpu import ",                       "from std.gpu import "),
    ("from gpu.memory import AddressSpace",    "from std.gpu.memory import AddressSpace"),
    # std.os.atomic does not exist in 26.4 consumer build.
    # Use the local _atomic.mojo shim that implements Atomic via LLVM IR intrinsics.
    ("from os.atomic import Atomic",           "from ._atomic import Atomic"),
    ("from memory import stack_allocation",    "from std.memory import stack_allocation"),
    # Type / ctx API
    ("DeviceContextPtr",                       "DeviceContext"),
    ("ctx.get_device_context().",              "ctx."),
    # enqueue_function_experimental is deprecated — use enqueue_function
    ("enqueue_function_experimental[",         "enqueue_function["),
    # Deprecation fixes
    ("alias LEFT_NO_MATCH",                    "comptime LEFT_NO_MATCH"),
]


def fix_managed_tensor_slice(text: str) -> str:
    """
    Convert old-style ManagedTensorSlice helper-function param types
    to the 26.4 InputTensor / OutputTensor aliases (from extensibility).

    Pattern A (output): mut=True + explicit dtype/rank + io_spec=_
      ManagedTensorSlice[mut=True, dtype=X, rank=Y, io_spec=_, static_spec=_]
      → OutputTensor[dtype=X, rank=Y, static_spec=_]

    Pattern B (input): no mut + explicit dtype/rank + io_spec=_
      ManagedTensorSlice[dtype=X, rank=Y, io_spec=_, static_spec=_]
      → InputTensor[dtype=X, rank=Y, static_spec=_]
    """
    text = re.sub(
        r'ManagedTensorSlice\[mut=True,\s*dtype=([^,]+),\s*rank=([^,]+),\s*io_spec=_,\s*static_spec=_\]',
        r'OutputTensor[dtype=\1, rank=\2, static_spec=_]',
        text,
    )
    text = re.sub(
        r'ManagedTensorSlice\[dtype=([^,]+),\s*rank=([^,]+),\s*io_spec=_,\s*static_spec=_\]',
        r'InputTensor[dtype=\1, rank=\2, static_spec=_]',
        text,
    )
    return text


def fix_fn_to_def(text: str) -> str:
    """
    Replace ALL `fn ` → `def ` at any indentation level.
    In Mojo 26.4 (1.0.0b2) the `fn` keyword is fully removed.
    """
    text = re.sub(r'^(\s*)fn ', r'\1def ', text, flags=re.MULTILINE)
    return text


def fix_uint_comparisons(text: str) -> str:
    """
    Fix UInt vs Int comparison mismatches in GPU kernel comparisons.

    In 26.4, block/thread index arithmetic may yield Int while explicit
    UInt() casts create a type mismatch.  Wrap `tid` in Int() and drop the
    UInt() cast so both sides are Int.

      if tid < UInt(expr): → if Int(tid) < expr:
      if tid >= UInt(expr): → if Int(tid) >= expr:
    """
    text = re.sub(r'\bif tid < UInt\(([^)]+)\)',    r'if Int(tid) < \1',  text)
    text = re.sub(r'\bif tid >= UInt\(([^)]+)\)',   r'if Int(tid) >= \1', text)
    return text


def fix_parameter_if(text: str) -> str:
    """
    @parameter
    if ... → comptime
    if ...
    """
    text = re.sub(
        r'@parameter\n(\s*)if\b',
        r'comptime\n\1if',
        text,
    )
    return text


def remove_unused_atomic_import(text: str, path: Path) -> str:
    """Remove Atomic import from files that don't actually call Atomic.xxx."""
    import re
    if "Atomic." not in text and "from ._atomic import Atomic" in text:
        text = text.replace("from ._atomic import Atomic\n", "")
        print(f"    (removed unused Atomic import from {path.name})")
    return text


def migrate_file(path: Path) -> bool:
    original = path.read_text()
    text = original

    for old, new in REPLACEMENTS:
        text = text.replace(old, new)

    text = fix_managed_tensor_slice(text)
    text = fix_fn_to_def(text)
    text = fix_uint_comparisons(text)
    text = fix_parameter_if(text)
    text = remove_unused_atomic_import(text, path)

    if text == original:
        print(f"  (no changes) {path.name}")
        return False
    path.write_text(text)
    print(f"  MIGRATED     {path.name}")
    return True


def main():
    mojo_files = sorted(KERNELS_DIR.glob("*.mojo"))
    mojo_files = [f for f in mojo_files if f.name != "__init__.mojo"]
    changed = 0
    for f in mojo_files:
        if migrate_file(f):
            changed += 1
    print(f"\nDone: {changed}/{len(mojo_files)} files updated.")


if __name__ == "__main__":
    main()
