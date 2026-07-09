"""
Migrate kernels/*.mojo from Mojo/MAX 26.2 API to 26.4 API.

Changes applied:
1. Import paths: math/gpu/memory/os.atomic/runtime.asyncrt/tensor → std.*
   + tensor → extensibility
2. `fn` (undecorated) → `def`
3. DeviceContextPtr → DeviceContext
4. ctx.get_device_context(). → ctx.
5. InputTensor[ → ManagedTensorSlice[io_spec=Input,
6. OutputTensor[ → ManagedTensorSlice[io_spec=Output,
"""

import re
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"

IMPORT_MAP = [
    # (old, new)
    ("from tensor import InputTensor, ManagedTensorSlice, OutputTensor",
     "from extensibility import ManagedTensorSlice, Input, Output"),
    ("from runtime.asyncrt import DeviceContextPtr",
     "from std.gpu.host import DeviceContext"),
    ("from math import ceildiv",
     "from std.math import ceildiv"),
    ("from gpu import ",
     "from std.gpu import "),
    ("from gpu.memory import AddressSpace",
     "from std.gpu.memory import AddressSpace"),
    ("from os.atomic import Atomic",
     "from std.os.atomic import Atomic"),
    ("from memory import stack_allocation",
     "from std.memory import stack_allocation"),
]

def migrate_file(path: Path) -> bool:
    original = path.read_text()
    text = original

    # 1. Fix imports (simple string replacements)
    for old, new in IMPORT_MAP:
        text = text.replace(old, new)

    # 2. InputTensor[  →  ManagedTensorSlice[io_spec=Input,
    text = text.replace("InputTensor[", "ManagedTensorSlice[io_spec=Input, ")

    # 3. OutputTensor[  →  ManagedTensorSlice[io_spec=Output,
    text = text.replace("OutputTensor[", "ManagedTensorSlice[io_spec=Output, ")

    # 4. DeviceContextPtr  →  DeviceContext  (type annotations and usages)
    text = text.replace("DeviceContextPtr", "DeviceContext")

    # 5. ctx.get_device_context().  →  ctx.
    text = text.replace("ctx.get_device_context().", "ctx.")

    # 6. Undecorated `fn ` at start of line  →  `def `
    #    Only lines where `fn ` is the first non-whitespace AND the previous
    #    non-empty line does NOT end with a Mojo decorator (`@...`).
    lines = text.splitlines(keepends=True)
    result = []
    prev_stripped = ""
    for line in lines:
        stripped = line.strip()
        # bare fn: starts with `fn ` and previous non-empty line is not a decorator
        if stripped.startswith("fn ") and not prev_stripped.startswith("@"):
            indent = len(line) - len(line.lstrip())
            line = " " * indent + "def " + stripped[3:] + ("\n" if not line.endswith("\n") else "\n")
        if stripped:
            prev_stripped = stripped
        result.append(line)
    text = "".join(result)

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
