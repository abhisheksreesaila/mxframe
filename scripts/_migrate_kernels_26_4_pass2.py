"""
Fix-up pass 2 for kernels/*.mojo migration to Mojo 26.4.

Issues fixed:
1. @staticmethod fn execute → @staticmethod def execute
   (In 26.4 ALL fn → def except @parameter fn, @export fn, @always_inline fn)
2. ManagedTensorSlice[io_spec=X, dtype=Y, rank=Z, ...] → drop dtype/rank (inferred)
3. ManagedTensorSlice[mut=True, dtype=Y, rank=Z, io_spec=_, ...] → [io_spec=Output, ...]
4. ManagedTensorSlice[dtype=Y, rank=Z, io_spec=_, ...] → [io_spec=Input, ...]
"""

import re
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"

# Decorators that allow the following `fn` to stay as `fn`
FN_KEEP_DECORATORS = {"@parameter", "@export", "@always_inline"}


def fix_fn_to_def(text: str) -> str:
    """Change `fn` → `def` for all non-special function declarations."""
    lines = text.splitlines(keepends=True)
    result = []
    prev_stripped = ""
    for line in lines:
        stripped = line.strip()
        # A `fn ` at any indentation level that is NOT preceded by a keep-decorator
        if stripped.startswith("fn ") and prev_stripped not in FN_KEEP_DECORATORS:
            indent = len(line) - len(line.lstrip())
            line = " " * indent + "def " + stripped[3:] + "\n"
        if stripped:
            prev_stripped = stripped
        result.append(line)
    return "".join(result)


def fix_managed_tensor_slice(text: str) -> str:
    """
    Fix ManagedTensorSlice parameter type annotations:

    Case 1: already has io_spec=Input/Output (from pass 1) but still has dtype/rank
      ManagedTensorSlice[io_spec=X, dtype=..., rank=..., static_spec=_]
      → ManagedTensorSlice[io_spec=X, static_spec=_]

    Case 2: old-style mut=True (output tensor)
      ManagedTensorSlice[mut=True, dtype=..., rank=..., io_spec=_, static_spec=_]
      → ManagedTensorSlice[io_spec=Output, static_spec=_]

    Case 3: old-style input tensor (no mut, wildcard io_spec)
      ManagedTensorSlice[dtype=..., rank=..., io_spec=_, static_spec=_]
      → ManagedTensorSlice[io_spec=Input, static_spec=_]
    """
    # Case 1: io_spec=X present but dtype/rank also present — drop dtype/rank
    # Pattern: ManagedTensorSlice[io_spec=(Input|Output), dtype=..., rank=..., static_spec=_]
    text = re.sub(
        r'ManagedTensorSlice\[io_spec=(Input|Output),\s*dtype=[^,]+,\s*rank=[^,]+,\s*static_spec=_\]',
        r'ManagedTensorSlice[io_spec=\1, static_spec=_]',
        text
    )

    # Case 2: mut=True output tensor
    text = re.sub(
        r'ManagedTensorSlice\[mut=True,\s*dtype=[^,]+,\s*rank=[^,]+,\s*io_spec=_,\s*static_spec=_\]',
        'ManagedTensorSlice[io_spec=Output, static_spec=_]',
        text
    )

    # Case 3: input tensor with wildcard io_spec
    text = re.sub(
        r'ManagedTensorSlice\[dtype=[^,]+,\s*rank=[^,]+,\s*io_spec=_,\s*static_spec=_\]',
        'ManagedTensorSlice[io_spec=Input, static_spec=_]',
        text
    )

    return text


def fix_file(path: Path) -> bool:
    original = path.read_text()
    text = original

    text = fix_fn_to_def(text)
    text = fix_managed_tensor_slice(text)

    if text == original:
        print(f"  (no changes) {path.name}")
        return False
    path.write_text(text)
    print(f"  FIXED        {path.name}")
    return True


def main():
    mojo_files = sorted(KERNELS_DIR.glob("*.mojo"))
    mojo_files = [f for f in mojo_files if f.name != "__init__.mojo"]
    changed = 0
    for f in mojo_files:
        if fix_file(f):
            changed += 1
    print(f"\nDone: {changed}/{len(mojo_files)} files updated.")


if __name__ == "__main__":
    main()
