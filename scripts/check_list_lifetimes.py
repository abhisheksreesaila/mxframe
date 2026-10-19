#!/usr/bin/env python3
"""Mojo frees a value at its last use. A List whose address is handed to a kernel or a hook has no use the compiler can see after that,
so it is freed before the kernel reads it, and the kernel reads freed memory (F-148, F-152 twice, F-152 review: a third time).

The structural fix is `mojo/mxframe/kargs.py`'s `KernelRun`: it owns the lists of a kernel call, hands out their addresses and makes the
call itself, so no list's address leaves its owner. This check covers whatever the owner does not: for every function (a method, a nested
function too) it finds each local value whose address is taken --

    xs.unsafe_ptr()          ranks[i].unsafe_ptr()          run.i64(r) / run.i32(r) / run.u8(r)   (a pointer into a KernelRun)

(also when the pointer is first held in a variable, or goes through Int(...)) -- and requires the name to appear again in the code
after the last such line: a real use, or the keep-alive `_ = xs^`. A mention in a comment or a string does not count. Parameters, fields
(`self.x`, `look.table`) and `ref` bindings belong to the caller and are not checked. A line that is inside a loop which uses the list
again can say `# lifetime: ok`.

Usage: python3 scripts/check_list_lifetimes.py [files...]   (default: mojo/mxframe/*.mojo mojo/cli/*.mojo mojo/tools/*.mojo)
"""
import glob
import os
import re
import sys

# an address taken from a local: `xs.unsafe_ptr()`, `xs[i].unsafe_ptr()`, `run.i64(`, `UnsafePointer(to=x)`,
# `s.as_bytes().unsafe_ptr()` (groups 1, 4, 5, 6 name the local)
TAKE = re.compile(r"(?<![\w.\])])([A-Za-z_]\w*)((?:\.\w+)*)((?:\[[^\]]*\])*)\.unsafe_ptr\(\)|(?<![\w.])([A-Za-z_]\w*)\.(?:i64|i32|u8)\("
                  r"|UnsafePointer\(to=([A-Za-z_]\w*)\)|(?<![\w.])([A-Za-z_]\w*)\.as_bytes\(\)\.unsafe_ptr\(\)")
DEF = re.compile(r"^(\s*)def \w+")


def strip(lines):
    """The lines with comments and string literals (docstrings too) blanked, same line numbers."""
    out = []
    in_triple = None
    for line in lines:
        res = []
        i = 0
        while i < len(line):
            if in_triple:
                j = line.find(in_triple, i)
                if j < 0:
                    i = len(line)
                else:
                    i = j + 3
                    in_triple = None
                continue
            c = line[i]
            if line.startswith('"""', i) or line.startswith("'''", i):
                in_triple = line[i:i + 3]
                i += 3
                continue
            if c == '"' or c == "'":
                j = i + 1
                while j < len(line) and line[j] != c:
                    j += 2 if line[j] == "\\" else 1
                i = j + 1
                res.append('""')
                continue
            if c == "#":
                break
            res.append(c)
            i += 1
        out.append("".join(res))
    return out


def functions(raw, code):
    n = len(raw)
    for i in range(n):
        m = DEF.match(raw[i])
        if not m:
            continue
        indent = len(m.group(1))
        j = i + 1
        while j < n and (raw[j].strip() == "" or len(raw[j]) - len(raw[j].lstrip()) > indent):
            j += 1
        yield i, raw[i:j], code[i:j]


def check(path):
    raw = open(path).read().split("\n")
    code = strip(raw)
    bad = []
    for start, rbody, cbody in functions(raw, code):
        last = {}
        for i, line in enumerate(cbody):
            if "# lifetime: ok" in rbody[i]:
                continue
            for m in TAKE.finditer(line):
                root = m.group(1) or m.group(4) or m.group(5) or m.group(6)
                if m.group(1) and m.group(2):                        # a field of something: `self.x`, `look.table` (the owner keeps it)
                    continue
                last[root] = (i, m.end())
        for name, (at, col) in last.items():
            if name == "self":
                continue
            declared = any(re.search(rf"\bvar {name}\b", l) for l in cbody[:at + 1])
            if not declared:                                         # a parameter or a `ref`: the caller owns it
                continue
            later = cbody[at][col:] + "\n" + "\n".join(cbody[at + 1:])   # a use later on the same line counts (`f(UnsafePointer(to=n)) or n != 0`)
            if not re.search(rf"\b{name}\b", later):
                bad.append((path, start + at + 1, name))
    return bad


if __name__ == "__main__":
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = sys.argv[1:] or sorted(glob.glob(root + "/mojo/mxframe/*.mojo") + glob.glob(root + "/mojo/cli/*.mojo") + glob.glob(root + "/mojo/tools/*.mojo"))
    problems = []
    for f in files:
        problems += check(f)
    for path, line, name in problems:
        print(f"{path}:{line}: `{name}` has its address taken here and is not used afterwards: it is freed before the kernel reads it "
              f"(own it in a KernelRun, or add `_ = {name}^` after the call)")
    sys.exit(1 if problems else 0)
