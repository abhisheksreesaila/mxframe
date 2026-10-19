#!/usr/bin/env python3
"""Mojo frees a value at its last use. A List whose address goes to a kernel or a hook (`Int(xs.unsafe_ptr())` written into a control
block, a table of addresses) is no longer used as far as the compiler can tell, so it is freed before the kernel reads it, and the
kernel reads freed memory (F-148, F-152: twice). Keep it alive past the call with `_ = xs^`.

This checks every function in the given Mojo files: for each local name whose address is taken with `Int(<name>.unsafe_ptr())`,
the name must appear again after the last line that takes its address (the keep-alive `_ = name^`, or a later real use).
Usage: python3 scripts/check_list_lifetimes.py [files...]   (default: mojo/mxframe/*.mojo mojo/cli/*.mojo mojo/tools/*.mojo)
"""
import glob
import os
import re
import sys

ADDR = re.compile(r"Int\((\w+)\.unsafe_ptr\(\)\)")
DEF = re.compile(r"^def \w+")


def functions(lines):
    start = None
    for i, line in enumerate(lines):
        if DEF.match(line):
            if start is not None:
                yield start, lines[start:i]
            start = i
    if start is not None:
        yield start, lines[start:]


def check(path):
    bad = []
    lines = open(path).read().split("\n")
    for start, body in functions(lines):
        last_addr = {}
        for i, line in enumerate(body):
            if "# lifetime: ok" in line:                            # a loop that uses the list again, which a line check cannot see
                continue
            for name in ADDR.findall(line):
                last_addr[name] = i
        for name, at in last_addr.items():
            declared = any(re.search(rf"\bvar {name}\b", l) for l in body[:at + 1])
            if not declared:                                        # a parameter or a field: the caller owns it
                continue
            later = "\n".join(body[at + 1:])
            if not re.search(rf"\b{name}\b", later):
                bad.append((path, start + at + 1, name))
    return bad


root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sys.argv[1:] or sorted(glob.glob(root + "/mojo/mxframe/*.mojo") + glob.glob(root + "/mojo/cli/*.mojo") + glob.glob(root + "/mojo/tools/*.mojo"))
problems = []
for f in files:
    problems += check(f)
for path, line, name in problems:
    print(f"{path}:{line}: `{name}` has its address taken here and is not used afterwards: it is freed before the kernel reads it (add `_ = {name}^` after the call)")
sys.exit(1 if problems else 0)
