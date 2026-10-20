"""scripts/check_list_lifetimes.py: a List whose address a kernel reads must be alive when it reads it (F-148, F-152). The shapes the
check must catch (the reviewer's mutations of F-152's code) fail it; the repo's own Mojo code passes."""
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "check_list_lifetimes.py"


def check(*paths):
    return subprocess.run([sys.executable, str(SCRIPT), *map(str, paths)], capture_output=True, text=True)


def test_the_repos_mojo_code_passes():
    r = check()
    assert r.returncode == 0, r.stdout


BAD = {
    "a list whose address goes into a call and is never named again": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    return kernel(Int(xs.unsafe_ptr()))
""",
    "a struct method": """
struct S:
    def run(self) -> Int:
        var xs = List[Int64](length=4, fill=0)
        var addr = Int(xs.unsafe_ptr())
        return kernel(addr)
""",
    "a pointer held in a variable": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    var p = xs.unsafe_ptr()
    return kernel(Int(p))
""",
    "the only later mention is a comment": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    var r = kernel(Int(xs.unsafe_ptr()))
    # xs is read by the kernel
    return r
""",
    "the only later mention is a string": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    var r = kernel(Int(xs.unsafe_ptr()))
    print("xs is read by the kernel")
    return r
""",
    "an indexed list of lists": """
def run() -> Int:
    var ranks = List[List[Int64]]()
    ranks.append(List[Int64](length=4, fill=0))
    var r = kernel(Int(ranks[len(ranks) - 1].unsafe_ptr()))
    return r
""",
    "a pointer taken from a KernelRun that is not used again": """
def run() -> Int:
    var run = KernelRun(p, sb, sm, t, 0)
    var r = run.region(4)
    var addr = Int(run.i64(r))
    return kernel(addr)
""",
}

GOOD = {
    "kept alive by `_ = xs^`": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    var r = kernel(Int(xs.unsafe_ptr()))
    _ = xs^
    return r
""",
    "read after the call": """
def run() -> Int:
    var xs = List[Int64](length=4, fill=0)
    var r = kernel(Int(xs.unsafe_ptr()))
    return r + Int(xs[0])
""",
    "a parameter (the caller owns it)": """
def run(xs: List[Int64]) -> Int:
    return kernel(Int(xs.unsafe_ptr()))
""",
    "a field": """
struct S:
    var xs: List[Int64]
    def run(self) -> Int:
        return kernel(Int(self.xs.unsafe_ptr()))
""",
    "the owner passed to the function that reads the region": """
def run() -> Int:
    var run = KernelRun(p, sb, sm, t, 0)
    var r = run.region(4)
    return read_it(run, r)
""",
}


@pytest.mark.parametrize("name", list(BAD))
def test_the_check_catches(tmp_path, name):
    f = tmp_path / "bad.mojo"
    f.write_text(BAD[name])
    r = check(f)
    assert r.returncode == 1 and "is not used afterwards" in r.stdout, (name, r.stdout)


@pytest.mark.parametrize("name", list(GOOD))
def test_the_check_lets_through(tmp_path, name):
    f = tmp_path / "good.mojo"
    f.write_text(GOOD[name])
    r = check(f)
    assert r.returncode == 0, (name, r.stdout)


def test_deleting_the_keep_alive_in_the_top_k_is_caught(tmp_path):
    """The reviewer's mutation: rows.mojo without `_ = ranks^` (the indexed shape) and without `_ = ctl^`-style keep-alives."""
    src = (ROOT / "mojo/omxcore/rows.mojo").read_text()
    assert "_ = ranks^" in src
    f = tmp_path / "rows_mut.mojo"
    f.write_text(src.replace("    _ = ranks^\n", ""))
    r = check(f)
    assert r.returncode == 1 and "`ranks`" in r.stdout, r.stdout
