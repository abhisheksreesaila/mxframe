"""F-009: auto's group-cardinality estimate must stay cheap on the first query of a process
(it is on the user-felt clock): no numpy.ma import, no first-call Arrow compute for numeric keys."""
import subprocess
import sys

import numpy as np
import pyarrow as pa
import pytest

from mxframe import auto_mode as am


def _reference(col, rows):
    """The estimator as it was before F-009 (np.unique sample indices + pc.count_distinct)."""
    import pyarrow.compute as pc
    idx = np.unique(np.linspace(0, len(col) - 1, min(am._SAMPLE_ROWS, len(col))).astype(np.int64))
    d = pc.count_distinct(col.take(pa.array(idx))).as_py()
    return max(1, d) if d < 0.5 * len(idx) else max(1, int(rows * d / len(idx)))


def _cols():
    r = np.random.default_rng(9)
    yield pa.chunked_array([pa.array(r.integers(0, 7, 1_000))])
    yield pa.chunked_array([pa.array(r.integers(0, 50, 100_000))])
    yield pa.chunked_array([pa.array(r.integers(0, 200_000, 800_000))])
    yield pa.chunked_array([pa.array(r.integers(0, 1_500_000, 2_000_000))])
    yield pa.chunked_array([pa.array(np.arange(300_000) // 3)])                       # sorted
    yield pa.chunked_array([pa.array(np.arange(300_000) % 10, mask=np.arange(300_000) % 4 == 0)])
    yield pa.chunked_array([pa.array(r.normal(size=500_000).round(1))])               # float
    yield pa.chunked_array([pa.array(r.integers(0, 90, 400_000).astype(np.int32)),     # two chunks
                            pa.array(r.integers(0, 90, 100_000).astype(np.int32))])
    yield pa.chunked_array([pa.array([f"s{i % 25}" for i in range(300_000)])])        # strings


@pytest.mark.parametrize("i", range(9))
def test_same_estimate_as_before(i):
    col = list(_cols())[i]
    n = len(col)
    est = am._key_card("k", [pa.table({"k": col})], n)
    ref = _reference(col, n)
    assert abs(est - ref) <= 0.05 * ref + 1, (est, ref)


def test_numeric_estimate_imports_nothing_heavy():
    code = (
        "import sys, numpy as np, pyarrow as pa\n"
        "from mxframe import auto_mode as am\n"
        "t = pa.table({'k': pa.array(np.arange(1_000_000) % 5000)})\n"
        "before = set(sys.modules)\n"
        "am._key_card('k', [t], 1_000_000)\n"
        "print(sorted(m for m in set(sys.modules) - before if m.startswith('numpy.ma')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_many_chunks_stay_cheap_and_agree():
    import time
    n, chunks = 2_000_000, 10_000
    r = np.random.default_rng(3)
    v = r.integers(0, 40_000, n)
    col = pa.chunked_array([pa.array(c) for c in np.array_split(v, chunks)])
    t0 = time.perf_counter()
    est = am._key_card("k", [pa.table({"k": col})], n)
    took = time.perf_counter() - t0
    ref = _reference(col, n)
    assert abs(est - ref) <= 0.05 * ref + 1
    assert took < 0.2, f"{took * 1e3:.0f} ms for {chunks} chunks"   # was 285 ms before the fix


def _str_cols():
    r = np.random.default_rng(11)
    yield pa.chunked_array([pa.array(np.array(["A", "N", "R"])[r.integers(0, 3, 600_000)])])
    yield pa.chunked_array([pa.array([f"customer#{i:09d}" for i in r.integers(0, 150_000, 300_000)])])
    yield pa.chunked_array([pa.array([f"k{i % 40}" for i in range(200_000)], pa.large_string())])
    yield pa.chunked_array([pa.array([None if i % 5 == 0 else f"v{i % 17}" for i in range(100_000)])])
    yield pa.chunked_array([pa.array([f"x{i % 9}" for i in range(80_000)]),
                            pa.array([f"x{i % 9}" for i in range(50_000)]).slice(7)])     # offset chunk
    yield pa.chunked_array([pa.array(["same prefix that is longer than sixty-four bytes, then a tail #"
                                      + str(i % 300) for i in range(120_000)])])
    yield pa.chunked_array([pa.array([b"\x00" * (i % 3) for i in range(90_000)], pa.binary())])   # "", "\0", "\0\0"
    # chunks whose longest strings differ (8 vs 16 bytes, 200 vs 300): equal values must count once (F-128 review)
    yield pa.chunked_array([pa.array(["a", "b", "c"] * 20_000), pa.array(["a", "b", "c", "0123456789abc"] * 15_000)])
    long = "y" * 200
    yield pa.chunked_array([pa.array([long, "s"] * 30_000), pa.array([long, "z" * 300, "s"] * 20_000)])
    yield pa.chunked_array([pa.array([f"v{i % 300}" for i in range(9_000)]),
                            pa.array([f"v{i % 300}" if i % 7 else "x" * (60 + i % 5) for i in range(9_000)])])


@pytest.mark.parametrize("i", range(10))
def test_string_keys_same_estimate_as_before(i):
    col = list(_str_cols())[i]
    n = len(col)
    est = am._key_card("k", [pa.table({"k": col})], n)
    ref = _reference(col, n)
    assert abs(est - ref) <= 0.05 * ref + 1, (est, ref)


def test_string_estimate_does_not_import_arrow_compute():
    """F-058: a string group key imported pyarrow.compute (~25 ms) on auto's first query; device="cpu" never does."""
    code = (
        "import sys, numpy as np, pyarrow as pa\n"
        "from mxframe import auto_mode as am\n"
        "n = 1_000_002\n"     # built from buffers: pa.array(list of str) imports pyarrow.compute itself
        "off = pa.py_buffer(np.arange(n + 1, dtype=np.int32).tobytes())\n"
        "data = pa.py_buffer((b'ANR' * (n // 3)))\n"
        "t = pa.table({'k': pa.StringArray.from_buffers(n, off, data)})\n"
        "assert 'pyarrow.compute' not in sys.modules\n"
        "print(am._key_card('k', [t], 1_000_002), 'pyarrow.compute' in sys.modules)\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["3", "False"]
