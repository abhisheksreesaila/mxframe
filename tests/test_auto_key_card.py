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
