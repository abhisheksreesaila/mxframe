"""F-011 engine correctness scenarios (run by tests/test_f011_correctness.py).

    python tests/_f011_scenarios.py <scenario> <device>

Each scenario asserts internally and exits non-zero on failure.  The CPU device runs
in-process under pytest; ``gpu`` and ``auto`` run in a subprocess so a GPU fault fails
one test instead of the session.
"""
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402  (makes `import mxframe` resolve to this checkout)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

from mxframe import LazyFrame, col, lit, when  # noqa: E402
from mxframe.lazy_frame import Scan  # noqa: E402


def _lf(t):
    return LazyFrame(Scan(t))


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def count_exact_beyond_2p24(device):
    """A group of 17M rows counts exactly (float32 would stop at 16,777,216), as int64."""
    n = (1 << 24) + 1
    t = pa.table({"k": pa.array(np.zeros(n, dtype=np.int32)),
                  "v": pa.array(np.ones(n, dtype=np.float32))})
    out = _lf(t).groupby("k").agg(col("v").count().alias("n")).compute(device=device)
    assert out.schema.field("n").type == pa.int64(), out.schema
    assert out.column("n").to_pylist() == [n], out.column("n").to_pylist()


def count_exact_beyond_2p24_unfused(device):
    with _env(MXFRAME_FUSED_PIPELINE="0"):
        count_exact_beyond_2p24(device)


def count_types_all_paths(device):
    """Grouped and global counts are int64 whatever path served them, incl. empty results."""
    t = pa.table({"k": pa.array([1, 1, 2, 2, 2], type=pa.int32()),
                  "v": pa.array([1, 2, 3, 4, 5], type=pa.float32())})
    for fused in ("1", "0"):
      with _env(MXFRAME_FUSED_PIPELINE=fused):
            out = _lf(t).groupby("k").agg(col("v").count().alias("n")).sort("k").compute(device=device)
            assert out.schema.field("n").type == pa.int64(), (fused, out.schema)
            assert out.column("n").to_pylist() == [2, 3]
            out = (_lf(t).filter(col("v") > lit(100.0)).groupby("k")
                   .agg(col("v").count().alias("n")).compute(device=device))
            assert out.num_rows == 0 and out.schema.field("n").type == pa.int64(), (fused, out.schema)
            out = _lf(t).groupby().agg(col("v").count().alias("n")).compute(device=device)
            assert out.schema.field("n").type == pa.int64() and out.column("n").to_pylist() == [5]


SCENARIOS = {k: v for k, v in globals().items()
             if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) == __name__}

if __name__ == "__main__":
    name, device = sys.argv[1], sys.argv[2]
    SCENARIOS[name](device)
    print("ok")
