"""F-018: joins (inner, left, string columns, composite keys) never import pandas (~150-200 ms per fresh process)."""
import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SCRIPT = textwrap.dedent("""
    import sys
    import numpy as np
    import pyarrow as pa
    from omxcore.np_arrow import array as na
    from omxcore import LazyFrame, col, lit

    n = 200_000
    a = pa.table({"k": na(np.arange(n, dtype=np.int32)), "g": na((np.arange(n) % 7).astype(np.int32)),
                  "v": na(np.random.rand(n))})
    b = pa.table({"k": na(np.arange(0, n, 3, dtype=np.int32)), "g": na((np.arange(0, n, 3) % 7).astype(np.int32)),
                  "w": na(np.random.rand(len(range(0, n, 3)))), "s": na([f"s{i % 5}" for i in range(0, n, 3)])})
    c = pa.table({"k": na(np.arange(0, n, 2, dtype=np.int32)), "z": na(np.random.rand(len(range(0, n, 2))))})
    LazyFrame(a).join(LazyFrame(b), on="k", how="left").compute(device="cpu")
    LazyFrame(a).join(LazyFrame(b), on=["k", "g"]).join(LazyFrame(c), on="k").filter(col("v") > lit(0.5)) \\
        .groupby("s").agg(col("w").sum().alias("t")).compute(device="cpu")
    # a composite key that does not pack into a small range, and a string join key
    LazyFrame(a).join(LazyFrame(b), left_on=["k", "g"], right_on=["k", "g"]).compute(device="cpu")
    LazyFrame(b).join(LazyFrame(b), on="s").limit(10).compute(device="cpu")
    assert "pandas" not in sys.modules, "pandas was imported by a join"
""")


def test_joins_do_not_import_pandas():
    import os
    env = dict(os.environ)
    tmp = Path(__file__).resolve().parent.parent.parent
    # run against this checkout: the repo root is the `omxcore` package (see conftest.py's shim)
    from conftest import _shim
    env["PYTHONPATH"] = str(_shim) + os.pathsep + env.get("PYTHONPATH", "")
    r = subprocess.run([sys.executable, "-c", SCRIPT], capture_output=True, text=True, env=env, timeout=300)
    assert r.returncode == 0, (r.stdout + r.stderr)[-2000:]
