import pyarrow as pa

import mxframe
from mxframe import LazyFrame, col


def test_import_resolves_to_this_checkout():
    from pathlib import Path
    assert Path(mxframe.__file__).resolve().parent == Path(__file__).resolve().parent.parent


def test_grouped_sum_cpu():
    t = pa.table({"k": [1, 1, 2], "v": [1.0, 2.0, 5.0]})
    out = LazyFrame(t).groupby("k").agg(col("v").sum().alias("s")).sort("k").compute(device="cpu")
    assert out.column("s").to_pylist() == [3.0, 5.0]
