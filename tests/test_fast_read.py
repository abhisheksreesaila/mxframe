"""Reading Parquet must not import pyarrow.dataset / pandas (~150 ms of a first query)."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import mxframe as mx
from mxframe import col, lit

ROOT = Path(__file__).resolve().parent.parent


def _fresh(code: str) -> str:
    shim = ROOT / "tests" / "conftest.py"
    prog = f"import sys; sys.path.insert(0, {str(shim.parent)!r}); import conftest\n" + code
    return subprocess.run([sys.executable, "-c", prog], capture_output=True, text=True, check=True).stdout


def _write(tmp_path, n=100_000, row_group_size=10_000):
    rng = np.random.default_rng(0)
    t = pa.table({"d": np.arange(n, dtype=np.int32), "v": rng.random(n).astype(np.float32),
                  "s": pa.array(np.where(rng.random(n) < 0.5, "x", "y")).dictionary_encode().cast(pa.string())})
    p = tmp_path / "t.parquet"
    pq.write_table(t, p, row_group_size=row_group_size)
    return p, t


def test_scan_and_read_do_not_import_pandas(tmp_path):
    p, _ = _write(tmp_path)
    out = _fresh(f"""
import mxframe as mx
from mxframe import col, lit
t = mx.read_parquet_table({str(p)!r}, columns=["d", "v"])
r = mx.scan_parquet({str(p)!r}).filter(col("d") < lit(500)).groupby().agg(col("v").sum()).compute(device="cpu")
print("pandas" in sys.modules, "pyarrow.dataset" in sys.modules)
""")
    assert out.split() == ["False", "False"]


def test_pushdown_by_row_group_stats_is_exact(tmp_path):
    p, t = _write(tmp_path)
    for pred, mask in [
        (col("d") < lit(25_000), t["d"].to_numpy() < 25_000),
        ((col("d") >= lit(31_000)) & (col("d") < lit(47_123)), (t["d"].to_numpy() >= 31_000) & (t["d"].to_numpy() < 47_123)),
        (col("d") > lit(10**9), np.zeros(t.num_rows, bool)),
        (col("s") == lit("x"), t["s"].to_numpy(zero_copy_only=False) == "x"),
        (col("d").isin([5, 50_005, 99_999]), np.isin(t["d"].to_numpy(), [5, 50_005, 99_999])),
    ]:
        got = mx.scan_parquet(p).filter(pred).groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n")).compute(device="cpu")
        want_n = int(mask.sum())
        assert got["n"][0].as_py() == want_n
        if want_n == 0:
            assert got["s"][0].as_py() is None         # SQL: a sum over no rows is NULL
        else:
            np.testing.assert_allclose(got["s"][0].as_py(), t["v"].to_numpy()[mask].astype(np.float64).sum(), rtol=1e-9)


def test_row_groups_are_skipped(tmp_path):
    from mxframe import io_parquet
    p, _ = _write(tmp_path)
    groups = io_parquet._row_groups_for(pq.ParquetFile(p), (("d", "lt", 25_000),))
    assert groups == [0, 1, 2]


def test_read_parquet_table_with_filters_keeps_exact_semantics(tmp_path):
    import pyarrow.dataset as ds
    p, t = _write(tmp_path)
    got = mx.read_parquet_table(p, columns=["d"], filters=ds.field("d") < 123)
    assert got.num_rows == 123
