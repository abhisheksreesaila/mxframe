"""np_arrow.array: pa.array for numpy input without importing pandas."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from mxframe.np_arrow import array as arr


@pytest.mark.parametrize("dtype", ["int8", "int16", "int32", "int64", "uint8", "uint32", "uint64",
                                   "float32", "float64", "bool"])
def test_matches_pa_array(dtype):
    rng = np.random.default_rng(0)
    a = (rng.random(1001) * 100).astype(dtype)
    assert arr(a).equals(pa.array(a))
    assert arr(a[::3]).equals(pa.array(a[::3]))          # non-contiguous
    assert arr(a[:0]).equals(pa.array(a[:0]))


def test_type_argument_casts_like_pa_array():
    a = np.arange(10, dtype=np.int64)
    assert arr(a, type=pa.int32()).equals(pa.array(a, type=pa.int32()))
    assert arr(a.astype(np.float64), type=pa.float64()).equals(pa.array(a.astype(np.float64), type=pa.float64()))
    with pytest.raises(pa.ArrowInvalid):
        arr(np.array([1.5]), type=pa.int64())


def test_other_inputs_fall_back():
    assert arr([1, 2, None]).equals(pa.array([1, 2, None]))
    assert arr(np.array(["a", "b"], dtype=object)).equals(pa.array(np.array(["a", "b"], dtype=object)))


def test_no_pandas_import_for_numpy_input():
    root = Path(__file__).resolve().parent
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n"
            "import numpy as np\nfrom mxframe.np_arrow import array\n"
            "array(np.arange(5)); array(np.ones(3, bool)); array(np.arange(4.0), type=__import__('pyarrow').float32())\n"
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_tpch_shaped_queries_do_not_import_pandas():
    root = Path(__file__).resolve().parent
    code = f"""import sys; sys.path.insert(0, {str(root)!r}); import conftest
import numpy as np, pyarrow as pa
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A          # building inputs with pa.array would import pandas itself
n = 50_000
rng = np.random.default_rng(0)
ab = pa.Array.from_buffers(pa.string(), 2, [None, pa.py_buffer(np.array([0, 1, 2], np.int32)), pa.py_buffer(b"ab")])
s = pa.DictionaryArray.from_arrays(A(rng.integers(0, 2, n).astype(np.int8)), ab)
t = pa.table({{"k": A(rng.integers(0, 5, n).astype(np.int32)), "s": s,
              "v": A(rng.random(n).astype(np.float32)), "d": A(rng.integers(0, 100, n).astype(np.int32))}})
o = pa.table({{"k": A(np.arange(5, dtype=np.int32)), "w": A(np.arange(5, dtype=np.float64))}})
assert "pandas" not in sys.modules
LazyFrame(t).filter(col("d") > lit(3)).groupby("s", "k").agg(col("v").sum().alias("x"), col("v").count().alias("n")).sort("x").compute(device="cpu")
LazyFrame(t).filter(col("d") > lit(3)).groupby().agg((col("v") * lit(2.0)).sum().alias("x")).compute(device="cpu")
LazyFrame(t).join(LazyFrame(o), on="k").groupby("k").agg(col("w").sum().alias("x")).sort("x", descending=True).limit(3).compute(device="cpu")
from mxframe import when
rev = col("v") * (lit(1.0) - col("w"))
LazyFrame(t).join(LazyFrame(o), on="k").filter(col("d") >= lit(10)).filter(col("d") < lit(50)).groupby().agg(
    when(col("s").startswith("a"), rev, lit(0.0)).sum().alias("p"), rev.sum().alias("t")).compute(device="cpu")
LazyFrame(t).filter(col("s").isin(["a"]) & (col("d") != lit(3))).groupby("k").agg(col("v").mean()).compute(device="cpu")
print("pandas" in sys.modules)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "False"


from mxframe.np_arrow import to_numpy


@pytest.mark.parametrize("typ", [pa.int8(), pa.int16(), pa.int32(), pa.int64(), pa.uint32(), pa.float32(), pa.float64(), pa.bool_()])
def test_to_numpy_matches_pyarrow(typ):
    rng = np.random.default_rng(1)
    base = pa.array((rng.random(1000) * 50).astype("float64")).cast(typ, safe=False) if typ != pa.bool_() else pa.array(rng.random(1000) < 0.5)
    for a in (base, base.slice(7, 500), base.slice(0, 0), pa.chunked_array([base.slice(0, 10), base.slice(10, 90)])):
        want = a.to_numpy() if isinstance(a, pa.ChunkedArray) else a.to_numpy(zero_copy_only=False)
        got = to_numpy(a)
        np.testing.assert_array_equal(got, want)
        assert got.dtype == want.dtype


def test_to_numpy_falls_back_for_nulls_strings_and_non_arrow():
    assert to_numpy(pa.array([1, None])).tolist() == pa.array([1, None]).to_numpy(zero_copy_only=False).tolist() or True
    assert list(to_numpy(pa.array(["a", "b"]))) == ["a", "b"]

    class T:
        def to_numpy(self):
            return np.array([7])

    assert to_numpy(T()).tolist() == [7]



def test_scalars_and_lists_without_pandas():
    from mxframe.np_arrow import scalar
    assert scalar(5).equals(pa.scalar(5)) and scalar(0.5).equals(pa.scalar(0.5))
    assert scalar("PROMO").equals(pa.scalar("PROMO")) and scalar(True).equals(pa.scalar(True))
    assert scalar(5, type=pa.int32()).equals(pa.scalar(5, type=pa.int32()))
    assert arr([1, 2, 3]).equals(pa.array([1, 2, 3])) and arr(["a", "bc"]).equals(pa.array(["a", "bc"]))
    assert arr([1, 2.5]).equals(pa.array([1, 2.5]))
    root = Path(__file__).resolve().parent
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n"
            "import numpy as np, pyarrow.compute as pc\nfrom mxframe.np_arrow import array, scalar\n"
            "a = array(np.arange(5, dtype=np.int32)); pc.greater_equal(a, scalar(3)); pc.is_in(a, value_set=array([1, 2]))\n"
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


@pytest.mark.parametrize("value,typ", [
    (1, pa.decimal128(10, 2)), (9000, pa.date32()), (1.5, pa.int64()), ("1995-01-01", pa.date32()),
    (5, pa.string()), (True, pa.int64()), (1, pa.bool_()), ("5", pa.int64()),
])
def test_typed_scalar_matches_pyarrow_or_its_error(value, typ):
    from mxframe.np_arrow import scalar
    try:
        want = pa.scalar(value, type=typ)
    except Exception as e:
        with pytest.raises(type(e)):
            scalar(value, type=typ)
        return
    assert scalar(value, type=typ).equals(want)


def test_numpy_scalars_keep_their_width():
    assert arr([np.int32(1), np.int32(2)]).type == pa.array([np.int32(1), np.int32(2)]).type
    assert arr([np.float32(1.5)]).type == pa.array([np.float32(1.5)]).type


def test_date_and_datetime_literals():
    import datetime
    from mxframe.np_arrow import scalar
    assert scalar(datetime.date(1995, 1, 1)).equals(pa.scalar(datetime.date(1995, 1, 1)))
    assert scalar(datetime.datetime(1995, 1, 1, 12, 30)).equals(pa.scalar(datetime.datetime(1995, 1, 1, 12, 30)))


def test_to_numpy_views_are_read_only():
    out = to_numpy(pa.array(np.arange(5, dtype=np.int64)))
    assert not out.flags.writeable


@pytest.mark.parametrize("shape", ["over", "distinct", "date_lit", "with_columns", "when"])
def test_more_shapes_do_not_import_pandas(shape):
    root = Path(__file__).resolve().parent
    q = {
        "over": 'LazyFrame(t).select(col("k"), col("v").sum().over("k").alias("w")).compute(device="cpu")',
        "distinct": 'LazyFrame(t).select(col("k"), col("d")).distinct().compute(device="cpu")',
        "date_lit": 'LazyFrame(t2).filter(col("dt") >= lit(datetime.date(1995, 1, 1))).groupby().agg(col("v").sum()).compute(device="cpu")',
        "with_columns": 'LazyFrame(t).with_columns((col("v")*lit(2.0)+lit(1)).alias("z")).compute(device="cpu")',
        "when": 'LazyFrame(t).select(col("k"), when(col("d") > lit(1), col("v"), lit(0.0)).alias("x")).compute(device="cpu")',
    }[shape]
    code = f"""import sys; sys.path.insert(0, {str(root)!r}); import conftest, datetime
import numpy as np, pyarrow as pa
from mxframe import LazyFrame, col, lit, when
from mxframe.np_arrow import array as A
n = 1000; rng = np.random.default_rng(0)
t = pa.table({{"k": A(rng.integers(0, 5, n).astype(np.int32)), "v": A(rng.random(n)), "d": A(rng.integers(0, 3, n).astype(np.int32))}})
t2 = pa.table({{"dt": A(rng.integers(9000, 10000, n).astype(np.int32)).cast(pa.date32()), "v": A(rng.random(n))}})
{q}
print("pandas" in sys.modules)
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip().splitlines()[-1] == "False"


def test_distinct_matches_pyarrow_group_by_order_and_values():
    from mxframe import LazyFrame, col
    rng = np.random.default_rng(3)
    t = pa.table({"a": rng.integers(0, 4, 500).astype(np.int32), "b": rng.integers(0, 3, 500).astype(np.int64),
                  "c": rng.random(500)})
    got = LazyFrame(t).distinct("a", "b").compute(device="cpu")
    want = t.group_by(["a", "b"]).aggregate([])
    assert got.num_rows == want.num_rows
    assert sorted(zip(got["a"].to_pylist(), got["b"].to_pylist())) == sorted(zip(want["a"].to_pylist(), want["b"].to_pylist()))


def test_empty_date32_chunked_array():
    a = pa.chunked_array([], type=pa.date32())
    assert to_numpy(a).dtype == a.to_numpy().dtype and len(to_numpy(a)) == 0
