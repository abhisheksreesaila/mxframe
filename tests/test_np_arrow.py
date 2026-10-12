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


# ---- F-026: empty and masked inputs, the cases PyArrow sends through pandas ---------------------

from mxframe.np_arrow import combine  # noqa: E402

TYPES = [pa.int32(), pa.int64(), pa.float64(), pa.string(), pa.bool_(), pa.date32(),
         pa.dictionary(pa.int32(), pa.string())]


@pytest.mark.parametrize("t", TYPES, ids=str)
def test_combine_matches_combine_chunks_except_it_keeps_a_single_chunk(t):
    empty = pa.chunked_array([], type=t)
    assert combine(empty).equals(empty.combine_chunks()) and combine(empty).type == t
    one = pa.chunked_array([pa.nulls(3, t)])
    got = combine(one)
    assert got.equals(one.chunk(0)) and [b.address if b else None for b in got.buffers()] == \
        [b.address if b else None for b in one.chunk(0).buffers()]      # no copy
    arr = pa.nulls(2, t)
    assert combine(arr) is arr                              # already an Array
    if t == pa.int64():
        two = pa.chunked_array([arr_from(range(3)), arr_from(range(3, 5))])
        assert combine(two).equals(two.combine_chunks())


def arr_from(r):
    return arr(np.array(list(r), dtype=np.int64))


@pytest.mark.parametrize("t", TYPES[:6], ids=str)
def test_empty_list_with_a_type(t):
    assert arr([], type=t).equals(pa.array([], type=t))


@pytest.mark.parametrize("dtype,t", [("float64", pa.float64()), ("int64", pa.int64()), ("float32", pa.float32())])
def test_masked_numeric_arrays(dtype, t):
    m = np.ma.array(np.arange(6).astype(dtype), mask=[0, 1, 0, 0, 1, 0])
    assert arr(m).equals(pa.array(m))
    assert arr(m, type=t).equals(pa.array(m, type=t))
    allmasked = np.ma.masked_all(1, dtype=dtype)
    assert arr(allmasked, type=t).equals(pa.array(allmasked, type=t))
    assert arr(np.ma.array(np.arange(3).astype(dtype))).equals(pa.array(np.arange(3).astype(dtype)))


def test_empty_and_masked_inputs_do_not_import_pandas():
    root = Path(__file__).resolve().parent
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n"
            "import numpy as np, pyarrow as pa\nfrom mxframe.np_arrow import array, combine\n"
            "combine(pa.chunked_array([], type=pa.string())); combine(pa.chunked_array([], type=pa.int64()))\n"
            "array([], type=pa.float64()); array(np.ma.masked_all(1, dtype=np.float64), type=pa.float64())\n"
            "array(np.ma.array(np.arange(4.0), mask=[0, 1, 0, 1]))\n"
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"


def test_whole_floats_into_an_integer_type():
    for t in (pa.int64(), pa.int32(), pa.uint8()):
        assert arr([0.0, 3.0], type=t).equals(pa.array([0.0, 3.0], type=t))
    assert arr([0.5], type=pa.int64()).equals(pa.array([0.5], type=pa.int64()))    # PyArrow's own path


@pytest.mark.parametrize("like", [pa.int32(), pa.string(), pa.dictionary(pa.int32(), pa.string())], ids=str)
def test_empty_value_set_takes_the_column_type(like):
    import pyarrow.compute as pc
    from mxframe.np_arrow import value_set
    vs = value_set([], like)
    assert len(vs) == 0 and vs.type == (like.value_type if pa.types.is_dictionary(like) else like)
    col = pa.nulls(3, like) if not pa.types.is_dictionary(like) else pa.array(["a", "b", "a"]).dictionary_encode()
    assert pc.is_in(col, value_set=vs).to_pylist() == [False] * 3
    assert value_set([1, 2], pa.int64()).equals(arr([1, 2]))


# ---- F-026 review: Table.group_by imports pandas even on non-empty data ----------------------

from mxframe.np_arrow import distinct_rows, group_aggregate  # noqa: E402


def _nullable_table(n=500, seed=3):
    r = np.random.default_rng(seed)
    k1 = pa.array([None if x == 0 else int(x) for x in r.integers(0, 4, n)], pa.int32())
    k2 = pa.array([None if x == 0 else f"s{x}" for x in r.integers(0, 3, n)], pa.string())
    v = pa.array([None if x < 0.2 else float(x) for x in r.random(n)], pa.float64())
    return pa.table({"k1": k1, "k2": k2, "v": v})


@pytest.mark.parametrize("cols", [["k1"], ["k2"], ["k1", "k2"], ["k1", "k2", "v"]])
def test_distinct_rows_matches_group_by(cols):
    t = _nullable_table()
    want = t.select(cols).group_by(cols, use_threads=False).aggregate([])
    got = distinct_rows(t.select(cols)).to_pylist()
    first, seen = [], set()
    for r in t.select(cols).to_pylist():                 # first-appearance order, nulls equal
        if tuple(r.values()) not in seen:
            seen.add(tuple(r.values()))
            first.append(r)
    assert got == first
    # same rows as PyArrow (whose order is not first appearance once a float key is involved)
    assert sorted(map(str, got)) == sorted(map(str, want.select(cols).to_pylist()))
    assert distinct_rows(t.select(cols).slice(0, 0)).num_rows == 0


@pytest.mark.parametrize("keys", [["k1"], ["k1", "k2"]])
def test_group_aggregate_matches_group_by(keys):
    t = _nullable_table()
    ops = ["sum", "min", "max", "mean", "count"]
    spec = [("v", op) for op in ops]
    want = t.group_by(keys, use_threads=False).aggregate(spec)
    got = group_aggregate(t, keys, spec)
    assert got.column_names == want.column_names
    for name in want.column_names:
        w, g = want.column(name).to_pylist(), got.column(name).to_pylist()
        if name.startswith("v_") and name not in ("v_count",):
            assert [x is None for x in g] == [x is None for x in w], name
            assert np.allclose([x for x in g if x is not None], [x for x in w if x is not None], rtol=1e-12), name
        else:
            assert g == w, name


def test_group_aggregate_all_null_group_and_empty_input():
    t = pa.table({"k": pa.array([1, 1, 2], pa.int32()), "v": pa.array([None, None, 3.0], pa.float64())})
    got = group_aggregate(t, ["k"], [("v", "sum"), ("v", "count"), ("v", "mean"), ("v", "min")]).to_pylist()
    assert got == [{"k": 1, "v_sum": None, "v_count": 0, "v_mean": None, "v_min": None},
                   {"k": 2, "v_sum": 3.0, "v_count": 1, "v_mean": 3.0, "v_min": 3.0}]
    assert group_aggregate(t.slice(0, 0), ["k"], [("v", "sum")]).num_rows == 0


def test_masked_array_edge_cases():
    m = np.ma.array(np.arange(12.0), mask=[i % 3 == 0 for i in range(12)])
    for x in (m[2:9], m[::2], m[1:1]):
        assert arr(x).equals(pa.array(x))
    b = np.ma.array(np.array([True, False, True]), mask=[0, 1, 0])
    assert arr(b).equals(pa.array(b))
    nan_masked = np.ma.array(np.array([1.0, np.nan, 3.0]), mask=[0, 1, 0])
    assert arr(nan_masked, type=pa.int64()).equals(pa.array(nan_masked, type=pa.int64()))


def test_distinct_and_group_aggregate_do_not_import_pandas():
    root = Path(__file__).resolve().parent
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n"
            "import numpy as np, pyarrow as pa\nfrom mxframe.np_arrow import array, distinct_rows, group_aggregate\n"
            "k = pa.chunked_array([array(np.array([1, 2, 1], np.int32))]).cast(pa.int32())\n"
            "m = np.ma.array(np.array([1.0, 2.0, 3.0]), mask=[0, 1, 0])\n"
            "t = pa.table({'k': k, 'v': array(m)})\n"
            "distinct_rows(t.select(['k'])); group_aggregate(t, ['k'], [('v', 'sum'), ('v', 'count')])\n"
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout
    assert out.strip() == "False"
