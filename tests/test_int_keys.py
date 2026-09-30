"""F-020: integer group keys are encoded by direct addressing (group_keys.encode_int / dense_sorted /
first_rows) with the same groups, the same numbering and the same order as Arrow's hash and np.unique."""
import numpy as np
import pyarrow as pa
import pytest

from mxframe import group_keys
from mxframe.custom_ops import CustomOpsCompiler

pytestmark = pytest.mark.skipif(not group_keys.available(), reason="libmxgroupkeys_cpu.so not built")

N = group_keys.IK_MIN_ROWS + 1234


def _rng():
    return np.random.default_rng(20)


@pytest.mark.parametrize("dtype", [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("hi", [3, 300, 5000, 40000])
def test_encode_int_matches_arrow_dictionary_encode(dtype, hi):
    info = np.iinfo(dtype)
    hi = min(hi, int(info.max))
    lo = max(int(info.min), -hi // 2) if info.min < 0 else 0
    a = _rng().integers(lo, hi, N).astype(dtype)
    got = group_keys.encode_int(a)
    assert got is not None
    ids, uniq, first = got
    want = pa.array(a).dictionary_encode()
    assert ids.dtype == np.int32
    assert np.array_equal(ids, want.indices.to_numpy(zero_copy_only=False))
    assert np.array_equal(uniq, want.dictionary.to_numpy(zero_copy_only=False).astype(np.int64))
    assert np.array_equal(a[first], uniq) and (np.diff(first) > 0).all()


def test_encode_int_first_appearance_order_and_negative_offsets():
    a = np.tile(np.array([7, -3, 7, 100, -3, 55], dtype=np.int64), N // 6 + 1)[:N]
    ids, uniq, first = group_keys.encode_int(a)
    assert uniq.tolist() == [7, -3, 100, 55]
    assert first.tolist() == [0, 1, 3, 5]
    assert ids[:6].tolist() == [0, 1, 0, 2, 1, 3]


def test_encode_int_declines_wide_ranges_small_inputs_and_wide_types():
    sparse = np.array([0, 10**12] * (N // 2), dtype=np.int64)
    assert group_keys.encode_int(sparse) is None                      # range >> rows: caller hashes
    assert group_keys.encode_int(np.arange(100, dtype=np.int64)) is None      # too few rows
    assert group_keys.encode_int(np.arange(N, dtype=np.float64)) is None
    assert group_keys.encode_int(np.arange(N, dtype=np.uint64)) is None


def test_encode_key_uses_it_for_plain_integer_columns_only():
    from mxframe.arrow_dict import encode_key
    a = _rng().integers(0, 1000, N).astype(np.int64)
    got = encode_key(pa.chunked_array([pa.array(a[:1000]), pa.array(a[1000:])]))
    want = pa.array(a).dictionary_encode()
    assert got.indices.equals(want.indices) and got.dictionary.equals(want.dictionary)
    with_null = pa.array(a.tolist()[:-1] + [None])
    assert encode_key(with_null).indices.equals(with_null.dictionary_encode().indices)


@pytest.mark.parametrize("space", ["dense", "sparse"])
def test_dense_sorted_matches_np_unique(space):
    hi = 20_000 if space == "dense" else 3 * 10**12
    a = _rng().integers(0, hi, N).astype(np.int64)
    a[::7] = a[0]                                    # repeated values
    got = group_keys.dense_sorted(a)
    assert got is not None
    ids, uniq, first = got
    wu, wi = np.unique(a, return_inverse=True)
    assert np.array_equal(uniq, wu) and np.array_equal(ids, wi.astype(np.int32))
    if first is not None:
        assert np.array_equal(a[first], uniq)
        assert np.array_equal(first, np.unique(a, return_index=True)[1])


def test_first_rows_are_ascending_first_occurrences():
    ids = _rng().integers(0, 5000, N).astype(np.int32)
    got = group_keys.first_rows(ids, 5000)
    want = np.sort(np.unique(ids, return_index=True)[1])
    assert np.array_equal(got, want)


def test_composite_matches_numpy():
    r = _rng()
    a = r.integers(0, 100, N).astype(np.int32)
    b = r.integers(0, 3000, N).astype(np.int16)
    c = r.integers(0, 7, N).astype(np.int64)
    got = group_keys.composite([a, b, c], [3000 * 7, 7, 1])
    want = a.astype(np.int64) * 21000 + b.astype(np.int64) * 7 + c
    assert np.array_equal(got, want)


def _np_reference_ids(table, keys):
    """The pre-F-020 numbering: per-key Arrow dictionary codes, composite, np.unique."""
    encs = [table.column(k).combine_chunks().dictionary_encode() for k in keys]
    sizes = [len(e.dictionary) for e in encs]
    strides = [int(np.prod(sizes[i + 1:], dtype=np.int64)) for i in range(len(sizes))]
    comp = sum(e.indices.to_numpy().astype(np.int64) * s for e, s in zip(encs, strides))
    uniq, ids = np.unique(comp, return_inverse=True)
    return ids.astype(np.int32), len(uniq)


@pytest.mark.parametrize("keys", [["a"], ["a", "b"], ["b", "a", "c"]])
def test_build_group_ids_is_unchanged_for_integer_keys(keys):
    r = _rng()
    t = pa.table({"a": r.integers(0, 200_000, N).astype(np.int64),
                  "b": r.integers(0, 40, N).astype(np.int32),
                  "c": r.integers(0, 3, N).astype(np.int64)})
    ids, n, cols = CustomOpsCompiler._build_group_ids(t, keys)
    if len(keys) == 1:
        want = t.column(keys[0]).combine_chunks().dictionary_encode()
        wids, wn = want.indices.to_numpy(), len(want.dictionary)
    else:
        wids, wn = _np_reference_ids(t, keys)
    assert n == wn and np.array_equal(ids, wids)
    for k in keys:      # the unique key columns line up with the ids
        assert np.array_equal(cols[k].take(pa.array(ids)).to_numpy(), t.column(k).to_numpy())


def test_distinct_keeps_first_appearance_order():
    from mxframe import LazyFrame
    r = _rng()
    t = pa.table({"a": r.integers(0, 90_000, N).astype(np.int64), "b": r.integers(0, 30, N).astype(np.int64)})
    got = LazyFrame(t).select("a", "b").distinct().compute(device="cpu")
    df = t.to_pandas().drop_duplicates()
    assert got.column("a").to_pylist() == df["a"].tolist()
    assert got.column("b").to_pylist() == df["b"].tolist()
