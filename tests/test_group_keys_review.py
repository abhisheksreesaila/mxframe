"""F-015 review fixes: index dtype overflow in the old path, empty chunks, the composite-space
limit, invalid codes, and an end-to-end dictionary-key query against DuckDB."""
import numpy as np
import pyarrow as pa
import pytest

from omxcore import group_keys
from omxcore.custom_ops import CustomOpsCompiler

needs_lib = pytest.mark.skipif(not group_keys.available(), reason="libmxgroupkeys_cpu.so not built")


def _dict_chunk(values, dictionary, itype=pa.int32()):
    idx = pa.array([dictionary.index(v) for v in values], type=itype)
    return pa.DictionaryArray.from_arrays(idx, pa.array(dictionary))


def _ground_truth(a_codes, b_codes):
    pairs = sorted(set(zip(a_codes.tolist(), b_codes.tolist())))
    rank = {p: i for i, p in enumerate(pairs)}
    return np.array([rank[p] for p in zip(a_codes.tolist(), b_codes.tolist())], dtype=np.int32), len(pairs)


@pytest.mark.parametrize("itype,na,nb", [(pa.int8(), 50, 7), (pa.int16(), 512, 512), (pa.uint8(), 200, 100)])
def test_old_path_no_index_dtype_overflow(itype, na, nb):
    """stride * code exceeds the index dtype's range (int8 50x7 = 350; int16 512x512 = 262144)."""
    rng = np.random.default_rng(2)
    n = 40_000
    a = rng.integers(0, na, n)
    b = rng.integers(0, nb, n)
    a[:na] = np.arange(na)
    b[:nb] = np.arange(nb)
    # the int16 case needs the codes to fit: 512 < 32767, 200 < 255, 50 < 127
    da = pa.DictionaryArray.from_arrays(pa.array(a, type=itype), pa.array([f"a{i:04d}" for i in range(na)]))
    db = pa.DictionaryArray.from_arrays(pa.array(b, type=itype), pa.array([f"b{i:04d}" for i in range(nb)]))
    t = pa.table({"a": da, "b": db})
    want_ids, want_n = _ground_truth(a, b)
    ids, n_groups, cols = CustomOpsCompiler._build_group_ids(t, ["a", "b"])
    assert n_groups == want_n
    assert np.array_equal(np.asarray(ids), want_ids)


@needs_lib
def test_empty_chunk_inside_chunked_key():
    c = _dict_chunk(["a", "b", "a"] * 10, ["a", "b"])
    empty = pa.DictionaryArray.from_arrays(pa.array([], type=pa.int32()), pa.array(["a", "b"]))
    d = _dict_chunk(["b", "a"] * 5, ["b", "a"])
    t = pa.table({"k": pa.chunked_array([c, empty, d, empty])})
    ids, n, cols = group_keys.build(t, ["k"])
    wids, wn, wcols = CustomOpsCompiler._build_group_ids(t, ["k"])
    assert n == wn and np.array_equal(ids, np.asarray(wids))
    assert cols["k"].to_pylist() == wcols["k"].to_pylist()


@needs_lib
def test_composite_space_exactly_max_and_one_over():
    n = 1 << 9                      # 512 * 512 = 2**18
    codes = np.arange(n * 4) % n
    d = pa.DictionaryArray.from_arrays(pa.array(codes, type=pa.int16()),
                                       pa.array([f"v{i}" for i in range(n)]))
    t = pa.table({"a": d, "b": d})
    assert n * n == group_keys.MAX_SPACE
    got = group_keys.build(t, ["a", "b"])
    assert got is not None and got[1] == n     # only the diagonal occurs
    wids, wn, _ = CustomOpsCompiler._build_group_ids(t, ["a", "b"])
    assert np.array_equal(got[0], np.asarray(wids)) and wn == n
    d2 = pa.DictionaryArray.from_arrays(pa.array(codes % 2, type=pa.int16()),
                                        pa.array([f"v{i}" for i in range(n + 1)]))
    t2 = pa.table({"a": d, "b": d2})           # 512 * 513 > 2**18
    assert group_keys.build(t2, ["a", "b"]) is None


@needs_lib
def test_invalid_dictionary_codes_are_declined():
    bad = pa.DictionaryArray.from_arrays(pa.array([0, 1, 5, -1], type=pa.int8()),
                                         pa.array(["a", "b"]), safe=False)
    ok = _dict_chunk(["x", "y", "x", "y"], ["x", "y"])
    assert group_keys.build(pa.table({"a": bad, "b": ok}), ["a", "b"]) is None
    assert group_keys.build(pa.table({"a": bad}), ["a"]) is None


@needs_lib
@pytest.mark.slow
def test_dictionary_key_query_matches_duckdb_sf1():
    duckdb = pytest.importorskip("duckdb")
    from conftest import TPCH_DATA
    path = TPCH_DATA / "tpch_sf1" / "lineitem.parquet"
    if not path.exists():
        pytest.skip("TPC-H SF1 data not present")
    import omxcore as mx
    from omxcore import LazyFrame, col, lit
    t = mx.read_parquet_table(path, dictionary="auto",
                              columns=["l_returnflag", "l_linestatus", "l_quantity", "l_extendedprice",
                                       "l_discount", "l_shipdate"])
    assert pa.types.is_dictionary(t.schema.field("l_returnflag").type)
    r = (LazyFrame(t).filter(col("l_shipdate") <= lit(10471)).groupby("l_returnflag", "l_linestatus")
         .agg(col("l_quantity").sum().alias("q"),
              (col("l_extendedprice") * (lit(1.0) - col("l_discount"))).sum().alias("rev"))
         .compute(device="cpu"))
    names = ["l_returnflag", "l_linestatus", "q", "rev"]
    got = {(a, b): (q, v) for a, b, q, v in zip(*[r.column(c).to_pylist() for c in names])}
    rows = duckdb.connect().execute(
        f"select l_returnflag, l_linestatus, sum(l_quantity), sum(l_extendedprice*(1.0-l_discount)) "
        f"from '{path}' where l_shipdate <= 10471 group by 1, 2").fetchall()
    assert len(rows) == len(got) > 0
    for a, b, q, v in rows:
        gq, gv = got[(a, b)]
        assert gq == pytest.approx(q, rel=1e-6) and gv == pytest.approx(v, rel=1e-6)
