"""F-046h: COUNT(DISTINCT b) per a on unsorted keys without hashing strings or serializing on a
few keys. A dictionary column (b, or the key a) reaches the native count as int32 codes of one
dictionary (chunks with their own dictionaries remapped), straight from the scan, the native filter
or a join's output; a small (key range) x (value range) is counted in a byte map; a few keys with a
wide value range are partitioned by the pair's hash, so every thread works. Each test asserts the
path that answered, and every result is compared with DuckDB."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, lit
from mxframe.np_arrow import array as A

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

WORDS = ["MAIL", "SHIP", "AIR", "RAIL", "TRUCK", "FOB", "REG AIR", "héllo"]


@pytest.fixture
def seen(monkeypatch):
    """(input column types, path) of every native count-distinct that answered."""
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    monkeypatch.setenv("MXFRAME_CPU_THREADS", "8")
    log = []
    real = nx.try_count_distinct

    def spy(comp, table, key, val, name):
        nx._last.pop("cd_path", None)
        r = real(comp, table, key, val, name)
        if r is not None:
            log.append((tuple(str(t).split("<")[0] for t in table.schema.types), nx._last.get("cd_path")))
        return r

    monkeypatch.setattr(nx, "try_count_distinct", spy)
    return log


def _dict_chunks(values, n_chunks, seed):
    """A dictionary column whose chunks each carry their own dictionary, in their own order."""
    r = np.random.default_rng(seed)
    cuts = np.linspace(0, len(values), n_chunks + 1).astype(int)
    parts = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        vals = values[a:b]
        d = list(set(vals))
        r.shuffle(d)
        idx = {v: i for i, v in enumerate(d)}
        parts.append(pa.DictionaryArray.from_arrays(pa.array([idx[v] for v in vals], pa.int32()), pa.array(d)))
    return pa.chunked_array(parts)


def _table(n=200_000, seed=5, keys=3000, chunks=4):
    r = np.random.default_rng(seed)
    words = [WORDS[i] for i in r.integers(0, len(WORDS), n)]
    return pa.table({
        "a": A(r.integers(0, keys, n).astype(np.int64)),          # unsorted
        "s": _dict_chunks(words, chunks, seed),
        "p": pa.array(words, pa.string()),                         # the same values, plain
        "v": A(r.integers(-40, 40, n).astype(np.int32)),
        "w": A(r.integers(-10**12, 10**12, n).astype(np.int64)),   # far too wide for a byte map
        "d": A(r.integers(0, 100, n).astype(np.int32)),
    })


def _duck(sql, **tables):
    con = duckdb.connect()
    for k, v in tables.items():
        con.register(k, v)
    return con.execute(sql).fetchall()


def _q(t, a, b, pred=None):
    lf = LazyFrame(t)
    if pred is not None:
        lf = lf.filter(pred)
    return lf.select(a, b).distinct().groupby(a).agg(col(b).count().alias("n")).compute(device="cpu")


def _same(got, exp, a):
    assert got.column_names == [a, "n"]
    assert got.column(a).to_pylist() == sorted(r[0] for r in exp)          # key order
    assert dict(zip(got.column(a).to_pylist(), got.column("n").to_pylist())) == dict(exp)


@pytest.mark.parametrize("chunks", [1, 4])
def test_dictionary_value_counts_codes_in_a_byte_map(seen, chunks):
    t = _table(chunks=chunks)
    got = _q(t, "a", "s", col("d") < lit(70))
    _same(got, _duck("SELECT a, COUNT(DISTINCT s) FROM t WHERE d < 70 GROUP BY a", t=t), "a")
    assert seen == [(("int64", "dictionary"), "byte map")]


def test_no_filter_and_negative_integer_values(seen):
    t = _table()
    got = _q(t, "a", "v")
    _same(got, _duck("SELECT a, COUNT(DISTINCT v) FROM t GROUP BY a", t=t), "a")
    assert seen == [(("int64", "int32"), "byte map")]


@pytest.mark.parametrize("keys", [1, 2])
@pytest.mark.parametrize("b", ["w", "p"])
def test_a_few_keys_partition_by_the_pair(seen, b, keys):
    t = _table(keys=keys)                      # one key-range partition per key: one or two workers
    got = _q(t, "a", b, col("d") >= lit(10))
    _same(got, _duck(f"SELECT a, COUNT(DISTINCT {b}) FROM t WHERE d >= 10 GROUP BY a", t=t), "a")
    assert seen and seen[0][1] == "pair partitions"


def test_wide_values_and_many_keys_keep_key_partitions(seen):
    t = _table(keys=150_000)
    got = _q(t, "a", "w")
    _same(got, _duck("SELECT a, COUNT(DISTINCT w) FROM t GROUP BY a", t=t), "a")
    assert seen == [(("int64", "int64"), "key partitions")]


def test_dictionary_key_comes_back_decoded_in_value_order(seen):
    t = _table()
    got = _q(t, "s", "v", col("d") < lit(50))
    assert got.schema.field("s").type == pa.string()
    _same(got, _duck("SELECT s, COUNT(DISTINCT v) FROM t WHERE d < 50 GROUP BY s", t=t), "s")
    assert seen == [(("dictionary", "int32"), "byte map")]


def test_dictionary_column_through_a_join(seen):
    t = _table(keys=500)
    r = np.random.default_rng(9)
    dim = pa.table({"dk": A(np.arange(500, dtype=np.int64)), "g": A(r.integers(0, 25, 500).astype(np.int32))})
    got = (LazyFrame(t).join(LazyFrame(dim), left_on="a", right_on="dk").select("g", "s").distinct()
           .groupby("g").agg(col("s").count().alias("n")).compute(device="cpu"))
    _same(got, _duck("SELECT g, COUNT(DISTINCT s) FROM t JOIN dim ON a = dk GROUP BY g", t=t, dim=dim), "g")
    assert seen == [(("int32", "dictionary"), "byte map")]


def test_from_parquet(seen, tmp_path):
    import mxframe as mx
    import pyarrow.parquet as pq
    t = _table()
    pq.write_table(t.select(["a", "p", "d"]), tmp_path / "t.parquet", row_group_size=50_000)
    got = (mx.scan_parquet(str(tmp_path / "t.parquet")).filter(col("d") < lit(30)).select("a", "p").distinct()
           .groupby("a").agg(col("p").count().alias("n")).compute(device="cpu"))
    _same(got, _duck("SELECT a, COUNT(DISTINCT p) FROM t WHERE d < 30 GROUP BY a", t=t), "a")
    assert seen == [(("int64", "dictionary"), "byte map")]


def test_without_the_executor_dictionaries_are_decoded_first(seen, monkeypatch):
    t = _table()
    monkeypatch.setenv("MXFRAME_NATIVE", "0")
    got = _q(t, "a", "s", col("d") < lit(70))
    exp = _duck("SELECT a, COUNT(DISTINCT s) FROM t WHERE d < 70 GROUP BY a", t=t)
    assert got.schema.field("a").type == pa.int64()
    assert dict(zip(got.column("a").to_pylist(), got.column("n").to_pylist())) == dict(exp)
    assert seen == []


def test_null_in_a_dictionary_column_takes_the_old_path(seen):
    t = _table(n=20_000)
    s = t.column("s").combine_chunks().to_pylist()
    s[7] = None
    t = t.set_column(t.schema.get_field_index("s"), "s", pa.array(s, pa.string()).dictionary_encode())
    got = _q(t, "a", "s")
    exp = _duck("SELECT a, COUNT(DISTINCT s) FROM t GROUP BY a", t=t)
    # COUNT(DISTINCT) skips the null in SQL; distinct().count() counts it like a value: compare
    # with the plain-string column through the same API instead
    plain = _q(t.set_column(t.schema.get_field_index("s"), "s", pa.array(s, pa.string())), "a", "s")
    assert got.equals(plain)
    assert len(exp) == got.num_rows
    assert seen == []


def test_duplicate_values_in_a_dictionary_count_once(seen):
    # a legal dictionary may repeat a value: codes 0 and 2 both mean "x"
    r = np.random.default_rng(4)
    n = 50_000
    codes = r.integers(0, 3, n).astype(np.int32)
    s = pa.DictionaryArray.from_arrays(pa.array(codes), pa.array(["x", "y", "x"]))
    t = pa.table({"a": A(r.integers(0, 200, n).astype(np.int64)), "s": s})
    got = _q(t, "a", "s")
    plain = t.set_column(1, "s", s.cast(pa.string()))
    _same(got, _duck("SELECT a, COUNT(DISTINCT s) FROM t GROUP BY a", t=plain), "a")
    assert seen == [(("int64", "dictionary"), "byte map")]


def test_narrow_dictionary_indices(seen):
    r = np.random.default_rng(6)
    n = 40_000
    s = pa.DictionaryArray.from_arrays(pa.array(r.integers(0, 5, n).astype(np.int8)), pa.array(WORDS[:5]))
    t = pa.table({"a": A(r.integers(0, 300, n).astype(np.int64)), "s": s})
    got = _q(t, "a", "s")
    _same(got, _duck("SELECT a, COUNT(DISTINCT s) FROM t GROUP BY a", t=t.set_column(1, "s", s.cast(pa.string()))), "a")
    assert seen == [(("int64", "dictionary"), "byte map")]


def test_dictionary_key_with_its_own_dictionary_per_chunk(seen):
    t = _table(chunks=5)
    got = _q(t, "s", "w")
    _same(got, _duck("SELECT s, COUNT(DISTINCT w) FROM t GROUP BY s", t=t), "s")
    assert seen and seen[0][0] == ("dictionary", "int64")


def test_count_of_a_literal_and_an_empty_result(seen):
    t = _table()
    got = (LazyFrame(t).select("a", "s").distinct().groupby("a").agg(lit(1).count().alias("n"))
           .compute(device="cpu"))
    _same(got, _duck("SELECT a, COUNT(DISTINCT s) FROM t GROUP BY a", t=t), "a")
    got = _q(t, "a", "s", col("d") > lit(1000))
    assert got.num_rows == 0 and got.column_names == ["a", "n"]


@pytest.mark.parametrize("keys", [[7], [0, 10**12]])
def test_sorted_keys_with_few_runs(seen, keys):
    # one run (or two, far apart): the runs path would leave one worker per run
    r = np.random.default_rng(8)
    n = 300_000
    a = np.sort(r.choice(np.array(keys, dtype=np.int64), n))
    t = pa.table({"a": A(a), "b": A(r.integers(0, 50_000, n).astype(np.int64))})
    got = _q(t, "a", "b")
    _same(got, _duck("SELECT a, COUNT(DISTINCT b) FROM t GROUP BY a", t=t), "a")
    # a range too sparse for the hash paths stays on the runs path instead of falling back
    assert seen == [(("int64", "int64"), "runs" if len(keys) == 2 else "byte map")]


def _gpu_usable():
    try:
        import mxframe.custom_ops as co
        return co._max_accelerator() is not None
    except Exception:
        return False


@pytest.mark.parametrize("device", [pytest.param("gpu", marks=pytest.mark.skipif(not _gpu_usable(), reason="no usable GPU"))])
def test_other_devices_see_decoded_columns(seen, device):
    t = _table(n=30_000)
    got = (LazyFrame(t).filter(col("d") < lit(70)).select("a", "s").distinct().groupby("a")
           .agg(col("s").count().alias("n")).compute(device=device))
    exp = _duck("SELECT a, COUNT(DISTINCT s) FROM t WHERE d < 70 GROUP BY a", t=t)
    assert dict(zip(got.column("a").to_pylist(), got.column("n").to_pylist())) == dict(exp)


@pytest.mark.parametrize("rows", [30_000, 400_000])          # under and over auto's small-input shortcut
def test_auto_counts_on_codes_like_cpu(seen, rows):
    """F-058: auto resolved the plan with decoded columns, so COUNT(DISTINCT) per key decoded the counted strings
    first (J4 at SF10: the GPU path took 6.7 s, the CPU on codes 0.66 s). The shape now runs as device="cpu"."""
    import mxframe as mx
    t = _table(n=rows)
    lf = LazyFrame(t).filter(col("d") < lit(70)).select("a", "s").distinct().groupby("a").agg(col("s").count().alias("n"))
    got = lf.compute(device="auto")
    _same(got, _duck("SELECT a, COUNT(DISTINCT s) FROM t WHERE d < 70 GROUP BY a", t=t), "a")
    assert seen and seen[0][0] == ("int64", "dictionary")
    assert lf.last_compile_provenance["auto_device"] == "cpu"
    assert "dictionary codes" in lf.last_compile_provenance["auto_reason"]


def test_auto_steered_to_the_gpu_decodes(seen):
    """prefer="gpu" asks for the GPU path, which needs plain columns: the plan is resolved decoded, as before."""
    t = _table(n=30_000)
    lf = LazyFrame(t).filter(col("d") < lit(70)).select("a", "s").distinct().groupby("a").agg(col("s").count().alias("n"))
    got = lf.compute(device="auto", prefer="gpu")
    exp = _duck("SELECT a, COUNT(DISTINCT s) FROM t WHERE d < 70 GROUP BY a", t=t)
    assert dict(zip(got.column("a").to_pylist(), got.column("n").to_pylist())) == dict(exp)   # no ORDER BY: any order
    assert all("dictionary" not in types for types, _ in seen)
    assert "dictionary codes" not in lf.last_compile_provenance["auto_reason"]


def test_auto_with_plain_columns_is_priced_by_the_cost_model(seen):
    t = _table(n=30_000)
    lf = LazyFrame(t).select("a", "p").distinct().groupby("a").agg(col("p").count().alias("n"))
    got = lf.compute(device="auto")
    _same(got, _duck("SELECT a, COUNT(DISTINCT p) FROM t GROUP BY a", t=t), "a")
    assert "dictionary codes" not in lf.last_compile_provenance["auto_reason"]


def test_explain_says_what_compute_does(seen):
    t = _table(n=400_000)
    lf = LazyFrame(t).select("a", "s").distinct().groupby("a").agg(col("s").count().alias("n"))
    text = lf.explain(device="auto")
    lf.compute(device="auto")
    assert lf.last_compile_provenance["auto_reason"] in text
