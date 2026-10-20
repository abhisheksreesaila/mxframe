"""F-043: native string kernels. `str_slice` builds its result in one parallel Mojo pass and a utf8
column's `isin` over string values is one pass too; both count UTF-8 code points the way Arrow does.
Every case is compared with Arrow's own kernels (utf8_slice_codeunits, is_in)."""
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pytest

import omxcore.native_exec as nx
from omxcore import LazyFrame, col, lit

pytestmark = pytest.mark.skipif(not nx.available(), reason="libmxexec_cpu.so not built")

WORDS = ["", "a", "ab", "abc", "héllo", "日本語テキスト", "naïve café", "x" * 40, "😀smile", "13-555", "29-1"]


def _arr(n=50_000, seed=1, nulls=False):
    r = np.random.default_rng(seed)
    vals = [WORDS[i] for i in r.integers(0, len(WORDS), n)]
    if nulls:
        vals = [None if i % 13 == 0 else v for i, v in enumerate(vals)]
    return pa.array(vals, pa.string())


@pytest.mark.parametrize("offset,length", [(0, 2), (1, 3), (0, None), (2, None), (5, 1), (60, 2), (0, 0)])
@pytest.mark.parametrize("nulls", [False, True])
def test_slice_matches_arrow(offset, length, nulls):
    a = _arr(nulls=nulls)
    got = nx.try_str_slice(a, offset, length)
    assert got is not None
    exp = pc.utf8_slice_codeunits(a, start=offset, stop=None if length is None else offset + length)
    assert got.to_pylist() == exp.to_pylist()


def test_slice_of_a_sliced_chunked_array():
    a = pa.chunked_array([_arr(seed=2).slice(17, 20_000), _arr(seed=3, nulls=True).slice(5, 9_000)])
    got = nx.try_str_slice(a, 1, 2)
    exp = pc.utf8_slice_codeunits(a, start=1, stop=3)
    assert got.to_pylist() == exp.to_pylist()


@pytest.mark.parametrize("offset,length", [(0, 2), (0, None), (1, 1), (3, 4)])
def test_slice_isin_matches_arrow(offset, length):
    a = _arr()
    values = ["ab", "13", "", "él", "日本", "héllo", "😀s"]
    got = nx.try_str_slice_in(a, offset, length, values)
    sliced = pc.utf8_slice_codeunits(a, start=offset, stop=None if length is None else offset + length)
    exp = pc.is_in(sliced, value_set=pa.array(values, pa.string()))
    assert got.to_pylist() == exp.to_pylist()


def test_isin_declines_nulls_and_non_strings():
    assert nx.try_str_slice_in(_arr(nulls=True), 0, None, ["a"]) is None
    assert nx.try_str_slice_in(pa.array([1, 2]), 0, None, ["a"]) is None
    assert nx.try_str_slice_in(_arr(), 0, None, [1]) is None


def test_q22_shape_through_the_api():
    r = np.random.default_rng(5)
    n = 200_000
    phones = [f"{r.integers(10, 35)}-{r.integers(100, 999)}-{r.integers(1000, 9999)}" for _ in range(n)]
    t = pa.table({"p": pa.array(phones, pa.string()), "v": r.random(n)})
    codes = ["13", "31", "23", "29", "30", "18", "17"]
    got = (LazyFrame(t).with_columns(col("p").str_slice(0, 2).alias("cc"))
           .filter(col("cc").isin(codes)).groupby("cc").agg(col("v").sum().alias("s")).sort("cc")
           .compute(device="cpu"))
    exp = {}
    for p, v in zip(phones, t.column("v").to_pylist()):
        if p[:2] in codes:
            exp[p[:2]] = exp.get(p[:2], 0.0) + v
    assert got.column("cc").to_pylist() == sorted(exp)
    assert got.column("s").to_pylist() == pytest.approx([exp[k] for k in sorted(exp)])


def _served(monkeypatch):
    log = []
    real = nx.try_str_slice_in

    def spy(*a, **k):
        r = real(*a, **k)
        log.append(r is not None)
        return r

    monkeypatch.setattr(nx, "try_str_slice_in", spy)
    return log


def _t(n=30_000, seed=8):
    r = np.random.default_rng(seed)
    return pa.table({"s": _arr(n, seed), "v": r.random(n)})


@pytest.mark.parametrize("negate", [False, True])
def test_fused_slice_isin_and_plain_isin_through_the_api(monkeypatch, negate):
    t = _t()
    log = _served(monkeypatch)
    pred = col("s").str_slice(0, 2).isin(["ab", "日本", "13"])
    plain = col("s").isin(["abc", "héllo", ""])
    for p, f in ((pred, lambda s: s[:2] in ("ab", "日本", "13")), (plain, lambda s: s in ("abc", "héllo", ""))):
        got = LazyFrame(t).filter(~p if negate else p).compute(device="cpu").num_rows
        exp = sum(1 for s in t.column("s").to_pylist() if f(s) != negate)
        assert got == exp
    assert log and all(log)


def test_value_list_edges(monkeypatch):
    t = _t()
    s = t.column("s").to_pylist()
    count = lambda vals: LazyFrame(t).filter(col("s").isin(vals)).compute(device="cpu").num_rows
    assert count([]) == 0
    assert count(["ab", "ab"]) == s.count("ab")
    assert count([""]) == s.count("")
    many = [f"v{i}" for i in range(500)] + ["abc"]                 # past the native limit: Arrow
    assert count(many) == s.count("abc")


def test_dictionary_column_and_literal_left_side():
    t = _t()
    d = t.set_column(0, "s", pa.compute.dictionary_encode(t.column("s")))
    s = t.column("s").to_pylist()
    assert LazyFrame(d).filter(col("s").isin(["abc", "ab"])).compute(device="cpu").num_rows == \
        s.count("abc") + s.count("ab")
    got = LazyFrame(t).filter((col("s") == lit("abc")) | lit("ab").isin(["zz"])).compute(device="cpu")
    assert got.num_rows == s.count("abc")


def test_slice_in_of_a_sliced_input():
    a = _arr(seed=4).slice(333, 20_000)
    got = nx.try_str_slice_in(a, 1, 2, ["bc", "él", "本語"])
    exp = pc.is_in(pc.utf8_slice_codeunits(a, start=1, stop=3), value_set=pa.array(["bc", "él", "本語"]))
    assert got.to_pylist() == exp.to_pylist()
