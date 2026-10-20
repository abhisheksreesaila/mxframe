"""F-015: dictionary-code group ids (group_keys.py) match the existing path exactly, and CPU
queries do not import MAX."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest

from omxcore import group_keys
from omxcore.custom_ops import CustomOpsCompiler

pytestmark = pytest.mark.skipif(not group_keys.available(), reason="libmxgroupkeys_cpu.so not built")

ROOT = Path(__file__).resolve().parent.parent


def _dict_chunk(values, dictionary, itype=pa.int32()):
    idx = pa.array([dictionary.index(v) for v in values], type=itype)
    return pa.DictionaryArray.from_arrays(idx, pa.array(dictionary))


def _same(table, keys):
    got = group_keys.build(table, keys)
    want = CustomOpsCompiler._build_group_ids(table, keys)
    assert got is not None
    ids, n, cols = got
    wids, wn, wcols = want
    assert n == wn
    assert ids.dtype == np.int32 and np.array_equal(ids, np.asarray(wids))
    for k in keys:
        assert cols[k].to_pylist() == wcols[k].to_pylist()
        assert cols[k].type == wcols[k].type


@pytest.mark.parametrize("itype", [pa.int8(), pa.int16(), pa.int32(), pa.uint8(), pa.uint16()])
def test_index_widths_single_and_multi_key(itype):
    rng = np.random.default_rng(1)
    n = 50_000
    a = rng.choice(["A", "N", "R"], n)
    b = rng.choice(["F", "O"], n)
    ta = pa.chunked_array([_dict_chunk(a[:20000], ["A", "N", "R"], itype),
                           _dict_chunk(a[20000:], ["A", "N", "R"], itype)])
    tb = pa.chunked_array([_dict_chunk(b, ["F", "O"], itype)])
    t = pa.table({"a": ta, "b": tb})
    _same(t, ["a"])
    _same(t, ["a", "b"])
    _same(t, ["b", "a"])


def test_chunks_with_different_dictionaries_and_unused_entries():
    # Different dictionary order per chunk, an entry no row uses, chunk boundaries that differ per key.
    c1 = _dict_chunk(["x", "y", "x", "z"] * 500, ["x", "y", "z", "unused"])
    c2 = _dict_chunk(["z", "w", "w", "x"] * 300, ["w", "z", "x"])
    c3 = _dict_chunk(["q", "x"] * 100, ["x", "q"])
    a = pa.chunked_array([c1, c2, c3])
    n = len(a)
    k = np.arange(n) % 3
    b = pa.chunked_array([_dict_chunk([["u", "v", "t"][i] for i in k[:1700]], ["t", "u", "v"]),
                          _dict_chunk([["u", "v", "t"][i] for i in k[1700:]], ["v", "u", "t"])])
    t = pa.table({"a": a, "b": b})
    _same(t, ["a"])
    _same(t, ["a", "b"])


def test_sliced_chunks():
    full = _dict_chunk(["p", "q", "r", "q", "p", "r"] * 100, ["p", "q", "r"])
    t = pa.table({"a": pa.chunked_array([full.slice(7, 200), full.slice(300, 250)])})
    _same(t, ["a"])


def test_absent_combinations_get_no_group():
    a = _dict_chunk(["a", "a", "b", "b"], ["a", "b"])
    b = _dict_chunk(["x", "y", "x", "x"], ["x", "y"])
    t = pa.table({"a": a, "b": b})
    _same(t, ["a", "b"])            # (b, y) never occurs
    assert group_keys.build(t, ["a", "b"])[1] == 3


def test_declines_what_it_does_not_handle():
    d = _dict_chunk(["a", "b"], ["a", "b"])
    assert group_keys.build(pa.table({"a": d, "n": pa.array([1, 2])}), ["a", "n"]) is None   # not a dictionary
    nulls = pa.DictionaryArray.from_arrays(pa.array([0, None], type=pa.int32()), pa.array(["a", "b"]))
    assert group_keys.build(pa.table({"a": nulls}), ["a"]) is None                            # nulls: existing path
    big = pa.DictionaryArray.from_arrays(pa.array([0, 1], type=pa.int32()),
                                         pa.array([str(i) for i in range(1 << 10)]))
    assert group_keys.build(pa.table({"a": big, "b": big, "c": big}), ["a", "b", "c"]) is None   # space too large


def test_cpu_query_does_not_import_max():
    code = (
        "import sys, pyarrow as pa\n"
        "sys.path.insert(0, %r)\n"
        "import conftest\n"
        "import omxcore as mx\n"
        "from omxcore import LazyFrame, col\n"
        "d = pa.DictionaryArray.from_arrays(pa.array([0,1,0,1]*1000, type=pa.int8()), pa.array(['a','b']))\n"
        "t = pa.table({'k': d, 'v': pa.array([1.0]*4000)})\n"
        "r = LazyFrame(t).groupby('k').agg(col('v').sum().alias('s')).compute(device='cpu')\n"
        "assert 'max.engine' not in sys.modules and 'max.graph' not in sys.modules, 'MAX imported'\n"
        "print(sorted(r.to_pydict()['s']))\n" % str(ROOT / "tests")
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    assert "[2000.0, 2000.0]" in out.stdout
