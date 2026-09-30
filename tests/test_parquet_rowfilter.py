"""F-017: integer conjuncts pushed into the Parquet reader drop rows per row group, in parallel."""
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
from mxframe import col
from mxframe import io_parquet as iop


def _table(n=6000, seed=3, with_nulls=False):
    rng = np.random.default_rng(seed)
    d = rng.integers(8000, 10500, n).astype(np.int32)
    t = {
        "id": pa.array(np.arange(n, dtype=np.int64)),
        "d": pa.array(d) if not with_nulls else pa.array([None if i % 97 == 0 else int(v) for i, v in enumerate(d)], pa.int32()),
        "p": pa.array(rng.uniform(1, 100, n).astype(np.float32)),
        "f": pa.array(rng.choice(["A", "B", "C"], n)),
    }
    return pa.table(t)


_REAL_ESTIMATE = iop._estimated_pass


@pytest.fixture(autouse=True)
def _always_row_filter(monkeypatch):
    monkeypatch.setattr(iop, "MIN_ROWS_FOR_ROW_FILTER", 0)      # test files are tiny
    monkeypatch.setattr(iop, "_estimated_pass", lambda *a: 0.0)  # exercise the mask code for every case


@pytest.fixture(params=[False, True], ids=["plain", "nulls"])
def path(tmp_path, request):
    p = tmp_path / "t.parquet"
    pq.write_table(_table(with_nulls=request.param), p, row_group_size=500)   # 12 row groups
    return str(p)


def _expect(path, cols, pred):
    """Rows the reader may drop: none that satisfy ``pred`` (a numpy predicate on the id order)."""
    full = pq.read_table(path)
    keep = pred(full)
    return full.select(cols).filter(pa.array(keep))


PUSHED_CASES = [
    ((("d", "ge", 9000), ("d", "lt", 9500)), lambda t: _np(t, "d", lambda a: (a >= 9000) & (a < 9500))),
    ((("d", "eq", 9000),), lambda t: _np(t, "d", lambda a: a == 9000)),
    ((("d", "ne", 9000),), lambda t: _np(t, "d", lambda a: a != 9000)),
    ((("d", "isin", (8500, 9000, 10000)),), lambda t: _np(t, "d", lambda a: np.isin(a, [8500, 9000, 10000]))),
    ((("d", "gt", 2**40),), lambda t: _np(t, "d", lambda a: np.zeros(len(a), bool))),
    ((("d", "lt", 2**40),), lambda t: _np(t, "d", lambda a: np.ones(len(a), bool))),
    ((("id", "ge", 2500), ("d", "le", 9000)), lambda t: _np(t, "id", lambda a: a >= 2500) & _np(t, "d", lambda a: a <= 9000)),
]


def _np(t, name, f):
    c = t.column(name)
    a = np.asarray(c.fill_null(-1).to_numpy(zero_copy_only=False)) if c.null_count else c.to_numpy()
    m = f(a)
    if c.null_count:
        m = m & ~np.asarray(c.is_null())
    return m


@pytest.mark.parametrize("pushed,pred", PUSHED_CASES)
def test_read_file_is_a_superset_of_matching_rows_and_exact_without_nulls(path, pushed, pred):
    cols = ["p", "id"]                                  # filter column 'd' is not in the output
    got = iop._read_file(path, cols, [], pushed)
    assert got.column_names == cols
    expected = _expect(path, cols, pred)
    got_ids = set(got.column("id").to_pylist())
    assert set(expected.column("id").to_pylist()) <= got_ids     # never drops a matching row
    if pq.read_table(path).column("d").null_count == 0:
        assert sorted(got.column("id").to_pylist()) == sorted(expected.column("id").to_pylist())


def test_rows_keep_file_order(path):
    got = iop._read_file(path, ["id"], [], (("d", "ge", 8500),))
    ids = got.column("id").to_pylist()
    assert ids == sorted(ids)


def test_everything_filtered_out_keeps_schema(path):
    got = iop._read_file(path, ["p", "d"], [], (("id", "ge", 10**9),))
    assert got.num_rows == 0 and got.column_names == ["p", "d"]
    got = iop._read_file(path, ["p", "d"], [], (("d", "gt", 20000),))
    assert got.num_rows == 0 and got.column_names == ["p", "d"]


def test_string_columns_and_dictionaries_survive_row_filtering(path):
    got = iop._read_file(path, ["f", "id"], ["f"], (("d", "lt", 9000),))
    ref = pq.read_table(path)
    keep = _np(ref, "d", lambda a: a < 9000)
    expect_ids = set(ref.column("id").to_pylist()[i] for i in np.flatnonzero(keep))
    assert expect_ids <= set(got.column("id").to_pylist())
    assert pa.types.is_dictionary(got.schema.field("f").type)
    ids = got.column("id").to_pylist()
    fs = got.column("f").to_pylist()
    full = dict(zip(ref.column("id").to_pylist(), ref.column("f").to_pylist()))
    assert all(full[i] == f for i, f in zip(ids, fs))


@pytest.mark.parametrize("threads", ["1", "3"])
def test_thread_count_does_not_change_results(path, monkeypatch, threads):
    base = iop._read_file(path, ["id", "p"], [], (("d", "ge", 9000),))
    monkeypatch.setenv("MXFRAME_READ_THREADS", threads)
    assert iop._read_file(path, ["id", "p"], [], (("d", "ge", 9000),)).equals(base)


def test_query_results_match_pyarrow_reference(path):
    lf = mx.scan_parquet(path).filter((col("d") >= 9000) & (col("d") < 9400)).select(col("id"), col("p"))
    got = lf.compute(device="cpu")
    ref = pq.read_table(path)
    d = ref.column("d")
    keep = [v is not None and 9000 <= v < 9400 for v in d.to_pylist()]
    assert sorted(got.column("id").to_pylist()) == sorted(np.asarray(ref.column("id"))[np.array(keep)].tolist())


def test_filter_reading_does_not_import_pandas():
    root = Path(__file__).resolve().parent
    code = (f"import sys; sys.path.insert(0, {str(root)!r}); import conftest\n"
            "import tempfile, os, numpy as np, pyarrow as pa, pyarrow.parquet as pq\n"
            "import mxframe as mx\n"
            "from mxframe import col\n"
            "d = tempfile.mkdtemp(); p = os.path.join(d, 't.parquet')\n"
            "from mxframe.np_arrow import array as A\n"
            "pq.write_table(pa.table({'a': A(np.arange(4000, dtype=np.int64)), 'b': A(np.ones(4000, np.float32))}), p, row_group_size=500)\n"
            "import mxframe.io_parquet as iop; iop.MIN_ROWS_FOR_ROW_FILTER = 0; iop._estimated_pass = lambda *a: 0.0\n"
            "r = mx.scan_parquet(p).filter(col('a') >= 100).select(col('b')).compute(device='cpu')\n"
            "assert r.num_rows == 3900\n"
            "assert iop._read_file(p, ['b'], [], (('a', 'ge', 100),)).num_rows >= 3900\n"
            "for m in ('pandas', 'pyarrow.dataset', 'concurrent.futures'):\n"
            "    assert m not in sys.modules, m + ' imported'\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]


def test_covers_and_estimates():
    assert iop._covers(5, 9, "ge", 5) and not iop._covers(5, 9, "gt", 5)
    assert iop._covers(5, 9, "lt", 10) and not iop._covers(5, 9, "le", 8)
    assert iop._covers(7, 7, "eq", 7) and not iop._covers(5, 9, "eq", 7)
    assert iop._covers(5, 9, "ne", 11) and not iop._covers(5, 9, "ne", 7)
    assert iop._covers(7, 7, "isin", (1, 7)) and not iop._covers(5, 9, "isin", (7,))


def test_unselective_filters_are_left_to_the_engine(tmp_path, monkeypatch):
    p = str(tmp_path / "u.parquet")
    pq.write_table(_table(), p, row_group_size=500)
    calls = []
    monkeypatch.setattr(iop, "_estimated_pass", _REAL_ESTIMATE)
    monkeypatch.setattr(iop, "_read_groups", lambda *a, **k: calls.append(1) or (_ for _ in ()).throw(AssertionError))
    got = iop._read_file(p, ["id"], [], (("d", "ge", 8010),))       # ~99% of rows survive
    assert not calls and got.num_rows == 6000


def test_small_reads_use_the_plain_reader(tmp_path, monkeypatch):
    monkeypatch.setattr(iop, "MIN_ROWS_FOR_ROW_FILTER", 10**9)
    p = str(tmp_path / "s.parquet")
    pq.write_table(_table(), p, row_group_size=500)
    monkeypatch.setattr(iop, "_read_groups", lambda *a, **k: (_ for _ in ()).throw(AssertionError))
    assert iop._read_file(p, ["id"], [], (("d", "ge", 9500),)).num_rows == 6000


def test_footer_cache_refreshes_when_file_is_rewritten(tmp_path):
    p = str(tmp_path / "r.parquet")
    pq.write_table(_table(n=1000), p, row_group_size=500)
    assert iop._metadata(p).num_rows == 1000
    pq.write_table(_table(n=2000), p, row_group_size=500)
    assert iop._metadata(p).num_rows == 2000
    assert iop._read_file(p, ["id"], [], ()).num_rows == 2000


def test_bad_thread_env_is_ignored(monkeypatch):
    monkeypatch.setenv("MXFRAME_READ_THREADS", "lots")
    assert iop._read_threads() >= 2


def test_worker_exception_propagates():
    def boom(x):
        if x == 3:
            raise ValueError("boom")
        return x
    with pytest.raises(ValueError, match="boom"):
        iop._parallel_map(boom, range(8), 4)
    assert iop._parallel_map(lambda x: x * 2, range(5), 3) == [0, 2, 4, 6, 8]


@pytest.mark.parametrize("dtype", [np.int8, np.uint64])
def test_narrow_and_unsigned_columns(tmp_path, dtype):
    p = str(tmp_path / "n.parquet")
    n = 2000
    a = (np.arange(n) % 100).astype(dtype)
    pq.write_table(pa.table({"k": pa.array(a), "id": pa.array(np.arange(n))}), p, row_group_size=250)
    for op, v, f in [("ge", 90, a >= 90), ("lt", 5, a < 5), ("eq", 7, a == 7), ("isin", (3, 4), np.isin(a, [3, 4]))]:
        got = iop._read_file(p, ["id"], [], (("k", op, v if op != "isin" else v),))
        assert set(np.flatnonzero(f)) <= set(got.column("id").to_pylist())


@pytest.mark.parametrize("value", [2**70, -2**70, 2**63, -2**63 - 1])
def test_out_of_range_literals_never_drop_rows(tmp_path, value):
    p = str(tmp_path / "o.parquet")
    a = np.arange(2000, dtype=np.int32)
    pq.write_table(pa.table({"k": pa.array(a), "id": pa.array(a.astype(np.int64))}), p, row_group_size=250)
    for op in ("eq", "ne", "lt", "le", "gt", "ge"):
        got = iop._read_file(p, ["id"], [], (("k", op, value),))
        ref = {"eq": a == value, "ne": a != value, "lt": a < value, "le": a <= value,
               "gt": a > value, "ge": a >= value}[op]
        assert set(np.flatnonzero(ref)) <= set(got.column("id").to_pylist())


def test_isin_mixing_huge_and_negative_values(tmp_path):
    p = str(tmp_path / "i.parquet")
    a = np.arange(-500, 1500, dtype=np.int64)
    pq.write_table(pa.table({"k": pa.array(a), "id": pa.array(a)}), p, row_group_size=250)
    got = iop._read_file(p, ["id"], [], (("k", "isin", (-3, 5, 2**64)),))
    assert {-3, 5} <= set(got.column("id").to_pylist())
