"""F-011: every TPC-H query gives the same answer as DuckDB, on cpu, gpu and auto (and the
pandas / Polars baseline versions at SF1).

The reference is DuckDB running the official TPC-H SQL over the very same Parquet files
(day-number dates and float32 decimals are turned back into DATE and DECIMAL(15,2) by
views, see tpch_duckdb_sql.py).  The mxframe side is ``scripts/benchmark_tpch.py``
``run_qN_mxframe`` on tables from ``mx.read_parquet_table``, run in a fresh process per
query so a GPU fault fails one test instead of the session.

Comparison: same row count; strings, integers and counts exactly (a count that comes
back as a float fails); floats to a relative 1e-6.  Rows are matched after sorting on
their non-float columns.  For queries with a LIMIT, rows tied with the last kept row on
the ORDER BY column may legitimately differ and are dropped from both sides.

SF1 always (when the data is present); SF10 with MXFRAME_TPCH_SF10=1 (60M rows, slow).
"""
import math
import os
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pytest

from conftest import TPCH_DATA
import tpch_duckdb_sql as ref

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "_tpch_runner.py"


def _data_dir(sf):
    """`tpch_sf<SF>_exact` (decimals rounded to the nearest float32, see gen_tpch_parquet.py) if
    present, else the older `tpch_sf<SF>` whose decimals were truncated (Q6 cannot match there)."""
    exact = TPCH_DATA / f"tpch_sf{sf}_exact"
    return exact if (exact / "lineitem.parquet").exists() else TPCH_DATA / f"tpch_sf{sf}"


SF_DIRS = {"1": _data_dir("1"), "10": _data_dir("10")}
REL_TOL = 1e-6
DEVICES = ["cpu", "gpu", "auto"]


def _sf_params():
    out = []
    for sf, path in SF_DIRS.items():
        marks = [pytest.mark.slow,
                 pytest.mark.skipif(not (path / "lineitem.parquet").exists(),
                                    reason=f"TPC-H SF{sf} data not present")]
        if sf == "10":
            marks.append(pytest.mark.skipif(os.environ.get("MXFRAME_TPCH_SF10") != "1",
                                            reason="set MXFRAME_TPCH_SF10=1 to run SF10"))
        out.append(pytest.param(sf, marks=marks, id=f"sf{sf}"))
    return out


@pytest.fixture(scope="module")
def duck():
    cons = {}

    def get(sf):
        if sf not in cons:
            cons[sf] = ref.connect(SF_DIRS[sf])
        return cons[sf]
    return get


_reference_cache = {}


def reference(duck, sf, q):
    if (sf, q) not in _reference_cache:
        _reference_cache[(sf, q)] = ref.run(duck(sf), q)
    return _reference_cache[(sf, q)]


def run_mxframe(sf, q, device, tmp_path):
    out = tmp_path / f"q{q}_{device}.arrow"
    env = dict(os.environ)
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    proc = subprocess.run([sys.executable, str(RUNNER), str(SF_DIRS[sf]), str(q), device, str(out)],
                          capture_output=True, text=True, timeout=1800, env=env)
    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout).strip().splitlines()[-6:])
        pytest.fail(f"Q{q} on {device} crashed (exit {proc.returncode}):\n{tail}")
    with pa.OSFile(str(out), "rb") as f:
        return pa.ipc.open_file(f).read_all()


def _is_int(t):
    return pa.types.is_integer(t)


def _is_float(t):
    return pa.types.is_floating(t) or pa.types.is_decimal(t)


def _close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    return math.isclose(float(a), float(b), rel_tol=REL_TOL, abs_tol=1e-9)


def _drop_tied_at_limit(t, order_cols):
    """Drop the rows equal to the last row on every ORDER BY column (the LIMIT boundary tie group)."""
    if t.num_rows == 0:
        return t
    cols = [t.column(c).to_pylist() for c in order_cols]
    edge = [c[-1] for c in cols]
    keep = [not all(_close(c[i], e) if isinstance(e, float) else c[i] == e for c, e in zip(cols, edge))
            for i in range(t.num_rows)]
    return t.filter(pa.array(keep))


def _assert_in_order(got, want, order_cols):
    """The ORDER BY columns of both results agree row by row (so ``got`` is in DuckDB's order)."""
    for c in order_cols:
        g, w = got.column(c).to_pylist(), want.column(c).to_pylist()
        for i, (a, b) in enumerate(zip(g, w)):
            ok = _close(a, b) if isinstance(b, float) else a == b
            assert ok, f"row {i} is out of ORDER BY order in {c}: got {a!r}, DuckDB {b!r}"


def _rows(t, sort_cols):
    cols = [t.column(i).to_pylist() for i in range(t.num_columns)]
    rows = list(zip(*cols))
    return sorted(rows, key=lambda r: tuple("" if r[i] is None else str(r[i]) for i in sort_cols))


def assert_matches_duckdb(got: pa.Table, want: pa.Table, meta: dict, check_names: bool = True, q=None):
    if check_names:
        assert got.column_names == want.column_names, (got.column_names, want.column_names)
    else:
        assert got.num_columns == want.num_columns, (got.column_names, want.column_names)
        got = got.rename_columns(want.column_names)
    assert got.num_rows == want.num_rows, f"rows: got {got.num_rows}, DuckDB {want.num_rows}"
    for name in meta.get("int_cols", []):
        assert _is_int(got.schema.field(name).type), \
            f"{name} must be an integer count, got {got.schema.field(name).type}"
    order_cols = ref.ORDER_BY.get(q, [])
    if order_cols:
        _assert_in_order(got, want, order_cols)
    if "limit" in meta:
        got, want = _drop_tied_at_limit(got, order_cols), _drop_tied_at_limit(want, order_cols)
        assert got.num_rows == want.num_rows, "rows differ outside the LIMIT boundary tie group"
    key_cols = [i for i, t in enumerate(want.schema.types) if not _is_float(t)]
    rows_got, rows_want = _rows(got, key_cols), _rows(want, key_cols)
    for n, (g, w) in enumerate(zip(rows_got, rows_want)):
        for i, (a, b) in enumerate(zip(g, w)):
            name = want.column_names[i]
            if _is_float(want.schema.types[i]):
                assert _close(a, b), f"row {n} {name}: got {a!r}, DuckDB {b!r} (rel tol {REL_TOL})"
            else:
                assert a == b, f"row {n} {name}: got {a!r}, DuckDB {b!r}"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", range(1, 23))
@pytest.mark.parametrize("sf", _sf_params())
def test_query_matches_duckdb(sf, q, device, duck, tmp_path):
    want = reference(duck, sf, q)
    got = run_mxframe(sf, q, device, tmp_path)
    assert_matches_duckdb(got, want, ref.REFERENCE[q][1], q=q)


@pytest.mark.parametrize("sf", _sf_params())
def test_float32_decimals_are_the_nearest_float32(sf):
    """Data check: a decimal stored as the float32 one ulp BELOW its value (0.049999997 for
    0.05) makes `l_discount >= 0.05` drop rows, and TPC-H Q6 comes out 28% low.  Regenerate
    with scripts/gen_tpch_parquet.py if this fails."""
    import numpy as np
    import pyarrow.parquet as pq
    d = pq.read_table(SF_DIRS[sf] / "lineitem.parquet", columns=["l_discount"]).column(0).to_numpy()
    u = np.unique(d)
    nearest = np.round(u.astype(np.float64), 2).astype(np.float32)
    assert (u == nearest).all(), f"not the nearest float32: {u[u != nearest]}"


# Polars sums float32 columns in float32: relative error 1e-4..1e-6 on sums of millions of rows.
# A property of the engine, shown (not hidden) by the scoreboard, so it is expected here.
POLARS_FLOAT32_SUMS = {1, 9}


@pytest.mark.parametrize("engine", ["pandas", "polars"])
@pytest.mark.parametrize("q", range(1, 23))
def test_baseline_engine_matches_duckdb(engine, q, duck, tmp_path, request):
    """The pandas and Polars versions in benchmark_tpch.py (the F-001 scoreboard's baselines)
    express the same TPC-H queries: same answers as DuckDB at SF1."""
    if not (SF_DIRS["1"] / "lineitem.parquet").exists():
        pytest.skip("TPC-H SF1 data not present")
    if engine == "polars":
        pytest.importorskip("polars")
        if q in POLARS_FLOAT32_SUMS:
            request.applymarker(pytest.mark.xfail(strict=True, reason="Polars accumulates float32 sums in float32"))
    want = reference(duck, "1", q)
    got = run_mxframe("1", q, engine, tmp_path)
    if q == 14:            # the baselines return the TPC-H ratio only, mxframe also the two sums it is made of
        want = want.select(["promo_pct"])
    assert_matches_duckdb(got, want, ref.REFERENCE[q][1], check_names=False, q=q)


# --- the comparison itself must be strict, or the gate means nothing -----------------------

def test_compare_rejects_a_float_count():
    want = pa.table({"k": ["a"], "n": pa.array([16777217], pa.int64())})
    got = pa.table({"k": ["a"], "n": pa.array([16777216.0], pa.float64())})
    with pytest.raises(AssertionError):
        assert_matches_duckdb(got, want, {"int_cols": ["n"]})


def test_compare_rejects_a_missing_row_and_a_small_float_error():
    want = pa.table({"k": ["a", "b"], "x": [1.0, 2.0]})
    with pytest.raises(AssertionError):
        assert_matches_duckdb(pa.table({"k": ["a"], "x": [1.0]}), want, {})
    with pytest.raises(AssertionError):
        assert_matches_duckdb(pa.table({"k": ["a", "b"], "x": [1.0, 2.00001]}), want, {})
    assert_matches_duckdb(pa.table({"k": ["b", "a"], "x": [2.0000001, 1.0]}), want, {})


def test_limit_ties_at_the_boundary_are_not_compared(monkeypatch):
    monkeypatch.setitem(ref.ORDER_BY, 99, ["v"])
    want = pa.table({"k": ["a", "b", "c"], "v": [3.0, 2.0, 2.0]})
    got = pa.table({"k": ["a", "b", "z"], "v": [3.0, 2.0, 2.0]})
    assert_matches_duckdb(got, want, {"limit": ("v", True)}, q=99)


def test_ties_on_one_order_key_only_are_still_compared(monkeypatch):
    """The boundary group is the rows tied on ALL order-by keys: a row that differs in the second key is a bug."""
    monkeypatch.setitem(ref.ORDER_BY, 99, ["v", "k"])
    want = pa.table({"k": ["a", "b", "c"], "v": [3.0, 2.0, 2.0]})
    got = pa.table({"k": ["a", "c", "b"], "v": [3.0, 2.0, 2.0]})
    with pytest.raises(AssertionError):
        assert_matches_duckdb(got, want, {"limit": ("v", True)}, q=99)


def test_result_order_is_checked(monkeypatch):
    monkeypatch.setitem(ref.ORDER_BY, 99, ["y", "k"])
    want = pa.table({"y": [2, 2, 1], "k": ["a", "b", "c"]})
    assert_matches_duckdb(pa.table({"y": [2, 2, 1], "k": ["a", "b", "c"]}), want, {}, q=99)
    with pytest.raises(AssertionError):
        assert_matches_duckdb(pa.table({"y": [1, 2, 2], "k": ["c", "a", "b"]}), want, {}, q=99)
    with pytest.raises(AssertionError):
        assert_matches_duckdb(pa.table({"y": [2, 2, 1], "k": ["b", "a", "c"]}), want, {}, q=99)
