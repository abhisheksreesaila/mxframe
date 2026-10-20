"""F-011 engine correctness scenarios (run by tests/test_f011_correctness.py).

    python tests/_f011_scenarios.py <scenario> <device>

Each scenario asserts internally and exits non-zero on failure.  The CPU device runs
in-process under pytest; ``gpu`` and ``auto`` run in a subprocess so a GPU fault fails
one test instead of the session.
"""
import contextlib
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import conftest  # noqa: F401,E402  (makes `import omxcore` resolve to this checkout)

import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.compute as pc  # noqa: E402

from omxcore import LazyFrame, col, lit, when  # noqa: E402
from omxcore.lazy_frame import Scan  # noqa: E402


def _lf(t):
    return LazyFrame(Scan(t))


@contextlib.contextmanager
def _env(**kv):
    old = {k: os.environ.get(k) for k in kv}
    os.environ.update(kv)
    try:
        yield
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def count_exact_beyond_2p24(device):
    """A group of 17M rows counts exactly (float32 would stop at 16,777,216), as int64."""
    n = (1 << 24) + 1
    t = pa.table({"k": pa.array(np.zeros(n, dtype=np.int32)),
                  "v": pa.array(np.ones(n, dtype=np.float32))})
    out = _lf(t).groupby("k").agg(col("v").count().alias("n")).compute(device=device)
    assert out.schema.field("n").type == pa.int64(), out.schema
    assert out.column("n").to_pylist() == [n], out.column("n").to_pylist()


def count_exact_beyond_2p24_unfused(device):
    with _env(MXFRAME_FUSED_PIPELINE="0"):
        count_exact_beyond_2p24(device)


def count_types_all_paths(device):
    """Grouped and global counts are int64 whatever path served them, incl. empty results."""
    t = pa.table({"k": pa.array([1, 1, 2, 2, 2], type=pa.int32()),
                  "v": pa.array([1, 2, 3, 4, 5], type=pa.float32())})
    for fused in ("1", "0"):
      with _env(MXFRAME_FUSED_PIPELINE=fused):
            out = _lf(t).groupby("k").agg(col("v").count().alias("n")).sort("k").compute(device=device)
            assert out.schema.field("n").type == pa.int64(), (fused, out.schema)
            assert out.column("n").to_pylist() == [2, 3]
            out = (_lf(t).filter(col("v") > lit(100.0)).groupby("k")
                   .agg(col("v").count().alias("n")).compute(device=device))
            assert out.num_rows == 0 and out.schema.field("n").type == pa.int64(), (fused, out.schema)
            out = _lf(t).groupby().agg(col("v").count().alias("n")).compute(device=device)
            assert out.schema.field("n").type == pa.int64() and out.column("n").to_pylist() == [5]


# --- nulls --------------------------------------------------------------------------------

def _nulls_table():
    return pa.table({"k": pa.array([1, 1, 2, None, None], pa.int32()),
                     "v": pa.array([1.0, None, 3.0, 4.0, 5.0], pa.float64()),
                     "s": pa.array(["a", None, "b", "a", None])})


def _rows(t):
    names = t.column_names
    return sorted(zip(*[t.column(n).to_pylist() for n in names]), key=lambda r: tuple(str(x) for x in r))


def nulls_filter_drops_null_comparisons(device):
    """A comparison with a null is not true: the row is filtered out, also in front of an aggregate."""
    t = _nulls_table()
    out = _lf(t).filter(col("v") > lit(2.0)).groupby("k").agg(col("v").sum().alias("s")).compute(device=device)
    assert _rows(out) == [(2, 3.0), (None, 9.0)], _rows(out)
    out = _lf(t).filter(col("k") == lit(1)).groupby().agg(col("v").count().alias("n")).compute(device=device)
    assert out.column("n").to_pylist() == [1], out.column("n").to_pylist()


def nulls_group_key_is_its_own_group(device):
    """Rows with a null group key form one group (never silently dropped); aggregates skip null values."""
    t = _nulls_table()
    out = (_lf(t).groupby("k").agg(col("v").sum().alias("s"), col("v").count().alias("n"),
                                   col("v").mean().alias("m"), col("v").min().alias("lo"),
                                   col("v").max().alias("hi")).compute(device=device))
    assert _rows(out) == [(1, 1.0, 1, 1.0, 1.0, 1.0), (2, 3.0, 1, 3.0, 3.0, 3.0),
                          (None, 9.0, 2, 4.5, 4.0, 5.0)], _rows(out)
    out = _lf(t).groupby("s").agg(col("v").sum().alias("s2")).compute(device=device)
    assert _rows(out) == [(None, 5.0), ("a", 5.0), ("b", 3.0)], _rows(out)


def nulls_global_aggregates_skip_nulls(device):
    t = _nulls_table()
    out = _lf(t).groupby().agg(col("v").sum().alias("s"), col("v").count().alias("n"),
                               col("v").mean().alias("m")).compute(device=device)
    assert out.column("s").to_pylist() == [13.0]
    assert out.column("n").to_pylist() == [4]
    assert out.column("m").to_pylist() == [3.25]


def string_ne_with_null_keys(device):
    """filter(s != "a") with null keys: null rows are dropped (SQL), nothing raises."""
    t = _nulls_table()
    out = _lf(t).filter(col("s") != lit("a")).compute(device=device)
    assert out.column("s").to_pylist() == ["b"], out.column("s").to_pylist()
    out = _lf(t).filter(col("s") != lit("a")).groupby("s").agg(col("v").sum().alias("x")).compute(device=device)
    assert _rows(out) == [("b", 3.0)], _rows(out)


def sort_with_null_keys(device):
    """Nulls sort last in either direction (DuckDB's default), instead of raising."""
    t = _nulls_table()
    out = _lf(t).sort("k").compute(device=device)
    assert out.column("k").to_pylist()[:3] == [1, 1, 2] and out.column("k").to_pylist()[3:] == [None, None]
    out = _lf(t).sort("s", descending=True).compute(device=device)
    assert out.column("s").to_pylist() == ["b", "a", "a", None, None], out.column("s").to_pylist()


# --- limit / sort / joins / caches ------------------------------------------------------------

def limit_below_aggregate_is_applied(device):
    t = pa.table({"g": pa.array([1, 1, 2, 2, 3], pa.int32()), "v": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], pa.float32())})
    out = _lf(t).limit(3).groupby("g").agg(col("v").sum().alias("s")).sort("g").compute(device=device)
    assert _rows(out) == [(1, 3.0), (2, 3.0)], _rows(out)          # only the first 3 rows are aggregated
    out = _lf(t).limit(2).groupby().agg(col("v").sum().alias("s")).compute(device=device)
    assert out.column("s").to_pylist() == [3.0]
    out = _lf(t).filter(col("v") > lit(1.5)).limit(2).groupby().agg(col("v").count().alias("n")).compute(device=device)
    assert out.column("n").to_pylist() == [2]                        # limit applies after the filter


def multi_key_sort_order(device):
    """Two- and three-key sorts, every direction mix, key ranges small and huge (the mixed-radix
    key overflows int32 at 1e5 x 1e5), strings, and exact float64 payload columns."""
    for n, ra, rb in ((5000, 5, 1000), (200_000, 100_000, 100_000)):
        rng = np.random.default_rng(3)
        a = rng.integers(0, ra, n).astype(np.int32)
        b = rng.integers(0, rb, n).astype(np.int32)
        x = 16_777_217.0 + np.arange(n, dtype=np.float64)              # not representable in float32
        t = pa.table({"a": pa.array(a), "b": pa.array(b), "x": pa.array(x)})
        for desc in ([False, False], [False, True], [True, False], [True, True]):
            out = _lf(t).sort("a", "b", descending=desc).compute(device=device)
            sa = -a.astype(np.int64) if desc[0] else a.astype(np.int64)
            sb = -b.astype(np.int64) if desc[1] else b.astype(np.int64)
            order = np.lexsort((np.arange(n), sb, sa))                 # stable: ties keep input order
            got = list(zip(out.column("a").to_pylist(), out.column("b").to_pylist()))
            want = list(zip(a[order].tolist(), b[order].tolist()))
            assert got == want, (n, desc, got[:5], want[:5])
            assert out.column("x").to_pylist() == x[order].tolist(), (n, desc, "float64 payload")
    s1 = np.array(["b", "a", "c", "a", "b", "c", "a"])
    k2 = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    t = pa.table({"s": pa.array(s1), "k": pa.array(k2)})
    out = _lf(t).sort("s", "k", descending=[False, True]).compute(device=device)
    assert list(zip(out.column("s").to_pylist(), out.column("k").to_pylist())) == \
        [("a", 7), ("a", 4), ("a", 2), ("b", 5), ("b", 1), ("c", 6), ("c", 3)]


def join_keys_of_different_integer_widths(device):
    """int64 key joined to an int32 key: both are cast to a common type instead of raising."""
    a = pa.table({"k": pa.array([1, 2, 3, 3], pa.int64()), "x": pa.array([10, 20, 30, 31], pa.int32())})
    b = pa.table({"k": pa.array([3, 1, 4], pa.int32()), "y": pa.array([1.5, 2.5, 3.5], pa.float32())})
    out = _lf(a).join(_lf(b), on="k").compute(device=device)
    assert _rows(out.select(["k", "x", "y"])) == [(1, 10, 2.5), (3, 30, 1.5), (3, 31, 1.5)], _rows(out)
    out = _lf(b).join(_lf(a), left_on="k", right_on="k").compute(device=device)
    assert out.num_rows == 3


def join_cache_is_keyed_by_device(device):
    """A join computed for one device is not reused by a query on another (GPU gathers may round)."""
    from omxcore import custom_ops
    big = 16_777_217.0                                               # not representable in float32
    left = pa.table({"k": pa.array(np.arange(100, dtype=np.int32)), "v": pa.array(np.full(100, big))})
    right = pa.table({"k": pa.array(np.arange(100, dtype=np.int32)), "w": pa.array(np.ones(100, np.float32))})
    custom_ops._JOIN_RESULT_CACHE.clear()
    for dev in ("gpu", "cpu"):
        out = _lf(left).join(_lf(right), on="k").compute(device=dev)
        assert set(out.column("v").to_pylist()) == {big}, (dev, set(out.column("v").to_pylist()))
    devices = {key[-1] for key in custom_ops._JOIN_RESULT_CACHE}
    assert devices == {"gpu", "cpu"}, devices


# --- expressions / kernels -------------------------------------------------------------------

def case_when_sums_are_float64(device):
    """A case_when aggregate argument is evaluated in float64, not float32 (the legacy GPU
    kernels are float32 by design; the fused GPU pipeline is float64)."""
    n = 6
    t = pa.table({"g": pa.array([0] * n, pa.int32()),
                  "big": pa.array([16_777_217.0] * n, pa.float64()),
                  "f": pa.array([1] * n, pa.int32())})
    for fused in ("1", "0") if device == "cpu" else ("1",):
        with _env(MXFRAME_FUSED_PIPELINE=fused):
            out = (_lf(t).groupby("g")
                   .agg(when(col("f") == lit(1), col("big"), lit(0.0)).sum().alias("s")).compute(device=device))
            assert out.schema.field("s").type == pa.float64()
            assert out.column("s").to_pylist() == [16_777_217.0 * n], (fused, out.column("s").to_pylist())


def min_max_identity_with_infinities(device):
    """A group whose only values are +inf (or -inf) reports that value for both min and max."""
    inf = float("inf")
    t = pa.table({"g": pa.array([0, 0, 1, 1, 2], pa.int32()),
                  "v": pa.array([inf, inf, -inf, -inf, 1.0], pa.float32())})
    for fused in ("1", "0"):
        with _env(MXFRAME_FUSED_PIPELINE=fused):
            out = (_lf(t).groupby("g").agg(col("v").min().alias("lo"), col("v").max().alias("hi"))
                   .sort("g").compute(device=device))
            assert out.column("lo").to_pylist() == [inf, -inf, 1.0], (fused, out.column("lo").to_pylist())
            assert out.column("hi").to_pylist() == [inf, -inf, 1.0], (fused, out.column("hi").to_pylist())


def float32_column_keeps_boundary_rows(device):
    """`d >= 0.05 and d <= 0.07` on a float32 column keeps the rows that equal the literals
    (the column is compared in its own precision, as DuckDB compares a DECIMAL), on every path."""
    d = np.round(np.array([0.04, 0.05, 0.06, 0.07, 0.08] * 4, dtype=np.float64), 2).astype(np.float32)
    t = pa.table({"g": pa.array(np.array([0, 1] * 10, dtype=np.int32)), "d": pa.array(d),
                  "p": pa.array(np.ones(20, dtype=np.float32))})
    pred = (col("d") >= lit(0.05)) & (col("d") <= lit(0.07))
    for fused in ("1", "0"):
        with _env(MXFRAME_FUSED_PIPELINE=fused):
            out = _lf(t).filter(pred).groupby().agg(col("p").sum().alias("n")).compute(device=device)
            assert out.column("n").to_pylist() == [12.0], (fused, "global", out.column("n").to_pylist())
            out = (_lf(t).filter(pred).groupby("g").agg(col("p").sum().alias("n")).sort("g")
                   .compute(device=device))
            assert out.column("n").to_pylist() == [6.0, 6.0], (fused, "grouped", out.column("n").to_pylist())
    out = _lf(t).filter(pred).compute(device=device)
    assert out.num_rows == 12, out.num_rows
    out = _lf(t).filter(col("d") == lit(0.06)).compute(device=device)
    assert out.num_rows == 4, out.num_rows
    out = _lf(t).filter(col("d") != lit(0.06)).compute(device=device)
    assert out.num_rows == 16, out.num_rows


def gpu_scratch_does_not_reserve_the_device(device):
    """The first Mojo-side buffer (the recursive scan makes one at >= 1M rows) must not make the
    kernel library's memory manager reserve ~90% of the GPU: at SF10 our own cuMemAlloc calls
    (uploads, join outputs, gathers) then ran out of memory."""
    if device == "cpu":
        return
    import ctypes
    from omxcore.aot_kernels import get_aot_gpu
    g = get_aot_gpu()
    assert g is not None
    lib = g._cu._lib
    free, total = ctypes.c_size_t(), ctypes.c_size_t()

    def used_bytes():
        lib.cuMemGetInfo_v2(ctypes.byref(free), ctypes.byref(total))
        return total.value - free.value

    before = used_bytes()
    n = 4_000_000
    src = g._alloc(n * 4)
    g._fill_i32(src, n, 1)
    out = g._alloc(n * 4)
    assert g._scan_i32(out, src, n) == n
    g._free(src, out)
    grown = used_bytes() - before
    assert grown < 0.5 * total.value, f"the scan made the device grow by {grown >> 20} MiB of {total.value >> 20}"


def join_duplicate_names_get_a_right_suffix(device):
    """Non-key columns with the same name on both sides: the right one is renamed `_right`
    (`_right2`, ... when taken, also by another right column), for inner and left joins, so the result can be queried again."""
    a = pa.table({"k": pa.array([1, 2, 3], pa.int32()), "v": pa.array([10.0, 20.0, 30.0])})
    b = pa.table({"k": pa.array([1, 2, 4], pa.int32()), "v": pa.array([1.0, 2.0, 4.0]),
                  "v_right": pa.array([7.0, 8.0, 9.0])})
    out = _lf(a).join(_lf(b), on="k").compute(device=device)
    assert out.column_names == ["k", "v", "v_right2", "v_right"], out.column_names
    assert _rows(out) == [(1, 10.0, 1.0, 7.0), (2, 20.0, 2.0, 8.0)], _rows(out)
    out = _lf(a).join(_lf(b), on="k", how="left").compute(device=device)
    assert out.column_names == ["k", "v", "v_right2", "v_right"], out.column_names
    assert sorted(out.column("v_right2").to_pylist(), key=str) == [1.0, 2.0, None]
    twice = _lf(a).join(_lf(a), on="k").join(_lf(a), on="k").compute(device=device)
    assert twice.column_names == ["k", "v", "v_right", "v_right2"], twice.column_names
    out = (_lf(a).join(_lf(a), on="k").filter(col("v_right") > lit(15.0))
           .compute(device=device))                    # the filter must not be pushed into either side wrongly
    assert _rows(out) == [(2, 20.0, 20.0), (3, 30.0, 30.0)], _rows(out)
    out = (_lf(a).join(_lf(a), on="k").filter(col("v") > lit(15.0)).compute(device=device))
    assert _rows(out) == [(2, 20.0, 20.0), (3, 30.0, 30.0)], _rows(out)


def self_join_then_aggregate(device):
    t = pa.table({"k": pa.array([1, 1, 2, None], pa.int32()), "v": pa.array([1.0, 2.0, None, 4.0])})
    j = _lf(t).join(_lf(t), on="v")
    out = j.groupby().agg(col("v").sum().alias("s"), col("k_right").count().alias("n")).compute(device=device)
    assert out.column("s").to_pylist() == [1.0 + 2.0 + 4.0], out.to_pydict()
    t2 = pa.table({"k": pa.array([1, 1, 2], pa.int32()), "j": pa.array([1, 2, 3], pa.int32()),
                   "v": pa.array([1.0, 2.0, 3.0])})
    out = (_lf(t2).join(_lf(t2), on="k").groupby("k")
           .agg(col("v_right").sum().alias("s")).sort("k").compute(device=device))
    assert _rows(out) == [(1, 6.0), (2, 3.0)], _rows(out)
    t3 = pa.table({"k": pa.array([1, 1, 2], pa.int32()), "v": pa.array([1.0, None, 3.0])})
    out = (_lf(t3).join(_lf(t3), on="k").groupby("k")
           .agg(col("v_right").sum().alias("s"), col("v").count().alias("n")).sort("k").compute(device=device))
    assert _rows(out) == [(1, 2.0, 2), (2, 3.0, 1)], _rows(out)


def literal_aggregates_with_nulls(device):
    """Only count(*) is a row count; sum / max of a literal broadcast the literal."""
    t = pa.table({"k": pa.array([1, 1, None, None, None], pa.int32()), "v": pa.array([1.0, None, 3.0, 4.0, 5.0])})
    out = (_lf(t).groupby("k").agg(lit(1.0).sum().alias("s"), lit(5.0).max().alias("m"),
                                   lit(1).count().alias("n")).compute(device=device))
    assert _rows(out) == [(1, 2.0, 5.0, 2), (None, 3.0, 5.0, 3)], _rows(out)
    out = _lf(t).groupby().agg(lit(2.0).sum().alias("s"), lit(1).count().alias("n")).compute(device=device)
    assert out.column("s").to_pylist() == [10.0] and out.column("n").to_pylist() == [5]


SCENARIOS = {k: v for k, v in globals().items()
             if callable(v) and not k.startswith("_") and getattr(v, "__module__", None) == __name__}

if __name__ == "__main__":
    name, device = sys.argv[1], sys.argv[2]
    SCENARIOS[name](device)
    print("ok")
