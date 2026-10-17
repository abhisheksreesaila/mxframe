"""F-060: every kernel that adds an integer sum does it in int64: the native executor (direct, compacted and hashed
groups, the fused join), the older fused pipeline (per-worker partials, the shared atomic output, the table of
sub-totals, the staged many-groups path) and the GPU (per-thread, shared-table, atomic and ordered modes). Each shape
is checked against DuckDB's exact total, with values whose sums pass 2^53 (so a float64 accumulator is wrong), and
a total past int64 raises OverflowError on each of them."""
import duckdb
import numpy as np
import pyarrow as pa
import pytest

import mxframe.native_exec as nx
from mxframe import LazyFrame, col, gpu_stream, lit
from mxframe.np_arrow import array as A

HAS_GPU = gpu_stream.cuda_available()

# (rows, groups): a global sum, a few groups, a thousand, many (the shared atomic output: ~1 row per group),
# and more than the table path holds (the staged path, in reproducible mode)
SHAPES = {
    "global": (1_500_000, 1),
    "few": (1_500_000, 7),
    "thousands": (1_500_000, 1000),
    "many": (1_200_000, 1_000_000),
    "staged": (400_000, 200_000),
}


def _table(rows, groups, hashed=False, seed=60):
    """Values of about 2^60 / (rows per group), so a group's total is about 2^60 (past 2^53, inside int64) unless the
    values would have to pass 2^53 themselves (the shapes with ~1 row per group exercise the paths, not the digits)."""
    r = np.random.default_rng(seed)
    per_group = max(rows // groups, 1)
    lo = min(1 << (60 - int(np.ceil(np.log2(per_group)))), 1 << 52)
    v = r.integers(lo, 2 * lo, rows).astype(np.int64)
    v = np.where(r.random(rows) < 0.2, -v, v)                    # both signs: partial totals cancel
    g = (r.permutation(rows) % groups).astype(np.int64)
    cols = {"v": A(v), "w": A(r.integers(-1000, 1000, rows).astype(np.int32)), "g": A(g)}
    if hashed:
        cols["g"] = A(g.astype(np.float64) * 0.5 + 0.25)         # float keys: hashed groups
    return pa.table(cols)


def _want(t, keyed):
    con = duckdb.connect()
    con.register("t", t)
    if not keyed:
        return [(None, int(a), int(b)) for a, b in con.execute("select sum(v), sum(w) from t").fetchall()]
    return [(g, int(a), int(b)) for g, a, b in con.execute(
        "select g, sum(v), sum(w) from t group by g order by g").fetchall()]


def _run(t, keyed, device, **kw):
    lf = LazyFrame(t).groupby("g") if keyed else LazyFrame(t).groupby()
    lf = lf.agg(col("v").sum().alias("sv"), col("w").sum().alias("sw"))
    if keyed:
        lf = lf.sort("g")
    got = lf.compute(device=device, **kw)
    assert got.schema.field("sv").type == pa.int64() and got.schema.field("sw").type == pa.int64()
    rows = got.to_pylist()
    return [(r.get("g"), r["sv"], r["sw"]) for r in rows], lf.last_compile_provenance


MODES = [
    pytest.param("1", "cpu", False, id="native"),
    pytest.param("1", "cpu", True, id="native-repro"),
    pytest.param("0", "cpu", False, id="pipeline"),
    pytest.param("0", "cpu", True, id="pipeline-repro"),
    pytest.param("0", "gpu", False, id="gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
    pytest.param("0", "gpu", True, id="gpu-repro", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
]


@pytest.mark.parametrize("hashed", [False, True], ids=["direct", "hashed"])
@pytest.mark.parametrize("shape", sorted(SHAPES))
@pytest.mark.parametrize("native,device,repro", MODES)
def test_the_sums_equal_duckdbs_exact_totals(monkeypatch, native, device, repro, shape, hashed):
    rows, groups = SHAPES[shape]
    if hashed and (groups == 1 or device == "gpu"):
        pytest.skip("hashed groups: the executor's float keys")
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    t = _table(rows, groups, hashed)
    keyed = groups > 1
    want = _want(t, keyed)
    got, prov = _run(t, keyed, device, reproducible=repro)
    assert got == want
    if rows // groups > 100:                    # a group's total passes 2^53: float64 would have rounded it
        assert max(abs(w[1]) for w in want) > 2 ** 53
    if native == "1" and device == "cpu":
        assert prov["path"] == "native", prov
    elif device == "gpu" and not repro:
        assert prov["path"] == "fused_pipeline_gpu", prov                    # (modes 0, 3, 1 and 2 by the group count)
    elif device == "gpu" and shape in ("global", "few", "thousands"):
        assert prov["path"] == "fused_pipeline_gpu", prov                    # the ordered mode 4 holds these


# -- a total past int64 raises on every kernel ---------------------------------------------------------

def _big(rows, groups):
    v = np.full(rows, 1 << 51, dtype=np.int64)
    return pa.table({"v": A(v), "g": A((np.arange(rows) % groups).astype(np.int64))})


@pytest.mark.parametrize("shape", ["global", "few", "thousands"])
@pytest.mark.parametrize("native,device,repro", MODES)
def test_a_total_past_int64_raises_on_every_kernel(monkeypatch, native, device, repro, shape):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    groups = SHAPES[shape][1]
    rows = 6000 * max(groups, 1) if groups > 1 else 6000                  # every group holds 6000 x 2^51 = 2^63.5
    t = _big(rows, groups)
    lf = LazyFrame(t).groupby("g") if groups > 1 else LazyFrame(t).groupby()
    with pytest.raises(OverflowError, match="sum of INT64"):
        lf.agg(col("v").sum().alias("s")).compute(device=device, reproducible=repro)


@pytest.mark.parametrize("native,device,repro", MODES)
def test_the_largest_total_that_fits_is_exact_on_every_kernel(monkeypatch, native, device, repro):
    monkeypatch.setenv("MXFRAME_NATIVE", native)
    if native == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    rows = 4095                                                      # 4095 x 2^51 = 2^63 - 2^51
    t = pa.table({"v": A(np.full(rows, 1 << 51, dtype=np.int64)), "g": A(np.zeros(rows, dtype=np.int64))})
    for lf in (LazyFrame(t).groupby().agg(col("v").sum().alias("s")), LazyFrame(t).groupby("g").agg(col("v").sum().alias("s"))):
        got = lf.compute(device=device, reproducible=repro)
        assert got.schema.field("s").type == pa.int64() and got.column("s").to_pylist() == [rows * (1 << 51)]


# -- the fused join + aggregate -------------------------------------------------------------------------

def test_the_fused_join_sums_in_int64(monkeypatch):
    monkeypatch.setenv("MXFRAME_NATIVE", "1")
    if not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    rows = 1_200_000
    r = np.random.default_rng(3)
    fact = pa.table({"k": A((np.arange(rows) % 50).astype(np.int64)), "v": A(r.integers(1 << 40, 1 << 41, rows).astype(np.int64))})
    dim = pa.table({"k": A(np.arange(50, dtype=np.int64)), "g": A((np.arange(50) % 5).astype(np.int64)),
                    "bonus": A(np.arange(50, dtype=np.int64) << 30)})
    con = duckdb.connect()
    con.register("fact", fact)
    con.register("dim", dim)
    lf = (LazyFrame(fact).join(LazyFrame(dim), on="k").groupby("g")
          .agg(col("v").sum().alias("sv"), col("bonus").sum().alias("sb")).sort("g"))
    got = lf.compute(device="cpu")
    assert got.schema.field("sv").type == pa.int64() and got.schema.field("sb").type == pa.int64()
    want = con.execute("select g, sum(v), sum(bonus) from fact join dim using (k) group by g order by g").fetchall()
    assert [(r["g"], r["sv"], r["sb"]) for r in got.to_pylist()] == [(g, int(a), int(b)) for g, a, b in want]
