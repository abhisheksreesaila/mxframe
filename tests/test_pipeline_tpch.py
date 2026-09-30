"""F-008: TPC-H Q1 / Q6 / Q14 / Q19 through the fused pipeline match DuckDB (CPU and GPU)."""
import sys
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from conftest import TPCH_DATA
from mxframe import custom_ops
from mxframe import pipeline_kernels as pk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark_tpch as bm  # noqa: E402

SFS = [sf for sf in ("sf1", "sf10") if (TPCH_DATA / f"tpch_{sf}" / "lineitem.parquet").exists()]


def _gpu_ok():
    if not pk.gpu_available():
        return False
    try:
        pk._get_cuda()
        return True
    except Exception:
        return False


DEVICES = ["cpu"] + (["gpu"] if _gpu_ok() else [])


@pytest.fixture(scope="module", params=SFS or ["missing"])
def tpch(request):
    if request.param == "missing":
        pytest.skip("TPC-H data not found")
    d = TPCH_DATA / f"tpch_{request.param}"
    tables = {}
    for name in ("lineitem", "part"):
        tables[name] = pq.read_table(d / f"{name}.parquet")
    con = duckdb.connect()
    for name, t in tables.items():
        con.register(name, t)
    return request.param, tables, con


@pytest.fixture
def fused_calls(monkeypatch):
    """Count how often the fused pipeline actually served a plan."""
    calls = {"served": 0, "declined": 0}
    real = custom_ops.try_fused_aggregate

    def spy(*a, **k):
        out = real(*a, **k)
        calls["served" if out is not None else "declined"] += 1
        return out

    monkeypatch.setattr(custom_ops, "try_fused_aggregate", spy)
    return calls


def _rows(t: pa.Table):
    return {n: np.asarray(t.column(n).to_pylist(), dtype=object) for n in t.column_names}


def _close(got, want, rtol):
    got = np.asarray(got, dtype=np.float64)
    want = np.asarray(want, dtype=np.float64)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=rtol, atol=1e-6)


@pytest.mark.parametrize("device", DEVICES)
def test_q1_matches_duckdb(tpch, device, fused_calls):
    _, T, con = tpch
    got = bm.run_q1_mxframe(T["lineitem"], device=device)
    want = con.sql(f"""
        select l_returnflag, l_linestatus,
               sum(l_quantity::double) sum_qty, sum(l_extendedprice::double) sum_base_price,
               sum(l_extendedprice::double * (1 - l_discount::double)) sum_disc_price,
               sum(l_extendedprice::double * (1 - l_discount::double) * (1 + l_tax::double)) sum_charge,
               avg(l_quantity::double) avg_qty, avg(l_extendedprice::double) avg_price,
               avg(l_discount::double) avg_disc, count(*) count_order
        from lineitem where l_shipdate <= {bm.CUTOFF_Q1}
        group by 1, 2 order by 1, 2""").arrow()
    if hasattr(want, "read_all"):
        want = want.read_all()
    assert fused_calls["served"] == 1
    assert got.column_names[:2] == ["l_returnflag", "l_linestatus"]
    assert got.column("l_returnflag").to_pylist() == want.column("l_returnflag").to_pylist()
    assert got.column("l_linestatus").to_pylist() == want.column("l_linestatus").to_pylist()
    for n in ("sum_qty", "sum_base_price", "sum_disc_price", "sum_charge", "avg_qty", "avg_price",
              "avg_disc", "count_order"):
        _close(got.column(n).to_numpy(), want.column(n).to_numpy(), 1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_q6_matches_duckdb(tpch, device, fused_calls):
    _, T, con = tpch
    got = bm.run_q6_mxframe(T["lineitem"], device=device)
    want = con.sql(f"""
        select sum(l_extendedprice::double * l_discount::double) revenue from lineitem
        where l_shipdate >= {bm.Q6_DATE_LO} and l_shipdate < {bm.Q6_DATE_HI}
          and l_discount::double >= {bm.Q6_DISC_LO} and l_discount::double <= {bm.Q6_DISC_HI}
          and l_quantity::double < {bm.Q6_QTY_HI}""").fetchone()[0]
    assert fused_calls["served"] == 1
    _close([got.column("revenue")[0].as_py()], [want], 1e-9)


@pytest.mark.parametrize("device", DEVICES)
def test_q14_matches_duckdb(tpch, device, fused_calls):
    _, T, con = tpch
    got = bm.run_q14_mxframe(T["part"], T["lineitem"], device=device)
    want = con.sql(f"""
        select sum(case when p_type like 'PROMO%' then l_extendedprice::double * (1 - l_discount::double) else 0 end),
               sum(l_extendedprice::double * (1 - l_discount::double))
        from lineitem join part on l_partkey = p_partkey
        where l_shipdate >= {bm.Q14_DATE_LO} and l_shipdate < {bm.Q14_DATE_HI}""").fetchone()
    _close([got.column("promo_revenue")[0].as_py(), got.column("total_revenue")[0].as_py()], list(want), 1e-9)
    assert fused_calls["served"] == 1


@pytest.mark.parametrize("device", DEVICES)
def test_q19_matches_duckdb(tpch, device, fused_calls):
    _, T, con = tpch
    got = bm.run_q19_mxframe(T["part"], T["lineitem"], device=device)

    def cont(x):
        return ", ".join(f"'{c}'" for c in x)
    want = con.sql(f"""
        select sum(l_extendedprice::double * (1 - l_discount::double)) from lineitem join part on l_partkey = p_partkey
        where (p_brand = 'Brand#12' and p_container in ({cont(bm.CONTAINERS_SM)}) and l_quantity::double >= 1 and l_quantity::double <= 11 and p_size between 1 and 5)
           or (p_brand = 'Brand#23' and p_container in ({cont(bm.CONTAINERS_MD)}) and l_quantity::double >= 10 and l_quantity::double <= 20 and p_size between 1 and 10)
           or (p_brand = 'Brand#34' and p_container in ({cont(bm.CONTAINERS_LG)}) and l_quantity::double >= 20 and l_quantity::double <= 30 and p_size between 1 and 15)
        """).fetchone()[0]
    _close([got.column("revenue")[0].as_py()], [want], 1e-9)
    assert fused_calls["served"] == 1
