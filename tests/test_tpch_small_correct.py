"""F-027: every engine answers all 22 TPC-H queries like DuckDB on tiny data, where filters and
joins often leave intermediates empty (SF 0.002 ~ 12k lineitem rows, SF 0.017 ~ 100k)."""
import importlib
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
from tpch_user import spec  # noqa: E402
from tpch_user.compare import compare_tables  # noqa: E402

ENGINES = [("pandas", "q_pandas", None), ("polars", "q_polars", None), ("mx-cpu", "q_mx", "cpu")]


@pytest.fixture(scope="module", params=["0.002", "0.017"])
def data_dir(request, tmp_path_factory):
    gen = importlib.import_module("gen_tpch_parquet")
    d = tmp_path_factory.mktemp("tpch") / f"tpch_sf{request.param}_exact"
    gen.generate(float(request.param), str(d))
    return str(d)


def _arrow(mod, res, q):
    tbl = mod.to_arrow(res)
    names = spec.OUTPUT[q]
    return tbl.select(names) if all(n in tbl.column_names for n in names) else tbl.rename_columns(names)


@pytest.mark.parametrize("mode", ["user-felt", "engine"])
def test_all_queries_match_duckdb(data_dir, mode):
    ref = importlib.import_module("tpch_user.q_duckdb")
    bad = []
    for q in sorted(spec.OUTPUT):
        want = _arrow(ref, ref.run(q, data_dir), q)
        for name, modname, device in ENGINES:
            mod = importlib.import_module(f"tpch_user.{modname}")
            try:
                res = (mod.run(q, data_dir, device) if mode == "user-felt"
                       else mod.run_mem(q, mod.load_mem(q, data_dir), device))
                got = _arrow(mod, res, q)
            except Exception as e:  # noqa: BLE001
                bad.append(f"Q{q} {name}: {type(e).__name__}: {e}")
                continue
            cmp = compare_tables(got, want, topn_metric=spec.TOPN_METRIC.get(q), order_by=spec.ORDER_BY.get(q),
                                 empty_sum_is_zero=name in ("pandas", "polars"))
            if cmp.verdict == "wrong":
                bad.append(f"Q{q} {name}: {cmp.message}")
    assert not bad, "\n".join(bad)
