"""F-056 step 2 on TPC-H: the SQL text of Q1, Q3, Q5, Q6 and Q10 over `scan_parquet` sources, with the host
budget set so that SF1's lineitem (6M rows, ~25 row groups) is "bigger than memory". The answers equal DuckDB's
over the same files; the fact table streams in batches and the dimension tables stay resident.

SF1 data is not in the repository (tests/conftest.py `TPCH_DATA`); the tests skip without it.
"""
import sys
import warnings
from pathlib import Path

import pytest

from conftest import TPCH_DATA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from tpch_sql import run  # noqa: E402
from tpch_sql.queries import SQL  # noqa: E402

DATA = TPCH_DATA / "tpch_sf1_exact"
pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(not (DATA / "lineitem.parquet").exists(), reason="TPC-H SF1 data not present")]

import mxframe as mx  # noqa: E402
from mxframe import gpu_stream  # noqa: E402

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]


def sources(text):
    import sqlglot
    import sqlglot.expressions as sge
    tree = sqlglot.parse_one(text, dialect="duckdb")
    return {t.name.lower(): mx.scan_parquet(str(DATA / f"{t.name.lower()}.parquet")) for t in tree.find_all(sge.Table)}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("q", [1, 3, 5, 6, 10])
def test_tpch_query_over_a_file_bigger_than_the_budget_equals_duckdb(q, device, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "256M")
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", "64M")
    text = SQL[q]
    lf = mx.sql(text, **sources(text))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        got = lf.compute(device=device)
    info = lf.last_compile_provenance
    assert info["path"] == "streamed", info
    assert info["streamed"]["batches"] >= 3, info["streamed"]
    assert run.compare(got, run.duck_result(text, str(DATA)), text) == ""
