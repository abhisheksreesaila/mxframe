"""F-056 step 3c and F-113: a result bigger than memory has a way to finish, in bounded memory.

1. `LazyFrame.sink_parquet(path)` streams the result (batches of a filter, spilled groups, Grace pairs, sorted ranges)
   straight into a Parquet file; `compute()` keeps raising its MemoryError, which now names `sink_parquet`.
2. ORDER BY with no LIMIT over a file bigger than memory: sorted runs per batch on disk, merged by key ranges.
3. count(distinct x) per group over the budget: the (group keys, x) pairs are hash-partitioned and deduplicated
   one partition at a time, then counted per group.

Budgets are artificial (`MXFRAME_HOST_MEM_BUDGET`) over small files; every answer equals DuckDB's over the same file.
"""
import json
import math
import os
import subprocess
import sys
import textwrap

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
from mxframe import col, gpu_stream
from mxframe.io_parquet import scan_parquet

from test_f056_streamed import close, prov, run, same

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]

N = 400_000
GROUPS = 150_000
HOST = "6M"


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056s3c")
    idx = np.arange(N)
    rng = np.random.default_rng(3)
    t = pa.table({
        "g": ((idx * 7919) % GROUPS).astype(np.int64),
        "v": rng.random(N) * 100,
        "q": rng.integers(1, 1000, N).astype(np.int64),
        "id": idx.astype(np.int64),                                     # unique: a sort key without ties
        "s": np.array([f"k{i:06d}" for i in range(GROUPS)])[(idx * 31) % GROUPS],
    })
    path = str(d / "big.parquet")
    pq.write_table(t, path, row_group_size=20_000)
    return path


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", HOST)
    spill = tmp_path / "tmp"
    spill.mkdir()
    monkeypatch.setenv("MXFRAME_SPILL_DIR", str(spill))
    return spill


def duck(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW big AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


def spill_leftovers(tmp):
    return sorted(os.listdir(tmp))


def sink(lf, path, device="cpu", **kw):
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return lf.sink_parquet(path, device=device, **kw)


# --- F-113: sink_parquet ------------------------------------------------------------------------------------

def kept_rows(big):
    return scan_parquet(big).filter(col("q") > 100).select("id", "g", "v", "q", "s")      # ~90% of the file


@pytest.mark.parametrize("device", DEVICES)
def test_a_filter_keeping_most_of_the_file_is_written_and_equals_duckdb(big, device, tmp_path, _env):
    lf = kept_rows(big)
    with pytest.raises(MemoryError, match=r"bigger than memory.*sink_parquet"):
        run(lf, device)
    out = str(tmp_path / "kept.parquet")
    n = sink(lf, out, device)
    want = duck("SELECT id, g, v, q, s FROM big WHERE q > 100 ORDER BY id", big)
    got = pq.read_table(out)
    assert n == want.num_rows == got.num_rows
    assert prov(lf)["path"] == "streamed"
    same(got.sort_by("id"), want)
    assert spill_leftovers(_env) == []
    assert [f for f in os.listdir(tmp_path) if f != "tmp" and f != "kept.parquet"] == []


def test_an_over_budget_all_groups_aggregate_is_written_and_equals_duckdb(big, tmp_path, _env):
    lf = (scan_parquet(big).groupby("g")
          .agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").mean().alias("mv"),
               col("v").min().alias("lo"), col("q").max().alias("hi"), col("v").count().alias("n")))
    with pytest.raises(MemoryError, match=r"bigger than memory.*sink_parquet"):
        run(lf, "cpu")
    out = str(tmp_path / "groups.parquet")
    n = sink(lf, out)
    want = duck("SELECT g, sum(v) AS sv, sum(q) AS sq, avg(v) AS mv, min(v) AS lo, max(q) AS hi, count(v) AS n "
                "FROM big GROUP BY g", big)
    got = pq.read_table(out)
    assert n == GROUPS == got.num_rows
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 2
    same(got, want, ("g",))
    assert spill_leftovers(_env) == []


def test_a_result_that_fits_is_written_too(big, tmp_path):
    out = str(tmp_path / "small.parquet")
    n = sink(scan_parquet(big).filter(col("id") % 1000 == 3).select("id", "v"), out)
    want = duck("SELECT id, v FROM big WHERE id % 1000 = 3", big)
    assert n == want.num_rows == 400
    same(pq.read_table(out).sort_by("id"), want.sort_by("id"))


def test_an_empty_result_writes_a_file_with_the_columns(big, tmp_path):
    out = str(tmp_path / "none.parquet")
    n = sink(scan_parquet(big).filter(col("q") > 5000).select("id", "s"), out)
    got = pq.read_table(out)
    assert n == 0 and got.num_rows == 0 and got.column_names == ["id", "s"]
    assert got.schema.field("s").type == pa.string() and got.schema.field("id").type == pa.int64()


def test_an_in_memory_frame_can_be_sunk_too(tmp_path):
    out = str(tmp_path / "mem.parquet")
    t = pa.table({"a": [3, 1, 2], "b": ["x", None, "z"]})
    assert mx.LazyFrame(t).sort("a").sink_parquet(out) == 3
    assert pq.read_table(out).to_pylist() == [{"a": 1, "b": None}, {"a": 2, "b": "z"}, {"a": 3, "b": "x"}]


def test_a_failing_query_leaves_no_file_behind(big, tmp_path):
    out = str(tmp_path / "bad.parquet")
    with pytest.raises(Exception):
        scan_parquet(big).select(col("nope")).sink_parquet(out)
    assert os.listdir(tmp_path).count("bad.parquet") == 0
    assert [f for f in os.listdir(tmp_path) if f != "tmp"] == []


def test_an_existing_file_survives_a_failed_sink(big, tmp_path):
    out = str(tmp_path / "keep.parquet")
    pq.write_table(pa.table({"a": [1]}), out)
    with pytest.raises(Exception):
        scan_parquet(big).select(col("nope")).sink_parquet(out)
    assert pq.read_table(out).to_pylist() == [{"a": 1}]


SINK_PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    from mxframe import col
    from mxframe.io_parquet import scan_parquet

    def status(key):
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith(key + ":"))

    path, mode, out = sys.argv[1], sys.argv[2], sys.argv[3]
    lf = scan_parquet(path)
    if mode == "filter":
        lf = lf.filter(col("q") > 100).select("id", "g", "v", "q", "s")
    else:
        lf = lf.groupby("g").agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").count().alias("n"))
    lf.schema
    lf.sink_parquet(out)                             # warm: the first run decodes and frees everything once
    pa.default_memory_pool().release_unused()
    n = lf.sink_parquet(out)
    print(json.dumps({"rows": n, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory()}))
""")


def sink_peak(path, mode, budget, tmp, out):
    import conftest
    env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_SPILL_DIR=str(tmp),
               PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
    done = subprocess.run([sys.executable, "-c", SINK_PEAK, path, mode, out], env=env, capture_output=True, text=True,
                          check=True, timeout=600)
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("mode", ["filter", "groups"])
def test_peak_memory_of_a_sink_is_bounded_by_the_budget(big, mode, tmp_path, _env):
    """The filter result is 11 MB and the group result 5 MB or more in memory; the Arrow pool never holds more than the
    6 MB budget while they are written, and the same query with a big budget holds the whole result."""
    small = sink_peak(big, mode, HOST, _env, str(tmp_path / "a.parquet"))
    whole = sink_peak(big, mode, "64G", _env, str(tmp_path / "b.parquet"))
    assert small["path"] == "streamed" and whole["path"] != "streamed"
    assert small["rows"] == whole["rows"]
    assert small["arrow_peak"] < 6 << 20, small
    assert whole["arrow_peak"] > 8 << 20, whole
    assert pq.read_table(str(tmp_path / "a.parquet")).num_rows == small["rows"]


# --- sorts larger than memory ---------------------------------------------------------------------------------

M = 240_000
MESSY_HOST = "4M"


@pytest.fixture(scope="module")
def messy(tmp_path_factory):
    """Sort keys with everything the order has rules for: NaN, +-inf, NULL, heavy duplicates, strings with NULLs."""
    d = tmp_path_factory.mktemp("f056s3c_messy")
    rng = np.random.default_rng(11)
    idx = np.arange(M)
    f = np.round(rng.random(M) * 50) / 2                                  # 51 distinct values: many ties
    f[idx % 97 == 0] = np.nan
    f[idx % 89 == 0] = np.inf
    f[idx % 83 == 0] = -np.inf
    i = rng.integers(-5, 6, M)
    words = np.array(["", "a", "B", "b", "ab", "zz", "Zz", "é", "k" * 40])
    s = words[rng.integers(0, len(words), M)].astype(object)
    s[idx % 7 == 0] = None
    t = pa.table({
        "id": rng.permutation(M).astype(np.int64),                        # unique, in no order
        "f": pa.array(f, mask=(idx % 61 == 0)),
        "i": pa.array(i.astype(np.int64), mask=(idx % 53 == 0)),
        "s": pa.array(s, type=pa.string()),
        "v": rng.random(M),
    })
    path = str(d / "messy.parquet")
    pq.write_table(t, path, row_group_size=12_000)
    return path


@pytest.fixture
def messy_env(monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", MESSY_HOST)


def duck_m(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW messy AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


SORTS = {
    "float asc, nan largest, nulls last": (("f", "id"), [False, False], "f, id"),
    "float desc": (("f", "id"), [True, False], "f DESC, id"),
    "int then string desc then id": (("i", "s", "id"), [False, True, False], "i, s DESC, id"),
    "string asc": (("s", "id"), [False, False], "s, id"),
    "string desc, float desc": (("s", "f", "id"), [True, True, False], "s DESC, f DESC, id"),
}


@pytest.mark.parametrize("name", list(SORTS))
@pytest.mark.parametrize("device", DEVICES)
def test_a_sort_of_a_file_bigger_than_memory_is_written_in_order_and_equals_duckdb(messy, messy_env, name, device, tmp_path, _env):
    by, desc, order = SORTS[name]
    lf = scan_parquet(messy).select("id", "f", "i", "s", "v").sort(*by, descending=desc)
    out = str(tmp_path / "sorted.parquet")
    n = sink(lf, out, device)
    p = prov(lf)
    assert p["path"] == "streamed" and p["streamed"]["rows"]["kind"] == "sort"
    assert p["streamed"]["rows"]["sort"]["runs"] >= 3 and p["streamed"]["rows"]["sort"]["ranges"] >= 3
    got = pq.read_table(out)
    assert n == got.num_rows == M
    same(got, duck_m(f"SELECT id, f, i, s, v FROM messy ORDER BY {order}", messy))      # in order, not re-sorted
    assert spill_leftovers(_env) == []


def test_compute_of_a_sort_that_fits_equals_duckdb_and_one_that_does_not_names_sink_parquet(messy, messy_env):
    lf = scan_parquet(messy).filter(col("id") % 40 == 3).select("id", "f", "s").sort("f", "s", "id", descending=[True, False, False])
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["rows"]["kind"] == "sort"
    same(got, duck_m("SELECT id, f, s FROM messy WHERE id % 40 = 3 ORDER BY f DESC, s, id", messy))
    big_sort = scan_parquet(messy).select("id", "f", "i", "s", "v").sort("f", "id")
    with pytest.raises(MemoryError, match=r"bigger than memory.*sink_parquet"):
        run(big_sort, "cpu")


def test_sql_order_by_without_limit_with_a_hidden_sort_key_is_written_in_order(messy, messy_env, tmp_path):
    lf = mx.sql("SELECT id, s FROM messy WHERE i IS NOT NULL ORDER BY f DESC, id", messy=scan_parquet(messy))
    out = str(tmp_path / "sql.parquet")
    n = sink(lf, out)
    assert prov(lf)["streamed"]["rows"]["kind"] == "sort"
    want = duck_m("SELECT id, s FROM messy WHERE i IS NOT NULL ORDER BY f DESC, id", messy)
    assert n == want.num_rows
    same(pq.read_table(out), want)


def test_an_empty_sort_result_writes_the_columns(messy, messy_env, tmp_path):
    out = str(tmp_path / "e.parquet")
    n = sink(scan_parquet(messy).filter(col("id") < 0).select("id", "s").sort("id"), out)
    got = pq.read_table(out)
    assert n == 0 and got.column_names == ["id", "s"]


def test_sorting_the_groups_of_a_spilled_aggregate_without_limit(big, tmp_path, _env):
    lf = (scan_parquet(big).groupby("g").agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"))
          .sort("sq", "g", descending=[True, False]))
    out = str(tmp_path / "g.parquet")
    n = sink(lf, out)
    p = prov(lf)
    assert p["streamed"]["spill"]["partitions"] >= 2 and p["streamed"]["rows"]["kind"] == "sort"
    assert n == GROUPS
    same(pq.read_table(out), duck("SELECT g, sum(v) AS sv, sum(q) AS sq FROM big GROUP BY g ORDER BY sq DESC, g", big))
    assert spill_leftovers(_env) == []


def test_a_sort_leaves_no_spill_files_after_an_error(messy, messy_env, _env, monkeypatch, tmp_path):
    from mxframe import extsort
    seen = {}

    def broken(self, i):
        seen["files"] = spill_leftovers(_env)
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(extsort._Files, "read", broken)
    with pytest.raises(RuntimeError, match="disk on fire"):
        sink(scan_parquet(messy).select("id", "f").sort("f", "id"), str(tmp_path / "x.parquet"))
    assert len(seen["files"]) == 1
    assert spill_leftovers(_env) == []
    assert [f for f in os.listdir(tmp_path) if f != "tmp"] == []


def test_ties_keep_the_order_of_the_file(messy, messy_env, tmp_path):
    """Sorting by a key with 11 values and no tiebreaker: rows with equal keys come out in file order (the sort is
    stable, as the in-memory one is)."""
    out = str(tmp_path / "stable.parquet")
    sink(scan_parquet(messy).select("id", "i").sort("i"), out)
    got = pq.read_table(out).to_pylist()
    src = {r["id"]: k for k, r in enumerate(pq.read_table(messy, columns=["id"]).to_pylist())}
    last = None
    for r in got:
        pos = src[r["id"]]
        if last is not None and last[0] == r["i"]:
            assert pos > last[1], (r, last)
        last = (r["i"], pos)


SORT_PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    from mxframe.io_parquet import scan_parquet

    path, out = sys.argv[1], sys.argv[2]
    lf = scan_parquet(path).select("id", "f", "i", "v").sort("f", "i", "id", descending=[True, False, False])
    lf.schema
    lf.sink_parquet(out)
    pa.default_memory_pool().release_unused()
    n = lf.sink_parquet(out)
    print(json.dumps({"rows": n, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory()}))
""")


def test_peak_memory_of_an_external_sort_is_bounded_by_the_budget(messy, tmp_path, _env):
    import conftest
    res = {}
    for budget in (MESSY_HOST, "64G"):
        env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_SPILL_DIR=str(_env),
                   PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
        done = subprocess.run([sys.executable, "-c", SORT_PEAK, messy, str(tmp_path / f"{budget}.parquet")], env=env,
                              capture_output=True, text=True, check=True, timeout=600)
        res[budget] = json.loads(done.stdout.strip().splitlines()[-1])
    assert res[MESSY_HOST]["path"] == "streamed" and res["64G"]["path"] != "streamed"
    assert res[MESSY_HOST]["rows"] == res["64G"]["rows"] == M
    assert res[MESSY_HOST]["arrow_peak"] < 4 << 20, res
    assert res["64G"]["arrow_peak"] > 8 << 20, res
    assert pq.read_table(str(tmp_path / f"{MESSY_HOST}.parquet")).column("id").equals(pq.read_table(str(tmp_path / "64G.parquet")).column("id"))
