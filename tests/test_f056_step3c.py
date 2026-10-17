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


# --- count(distinct x) over the budget -----------------------------------------------------------------------

D = 400_000
KEYS = 3_000


@pytest.fixture(scope="module")
def dist(tmp_path_factory):
    """3,000 groups with up to 300 distinct x each (about 340k distinct (g, x) pairs, 5 MB of state: over the budget),
    NULLs in the group key and in x, a float column with NaN and both zeros, strings."""
    d = tmp_path_factory.mktemp("f056s3c_dist")
    rng = np.random.default_rng(5)
    idx = np.arange(D)
    x = rng.integers(0, 300, D)
    fx = x.astype(np.float64) / 4
    fx[idx % 211 == 0] = np.nan
    nan_neg = np.frombuffer(np.uint64(0xFFF8000000000001).tobytes(), dtype=np.float64)[0]
    fx[idx % 223 == 0] = nan_neg
    fx[idx % 227 == 0] = -0.0
    fx[idx % 229 == 0] = 0.0
    g = rng.integers(0, KEYS, D).astype(np.int64)
    t = pa.table({
        "g": pa.array(g, mask=(idx % 97 == 0)),
        "k": pa.array(g % 50, mask=(idx % 97 == 0)),                      # 50 groups with many x each
        "x": pa.array(x.astype(np.int64), mask=(idx % 13 == 0)),
        "fx": pa.array(fx, mask=(idx % 191 == 0)),
        "sx": pa.array(np.array([f"v{i:03d}" for i in range(300)])[x], mask=(idx % 17 == 0)),
        "v": rng.random(D),
        "q": rng.integers(1, 1000, D).astype(np.int64),
    })
    path = str(d / "dist.parquet")
    pq.write_table(t, path, row_group_size=20_000)
    return path


def duck_d(query, path):
    con = duckdb.connect()
    con.execute(f"CREATE VIEW dist AS SELECT * FROM read_parquet('{path}')")
    return con.sql(query).arrow().read_all()


@pytest.mark.parametrize("device", DEVICES)
def test_count_distinct_per_group_over_the_budget_equals_duckdb(dist, device, _env):
    lf = mx.sql("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist=scan_parquet(dist))
    got = run(lf, device)
    p = prov(lf)
    assert p["path"] == "streamed" and p["streamed"]["spill"]["partitions"] >= 2
    want = duck_d("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist)
    assert got.num_rows == want.num_rows == KEYS + 1                       # the NULL group is one of them
    same(got, want, ("g",))
    assert spill_leftovers(_env) == []


def test_count_distinct_ignores_nulls_and_counts_a_group_of_only_nulls_as_zero(tmp_path, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "1M")
    n = 120_000
    idx = np.arange(n)
    g = (idx % 4_000).astype(np.int64)
    x = pa.array((idx * 7 % 5_000).astype(np.int64), mask=(g % 10 == 0) | (idx % 11 == 0))      # groups 0, 10, ... have only NULLs
    path = str(tmp_path / "n.parquet")
    pq.write_table(pa.table({"g": g, "x": x}), path, row_group_size=4_000)
    lf = mx.sql("SELECT g, count(DISTINCT x) AS nd FROM n GROUP BY g", n=scan_parquet(path))
    got = run(lf, "cpu")
    assert prov(lf)["streamed"]["spill"]["partitions"] >= 2
    con = duckdb.connect()
    want = con.execute(f"SELECT g, count(DISTINCT x) AS nd FROM read_parquet('{path}') GROUP BY g").arrow().read_all()
    same(got, want, ("g",))
    zeros = [r["g"] for r in got.to_pylist() if r["nd"] == 0]
    assert 0 in zeros and 10 in zeros and len(zeros) == 400


def test_a_global_count_distinct_over_the_budget(dist):
    lf = mx.sql("SELECT count(DISTINCT g) AS nd FROM dist", dist=scan_parquet(dist))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    same(got, duck_d("SELECT count(DISTINCT g) AS nd FROM dist", dist))


def test_count_distinct_with_other_aggregates_equals_duckdb(dist, _env):
    q = ("SELECT g, count(DISTINCT x) AS nd, sum(v) AS sv, sum(q) AS sq, avg(q) AS aq, min(v) AS lo, max(q) AS hi, "
         "count(*) AS n FROM dist GROUP BY g")
    lf = mx.sql(q, dist=scan_parquet(dist))
    got = run(lf, "cpu")
    p = prov(lf)
    assert p["path"] == "streamed" and p["streamed"]["spill"]["partitions"] >= 2 and p["streamed"].get("stages") == 2
    same(got, duck_d(q, dist), ("g",))
    assert spill_leftovers(_env) == []


def test_count_distinct_with_having_order_by_and_limit(dist):
    q = ("SELECT g, count(DISTINCT x) AS nd, sum(q) AS sq FROM dist GROUP BY g HAVING count(DISTINCT x) > 150 "
         "ORDER BY nd DESC, g LIMIT 25")
    lf = mx.sql(q, dist=scan_parquet(dist))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    same(got, duck_d(q, dist))


def test_the_dataframe_form_select_distinct_groupby_count_equals_duckdb(dist):
    lf = scan_parquet(dist).select("g", "x").distinct().groupby("g").agg(col("x").count().alias("nd"))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed" and prov(lf)["streamed"]["spill"]["partitions"] >= 2
    same(got, duck_d("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist), ("g",))


@pytest.mark.parametrize("key", ["k", "g"])
@pytest.mark.parametrize("column", ["fx", "sx"])
def test_count_distinct_of_floats_with_nan_and_signed_zero_and_of_strings(dist, column, key):
    """NaN of any bit pattern is one value, -0.0 and 0.0 are one value, NULL is none; with 50 groups the pairs fit and
    are merged, with 3,000 they are spilled and deduplicated one hash partition at a time."""
    q = f"SELECT {key}, count(DISTINCT {column}) AS nd FROM dist GROUP BY {key}"
    lf = mx.sql(q, dist=scan_parquet(dist))
    got = run(lf, "cpu")
    assert prov(lf)["path"] == "streamed"
    assert ("spill" in prov(lf)["streamed"]) == (key == "g")
    same(got, duck_d(q, dist), (key,))


def test_a_sink_of_count_distinct_writes_the_groups(dist, tmp_path):
    lf = mx.sql("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist=scan_parquet(dist))
    out = str(tmp_path / "cd.parquet")
    assert sink(lf, out) == KEYS + 1
    same(pq.read_table(out), duck_d("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist), ("g",))


CD_PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    import mxframe as mx
    from mxframe.io_parquet import scan_parquet

    lf = mx.sql("SELECT g, count(DISTINCT x) AS nd FROM dist GROUP BY g", dist=scan_parquet(sys.argv[1]))
    lf.compute(device="cpu")
    pa.default_memory_pool().release_unused()
    n = lf.compute(device="cpu").num_rows
    print(json.dumps({"rows": n, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory()}))
""")


def test_peak_memory_of_a_spilled_count_distinct_is_bounded_by_the_budget(dist, _env):
    import conftest
    res = {}
    for budget in (HOST, "64G"):
        env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_SPILL_DIR=str(_env),
                   PYTHONPATH=os.pathsep.join([str(conftest._shim)] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH") else [])))
        done = subprocess.run([sys.executable, "-c", CD_PEAK, dist], env=env, capture_output=True, text=True, check=True, timeout=600)
        res[budget] = json.loads(done.stdout.strip().splitlines()[-1])
    assert res[HOST]["path"] == "streamed" and res["64G"]["path"] != "streamed"
    assert res[HOST]["rows"] == res["64G"]["rows"] == KEYS + 1
    assert res[HOST]["arrow_peak"] < 6 << 20, res
    assert res["64G"]["arrow_peak"] > 8 << 20, res
    assert spill_leftovers(_env) == []


# --- the other streams, into a sink -------------------------------------------------------------------------------

def test_a_grace_join_result_bigger_than_memory_is_written_pair_by_pair(big, tmp_path, _env):
    left = scan_parquet(big).select("id", "g", "v")
    right = scan_parquet(big).filter(col("q") > 400).select("id", "q")
    lf = left.join(right, on="id")
    with pytest.raises(MemoryError, match="sink_parquet"):
        run(lf, "cpu")
    out = str(tmp_path / "join.parquet")
    n = sink(lf, out)
    assert "grace" in prov(lf)["streamed"]
    want = duck("SELECT a.id, a.g, a.v, b.q FROM big a JOIN (SELECT id, q FROM big WHERE q > 400) b ON a.id = b.id", big)
    assert n == want.num_rows
    got = pq.read_table(out)
    same(got.sort_by("id"), want.sort_by("id"))
    assert spill_leftovers(_env) == []


# --- SQL COPY ------------------------------------------------------------------------------------------------------

def test_sql_copy_to_parquet_streams_a_result_bigger_than_memory(big, tmp_path):
    out = str(tmp_path / "copy.parquet")
    res = mx.sql(f"COPY (SELECT id, g, v, q, s FROM big WHERE q > 100) TO '{out}' (FORMAT PARQUET)",
                 big=scan_parquet(big)).compute()
    want = duck("SELECT id, g, v, q, s FROM big WHERE q > 100 ORDER BY id", big)
    assert res.to_pylist() == [{"Count": want.num_rows}]
    same(pq.read_table(out).sort_by("id"), want)


def test_sql_copy_without_options_and_unsupported_forms(big, tmp_path):
    out = str(tmp_path / "c2.parquet")
    assert mx.sql(f"COPY (SELECT id FROM big LIMIT 5) TO '{out}'", big=scan_parquet(big)).compute().to_pylist() == [{"Count": 5}]
    assert pq.read_table(out).num_rows == 5
    with pytest.raises(NotImplementedError, match="FORMAT PARQUET"):
        mx.sql(f"COPY (SELECT id FROM big) TO '{out}' (FORMAT CSV)", big=scan_parquet(big))
    with pytest.raises(NotImplementedError, match=r"COPY \(SELECT"):
        mx.sql(f"COPY big TO '{out}'", big=scan_parquet(big))


def test_median_quantile_stddev_and_variance_are_refused_by_name_not_with_a_key_error():
    """The engine has no such aggregate (in memory or streamed): a SQL query that asks for one says so."""
    t = pa.table({"g": [1, 1, 2], "x": [1.0, 2.0, 4.0]})
    for q in ("median(x)", "stddev(x)", "var_samp(x)", "quantile_cont(x, 0.5)"):
        with pytest.raises(NotImplementedError, match="Unsupported aggregate function"):
            mx.sql(f"SELECT g, {q} FROM t GROUP BY g", t=t)


def test_reproducible_mode_gives_the_same_bits_for_a_spilled_count_distinct_and_a_sorted_sink(dist, messy, tmp_path, monkeypatch):
    q = "SELECT g, count(DISTINCT x) AS nd, sum(v) AS sv, avg(v) AS av FROM dist GROUP BY g"
    first = run(mx.sql(q, dist=scan_parquet(dist)), "cpu", reproducible=True)
    again = run(mx.sql(q, dist=scan_parquet(dist)), "cpu", reproducible=True)
    assert first.to_pylist() == again.to_pylist() and first.num_rows == KEYS + 1
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", MESSY_HOST)
    for name in ("a", "b"):
        sink(scan_parquet(messy).select("id", "f", "v").sort("f", "id"), str(tmp_path / f"{name}.parquet"), reproducible=True)
    assert pq.read_table(str(tmp_path / "a.parquet")).column("id").to_pylist() == \
        pq.read_table(str(tmp_path / "b.parquet")).column("id").to_pylist()
