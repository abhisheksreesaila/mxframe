"""F-056 step 2: a query over a Parquet file bigger than memory finishes, with bounded memory.

`MXFRAME_HOST_MEM_BUDGET` and `MXFRAME_GPU_MEM_BUDGET` make a ~200 MB file with 60 row groups "bigger than
the machine". The query is cut into batches of row groups (filter, project, partial aggregate per batch), the
partials are merged, and the result must equal DuckDB's over the same file on cpu, gpu and auto. The path
(`provenance["path"] == "streamed"`, batch count) and the peak memory are asserted.
"""
import math
import os
import subprocess
import sys
import textwrap
import warnings

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from mxframe import LazyFrame, col, gpu_stream
from mxframe.io_parquet import scan_parquet

N = 6_000_000
GROUP_ROWS = 100_000          # 60 row groups of ~3.4 MB decoded
HOST = "40M"
GPU = "8M"

HAS_GPU = gpu_stream.cuda_available()
DEVICES = ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU")),
           pytest.param("auto", marks=pytest.mark.skipif(not HAS_GPU, reason="no CUDA GPU"))]


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("f056")
    rng = np.random.default_rng(56)
    idx = np.arange(N)
    fact = pa.table({
        "k": (idx % 4).astype(np.int32),
        "j": rng.integers(0, 2000, N).astype(np.int32),
        "v": rng.random(N) * 100,
        "q": rng.integers(1, 51, N).astype(np.int64),
        "d": (idx * 2500 // N).astype(np.int32),                        # sorted: a filter on it prunes row groups
        "s": np.array(["A", "B", "C"])[rng.integers(0, 3, N)],
        "m": ((idx * 7919) % 300_000).astype(np.int32),                 # 300k groups
    })
    dim1 = pa.table({"id": np.arange(2000, dtype=np.int32), "region": (np.arange(2000) % 25).astype(np.int32),
                     "w": (np.arange(2000) % 7).astype(np.int64)})
    dim2 = pa.table({"region": np.arange(25, dtype=np.int32), "rname": np.array([f"R{i:02d}" for i in range(25)])})
    out = {n: str(d / f"{n}.parquet") for n in ("fact", "dim1", "dim2")}
    pq.write_table(fact, out["fact"], row_group_size=GROUP_ROWS)
    pq.write_table(dim1, out["dim1"])
    pq.write_table(dim2, out["dim2"])
    return out


@pytest.fixture(autouse=True)
def _budgets(monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", HOST)
    monkeypatch.setenv("MXFRAME_GPU_MEM_BUDGET", GPU)


def sql(query, files):
    con = duckdb.connect()
    for n, p in files.items():
        con.execute(f"CREATE VIEW {n} AS SELECT * FROM read_parquet('{p}')")
    return con.sql(query).arrow().read_all()


def close(a, b):
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) or isinstance(b, float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def same(got, want, keys=()):
    """Same columns, same rows (sorted by `keys` when given), floats to 1e-9 relative, everything else exact."""
    assert got.column_names == want.column_names
    assert got.num_rows == want.num_rows
    g, w = got.to_pylist(), want.to_pylist()
    if keys:
        order = lambda r: tuple((r[k] is None, r[k]) for k in keys)       # noqa: E731
        g, w = sorted(g, key=order), sorted(w, key=order)
    for gr, wr in zip(g, w):
        for name in want.column_names:
            assert close(gr[name], wr[name]), (name, gr, wr)


def prov(lf):
    return lf.last_compile_provenance


def run(lf, device, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)         # gpu says when a piece leaves the GPU
        return lf.compute(device=device, **kw)


def assert_streamed(lf, min_batches=3):
    p = prov(lf)
    assert p["path"] == "streamed", p
    assert p["streamed"]["batches"] >= min_batches
    return p["streamed"]


# --- TPC-H shaped aggregates ---------------------------------------------------------------------------------

def q1(files):
    lf = (scan_parquet(files["fact"]).filter(col("d") <= 2000).groupby("k", "s")
          .agg(col("q").sum().alias("sum_q"), col("v").sum().alias("sum_v"),
               (col("v") * col("q")).sum().alias("sum_vq"), col("v").mean().alias("avg_v"),
               col("q").mean().alias("avg_q"), col("v").min().alias("min_v"), col("q").max().alias("max_q"),
               col("v").count().alias("n")))
    want = sql("SELECT k, s, sum(q) AS sum_q, sum(v) AS sum_v, sum(v * q) AS sum_vq, avg(v) AS avg_v, "
               "avg(q) AS avg_q, min(v) AS min_v, max(q) AS max_q, count(v) AS n FROM fact WHERE d <= 2000 "
               "GROUP BY k, s", files)
    return lf, want, ("k", "s")


def q6(files):
    lf = (scan_parquet(files["fact"]).filter((col("d") >= 500) & (col("d") < 1500) & (col("q") < 24))
          .groupby().agg((col("v") * col("q")).sum().alias("revenue"), col("v").count().alias("n")))
    want = sql("SELECT sum(v * q) AS revenue, count(v) AS n FROM fact WHERE d >= 500 AND d < 1500 AND q < 24", files)
    return lf, want, ()


def q3(files):
    """Fact x two nested dimensions, a HAVING-like filter, ORDER BY ... LIMIT above the aggregate."""
    fact, dim1, dim2 = (scan_parquet(files[n]) for n in ("fact", "dim1", "dim2"))
    lf = (fact.filter(col("d") < 2200)
          .join(dim1.join(dim2, on="region"), left_on="j", right_on="id")
          .groupby("rname", "w").agg((col("v") * col("q")).sum().alias("revenue"), col("v").count().alias("n"))
          .sort("revenue", descending=True).limit(7))
    want = sql("SELECT rname, w, sum(v * q) AS revenue, count(v) AS n FROM fact JOIN dim1 ON fact.j = dim1.id "
               "JOIN dim2 ON dim1.region = dim2.region WHERE d < 2200 GROUP BY rname, w "
               "ORDER BY revenue DESC LIMIT 7", files)
    return lf, want, ()


def q5_dim_filter(files):
    """A filtered dimension (the build side shrinks) and a left-deep join; group by a dimension column."""
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"]).filter(col("w") < 3)
    lf = (fact.join(dim1, left_on="j", right_on="id").groupby("region")
          .agg(col("v").sum().alias("rev"), col("q").sum().alias("sq")))
    want = sql("SELECT region, sum(v) AS rev, sum(q) AS sq FROM fact JOIN dim1 ON fact.j = dim1.id "
               "WHERE w < 3 GROUP BY region", files)
    return lf, want, ("region",)


def many_groups(files):
    lf = (scan_parquet(files["fact"]).groupby("m")
          .agg(col("v").sum().alias("sv"), col("q").sum().alias("sq"), col("v").count().alias("n")))
    want = sql("SELECT m, sum(v) AS sv, sum(q) AS sq, count(v) AS n FROM fact GROUP BY m", files)
    return lf, want, ("m",)


QUERIES = {"q1": q1, "q6": q6, "q3": q3, "q5": q5_dim_filter, "many_groups": many_groups}


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", sorted(QUERIES))
def test_a_file_bigger_than_the_budget_is_streamed_and_equals_duckdb(files, name, device):
    lf, want, keys = QUERIES[name](files)
    got = run(lf, device)
    info = assert_streamed(lf)
    same(got, want, keys)
    assert info["host_budget"] == 40 << 20
    assert sum(info["devices"].values()) == info["batches"]
    if device == "cpu":
        assert set(info["devices"]) == {"cpu"}


def test_without_a_budget_that_is_exceeded_nothing_streams(files, monkeypatch):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    lf, want, keys = q1(files)
    got = run(lf, "cpu")
    assert prov(lf)["path"] != "streamed"
    same(got, want, keys)


def test_a_streamed_result_has_the_dtypes_of_the_in_memory_one(files, monkeypatch):
    lf, _, _ = q1(files)
    streamed = run(lf, "cpu")
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "64G")
    whole = run(q1(files)[0], "cpu")
    assert streamed.schema == whole.schema


def test_batches_come_from_the_row_groups_the_filter_leaves(files, monkeypatch):
    """d is sorted, so d < 250 keeps the first six of 60 row groups: the rest are never read."""
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "10M")
    small = scan_parquet(files["fact"]).filter(col("d") < 250).groupby("k").agg(col("v").sum().alias("s"))
    got = run(small, "cpu")
    info = assert_streamed(small, 3)
    assert info["row_groups"] == 60 and info["row_groups_read"] == 6
    same(got, sql("SELECT k, sum(v) AS s FROM fact WHERE d < 250 GROUP BY k", files), ("k",))


def test_every_batch_empty_gives_the_empty_answers(files):
    """The filter cannot be answered from statistics, so every batch is read and filtered down to nothing."""
    nothing = (col("q") + col("k")) > 500
    keyed = scan_parquet(files["fact"]).filter(nothing).groupby("k").agg(col("v").sum().alias("s"))
    assert run(keyed, "cpu").num_rows == 0
    assert_streamed(keyed)
    glob = (scan_parquet(files["fact"]).filter(nothing)
            .groupby().agg(col("v").sum().alias("s"), col("q").sum().alias("sq"), col("v").count().alias("n"),
                           col("v").mean().alias("m"), col("q").min().alias("mn")))
    got = run(glob, "cpu")
    assert_streamed(glob)
    assert got.to_pylist() == [{"s": None, "sq": None, "n": 0, "m": None, "mn": None}]
    same(got, sql("SELECT sum(v) AS s, sum(q) AS sq, count(v) AS n, avg(v) AS m, min(q) AS mn FROM fact "
                  "WHERE q + k > 500", files))


def test_a_group_found_in_some_batches_only_is_kept(files):
    """d is sorted: each of its 2,500 values lives in one or two batches, the other batches never see it."""
    lf = scan_parquet(files["fact"]).groupby("d").agg(col("v").count().alias("n"), col("v").sum().alias("s"))
    got = run(lf, "cpu")
    assert_streamed(lf)
    same(got, sql("SELECT d, count(v) AS n, sum(v) AS s FROM fact GROUP BY d", files), ("d",))


@pytest.mark.parametrize("device", DEVICES)
def test_batches_emptied_by_the_row_filter_next_to_batches_that_are_not(files, device):
    """d is sorted and `d + k` cannot be answered from statistics: the early batches filter down to nothing
    (a partial of no groups, whose column types may differ), the late ones do not."""
    lf = (scan_parquet(files["fact"]).filter((col("d") + col("k")) > 2000).groupby("k")
          .agg(col("q").sum().alias("sq"), col("v").mean().alias("m"), col("q").min().alias("lo"),
               col("v").count().alias("n")))
    got = run(lf, device)
    assert_streamed(lf)
    same(got, sql("SELECT k, sum(q) AS sq, avg(v) AS m, min(q) AS lo, count(v) AS n FROM fact WHERE d + k > 2000 "
                  "GROUP BY k", files), ("k",))


@pytest.mark.parametrize("how", ["left", "semi", "anti"])
def test_the_streamed_table_is_the_left_side_of_left_semi_and_anti_joins(files, how):
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"]).filter(col("w") < 3)
    lf = (fact.join(dim1, left_on="j", right_on="id", how=how).groupby("k")
          .agg(col("v").sum().alias("s"), col("v").count().alias("n")))
    got = run(lf, "cpu")
    assert_streamed(lf)
    kind = {"left": "LEFT JOIN", "semi": "SEMI JOIN", "anti": "ANTI JOIN"}[how]
    want = sql(f"SELECT k, sum(v) AS s, count(v) AS n FROM fact {kind} (SELECT * FROM dim1 WHERE w < 3) d "
               f"ON fact.j = d.id GROUP BY k", files)
    same(got, want, ("k",))


def test_a_join_where_the_big_file_is_the_right_side_of_a_left_join_is_not_streamed(files):
    fact, dim1 = scan_parquet(files["fact"]), scan_parquet(files["dim1"])
    lf = dim1.join(fact, left_on="id", right_on="j", how="left").groupby("region").agg(col("v").sum().alias("s"))
    with pytest.warns(RuntimeWarning, match="not streamed: the big file is the build side of a left join"):
        lf.compute(device="cpu")
    assert prov(lf)["path"] != "streamed"


# --- exactness ----------------------------------------------------------------------------------------------

def small_file(tmp_path, table, rows_per_group=2000):
    path = str(tmp_path / "small.parquet")
    pq.write_table(table, path, row_group_size=rows_per_group)
    return path


def tiny(monkeypatch, host="100K"):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", host)


@pytest.mark.parametrize("device", DEVICES)
def test_integer_sums_stay_exact_past_2_53(tmp_path, monkeypatch, device):
    tiny(monkeypatch)
    n = 200_000
    rng = np.random.default_rng(1)
    t = pa.table({"k": (np.arange(n) % 3).astype(np.int32),
                  "x": (2 ** 45 + rng.integers(0, 1000, n)).astype(np.int64)})     # every group sums to ~2^62
    path = small_file(tmp_path, t)
    lf = scan_parquet(path).groupby("k").agg(col("x").sum().alias("s"), col("x").min().alias("lo"),
                                            col("x").max().alias("hi"), col("x").count().alias("n"))
    got = run(lf, device)
    assert_streamed(lf, 5)
    exact = {k: int(t.column("x").to_numpy()[np.arange(n) % 3 == k].sum()) for k in range(3)}
    assert {r["k"]: r["s"] for r in got.to_pylist()} == exact
    assert max(exact.values()) > 2 ** 53
    ref = duckdb.sql(f"SELECT k, min(x) AS lo, max(x) AS hi, count(x) AS n FROM read_parquet('{path}') GROUP BY k").arrow().read_all()
    assert {r["k"]: (r["lo"], r["hi"], r["n"]) for r in got.to_pylist()} == \
        {r["k"]: (r["lo"], r["hi"], r["n"]) for r in ref.to_pylist()}


def test_integer_sums_whose_batches_are_small_but_total_is_not(tmp_path, monkeypatch):
    """Each batch's sum is below 2^53 (the engine answers it as a double); the total is not: the partials are
    merged as int64, not rounded as doubles."""
    tiny(monkeypatch, "20K")
    n = 40_000
    t = pa.table({"k": np.zeros(n, dtype=np.int32), "x": np.full(n, 2 ** 38 + 1, dtype=np.int64)})
    lf = scan_parquet(small_file(tmp_path, t, 1000)).groupby("k").agg(col("x").sum().alias("s"))
    got = run(lf, "cpu")
    info = assert_streamed(lf, 8)
    assert got.column("s").to_pylist() == [n * (2 ** 38 + 1)]          # ~2^53.3: a double would round it
    assert n * (2 ** 38 + 1) > 2 ** 53 and info["batches"] > 8


@pytest.mark.parametrize("device", DEVICES)
def test_nulls_and_nans_follow_the_engine_rules(tmp_path, monkeypatch, device):
    tiny(monkeypatch)
    n = 60_000
    rng = np.random.default_rng(2)
    k = (np.arange(n) % 5).astype(np.int32)
    v = rng.random(n)
    v[(np.arange(n) % 97) == 0] = np.nan
    v[k == 4] = np.nan                                    # group 4 is all NaN
    x = pa.array(rng.integers(0, 100, n).astype(np.float64), mask=(k == 3) | (np.arange(n) % 11 == 0))   # group 3 is all NULL
    t = pa.table({"k": pa.array(k, mask=(np.arange(n) % 50 == 7)), "v": v, "x": x})   # some NULL keys
    path = small_file(tmp_path, t, 1500)
    lf = scan_parquet(path).groupby("k").agg(col("v").max().alias("vmax"), col("v").min().alias("vmin"),
                                            col("x").sum().alias("xs"), col("x").mean().alias("xm"),
                                            col("x").count().alias("xn"), col("v").count().alias("vn"))
    got = run(lf, device)
    assert_streamed(lf, 5)
    want = duckdb.sql(f"SELECT k, max(v) AS vmax, min(v) AS vmin, sum(x) AS xs, avg(x) AS xm, count(x) AS xn, "
                      f"count(v) AS vn FROM read_parquet('{path}') GROUP BY k").arrow().read_all()
    same(got, want, ("k",))


def test_reproducible_mode_gives_the_same_bits_every_run(files):
    lf, _, _ = q1(files)
    a = run(lf, "cpu", reproducible=True).sort_by([("k", "ascending"), ("s", "ascending")])
    b = run(q1(files)[0], "cpu", reproducible=True).sort_by([("k", "ascending"), ("s", "ascending")])
    assert prov(lf)["path"] == "streamed"
    assert a.equals(b)
    for name in ("sum_v", "sum_vq", "avg_v"):
        assert a.column(name).to_numpy().tobytes() == b.column(name).to_numpy().tobytes()


def test_reproducible_batches_do_not_depend_on_free_memory(monkeypatch):
    """With no override the budget is read from MemTotal in reproducible mode, not from what is free now."""
    from mxframe import streamed
    monkeypatch.delenv("MXFRAME_HOST_MEM_BUDGET")
    monkeypatch.setattr(streamed, "_meminfo", lambda key: {"MemAvailable": 1 << 30, "MemTotal": 64 << 30}[key])
    a = streamed.host_budget(reproducible=True)
    monkeypatch.setattr(streamed, "_meminfo", lambda key: {"MemAvailable": 40 << 30, "MemTotal": 64 << 30}[key])
    assert a == streamed.host_budget(reproducible=True) == 32 << 30
    assert streamed.host_budget() == int(0.7 * (40 << 30))


# --- what does not stream ---------------------------------------------------------------------------------

def test_a_plan_that_cannot_stream_warns_and_still_runs(tmp_path, monkeypatch):
    tiny(monkeypatch, "20K")
    t = pa.table({"k": np.arange(20_000, dtype=np.int32), "v": np.arange(20_000, dtype=np.float64)})
    path = small_file(tmp_path, t, 1000)
    lf = scan_parquet(path).sort("v", descending=True).limit(3)
    with pytest.warns(RuntimeWarning, match="does not fit in memory .* not streamed: no aggregate"):
        got = lf.compute(device="cpu")
    assert prov(lf)["path"] != "streamed"
    assert got.column("v").to_pylist() == [19999.0, 19998.0, 19997.0]


def test_count_distinct_declines_with_the_reason(tmp_path, monkeypatch):
    tiny(monkeypatch, "20K")
    t = pa.table({"k": (np.arange(20_000) % 3).astype(np.int32), "v": (np.arange(20_000) % 10).astype(np.int32)})
    path = small_file(tmp_path, t, 1000)
    lf = scan_parquet(path).distinct("k", "v").groupby("k").agg(col("v").count().alias("n"))
    with pytest.warns(RuntimeWarning, match="not streamed: a Distinct below the aggregate"):
        got = lf.compute(device="cpu")
    assert sorted((r["k"], r["n"]) for r in got.to_pylist()) == [(0, 10), (1, 10), (2, 10)]


def test_group_state_over_the_budget_raises_a_clear_memory_error(tmp_path, monkeypatch):
    tiny(monkeypatch, "400K")
    n = 100_000
    t = pa.table({"k": np.arange(n, dtype=np.int64), "v": np.ones(n)})          # every row its own group
    lf = scan_parquet(small_file(tmp_path, t, 5000)).groupby("k").agg(col("v").sum().alias("s"))
    with pytest.raises(MemoryError, match=r"group state of [\d,]+ groups .* needs hash-partitioned groups \(F-056 step 3\)"):
        lf.compute(device="cpu")


def test_a_dimension_bigger_than_half_the_budget_declines_with_the_reason(tmp_path, monkeypatch):
    tiny(monkeypatch, "300K")
    n = 40_000
    big = pa.table({"id": np.arange(n, dtype=np.int32), "x": np.ones(n)})
    other = pa.table({"id": (np.arange(n) % n).astype(np.int32), "y": np.ones(n)})
    a, b = str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")
    pq.write_table(big, a, row_group_size=2000)
    pq.write_table(other, b, row_group_size=2000)
    lf = scan_parquet(a).join(scan_parquet(b), on="id").groupby().agg(col("y").sum().alias("s"))
    with pytest.warns(RuntimeWarning, match="not streamed: the other tables"):
        got = lf.compute(device="cpu")
    assert got.column("s").to_pylist() == [float(n)]


# --- memory is bounded ------------------------------------------------------------------------------------

PEAK = textwrap.dedent("""
    import json, sys
    import pyarrow as pa
    from mxframe import col
    from mxframe.io_parquet import scan_parquet

    def status(key):
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) * 1024 for l in f if l.startswith(key + ":"))

    lf = (scan_parquet(sys.argv[1]).filter(col("d") < 2400).groupby("k", "s")
          .agg(col("v").sum().alias("sv"), (col("v") * col("q")).sum().alias("svq"), col("q").mean().alias("aq")))
    lf.schema                                       # imports and warms up before the peak is reset
    lf.compute(device="cpu")
    open("/proc/self/clear_refs", "w").write("5")   # VmHWM starts again from the current size
    base = status("VmRSS")
    out = lf.compute(device=sys.argv[2])
    print(json.dumps({"rows": out.num_rows, "path": lf.last_compile_provenance["path"],
                      "arrow_peak": pa.default_memory_pool().max_memory(), "rss_peak_growth": status("VmHWM") - base}))
""")


def peak(files, device, budget):
    import json
    env = dict(os.environ, MXFRAME_HOST_MEM_BUDGET=budget, MXFRAME_GPU_MEM_BUDGET=GPU)
    out = subprocess.run([sys.executable, "-c", PEAK, files["fact"], device], env=env, capture_output=True, text=True,
                         check=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_peak_memory_is_bounded_by_the_budget_not_the_file(files):
    """Warm both runs (the first compute reads and frees everything once, so the allocator's pages are not
    the difference), then compare the peak of a second compute."""
    decoded = N * (4 + 4 + 8 + 8 + 4 + 1)          # the file's columns as Arrow holds them (~170 MB)
    streamed = peak(files, "cpu", HOST)
    whole = peak(files, "cpu", "64G")
    assert streamed["path"] == "streamed" and whole["path"] != "streamed"
    assert streamed["rows"] == whole["rows"] == 12
    assert streamed["arrow_peak"] < 40 << 20, streamed          # under the 40 MB budget
    assert whole["arrow_peak"] > 0.7 * decoded                  # the contrast: reading it whole holds the file
    assert streamed["rss_peak_growth"] < 0.5 * whole["rss_peak_growth"], (streamed, whole)
