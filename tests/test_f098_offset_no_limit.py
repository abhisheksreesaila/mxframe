"""F-098: OFFSET without LIMIT is a `Limit` with `n=None`, not a made-up row count.

Before: the SQL front end wrote `Limit(n=2**62)` (`explain()` printed 4611686018427387904) and an OFFSET near
2^63 overflowed `n + offset` in int64 on the way to the native kernels / the top-n paths.

Expected answers come from a NATIVE DuckDB table, on cpu and gpu, native executor on and off. The streamed cases
(F-056 step 3a sinks) run over a Parquet file bigger than an artificial host budget.
"""
import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
import mxframe.native_exec as nx
from mxframe import col, gpu_stream
from mxframe.io_parquet import scan_parquet
from mxframe.lazy_frame import Limit

GPU = gpu_stream.cuda_available()
N = 400


def _t():
    r = np.random.default_rng(5)
    b = r.integers(0, 12, N)
    return pa.table({
        "i": pa.array(np.arange(N), pa.int64()),
        "b": pa.array([None if x == 11 else int(x) for x in b], pa.int64()),
        "c": pa.array(r.integers(0, 50, N), pa.int64()),
        "k": pa.array(np.array(list("xyz"))[r.integers(0, 3, N)].tolist(), pa.string()),
        "v": pa.array(r.random(N) * 100, pa.float64()),
    })


def _w():
    return pa.table({"k": pa.array(["x", "y", "z"], pa.string()), "x": pa.array([10, 20, 30], pa.int64())})


T, W = _t(), _w()
CON = duckdb.connect()
CON.register("_t", T)
CON.register("_w", W)
CON.execute("create table t as select * from _t")
CON.execute("create table w as select * from _w")


@pytest.fixture(params=[(dev, nat) for dev in ("cpu", "gpu") for nat in ("1", "0")],
                ids=lambda p: f"{p[0]}-native{p[1]}")
def cfg(request, monkeypatch):
    dev, nat = request.param
    if dev == "gpu" and not GPU:
        pytest.skip("no CUDA GPU")
    if nat == "1" and not nx.available():
        pytest.skip("libmxexec_cpu.so not built")
    monkeypatch.setenv("MXFRAME_NATIVE", nat)
    return dev


HUGE = [2 ** 62 - 1, 2 ** 62, 2 ** 62 + 1, 2 ** 63 - 1]            # (DuckDB itself refuses an OFFSET past int64)
BEYOND = [2 ** 63, 2 ** 64, 10 ** 20]

QUERIES = (
    [f"select i from t order by i offset {h}" for h in HUGE]
    + [f"select i from t order by c, i limit 5 offset {h}" for h in HUGE]
    + [f"select * from t offset {h}" for h in HUGE[:3]]
    + [
        "select i from t order by c, i offset 390",
        "select i from t offset 395",
        "select i from t where c > 10 order by c desc, i offset 350",
        "select i from t order by i offset 400",                       # exactly all rows
        "select i from t order by i offset 399",
        "select i from t order by b, i offset 395",                    # NULL keys
        "select i, b from t order by b desc, i offset 3",
        "select k, sum(c) as s from t group by k order by k offset 1",
        "select k, count(*) as n from t group by k order by n desc, k offset 2",
        "select k, sum(c) as s from t group by k order by s, k offset 5",
        "select distinct k from t order by k offset 1",
        "select t.i, w.x from t join w on t.k = w.k order by t.i offset 390",
        "select i from (select i, c from t order by c, i offset 380) order by i desc",
        "select count(*) as n from (select i from t order by i offset 300)",
        "select i from t order by i offset 5 limit 3",                 # (either clause order)
        "select i from (select i from t order by i offset 390) order by i desc limit 2",
        "select i from t order by i limit 3 offset 2",                 # with a LIMIT: unchanged
    ]
)


@pytest.mark.parametrize("q", QUERIES)
def test_matches_duckdb(cfg, q):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T, w=W).compute(device=cfg)
    assert got.column_names == want.column_names
    assert got.to_pydict() == want.to_pydict(), q


def test_huge_offset_returns_no_rows():
    for h in HUGE + BEYOND:
        for q in (f"select i from t order by i offset {h}", f"select i from t order by c, i limit 5 offset {h}",
                  f"select * from t offset {h}"):
            out = mx.sql(q, t=T).compute()
            assert out.num_rows == 0, q
            assert out.column_names == CON.execute(q.replace(str(h), "1")).to_arrow_table().column_names


def test_no_limit_is_none_and_explain_shows_no_sentinel():
    lf = mx.sql("select i from t order by i offset 7", t=T)
    node = lf.plan
    while not isinstance(node, Limit):
        node = node.input
    assert node.n is None and node.offset == 7
    text = lf.explain()
    assert "Limit(offset=7)" in text
    assert "4611686018427387904" not in text and "n=" not in text.split("Limit(")[1].split("\n")[0]
    assert "Limit(n=3, offset=2)" in mx.sql("select i from t order by i limit 3 offset 2", t=T).explain()


def test_api_limit_none_keeps_the_rest():
    out = mx.LazyFrame.from_arrow(T).sort("i").limit(None, 395).compute() if hasattr(mx.LazyFrame, "from_arrow") \
        else mx.sql("select i from t order by i", t=T).limit(None, 395).compute()
    assert out.column("i").to_pylist() == [395, 396, 397, 398, 399]


def test_optimizer_merges_limits_with_none():
    lf = mx.sql("select i from (select i from t order by i limit 10) limit 4", t=T)
    assert lf.compute().column("i").to_pylist() == [0, 1, 2, 3]
    lf = mx.sql("select i from (select i from t order by i limit 10) offset 0", t=T)
    assert lf.compute().column("i").to_pylist() == list(range(10))


# ---- streamed sinks over a file bigger than the host budget -----------------------------------------------------

M = 120_000


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    d = tmp_path_factory.mktemp("f098")
    idx = np.arange(M)
    t = pa.table({"id": idx.astype(np.int64), "g": (idx % 97).astype(np.int64),
                  "v": (idx * 31 % 1000).astype(np.int64)})
    path = str(d / "big.parquet")
    pq.write_table(t, path, row_group_size=10_000)
    return path


def _duck_big(q, path):
    con = duckdb.connect()
    con.execute(f"create view big as select * from read_parquet('{path}')")
    return con.execute(q).to_arrow_table()


@pytest.mark.parametrize("q", [
    "select id from big where v > 990 order by id offset 1000",
    "select id from big where v > 990 order by v desc, id offset 3",
    "select id from big where v > 990 order by id offset 1000000",
    "select id from big where v > 990 order by id offset 9223372036854775807",
    "select id from big where v > 990 order by id limit 4 offset 9223372036854775807",
    "select id, v from big where id > 119990 offset 4",
    "select g, sum(v) as s from big group by g order by g offset 90",
    "select id from big where v > 990 order by id limit 5 offset 2",
])
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not GPU, reason="no CUDA GPU"))])
def test_streamed_offset_without_limit(big, monkeypatch, q, device):
    monkeypatch.setenv("MXFRAME_HOST_MEM_BUDGET", "6M")
    want = _duck_big(q, big)
    got = mx.sql(q, big=scan_parquet(big)).compute(device=device)
    assert got.column_names == want.column_names
    assert got.to_pydict() == want.to_pydict(), q


def test_offset_zero_without_limit_keeps_every_row():
    """DuckDB 1.5.1 returns NO rows for `ORDER BY .. OFFSET 0` over a native table (its STREAMING_LIMIT of 0), so this is
    not checked against it: OFFSET 0 skips nothing."""
    out = mx.sql("select i from t order by c desc, i offset 0", t=T).compute()
    assert out.num_rows == N
    assert out.column("i").to_pylist() == CON.execute("select i from t order by c desc, i").to_arrow_table().column("i").to_pylist()


# ---- a derived table with OFFSET and no LIMIT inside a cross / theta join (the F-096 cap reads its row count) --------

U = pa.table({"k": pa.array([1, 2], pa.int64())})
CON.register("_u", U)
CON.execute("create table u as select * from _u")

JOINS = [
    "select a.i, u.k from (select i from t order by i offset 397) a cross join u",
    "select a.i, u.k from (select i from t order by i offset 397) a, u where a.i > u.k",
    "select a.i, u.k from (select i from t order by i offset 397) a join u on a.i > u.k",
    "select i, k from u, (select i from t order by i offset 398) a",
    "select a.i, u.k from (select i from t order by i offset 1000) a cross join u",
]


@pytest.mark.parametrize("q", JOINS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("gpu", marks=pytest.mark.skipif(not GPU, reason="no CUDA GPU")),
                                    pytest.param("auto", marks=pytest.mark.skipif(not GPU, reason="no CUDA GPU"))])
def test_join_over_an_offset_only_derived_table(q, device):
    want = CON.execute(q).to_arrow_table()
    got = mx.sql(q, t=T, u=U).compute(device=device)
    rows = lambda x: sorted(zip(*(x.column(i).to_pylist() for i in range(x.num_columns))))     # noqa: E731
    assert got.column_names == want.column_names
    assert rows(got) == rows(want), q
