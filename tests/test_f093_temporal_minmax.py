"""F-093: MIN / MAX of date and timestamp columns return the input type (DuckDB), on every path.
Expected answers come from DuckDB tables."""
import datetime as dt

import duckdb
import pyarrow as pa
import pytest

import omxcore as mx
from omxcore import LazyFrame, col

DEVICES = ["cpu", "gpu"]
D = dt.date
BIG = 2 ** 53 + 1          # in ns: past float64's exact range


def _types():
    return {
        "date32": pa.date32(), "date64": pa.date64(),
        "ts_s": pa.timestamp("s"), "ts_ms": pa.timestamp("ms"), "ts_us": pa.timestamp("us"),
        "ts_ns": pa.timestamp("ns"), "ts_tz": pa.timestamp("us", tz="UTC"),
    }


def _table(kind):
    ty = _types()[kind]
    if pa.types.is_date(ty):
        vals = [D(2020, 1, 5), D(2021, 3, 1), None, D(1999, 12, 31), D(2020, 1, 4), None, D(2030, 6, 7)]
    else:
        base = 1_600_000_000
        vals = [base + 5, base + 90, None, base - 7, base + 1, None, base + 4000]
        vals = [None if v is None else dt.datetime(1970, 1, 1) + dt.timedelta(seconds=v) for v in vals]
        if ty.tz:
            vals = [None if v is None else v.replace(tzinfo=dt.timezone.utc) for v in vals]
    return pa.table({"g": pa.array([1, 1, 1, 2, 2, 3, 3], pa.int64()),
                     "d": pa.array(vals, ty),
                     "k": pa.array([1, 2, 3, 4, 5, 6, 7], pa.int64())})


def _duck(q, **tables):
    con = duckdb.connect()
    for n, t in tables.items():
        con.register(n, t)
    return con.execute(q).to_arrow_table()


def _same(got, want, by, src=None):
    # DuckDB has no date64 and shows a tz column in its session zone; we keep the input type (the ticket's
    # rule), so the DuckDB answer is compared after casting its date / timestamp columns to ours.
    want = pa.table({n: want.column(n).cast(got.schema.field(n).type)
                     if pa.types.is_temporal(want.schema.field(n).type) else want.column(n) for n in want.column_names})
    if src is not None:         # the result is the input column's own type, whatever DuckDB says
        for n in want.column_names:
            if n != "g":
                assert got.schema.field(n).type == src, n
    if by:
        got, want = got.sort_by(by), want.sort_by(by)
    assert got.column_names == want.column_names
    for n in want.column_names:
        assert got.schema.field(n).type == want.schema.field(n).type, n
    assert got.to_pylist() == want.to_pylist()


KINDS = list(_types())


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["min", "max"])
@pytest.mark.parametrize("kind", KINDS)
def test_grouped_minmax(kind, fn, device):
    t = _table(kind)
    want = _duck(f"SELECT g, {fn}(d) AS m FROM t GROUP BY g", t=t)
    got = LazyFrame(t).groupby("g").agg(getattr(col("d"), fn)().alias("m")).compute(device=device)
    _same(got, want, [("g", "ascending")], src=t.schema.field("d").type)
    got = mx.sql(f"SELECT g, {fn}(d) AS m FROM t GROUP BY g", t=t).compute(device=device)
    _same(got, want, [("g", "ascending")], src=t.schema.field("d").type)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["min", "max"])
@pytest.mark.parametrize("kind", KINDS)
def test_global_minmax(kind, fn, device):
    t = _table(kind)
    want = _duck(f"SELECT {fn}(d) AS m FROM t", t=t)
    got = LazyFrame(t).groupby().agg(getattr(col("d"), fn)().alias("m")).compute(device=device)
    _same(got, want, [], src=t.schema.field("d").type)
    _same(mx.sql(f"SELECT {fn}(d) AS m FROM t", t=t).compute(device=device), want, [], src=t.schema.field("d").type)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["date32", "ts_us"])
def test_all_null_group_and_empty(kind, device):
    t = _table(kind)
    q = "SELECT g, min(d) AS lo, max(d) AS hi FROM t WHERE k IN (3, 6) GROUP BY g"
    _same(mx.sql(q, t=t).compute(device=device), _duck(q, t=t), [("g", "ascending")], src=t.schema.field("d").type)
    q = "SELECT min(d) AS lo, max(d) AS hi FROM t WHERE k > 100"
    _same(mx.sql(q, t=t).compute(device=device), _duck(q, t=t), [], src=t.schema.field("d").type)


@pytest.mark.parametrize("device", DEVICES)
def test_having(device):
    t = _table("date32")
    lit = "DATE '2020-01-05'"
    for q in (f"SELECT g, min(d) AS lo FROM t GROUP BY g HAVING min(d) > {lit}",
              f"SELECT g, max(d) AS hi FROM t GROUP BY g HAVING max(d) < {lit}",
              f"SELECT g FROM t GROUP BY g HAVING min(d) > {lit}"):
        _same(mx.sql(q, t=t).compute(device=device), _duck(q, t=t), [("g", "ascending")], src=t.schema.field("d").type)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["date32", "ts_ns"])
def test_through_join(kind, device):
    t = _table(kind)
    u = pa.table({"k": pa.array([1, 2, 3, 4, 5, 6], pa.int64()), "w": pa.array([1, 2, 3, 4, 5, 6], pa.int64())})
    q = "SELECT t.g AS g, min(d) AS lo, max(d) AS hi FROM t JOIN u ON t.k = u.k GROUP BY t.g"
    _same(mx.sql(q, t=t, u=u).compute(device=device), _duck(q, t=t, u=u), [("g", "ascending")], src=t.schema.field("d").type)
    q = "SELECT min(d) AS lo, max(d) AS hi FROM t JOIN u ON t.k = u.k"
    _same(mx.sql(q, t=t, u=u).compute(device=device), _duck(q, t=t, u=u), [], src=t.schema.field("d").type)


@pytest.mark.parametrize("device", DEVICES)
def test_date_min_with_big_int_max_and_sum(device):
    big = 2 ** 53 + 3
    t = pa.table({"g": pa.array([1, 1, 2], pa.int64()),
                  "d": pa.array([D(2020, 1, 5), D(2019, 1, 1), D(2021, 1, 1)], pa.date32()),
                  "x": pa.array([big, big + 2, big + 4], pa.int64())})
    q = "SELECT g, min(d) AS lo, max(x) AS hx, sum(x) AS sx FROM t GROUP BY g"
    got = mx.sql(q, t=t).compute(device=device).sort_by("g")
    want = _duck(q, t=t).sort_by("g")
    assert got.schema.field("lo").type == pa.date32() and got.schema.field("hx").type == pa.int64()
    assert got.to_pylist() == [{**r, "sx": int(r["sx"])} for r in want.to_pylist()]
    q = "SELECT min(d) AS lo, max(x) AS hx, sum(x) AS sx FROM t"
    got, want = mx.sql(q, t=t).compute(device=device), _duck(q, t=t)
    assert got.to_pylist() == [{**r, "sx": int(r["sx"])} for r in want.to_pylist()]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("fn", ["min", "max"])
def test_timestamps_past_2_53_ns(fn, device):
    ns = [BIG, BIG + 2, BIG + 1, None, BIG + 3, BIG - 1]
    t = pa.table({"g": pa.array([1, 1, 1, 1, 2, 2], pa.int64()), "d": pa.array(ns, pa.timestamp("ns"))})
    q = f"SELECT g, {fn}(d) AS m FROM t GROUP BY g"
    want = _duck(q, t=t)
    got = mx.sql(q, t=t).compute(device=device)
    _same(got, want, [("g", "ascending")], src=t.schema.field("d").type)
    assert sorted(got.column("m").cast(pa.int64()).to_pylist()) == sorted(want.column("m").cast(pa.int64()).to_pylist())
    q = f"SELECT {fn}(d) AS m FROM t"
    got = mx.sql(q, t=t).compute(device=device)
    assert got.column("m").cast(pa.int64()).to_pylist() == _duck(q, t=t).column("m").cast(pa.int64()).to_pylist()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("kind", ["date32", "ts_ns"])
def test_no_nulls_and_parquet_scan(kind, device, tmp_path):
    t = _table(kind).filter(pa.array([True, True, False, True, True, False, True]))
    assert t.column("d").null_count == 0
    q = "SELECT g, min(d) AS lo, max(d) AS hi FROM t GROUP BY g"
    want = _duck(q, t=t)
    _same(mx.sql(q, t=t).compute(device=device), want, [("g", "ascending")], src=t.schema.field("d").type)
    import pyarrow.parquet as pq
    pq.write_table(t, tmp_path / "t.parquet")
    lf = mx.scan_parquet(str(tmp_path / "t.parquet"))
    got = lf.groupby("g").agg(col("d").min().alias("lo"), col("d").max().alias("hi")).compute(device=device)
    _same(got, want, [("g", "ascending")], src=t.schema.field("d").type)
