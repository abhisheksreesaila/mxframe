"""F-135 (M4): Parquet written by the Python-free Mojo package reads back in pyarrow and DuckDB with every value,
type, null and the "omx" metadata, for each codec and several row-group sizes."""
import json
import subprocess
from pathlib import Path

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
WORDS = ["West", "East", "Nörth", "", "South-East"]


@pytest.fixture(scope="module")
def writer(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_write_sample"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_write_sample.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def expected(n):
    rows = []
    for i in range(n):
        rev = None if i % 13 == 4 else (i * 1_000_003 + (1 << 53) if i % 97 == 0 else i * 101 - 50_000)
        rows.append({
            "qty": i * 7 - 3000,
            "revenue": rev,
            "region": None if i % 17 == 2 else WORDS[i % 5],
            "day": 18_000 + i % 400,
            "paid": None if i % 11 == 6 else (i % 3 == 0),
        })
    return rows


@pytest.mark.parametrize("codec", ["zstd", "snappy", "none"])
@pytest.mark.parametrize("n,group", [(0, 1024), (1, 8), (1000, 1024), (5000, 1024)])
def test_pyarrow_and_duckdb_read_what_mojo_wrote(writer, tmp_path, codec, n, group):
    path = tmp_path / "t.parquet"
    subprocess.run([str(writer), str(path), codec, str(n), str(group)], check=True)
    t = pq.read_table(path)
    md = pq.read_metadata(path)
    assert md.num_row_groups == max(1, -(-n // group))
    assert json.loads(t.schema.metadata[b"omx"]) == {"formats": {"revenue": "money"}}
    assert t.schema.field("qty").type == pa.int64()
    assert t.schema.field("revenue").type == pa.decimal128(18, 2)
    assert t.schema.field("region").type == pa.string()
    assert t.schema.field("day").type == pa.date32()
    assert t.schema.field("paid").type == pa.bool_()
    want = expected(n)
    got = t.to_pylist()
    for g, w in zip(got, want):
        assert g["qty"] == w["qty"] and g["region"] == w["region"] and g["paid"] == w["paid"]
        assert (g["revenue"] is None) == (w["revenue"] is None)
        if w["revenue"] is not None:
            assert int(g["revenue"].scaleb(2)) == w["revenue"]          # the exact scaled integer
        assert (g["day"] - __import__("datetime").date(1970, 1, 1)).days == w["day"]
    assert len(got) == n
    con = duckdb.connect()
    s = con.execute(f"SELECT count(*), sum(qty), count(revenue), count(region), sum(CASE WHEN paid THEN 1 ELSE 0 END) "
                    f"FROM read_parquet('{path}')").fetchone()
    assert s[0] == n
    assert (s[1] or 0) == sum(w["qty"] for w in want)
    assert s[2] == sum(w["revenue"] is not None for w in want)
    assert s[3] == sum(w["region"] is not None for w in want)
    assert (s[4] or 0) == sum(bool(w["paid"]) for w in want)


# ── reading files other writers made ─────────────────────────────────────────────────────────────────

import datetime as _dt
import decimal as _dec
import random as _random


@pytest.fixture(scope="module")
def dumper(tmp_path_factory):
    out = tmp_path_factory.mktemp("mojo") / "pq_dump"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(ROOT / "mojo/tools/pq_dump.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


def mojo_read(dumper, path):
    """(columns [(name, kind, scale)], rows of stored values with TEXT as the word)."""
    out = subprocess.run([str(dumper), str(path)], check=True, capture_output=True).stdout.decode("utf-8")
    lines = out.split("\n")
    head = json.loads(lines[0])
    words = {}
    rows = []
    for line in lines[1:]:
        if line.startswith("WORDS "):
            _, c, rest = line.split(" ", 2)
            words[int(c)] = rest.split("\x1f")[:-1]
        elif line:
            rows.append(json.loads(line))
    for r in rows:
        for c, v in enumerate(r):
            if isinstance(v, dict):
                r[c] = words[c][v["word"]]
    return [tuple(c) for c in head["columns"]], rows


def stored(v, t):
    """A pyarrow value as MX frame stores it."""
    if v is None:
        return None
    if pa.types.is_decimal(t):
        return int(v.scaleb(t.scale))
    if pa.types.is_date(t):
        return (v - _dt.date(1970, 1, 1)).days
    if pa.types.is_timestamp(t):
        return (v.replace(tzinfo=None) - _dt.datetime(1970, 1, 1)) // _dt.timedelta(days=1)
    if pa.types.is_boolean(t):
        return int(v)
    return v


def sample(n, seed=3):
    r = _random.Random(seed)
    def maybe(v, k):
        return None if r.random() < k else v
    return pa.table({
        "i64": pa.array([maybe(r.randint(-2**62, 2**62), 0.1) for _ in range(n)], pa.int64()),
        "i32": pa.array([r.randint(-2**31, 2**31 - 1) for _ in range(n)], pa.int32()),
        "dec2": pa.array([maybe(_dec.Decimal(r.randint(-10**9, 10**9)).scaleb(-2), 0.2) for _ in range(n)], pa.decimal128(12, 2)),
        "dec4": pa.array([_dec.Decimal(r.randint(-10**17, 10**17)).scaleb(-4) for _ in range(n)], pa.decimal128(18, 4)),
        "d": pa.array([maybe(_dt.date(1960, 1, 1) + _dt.timedelta(days=r.randint(0, 40000)), 0.05) for _ in range(n)], pa.date32()),
        "ts_us": pa.array([_dt.datetime(1950, 1, 1) + _dt.timedelta(seconds=r.randint(0, 3 * 10**9)) for _ in range(n)], pa.timestamp("us")),
        "ts_ms": pa.array([_dt.datetime(2020, 1, 1) + _dt.timedelta(seconds=r.randint(-10**8, 10**8)) for _ in range(n)], pa.timestamp("ms")),
        "ts_ns": pa.array([_dt.datetime(2001, 1, 1) + _dt.timedelta(seconds=r.randint(0, 10**8)) for _ in range(n)], pa.timestamp("ns")),
        "b": pa.array([maybe(r.random() < 0.5, 0.1) for _ in range(n)], pa.bool_()),
        "s": pa.array([maybe(r.choice(["West", "East", "", "Ünïcode ✓", "x" * 300]), 0.1) for _ in range(n)], pa.string()),
        "s_many": pa.array([f"key-{r.randint(0, n)}" for _ in range(n)], pa.string()),
    })


def check_equal(dumper, path, table):
    cols, rows = mojo_read(dumper, path)
    assert [c[0] for c in cols] == table.column_names
    want = table.to_pylist()
    assert len(rows) == len(want)
    for i, (g, w) in enumerate(zip(rows, want)):
        for c, name in enumerate(table.column_names):
            assert g[c] == stored(w[name], table.schema.field(name).type), (i, name, g[c], w[name])


@pytest.mark.parametrize("opts", [
    dict(compression="zstd"),
    dict(compression="snappy"),
    dict(compression="none"),
    dict(compression="zstd", use_dictionary=False),
    dict(compression="snappy", data_page_version="2.0"),
    dict(compression="zstd", data_page_version="2.0", use_dictionary=False),
    dict(compression="none", use_dictionary=False, column_encoding={"i64": "DELTA_BINARY_PACKED", "i32": "DELTA_BINARY_PACKED"}),
    dict(compression="zstd", row_group_size=777, data_page_size=4096),
])
def test_mojo_reads_what_pyarrow_wrote(dumper, tmp_path, opts):
    t = sample(5000)
    path = tmp_path / "p.parquet"
    pq.write_table(t, path, **opts)
    check_equal(dumper, path, t)


def test_mojo_reads_what_duckdb_wrote(dumper, tmp_path):
    t = sample(3000, seed=9)
    path = tmp_path / "d.parquet"
    con = duckdb.connect()
    con.register("t", t)
    con.execute(f"COPY t TO '{path}' (FORMAT parquet)")
    back = pq.read_table(path)            # DuckDB's own types for the same values (its timestamps, decimals)
    check_equal(dumper, path, back)


def test_round_trip_keeps_the_omx_metadata(writer, dumper, tmp_path):
    path = tmp_path / "rt.parquet"
    subprocess.run([str(writer), str(path), "zstd", "2000", "512"], check=True)
    cols, rows = mojo_read(dumper, path)
    assert cols == [("qty", "INT", 0), ("revenue", "DECIMAL", 2), ("region", "TEXT", 0), ("day", "DATE", 0), ("paid", "BOOL", 0)]
    want = expected(2000)
    for g, w in zip(rows, want):
        assert g == [w["qty"], w["revenue"], w["region"], w["day"], None if w["paid"] is None else int(w["paid"])]


def test_unsupported_columns_raise_readably(dumper, tmp_path):
    path = tmp_path / "f.parquet"
    pq.write_table(pa.table({"x": pa.array([1.5, 2.5])}), path)
    p = subprocess.run([str(dumper), str(path)], capture_output=True, text=True)
    assert p.returncode != 0 and "column `x` is DOUBLE: MX frame has no kind for it yet" in (p.stdout + p.stderr)
