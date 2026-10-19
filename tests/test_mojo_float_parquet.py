"""F-157: FLOAT64 as a column kind, Parquet side. Mojo reads DOUBLE and FLOAT columns (FLOAT widened to double), keeps every bit
(-0.0, NaN payloads, subnormals) and writes DOUBLE back; a file with a float column opens even when a query never touches it."""
import json
import struct
import subprocess
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from f157_data import bits, random_doubles, SPECIALS

ROOT = Path(__file__).resolve().parent.parent


def build(tmp_path_factory, name):
    out = tmp_path_factory.mktemp("mojo") / name
    subprocess.run(["bash", str(ROOT / f"scripts/mojo_build.sh"), str(ROOT / f"mojo/tools/{name}.mojo"), str(out)],
                   check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope="module")
def dumper(tmp_path_factory):
    return build(tmp_path_factory, "pq_dump")


@pytest.fixture(scope="module")
def copier(tmp_path_factory):
    return build(tmp_path_factory, "pq_copy")


def mojo_read(dumper, path):
    out = subprocess.run([str(dumper), str(path)], check=True, capture_output=True).stdout.decode("utf-8")
    lines = out.split("\n")
    head = json.loads(lines[0])
    rows = [json.loads(line) for line in lines[1:] if line and not line.startswith("WORDS ")]
    return [tuple(c) for c in head["columns"]], rows


def as_bits(v):
    """A pyarrow float (python float, or None) as the signed int64 Mojo stores."""
    return None if v is None else bits(float(np.float64(v)))


def float_table(n, seed):
    d = random_doubles(n, seed)
    f = [None if v is None else v for v in random_doubles(n, seed + 1, spread=1e4)]
    # a float32 column must hold float32-representable values (or the cast would round): cast through numpy, huge ones become inf
    with np.errstate(over="ignore"):
        f32 = [None if v is None else float(np.float32(v)) for v in f]
    return pa.table({"id": pa.array(range(n), pa.int64()), "d": pa.array(d, pa.float64()), "f": pa.array(f32, pa.float32())})


@pytest.mark.parametrize("opts", [
    dict(compression="zstd"),
    dict(compression="snappy", use_dictionary=False),
    dict(compression="none", data_page_version="2.0"),
    dict(compression="zstd", use_dictionary=False, row_group_size=777, data_page_size=2048),
    dict(compression="none", use_dictionary=True, dictionary_pagesize_limit=512),
])
def test_mojo_reads_pyarrow_doubles_and_floats_bit_for_bit(dumper, tmp_path, opts):
    t = float_table(5000, 7)
    path = tmp_path / "f.parquet"
    pq.write_table(t, path, **opts)
    cols, rows = mojo_read(dumper, path)
    assert cols == [("id", "INT", 0), ("d", "FLOAT", 0), ("f", "FLOAT", 32)]
    want = t.to_pydict()
    assert len(rows) == 5000
    for i, row in enumerate(rows):
        assert row[1] == as_bits(want["d"][i]), (i, row[1], want["d"][i])
        assert row[2] == as_bits(want["f"][i]), (i, row[2], want["f"][i])


def test_a_file_of_duckdb_doubles_opens(dumper, tmp_path):
    path = tmp_path / "d.parquet"
    con = duckdb.connect()
    con.execute(f"COPY (SELECT i AS id, i / 7.0 AS d, CASE WHEN i % 5 = 0 THEN NULL ELSE i * CAST(0.1 AS DOUBLE) END AS e FROM range(3000) t(i)) TO '{path}' (FORMAT parquet)")
    cols, rows = mojo_read(dumper, path)
    assert cols == [("id", "INT", 0), ("d", "FLOAT", 0), ("e", "FLOAT", 0)]
    got = duckdb.connect().execute(f"SELECT d, e FROM read_parquet('{path}')").fetchall()
    for row, (d, e) in zip(rows, got):
        assert row[1] == as_bits(d) and row[2] == as_bits(e)


def test_mojo_writes_double_and_pyarrow_duckdb_read_it_back(dumper, copier, tmp_path):
    t = float_table(3500, 11)
    src, dst = tmp_path / "src.parquet", tmp_path / "dst.parquet"
    pq.write_table(t, src, use_dictionary=False)
    subprocess.run([str(copier), str(src), str(dst)], check=True)
    back = pq.read_table(dst)
    assert back.schema.field("d").type == pa.float64()
    assert back.schema.field("f").type == pa.float64()                         # FLOAT is written as DOUBLE
    for name in ("d", "f"):
        want = t[name].to_pylist()
        got = back[name].to_pylist()
        assert [as_bits(v) for v in got] == [as_bits(v) for v in want], name   # every bit, NaN payloads and -0.0 included
    rows = duckdb.connect().execute(f"SELECT count(*), count(d), sum(CASE WHEN isfinite(d) THEN d ELSE 0 END) FROM read_parquet('{dst}')").fetchone()
    assert rows[0] == 3500 and rows[1] == sum(v is not None for v in t["d"].to_pylist())


def test_a_double_column_nobody_asks_about_does_not_stop_the_file_opening(dumper, tmp_path):
    path = tmp_path / "mix.parquet"
    pq.write_table(pa.table({"q": pa.array([1, 2, 3], pa.int64()), "price": pa.array([1.5, float("nan"), None])}), path)
    cols, rows = mojo_read(dumper, path)
    assert cols[1] == ("price", "FLOAT", 0)
    assert rows[0][1] == bits(1.5) and rows[2][1] is None


def test_byte_stream_split_doubles_and_floats_read(dumper, tmp_path):
    t = float_table(2500, 5)
    path = tmp_path / "bss.parquet"
    pq.write_table(t, path, use_dictionary=False, use_byte_stream_split=["d", "f"])
    cols, rows = mojo_read(dumper, path)
    want = t.to_pydict()
    for i, row in enumerate(rows):
        assert row[1] == as_bits(want["d"][i]) and row[2] == as_bits(want["f"][i])
