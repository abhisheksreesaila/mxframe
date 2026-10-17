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
