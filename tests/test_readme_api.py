"""README.md (F-048): every code sample runs as written, every API-table example evaluates against
real data, and the headline numbers match the release benchmark files."""
import csv
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mxframe as mx
from mxframe import LazyFrame, col, lit, when

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def _section(title):
    m = re.search(rf"^## [^\n]*{re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", README, re.S | re.M)
    assert m, f"README has no '## {title}' section"
    return m.group(1)


@pytest.fixture
def data(tmp_path, monkeypatch):
    """Small tables with the columns the README's samples and examples use."""
    monkeypatch.chdir(tmp_path)
    lineitem = pa.table({
        "l_returnflag": ["A", "N", "R", "N", "A", "R"],
        "l_quantity": [10.0, 30.0, 5.0, 12.0, 40.0, 23.0],
        "l_extendedprice": [100.0, 300.0, 50.0, 120.0, 400.0, 230.0],
    })
    pq.write_table(lineitem, "lineitem.parquet")
    t = pa.table({
        "id": [1, 2, 3, 4], "dept": ["eng", "ops", "eng", "ops"], "x": [5, 20, 15, 1],
        "price": [10.0, 20.0, 30.0, 40.0], "name": ["alpha", "beta", "gamma", "alfa"],
        "day": pa.array([8000, 9000, 10000, 11000], pa.int32()).cast(pa.date32()),
    })
    other = pa.table({"id": [1, 2, 5], "region": ["eu", "us", "apac"]})
    pq.write_table(t, "t.parquet")
    return {"lf": LazyFrame(t), "lf2": LazyFrame(other), "t": t, "other": other}


def test_code_samples_run(data):
    blocks = re.findall(r"```python\n(.*?)```", README, re.S)
    assert blocks, "README has no python samples"
    ns = {}
    for code in blocks:
        exec(compile(code, "README.md", "exec"), ns)


def _examples():
    rows = []
    for line in _section("API reference").splitlines():
        cells = [c.strip() for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))] if line.startswith("|") else []
        if len(cells) >= 3 and cells[-1].startswith("`") and not set(cells[0]) <= set("-: "):
            rows.append((cells[0], cells[-1].strip("`")))
    return rows


def test_api_tables_have_examples():
    assert len(_examples()) >= 25


@pytest.mark.parametrize("name,example", _examples(), ids=[n for n, _ in _examples()])
def test_api_example_runs(data, name, example):
    ns = {"mx": mx, "col": col, "lit": lit, "when": when, **data}
    out = eval(example, ns)
    if isinstance(out, LazyFrame):
        out = out.compute(device="cpu")
    if isinstance(out, mx.Expr):
        out = data["lf"].select(out.alias("v")).compute(device="cpu")
    assert out is not None


def _total(path, engine, sf):
    with open(ROOT / "docs" / "bench" / path) as f:
        return sum(float(r["first_ms"]) for r in csv.DictReader(f)
                   if r["engine"] == engine and r["sf"] == str(sf)) / 1000


@pytest.mark.parametrize("label,path,sf", [
    ("SF10", "m2s-engine.csv", 10), ("SF1", "m2s-engine.csv", 1),
    ("SF10 shuffled", "m2s-engine-shuffled.csv", 10), ("SF1 shuffled", "m2s-engine-shuffled.csv", 1),
])
def test_headline_numbers_match_bench_files(label, path, sf):
    row = re.search(rf"^\| {re.escape(label)} \|([^\n]+)$", _section("Latest release"), re.M)
    assert row, f"no '{label}' row in the release table"
    shown = [float(v) for v in re.findall(r"([\d.]+) s", row.group(1))]
    want = [_total(path, e, sf) for e in ("mx-cpu", "duckdb", "polars")]
    assert shown == pytest.approx(want, abs=0.006), (label, shown, want)


def test_readme_is_short():
    assert len(README.splitlines()) <= 140
