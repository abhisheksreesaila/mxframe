"""README.md and docs/reference.md (F-048; the README became a short front door that points to the story, and its
API tables, release numbers and install moved to docs/reference.md): every code sample in both runs as written, every
API-table example evaluates against real data, and the headline numbers match the release benchmark files."""
import csv
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import omxcore as mx
from omxcore import LazyFrame, col, lit, when

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
REFERENCE = (ROOT / "docs" / "reference.md").read_text()


def _section(title):
    m = re.search(rf"^## [^\n]*{re.escape(title)}[^\n]*\n(.*?)(?=^## |\Z)", REFERENCE, re.S | re.M)
    assert m, f"docs/reference.md has no '## {title}' section"
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
    for name, text in (("README.md", README), ("docs/reference.md", REFERENCE)):
        blocks = re.findall(r"```python\n(.*?)```", text, re.S)
        assert blocks, f"{name} has no python samples"
        ns = {}
        for code in blocks:
            exec(compile(code, name, "exec"), ns)


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


def test_readme_is_short_and_points_to_the_story():
    assert len(README.splitlines()) <= 60
    assert "story.html" in re.split(r"^## ", README, flags=re.M)[0], "the story link comes before the first section"
    assert "docs/reference.md" in README


def _rows(path, engine, sf):
    with open(ROOT / "docs" / "bench" / path) as f:
        return {r["query"]: float(r["first_ms"]) for r in csv.DictReader(f)
                if r["engine"] == engine and r["sf"] == str(sf)}


def test_release_prose_numbers_match_bench_files():
    text = _section("Latest release")
    for path, label in (("m2s-engine.csv", "2.27 s to 2.41 s at SF10"), ("m2s-engine-shuffled.csv", "2.79 s to 3.63 s at SF10 shuffled")):
        dk = sum(v for q, v in _rows(path, "duckdb", 10).items() if q != "21") / 1000
        mx_ = sum(v for q, v in _rows(path, "mx-cpu", 10).items() if q != "21") / 1000
        assert f"{dk:.2f} s to {mx_:.2f} s" in label and label in text, (path, dk, mx_)
    dk, mxr = _rows("m2s-engine.csv", "duckdb", 10), _rows("m2s-engine.csv", "mx-cpu", 10)
    assert f"faster on {sum(dk[q] < mxr[q] for q in dk)} of the 22 SF10 queries" in text
    for run, n in (("sf10", 50), ("sf1", 100), ("sf1-shuffled", 100)):
        all_row = re.search(r"^\| all \| (\d+) \|(?:[^|]*\|){4} ([\d.]+x) \|", (ROOT / "docs" / "bench" / f"random-{run}-m2s.md").read_text(), re.M)
        assert int(all_row.group(1)) == n and f"{all_row.group(2)} at {run.upper().replace('-SHUFFLED', ' shuffled')} ({n}" in text, (run, all_row.groups())


# what the examples must return on the fixture data (rows, or a column's values)
EXPECTED = {
    "`.filter(expr)`": 2, "`.join(other, on, how)`": 4, "`.limit(n)` / `.head(n)`": 2,
    "`.distinct(*subset)`": 2, "`.groupby(*keys).agg(*aggs)`": 2, "`mx.sql(query, **tables)`": 2,
    "COUNT(DISTINCT b) per a": {"n": [2, 2]}, "`.isin(values)`": {"v": [True, False, True, False]},
    "`.between(lo, hi)`": {"v": [True, False, True, False]}, "`.str_slice(offset, length)`": {"v": ["al", "be", "ga", "al"]},
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_api_example_results(data, name):
    example = dict(_examples())[name]
    ns = {"mx": mx, "col": col, "lit": lit, "when": when, **data}
    out = eval(example, ns)
    out = out if not isinstance(out, mx.Expr) else data["lf"].select(out.alias("v"))
    table = out.compute(device="cpu")
    want = EXPECTED[name]
    if isinstance(want, int):
        assert table.num_rows == want
    else:
        for c, vals in want.items():
            got = table.column(c).to_pylist()
            assert (sorted(got) if c == "n" else got) == vals


def test_quickstart_runs_on_the_native_executor(data):
    """The README's numbers are the native executor's; its quickstart must take that path."""
    from omxcore import native_exec
    code = re.findall(r"```python\n(.*?)```", REFERENCE, re.S)[0]
    ns = {}
    exec(compile(code, "README.md", "exec"), ns)
    assert native_exec.last_path() == "native", native_exec.last_decline()
