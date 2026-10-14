"""F-049: the benchmarks page reads only bench files that exist, and its totals logic agrees with the data.

docs/site/js/bench.js keeps its logic free of the DOM; these tests run it under node and compare it with a
plain-Python reading of the same CSV / JSON, and with the summaries printed in the random-suite .md files."""
import csv
import json
import math
import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
BENCH = ROOT / "docs" / "bench"
PAGE = SITE / "benchmarks.html"
JS = SITE / "js" / "bench.js"


def _rows(name):
    return list(csv.DictReader(open(BENCH / name)))


def _total(rows, engine, sf, skip=None):
    return sum(float(r["first_ms"]) for r in rows
               if r["engine"] == engine and r["sf"] == str(sf) and r["query"] != str(skip))


def test_page_and_js_name_only_existing_bench_files():
    text = PAGE.read_text() + JS.read_text()
    names = set(re.findall(r"bench/([\w.-]+\.(?:csv|json))", text))
    assert {"m2s-engine.csv", "m2s-engine-shuffled.csv", "random-sf1-m2s.json",
            "random-sf1-shuffled-m2s.json", "random-sf10-m2s.json"} <= names
    for n in names:
        assert (BENCH / n).is_file(), n


def test_page_shell_and_sections():
    html = PAGE.read_text()
    assert 'class="read"' in html and 'src="js/bench.js"' in html
    assert html.count("<details>") >= 3, "a data table under each chart"
    assert "snapper" in html


def test_known_sf10_totals_in_the_csv():
    rows = _rows("m2s-engine.csv")
    assert _total(rows, "mx-cpu", 10) / 1000 == pytest.approx(3.15, abs=0.005)
    assert _total(rows, "duckdb", 10) / 1000 == pytest.approx(5.08, abs=0.005)
    assert _total(rows, "polars", 10) / 1000 == pytest.approx(14.40, abs=0.005)
    # the honest sentence on the page: without Q21 DuckDB's total is the lower one at SF10
    assert _total(rows, "duckdb", 10, skip=21) < _total(rows, "mx-cpu", 10, skip=21)
    wrong = [(r["engine"], r["sf"], r["query"]) for r in rows if r["verdict"] == "wrong"]
    assert wrong == [("polars", "10", "1")]


node = pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")


def _run_js(body):
    script = f"const b = require({str(JS)!r}); const fs = require('fs'); {body}"
    out = subprocess.run(["node", "-e", script], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@node
@pytest.mark.parametrize("name", ["m2s-engine.csv", "m2s-engine-shuffled.csv"])
@pytest.mark.parametrize("sf", [1, 10])
def test_js_totals_match_python(name, sf):
    got = _run_js(f"""
      const [head, ...lines] = fs.readFileSync({str(BENCH / name)!r}, 'utf8').trim().split(/\\r?\\n/);
      const cols = head.split(',');
      const rows = lines.map(l => Object.fromEntries(l.split(',').map((v, i) => [cols[i], v])));
      const m = b.queryMatrix(rows, {sf});
      const o = {{}};
      for (const e of ['mx-cpu', 'duckdb', 'polars']) o[e] = [b.total(m, e), b.total(m, e, 21)];
      o.wrong = b.wrongQueries(m, 'polars'); o.n = m.queries.length;
      console.log(JSON.stringify(o));""")
    rows = _rows(name)
    for e in ("mx-cpu", "duckdb", "polars"):
        assert got[e][0] == pytest.approx(_total(rows, e, sf))
        assert got[e][1] == pytest.approx(_total(rows, e, sf, skip=21))
    assert got["n"] == 22
    assert got["wrong"] == ([1] if sf == 10 else [])


def _md_ratios(name):
    """shape -> DuckDB / MXFrame (geo. mean) as printed in the run's summary table."""
    out = {}
    for line in (BENCH / name).read_text().splitlines():
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and cells[0] not in ("shape", "---") and cells[6].endswith("x"):
            out[cells[0]] = float(cells[6][:-1])
    return out


@node
@pytest.mark.parametrize("stem", ["random-sf1-m2s", "random-sf1-shuffled-m2s", "random-sf10-m2s"])
def test_js_random_shapes_match_the_md_summary(stem):
    got = _run_js(f"""
      const res = JSON.parse(fs.readFileSync({str(BENCH / (stem + '.json'))!r}, 'utf8'));
      console.log(JSON.stringify(b.randomShapes(res)));""")
    want = _md_ratios(stem + ".md")
    assert want and {s["shape"] for s in got} == set(want)
    for s in got:
        assert s["ratio"] == pytest.approx(want[s["shape"]], abs=0.006), s["shape"]

    # and a plain-Python reading of the JSON agrees for the whole run
    res = json.load(open(BENCH / (stem + ".json")))
    logs = [math.log(r["duckdb"]["ms"] / r["mx"]["ms"]) for r in res]
    assert next(s for s in got if s["shape"] == "all")["ratio"] == pytest.approx(math.exp(sum(logs) / len(logs)))
