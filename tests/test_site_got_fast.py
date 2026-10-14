"""F-051: the "How it got fast" page. Every bench file it names exists, every milestone total matches
the value computed from its CSV (the way scripts/plot_progress.py computes it) or the number written
in docs/optimizations.md, and every number on an optimization card is quoted from that document."""
import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "docs" / "site" / "how-it-got-fast.html"
BENCH = ROOT / "docs" / "bench"
MD = (ROOT / "docs" / "optimizations.md").read_text()
FEATURES = {
    "compile-time specialization",
    "closures inlined into parallelize",
    "SIMD widths",
    "AOT shared libraries over the Arrow C Data Interface",
}


class _Data(HTMLParser):
    def __init__(self):
        super().__init__()
        self._in, self.json = False, ""

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "got-fast-data":
            self._in = True

    def handle_endtag(self, tag):
        if tag == "script":
            self._in = False

    def handle_data(self, data):
        if self._in:
            self.json += data


def _data():
    p = _Data()
    p.feed(PAGE.read_text())
    return json.loads(p.json)


def _total(file, engine, sf):
    rows = [r for r in csv.DictReader(open(BENCH / file)) if r["engine"] == engine and r["sf"] == sf and r["first_ms"]]
    assert len(rows) == 22, f"{file}: {engine} SF{sf} has {len(rows)} queries"
    return sum(float(r["first_ms"]) for r in rows) / 1000


def _section(heading):
    m = re.search(rf"^### {re.escape(heading)}\b.*?(?=^#{{2,3}} |\Z)", MD, re.S | re.M)
    assert m, f"optimizations.md has no section '### {heading}'"
    return m.group(0)


TIMELINE = MD[MD.index("## The milestones"):MD.index("### M0 / M1")]


def test_page_exists_with_read_box_and_data():
    html = PAGE.read_text()
    assert 'class="read"' in html
    assert _data()["milestones"]


def test_every_bench_file_named_exists():
    d = _data()
    names = set(re.findall(r"bench/([\w.-]+\.(?:csv|md|json))", PAGE.read_text()))
    for m in d["milestones"]:
        names |= {m["sf10"]["file"], m["sf1"]["file"]}
    names |= {d["reference"]["file"], d["perquery"]["before"]["file"], d["perquery"]["after"]["file"]}
    for c in d["cards"]:
        for n in c["nums"]:
            names |= {f.split("/")[-1] for f in re.findall(r"[\w./-]+\.(?:csv|md|json)", n["source"]) if "optimizations" not in f}
    assert len(names) > 10
    for n in names:
        assert (BENCH / n).is_file(), f"missing docs/bench/{n}"


def test_milestones_in_timeline_order_and_match_the_timeline_block():
    d = _data()
    order = [TIMELINE.index(m["md"] + " s") for m in d["milestones"]]
    assert order == sorted(order), "milestones must follow the timeline in optimizations.md"


@pytest.mark.parametrize("sf", ["10", "1"])
def test_milestone_totals_match_bench_csv(sf):
    for m in _data()["milestones"]:
        e = m[f"sf{sf}"]
        assert abs(_total(e["file"], "mx-cpu", sf) - e["total"]) <= 0.01, (m["id"], e["file"])


def test_milestone_totals_match_numbers_in_optimizations_md():
    for m in _data()["milestones"]:
        written = float(m["md"])
        assert abs(_total(m["sf10"]["file"], "mx-cpu", "10") - written) <= 0.05, (m["id"], written)


@pytest.mark.parametrize("sf", ["10", "1"])
def test_reference_lines_match_bench_csv(sf):
    ref = _data()["reference"]
    for engine in ("polars", "duckdb"):
        assert abs(_total(ref["file"], engine, sf) - ref[f"sf{sf}"][engine]) <= 0.01


def test_reference_lines_match_optimizations_md_table():
    ref = _data()["reference"]
    assert f"{ref['sf10']['polars']:.2f} s" in MD and f"{ref['sf10']['duckdb']:.2f} s" in MD
    assert f"{ref['sf1']['polars']:.2f} s" in MD and f"{ref['sf1']['duckdb']:.2f} s" in MD


def test_cards_follow_optimizations_md_order_and_have_every_part():
    cards = _data()["cards"]
    pos = [MD.index(f"### {c['section']}") for c in cards]
    assert pos == sorted(pos), "cards must follow optimizations.md order"
    headings = re.findall(r"^### (M0|M2\w|F-0\d+)", MD, re.M)
    covered = {c["section"].split(" ")[0] for c in cards}
    assert set(headings) <= covered | {"M2b", "M2c"}, set(headings) - covered
    for c in cards:
        assert c["slow"] and c["now"], c["id"]
        assert c["badges"], c["id"]
        for b in c["badges"]:
            assert b["kind"] in ("mojo", "plain")
            if b["kind"] == "mojo":
                assert b["name"] in FEATURES, b


def test_every_card_number_is_quoted_from_optimizations_md():
    for c in _data()["cards"]:
        text = _section(c["section"]) + TIMELINE
        for n in c["nums"]:
            assert n["source"].strip(), c["id"]
            for key in ("before", "after"):
                token = n[key].replace("~", "")
                assert re.search(rf"(?<![\d.,]){re.escape(token)}(?!\d|,\d|\.\d)", text), f"{c['id']}: {n['label']} {key}={n[key]} not in optimizations.md"


def test_non_mojo_wins_are_labelled_plain():
    plain = {c["id"] for c in _data()["cards"] if all(b["kind"] == "plain" for b in c["badges"])}
    assert {"m2f", "m2i", "m2k", "m2m"} <= plain, "optimizations.md calls these ordinary engineering"


def test_per_query_files_have_duckdb_and_mxframe_for_all_22():
    d = _data()["perquery"]
    for side in ("before", "after"):
        for sf in ("10", "1"):
            _total(d[side]["file"], "mx-cpu", sf)
            _total(d[side]["file"], "duckdb", sf)


def test_page_redraws_on_themechange_and_uses_engine_tokens():
    html = PAGE.read_text()
    assert "themechange" in html
    for tok in ("--mx", "--dk", "--pl"):
        assert tok in html
    assert "<details" in html and "legend" in html


def test_progress_chart_is_drawn_at_container_width_with_readable_text():
    html = PAGE.read_text()
    assert "host.clientWidth" in html and "narrow" in html
    assert 'width: "100%"' not in html.split("function drawProgress")[1].split("function progressTable")[0]
    assert "#progress svg { min-width: 0" in html
    css = (ROOT / "docs" / "site" / "site.css").read_text()
    assert re.search(r"svg text \{[^}]*font-size: (1\d|[2-9]\d)px", css), "axis text stays at 10px or more"
