"""The Mojo engine page and README art (docs/site/engine.html, docs/img/engine-*.svg): one data file,
docs/bench/engine-informal.json, quotes docs/mojo-engine.md; every chart is drawn from it; the committed
SVGs are what scripts/make_engine_art.py draws; GitHub can render them (no scripts, no web fonts)."""
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = json.loads((ROOT / "docs" / "bench" / "engine-informal.json").read_text())
MD = (ROOT / "docs" / "mojo-engine.md").read_text()
PAGE = ROOT / "docs" / "site" / "engine.html"
IMG = ROOT / "docs" / "img"


def _art():
    spec = importlib.util.spec_from_file_location("make_engine_art", ROOT / "scripts" / "make_engine_art.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nums(text):
    return {float(n.replace(",", "")) for n in re.findall(r"(?<![\w.])\d[\d,]*(?:\.\d+)?", text)}


MD_NUMS = _nums(MD)


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield from _leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, path + (i,))
    else:
        yield path, node


def test_every_number_is_quoted_from_mojo_engine_md():
    for path, v in _leaves(DATA):
        if isinstance(v, bool):
            continue
        if isinstance(v, (int, float)):
            # a millisecond figure may be written in seconds there (DuckDB 1.9 s = 1900 ms)
            assert v in MD_NUMS or v / 1000 in MD_NUMS, f"{path} = {v} is not in docs/mojo-engine.md"
        elif isinstance(v, str) and path[0] in ("headline", "areas", "correct"):
            for n in _nums(v) - {20.0}:  # "20M rows" etc. are row counts, also in the text
                assert n in MD_NUMS, f"{path}: {n} in {v!r} is not in docs/mojo-engine.md"


def test_ten_areas_four_headlines_and_a_visible_caveat():
    assert len(DATA["areas"]) == 10 and len(DATA["headline"]) == 4
    assert "informal" in DATA["caveat"].lower() and "quiet-machine scoreboard" in DATA["caveat"]
    assert sum(1 for h in DATA["headline"] if h.get("focus")) == 1


@pytest.mark.parametrize("h", DATA["headline"], ids=lambda h: h["label"])
def test_headline_text_matches_the_row_it_points_at(h):
    art = _art()
    row = art.lookup(DATA, h["ref"])
    shown = _nums(h["big"] + " " + h["vs"])
    if h["ref"] == ["memory"]:
        want = {row["file_gb"], row["peak_mb"]}
    else:
        want = {art.lo(row["mx"]), art.hi(row["duckdb"])}
    assert want <= shown, (h, want, shown)


def test_gaps_are_really_behind():
    art = _art()
    for g in art.gaps(DATA):
        assert g["lo"] > 1, g


def test_committed_svgs_match_the_generator():
    for name, svg in _art().build().items():
        assert (IMG / name).is_file(), f"run scripts/make_engine_art.py: docs/img/{name} missing"
        assert (IMG / name).read_text() == svg, f"docs/img/{name} is stale: run scripts/make_engine_art.py"


@pytest.mark.parametrize("name", sorted(p.name for p in IMG.glob("engine-*.svg")))
def test_svgs_render_on_github(name):
    s = (IMG / name).read_text()
    assert "<script" not in s and "@import" not in s and "@font-face" not in s and "http://" not in s.replace("http://www.w3.org/2000/svg", "")
    assert "<title" in s and "<desc" in s, "an accessible name and description"
    assert "informal" in s.lower(), "every engine SVG keeps the informal caveat"


def test_page_is_in_the_nav_and_reads_the_one_data_file():
    js = (ROOT / "docs" / "site" / "site.js").read_text()
    assert '["engine.html", "The Mojo engine"]' in js
    html = PAGE.read_text()
    assert '"bench/engine-informal.json"' in html
    assert "prefers-reduced-motion" in html


def test_readme_shows_the_engine_art_and_links_the_page():
    readme = (ROOT / "README.md").read_text()
    raw = "https://raw.githubusercontent.com/abhisheksreesaila/mxframe/main/docs/img/"
    used = re.findall(re.escape(raw) + r"([\w.-]+\.svg)", readme)
    assert "engine-hero.svg" in used and len(used) >= 2
    for n in used:
        assert (IMG / n).is_file(), n
    assert "engine.html" in readme and "docs/mojo-engine.md" in readme
