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


def test_one_tile_per_glance_row_four_headlines_and_a_visible_caveat():
    glance = MD[MD.index("## At a glance"):].split("\n## ", 1)[0]
    rows = [l for l in glance.splitlines() if l.startswith("| ") and not l.startswith("| Area")]
    assert len(DATA["areas"]) == len(rows), "one tile per row of the doc's At a glance table"
    assert len(DATA["headline"]) == 4
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
    shown = re.sub(r"<desc.*?</desc>", "", s, flags=re.S)
    if re.search(r">[^<]*\d\s?(?:ms|s)\b", shown):
        assert "informal" in s.lower(), "every engine SVG that shows a timing keeps the informal caveat"


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


BRIEF = (ROOT / "docs" / "briefs" / "positioning.md").read_text()
PLAN = (ROOT / "docs" / "plan.md").read_text()


def test_roadmap_says_what_the_approved_brief_says():
    rm = DATA["roadmap"]
    one_line = " ".join(BRIEF.split())
    assert f"{rm['vision']}. {rm['vision_2']}." in one_line
    assert len(rm["can_be"]) == 7
    assert sum(1 for s in rm["path"] if s["id"] == rm["now"]) == 1


def test_roadmap_steps_name_real_plan_items():
    for s in DATA["roadmap"]["path"]:
        for ref in re.findall(r"\b(?:F-\d+|M\d+)\b", s["tag"]):
            assert re.search(rf"^## {ref}\b", PLAN, re.M), f"{ref} is not a heading in docs/plan.md"


def test_agent_features_are_not_shown_as_done():
    """Only what ships today may say so; the plugins are planned (M10)."""
    items = DATA["roadmap"]["agents"]["items"]
    assert {i["status"] for i in items} <= {"today", "with 0.7.0", "planned"}
    assert [i["name"] for i in items if i["status"] == "today"] == ["mx"]
    assert "PLANNED" in DATA["roadmap"]["agents"]["when"]


def test_readme_names_omxcore_and_shows_the_roadmap():
    readme = (ROOT / "README.md").read_text()
    assert "omxcore" in readme and "docs/briefs/positioning.md" in readme
    for n in ("engine-roadmap.svg", "engine-hood.svg", "engine-family.svg"):
        assert f"docs/img/{n}" in readme, n
