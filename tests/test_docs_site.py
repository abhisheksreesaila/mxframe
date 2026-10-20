"""The docs site in docs/site/ (F-047..F-052): every page loads the shared CSS and JS, opens with a
"how to read this" box, and every local link, script, image and data file it names exists."""
from html.parser import HTMLParser
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "docs" / "site"
PAGES = sorted(SITE.glob("*.html"))


class _Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.refs, self.classes, self.theme = [], [], None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == "html":
            self.theme = a.get("data-theme")
        for key in ("href", "src"):
            if a.get(key):
                self.refs.append(a[key])
        self.classes += (a.get("class") or "").split()


def _parse(path):
    p = _Page()
    p.feed(path.read_text())
    return p


def _local(ref):
    return not re.match(r"^(https?:|mailto:|#|data:)", ref)


def test_site_has_index_and_shared_assets():
    assert (SITE / "index.html").is_file()
    assert (SITE / "site.css").is_file() and (SITE / "site.js").is_file()
    css = (SITE / "site.css").read_text()
    for token in ("--bg", "--surface", "--accent", "--mx", "--dk", "--pl", '[data-theme="dark"]'):
        assert token in css, token


def test_bench_link_reaches_benchmark_files():
    assert (SITE / "bench" / "m2s-engine.csv").is_file()


LANDING = {"story.html", "deck.html"}   # the story and its slides: their own frame instead of the docs top bar, the same tokens (below)


@pytest.mark.parametrize("page", [p for p in PAGES if p.name not in LANDING], ids=lambda p: p.name)
def test_page_shell(page):
    p = _parse(page)
    assert p.theme == "light", "light is the default theme"
    assert "site.css" in p.refs and "site.js" in p.refs
    assert "read" in p.classes, 'every page opens with a "how to read this" box'


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_local_links_exist(page):
    for ref in filter(_local, _parse(page).refs):
        target = (page.parent / ref.split("#")[0].split("?")[0]).resolve()
        assert target.exists(), f"{page.name} links to missing {ref}"


def test_nav_pages_exist():
    js = (SITE / "site.js").read_text()
    for href in re.findall(r'\["([\w-]+\.html)"', js):
        assert (SITE / href).is_file(), f"nav lists missing page {href}"


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.name)
def test_data_files_named_in_pages_exist(page):
    text = page.read_text()
    for name in re.findall(r'bench/([\w.-]+\.(?:csv|json))', text):
        assert (SITE / "bench" / name).is_file(), f"{page.name} reads missing bench/{name}"


def test_pages_workflow_copies_bench_files():
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "docs/site" in wf and "-L" in wf, "deploy must dereference the bench link"


def test_csv_files_the_pages_read_have_no_quotes():
    """site.js loadCSV splits on bare commas; a quoted field with a comma would be torn apart."""
    bench = ROOT / "docs" / "bench"
    names = set()
    for f in [*SITE.glob("*.html"), *SITE.glob("*.js"), *(SITE / "js").glob("*.js")]:
        names |= {Path(n).name for n in re.findall(r"bench/([\w.-]+\.csv)", f.read_text())}
    assert len(names) >= 10
    for n in sorted(names):
        assert (bench / n).is_file(), n
        assert '"' not in (bench / n).read_text(), f"{n} has a quote: site.js loadCSV cannot parse quoted fields"


def test_pages_workflow_publishes_the_kernel_visualizer():
    """site.js links visualize/omxcore_pipeline.html; it lives outside docs/site and is copied in at deploy."""
    wf = (ROOT / ".github" / "workflows" / "pages.yml").read_text()
    assert "cp -r visualize _site/visualize" in wf
    assert (ROOT / "visualize" / "omxcore_pipeline.html").is_file()
    assert "visualize/omxcore_pipeline.html" in (SITE / "site.js").read_text()


def test_landing_page_and_docs_share_one_design_language():
    """Story and docs are one product: the same Ledger / Graphite colours, the same two fonts, no gradients."""
    story = (SITE / "story.html").read_text()
    css = (SITE / "site.css").read_text()
    for value in ["#eef0f3", "#ffffff", "#e1e4e8", "#1b1f24", "#0e7c86",            # Ledger: page, surface, line, ink, accent
                  "#161618", "#1d1e21", "#ececee", "#3fb8c2",                       # Graphite (night)
                  "JetBrains Mono", "Inter"]:
        assert value in story and value in css, value
    assert "gradient(" not in css.split("*/", 1)[1] and "Space Mono" not in css
    for page in PAGES:
        assert "Space+Mono" not in page.read_text(), page.name
