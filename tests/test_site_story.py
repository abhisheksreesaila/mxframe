"""The story page (docs/site/story.html): every image ships with the site and is described, the timeline is complete,
and the claims keep their caveats."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"
PAGE = (SITE / "story.html").read_text()


def test_every_image_is_in_the_site_and_has_alt_text():
    imgs = re.findall(r"<img\b[^>]*>", PAGE)
    assert len(imgs) >= 1
    for tag in imgs:
        src = re.search(r'src="([^"]+)"', tag).group(1)
        assert (SITE / src).is_file(), src
        alt = re.search(r'alt="([^"]*)"', tag)
        assert alt and len(alt.group(1)) > 20, tag


def test_the_page_has_its_sections_in_order_calm_to_detailed():
    order = ['class="hero"', 'id="journey"', 'id="mojo"', 'id="agents"', 'id="unique"', 'id="read"']
    at = [PAGE.index(k) for k in order]
    assert at == sorted(at)
    for when in ["April 2026", "0.5.0", "0.6.0", "October 20"]:
        assert when in PAGE


def test_claims_keep_their_caveats():
    assert "informal timings" in PAGE and "quiet-machine scoreboard" in PAGE
    soon = PAGE[PAGE.index("Coming soon"):]
    assert "MCP server" in soon and "MCP server" not in PAGE[:PAGE.index("Coming soon")].split('id="agents"')[1], \
        "the engine's MCP server is not built yet: it may only appear under Coming soon"
    assert "validated on NVIDIA" in PAGE
    assert "prefers-reduced-motion" in PAGE


def test_no_gradients_glass_or_blur_on_chrome():
    css = PAGE[PAGE.index("<style>"):PAGE.index("</style>")]
    for banned in ["gradient(", "backdrop-filter", "blur("]:
        assert banned not in css, banned


DECK = (SITE / "deck.html").read_text()


def test_deck_is_linked_and_pinned():
    assert 'href="deck.html"' in PAGE and 'href="deck.html"' in (SITE / "index.html").read_text()
    assert "deck.html" in (ROOT / "README.md").read_text()
    cdn = re.findall(r'https://cdn\.jsdelivr\.net/npm/reveal\.js@([\d.]+)/', DECK)
    assert cdn and set(cdn) == {"5.2.1"}, "reveal.js is pinned to one version"


def test_deck_keeps_the_story_honest():
    assert DECK.count("<section") >= 12
    assert "Q21" in DECK and "written down" in DECK, "the numbers slide says where it's behind"
    soon = DECK[DECK.index("Coming soon"):]
    assert "MCP server" in soon and "MCP server" not in DECK[:DECK.index("Coming soon")]
    assert "prefers-reduced-motion" in DECK and "mx-theme" in DECK
    for src in re.findall(r'<img[^>]+src="([^"]+)"', DECK):
        assert (SITE / src).is_file(), src
