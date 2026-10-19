# Docs site design (F-047..F-052)

Mockup: `design/docs-site.html` (options A, B and C; open it in a browser). Chosen 2026-10-14: **light by default**, with:
- **Layout:** A's left sidebar of pages, collapsing to a top menu under 820px.
- **Benchmarks:** B's dot plot (one row per query, one dot per engine, log scale) as the main view, plus C's speedup-vs-DuckDB chart (diverging around 1x) as a second view.
- **Internals:** A's layer stack, with a detail panel beside it.

## Tokens
- House style, Terminal Ledger: `fx-theme.css` tokens (`--bg`, `--surface*`, `--border*`, `--accent*`, `--text*`, `--green/--red/--yellow` with their `-dim` fills).
- `<html data-theme="light">` is the default. A toggle switches to dark, which keeps the scanlines and film grain. The choice is kept in localStorage.
- Fonts: Space Mono (labels, buttons, numbers, code) and Inter (body).
- Engine colours, validated with the dataviz validator on both surfaces (`--pairs all` passes):

  | Engine | Light | Dark |
  |---|---|---|
  | MXFrame `--mx` | `#2a78d6` | `#3987e5` |
  | DuckDB `--dk` | `#1baf7a` | `#199e70` |
  | Polars `--pl` | `#eb6834` | `#d95926` |
  | pyarrow `--pa` (engine page) | `#eda100` | `#c98500` |

  - MX, DuckDB and pyarrow together pass `--pairs all` on both surfaces (2026-10-19); violet failed against MX blue, so pyarrow is slot 4 yellow.
  - Light mode warns that DuckDB's green (and pyarrow's yellow) is below 3:1 contrast, so every chart has a legend and a data-table view.
  - Speedup chart: `--fast` = MXFrame's blue, `--slow` = `#e34948` light / `#e66767` dark, `--mid` = `#8f8e88` for the 1x line.

## Components
- **Sidebar nav (`.side`):** a mono 12px link list; the active page is `--accent-dim` fill with `--accent-hi` text.
- **"How to read this" box (`.read`):** a 2px accent left border on an `--accent-dim` fill, with a mono uppercase title. It's the first element on every page, and the brief requires it.
- **Cards (`.card`, `.card-h`), badges (`.badge.ok`), buttons (`.btn`, `.btn.on`):** from the house style.
- **Charts:** SVG built by plain JS, styled from the tokens.
  - A tooltip on hover (`.tip`), a legend, and a `<details>` data table under each chart.
  - Redraw on theme change.
- **Layer stack (`.layer`):** a numbered button per layer. The selected one gets the accent border, and the detail panel shows its description, files and off switch.

## States
- **Loading:** charts show "loading benchmark files…" until the CSV fetch resolves.
- **Error:** a fetch failure shows the file path that failed and a link to it in `docs/bench/`.
- **Long content:** query rows scroll inside the chart card. Detail panels wrap file paths.
- **Mobile (<820px):**
  - the sidebar becomes a top menu;
  - grids collapse to one column;
  - the dot plot keeps its log axis, with the chart scrolling sideways at a fixed minimum width of 560px.

## The Mojo engine page and README art (2026-10-19)
- `docs/site/engine.html` and `docs/img/engine-*.svg` are drawn from one file, `docs/bench/engine-informal.json`, which quotes `docs/mojo-engine.md` (`tests/test_site_engine.py` checks every number). Regenerate the SVGs with `pixi run python scripts/make_engine_art.py`.
- Look: **editorial paper** (picked 2026-10-19 over a dark Terminal Ledger glow; record in `docs/design/engine-compare.html`, the dark option is in commit `a0151ff`). Light ruled paper, one huge serif number (`--font-display`), the other headline numbers in a ruled ledger column, plum accent.
- The page hero uses the site tokens, so it follows the light / dark toggle (dark uses `--accent-hi` for the serif units). The SVGs use system font stacks only (GitHub strips web fonts and scripts).
