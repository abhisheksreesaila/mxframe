# Docs overhaul: a simple README, plus interactive pages

Status: approved

## Why (the mission)
- Why this exists, in one sentence: the README is 313 lines, about half of it benchmark history, so a data engineer can't tell in 30 seconds why to try MXFrame or how to use it.
- Why now: 0.6.0 is on PyPI with the first numbers that beat DuckDB and Polars on TPC-H SF10, so people will start arriving.
- If it disappeared: visitors bounce, and the porting and optimization story stays buried in long markdown.

## Who it's for
- First reader: a data engineer who uses Polars or DuckDB and lands on the repo or the PyPI page.
- Their trigger: "is this faster than what I use, and can I switch without rewriting?"
- What's wrong today: the answer is under three scoreboards and a changelog.
- Second reader (assumed): a Mojo/MAX developer, or someone porting the engine, who follows the "behind the scenes" link.

## What it does
- Core job: get a reader from "what is this?" to a running query in two minutes, with the details one click away.
- v1:
  1. **README, under about 120 lines (assumed).** Sections:
     - Why: 3 bullets, faster, same DataFrame and SQL API, GPU when you have one.
     - Latest release: 0.6.0, the SF10 table, with the DuckDB caveat in one line.
     - Install and a code + output "screenshot".
     - API reference as tables: LazyFrame methods, expressions, aggregations, SQL, devices.
     - Links to the pages.
  2. **GitHub Pages site from `docs/`.** One page per topic, with an index that has a thumbnail for each:
     - **Benchmarks (interactive, assumed):** toggle SF1/SF10, stored/shuffled, and engine. Per-query bars, the random-query shapes, and the run conditions.
     - **How `device="auto"` decides:** the existing `auto-mode.html`, restyled.
     - **Behind the scenes, for porting (assumed):** the path a query takes, LazyFrame → plan → optimizer → native Mojo executor, then the fused pipeline, then AOT kernels. Each layer is clickable to show its files, its ctypes/Arrow C-Data seam, and how to add an operator.
     - **How it got fast:** the optimization timeline from `optimizations.md`, one card per step, showing which wins are Mojo and which aren't.
  3. Every page says, at the top, what you're looking at and how to read it ("what each screen means").
- Not doing:
  - no docs framework (MkDocs/Sphinx): plain HTML + JS;
  - no auto-generated API docs from docstrings;
  - no rewrite of `plan.md`, `lessons.md` or the specs.
  - Long markdown (`benchmarks.md`, `optimizations.md`) stays as the source of record.

## How it should look and feel
- Three words: fast, honest, plain.
- Like: DuckDB's docs (short, tables), Polars' benchmark page (charts first).
- House style, Terminal Ledger (assumed), shared by all pages through one CSS file.
- The most important screen: the README's top half. Why, the numbers, install, the first query.

## How we'll know it worked
- A newcomer can install and run a query from the README alone, in under two minutes.
- Every API call in the README tables runs as written (checked by a test).
- Every number on the pages comes from `docs/bench/*.csv`, not typed by hand (assumed: the pages load or embed the CSVs).

## Constraints and risks
- GitHub Pages needs a one-time switch in the repo settings, or `gh api` with your OK.
- Risk: the numbers drift between the README, the pages and the CSVs. To prevent it, build the pages from the CSVs and keep only the headline table in the README.
- Risk: the API tables go stale. A test runs each documented call.

## Decisions and trade-offs
- README for data engineers first. Pro: the people who'd install it. Con: the Mojo depth lives on the pages.
- GitHub Pages. Pro: rendered, linkable, free. Con: one settings switch, and a deploy workflow.
- "Screenshots" means a code + output snippet. Pro: shows the API working at a glance. Con: none of the HTML pages are pictured in the README itself (the site index has thumbnails).
- Plain HTML + JS, no framework (assumed). Pro: nothing to install, matches the stack. Con: shared nav and CSS are copied by hand or by a tiny build script.

## Open questions
- (none; assumed items above are my recommendations, change any)
