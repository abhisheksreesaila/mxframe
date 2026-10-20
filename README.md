# omxcore

**The data engine for the agent era.**

omxcore runs SQL on your CPU *and* your GPU, over the CSV and Parquet files you already have. It's one native library,
written in Mojo, with no Python inside. Apps link it like SQLite. Agents talk to it in JSON.

### ✨ [Read the story →](https://abhisheksreesaila.github.io/omxcore/story.html) · ▶ [Watch it as slides](https://abhisheksreesaila.github.io/omxcore/deck.html)
The journey, why Mojo, why it's built for agents, the numbers, and where it's going.

[![Story](https://img.shields.io/badge/the%20story-read-0e7c86)](https://abhisheksreesaila.github.io/omxcore/story.html) [![Mojo](https://img.shields.io/badge/written%20in-Mojo%201.1-0e7c86)](https://github.com/abhisheksreesaila/omxcore/tree/main/mojo) [![PyPI](https://img.shields.io/pypi/v/mxframe?label=PyPI%20(mxframe)&color=5d6571)](https://pypi.org/project/mxframe/) [![License](https://img.shields.io/badge/license-Apache--2.0-5d6571)](LICENSE)

- **Your files stay yours.** No import, no new format. Point it at a CSV or a Parquet file and ask.
- **Your hardware, all of it.** CPU or GPU, chosen per query, with the reason.
- **Answers an agent can trust.** Exact decimals, errors with a line, a column and a hint, a note for every decision.

## Try it

```sh
mx sql "SELECT region, sum(revenue) AS revenue FROM 'orders.csv' GROUP BY region"    # JSON out, no Python
```

```python
import omxcore as mx   # 0.6.0 from PyPI: pip install "mxframe[runtime,sql]", then import mxframe as mx

out = mx.sql("SELECT dept, SUM(x) AS s FROM t GROUP BY dept ORDER BY dept", t=mx.scan_parquet("t.parquet").compute())
print(out.compute())
```

On 0.6.0, all 22 TPC-H queries at scale factor 10 ran in **3.15 s** (DuckDB 5.08 s, Polars 14.40 s), every answer equal
to DuckDB's. Where it's still behind is [on the engine page](https://abhisheksreesaila.github.io/omxcore/engine.html).

## Read on

| | |
|---|---|
| ✨ [The story](https://abhisheksreesaila.github.io/omxcore/story.html) | why omxcore, why Mojo, why it's built for agents |
| ⚡ [The engine](https://abhisheksreesaila.github.io/omxcore/engine.html) | what it does and how it got fast, every number ([long form](docs/mojo-engine.md)) |
| 📊 [Benchmarks](https://abhisheksreesaila.github.io/omxcore/benchmarks.html) | every TPC-H query against DuckDB and Polars |
| 🧭 [CPU or GPU](https://abhisheksreesaila.github.io/omxcore/auto-mode.html) | how AUTO decides where a query runs |
| 📘 [Reference](docs/reference.md) | install, the Python API tables, the latest release, the Mojo example |
| 🔥 [Mojo API](docs/omxcore-mojo-api.md) | tables, frames, SQL, jobs and blocks, with no Python |
| 🧱 [Internals](https://abhisheksreesaila.github.io/omxcore/internals.html) | how a query runs, layer by layer |
| 🗺️ [Where it's going](docs/briefs/positioning.md) | the positioning and the path |

Build from source with [pixi](https://pixi.sh): `git clone https://github.com/abhisheksreesaila/omxcore && cd omxcore && pixi install && pixi run test`.
[Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md) · License: [Apache-2.0](LICENSE).
