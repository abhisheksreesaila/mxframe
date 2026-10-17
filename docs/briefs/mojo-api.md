# Brief: MX frame as a Mojo engine with a thin Python layer

Status: approved (Abhishek, 2026-10-17: "m1-m7 first, m8 can wait"; "use small C libraries, it's ok"; planner decision left to me)
Spec: `docs/mxframe-mojo-api.md` (from omxapp / omxplorer)

## Why
MX frame is a Mojo engine driven by about 28k lines of Python. That ties every user to Python: omxplorer's Mac, iPhone
and Android builds can't open a file, and the API is whatever Python glue grew. Agents now do most of the running, so
the engine should be a clean Mojo core that a command line, an MCP server, the app and Python all call, with Python
kept for what only its ecosystem does well: `pip install` and importing data through pyarrow.

## What
- A Mojo package (`.mojopkg`) with no libpython: tables (any width, nulls, dictionary text, scaled decimals, dates,
  bools), CSV and Parquet read/write (Parquet key-value metadata), queries (filters, group-by, exact int64 sums,
  counts, then min / max / mean / count distinct), a per-session edit overlay with undo, formula columns, sort
  permutations, and a GPU copy shared per process. Every operation has a CPU path with identical results.
- Readable errors ("column `cost` not found", "row 41,207: `12.x` isn't a number").
- Arrow-compatible memory, so pyarrow hands tables over without a copy (Arrow C Data Interface).
- One plan representation in Mojo (scan, filter, project, aggregate, sort, limit; joins later) under the `Query` API,
  lowered to the existing native executor. The smart planner (optimizer, joins, SQL) moves onto it later without a
  rewrite.

## Order
M0 spike (package + no-libpython binary + Parquet codec choice) → M1 tables → M2 queries on the plan IR → M3 CSV →
M4 Parquet → M5 edits, formulas, sort → M6 GPU → M7 omxplorer switched over ("done means" in the spec).
Later (M8, not now): the `mx` command line for agents, then the planner and SQL in Mojo, Python thinned to a binding.

## Decisions
- Compression codecs (zstd, snappy, maybe gzip) are small C libraries, replaceable later: linked dynamically from the pixi env now, statically for iOS (M7).
- The Python API keeps working throughout; nothing on today's path changes until M8.
- Benchmarks are paused; checks are correctness against DuckDB and omxplorer's own numbers (`scripts/check`).
- Speed targets from the spec are tracked, not gated: CSV parity with pyarrow, 100M-row GPU sum ~19 ms.

## Out of scope for now
XLSX / JSON lines / Arrow IPC files, Metal, joins and SQL in Mojo, data bigger than memory in the Mojo API.
