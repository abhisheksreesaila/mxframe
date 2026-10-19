# omxcore: positioning

Status: approved (owner, 2026-10-19)

## One line
**DuckDB-class SQL that uses your GPU and links into any app with no Python. Plug and play with the CSV and Parquet files you already have.** omxcore is a compute engine (CPU and
GPU, SQL, joins, streaming) with persistence to open Parquet files. It is the core that apps and agents build on, like
SQLite. (Renamed from MX frame: F-165.)

## What omxcore can be
1. **A faster drop-in for DuckDB** in notebooks, ETL and agents: SQL over Parquet and CSV, Arrow out, GPU when there
   is one.
2. **The GPU-native analytics engine for one machine:** many laptops and workstations have a GPU that DuckDB and
   Polars never use, and AUTO puts each query on the faster device. "Spark-cluster answers on one box."
3. **An embedded engine for apps:** a compiled library, no Python, callable from Mojo or any language through a C
   interface. The first app is omxplorer; others can follow (BI, spreadsheets, trading, IoT, on-device analytics on
   Mac and phones).
   **Pairs with omxapp:** omxapp (the Mojo framework for building apps once and running them natively on Linux,
   macOS, iPhone, Android and the web) + omxcore (the data engine) = a full stack for data apps with no Python and no
   server: the app and its engine compile into one native binary. omxplorer is the first app built this way.
4. **A data engine for AI agents:** `mx` returns JSON with exact errors, starts in 12 ms, and needs no Python
   environment. An MCP server around it makes omxcore the default data tool for agents.
5. **The compute half of a lakehouse:** Parquet today; Delta Lake / Iceberg tables if apps need them (M9, parked).
6. **Live dataframes for simulations and real-time state:** edits, undo and formulas, queryable fast on CPU or GPU.
7. **Feature and ML data prep:** joins and group-bys feeding training data to the GPU, with zero-copy handoff to
   PyTorch and JAX through the Arrow C Data Interface (F-139).

## Agent-native (owner, 2026-10-19)
omxcore is built for agents first: `mx` speaks JSON, starts in 12 ms and needs no Python. On top of the core, a set of
**plugins** makes it the place agents keep, search and reason over data. Each plugin is thin; what it needs from the
engine becomes a generic core primitive any app can use.

| Plugin | What agents get | Core primitive it needs |
|---|---|---|
| **Vector search** (GPU-native) | embeddings stored beside the data; `nearest(k)` with SQL filters in one query | a fixed-size float vector column; a distance + top-k kernel on CPU and GPU; an approximate index (IVF / graph) later |
| **Agent search** | one call that mixes keyword, vector and SQL filters and returns ranked rows with reasons | text tokens per dictionary word (exists), vector top-k, rank fusion |
| **Classification / decision API** | label or route rows (rules, small models, embeddings vs examples), batched on the GPU | user-defined functions run per block; per-word text functions (F-156) |
| **MCP server** (F-169) | `sql`, `explain`, `schema`, `search` as tools any agent can call | `mx` |
| **Memory for agents** | append-only facts with time and source, queried like any table; versions later (M9, Delta-style) | Parquet + append; nothing new |

**Plug and play:** every plugin works on the CSV and Parquet files you already have: point it at a file or folder and query, search or classify. No import, no new format, nothing to convert (`mx sql "… FROM 'tickets.csv'"` works today; vectors are a column in the same Parquet file).

Rule: the core never learns about agents; it gains vectors, a top-k distance kernel and UDFs, and the agent features
live in plugins over them.

## Why it can win
- **GPU-native across vendors:** DuckDB is CPU-only by design; cuDF and Sirius are tied to NVIDIA, CUDA and Python.
  Mojo gives one codebase for NVIDIA, AMD and Apple Silicon.
- **Modern algorithms, not a port of old ones:** morsel-driven parallelism (each core owns a block, no barriers),
  radix-partitioned hash joins built without atomics, parallel radix sort and radix select, HyperLogLog-sized hash
  sets, bounded top-k heaps, SIMD bulk Parquet decode, two-phase parallel dictionaries, planner choices from exact
  statistics, reproducible float sums.
- **A modern compiled language:** Mojo compiles ahead of time to native code with SIMD types, compile-time
  specialisation and no garbage collector, and the same kernels run on CPU and GPU. Python's ease with C++'s speed,
  in one codebase.
- **No Python inside:** an app or agent links one library; nothing to install or break.
- **Open files only:** no proprietary format, no lock-in. Anything omxcore writes, every other tool reads.

## Principles (owner)
- **The engine is the kernel.** Apps (omxplorer, future ones) and Python are plugins; an app's request becomes a
  generic engine feature, never app-specific code in the core.
- **Beat DuckDB and Polars on CPU and GPU,** on general workloads, not benchmark tricks. Honest about every gap.
- **Not a database.** If database-like features are ever needed, they mimic Delta Lake (Parquet + its transaction
  log): no new format, and no database work until an app needs it.

## What "replace DuckDB" requires
| Must have | Status (2026-10-19) |
|---|---|
| Correct answers on real SQL (TPC-H 22, then TPC-DS and the long tail) | M8 in progress; the gate is F-162 |
| Faster on CPU alone (most users have no big GPU) | wins on many joins and aggregates; behind on sorts, big outputs, filtered and one-to-many joins |
| Clearly faster on GPU where one exists | works in memory and streamed; needs scoreboard proof |
| Window functions, more types (timestamps, lists, structs) | not started |
| Joins bigger than RAM (spill) | deferred (F-163) |
| Simple install and Arrow interop | Python package exists; Arrow C interface is F-139 |

## Path
1. Finish M8 and pass the TPC-H 22 gate through `mx` (F-162).
2. Quiet-machine scoreboard: omxcore against DuckDB and Polars, CPU and GPU, published with the gaps.
3. Close the top gaps: join speed (F-161), sort and large outputs, then window functions.
4. Rename (F-165), release 0.7.0, an MCP server for `mx` (F-169), and the agent-native plugins (M10: vector search on the GPU first).
5. Then joins bigger than RAM and the SQL long tail.
