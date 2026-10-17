# MX frame: the Mojo API omxapp needs

**For:** the MX frame project. **From:** omxapp (the framework) and omxplorer (the Numbers-style data app), as of goal 15 (2026-10-17).
**Replaces:** the goal 8–9 wish list that was here (items 1–22, kept in git history). Everything in it is folded in below.

## Why
Today omxplorer runs on a **stand-in engine** inside this repo: `mojoui/frame.mojo` (columns, edits, formulas, sort),
`mojoui/frame_gpu.mojo` (GPU group-by) and two pyarrow shims (`mojoui/csvio.mojo` reads CSV, `mojoui/omxfile.mojo`
reads and writes Parquet). It is checked against an independent pass over the CSV and works to 100M rows
(`docs/scale.md`). But:
- the shims need Python, so Mac, iPhone and Android builds can't open a file;
- it's capped at 16 columns, int32 on the GPU, no nulls, one table;
- it's the app's own engine, not the product's.

The switch to MX frame should be **one layer**: the components (DataGrid, PivotTable, ChartView, KpiView, the
sheet and report pages) don't change. Only `Frame`, `GpuFrame`, `group_sum`, `read_csv` and `save_omx/open_omx`
change, and they're shaped like the API below on purpose.

## Hard requirements
- **A Mojo package (`.mojopkg`) with no Python or libpython**, so it links into the GTK, AppKit, UIKit and Android hosts.
- **Exact integer arithmetic:** money and quantities are scaled int64; sums are exact 64-bit (the checks compare to the cent).
- **Raises with readable messages** ("column `cost` not found", "row 41,207: `12.x` isn't a number").
- **GPU optional:** every operation has a CPU path giving identical results; the GPU is used when there is one.

## 1. Tables and columns

```mojo
from mxframe import Table, ColumnKind, INT, DECIMAL, TEXT, DATE, BOOL

var t = Table.empty()                              # "New blank sheet"
t.add_column("revenue", DECIMAL, scale=2)          # values stored as int64 hundredths
t.add_column("region", TEXT)                       # dictionary-encoded: int32 codes + a string table
t.rows(); t.columns(); t.name(c); t.kind(c); t.scale(c); t.index("region") -> Int  # -1 if absent
t.get(c, row) -> Int64                             # the stored value (a code for TEXT, days for DATE)
t.is_null(c, row) -> Bool
t.dictionary(c) -> Span[String]; t.text_of(c, code) -> String; t.code_of(c, "West") -> Int  # -1
```

| Need | Today (stand-in) | Why |
|---|---|---|
| Kinds INT, DECIMAL(scale), TEXT (dictionary), DATE (days since 1970), BOOL | INT, DECIMAL, TEXT, DATE; bools import as 0/1 | the Import step and the format inspector branch on kind |
| **Any number of columns** | 16 (`MAX_COLS`, a SIMD row vector) | wider CSVs lose columns past 16 today |
| **Nulls per column** (a validity bitmap) | none: empty cells read as 0 or "" and are only counted | Delete on a number cell; honest averages and counts |
| Column-major, contiguous int64 buffers the app can borrow as `Span[Int64]` | `List[Int64]` per column | the grid reads a screenful at a time; the GPU upload reads whole columns |
| Shared, read-only base data across sessions (web: many tabs, one table in memory) | `share()` returns an address; `release_columns()` empties it | the web host serves one process to many sessions |

## 2. Reading and writing files (the biggest gap: Python today)

```mojo
var r = read_csv(path, options=CsvOptions(max_columns=..., infer_rows=100_000))
r.table; r.rows; r.seconds
for c in r.columns: c.name, c.kind, c.scale, c.empty, c.distinct, c.sample   # the Import step's preview
write_parquet(path, table, metadata: String)    # metadata: one JSON string under a key ("omx")
var p = read_parquet(path); p.table; p.metadata -> String
```

| Need | Detail |
|---|---|
| **CSV/TSV reader** with type inference | integers → INT; decimals → DECIMAL with scale 2 if every value has ≤ 2 places, else 4; ISO dates and timestamps → DATE; true/false → BOOL; else TEXT. Quoting, embedded newlines, BOM, `\r\n`. |
| Per-column import report | empty cells, distinct count (text), the first few raw values, rows that failed to parse (row number + text) |
| Speed | today 1M rows: 0.5 s; 100M rows: 106 s on a loaded machine (pyarrow, multi-threaded). Parity or better, streaming (no full-file copy in memory). |
| **Parquet read/write with key-value metadata** | the `.omx` document is one Parquet file; its JSON metadata holds formats, formulas, charts, pivots and the Report. Edits are baked into the columns on save (D-138). |
| Later | XLSX, JSON lines, Arrow IPC; the Arrow C Data Interface both ways (zero-copy hand-over) |

## 3. Edits, rows and undo (a session's overlay on shared data)

```mojo
var f = Frame(table)              # one session's view: its edits over the shared table
f.set(c, row, value); f.set_text(c, row, "West")     # set_text parses by kind; new words get new codes
f.append(values: List[String]) -> Int                # a new row, returns its number
f.remove(row)                                        # a tombstone; f.alive(row)
f.undo() -> Bool; f.edits -> Int                     # edit counter: hosts re-aggregate when it moves
f.changed_rows() -> List[Int]                        # rows that differ from the shared base
f.merge(c, codes, into) -> Int                       # fold text values into one (a cleaning step), undoable
f.bake() -> Table                                    # edits applied, for saving
```

| Need | Why |
|---|---|
| Sparse cell edits, appended rows and removed rows over a read-only base | many sessions share one 100M-row table |
| One undo log covering sets, appends, removes and merges | the toolbar's Undo |
| **Edits visible to GPU aggregations without re-uploading** | today: the GPU sums the base, then the CPU applies each changed row as an exact delta (`group_sum` in `frame_gpu.mojo`). MX frame can do this its own way (patch kernel, dirty columns), as long as results include edits. |

## 4. Formula columns

```mojo
var m = f.formula("margin = revenue - cost")   # parses, type-checks, returns the new column's index
f.formula_text(m); f.depends(m, c); f.dependants(c) -> List[String]
f.get(m, row)                                  # evaluated
f.program(m) -> Expr                           # the expression with other formulas inlined, for aggregation
```

- Column expressions, never cell addresses: `+ - * / // %`, `min`, `max`, unary minus, constants, parentheses.
- **Decimal-aware:** `revenue / 2` keeps revenue's scale; `revenue * qty` rescales; division rounds the stand-in's way (pre-scaled numerator).
- Formulas may use formulas; aggregations over them must not materialise them (the stand-in inlines with `flatten`).
- Later: `year()`, `month()`, `day()` on dates; `if`; text functions; column plugins (embeddings, classifiers).

## 5. Aggregation (the engine's hot path)

```mojo
var q = Query(f).where(col("region") == code)             # filters: ranges and equality on any column, ANDed
              .group_by(col("carrier"), groups=n)            # a key expression with a known dense group count
              .agg(sum(col("revenue")), count())
var r = q.compute(device=AUTO)    # r.sums: List[Int] (index = key), r.counts, r.milliseconds
```

What omxplorer runs today, all through one call, `group_sum(frame, gf, key, value, ranges, groups)`:

| Caller | Key | Value | Filter |
|---|---|---|---|
| Chart on the sheet (`sheetpage.compute`) | a text column's code | a value column or formula | none |
| Pivot | `row_code × n_cols + col_code` (`pivot_program`) | value column | none |
| Report Sum / Average / Count tiles | constant 0 | value column | the Report filter (`region == West`) |
| Report chart tiles | a text column | value column | the Report filter |
| Column header histograms, status bar sum | a binned expression / constant | the column | none |

| Need | Detail |
|---|---|
| Group by an **integer key expression** with a known small group count (dense output, missing groups = 0) | covers text columns, pivots (combined keys) and bins |
| Aggregates | sum (exact int64), count; next: min, max, mean, count distinct, percentiles |
| Filters | `lo ≤ col ≤ hi` and `==` on any column (up to 8 today), ANDed; next: `is_in`, OR, nulls |
| **Includes the session's edits and formula columns** | §3 and §4 |
| Speed | today, 100M rows on the GPU: 19 ms for a sum, under 1 ms at 1M |
| Next | several group-bys in one pass (`compute_many`); reuse a compiled query with new constants (slider drags) |

## 6. Sort

```mojo
f.permutation(c, descending=False) -> List[Int]   # row order; the data never moves
```
Stable, over any column including formulas and edited cells. Text should sort alphabetically; today it sorts by
dictionary code (the order values first appeared), a known gap. Today: a radix sort, 0.02 s at 1M rows, 3.7 s at
100M. Next: multi-column sort.

## 7. GPU memory
- A device copy of a table, **uploaded once per process**, shared by every session; `release()` to give it back when a document closes.
- Values over 32 bits must not break it (today the stand-in uploads int32 and refuses to upload otherwise; a single amount over $21.4M, stored in cents, doesn't fit).
- Share the app's `DeviceContext` so the app's own kernels (charts, point clouds) and MX frame's use the same device memory.
- Later: data bigger than GPU or host memory (chunked, streamed); Apple GPUs (Metal) without Python.

## 8. Later (Query and Explore modes)
More than one table and joins; a SQL front end (SQL cells, EXPLAIN); pipelines with cached steps; versioned column
chunks for a time machine and branches; DuckDB as the test oracle and TPC-H.

## Memory rules
- A shared `Table` is read-only once shared and outlives every `Frame` built on it.
- Borrowed spans stay valid until the next edit to that column.
- Results own their buffers. Device buffers belong to the `DeviceContext` they were made on.

## Done means
omxplorer imports a CSV, edits, adds formulas, sorts, builds charts, pivots and a filtered Report, and saves and
reopens a `.omx`, with `from mxframe import …` in place of `mojoui/frame*.mojo`, `csvio.mojo` and `omxfile.mojo`;
`scripts/check` passes with the same numbers; and `scripts/build.sh omxplorer mac` links with no libpython.
