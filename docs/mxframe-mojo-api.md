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
t.add_column("region", TEXT)                       # dictionary-encoded: codes + a string table (MX frame stores the codes as Int64, like every value)
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
var shared = r.take_table().share()             # moves the table out; the report / metadata stay readable
```

| Need | Detail |
|---|---|
| **CSV/TSV reader** with type inference | integers → INT; decimals → DECIMAL with scale 2 if every value has ≤ 2 places, else 4; ISO dates and timestamps → DATE; true/false → BOOL; else TEXT. Quoting, embedded newlines, BOM, `\r\n`. |
| Per-column import report | empty cells, distinct count (text), the first few raw values, rows that failed to parse (row number + text) |
| Speed | today 1M rows: 0.5 s; 100M rows: 106 s on a loaded machine (pyarrow, multi-threaded). Parity or better, streaming (no full-file copy in memory). |
| **Parquet read/write with key-value metadata** | the `.omx` document is one Parquet file; its JSON metadata holds formats, formulas, charts, pivots and the Report. Edits are baked into the columns on save (D-138). |
| **Bigger than RAM** (F-140) | `var scan = scan_parquet(path)`; `Query(scan).where(...).group_by(col("region"), groups=scan.word_count(scan.index("region"))).agg(...).compute()` decodes one row group at a time (only the columns the plan reads) and equals `read_parquet` + the same query exactly; TEXT codes are the whole-file read's (`scan.code_of(c, word)`); CPU only. `count_distinct` and `top_k` stream too, exactly, by merging their state per row group; `percentile` / `median` keep every non-null value (12 bytes a row), see §5 |
| Later | XLSX, JSON lines, Arrow IPC; the Arrow C Data Interface both ways (zero-copy hand-over) |

## 3. Edits, rows and undo (a session's overlay on shared data)

```mojo
var f = Frame(table)              # one session's view: its edits over the shared table
f.set(c, row, value); f.set_text(c, row, "West")     # set_text parses by kind; new words get new codes
f.append(values: List[String]) -> Int                # a new row, returns its number
f.remove(row)                                        # a tombstone; f.alive(row)
f.undo() -> Bool; f.edits -> Int                     # edit counter: hosts re-aggregate when it moves
f.edited(c, row) -> Bool                              # the cell holds a session edit (grids mark it)
f.word_count(c) -> Int                                # shared words + words this session added
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
| Aggregates | sum (exact int64), count, count(e), min, max, mean, **count distinct, median / percentile, top-k** (below) |
| Filters | `lo ≤ col ≤ hi` and `==` on any column (up to 8 today), ANDed; next: `is_in`, OR, nulls |
| **Includes the session's edits and formula columns** | §3 and §4 |
| Speed | today, 100M rows on the GPU: 19 ms for a sum, under 1 ms at 1M |
| Next | several group-bys in one pass (`compute_many`); reuse a compiled query with new constants (slider drags) |

### Count distinct, percentiles and top-k (F-148)

Aggregates that need more than a running number per group, exact, over a table, a Frame with edits or a Parquet scan:

```mojo
var r = Query(f).where(col("day").between(0, 364))
              .group_by(col("region"), groups=n)
              .agg(count_distinct(col("customer")),          # r.values(0)[g]   distinct non-null values
                   median(col("revenue")),                   # r.values(1)[g] (scale + 1)   r.present(1, g)
                   percentile(col("revenue"), 0.9),          # r.values(2)[g]   (the value itself, unscaled)
                   top_k(col("revenue"), 3),                 # r.top(3, g)      List[Int], largest first
                   sum(col("revenue")))                      # plain aggregates sit beside them, one pass
              .compute()
```

| Aggregate | Meaning | Result |
|---|---|---|
| `count_distinct(e)` | the number of different non-null values of `e` in the group (text: different words; a merged spelling counts once) | `r.values(i)[g]`, 0 for a group with none |
| `percentile(e, p)` (discrete; `percentile_cont`, interpolating, is a follow-up) | `p` in 0..1: the value at index `max(ceil(p * n) - 1, 0)` of the group's `n` non-null values sorted ascending. This is DuckDB's **`quantile_disc`** (checked against it), so the answer is always an actual value of the column, in the column's stored units (a DECIMAL(2) keeps its scale 2), and exact | `r.values(i)[g]`; `r.present(i, g)` is False for a group with no non-null value (its value is then 0) |
| `median(e)` | DuckDB's **`median`** (`quantile_cont` at 0.5): the middle value, or for an even count the mean of the two middle values, **exact**: the result is the stored value times 10, i.e. at the column's scale plus one decimal place, so [1, 2, 3, 4] gives 25 (2.5), and 5 * (a + b) never rounds. A median whose middle passes about 9.2e17 does not fit one more place in int64 and raises. A **text** column has no mean: its median stays discrete, the lower middle word (a code), unscaled | `r.values(i)[g]` as above |
| `top_k(e, k)` | the `k` largest non-null values of the group, repeats kept (`ORDER BY e DESC LIMIT k`; DuckDB's `max(e, k)`); fewer when the group has fewer | `r.top(i, g)`: `List[Int]`, largest first |

- **Shape.** Distinct counts and percentile values are a number per group like the other aggregates (`values(i)`, `present(i, g)`). Top-k is a list per group (`top(i, g)`) because its size varies; it holds values, not row numbers: ties make "which row" ambiguous, and an app that wants the rows filters on `value >= top(i, g)[k-1]`. Several aggregates over the same expression share one pass over its values.
- **Nulls** are ignored, as in SQL: a null value is not counted, not ranked, not listed. A row whose group is outside `0 .. groups-1`, or null, or that a filter drops, is left out as for every aggregate.
- **Text** columns use their codes for `count_distinct`. For `percentile`, `median` and `top_k` the order is the words' alphabetical order (ASCII case ignored, bytes breaking ties: the order of `permutation` and "apple" before "Zebra"), not the codes'; the values come back as codes (`text_of`). They need the plain column: a text column inside an expression raises. In a Frame the order includes the words the session added, and merged spellings are one word. **This differs from DuckDB's default**, which orders `VARCHAR` by bytes ("Zebra" before "apple"); DuckDB's `NOCASE` collation is not honoured by its `quantile_disc` / `max(e, k)`, so the tests compare against `lower(s)`. For words that differ only in case the order is the bytes', as in `permutation`.
- **Algorithms** (all on every core, `holistic.mojo`). The query kernel's per-block hook writes the (group, value) pairs of the selected rows, packed; then: *count distinct* is a hash set of (group, value) pairs in 4 x cores hash partitions (scatter the pairs, then each partition inserts alone: no locks), tables sized from a HyperLogLog sketch of the pairs and probed with software prefetch; the first window is counted in tables that live one task, only a second window (a scan's next row group, or a table over 20M rows) turns them into persistent tables. *Percentile* groups the values with a parallel counting sort and **selects** instead of sorting: three-way quickselect for small groups, a parallel most-significant-byte radix select for groups of 262,144 values or more (two passes for values in a small range, eight at most). *Top-k* is a bounded min-heap of `k` values per group (one compare rejects most rows). When top-k is the only aggregate reading its values and the groups are few, the kernel's own hook offers each value to a heap private to the worker, so no pairs are written at all (20M rows, 8 groups: 17 ms, a plain sum is 15 ms); otherwise private heaps per core over the pairs when groups are few, and the pairs scattered by group when many.
- **Streamed scans.** Distinct and top-k stay exact by merging state per row group (the hash set; the heaps), so they use memory for their state, not the file. **Percentiles need every value:** the state is the non-null (group, value) pairs of the whole file, 12 bytes a non-null row, and the answer is computed from a grouped copy (8 bytes a row more): a peak of about 27 bytes per non-null row measured (19M rows: 520 MB over the scan's 77 MB; a 100M-row column: about 2.7 GB). Distinct state is about 35 bytes per distinct (group, value) pair measured (19M distinct pairs: 680 MB), however many row groups; top-k is `k` words a group (`groups * k` may not pass 268M), nothing per row.
- **Tables and frames** run in windows of up to 256 MB of pairs (about 20M rows for one value source) so the transient pairs are bounded; the plain aggregates are merged across windows. A frame's changed and appended rows are added after the kernel pass (the shared rows they replace are skipped by the mask), as for the plain aggregates.
- **GPU:** CPU only. These need every value, not a running number; the GPU kernel's per-group accumulators cannot hold them, and moving the pairs to the device and back costs more than the CPU's pass. `compute(AUTO)` runs them on the CPU and says why in `fallback_reason` when it would have used a device table; `compute(GPU)` raises.
- **Also:** `r.top(i, g)` is empty for an aggregate that is not a `top_k`. `k` is clamped to the table's or file's row count (a Frame cannot know, so a `k` with `groups * k` past 268M raises). `r.sums` stays right across windows. `percentile` raises for `p` outside 0..1, `top_k` for `k < 1`; a value that is a comparison raises as for any aggregate.

Informal timings (F-148; 12-core Ryzen 9 5900X, 24 threads, load about 3, a 20M-row table in memory, `compute()` alone, best of 3, results checked against DuckDB elsewhere; the DuckDB column is the same query in DuckDB 1.5.1 on 24 threads over a table it loaded, not a quiet-machine benchmark):

| 20M rows | plain `sum` + `count` | `count_distinct` | `median` | `top_k(.., 10)` | all three + `sum` | DuckDB: distinct / median / top-k |
|---|---|---|---|---|---|---|
| 8 groups, ~19M distinct values | 17 ms | 148 ms | 120 ms | 19 ms | 265 ms | 258 / 322 / 14 ms |
| 1,000 groups, 1,000 values | 18 ms | 93 ms | 94 ms | 21 ms | 190 ms | 132 / 248 / 17 ms |
| 1,000,000 groups, ~19M distinct values | 108 ms | 372 ms | 398 ms | 449 ms | 739 ms | 617 / 2,208 / 2,295 ms |

A scan of the same 20 row-group file (decode included): `sum` 107 ms, `count_distinct` 490 ms (756 MB peak), `median` 187 ms (598 MB), `top_k` 105 ms (77 MB, the scan's own).

## 6. Sort

```mojo
f.permutation(c, descending=False) -> List[Int]   # row order; the data never moves
```
Stable, over any column including formulas and edited cells; nulls last in both directions, removed rows left out.
Text sorts alphabetically ignoring ASCII case, byte order breaking ties ("apple" before "Zebra"; pending the owner's
review). A radix sort over Int64 keys: about 25-30 ms at 1M rows for text and narrow integers, 55-75 ms for full-range
int64 (informal, F-138a). Next: multi-column sort.

## 7. GPU memory
- A device copy of a table, **uploaded once per process**, shared by every session; `release()` to give it back when a document closes.
- Values over 32 bits must not break it (today the stand-in uploads int32 and refuses to upload otherwise; a single amount over $21.4M, stored in cents, doesn't fit).
- Share the app's `DeviceContext` so the app's own kernels (charts, point clouds) and MX frame's use the same device memory.
- `compute(AUTO)` falls back to the CPU only when the device cannot hold the query's buffers or the kernel declines the program; `QueryResult.fallback_reason` says which ("" when it ran where it was sent). `compute(GPU)` never falls back.
- A build without a GPU (every non-Linux target, or `-D MXFRAME_NO_GPU`): `gpu_available()` is False, `DeviceTable(...)` and `compute(GPU)` raise "no GPU on this build", AUTO is the CPU (see `mojo/README.md`, build switches).
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
