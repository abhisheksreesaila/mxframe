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
| Kinds INT, DECIMAL(scale), TEXT (dictionary), DATE (days since 1970), BOOL; and FLOAT (float64 bits in the value slot), a result-only kind `sql()` returns for AVG and `/`, which no file, Frame or query input takes until F-157 | INT, DECIMAL, TEXT, DATE; bools import as 0/1 | the Import step and the format inspector branch on kind |
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
| **Bigger than RAM** (F-140) | `var scan = scan_parquet(path)`; `Query(scan).where(...).group_by(col("region"), groups=scan.word_count(scan.index("region"))).agg(...).compute()` decodes one row group at a time (only the columns the plan reads) and equals `read_parquet` + the same query exactly; TEXT codes are the whole-file read's (`scan.code_of(c, word)`); CPU only. `count_distinct` and `top_k` stream too, exactly, by merging their state per row group; `percentile` / `median` keep every non-null value (12 bytes a row), see §5 | Edit it with `Frame(scan)` (§3, F-147) |
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
f.bake() -> Table                                    # edits applied, for saving (refused over a scan: use save)
f.save(path, metadata="", compression=ZSTD)          # the edited rows to a new Parquet file (F-147; below)
```

| Need | Why |
|---|---|
| Sparse cell edits, appended rows and removed rows over a read-only base | many sessions share one 100M-row table |
| One undo log covering sets, appends, removes and merges | the toolbar's Undo |
| **Edits visible to GPU aggregations without re-uploading** | today: the GPU sums the base, then the CPU applies each changed row as an exact delta (`group_sum` in `frame_gpu.mojo`). MX frame can do this its own way (patch kernel, dirty columns), as long as results include edits. |

### Editing a file bigger than memory (F-147)

```mojo
var f = Frame(scan_parquet(path))      # the same API as Frame(shared): set, set_null, set_text, append, remove, merge,
                                       # formula, undo, edited, word_count, text_of, code_of, get, is_null, permutation
f.set_cache_limit(64 * 1024 * 1024)    # optional; 128 MiB by default
var r = Query(f).where(...).group_by(...).agg(...).compute()     # streams the file with the edits applied
f.save("edited.parquet", metadata)     # streams the edited rows out; the file may replace the one the frame reads
```

| What | How it behaves |
|---|---|
| **Exactness** | A query, `get`, `is_null`, `edited`, `permutation` and `save` give exactly what `Frame(read_parquet(path))` with the same edits gives (tested: random sets, nulls, typed text with new words, appends, removes, merges and formulas over a 13-row-group file, again after every undo). TEXT codes are `scan.code_of`'s; session words follow the file's dictionary, which is built on first use, so codes never move. |
| **`get` / `is_null`** | Read the row group that holds the row through a cache of decoded **column chunks** (one column of one row group, 8 bytes a value plus its null bits). **The bound is `cache_limit` = 128 MiB by default** (`set_cache_limit(bytes)`, `cache_held()`, `cache_loads()`); least recently used chunks go first (lookup, use and eviction are O(1)), and the chunk just read is always kept (so one chunk bigger than the limit still reads: the bound is `max(limit, one chunk)`). A screenful of rows of one group costs one decode per column shown, not per cell; a grid wider than `limit / chunk` columns decodes on every pass, so size the limit to `columns shown x rows per group x 8`. A formula cell reads only the columns it uses. |
| **Queries** | `Query(frame)` streams the file as `Query(scan)` does (row groups on the workers by blocks, or group by group), each group or block skipping the rows the frame changed or removed (one bit per file row, 12.5 MB per 100M rows), and the changed and appended rows are added once at the end, exactly as for a frame over a table. Merges, formula columns, count_distinct, median, percentile and top_k work too (text order includes the session's words). CPU only. |
| **`permutation`** | Streams the column one row group at a time but holds **O(rows) keys** (about 50 bytes a row: 20M rows peaked at 1.0 GB, 1.75 s). A formula column evaluates row by row through the cache. |
| **`bake()`** | **Refused** over a scan ("a Frame over a streamed file cannot be baked into memory: save it to a new file with `save(path)`"). |
| **`save(path, metadata="", compression=ZSTD)`** | Streams the edited rows into a temporary file and renames it over `path` (atomic; an error removes the temporary file). One output row group per input row group (less its removed rows; a group with none left is skipped), then the appended rows in groups of 1M rows; all the frame's columns of one group are decoded at once, so the memory is one row group's columns, the output group, the file's dictionaries and the edits. A column is written OPTIONAL when the file declared it so or an edit put a null in it. Merged spellings are written as the word they read as; each group's dictionary page holds all the column's words (as `write_parquet` does). Formulas are not stored (their texts go in `metadata`). Over a table, `save` writes `bake()`. |
| **Metadata kept** | `save` writes `metadata` under "omx" (the caller's JSON; the source's is not carried over) and **writes back the file's other key-value metadata** (`scan.extra_metadata()`). Arrow's own `ARROW:schema` key is not kept: it describes the source's types, which the saved file no longer has. |
| **Narrowed columns** | Only a loss of values is refused: reading holds a TIMESTAMP as its UTC day, so **`save` raises, before writing anything and for any path, the source's included**: "column `ts` is TIMESTAMP in the file and is held as a DATE: saving would lose the time of day. ... pass allow_narrowing=True to save anyway" (`scan.narrowing(c)`, "" for none). A change of type that keeps every value is written as held, without asking: DECIMAL(p, s) as DECIMAL(18, s), ENUM / JSON as text, a narrower or unsigned integer as int64 (`scan.type_change(c)` says which). Arrow's own schema key is not written back, so Arrow-only types (dictionary, large_string, a timezone) come back plain. |
| **Threads** | **A Frame is not thread-safe**: `get`, `is_null` and `text_of` on a scan frame update the chunk cache (and build dictionaries), so one thread at a time per Frame; give each thread its own `Frame(scan)`. (`Query(frame)` copies what it needs and reads nothing from the frame while it runs.) |
| **API change** | `Frame.word_count`, `text_of` and `code_of` now `raises` (a file's dictionary is built on first use and can fail to decode); `get`, `is_null`, `permutation` and `overlay` already did. |
| **Measured** (12-core, 20M rows, 20 row groups, 567 MB file, 5 columns, 16 MiB cache) | edits 0.8 s for 1,350 edits (a cold chunk per set); query over the edited frame 0.1 s, 70 MB peak; a grid walk (1 row in 997, `get`) 25 ms, 88 MB; `save` 3.1 s, 180 MB peak (a 1M-row group is 40 MB decoded); the process alone is 21 MB. With 250k-row groups the save peaks at 84 MB on a 332 MB file. |


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
Stable, over any column including formulas and edited cells; nulls last in both directions, removed rows left out. Over a `Frame(scan)` it streams the column but holds O(rows) keys (§3).
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

## 8. SQL over one table (F-150)

```mojo
var cat = Catalog()
cat.add_table("orders", orders)              # a SharedTable
cat.add_frame("edited", frame)               # a Frame, snapshotted now (its edit counter is `SqlSource.version`): add it again after more edits
cat.add_scan("big", scan_parquet(path))      # a Parquet file, read a row group at a time (a Frame over a scan works too)
var r = sql(cat, "SELECT region, sum(revenue) AS total FROM orders WHERE qty > 2 GROUP BY region HAVING count(*) > 5 ORDER BY total DESC LIMIT 10")
r.table                                      # an owned Table: INT, DECIMAL, TEXT, DATE, BOOL and FLOAT (a result-only kind until F-157)
r.notes                                      # what the planner decided and what ran, one line each
r.explain; r.ms; r.device (CPU); r.fallback_reason ("")
var e = sql(cat, "EXPLAIN SELECT ...")       # the plan as a one-column Table `plan` (also `e.explain`); nothing runs
var o = PlannerOptions(); o.group = GROUP_HASHED          # GROUP_AUTO, GROUP_DENSE or GROUP_HASHED; o.dense_limit; o.threads
var h = sql(cat, "SELECT ...", o)
```

**The dialect** is a subset of DuckDB's; anything else raises `SQL error at line L, column C: <what>`, naming the construct
(`` `JOIN` is not supported yet (F-153) ``, `column `revenu` not found in table `orders` (did you mean `revenue`?)`).

| Part | Supported |
|---|---|
| Statement | `[EXPLAIN] SELECT ... FROM table [AS alias]` (table names and columns match without regard to case; `;` and `--` / `/* */` comments allowed) |
| Select list | expressions with aliases; group keys, aggregates, and expressions over them (`sum(a) / count(*)`, `max(d) - min(d)`, `g + 1`); a number or date constant |
| `WHERE` | `= <> != < <= > >=`, `BETWEEN`, `IN (literals)` (numbers: up to 40; text: any number), `IS [NOT] NULL`, `AND OR NOT`, a BOOL column; text, DATE (`DATE '2024-01-31'` or a string next to a DATE column) and decimal literals (aligned to the column's scale); `+ - *`, `//`, `%` on INT and DECIMAL (scales aligned as DuckDB does) and `date + n`, `date - date` |
| `GROUP BY` | columns, expressions, ordinals (`GROUP BY 1`), select aliases; none at all (one group) |
| `HAVING`, `ORDER BY` | aggregates and group keys in any expression; `ORDER BY` keys, aggregates, aliases, ordinals; `ASC` / `DESC`; `NULLS FIRST` / `LAST` (NULLS LAST by default, also for `DESC`, as DuckDB); `LIMIT n [OFFSET m]` |
| Aggregates | `SUM`, `COUNT(*)`, `COUNT(x)`, `COUNT(DISTINCT x)`, `MIN`, `MAX` (also of TEXT), `AVG`, `MEDIAN`, `QUANTILE_DISC(x, p)` |

**SQL's rules** (the Query builder above keeps its own; both run on the same kernels):
three-valued logic in `WHERE` / `HAVING` (a row is kept only when the condition is true; `x IN (1, NULL)` is true or unknown; `NOT` of unknown is unknown);
NULL is a group of its own; only groups with rows exist (a global aggregate over no rows still returns its one row); `SUM` / `MIN` / `MAX` / `AVG` / `MEDIAN`
of no non-null value are NULL, `COUNT` is 0; `x // 0` and `x % 0` are NULL and both truncate toward zero; a word the column never held matches nothing;
`/` always gives a FLOAT, `AVG` gives a FLOAT that is exactly what DuckDB computes (its long-double division `sum / (count * 10^scale)`, reproduced bit for bit).
Text order everywhere (`ORDER BY`, `MIN` / `MAX`, `QUANTILE_DISC`, `<` and `BETWEEN` against a literal) is alphabetical ignoring ASCII case, with byte order breaking ties; `=` and `IN` compare bytes.

**Differences from DuckDB, on purpose**: a `SUM` (or any int64 arithmetic) past int64 raises `a sum or an expression passed the int64 range (about 9.2e18)` where DuckDB
widens to HUGEINT (decimal `*` too); `MEDIAN` of a DECIMAL is exact at one more decimal place (DuckDB rounds to the column's scale; the median of integers is a FLOAT, as DuckDB's DOUBLE);
`//` on a DECIMAL and `/` below an aggregate raise (they are FLOATs until F-157); the text order above.

**How a query runs.** The statement is parsed (`sql.mojo`), bound to a typed tree (`bind.mojo`: types, scales, which expressions run per row and which after aggregation), planned from
statistics only (`planner.mojo`: the `Planner` trait, `RulePlanner`, `Stats`, `PlannerOptions`), lowered to one kernel program in SQL's mode (`sqlprog.mojo`), run over the source's
chunks, and finished in Mojo (`sqleval.mojo`, `sqlexec.mojo`: HAVING, ORDER BY, LIMIT, select expressions, the result Table). A table is one chunk; a frame is its shared table with
the changed rows masked out plus a small table of the changed and appended rows; a scan is one chunk per row group. The chunks' partial results (sums and counts add, min / max combine,
hashed groups merge by key) fold into one.

| Group strategy | When | How |
|---|---|---|
| none | no `GROUP BY` | one group, the plain kernel |
| dense | every key is a column whose values lie in a known range: a TEXT dictionary, BOOL, or the min / max of an INT / DECIMAL / DATE column of a table or frame (one pass, cached per source), and the domain (a NULL slot per nullable key, keys mixed-radix) is at most `dense_limit` (4,194,304) and either at most 65,536 or at most 4x the rows | the kernel adds into slot `g` directly |
| hashed | a key is an expression, or has no known range (an INT column of a scan), or the domain is too wide or too sparse | `hashagg_cpu.hash_aggregate`: per-worker hash tables over the key words (the value, plus a NULL flag word for a nullable key), a partitioned merge |

`count(DISTINCT)`, `MEDIAN`, `QUANTILE_DISC` and `MIN` / `MAX` of TEXT use the holistic states of §5; over hashed groups they take a second pass that maps each row to its group
by key (`holistic: a second pass ...`). `ORDER BY` with `LIMIT + OFFSET <= 4096` over more than 8x as many groups takes a bounded heap; otherwise stable radix passes, last key first.

`r.notes` is the contract the tests assert: `source: ...`, `group: dense, 2 keys g 7 (with a NULL slot) x s 17 ... = 119 slots: <reason>` or `group: hashed on N key words: <reason>`
(`forced by the options` when `PlannerOptions` said so), `kernel: N instructions a row (M slots)`, `ran: table|frame|scan ...: <chunks>`, `holistic: ...`, `order: ...`, `having: ...`, `time: ...`.

**A frame's formula columns** (§4) are columns of the table: the query reads the formula's expression over the stored columns, inlined, with the builder's own rules for it (floor `//` and `%`,
`x // 0` is 0, `min` / `max`) and exact decimals; they work in `WHERE`, `GROUP BY` (hashed) and the aggregates, over a table frame and a frame over a scan.

**Limits**: one table (joins: F-153); no row queries, `DISTINCT`, `SELECT *` (F-152); no `CASE`, `CAST`, `LIKE`, text or date functions (F-156); CPU only;
at most 48 registers (about 20 aggregates over different expressions) and 64 distinct constants in one program, said as an SQL error when passed; an AVG whose exact sum passes int64 raises like SUM.

Informal timings (F-150; 12-core Ryzen 9 5900X under a load average of about 6 from other jobs, so read them as rough; 5,000,000 rows generated with 8 / 10,000 / 1,000,000 groups; MX frame is
`sql()` on a table already in memory, best of 3 in one process, which includes planning and, the first time, the pass that finds a key's range; DuckDB 1.5.1 on a table it loaded, best of 3, its result
fetched into Python; neither is a quiet-machine benchmark). Every result was checked against DuckDB first.

| query | MX frame SQL | DuckDB | group strategy |
|---|---|---|---|
| `count(*), sum(v), avg(d), min(dt)` | 7 ms | 5 ms | none (one result row) |
| 8 groups, 4 aggregates | 8 ms | 6 ms | dense |
| `WHERE v > 100`, 2 keys (56 groups) | 8 ms | 11 ms | dense |
| 10k groups, `ORDER BY 2 DESC, 1 LIMIT 10` | 9 ms | 13 ms | dense |
| 1M groups, `ORDER BY 2 DESC, 1 LIMIT 10` | 133 ms | 92 ms | dense |
| 1M groups, all 1M rows out | 159 ms | 1,900 ms | dense |
| key `g % 100000` (an expression, bounded by interval arithmetic) | 21 ms | 70 ms | dense |
| `count(DISTINCT)` + `median`, 8 groups | 82 ms | 44 ms | dense |
| `quantile_disc`, 10k groups | 38 ms | 67 ms | dense |
| 10k x 7 groups, hashed (forced) | 152 ms | 71 ms | hashed |

The same queries over the Parquet file (one chunk per row group) are decode-bound: 80 to 110 ms to decode the three columns of 50 row groups, plus the kernel. A scan with very many groups merges a
partial result per row group (a dense partial per group is the whole domain), which is slow at 1M groups (1 to 2.5 s): batching row groups and the other big-GROUP-BY work over streamed files is F-158.

## 9. Later (Query and Explore modes)
More than one table and joins, SQL row queries, expression functions (M8: F-151 to F-162); pipelines with cached steps; versioned column
chunks for a time machine and branches; TPC-H through SQL.

## Memory rules
- A shared `Table` is read-only once shared and outlives every `Frame` built on it.
- Borrowed spans stay valid until the next edit to that column.
- Results own their buffers. Device buffers belong to the `DeviceContext` they were made on.

## Done means
omxplorer imports a CSV, edits, adds formulas, sorts, builds charts, pivots and a filtered Report, and saves and
reopens a `.omx`, with `from mxframe import …` in place of `mojoui/frame*.mojo`, `csvio.mojo` and `omxfile.mojo`;
`scripts/check` passes with the same numbers; and `scripts/build.sh omxplorer mac` links with no libpython.
