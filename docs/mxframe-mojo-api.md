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
| Kinds INT, DECIMAL(scale), TEXT (dictionary), DATE (days since 1970), BOOL; and FLOAT (float64 bits in the value slot; `append_float`, `get_float`; F-157, the section below) | INT, DECIMAL, TEXT, DATE; bools import as 0/1 | the Import step and the format inspector branch on kind |
| **Any number of columns** | 16 (`MAX_COLS`, a SIMD row vector) | wider CSVs lose columns past 16 today |
| **Nulls per column** (a validity bitmap) | none: empty cells read as 0 or "" and are only counted | Delete on a number cell; honest averages and counts |
| Column-major, contiguous int64 buffers the app can borrow as `Span[Int64]` | `List[Int64]` per column | the grid reads a screenful at a time; the GPU upload reads whole columns |
| Shared, read-only base data across sessions (web: many tabs, one table in memory) | `share()` returns an address; `release_columns()` empties it | the web host serves one process to many sessions |

### FLOAT64 (F-157)

FLOAT is a full column kind: the float64's bits in the Int64 value slot, the validity layout unchanged (`t.append_float(c, x)`, `t.get_float(c, row)`;
`Frame.set_float`, `Frame.get_float`). Bits are kept as given (-0.0, a NaN's payload, subnormals) in storage; the rules below make them one value where a
query compares, groups or counts them.

| Where | Rule |
|---|---|
| **Parquet** | DOUBLE is read as it is, FLOAT (32-bit) is widened to float64 exactly (its column has `scale == F32_COLUMN` (32), see "a 32-bit column" below); PLAIN, dictionary and BYTE_STREAM_SPLIT pages. Written as DOUBLE (PLAIN, no statistics): every bit survives a round trip. A float column's footer min / max are never read (`ParquetScan.column_range` says "unknown" for a FLOAT), so NaN in a footer cannot reach the planner's dense-group logic; null counts are used as for any column. |
| **CSV** | A column is FLOAT when its first `infer_rows` rows show a number with an exponent (`1e5`, `2.5E-3`), `nan` / `inf` / `-inf` / `infinity` (any case) beside numbers, more than 9 decimal places, or a number too wide for int64 at its scale (DuckDB reads those as DOUBLE); or when named in `CsvOptions(float_columns=[...])`. A decimal with 1 to 9 places stays an exact DECIMAL: scale 2 up to 2 places, 4 up to 4, and for 5 to 9 places the scale is that number of places, when every value of the first rows fits int64 at it (money with more places is not turned into a float). The value of a FLOAT cell is the correctly rounded float64 (an exact fast path, `strtod_l` in the C locale for the rest, whatever locale the host set); every NaN is the one positive quiet NaN; a cell that is not a number, or is past 1.8e308, is a failed cell (null). Past the window an exponent in an INT or DECIMAL column is a failed cell, and a DECIMAL column keeps rounding (counted in `rounded`). |
| **Comparisons** | DuckDB's order: NaN equals itself and is above every number (+inf included); -0.0 equals 0.0; NULL is unknown (SQL) or false (builder). |
| **Arithmetic** | INT and DECIMAL next to a FLOAT become float64 as DuckDB casts them: `float(v) / 10^scale` (an integer past 2^53 raises: it is not exact as a double). `+ - *` and `/` (always a division of doubles, IEEE: `x / 0` is inf, -inf or NaN); `//` and `%` on a FLOAT are not supported. A literal with an exponent is a DOUBLE; a decimal literal next to a FLOAT is read as `float(v) / 10^scale`. |
| **SUM / AVG** | float64 in a defined order, the kernel's reproducible mode (F-064): within one run, each 65,536-row morsel adds its rows in order and the morsels' totals are added in order, so the bits do **not** depend on the number of threads (`Query.threads(n)`, `PlannerOptions.threads`), the morsel setting or the block size (tested). They do depend on the **plan and the source layout**, because those change which additions happen in which order: (1) dense and hashed groups give different last bits for the same query (a dense run keeps up to 4 partial sums per group below 1,025 groups, a hashed run one per morsel: `PlannerOptions.group` forces one); (2) a table read whole is one run over its rows, whereas a Parquet scan is one run per row group folded in file order (morsels restart at each group's first row), so `read_parquet` + a query and `scan_parquet` + the same query can differ in the last bits (the exact integers, counts, min, max and medians never do); (3) a Frame is its shared rows, then the changed rows added one by one; (4) the SIMD width of the machine sets the lanes of a global sum, so two machines may differ. The same query on the same data, plan and machine gives the same bits every run. A scan with a float sum always runs row group by row group, never by row-range morsels (those are cut by the number of cores). The tests compare with DuckDB to 1e-12 relative. AVG is the float sum over the count. |
| **MIN / MAX** | NaN is the largest value: MAX is NaN when the group holds one, MIN is NaN only when it holds nothing else (a hidden count of the NaNs, so the kernel never compares one). |
| **Group keys** | SQL only (the builder's key is a dense integer): hashed, never dense. -0.0 and 0.0 are one group, every NaN is one group (reported as the canonical NaN), NULL is its own group; ORDER BY puts NaN after +inf and NULLs last. `PlannerOptions(group=GROUP_DENSE)` on a FLOAT key raises and says why. |
| **count distinct, median, quantile_disc, top_k** | on order-preserving keys (-0.0 = 0.0, one NaN above +inf): `median` is the middle value, or for an even count `lo * 0.5 + hi * 0.5` (DuckDB's interpolation); `quantile_disc` is an actual value. `QueryResult.floats(i)` gives a float aggregate's values as Float64 (`is_float(i)`); `values(i)` and `top(i, g)` give their bits. |
| **Frame** | `set_float`, `set_text` (`"1e3"`, `"nan"`, `"-0.0"`, `""` is null), `append` strings; a formula with a float (an exponent literal, a FLOAT column, `/` of a FLOAT) is a FLOAT formula column (`+ - * /`, `min`, `max`; `//` and `%` are refused); `permutation` orders floats (NaN last, nulls last either way); `bake` and `save` write DOUBLE. A base cell is null by its validity bit only, so any INT / DECIMAL / DATE / FLOAT value is data, including 9221120237041090561 and -9223372036854775807. Edited and appended cells mark a null with the NaN `0x7FF8000000000001` (`NULL_EDIT`; `set` refuses it as a value, `set_float` stores that NaN as the canonical NaN), and the changed rows a query reads (`Overlay.delta`) carry their own `null_marker`, which differs from it when a changed row holds it as data. |
| **A 32-bit column** | DuckDB's REAL. A Parquet FLOAT column is read widened to float64 (`scale == 32`), and SQL gives it DuckDB's type rules: REAL next to REAL, INT or DECIMAL is **REAL** (a literal or DECIMAL column is cast to 32 bits, so `l_discount <= 0.07` keeps the rows holding 0.07, and `f = d` compares in 32 bits); REAL next to a DOUBLE (a double column, an exponent literal such as `7e-2`) is **DOUBLE**; one DOUBLE in an `IN` list makes the whole list DOUBLE. REAL `+ - * /` and negation stay REAL (each result rounded to 32 bits, as DuckDB computes it); `sum` and `avg` of REAL give DOUBLE, `min`, `max`, `median` and `quantile_disc` give REAL. Forms not modelled raise by name: REAL meeting a non-literal INT or DECIMAL, or REAL arithmetic, *after* aggregation (`max(f) + min(f)`), and a DECIMAL with more than 10 places next to REAL. The builder has no 32-bit rule: it reads the column as a double, with `lit_float(x)` (a double) and `lit_f32(x)` (rounded to 32 bits). |
| **GPU** | A query that reads a FLOAT column or does float arithmetic runs on the CPU: `compute(GPU)` raises, AUTO says why in `fallback_reason` (a float64 sum has one defined order there). |
| **Builder** | `col("x") * 1.5 + col("y") / col("n")`, `lit_float`, `lit_f32`, `Query.threads(n)`; a float group key is refused (use SQL). Aggregates: `sum`, `mean`, `min_of`, `max_of`, `count_distinct`, `median`, `percentile`, `top_k`; read them with `r.floats(i)`. |

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
| **CSV/TSV reader** with type inference | integers → INT; decimals → DECIMAL with scale 2 if every value has ≤ 2 places, else 4; an exponent, more than 4 places or nan / inf → FLOAT (F-157); ISO dates and timestamps → DATE; true/false → BOOL; else TEXT. Quoting, embedded newlines, BOM, `\r\n`. |
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

## 8. SQL over one table (F-150, row queries F-152)

```mojo
var cat = Catalog()
cat.add_table("orders", orders)              # a SharedTable
cat.add_frame("edited", frame)               # a Frame, snapshotted now (its edit counter is `SqlSource.version`): add it again after more edits
cat.add_scan("big", scan_parquet(path))      # a Parquet file, read a row group at a time (a Frame over a scan works too)
var r = sql(cat, "SELECT region, sum(revenue) AS total FROM orders WHERE qty > 2 GROUP BY region HAVING count(*) > 5 ORDER BY total DESC LIMIT 10")
r.table                                      # an owned Table: INT, DECIMAL, TEXT, DATE, BOOL and FLOAT
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
| Select list | expressions with aliases; group keys, aggregates, and expressions over them (`sum(a) / count(*)`, `max(d) - min(d)`, `g + 1`); a number or date constant; with no aggregate and no `GROUP BY` the query returns rows (below): `*`, `t.*`, columns and expressions; `SELECT DISTINCT` |
| `WHERE` | `= <> != < <= > >=`, `BETWEEN`, `IN (literals)` (numbers: up to 40; text: any number), `IS [NOT] NULL`, `AND OR NOT`, a BOOL column; text, DATE (`DATE '2024-01-31'` or a string next to a DATE column) and decimal literals (aligned to the column's scale); `+ - *`, `//`, `%` on INT and DECIMAL (scales aligned as DuckDB does) and `date + n`, `date - date` |
| `GROUP BY` | columns, expressions, ordinals (`GROUP BY 1`), select aliases; none at all (one group) |
| `HAVING`, `ORDER BY` | aggregates and group keys in any expression (a row query: any column or expression, selected or not, several keys); `ORDER BY` keys, aggregates, aliases, ordinals; `ASC` / `DESC`; `NULLS FIRST` / `LAST` (NULLS LAST by default, also for `DESC`, as DuckDB); `LIMIT n [OFFSET m]` |
| Aggregates | `SUM`, `COUNT(*)`, `COUNT(x)`, `COUNT(DISTINCT x)`, `MIN`, `MAX` (also of TEXT), `AVG`, `MEDIAN`, `QUANTILE_DISC(x, p)` |

**SQL's rules** (the Query builder above keeps its own; both run on the same kernels):
three-valued logic in `WHERE` / `HAVING` (a row is kept only when the condition is true; `x IN (1, NULL)` is true or unknown; `NOT` of unknown is unknown);
NULL is a group of its own; only groups with rows exist (a global aggregate over no rows still returns its one row); `SUM` / `MIN` / `MAX` / `AVG` / `MEDIAN`
of no non-null value are NULL, `COUNT` is 0; `x // 0` and `x % 0` are NULL and both truncate toward zero; a word the column never held matches nothing;
`/` always gives a FLOAT with IEEE division by zero as DuckDB (`1 / 0` is inf, `-1 / 0` is -inf, `0 / 0` is NaN; NaN equals itself and sorts after every number), `AVG` gives a FLOAT that is exactly what DuckDB computes (its long-double division `sum / (count * 10^scale)`, reproduced bit for bit).
Text order everywhere (`ORDER BY`, `MIN` / `MAX`, `QUANTILE_DISC`, `<` and `BETWEEN` against a literal) is alphabetical ignoring ASCII case, with byte order breaking ties; `=` and `IN` compare bytes.

**Differences from DuckDB, on purpose**: a `SUM` (or any int64 arithmetic) past int64 raises `a sum or an expression passed the int64 range (about 9.2e18)` where DuckDB
widens to HUGEINT (decimal `*` too); `MEDIAN` of TEXT is the lower middle word (DuckDB's, in the text order), `MEDIAN` of a DATE raises (DuckDB's is a TIMESTAMP, a kind MX frame does not have; use `quantile_disc(d, 0.5)`), `MEDIAN` of a DECIMAL is exact at one more decimal place (DuckDB rounds to the column's scale; the median of integers is a FLOAT, as DuckDB's DOUBLE);
`//` and `%` on a FLOAT, and `//` on a DECIMAL (a floor of doubles in DuckDB), raise; the text order above; FLOAT64 in SQL follows the section below.

**How a query runs.** The statement is parsed (`sql.mojo`), bound to a typed tree (`bind.mojo`: types, scales, which expressions run per row and which after aggregation), turned into a
logical plan (`plan.mojo`: Scan, Filter, Aggregate, Filter (HAVING), Project, Sort, Limit, with the rewrites that look at its shape), planned from statistics only (`planner.mojo`: the `Planner`
trait, `RulePlanner`, `Stats`, `PlannerOptions`; it chooses the group strategy and the sink), lowered to one kernel program in SQL's mode (`sqlprog.mojo`), and run by the executor over the
source's chunks (`exec.mojo`: the Sources; `sqlexec.mojo`: the aggregate Sinks; `rows.mojo`: the row Sinks; `post.mojo`: the stage after aggregation). A table is one chunk; a frame is its shared table with
the changed rows masked out plus a small table of the changed and appended rows; a scan is one chunk per row group. The chunks' partial results (sums and counts add, min / max combine,
hashed groups merge by key) fold into one.

After aggregation the groups become a small table and HAVING, the select list's expressions, ORDER BY, OFFSET and LIMIT run over it as a row query, by the same executor and the same kernel
program lowering as any other row query (there is no second expression evaluator; a FLOAT is a float register, a number that meets one is converted as DuckDB casts it).

| Group strategy | When | How |
|---|---|---|
| none | no `GROUP BY` | one group, the plain kernel |
| dense | every key is a column whose values lie in a known range: a TEXT dictionary, BOOL, or the min / max of an INT / DECIMAL / DATE column of a table or frame (one pass, cached per source), and the domain (a NULL slot per nullable key, keys mixed-radix) is at most `dense_limit` (4,194,304) and either at most 65,536 or at most 4x the rows | the kernel adds into slot `g` directly |
| hashed | a key is an expression, or has no known range (an INT column of a scan), or the domain is too wide or too sparse | `hashagg_cpu.hash_aggregate`: per-worker hash tables over the key words (the value, plus a NULL flag word for a nullable key), a partitioned merge |

`count(DISTINCT)`, `MEDIAN`, `QUANTILE_DISC` and `MIN` / `MAX` of TEXT use the holistic states of §5; over hashed groups they take a second pass that maps each row to its group
by key (`holistic: a second pass ...`). `ORDER BY` with `LIMIT + OFFSET <= 4096` over more than 8x as many groups takes a bounded heap; otherwise stable radix passes, last key first.

`r.notes` is the contract the tests assert: `source: ...`, `group: dense, 2 keys g 7 (with a NULL slot) x s 17 ... = 119 slots: <reason>` or `group: hashed on N key words: <reason>`
(`forced by the options` when `PlannerOptions` said so), `kernel: N instructions a row (M slots)`, `ran: table|frame|scan ...: <chunks>`, `holistic: ...`, `order: ...`, `having: ...`, `time: ...`.

**A file's footer statistics are a hint, not a promise.** The planner uses a Parquet footer's min / max (dense groups for an integer key) and null counts (no NULL checks) but every row group is verified as it is read: a decoded group whose null count differs from the stated one, or a row outside a stated range (dense keys over a file carry one extra group that such rows fall into), makes the query run again without the statistics (hashed groups, every column nullable) and `notes` has `statistics: the footer of `t` cannot be believed (...); ran again without them`.

**Nesting**: an expression may nest 256 levels deep (parentheses, `NOT`, a repeated minus, or a chain of that many terms joined by one operator); more raises a positioned error.

**A frame's formula columns** (§4) are columns of the table: the query reads the formula's expression over the stored columns, inlined, with the builder's own rules for it (floor `//` and `%`,
`x // 0` is 0, `min` / `max`) and exact decimals; they work in `WHERE`, `GROUP BY` (hashed) and the aggregates, over a table frame and a frame over a scan.

**Limits**: one table (joins: F-153); `DISTINCT` only without aggregates; no `CASE`, `CAST`, `LIKE`, text or date functions (F-156); CPU only;
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

### Row queries (F-152)

A query with no aggregate and no `GROUP BY` returns rows: `SELECT *`, `SELECT t.*, x + 1 AS y`, `WHERE`, `ORDER BY` (several keys, ASC / DESC, NULLS FIRST / LAST, any expression), `LIMIT`, `OFFSET`, `SELECT DISTINCT`.
The result is an owned Table, as always; a frame's formula columns are columns (and part of `*`); text keeps the source's words (the result's dictionary holds the words present). A select item can be a column, an
arithmetic expression over INT / DECIMAL / DATE columns, a comparison or logical expression (a BOOL column of true / false / NULL) or a constant; a string constant and a bare `NULL` are refused by name; a FLOAT (a column of the file, or `/`) is selected, ordered and made distinct like any other kind, in DuckDB's order (NaN equal to itself and above +inf, -0.0 equal to 0.0).

| Part | What runs |
|---|---|
| materialize (the sink) | each chunk runs one kernel program: the WHERE (and a frame's mask) and the expressions of the select list and ORDER BY that are not plain columns, into registers. The kernel's block hook (`rows._rows_hook`, the `_mat_hook` pattern) writes, per block of 1,024 rows, the numbers of the selected rows and the computed values at the block's own place, and a prefix sum over the blocks' counts compacts them. A plain column is not carried through registers: it is gathered from the chunk by the selected rows' numbers, so a wide `SELECT *` costs no registers |
| `ORDER BY` | the collected rows are sorted by a stable LSD radix pass per key, from the last key to the first (text under the owner's order, NULLS FIRST / LAST per key); ties keep the source order (a table's rows, a frame's changed rows after them) |
| `ORDER BY ... LIMIT k [OFFSET m]` | when `k + m` is at most 4,096 and the source has more than 8x that many rows, no row is materialized: each worker keeps its best `k + m` rows in a heap inside the hook (a first-key test drops most rows before anything is copied), the workers' heaps are merged with the best so far after each chunk, and the result is cut to `[m, m + k)`. Ties keep the source order. EXPLAIN says `order: top-K by per-worker heaps in the hook` and the notes `rows: the best K of N rows kept in per-worker heaps`; otherwise `order: all rows sorted (why)` |
| `LIMIT n` with no `ORDER BY` | the scan stops once it has `n + offset` rows: a table is read in slices, a scan stops reading row groups (`ran: scan t: 1 of 13 row groups read`, `rows: N rows selected (stopped early: the LIMIT was met)`). Which rows come first is the source order |
| `SELECT DISTINCT a, b` | the group-by of the select list (the binder rewrites it; EXPLAIN shows `Distinct keys=[a, b]`), so it takes the dense or hashed strategy with the reason; `DISTINCT` with aggregates or `GROUP BY` raises (use `GROUP BY`); its `ORDER BY` terms must be in the select list |

Informal timings (F-152; the same 12-core Ryzen 9 5900X under a load average of about 5 from other jobs, so rough; the F-150 test generator's table of 1,200,000 rows and 9 columns written to Parquet; MX frame is the in-process
`sql()` time reported by `mx` (the file read apart), best of 3 fresh processes, the result a Table; DuckDB 1.5.1 on a table it loaded, best of 3, the result an Arrow table; every result is checked against DuckDB in `tests/test_mojo_rows.py`):

| query | rows out | MX frame | DuckDB |
|---|---|---|---|
| `SELECT id, x, d FROM t WHERE x > 500` | 275,347 | 12 ms | 9 ms |
| `SELECT * FROM t` | 1,200,000 | 114 ms | 59 ms |
| `SELECT id, x * 2 + 1 AS y, d + e AS de, dt FROM t WHERE k = 1` | 215,586 | 16 ms | 12 ms |
| `... WHERE x > 0 LIMIT 100` (the scan stops early) | 100 | 0.2 ms | 0.2 ms |
| `SELECT id, x FROM t ORDER BY x DESC, id LIMIT 10` (heaps) | 10 | 2.9 ms | 1.4 ms |
| `SELECT id, s, dt FROM t ORDER BY s, dt DESC LIMIT 50 OFFSET 100` (heaps, text key) | 50 | 3.5 ms | 6.0 ms |
| `SELECT id, x, s FROM t WHERE x IS NOT NULL ORDER BY x, id` (full sort) | 1,103,764 | 114 ms | 47 ms |
| `SELECT id, s FROM t ORDER BY s, id` (full sort, text) | 1,200,000 | 116 ms | 104 ms |
| `SELECT DISTINCT s FROM t` | 16 | 1.3 ms | 4.0 ms |
| `SELECT DISTINCT k, b FROM t` | 21 | 2.4 ms | 3.5 ms |
| `SELECT DISTINCT g FROM t` (hashed, 983,770 groups) | 983,770 | 45 ms | 22 ms |

Aggregates after the move of HAVING / ORDER BY / the select list onto the executor (same data; 983,770 groups for `g`): `GROUP BY g ORDER BY 2 DESC, 1 LIMIT 10` 57 ms (DuckDB 20),
`GROUP BY g` with all groups out 108 ms (61), `GROUP BY g HAVING count(*) > 1 ORDER BY sum(x) / count(*) DESC LIMIT 10` 67 ms (18), 7 groups 3.6 ms (2.0). DuckDB is faster where the result is a million rows or the query is a
full sort; the heaps, the early stop and `DISTINCT` over few values are where this executor wins or ties.

### The `mx` command line (F-151, for agents)

`mx` is the SQL of section 8 as one native program, built by `pixi run bash scripts/build_mx.sh` into `build/mx` (no Python in it; the script checks that). A cold `mx sql` on a small
file starts, runs and exits in about 11 ms wall (20 fresh processes in a shell loop, process creation included).

```
mx sql "<SQL>" [-t name=path ...] [-f query.sql] [--format json|jsonl|csv|parquet] [--out file] [--max-rows N] [--device cpu|gpu|auto]
mx explain [--analyze] "<SQL>" [-t name=path ...] [--format text|json]
mx schema <file> [--rows N]
```

- **Files.** `FROM 'orders.parquet'` and `FROM 'orders.csv'` open the file named in the query; `-t name=path` registers a file under a name (`FROM name`). A Parquet file is streamed a row group at a
  time when it is over a quarter of the machine's memory (or `--stream always`; `--stream never` reads it whole; `--memory BYTES` changes the memory it compares with), otherwise read into memory.
  The notes say which. A CSV is read whole, and as UTF-8 only: a file with a byte that is not UTF-8 (Latin-1 `caf\xe9`) or a UTF-16 byte order mark is an error naming the byte and line, with a hint to re-save it (`iconv -f LATIN1 -t UTF-8`); a UTF-8 byte order mark is skipped. The result is therefore always valid UTF-8. The advanced switches `--group auto|dense|hashed`, `--dense-limit N` and `--threads N` are F-150's `PlannerOptions`.
- **Device.** SQL runs on the CPU. `--device gpu` is accepted and the notes say it ran on the CPU; `device` in the result is `"cpu"`.
- **Result (stdout, one JSON object).**
  `{"columns":[{"name","kind","scale"}],"rows":[[...]],"row_count","truncated","ms","device","notes"}`.
  `kind` is INT, DECIMAL, TEXT, DATE, BOOL or FLOAT. An INT is a JSON integer. **A DECIMAL is a JSON number whose text is exact** (`1234.50`, scale digits kept): parse it with a decimal parser
  (`json.loads(text, parse_float=Decimal)`), never a float one. A DATE is `"yyyy-mm-dd"`, TEXT is the word, BOOL is `true` / `false`, NULL is `null`. A FLOAT is its shortest round-trip number;
  infinity and NaN, which JSON cannot hold, are the strings `"Infinity"`, `"-Infinity"` and `"NaN"`. `rows` holds at most `--max-rows` rows (default 1000; 0 means all); `row_count` is the true
  count and `truncated` says whether `rows` is shorter. `notes` is the planner's account of what ran (section 8): read it to see the group strategy, the columns read, the chunks, any re-run.
- **Other formats.** `jsonl` is one object per row (`{"k":0,"total":12.50}`), `csv` has a header and is written as DuckDB writes it: a cell is quoted when it holds a comma, a quote or a line break, **the empty string is `""` and NULL is an empty cell**, dates and decimals as in JSON. `--out file` writes every row to the file
  (whatever `--max-rows` says) and prints the JSON summary (`columns`, `row_count`, `ms`, `notes`, `out`, no `rows`). `--format parquet` needs `--out` (a FLOAT result column is written as Parquet DOUBLE). `--out` never creates directories: a missing one is an exit-1 error `couldn't write ...`; JSON, JSON lines and CSV are written in chunks of about a megabyte.
- **Errors.** A query error (bad SQL, unknown column, unsupported form, a file that cannot be opened) prints `{"error":{"message","line","column","hint"}}` on stdout and exits **1**; `line` and `column`
  are 1-based into the SQL text (null when the error is not about a position), `hint` carries the "did you mean" suggestion or what to do. A usage error (unknown command or option, wrong
  number of arguments, a bad value) prints the same object with the usage text as the hint and exits **2**. Exit 0 is success.
- **`mx explain`** prints the logical and physical plan as text (`--format json`: `{"plan":[lines],"analyzed","ms","notes"}`); `--analyze` also runs the SELECT and adds what ran and the time.
  `mx sql "EXPLAIN SELECT ..."` works too.
- **`mx schema file`** (Parquet or CSV): `{"file","format","bytes","rows","row_groups","columns":[{"name","kind","scale","nulls","distinct_words","first_values"}]}`; `distinct_words` is the size of a TEXT
  column's dictionary (null for other kinds), `first_values` the first `--rows` values (default 5) in the result's JSON form. A Parquet file is read from its footer and a few row groups.

Things an agent should know:

- **A long OR chain fails.** The parser stops expressions nested 256 levels deep, and an `a OR b OR c ...` chain of about 255 terms is that deep. Write `x IN (a, b, c)` instead. A list of **numbers** in
  an IN takes at most 40 values (`the most is 40`); split a longer one into several queries or filter by a range. A list of text words has no such limit.
- **A table is one file.** One table per query until joins (F-153); `FROM` is a registered name or a quoted file name. A row query (`SELECT *`, `WHERE`, `ORDER BY`, `LIMIT`) works; `--max-rows` caps what is printed
  (default 1000, `row_count` is the real count).
- **A Parquet file with a DOUBLE (or FLOAT) column opens (F-157).** FLOAT is a column kind: see "FLOAT64 (F-157)" in section 1 for the rules (NaN order, -0.0, the reproducible sum, a 32-bit column compared with a literal in 32 bits). A Parquet text value that is not UTF-8 is refused.
- **A footer that lies.** When a Parquet footer's min / max or null count turns out wrong while reading, the query runs again without the statistics and the notes say `statistics: ... ran again without
  them`; an EXPLAIN of that run says `statistics not believed` where a file with no statistics says `the source has no statistics on it`.
- Text order is the owner's (case-insensitive, bytes as the tie-break), so `min(s)` / `ORDER BY s` on text differ from DuckDB's byte order; DECIMAL `median` has one more digit of scale than DuckDB.

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


**FLOAT as text.** Wherever a FLOAT is written as text (`mx` JSON / JSON lines / CSV), it is the shortest decimal that reads back to the same bits
(`0.1` is `0.1`), in Python's `repr` layout (`float_text` in `floatfmt.mojo`; Mojo's own `String(Float64)` is shortest for all but about 1 double in 1,000, which it
prints one digit short, so the text is read back and rebuilt from the exact expansion when it fails). inf, -inf and NaN are `"Infinity"`, `"-Infinity"` and `"NaN"` in JSON.
