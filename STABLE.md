# Stable engine version for apps

Apps that vendor the Mojo package (`mxframe.mojoc`, soon `omxcore`) read this file to know which commit to build from.
Scripts read the one-line file `STABLE` (just the commit). When a new version becomes stable, an issue labelled
`mxframe` is also opened on the app's repo (abhisheksreesaila/mxapp), and the app's scheduled check updates itself.
Only commits listed here are stable: every ticket in them passed adversarial review, and the full test suites were
green. Anything newer on `main` may still be in review: don't vendor it.

## Current stable

**`23edc59`** (2026-10-20)

```sh
# build the package from a clean copy of the stable commit (never from a working checkout)
git -C <mxframe> archive 23edc59 | tar -x -C <scratch>
cd <scratch> && pixi run bash scripts/build_mojo_pkg.sh      # -> build/mxframe.mojoc
cp build/mxframe.mojoc <app>/vendor/ && echo 23edc59 > <app>/vendor/MXFRAME_COMMIT
```

## What changed since the previous stable (321870f), for apps

| Area | What you get | Ticket |
|---|---|---|
| Background work | `compute_job`, `sql_job`, `read_csv_job`, `read_parquet_job`, `save_job`, `write_parquet_job`, `permutation_job` return a `Job[T]`: `poll()`, `wait()`, `result()`, `cancel()` (answered at once; a cancelled save keeps the old file), `progress()` (done / total / stage), an optional `JobDone` callback. On Linux each job runs on its own thread; on macOS / serial builds jobs run inside the start call unless the app builds with `-D MXFRAME_JOB_THREADS` and allows `_pthread_create _pthread_join _pthread_detach _pthread_attr_init _pthread_attr_setstacksize _pthread_attr_destroy _sched_yield` (check `JOBS_ARE_THREADED`) | F-175 |
| Ranges and blocks | `range_aggregate(source, [order,] cols, a, b)` (count / sum / mean / min / max per column, exact at each scale) and `read_block(source, [order,] cols, a, b, values, nulls)` (rows a..b of n columns in one call, column-major) over a Table or Frame (edits, merges, over a file); formula columns raise for now | F-176 |

## What changed in 321870f (since 1889b7f), for apps

| Area | What you get | Ticket |
|---|---|---|
| SQL | `sql(catalog, text) -> SqlResult` (an owned `Table`): SELECT, WHERE, GROUP BY (several keys, NULL is its own group), HAVING, ORDER BY (several keys, NULLS FIRST / LAST), LIMIT / OFFSET, `SELECT *`, DISTINCT, EXPLAIN [ANALYZE] | F-150, F-152 |
| Aggregates | sum, count, avg, min, max, count(DISTINCT), median, quantile_disc per group | F-148, F-150 |
| Expressions | CASE, COALESCE, NULLIF, CAST / TRY_CAST, long IN lists, text functions once per word (lower, upper, trim, substr, replace, LIKE / ILIKE, contains, `\|\|`), dates (EXTRACT year / quarter / month / dow, date_trunc, ± INTERVAL, date differences) | F-156 |
| Joins | inner, LEFT, SEMI, ANTI; IN / EXISTS / NOT IN subqueries; 3-9 tables with a cost-based planner | F-153..F-155 |
| Types | FLOAT64 columns (Parquet DOUBLE / FLOAT, CSV floats); `CsvOptions(decimal_columns=)` for exact money | F-157 |
| `mx` | command line for agents and scripts: SQL in, JSON / CSV / Parquet out | F-151 |

**API changes to check when you update:** `Frame` base nulls come from validity bits; `median` is interpolated (scale
+1); `word_count` / `text_of` / `code_of` raise; CSV columns with more than 4 decimal places read as FLOAT unless
named in `decimal_columns`.

**Mac link (apps that build for macOS):** allow these libSystem symbols in the app's stub and allowed list:
`_sysconf _memset_pattern16 _strtod _strtod_l _newlocale _freelocale` (the package's `scripts/check_mac_link.sh`
prints the current list).

## Asking for something
Open an issue on this repo with the label **`omxplorer`** (or your app's name): what you need, what you do meanwhile,
and a repro for a bug. Needs that are generic go into the engine; app-specific ones are built as plugins outside it.

## History
| Stable | Date | Highlights |
|---|---|---|
| `23edc59` | 2026-10-20 | background jobs (cancel, progress), range aggregates and block reads |
| `321870f` | 2026-10-19 | SQL, joins, planner, expressions, FLOAT64, `mx` |
| `1889b7f` | 2026-10-19 | GPU for streamed files |
| `7cbcc04` | 2026-10-18 | editing streamed files, count distinct / median / top-k, faster streaming |
| `581250b` | 2026-10-18 | faster Parquet decode |
