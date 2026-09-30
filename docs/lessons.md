# Lessons

<!-- One line per lesson, added when you correct Claude or a mistake gets caught:
- YYYY-MM-DD · the rule that prevents it, and why
Claude reads this at the start of each session. -->
- 2026-09-30 · A cache keyed by `id(obj)` must hold a reference to `obj` and check identity on lookup; ids are reused after garbage collection, which returned another table's results (found in F-010 review).
- 2026-09-30 · Judge correctness against DuckDB, never against our own CPU path: GPU Q6 (8.865e7) looked wrong next to CPU (4.07e7), but DuckDB gives 8.865e7 — the CPU path dropped boundary rows comparing float32 columns to float64 literals. GPU paths not yet verified against DuckDB go in `auto_mode.KNOWN_BROKEN_GPU`.
