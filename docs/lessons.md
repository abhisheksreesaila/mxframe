# Lessons

<!-- One line per lesson, added when you correct Claude or a mistake gets caught:
- YYYY-MM-DD · the rule that prevents it, and why
Claude reads this at the start of each session. -->
- 2026-09-30 · A cache keyed by `id(obj)` must hold a reference to `obj` and check identity on lookup; ids are reused after garbage collection, which returned another table's results (found in F-010 review).
