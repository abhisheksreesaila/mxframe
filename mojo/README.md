# The Mojo package (no Python)

MX frame's Python-free API (`docs/mxframe-mojo-api.md`, brief `docs/briefs/mojo-api.md`).

- **Source:** the API in `mojo/mxframe/`. The native kernels stay in `kernels_aot/` (the Python path builds its
  `.so` files from them); `scripts/build_mojo_pkg.sh` copies them into the package as `mxframe/kernels/` with their
  imports made package-relative. Edit kernels in `kernels_aot/`, never in `build/`.
- **Build:** `pixi run bash scripts/build_mojo_pkg.sh` -> `build/mxframe.mojoc`. A program uses it with
  `mojo build app.mojo -I build`, plus `-Xlinker -lzstd -Xlinker -lsnappy` (Parquet codecs, C libraries) and their
  `-L` / `-rpath` (the pixi env's `lib/` here).
- **Tests:** `pixi run mojo-test` builds the package, then builds and runs each `mojo/tests/test_*.mojo` (a `main()`
  that raises on failure). A test binary that links libpython fails.
- **Lifetimes:** Mojo frees a value at its last use. A raw address taken from a `List` is dangling once the list
  is no longer used: keep the owner alive past the call (`_ = owner^`), or better, pass owned types, not addresses.

Why the kernels are copied, not imported from `-I kernels_aot`: a `.mojoc` does not carry modules that were only on
the `-I` path (a consumer then fails with "unable to locate module"), and a shared library that only re-imports
`@export` functions exports nothing, so the `.so` builds must keep compiling the original files.
