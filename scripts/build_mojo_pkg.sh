#!/usr/bin/env bash
# Build the Python-free Mojo package build/mxframe.mojoc (F-131).
#
# The API lives in mojo/mxframe/. The kernels stay in kernels_aot/ (one source: the Python path builds its .so
# files from them with flat imports, `from pipeline_cpu import ...`). They are copied into the package as
# mxframe/kernels/ with those imports made package-relative, because a .mojoc does not carry modules that were
# only on the -I path, and a shared library that re-imports @export functions does not export them.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
out="$root/build/mxframe.mojoc"
mkdir -p "$root/build"
# Many builds may run at once (pytest -n workers, mojo-test beside pytest): one at a time behind a lock, staged in
# a private directory and published by rename, so a reader never sees a half-written package; skipped when the
# sources and the compiler are unchanged.
exec 9>"$root/build/.mxframe-pkg.lock"
flock 9
stamp="$( { mojo --version 2>/dev/null; cat "$root"/mojo/mxframe/*.mojo "$root"/kernels_aot/*.mojo "$0"; } | sha1sum | cut -c1-40)"
if [ -f "$out" ] && [ "$(cat "$out.stamp" 2>/dev/null)" = "$stamp" ]; then
  echo "build/mxframe.mojoc is up to date"
  exit 0
fi
work="$(mktemp -d "$root/build/pkg.XXXXXX")"
trap 'rm -rf "$work"' EXIT
stage="$work/mxframe"
mkdir -p "$stage/kernels"
cp "$root"/mojo/mxframe/*.mojo "$stage/"
cp "$root"/kernels_aot/*.mojo "$stage/kernels/"
echo '"""The native kernels (copied from kernels_aot/ by scripts/build_mojo_pkg.sh: edit them there)."""' > "$stage/kernels/__init__.mojo"
for f in "$root"/kernels_aot/*.mojo; do
  m="$(basename "$f" .mojo)"
  sed -i -E "s/^from $m import/from .$m import/" "$stage"/kernels/*.mojo
done
mojo precompile --disable-warnings "$stage" -o "$work/mxframe.mojoc"
mv -f "$work/mxframe.mojoc" "$out"
echo "$stamp" > "$out.stamp"
echo "built build/mxframe.mojoc"
