#!/usr/bin/env bash
# Build the Python-free Mojo package build/mxframe.mojoc (F-131).
#
# The API lives in mojo/mxframe/. The kernels stay in kernels_aot/ (one source: the Python path builds its .so
# files from them with flat imports, `from pipeline_cpu import ...`). They are copied into the package as
# mxframe/kernels/ with those imports made package-relative, because a .mojoc does not carry modules that were
# only on the -I path, and a shared library that re-imports @export functions does not export them.
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
stage="$root/build/pkg/mxframe"
rm -rf "$root/build/pkg"
mkdir -p "$stage/kernels"
cp "$root"/mojo/mxframe/*.mojo "$stage/"
cp "$root"/kernels_aot/*.mojo "$stage/kernels/"
echo '"""The native kernels (copied from kernels_aot/ by scripts/build_mojo_pkg.sh: edit them there)."""' > "$stage/kernels/__init__.mojo"
for f in "$root"/kernels_aot/*.mojo; do
  m="$(basename "$f" .mojo)"
  sed -i -E "s/^from $m import/from .$m import/" "$stage"/kernels/*.mojo
done
mojo package "$stage" -o "$root/build/mxframe.mojoc"
echo "built build/mxframe.mojoc"
