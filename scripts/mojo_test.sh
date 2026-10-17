#!/usr/bin/env bash
# Build the package, then build and run every mojo/tests/test_*.mojo against it (each has a main() that raises on
# failure). Usage: pixi run mojo-test [test files...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bash "$root/scripts/build_mojo_pkg.sh" > "$root/build/pkg-build.log" 2>&1 || { grep -E "error:" -A4 "$root/build/pkg-build.log" | head -40; echo "FAIL (package build)"; exit 1; }
lib="$root/.pixi/envs/default/lib"
out="$root/build/mojo-tests"
mkdir -p "$out"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=("$root"/mojo/tests/test_*.mojo)
fail=0
for t in "${files[@]}"; do
  name="$(basename "$t" .mojo)"
  if ! mojo build --disable-warnings "$t" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy -Xlinker -rpath="$lib" \
       -o "$out/$name" > "$out/$name.build.log" 2>&1; then
    echo "FAIL (build) $name"; tail -20 "$out/$name.build.log"; fail=1; continue
  fi
  # The package must never need Python (brief: no libpython). ldd alone is not enough: Mojo's Python interop
  # dlopens libpython at run time, so look for the interop's runtime symbols in the binary too.
  if ldd "$out/$name" | grep -i python > /dev/null \
     || nm -D "$out/$name" 2>/dev/null | grep KGEN_CompilerRT_Python > /dev/null \
     || strings "$out/$name" | grep -E "^Py_Initialize" > /dev/null; then
    echo "FAIL $name uses Python (libpython linked or Mojo Python interop)"; fail=1; continue
  fi
  if "$out/$name" > "$out/$name.log" 2>&1; then echo "ok   $name"; else echo "FAIL $name"; tail -20 "$out/$name.log"; fail=1; fi
done
exit $fail
