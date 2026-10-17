#!/usr/bin/env bash
# Build the package, then build and run every mojo/tests/test_*.mojo against it (each has a main() that raises on
# failure). Usage: pixi run mojo-test [test files...]
set -euo pipefail
root="$(cd "$(dirname "$0")/.." && pwd)"
bash "$root/scripts/build_mojo_pkg.sh" >/dev/null
lib="$root/.pixi/envs/default/lib"
out="$root/build/mojo-tests"
mkdir -p "$out"
files=("$@")
[ ${#files[@]} -eq 0 ] && files=("$root"/mojo/tests/test_*.mojo)
fail=0
for t in "${files[@]}"; do
  name="$(basename "$t" .mojo)"
  if ! mojo build "$t" -I "$root/build" -Xlinker -L"$lib" -Xlinker -lzstd -Xlinker -lsnappy -Xlinker -rpath="$lib" \
       -o "$out/$name" > "$out/$name.build.log" 2>&1; then
    echo "FAIL (build) $name"; tail -20 "$out/$name.build.log"; fail=1; continue
  fi
  if ldd "$out/$name" | grep -qi python; then        # the package must never need Python (brief: no libpython)
    echo "FAIL $name links libpython:"; ldd "$out/$name" | grep -i python; fail=1; continue
  fi
  if "$out/$name" > "$out/$name.log" 2>&1; then echo "ok   $name"; else echo "FAIL $name"; tail -20 "$out/$name.log"; fail=1; fi
done
exit $fail
