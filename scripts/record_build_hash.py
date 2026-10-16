#!/usr/bin/env python3
"""F-066: record, and check, which Mojo sources each committed kernels_aot/*.so was built from.

Each library gets a sidecar `kernels_aot/<lib>.so.sources.json` with the SHA-256 of every source it
compiles: its own .mojo file plus the kernels_aot/*.mojo files it imports, found by following
`from X import` / `import X` lines. One sidecar per library, so branches that rebuild different
libraries never conflict.

    python scripts/record_build_hash.py [lib.so ...]   record (default: every library); the build
                                                       scripts call this after set_runpath.sh
    python scripts/record_build_hash.py --check        exit 1 and name every stale library (CI)

Record only right after building from the current sources: the hash states "built from these".
"""
import hashlib
import json
import re
import sys
from pathlib import Path

# library -> (its main source, the `bash scripts/build_pipeline.sh <target>` that rebuilds it)
LIBRARIES = {
    "libmxexec_cpu.so": ("executor_cpu.mojo", "exec"),
    "libmxgroupkeys_cpu.so": ("groupkeys_cpu.mojo", "keys"),
    "libmxjoin_cpu.so": ("joinsemi_cpu.mojo", "join"),
    "libmxkernels_aot.so": ("kernels_aot.mojo", "aot"),
    "libmxkernels_aot_gpu.so": ("kernels_aot_gpu.mojo", "aot"),
    "libmxpipeline_cpu.so": ("pipeline_cpu.mojo", "cpu"),
    "libmxpipeline_gpu.so": ("pipeline_gpu.mojo", "gpu"),
}
_IMPORT = re.compile(r"^\s*(?:from\s+(\w+)\s+import|import\s+(\w+))", re.M)


def sources(root: Path, lib: str) -> list:
    """The library's main source and every kernels_aot module it imports, transitively, sorted."""
    kdir = Path(root) / "kernels_aot"
    seen, todo = set(), [LIBRARIES[lib][0]]
    while todo:
        name = todo.pop()
        if name in seen:
            continue
        seen.add(name)
        for m in _IMPORT.finditer((kdir / name).read_text()):
            dep = (m.group(1) or m.group(2)) + ".mojo"
            if (kdir / dep).exists():
                todo.append(dep)
    return [kdir / n for n in sorted(seen)]


def _hashes(root: Path, lib: str) -> dict:
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources(root, lib)}


def _sidecar(root: Path, lib: str) -> Path:
    return Path(root) / "kernels_aot" / (lib + ".sources.json")


def record(root: Path, libs) -> None:
    for lib in libs:
        _sidecar(root, lib).write_text(json.dumps({"sources": _hashes(root, lib)}, indent=2) + "\n")


def stale(root: Path, lib: str):
    """None when the recorded hashes match the sources, else the message to show."""
    fix = f"Rebuild with bash scripts/build_pipeline.sh {LIBRARIES[lib][1]}"
    sidecar = _sidecar(root, lib)
    if not sidecar.exists():
        return f"{lib} is stale: no recorded source hash. {fix}"
    recorded = json.loads(sidecar.read_text())["sources"]
    now = _hashes(root, lib)
    changed = sorted(n for n in now.keys() | recorded.keys() if now.get(n) != recorded.get(n))
    if not changed:
        return None
    return f"{lib} is stale: {', '.join(changed)} changed. {fix}"


def main(argv) -> int:
    root = Path(__file__).resolve().parent.parent
    if argv[:1] == ["--check"]:
        problems = [m for lib in LIBRARIES if (m := stale(root, lib))]
        print("\n".join(problems) or "kernel libraries match their sources")
        return 1 if problems else 0
    libs = [Path(a).name for a in argv] or list(LIBRARIES)
    record(root, libs)
    print("recorded source hashes for:", " ".join(libs))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
