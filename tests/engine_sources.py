"""The engine sources a test binary is built from, for the rebuild-when-newer checks. One place, so a renamed or moved
package directory can't leave a check comparing against nothing (F-165 review)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def engine_sources(*extra):
    """Every Mojo source of the package and the kernels, plus `extra` paths; raises if a directory matched nothing."""
    pkg = sorted((ROOT / "mojo/omxcore").glob("*.mojo"))
    kernels = sorted((ROOT / "kernels_aot").glob("*.mojo"))
    if not pkg or not kernels:
        raise RuntimeError("engine_sources: mojo/omxcore or kernels_aot has no .mojo files; was the package moved?")
    return pkg + kernels + [Path(p) for p in extra]
