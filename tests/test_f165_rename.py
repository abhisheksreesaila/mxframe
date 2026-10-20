"""F-165: the rename to omxcore keeps the old names working for one release.

Mojo: `bash scripts/build_mojo_pkg.sh` (the command in STABLE.md) makes build/omxcore.mojoc and build/mxframe.mojoc; a tiny
program compiled against each (`from omxcore import ...`, `from mxframe import ...`, the old one unchanged) runs and prints
the same sum.
Python: `import omxcore as mx` and `import mxframe` give the same module, submodules included, so one process never holds
two copies of the engine.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import omxcore

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
SHIM = str(Path(os.path.abspath(omxcore.__file__)).parent.parent)     # the directory holding `omxcore` and `mxframe`

PROGRAM = """from std.sys import argv
from {pkg} import read_csv, CsvOptions, Query, col, sum, CPU
from {pkg}.kernels._fpenv import core_count


def main() raises:
    var csv = read_csv(String(argv()[1]), CsvOptions())
    var t = csv.take_table().share()
    var out = Query(t).agg(sum(col("qty"))).compute(CPU)
    print("sum", out.values(0)[0], "cores>0", core_count() > 0)
"""


@pytest.mark.parametrize("pkg", ["omxcore", "mxframe"])
def test_mojo_program_compiles_and_runs_against_each_package_name(pkg, tmp_path):
    csv = tmp_path / "q.csv"
    csv.write_text("qty\n3\n5\n")
    src = tmp_path / f"prog_{pkg}.mojo"
    src.write_text(PROGRAM.format(pkg=pkg))
    exe = tmp_path / f"prog_{pkg}"
    subprocess.run(["bash", str(ROOT / "scripts/mojo_build.sh"), str(src), str(exe)], check=True, capture_output=True, text=True)
    assert (BUILD / f"{pkg}.mojoc").is_file()
    out = subprocess.run([str(exe), str(csv)], check=True, capture_output=True, text=True).stdout
    assert out.strip() == "sum 8 cores>0 True"


def _python(code):
    env = {**os.environ, "PYTHONPATH": SHIM}
    return subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True, env=env, cwd=SHIM).stdout


def test_mxframe_is_omxcore():
    out = _python(
        "import json, sys\n"
        "import omxcore as mx\n"
        "import mxframe\n"
        "import mxframe.native_exec as nx_old\n"
        "from mxframe.lazy_frame import LazyFrame as OldLazyFrame\n"
        "import omxcore.native_exec as nx_new\n"
        "from omxcore.lazy_frame import LazyFrame\n"
        "print(json.dumps({'same_module': mxframe is mx, 'same_submodule': nx_old is nx_new, 'same_class': OldLazyFrame is LazyFrame,\n"
        "                  'spec': nx_new.__spec__.name, 'name': nx_new.__name__}))\n")
    assert json.loads(out) == {"same_module": True, "same_submodule": True, "same_class": True,
                               "spec": "omxcore.native_exec", "name": "omxcore.native_exec"}


def test_old_import_name_still_runs_a_query():
    out = _python(
        "import json, pyarrow as pa\n"
        "import mxframe as mx\n"
        "t = pa.table({'k': [1, 1, 2], 'v': [10, 20, 30]})\n"
        "r = mx.LazyFrame(mx.Scan(t)).groupby('k').agg(mx.col('v').sum().alias('s')).compute(device='cpu').sort_by('k')\n"
        "print(json.dumps(r.to_pydict()))\n")
    assert json.loads(out) == {"k": [1, 2], "s": [30, 30]}
