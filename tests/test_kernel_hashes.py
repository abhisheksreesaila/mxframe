"""F-066: every committed kernel library records the hash of the Mojo sources it was built from,
and a source that changed without a rebuild fails here, naming the build command."""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
KERNELS = ROOT / "kernels_aot"
LIBS = sorted(p.name for p in KERNELS.glob("*.so"))

_spec = importlib.util.spec_from_file_location("record_build_hash", ROOT / "scripts" / "record_build_hash.py")
rbh = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rbh)


@pytest.mark.parametrize("lib", LIBS)
def test_committed_library_matches_its_sources(lib):
    assert rbh.stale(ROOT, lib) is None


def test_every_library_is_registered():
    assert LIBS == sorted(rbh.LIBRARIES)


def test_source_sets_follow_imports():
    names = lambda lib: [p.name for p in rbh.sources(ROOT, lib)]
    assert names("libmxexec_cpu.so") == ["executor_cpu.mojo", "hashagg_cpu.mojo", "pipeline_cpu.mojo"]
    assert names("libmxkernels_aot_gpu.so") == ["_atomic.mojo", "kernels_aot_gpu.mojo"]
    assert names("libmxgroupkeys_cpu.so") == ["groupkeys_cpu.mojo"]


@pytest.fixture
def tree(tmp_path):
    shutil.copytree(KERNELS, tmp_path / "kernels_aot", ignore=shutil.ignore_patterns("*.so", "__pycache__"))
    for lib in LIBS:
        (tmp_path / "kernels_aot" / lib).write_bytes(b"x")
    rbh.record(tmp_path, LIBS)
    return tmp_path


def test_edited_import_makes_the_dependent_library_stale(tree):
    assert all(rbh.stale(tree, lib) is None for lib in LIBS)
    with open(tree / "kernels_aot" / "pipeline_cpu.mojo", "a") as f:
        f.write("\n# edit\n")
    assert rbh.stale(tree, "libmxexec_cpu.so") == (
        "libmxexec_cpu.so is stale: pipeline_cpu.mojo changed. "
        "Rebuild with bash scripts/build_pipeline.sh exec")
    assert rbh.stale(tree, "libmxpipeline_cpu.so") == (
        "libmxpipeline_cpu.so is stale: pipeline_cpu.mojo changed. "
        "Rebuild with bash scripts/build_pipeline.sh cpu")
    assert rbh.stale(tree, "libmxjoin_cpu.so") is None


def test_missing_record_is_stale(tree):
    (tree / "kernels_aot" / "libmxjoin_cpu.so.sources.json").unlink()
    assert rbh.stale(tree, "libmxjoin_cpu.so") == (
        "libmxjoin_cpu.so is stale: no recorded source hash. "
        "Rebuild with bash scripts/build_pipeline.sh join")


def test_record_is_a_per_library_sorted_json_file(tree):
    data = json.loads((tree / "kernels_aot" / "libmxexec_cpu.so.sources.json").read_text())
    assert list(data["sources"]) == ["executor_cpu.mojo", "hashagg_cpu.mojo", "pipeline_cpu.mojo"]
    assert all(len(h) == 64 for h in data["sources"].values())


def test_ci_runs_the_check():
    ci = (ROOT / ".github" / "workflows" / "test.yml").read_text()
    assert "scripts/record_build_hash.py --check" in ci
