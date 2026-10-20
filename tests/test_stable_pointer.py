"""STABLE (one line: the stable commit) is what apps' scheduled checks read; STABLE.md says the same for people."""
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_stable_pointer_matches_stable_md_and_is_a_commit():
    commit = (ROOT / "STABLE").read_text().strip()
    assert re.fullmatch(r"[0-9a-f]{7,40}", commit), commit
    md = (ROOT / "STABLE.md").read_text()
    current = re.search(r"## Current stable\s+\*\*`([0-9a-f]+)`\*\*", md)
    assert current and current.group(1) == commit, "STABLE and STABLE.md's current stable must agree"
    p = subprocess.run(["git", "-C", str(ROOT), "cat-file", "-t", commit], capture_output=True, text=True)
    if p.returncode == 0:
        assert p.stdout.strip() == "commit"
