"""No tracked text file may contain git conflict markers: a scripted merge twice committed docs/plan.md
with them (lessons, 2026-10-16 / 17)."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_no_tracked_file_has_conflict_markers():
    out = subprocess.run(["git", "-C", str(ROOT), "grep", "-lI", "-E", "^(<<<<<<< |>>>>>>> )", "--", "."],
                         capture_output=True, text=True)
    hits = [f for f in out.stdout.split() if f and not f.startswith("tests/test_no_conflict_markers")]
    assert not hits, f"conflict markers in: {hits}"
