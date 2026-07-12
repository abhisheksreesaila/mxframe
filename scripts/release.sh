#!/usr/bin/env bash
set -euo pipefail

VERSION="0.5.0"
TAG="v${VERSION}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TAG_RELEASE=false
UPLOAD=false

usage() {
    cat <<EOF
Usage: scripts/release.sh [--tag] [--upload]

Without flags, validates tests, metadata, and distribution artifacts only.
  --tag     create ${TAG} and a GitHub release from RELEASE_NOTES_${VERSION}.md
  --upload  upload checked artifacts to PyPI with twine

Run --tag and --upload only from a clean, committed release tree.
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG_RELEASE=true ;;
        --upload) UPLOAD=true ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

cd "$ROOT"

PYTHON=(pixi run python)
PYPROJECT_VERSION="$("${PYTHON[@]}" -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
PACKAGE_VERSION="$("${PYTHON[@]}" -c 'import ast; tree=ast.parse(open("__init__.py").read()); print(next(n.value.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__version__" for t in n.targets)))')"

if [[ "$PYPROJECT_VERSION" != "$VERSION" || "$PACKAGE_VERSION" != "$VERSION" ]]; then
    echo "Version mismatch: script=$VERSION pyproject=$PYPROJECT_VERSION package=$PACKAGE_VERSION" >&2
    exit 1
fi

if [[ ! -f "RELEASE_NOTES_${VERSION}.md" ]]; then
    echo "Missing RELEASE_NOTES_${VERSION}.md" >&2
    exit 1
fi

if $TAG_RELEASE || $UPLOAD; then
    if [[ -n "$(git status --porcelain)" ]]; then
        echo "Refusing release action from a dirty worktree. Commit the validated release first." >&2
        exit 1
    fi
fi

echo "[1/5] Running regression suites"
pixi run test-all
pixi run test6-gpu

echo "[2/5] Rebuilding AOT libraries"
pixi run mojo build kernels_aot/kernels_aot.mojo -I kernels_aot \
    --emit shared-lib -o kernels_aot/libmxkernels_aot.so
pixi run mojo build kernels_aot/kernels_aot_gpu.mojo -I kernels_aot \
    --emit shared-lib -o kernels_aot/libmxkernels_aot_gpu.so

echo "[3/5] Building wheel and sdist"
rm -rf dist build
"${PYTHON[@]}" -m build

echo "[4/5] Checking package metadata and compiled artifacts"
if "${PYTHON[@]}" -m twine --version >/dev/null 2>&1; then
    TWINE=("${PYTHON[@]}" -m twine)
elif command -v uvx >/dev/null 2>&1; then
    TWINE=(uvx --from twine twine)
else
    echo "twine is required: install mxframe[dev] or install uv for isolated checks" >&2
    exit 1
fi
"${TWINE[@]}" check dist/*
"${PYTHON[@]}" - <<'PY'
from pathlib import Path
from zipfile import ZipFile

wheel = next(Path("dist").glob("mxframe-*.whl"))
with ZipFile(wheel) as archive:
    names = set(archive.namelist())
required = (
    "mxframe/kernels.mojopkg",
    "mxframe/kernels_aot/libmxkernels_aot.so",
    "mxframe/kernels_aot/libmxkernels_aot_gpu.so",
)
missing = [suffix for suffix in required if not any(name.endswith(suffix) for name in names)]
if missing:
    raise SystemExit(f"wheel missing compiled artifacts: {missing}")
print(f"wheel artifact check passed: {wheel}")
PY

echo "[5/5] Release actions"
if $TAG_RELEASE; then
    if git rev-parse "$TAG" >/dev/null 2>&1; then
        echo "Tag already exists: $TAG" >&2
        exit 1
    fi
    command -v gh >/dev/null || { echo "gh CLI is required for --tag" >&2; exit 1; }
    git tag -a "$TAG" -m "MXFrame ${TAG}"
    git push origin "$TAG"
    gh release create "$TAG" dist/* --title "MXFrame ${TAG}" \
        --notes-file "RELEASE_NOTES_${VERSION}.md"
fi

if $UPLOAD; then
    "${TWINE[@]}" upload dist/*
fi

if ! $TAG_RELEASE && ! $UPLOAD; then
    echo "Validation complete. Commit changes, then run with --tag and/or --upload."
fi
