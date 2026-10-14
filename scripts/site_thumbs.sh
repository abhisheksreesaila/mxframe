#!/usr/bin/env bash
# Thumbnails for the docs site's front page: the top of each page, light theme, 1280x720.
# Needs chromium. Run from the repo root after changing a page: scripts/site_thumbs.sh
set -euo pipefail
PORT=8766
pixi run python -m http.server "$PORT" -d docs/site >/dev/null 2>&1 &
SERVER=$!
trap 'kill $SERVER' EXIT
sleep 2
for page in benchmarks auto-mode internals how-it-got-fast; do
    chromium --headless=new --disable-gpu --hide-scrollbars --window-size=1280,720 \
        --virtual-time-budget=6000 --screenshot="docs/site/img/$page.png" \
        "http://localhost:$PORT/$page.html" 2>/dev/null
done
ls -la docs/site/img
