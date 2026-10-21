#!/usr/bin/env bash
# Thumbnails for the docs site's front page: the top of each page, light theme, 1280x720.
# Needs chromium. Run from the repo root after changing a page: scripts/site_thumbs.sh
# PORT=<n> when 8766 is taken (another checkout may be serving docs/site); BASE=http://127.0.0.1:<n> to use a server that is already running.
set -euo pipefail
PORT=${PORT:-8766}
if [ -z "${BASE:-}" ]; then
    pixi run python -m http.server "$PORT" -d docs/site >/dev/null 2>&1 &
    SERVER=$!
    trap 'kill $SERVER' EXIT
    sleep 4
    BASE="http://127.0.0.1:$PORT"
fi
for page in benchmarks auto-mode internals how-it-got-fast; do
    chromium --headless=new --no-sandbox --disable-gpu --hide-scrollbars --window-size=1280,720 \
        --virtual-time-budget=8000 --screenshot="docs/site/img/$page.png" \
        "$BASE/$page.html" 2>/dev/null
done
ls -la docs/site/img
