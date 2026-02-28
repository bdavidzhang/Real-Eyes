#!/bin/bash
# Launch modal_streaming.py (serve) and modal_app.py (batch) simultaneously.
# Prints the streaming URL as soon as modal serve emits it.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
IMAGE_FOLDER="${1:-./office_loop}"
SUBMAP_SIZE="${2:-16}"
MAX_LOOPS="${3:-1}"

STREAM_LOG=$(mktemp /tmp/modal_stream_XXXXX.log)

echo "========================================================"
echo "  VGGT-SLAM Modal Launcher"
echo "========================================================"
echo "  Image folder : $IMAGE_FOLDER"
echo "  Submap size  : $SUBMAP_SIZE"
echo "  Max loops    : $MAX_LOOPS"
echo "  Stream log   : $STREAM_LOG"
echo "========================================================"
echo ""

# ── 1. Start streaming server in background, tee output to log ──────────────
echo "[streaming] Starting modal serve..."
modal serve "$SCRIPT_DIR/modal_streaming.py" 2>&1 | tee "$STREAM_LOG" &
STREAM_PID=$!

# ── 2. Wait until the *.modal.run URL appears in the log ────────────────────
echo "[streaming] Waiting for URL..."
STREAM_URL=""
for i in $(seq 1 60); do
    STREAM_URL=$(grep -oP 'https://[a-z0-9\-]+\.modal\.run' "$STREAM_LOG" | head -1)
    if [ -n "$STREAM_URL" ]; then
        break
    fi
    sleep 2
done

if [ -z "$STREAM_URL" ]; then
    echo "[streaming] ERROR: URL not found after 120s. Check $STREAM_LOG"
    kill "$STREAM_PID" 2>/dev/null
    exit 1
fi

echo ""
echo "========================================================"
echo "  STREAMING LIVE AT:"
echo ""
echo "    $STREAM_URL"
echo ""
echo "========================================================"
echo ""

# ── 3. Start batch SLAM job in foreground ───────────────────────────────────
echo "[batch] Starting modal run..."
modal run "$SCRIPT_DIR/modal_app.py" \
    --image-folder "$IMAGE_FOLDER" \
    --submap-size "$SUBMAP_SIZE" \
    --max-loops "$MAX_LOOPS"

# ── 4. Cleanup streaming server on exit ─────────────────────────────────────
echo "[streaming] Shutting down serve..."
kill "$STREAM_PID" 2>/dev/null
rm -f "$STREAM_LOG"
