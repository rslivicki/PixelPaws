#!/usr/bin/env bash
# THC orchestrator: poll for 30 *_filtered.h5 (DLC + filterpredictions complete
# for all 30 sessions), then run the post-DLC chain (extract-with-flow ->
# predict-all -> group-analyze -> Discord push).
#
# Designed to run in background:
#   bash scripts/research/thc_orchestrator.sh >> scripts/research/thc_orchestrator.log 2>&1 &

WEBHOOK=""
THC_DIR="/e/RSVIDS/Blackbox/260506_RS_THC_Withdrawal/Videos"
TARGET=30

echo "[$(date +%Y-%m-%d_%H:%M:%S)] Orchestrator started. Waiting for $TARGET filtered .h5 files in $THC_DIR"

last_count=-1
while true; do
  cur=$(ls "$THC_DIR" 2>/dev/null | grep -E "shuffle9.*_filtered\.h5$" | wc -l)
  if [ "$cur" -ne "$last_count" ]; then
    echo "[$(date +%H:%M:%S)] filtered h5 count: $cur / $TARGET"
    last_count=$cur
  fi
  if [ "$cur" -ge "$TARGET" ]; then
    echo "[$(date +%H:%M:%S)] Hit target. DLC done."
    break
  fi
  sleep 60
done

echo "[$(date +%H:%M:%S)] Running post-DLC chain (extract -> predict -> analyze)..."
cd /e/PixelPaws
PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_post_dlc_chain.py
chain_rc=$?
echo "[$(date +%H:%M:%S)] Post-DLC chain exit: $chain_rc"

if [ "$chain_rc" -ne 0 ]; then
  curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"content\":\"THC chain FAILED (exit $chain_rc) - check thc_orchestrator.log\"}" \
    "$WEBHOOK" >/dev/null
fi

echo "[$(date +%H:%M:%S)] Orchestrator done."
exit $chain_rc
