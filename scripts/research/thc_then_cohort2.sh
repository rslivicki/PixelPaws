#!/usr/bin/env bash
# Orchestrator: wait for THC DLC to finish (30 h5s), run post-DLC chain,
# then restart cohort2 DLC. Designed to run in background.

WEBHOOK=""
THC_DIR="/e/RSVIDS/Blackbox/260506_RS_THC_Withdrawal/Videos"

echo "[$(date +%H:%M:%S)] Orchestrator: waiting for THC DLC to hit 30 filtered h5s..."
# Count only *_filtered.h5 so we know analyze_videos AND filterpredictions
# are complete for every video.
while true; do
  cur=$(ls "$THC_DIR" 2>/dev/null | grep -E "shuffle9.*_filtered\.h5$" | wc -l)
  if [ "$cur" -ge 30 ]; then
    echo "[$(date +%H:%M:%S)] THC filtered h5 count = $cur. All done."
    break
  fi
  sleep 60
done

echo "[$(date +%H:%M:%S)] Running post-DLC chain..."
cd /e/PixelPaws
PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_post_dlc_chain.py
chain_rc=$?
echo "[$(date +%H:%M:%S)] Chain exit: $chain_rc"

if [ "$chain_rc" -ne 0 ]; then
  curl -s -X POST -H "Content-Type: application/json" \
    -d "{\"content\":\"THC chain failed (exit $chain_rc). Cohort2 DLC NOT restarted - check logs.\"}" "$WEBHOOK" >/dev/null
  exit $chain_rc
fi

echo "[$(date +%H:%M:%S)] Restarting cohort2 SNLT DLC..."
curl -s -X POST -H "Content-Type: application/json" \
  -d '{"content":"Restarting cohort2 SNLT DLC (29 videos remaining)."}' "$WEBHOOK" >/dev/null
PYTHONIOENCODING=utf-8 "C:/Users/Gereau/anaconda3/envs/DEEPLABCUT/python.exe" scripts/research/dlc_batch_2605_cohort2.py
echo "[$(date +%H:%M:%S)] Cohort2 DLC done."
