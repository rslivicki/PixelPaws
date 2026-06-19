"""PixelPaws pipeline — single source of truth for Discord webhooks + canonical constants.

Import from here instead of hardcoding. Two channels:
  - PROCESSING_WEBHOOK : live progress / status trackers / training watchers
  - RESULTS_WEBHOOK    : final figures + stats for a finished analysis
"""
from __future__ import annotations

# --- Discord channels -------------------------------------------------------
# URLs are NOT committed: loaded from the gitignored pp_secrets.py, then env vars.
# See pp_secrets.example.py. (PROCESSING = live trackers; RESULTS = final figures/stats.)
import os as _os
try:
    from pp_secrets import PROCESSING_WEBHOOK, RESULTS_WEBHOOK
except Exception:
    PROCESSING_WEBHOOK = _os.environ.get("PP_PROCESSING_WEBHOOK", "")
    RESULTS_WEBHOOK = _os.environ.get("PP_RESULTS_WEBHOOK", "")

# Legacy single channel some older scripts still import as WEBHOOK; default it to processing.
WEBHOOK = PROCESSING_WEBHOOK

# --- Feature extraction (canonical 8aed1c22) --------------------------------
FEATURE_HASH = "8aed1c22"
FEATURE_CFG = dict(
    bp_pixbrt_list=["hrpaw", "hlpaw", "snout"],
    square_size=[40, 40, 40],
    pix_threshold=0.3,
    include_optical_flow=True,
    bp_optflow_list=["hrpaw", "hlpaw", "snout"],
)
EXPECTED_FEATURE_COLS = 635

# --- DeepLabCut -------------------------------------------------------------
DLC_CONFIG = r"E:\Sync_from_lab\2511_RSGNKK_Blackbox\config_local.yaml"
DLC_PYTHON = r"C:\ProgramData\Anaconda3\envs\DEEPLABCUT\python.exe"
# Retrained model adopted 2026-06-06: shuffle1 (validated more accurate than the old shuffle9/best-190).
# config_local.yaml is iteration:2, snapshotindex:best, so analyze(shuffle=1) resolves to the iteration-2
# snapshot best-460 — the most recent model. (Prior comment said "iteration-1 best-260"; that was stale.)
SHUFFLE = 1
DLC_BATCH = 64

# --- Transcode --------------------------------------------------------------
CODEC = "libx265"
CRF = "23"
PRESET = "slow"

# --- Repo -------------------------------------------------------------------
REPO = r"E:\Sync_from_lab\PixelPaws"


def post(msg: str, webhook: str = PROCESSING_WEBHOOK) -> None:
    """Fire-and-forget plain-text Discord post (truncates to 1900 chars)."""
    import json, urllib.request
    try:
        req = urllib.request.Request(
            webhook, data=json.dumps({"content": msg[:1900]}).encode(),
            method="POST", headers={"Content-Type": "application/json", "User-Agent": "PP/1.0"})
        urllib.request.urlopen(req, timeout=20).read()
    except Exception as e:
        print("discord err:", e, flush=True)
