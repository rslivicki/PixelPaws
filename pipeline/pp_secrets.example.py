"""Template for pp_secrets.py (which is gitignored). Copy this to pp_secrets.py
and fill in the real Discord webhook URLs, OR set the PP_PROCESSING_WEBHOOK /
PP_RESULTS_WEBHOOK environment variables instead. pp_config.py loads from
pp_secrets.py first, then falls back to env vars.
"""
PROCESSING_WEBHOOK = ""  # live progress / status trackers
RESULTS_WEBHOOK = ""     # final figures + stats
