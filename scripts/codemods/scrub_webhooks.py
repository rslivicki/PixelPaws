"""One-off: strip hardcoded Discord webhook URLs from the repo before pushing.

Handles BOTH forms:
  (a) single literal:  "https://discord.com/api/webhooks/<id>/<token>"
  (b) split literals:  "https://discord.com/api/webhooks/<id>/"  "<token>"
Replaces the whole thing with an empty string literal "" (valid Python).
Skips pp_secrets.py (gitignored, holds the real URLs), .git, __pycache__.
"""
import re
from pathlib import Path

ROOT = Path(r"E:\Sync_from_lab\PixelPaws")
# Quote-inclusive (handles BOTH " and ' quotes): URL literal + optional adjacent token literal.
PAT = re.compile(
    r'''(["'])https://discord\.com/api/webhooks/\d+/[A-Za-z0-9_-]*\1(\s*(["'])[A-Za-z0-9_-]+\3)?''')
# Bare (non-quoted, e.g. markdown/shell) full URL fallback.
BARE = re.compile(r"https://discord\.com/api/webhooks/\d+/[A-Za-z0-9_-]+")
EXTS = {".py", ".md", ".sh", ".txt", ".json", ".ipynb", ".cfg", ".yaml", ".yml"}
SKIP_DIRS = {".git", "__pycache__"}
SKIP_FILES = {"pp_secrets.py"}

changed = []
for p in ROOT.rglob("*"):
    if not p.is_file() or p.suffix.lower() not in EXTS:
        continue
    if any(part in SKIP_DIRS for part in p.parts) or p.name in SKIP_FILES:
        continue
    try:
        txt = p.read_text(encoding="utf-8")
    except Exception:
        continue
    new = PAT.sub('""', txt)
    new = BARE.sub("", new)
    if new != txt:
        p.write_text(new, encoding="utf-8")
        changed.append(str(p.relative_to(ROOT)))

print(f"scrubbed {len(changed)} file(s)")
for f in sorted(changed):
    print("  ", f)
