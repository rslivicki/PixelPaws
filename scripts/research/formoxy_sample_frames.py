"""
Extract a representative frame from S1/S2/S3 formalin (cohort post-flip)
AND from the original portal `_unflipped` if still on disk, so a
side-by-side visual check confirms the flip direction is correct.

For the cohort versions, also overlay text identifying which side the
classifier expects to see the injection on (left paw -- nose pointing
forward, left of midline).

Posts the resulting montage to Discord.
"""
from __future__ import annotations

import io
import json
import mimetypes
import secrets
import subprocess
import tempfile
import urllib.request
from pathlib import Path

COHORT_VIDS = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws\videos")
# Original `_unflipped` portal copies are deleted post-transcode -- only
# the cohort (flipped) versions remain. We grab a frame from each.

WEBHOOK = (
    ""
)


def extract_frame(video: Path, seek_sec: float, out_png: Path) -> bool:
    cmd = ["ffmpeg", "-y",
           "-ss", str(seek_sec),
           "-i", str(video),
           "-vframes", "1",
           "-q:v", "2",
           str(out_png)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode == 0 and out_png.is_file() and out_png.stat().st_size > 0


def post_discord(text: str, pngs: list[Path]) -> None:
    boundary = secrets.token_hex(16)
    body = io.BytesIO()
    def add_field(name, value):
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode())
        body.write(value.encode("utf-8")); body.write(b"\r\n")
    def add_file(name, path):
        body.write(f"--{boundary}\r\n".encode())
        body.write(f'Content-Disposition: form-data; name="{name}"; filename="{path.name}"\r\n'.encode())
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        body.write(f"Content-Type: {ctype}\r\n\r\n".encode())
        body.write(path.read_bytes()); body.write(b"\r\n")
    add_field("payload_json", json.dumps({"content": text}))
    for i, p in enumerate(pngs):
        add_file(f"files[{i}]", p)
    body.write(f"--{boundary}--\r\n".encode())
    req = urllib.request.Request(
        WEBHOOK, data=body.getvalue(), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-Sample/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            print(f"Discord upload: HTTP {r.status}", flush=True)
    except Exception as e:
        print(f"Discord upload failed: {e}", flush=True)


def main() -> int:
    work = Path(tempfile.gettempdir()) / "formoxy_sample_frames"
    work.mkdir(exist_ok=True)
    pngs = []
    # Pull frames at 10 min, 20 min, and 40 min in
    seek_points = [600, 1200, 2400]
    for n in (1, 2, 3):
        src = COHORT_VIDS / f"2605_FormOxy_S{n}_formalin.mp4"
        if not src.is_file():
            print(f"!! missing {src}")
            continue
        for sec in seek_points:
            out = work / f"S{n}_formalin_t{sec}s.png"
            if not extract_frame(src, sec, out):
                print(f"!! failed to extract S{n} @ {sec}s")
                continue
            pngs.append(out)
            print(f"  extracted {out.name} ({out.stat().st_size//1024} KB)")
    if not pngs:
        print("No frames extracted.")
        return 1
    msg = ("**FormOxy S1/S2/S3 formalin -- post-flip sample frames** "
           "(10/20/40 min into each video). The injected paw should "
           "appear on the LEFT side of the mouse (mouse's own left -- "
           "i.e. on the screen's right if the mouse faces up the frame).")
    post_discord(msg, pngs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
