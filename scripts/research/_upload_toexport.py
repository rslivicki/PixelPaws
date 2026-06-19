"""Upload everything in C:/Users/Gereau/Desktop/Toexport to #results."""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

FOLDER = Path(r"C:\Users\Gereau\Desktop\Toexport")
WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".svg": "image/svg+xml",
                ".pdf": "application/pdf", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-Toexport-Upload/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}  ({len(files)} files)")


def main():
    files = sorted(FOLDER.iterdir())
    files = [f for f in files if f.is_file()]
    step(f"Uploading {len(files)} files from {FOLDER}:")
    for f in files:
        step(f"  - {f.name}  ({f.stat().st_size/1024:.0f} KB)")

    # Discord allows up to 10 attachments per message; we have 6 so one shot.
    head = (
        "**SNTX vs Naive paw-contour + intensity-ratio export package**\n"
        "Six files from the GUI export folder:\n"
        "* `contour_print_HL.pdf` / `contour_print_HR.pdf` -- representative "
        "paw-print outlines (HL, HR) for SNTX (red) vs Naive (black), with "
        "paw-like-filtered counts on each panel.\n"
        "* `contour_shape_HL.png` / `contour_shape_HR.png` -- group-mean "
        "contour overlay (the *averaged version* of the paw shape) with SD "
        "envelope per group, n=1500 each.\n"
        "* `Intensity_Ratio_-_TC.png` -- 5-min-bin HL/HR contour intensity "
        "ratio timecourse, SNTX vs Naive, mean +/- SEM. Two-way ANOVA: "
        "Treatment p<0.001***, Interaction p=0.518 ns. Per-bin asterisks at "
        "10, 15, 20, 25 min.\n"
        "* `Intensity_Ratio_-_TC_data_transformto_5minbins.csv` -- the "
        "underlying per-session per-minute intensity-ratio data."
    )
    upload(head, files)


if __name__ == "__main__":
    main()
