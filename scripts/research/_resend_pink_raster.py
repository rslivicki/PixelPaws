import json, urllib.request, uuid
from pathlib import Path
WEBHOOK = ("")
base = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected/Classifiers/plots/pink_rasters")
files = [
    base / "PixelPaws_Scratching_PinkRaster_251030_S3_Rimonabant.png",
    base / "PixelPaws_Scratching_PinkRaster_251030_S3_Rimonabant.pdf",
    base / "PixelPaws_Scratching_PinkRaster_251030_S3_Rimonabant.svg",
]
content = (
    "**Re-send — Scratching pink-style evaluation raster** "
    "(session 251030_S3_Rimonabant, F1 = 0.878). Three panels: "
    "ethogram (model vs human) | confusion matrix (RdPu colormap) | "
    "10-s binned correlation. PNG + PDF (editable text) + SVG."
)
boundary = "----PP" + uuid.uuid4().hex
mime_map = {".png": "image/png", ".pdf": "application/pdf", ".svg": "image/svg+xml"}
body = [
    f"--{boundary}".encode(),
    b'Content-Disposition: form-data; name="payload_json"',
    b'Content-Type: application/json', b"",
    json.dumps({"content": content}).encode(),
]
for i, p in enumerate(files):
    body += [
        f"--{boundary}".encode(),
        f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
        f"Content-Type: {mime_map[p.suffix.lower()]}".encode(), b"",
        p.read_bytes(),
    ]
body.append(f"--{boundary}--".encode())
req = urllib.request.Request(
    WEBHOOK, data=b"\r\n".join(body), method="POST",
    headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
             "User-Agent": "PixelPaws-Resend/1.0"})
with urllib.request.urlopen(req, timeout=120) as r:
    print("HTTP", r.status)
