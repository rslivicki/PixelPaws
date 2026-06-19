"""Re-composite Left_licking threshold + SHAP panels to match the
Scratching reference SVG layout exactly (page 955.65 x 273.6,
threshold panel 356.4 x 212.4 at (0, 20.28), SHAP panel
538.65 x 271.8 at (413.43, 0)).

Reuses the already-generated PDFs from licking_panels_plasma.py
so we keep the exact same plasma palette / density-jittered beeswarm
/ purple bar style as Scratching.
"""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import fitz

PANEL_DIR = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/licking_plasma_panels"
)
THR_PDF = PANEL_DIR / "PixelPaws_Licking_panelC_threshold.pdf"
SHAP_PDF = PANEL_DIR / "PixelPaws_Licking_panelD_shap.pdf"
OUT_DIR = PANEL_DIR

# Scratching SVG geometry (viewBox 955.65 x 273.6)
PAGE_W, PAGE_H = 955.65, 273.6
THR_X, THR_Y, THR_W, THR_H = 0.0, 20.28, 356.4, 212.4
SHAP_X, SHAP_Y, SHAP_W, SHAP_H = 413.43, 0.0, 538.65, 271.8

WEBHOOK = (
    ""
)


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        mime = {
            ".png": "image/png", ".pdf": "application/pdf",
            ".svg": "image/svg+xml",
        }.get(p.suffix.lower(), "application/octet-stream")
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; '
            f'filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "PixelPaws-LickingRecomposite/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def fit_rect(bb_w, bb_h, dest):
    dw, dh = dest.width, dest.height
    scale = min(dw / bb_w, dh / bb_h)
    new_w = bb_w * scale
    new_h = bb_h * scale
    cx = (dest.x0 + dest.x1) / 2
    cy = (dest.y0 + dest.y1) / 2
    return fitz.Rect(
        cx - new_w / 2, cy - new_h / 2,
        cx + new_w / 2, cy + new_h / 2,
    )


def main():
    for p in (THR_PDF, SHAP_PDF):
        if not p.exists():
            step(f"! missing {p}")
            return 1

    step("compositing to scratching-layout geometry "
         f"({PAGE_W} x {PAGE_H})")
    out = fitz.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)

    src_thr = fitz.open(THR_PDF)
    src_shap = fitz.open(SHAP_PDF)
    thr_page = src_thr[0]
    shap_page = src_shap[0]

    thr_dest = fitz.Rect(THR_X, THR_Y, THR_X + THR_W, THR_Y + THR_H)
    thr_fit = fit_rect(thr_page.rect.width, thr_page.rect.height, thr_dest)
    page.show_pdf_page(thr_fit, src_thr, 0)
    step(f"  threshold panel placed at {thr_fit}")

    shap_dest = fitz.Rect(SHAP_X, SHAP_Y, SHAP_X + SHAP_W, SHAP_Y + SHAP_H)
    shap_fit = fit_rect(shap_page.rect.width, shap_page.rect.height,
                        shap_dest)
    page.show_pdf_page(shap_fit, src_shap, 0)
    step(f"  shap panel placed at {shap_fit}")

    out_pdf = OUT_DIR / "PixelPaws_Licking_threshold_shap_scratching_layout.pdf"
    out.save(out_pdf, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = page.get_pixmap(matrix=mat, alpha=False)
    pix_t = page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = OUT_DIR / (
        "PixelPaws_Licking_threshold_shap_scratching_layout_white.png"
    )
    out_png_t = OUT_DIR / (
        "PixelPaws_Licking_threshold_shap_scratching_layout.png"
    )
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)

    src_thr.close()
    src_shap.close()
    out.close()
    step(f"wrote {out_pdf.name}, {out_png_w.name}, {out_png_t.name}")

    head = (
        "**Left_licking threshold + SHAP composite — re-laid-out to "
        "match Scratching reference figure.**\n"
        f"Page geometry: {PAGE_W}×{PAGE_H} (matches Scratching SVG "
        "viewBox exactly).\n"
        f"Threshold panel: {THR_W}×{THR_H} at ({THR_X}, {THR_Y}) — "
        "shifted down 20.28 to match Scratching.\n"
        f"SHAP panel: {SHAP_W}×{SHAP_H} at ({SHAP_X}, {SHAP_Y}) — "
        "top-aligned, takes ~56% of width.\n"
        "Plasma palette identical to Scratching panels (Recall=orange, "
        "Precision=pink, F1=purple, beeswarm=plasma Low→High, "
        "importance bars=purple)."
    )
    discord_upload(head, [out_png_w, out_png_t, out_pdf])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
