"""Combine Scratching + Left_licking threshold/SHAP panels into ONE
composite figure, with tight gap between threshold curve and SHAP
beeswarm (no excess whitespace).

Layout:
  Row 1 (Scratching):  [thr] [shap]
  Row 2 (Left_licking): [thr] [shap]

Panel aspect ratios preserved from source PDFs:
  threshold: 1.241  -> 248 x 200 per row
  shap:      1.982  -> 396 x 200 per row
"""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import fitz

SCRATCH_THR = Path(
    r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected/"
    r"Classifiers/plots/pdf_exports/PixelPaws_Scratching_panelC_threshold.pdf"
)
SCRATCH_SHAP = Path(
    r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected/"
    r"Classifiers/plots/pdf_exports/PixelPaws_Scratching_panelD_shap.pdf"
)
LICK_THR = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/licking_plasma_panels/"
    r"PixelPaws_Licking_panelC_threshold.pdf"
)
LICK_SHAP = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/licking_plasma_panels/"
    r"PixelPaws_Licking_panelD_shap.pdf"
)

OUT_DIR = Path(
    r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/"
    r"PixelPaws_Classifiers/HomeRig/licking_plasma_panels"
)

# --- Layout knobs (units = PDF points) ---
ROW_H = 200                       # per-row panel height
THR_W = round(ROW_H * 1.241, 2)   # match source PDF aspect (1.241)
SHAP_W = round(ROW_H * 1.982, 2)  # match source PDF aspect (1.982)
GAP_X = 4                         # horizontal gap (was 57 in scratching ref)
GAP_Y = 10                        # vertical gap between rows
PAD_L = 4
PAD_R = 4
PAD_T = 4
PAD_B = 4

PAGE_W = PAD_L + THR_W + GAP_X + SHAP_W + PAD_R
PAGE_H = PAD_T + 2 * ROW_H + GAP_Y + PAD_B

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
            "User-Agent": "PixelPaws-ScratchingLickingCombined/1.0",
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


def place(page, src_pdf, dest):
    src = fitz.open(src_pdf)
    sp = src[0]
    fit = fit_rect(sp.rect.width, sp.rect.height, dest)
    page.show_pdf_page(fit, src, 0)
    src.close()
    return fit


def main():
    for p in (SCRATCH_THR, SCRATCH_SHAP, LICK_THR, LICK_SHAP):
        if not p.exists():
            step(f"! missing {p}")
            return 1

    step(f"page geometry: {PAGE_W:.1f} x {PAGE_H:.1f}")
    out = fitz.open()
    page = out.new_page(width=PAGE_W, height=PAGE_H)

    # Row 1 — Scratching
    row1_y = PAD_T
    thr_dest = fitz.Rect(PAD_L, row1_y, PAD_L + THR_W, row1_y + ROW_H)
    shap_x0 = PAD_L + THR_W + GAP_X
    shap_dest = fitz.Rect(shap_x0, row1_y, shap_x0 + SHAP_W, row1_y + ROW_H)
    f1 = place(page, SCRATCH_THR, thr_dest)
    f2 = place(page, SCRATCH_SHAP, shap_dest)
    step(f"  row 1 (Scratching) thr={f1}, shap={f2}")

    # Row 2 — Left_licking
    row2_y = PAD_T + ROW_H + GAP_Y
    thr_dest2 = fitz.Rect(PAD_L, row2_y, PAD_L + THR_W, row2_y + ROW_H)
    shap_dest2 = fitz.Rect(shap_x0, row2_y, shap_x0 + SHAP_W, row2_y + ROW_H)
    f3 = place(page, LICK_THR, thr_dest2)
    f4 = place(page, LICK_SHAP, shap_dest2)
    step(f"  row 2 (Left_licking) thr={f3}, shap={f4}")

    out_pdf = OUT_DIR / "Scratching_Licking_combined_panels.pdf"
    out.save(out_pdf, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = page.get_pixmap(matrix=mat, alpha=False)
    pix_t = page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = OUT_DIR / "Scratching_Licking_combined_panels_white.png"
    out_png_t = OUT_DIR / "Scratching_Licking_combined_panels.png"
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)

    out.close()
    step(f"wrote {out_pdf.name}, {out_png_w.name}, {out_png_t.name}")

    head = (
        "**Scratching + Left_licking combined panels** (one file, tight gap).\n"
        f"Page {PAGE_W:.0f}×{PAGE_H:.0f}.  Row 1 = Scratching, "
        "Row 2 = Left_licking.  Per row: threshold curve (left) + "
        "SHAP beeswarm+bar (right).  Horizontal gap between threshold "
        f"and SHAP reduced to {GAP_X}pt (was 57 in scratching ref).  "
        "Same plasma palette across both rows (Recall=orange, "
        "Precision=pink, F1=purple, beeswarm=plasma Low→High, "
        "importance bars=purple)."
    )
    discord_upload(head, [out_png_w, out_png_t, out_pdf])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
