"""Composite the existing Left_licking honest threshold curve and SHAP
beeswarm into a single side-by-side figure that matches the layout of
the Scratching composite figure.

Reuses the already-generated PDFs at
  E:\RSVIDS\Blackbox\2511_Blackbox_Formalin\Barefoot_gui_test_claude\
  PixelPaws_Classifiers\HomeRig\licking_honest_correct_plots\
"""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import fitz

PLOT_DIR = Path(r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/licking_honest_correct_plots")
THR_PDF  = PLOT_DIR / "licking_threshold_curve.pdf"
SHAP_PDF = PLOT_DIR / "licking_shap_beeswarm.pdf"
OUT_DIR  = PLOT_DIR / "composite"
OUT_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = ("")

THR_W       = 360
THR_H       = 230
SHAP_W      = 640
SHAP_H      = 230
GAP_X       = 12
PAD_L = 6; PAD_R = 6; PAD_T = 6; PAD_B = 6


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-LickingComposite/1.0"})
    with urllib.request.urlopen(req, timeout=180) as r:
        step(f"  HTTP {r.status}")


def fit_rect(bb_w, bb_h, dest):
    dw, dh = dest.width, dest.height
    scale = min(dw / bb_w, dh / bb_h)
    new_w = bb_w * scale
    new_h = bb_h * scale
    cx = (dest.x0 + dest.x1) / 2
    cy = (dest.y0 + dest.y1) / 2
    return fitz.Rect(cx - new_w / 2, cy - new_h / 2,
                     cx + new_w / 2, cy + new_h / 2)


def main():
    for p in (THR_PDF, SHAP_PDF):
        if not p.exists():
            step(f"! missing {p}"); return 1

    out_w = PAD_L + THR_W + GAP_X + SHAP_W + PAD_R
    out_h = PAD_T + max(THR_H, SHAP_H) + PAD_B

    out = fitz.open()
    page = out.new_page(width=out_w, height=out_h)

    src_thr  = fitz.open(THR_PDF)
    src_shap = fitz.open(SHAP_PDF)
    thr_page  = src_thr[0]
    shap_page = src_shap[0]

    thr_dest = fitz.Rect(PAD_L, PAD_T, PAD_L + THR_W, PAD_T + THR_H)
    thr_fit  = fit_rect(thr_page.rect.width, thr_page.rect.height, thr_dest)
    page.show_pdf_page(thr_fit, src_thr, 0)

    shap_x0 = PAD_L + THR_W + GAP_X
    shap_dest = fitz.Rect(shap_x0, PAD_T, shap_x0 + SHAP_W, PAD_T + SHAP_H)
    shap_fit  = fit_rect(shap_page.rect.width, shap_page.rect.height, shap_dest)
    page.show_pdf_page(shap_fit, src_shap, 0)

    out_pdf = OUT_DIR / "licking_threshold_shap_composite.pdf"
    out.save(out_pdf, deflate=True, garbage=4)

    zoom = 240 / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix_w = page.get_pixmap(matrix=mat, alpha=False)
    pix_t = page.get_pixmap(matrix=mat, alpha=True)
    out_png_w = OUT_DIR / "licking_threshold_shap_composite_white.png"
    out_png_t = OUT_DIR / "licking_threshold_shap_composite.png"
    pix_w.save(out_png_w)
    pix_t.save(out_png_t)

    svg_str = page.get_svg_image(matrix=fitz.Matrix(1, 1))
    out_svg = OUT_DIR / "licking_threshold_shap_composite.svg"
    out_svg.write_text(svg_str, encoding="utf-8")

    src_thr.close(); src_shap.close(); out.close()
    step(f"wrote {out_pdf.name}, {out_png_w.name}, {out_png_t.name}, {out_svg.name}")

    head = ("**Left_licking — threshold curve + SHAP composite** (matches Scratching layout)\n"
            "Left: smoothed Precision / Recall / F1 vs threshold "
            "(honest 3-fold session-level GroupKFold, deployed pkl methodology, mean CV F1 = 0.907).\n"
            "Right: SHAP beeswarm — top 12 features ranked by mean |SHAP|, "
            "with horizontal bar chart of importance. "
            "Top features: Dis_snout-hlpaw, Ang_snout-centroid-hlpaw, hlpaw_Vel10, "
            "Ang_tailbase-flpaw-hlpaw, hlpaw_Vel2, snout_Vel10.")
    discord_upload(head, [out_pdf, out_png_w, out_png_t, out_svg])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
