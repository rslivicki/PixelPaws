"""Replace the 8 blurry raster images on the Academic Showcase poster
with their vector PDF counterparts, in place. The original raster's
placement rectangle is preserved; the vector is centered+scaled to fit.

Output: Academic_Showcase_poster_vectorswap.pdf (next to the source).
"""
from __future__ import annotations
from pathlib import Path
import fitz

SRC = Path(r"C:/Users/Gereau/Downloads/Academic_Showcase_poster copy.pdf")
OUT = SRC.with_name("Academic_Showcase_poster_vectorswap.pdf")

# xref in the source PDF -> vector PDF to drop in its place
SWAPS = {
    37: r"E:/RSVIDS/Blackbox/260515_Rim_DoseResp/analysis/rim_scratching_time_metrics.pdf",
    39: r"E:/RSVIDS/Blackbox/2605_FormOxy_LeftPaws/analysis/leftlicking_timecourse_5min_total_s.pdf",
    55: r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/scratch_lick_combined/scratching_licking_ethogram_figure.pdf",
    58: r"E:/RSVIDS/Blackbox/2511_Blackbox_Formalin/Barefoot_gui_test_claude/PixelPaws_Classifiers/HomeRig/licking_plasma_panels/Scratching_Licking_combined_panels_nogrid.pdf",
    61: r"E:/RSVIDS/Blackbox/2605_FormOxy_LeftPaws/analysis/leftlicking_phase_auc_bars.pdf",
    63: r"E:/RSVIDS/Blackbox/2605_FormOxy_LeftPaws/analysis/oxy_licking_time_metrics.pdf",
    65: r"E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/analysis/sntx_contour_intensity_tc.pdf",
    77: r"E:/RSVIDS/Blackbox/2603_SNLT_JG/Baseline/transitions/_classifier_f1.pdf",
}


def fit_rect(bb_w: float, bb_h: float, dest: fitz.Rect) -> fitz.Rect:
    dw, dh = dest.width, dest.height
    s = min(dw / bb_w, dh / bb_h)
    nw, nh = bb_w * s, bb_h * s
    cx, cy = (dest.x0 + dest.x1) / 2, (dest.y0 + dest.y1) / 2
    return fitz.Rect(cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2)


def main():
    for p in SWAPS.values():
        if not Path(p).exists():
            raise SystemExit(f"missing vector: {p}")

    doc = fitz.open(SRC)
    page = doc[0]

    swapped = 0
    for xref, vec_path in SWAPS.items():
        rects = page.get_image_rects(xref)
        if not rects:
            print(f"  ! xref {xref}: no placement rect, skipping")
            continue

        rect = rects[0]
        # Cover original raster with white before overlaying vector,
        # so any aspect-ratio mismatch shows white not pixels behind.
        page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1), width=0,
                       overlay=True)

        src = fitz.open(vec_path)
        sp = src[0]
        fit = fit_rect(sp.rect.width, sp.rect.height, rect)
        page.show_pdf_page(fit, src, 0)
        src.close()

        swapped += 1
        print(f"  xref {xref}: placed {Path(vec_path).name} "
              f"in {rect.width:.1f}x{rect.height:.1f} pt rect "
              f"at ({rect.x0:.0f},{rect.y0:.0f})")

    doc.save(OUT, deflate=True, garbage=4)
    doc.close()
    print(f"\nwrote {OUT} ({swapped}/{len(SWAPS)} swaps)")


if __name__ == "__main__":
    main()
