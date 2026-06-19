"""
White-background, portrait (stacked) version of the raw ↔ schematic
two-panel figure.

Top panel: raw cropped frame (grayscale).
Bottom panel: white-bg schematic — mouse rendered as a magenta-tinted
silhouette over white, dim dark-grey skeleton, bright keypoint dots
with dark outlines, feature annotations (Pix_/Ang_/Dis_/Vel_).
"""
from __future__ import annotations

import argparse, json, os, sys, time, urllib.request, uuid
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from feature_schematic import (
    find_video, find_dlc_h5, find_cache_pkl,
    grab_frame_gray, grab_frame_bgr,
    load_dlc_xy, keypoints_at,
    angle_at_vertex,
    SKELETON_EDGES, PAW_BRIGHTNESS_BPS, ROI_HALF,
)


WEBHOOK = (
    ""
)


def step(msg): print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml"}.get(p.suffix.lower(),
                                              "application/octet-stream")
        body += [
            f"--{boundary}".encode(),
            f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
            f"Content-Type: {mime}".encode(), b"",
            p.read_bytes(),
        ]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-WhiteSchematic/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def render_white_schematic(bgr_full, kps_full, frame_idx, kps_future,
                           out_stem: Path, size: int = 600,
                           vel_trail=None, vel_step: int = 10):
    H, W = bgr_full.shape[:2]
    half = size // 2

    cx0, cy0 = kps_full['centroid']
    x0 = int(max(0, min(W - size, round(cx0) - half)))
    y0 = int(max(0, min(H - size, round(cy0) - half)))
    bgr = bgr_full[y0:y0 + size, x0:x0 + size].copy()
    kps = {bp: (x - x0, y - y0) for bp, (x, y) in kps_full.items()}
    fut = {bp: (x - x0, y - y0) for bp, (x, y) in kps_future.items()}

    # White canvas — render only the paw / snout *contours*, not the whole
    # mouse body. For each Pix_<bp> ROI, Otsu-threshold inside the ROI to
    # get the bright paw silhouette, then alpha-blend it magenta onto white.
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    paw_halfs = {'hlpaw': 40, 'hrpaw': 40, 'snout': 24,
                 'flpaw': 18, 'frpaw': 18}
    paw_mask = np.zeros((size, size), dtype=np.uint8)
    for bp, hp in paw_halfs.items():
        if bp not in kps:
            continue
        cx, cy = int(round(kps[bp][0])), int(round(kps[bp][1]))
        x1, x2 = max(0, cx - hp), min(size, cx + hp)
        y1, y2 = max(0, cy - hp), min(size, cy + hp)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = gray[y1:y2, x1:x2]
        # Otsu picks paw vs (darker) surrounding fur / floor
        _, m = cv2.threshold(roi, 0, 255,
                             cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8), iterations=1)
        paw_mask[y1:y2, x1:x2] = np.maximum(paw_mask[y1:y2, x1:x2], m)

    soft = (paw_mask.astype(np.float32) / 255.0)
    # Slight darken inside paw silhouette so contour reads firmly
    soft = soft * 0.85
    canvas = np.full((size, size, 3), 255, dtype=np.float32)  # white BGR
    magenta = np.array([180, 60, 200], dtype=np.float32)
    for c in range(3):
        canvas[..., c] = (1 - soft) * canvas[..., c] + soft * magenta[c]

    # Skeleton lines — dark grey-blue, readable on white.
    SKEL_BGR = (90, 70, 60)
    for a, b in SKELETON_EDGES:
        if a in kps and b in kps:
            xa, ya = int(round(kps[a][0])), int(round(kps[a][1]))
            xb, yb = int(round(kps[b][0])), int(round(kps[b][1]))
            cv2.line(canvas, (xa, ya), (xb, yb),
                     SKEL_BGR, 1, cv2.LINE_AA)

    # Keypoint dots — bright but with dark outlines.
    DLC_BGR = {
        'snout':    (200, 140,  20),
        'neck':     (180, 120,  60),
        'centroid': (150, 110,  80),
        'tailbase': ( 30,  90, 200),
        'tailtip':  ( 50, 110, 220),
        'frpaw':    (180,  40, 200),
        'flpaw':    (180,  40, 200),
        'hrpaw':    (150,  20, 200),
        'hlpaw':    (150,  20, 200),
    }
    canvas = canvas.astype(np.uint8)
    for bp, color in DLC_BGR.items():
        if bp not in kps:
            continue
        x, y = int(round(kps[bp][0])), int(round(kps[bp][1]))
        cv2.circle(canvas, (x, y), 5, color,        -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, (0, 0, 0),     1, cv2.LINE_AA)

    # ── matplotlib annotations on the white panel ──────────────────
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100,
                     facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('white')
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Color key — darker accents so they read on white.
    BRIGHT_C = '#1e9d4a'   # green
    ANGLE_C  = '#c1245a'   # deep pink
    DIST_C   = '#c25a17'   # burnt orange
    VEL_C    = '#5b3a9a'   # purple

    FONT_KW = dict(family=['Arial', 'DejaVu Sans'], weight='bold', size=8)

    def _bbox(color):
        return dict(boxstyle='round,pad=0.16',
                    facecolor='white', alpha=0.92,
                    edgecolor=color, linewidth=0.7)

    # Brightness ROI squares
    for bp in PAW_BRIGHTNESS_BPS:
        if bp not in kps:
            continue
        x, y = kps[bp]
        ax.add_patch(mpatches.Rectangle(
            (x - ROI_HALF, y - ROI_HALF),
            2 * ROI_HALF, 2 * ROI_HALF,
            linewidth=1.6, edgecolor=BRIGHT_C,
            facecolor='none', zorder=5))
        if y - ROI_HALF < 24:
            ly, va = y + ROI_HALF + 4, 'top'
        else:
            ly, va = y - ROI_HALF - 4, 'bottom'
        ax.annotate(f'Pix_{bp}', (x, y), xytext=(x, ly),
                    color=BRIGHT_C, ha='center', va=va,
                    bbox=_bbox(BRIGHT_C), zorder=6,
                    fontfamily=FONT_KW['family'],
                    fontweight=FONT_KW['weight'],
                    fontsize=FONT_KW['size'])

    angle_specs = [
        ('snout',  'neck',     'centroid', 'forebody bend',  145,  35, 'center'),
        ('neck',   'centroid', 'tailbase', 'spine flex',      10, 285, 'left'),
        ('flpaw',  'centroid', 'frpaw',    'forepaw spread',
            size - 10,  90, 'right'),
    ]
    for a, v, c, tag, lx, ly, ha in angle_specs:
        if not all(k in kps for k in (a, v, c)):
            continue
        th1, th2, _ = angle_at_vertex(kps[a], kps[v], kps[c])
        ax.add_patch(mpatches.Arc(
            kps[v], 34, 34, angle=0,
            theta1=min(th1, th2), theta2=max(th1, th2),
            color=ANGLE_C, linewidth=2.0, zorder=6))
        vx, vy = kps[v]
        ax.annotate(f'∠ {tag}\nAng_{a}-{v}-{c}',
                    (vx, vy), xytext=(lx, ly),
                    color=ANGLE_C, ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', color=ANGLE_C,
                                    lw=0.7, alpha=0.85),
                    bbox=_bbox(ANGLE_C), zorder=7,
                    fontfamily=FONT_KW['family'],
                    fontweight=FONT_KW['weight'],
                    fontsize=FONT_KW['size'])

    distance_specs = [
        ('snout', 'tailbase', 'body length',     10, 555, 'left'),
        ('hlpaw', 'hrpaw',    'hindpaw spread',
            size - 10, 555, 'right'),
    ]
    for a, b, tag, lx, ly, ha in distance_specs:
        if a not in kps or b not in kps:
            continue
        ax.annotate('', xy=kps[b], xytext=kps[a],
                    arrowprops=dict(arrowstyle='<->', color=DIST_C,
                                    lw=1.6, shrinkA=8, shrinkB=8,
                                    alpha=0.95),
                    zorder=4)
        mx = (kps[a][0] + kps[b][0]) / 2
        my = (kps[a][1] + kps[b][1]) / 2
        ax.annotate(f'↔ {tag}\nDis_{a}-{b}',
                    (mx, my), xytext=(lx, ly),
                    color=DIST_C, ha=ha, va='center',
                    arrowprops=dict(arrowstyle='-', color=DIST_C,
                                    lw=0.7, alpha=0.85),
                    bbox=_bbox(DIST_C), zorder=7,
                    fontfamily=FONT_KW['family'],
                    fontweight=FONT_KW['weight'],
                    fontsize=FONT_KW['size'])

    # Velocity overlay — comet trail (solid past, dashed future) when a
    # vel_trail is provided; falls back to single dashed arrow otherwise.
    # Hind paws only (lift/swing is where the trail reads most clearly).
    velocity_specs = [
        ('hlpaw',  -90,   30, 'right'),
        ('hrpaw',   90,  -30, 'left'),
    ]
    if vel_trail is not None:
        cur_idx = vel_step
        for bp, ldx, ldy, ha in velocity_specs:
            local = []
            for kf in vel_trail:
                if isinstance(kf, dict) and bp in kf:
                    xx, yy = kf[bp]
                    local.append((xx - x0, yy - y0))
                else:
                    local.append(None)
            valid = [p for p in local if p is not None]
            if len(valid) < 3:
                continue
            past   = [p for p in local[: cur_idx + 1] if p is not None]
            future = [p for p in local[cur_idx:]     if p is not None]
            if past:
                ax.plot([p[0] for p in past], [p[1] for p in past],
                        color=VEL_C, linewidth=1.8, alpha=0.85, zorder=4)
            if future:
                ax.plot([p[0] for p in future], [p[1] for p in future],
                        color=VEL_C, linewidth=1.8, alpha=0.95,
                        linestyle='--', zorder=4)
            markers = [
                (0,                50, 0.40, 0),
                (cur_idx,         110, 0.95, 0.6),
                (len(local) - 1,   70, 0.55, 0),
            ]
            for idx, sz, alpha, edge in markers:
                if 0 <= idx < len(local) and local[idx] is not None:
                    mx_, my_ = local[idx]
                    ax.scatter([mx_], [my_], s=sz, color=VEL_C,
                               alpha=alpha,
                               edgecolor='black' if edge else 'none',
                               linewidth=edge,
                               zorder=5)
            cur = local[cur_idx] or valid[len(valid) // 2]
            mx_, my_ = cur
            ax.annotate(
                f'Vel{vel_step}_{bp}\n(t-{vel_step} … t+{vel_step})',
                (mx_, my_), xytext=(mx_ + ldx, my_ + ldy),
                color=VEL_C, ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color=VEL_C,
                                lw=0.7, alpha=0.85),
                bbox=_bbox(VEL_C), zorder=7,
                fontfamily=FONT_KW['family'],
                fontweight=FONT_KW['weight'],
                fontsize=FONT_KW['size'])
    else:
        for bp, ldx, ldy, ha in velocity_specs:
            if bp not in kps or bp not in fut:
                continue
            x_a, y_a = kps[bp]; x_b, y_b = fut[bp]
            if ((x_b - x_a)**2 + (y_b - y_a)**2) ** 0.5 < 4:
                continue
            ax.scatter([x_b], [y_b], s=120, color=VEL_C, alpha=0.30,
                       edgecolor='black', linewidth=0.5, zorder=4)
            ax.scatter([x_b], [y_b], s=24,  color=VEL_C, alpha=0.95, zorder=5)
            ax.annotate('', xy=(x_b, y_b), xytext=(x_a, y_a),
                        arrowprops=dict(arrowstyle='->', color=VEL_C,
                                        lw=1.8, linestyle='--',
                                        shrinkA=5, shrinkB=5, alpha=0.95),
                        zorder=6)
            mx = (x_a + x_b) / 2; my = (y_a + y_b) / 2
            ax.annotate(f'⇢ Vel{vel_step}_{bp}\n(t → t+{vel_step})',
                        (mx, my), xytext=(mx + ldx, my + ldy),
                        color=VEL_C, ha=ha, va='center',
                        arrowprops=dict(arrowstyle='-', color=VEL_C,
                                        lw=0.7, alpha=0.85),
                        bbox=_bbox(VEL_C), zorder=7,
                        fontfamily=FONT_KW['family'],
                        fontweight=FONT_KW['weight'],
                        fontsize=FONT_KW['size'])

    schematic_png = out_stem.with_name(out_stem.name + "_schematic.png")
    fig.savefig(schematic_png, dpi=100, facecolor='white')
    plt.close(fig)

    # Top panel: same crop, raw grayscale, on a white-bg figure
    raw_png = out_stem.with_name(out_stem.name + "_raw.png")
    gray_crop = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    fig2 = plt.figure(figsize=(size / 100, size / 100), dpi=100,
                      facecolor='white')
    ax2 = fig2.add_axes([0, 0, 1, 1])
    ax2.imshow(gray_crop, cmap='gray', vmin=0, vmax=255)
    ax2.set_xticks([]); ax2.set_yticks([])
    for s in ax2.spines.values():
        s.set_visible(False)
    fig2.savefig(raw_png, dpi=100, facecolor='white')
    plt.close(fig2)

    return raw_png, schematic_png


def stack_portrait(raw_png: Path, schematic_png: Path,
                   out_png: Path, out_pdf: Path, out_svg: Path):
    """Vertical stack: raw on top, schematic on bottom, both already
    same width on a white background. Produces PNG via cv2 and a
    matplotlib PDF/SVG with editable text by re-loading the PNGs."""
    raw = cv2.imread(str(raw_png))
    sch = cv2.imread(str(schematic_png))
    assert raw.shape[1] == sch.shape[1], "panels must be same width"
    sep = np.full((4, raw.shape[1], 3), 255, dtype=np.uint8)
    combo = np.vstack([raw, sep, sch])
    cv2.imwrite(str(out_png), combo)

    # Also re-render as a matplotlib figure (so PDF/SVG are vector
    # text-editable even though the panels are raster).
    h_in = combo.shape[0] / 100
    w_in = combo.shape[1] / 100
    fig = plt.figure(figsize=(w_in, h_in), dpi=100, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(cv2.cvtColor(combo, cv2.COLOR_BGR2RGB),
              interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    fig.savefig(out_pdf, format='pdf', facecolor='white',
                bbox_inches='tight', pad_inches=0)
    fig.savefig(out_svg, format='svg', facecolor='white',
                bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',
                    default=r'E:\RSVIDS\Blackbox\2604_DV_DSS')
    ap.add_argument('--session',
                    default='202604_Ai32_61_BL_cropped')
    ap.add_argument('--frame', type=int, default=24284)
    ap.add_argument('--out_dir',
                    default=str(Path(__file__).parent / 'analysis_output'
                                / 'feature_schematic_white_portrait'))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    step("resolving inputs")
    video_path = find_video(args.project, args.session)
    dlc_path   = find_dlc_h5(args.project, args.session)

    step("loading frame + keypoints")
    bgr_frame = grab_frame_bgr(video_path, args.frame)
    x_df, y_df = load_dlc_xy(dlc_path)
    kps = keypoints_at(x_df, y_df, args.frame)
    VEL_STEP = 10
    fut_idx = min(args.frame + VEL_STEP, len(x_df) - 1)
    kps_future = keypoints_at(x_df, y_df, fut_idx)
    vel_trail = []
    for offset in range(-VEL_STEP, VEL_STEP + 1):
        f = args.frame + offset
        if 0 <= f < len(x_df):
            vel_trail.append(keypoints_at(x_df, y_df, f))
        else:
            vel_trail.append({})

    step("rendering white-bg panels")
    stem = out_dir / f"feature_schematic_white_portrait_f{args.frame}"
    raw_png, sch_png = render_white_schematic(
        bgr_frame, kps, args.frame, kps_future, stem, size=600,
        vel_trail=vel_trail, vel_step=VEL_STEP)

    step("stacking portrait")
    combo_png = stem.with_suffix(".png")
    combo_pdf = stem.with_suffix(".pdf")
    combo_svg = stem.with_suffix(".svg")
    stack_portrait(raw_png, sch_png, combo_png, combo_pdf, combo_svg)
    step(f"  -> {combo_png}")

    step("uploading")
    discord_upload(
        "**Feature schematic — white background, portrait stack**\n"
        "Top: raw cropped frame. Bottom: white-bg schematic with magenta-"
        "tinted mouse silhouette, dark skeleton, bright keypoints, and "
        "the same Pix_/Ang_/Dis_/Vel_ feature annotations. PNG + PDF + SVG.",
        [combo_png, combo_pdf, combo_svg],
    )


if __name__ == '__main__':
    main()
