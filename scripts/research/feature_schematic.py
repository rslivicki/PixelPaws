"""
feature_schematic.py — annotated schematic of PixelPaws features over a
single mouse video frame. Produces two PNGs:

  pose_schematic.png                  keypoints, skeleton, example
                                       angles, example distances, height
                                       reference
  brightness_silhouette_schematic.png brightness ROI squares per
                                       keypoint, whole-frame silhouette
                                       outline, largest-blob bbox

Defaults to a known-good frame on the 2604_DV_DSS Ai32_61 BL session
(belly-pressed lying, all 9 DLC keypoints clearly placed).

Usage:
    py feature_schematic.py
        [--project E:\\RSVIDS\\Blackbox\\2604_DV_DSS]
        [--session 202604_Ai32_61_BL_cropped]
        [--frame   52282]
        [--out     <auto-timestamped under analysis_output>]
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)                                # peer scripts in this folder
sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)
from pose_features import PoseFeatureExtractor


# ─────────────────────────────────────────────────────────────────────
# Config (mirrors render_skeleton_video.py and brightness_features.py)
# ─────────────────────────────────────────────────────────────────────

BODYPARTS = ['snout', 'neck', 'centroid', 'tailbase', 'tailtip',
             'frpaw', 'flpaw', 'hrpaw', 'hlpaw']

SKELETON_EDGES = [
    ('snout', 'neck'),
    ('neck', 'centroid'),
    ('centroid', 'tailbase'),
    ('tailbase', 'tailtip'),
    ('neck', 'frpaw'),
    ('neck', 'flpaw'),
    ('centroid', 'hrpaw'),
    ('centroid', 'hlpaw'),
]

# RGB tuples copied from render_skeleton_video.py (lines 38–49); /255 for matplotlib.
KP_COLORS_BGR_OR_RGB = {
    'hrpaw':    (220, 210, 0),
    'hlpaw':    (200, 0, 200),
    'frpaw':    (0, 155, 255),
    'flpaw':    (160, 210, 0),
    'snout':    (0, 220, 220),
    'neck':     (180, 180, 180),
    'centroid': (120, 120, 120),
    'tailbase': (70, 120, 240),
    'tailtip':  (90, 150, 255),
}
KP_COLORS = {bp: tuple(c / 255 for c in rgb)
              for bp, rgb in KP_COLORS_BGR_OR_RGB.items()}

EDGE_COLOR = tuple(c / 255 for c in (100, 110, 100))   # dim green-grey

BRIGHTNESS_BPS = ['centroid', 'tailbase', 'snout', 'hlpaw', 'hrpaw']
PAW_BRIGHTNESS_BPS = ['hlpaw', 'hrpaw', 'snout']  # classic feature-set bodyparts
ROI_HALF = 10                # square_size 20 → ±10 px
SILHOUETTE_FLOOR = 35


# ─────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────

def find_video(project: str, session: str) -> str:
    for ext in ('.mp4', '.avi', '.mov', '.mkv'):
        p = os.path.join(project, 'videos', session + ext)
        if os.path.isfile(p):
            return p
    sys.exit(f"No video '{session}.<mp4|avi|mov|mkv>' in {project}/videos")


def find_dlc_h5(project: str, session: str) -> str:
    cands = sorted(glob.glob(os.path.join(
        project, 'videos', f'{session}DLC*.h5')))
    raw = [c for c in cands if '_filtered' not in c]
    chosen = (raw or cands)
    if not chosen:
        sys.exit(f"No DLC .h5 for '{session}' in {project}/videos")
    return chosen[0]


def find_cache_pkl(project: str, session: str) -> str:
    cands = sorted(glob.glob(os.path.join(
        project, 'features', f'{session}_features_*.pkl')),
        key=os.path.getmtime, reverse=True)
    if not cands:
        sys.exit(f"No cache pkl for '{session}' in {project}/features")
    return cands[0]


def grab_frame_gray(video_path: str, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        sys.exit(f"Could not read frame {frame_idx} from {video_path}")
    return cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)


def grab_frame_bgr(video_path: str, frame_idx: int) -> np.ndarray:
    """Color BGR frame; needed by render_simple_red_schematic for the
    paw-pixel stamps (greyscale loses the colour info)."""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ok, fr = cap.read()
    cap.release()
    if not ok or fr is None:
        sys.exit(f"Could not read frame {frame_idx} from {video_path}")
    return fr


def load_dlc_xy(dlc_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (x_df, y_df) covering all bodyparts. Single DLC read; the
    columns may not be named exactly {bp} — handle the common case where
    PoseFeatureExtractor returns columns like 'centroid' OR full DLC
    column names. We reduce both layouts to canonical bp-named columns."""
    ext = PoseFeatureExtractor(BODYPARTS)
    df = ext.load_dlc_data(dlc_path)
    x_df, y_df, _ = ext.get_bodypart_coords(df)

    def normalize(d: pd.DataFrame) -> pd.DataFrame:
        # If columns are already exactly the bodyparts, pass through.
        if set(d.columns) >= set(BODYPARTS):
            return d[BODYPARTS]
        # Otherwise map each bodypart to the column whose name contains it.
        rename: dict[str, str] = {}
        for bp in BODYPARTS:
            for c in d.columns:
                cl = c.lower()
                if bp.lower() in cl and (cl.endswith('_x') or cl.endswith('_y')
                                            or cl.endswith(bp.lower())):
                    rename[bp] = c
                    break
        return pd.DataFrame({bp: d[col] for bp, col in rename.items()})

    return normalize(x_df), normalize(y_df)


def keypoints_at(x_df: pd.DataFrame, y_df: pd.DataFrame,
                  frame_idx: int) -> dict[str, tuple[float, float]]:
    out = {}
    for bp in BODYPARTS:
        if bp in x_df.columns and bp in y_df.columns:
            x = x_df[bp].iloc[frame_idx]
            y = y_df[bp].iloc[frame_idx]
            if pd.notna(x) and pd.notna(y):
                out[bp] = (float(x), float(y))
    return out


def rolling_floor_y(y_df: pd.DataFrame, bp: str,
                     window: int = 500) -> pd.Series:
    return y_df[bp].rolling(window, min_periods=1).max()


# ─────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────

def angle_at_vertex(a: tuple[float, float], v: tuple[float, float],
                     c: tuple[float, float]) -> tuple[float, float, float]:
    """Return (theta1_deg, theta2_deg, interior_deg) for an arc at vertex v
    sweeping from edge v→a to edge v→c. matplotlib's Arc expects degrees,
    counter-clockwise from +x.
    """
    th_a = np.degrees(np.arctan2(a[1] - v[1], a[0] - v[0]))
    th_c = np.degrees(np.arctan2(c[1] - v[1], c[0] - v[0]))
    # Interior angle (smaller of the two sweeps)
    diff = (th_c - th_a) % 360
    if diff > 180:
        diff = 360 - diff
        th1, th2 = th_c, th_a
    else:
        th1, th2 = th_a, th_c
    # Clamp against floating-point drift; interior angles can never
    # legitimately be < 0 or > 180.
    diff = max(0.0, min(180.0, float(diff)))
    return th1, th2, diff


def euclid(a: tuple[float, float], b: tuple[float, float]) -> float:
    return float(np.hypot(b[0] - a[0], b[1] - a[1]))


# ─────────────────────────────────────────────────────────────────────
# Background helpers — cartoon vs photo
# ─────────────────────────────────────────────────────────────────────

def draw_background(ax, frame: np.ndarray, mode: str) -> None:
    """Either: 'photo' (greyed video frame) OR 'cartoon' (silhouette mask
    as a light grey fill on white). Both modes use the same coordinate
    system so the same overlay layer code lands correctly."""
    h, w = frame.shape[:2]
    if mode == 'photo':
        ax.imshow(frame, cmap='gray', vmin=0, vmax=255, alpha=0.55)
    else:  # cartoon
        # White background; silhouette body in light grey.
        sil_mask = (frame > SILHOUETTE_FLOOR)
        canvas = np.full((h, w), 255, dtype=np.uint8)   # white
        canvas[sil_mask] = 215                            # light grey body
        ax.imshow(canvas, cmap='gray', vmin=0, vmax=255)
    # Outline the actual frame in both modes
    ax.add_patch(mpatches.Rectangle((0, 0), w, h, fill=False,
                                      edgecolor='black', linewidth=0.8,
                                      zorder=1))


def draw_skeleton(ax, kps: dict, label_keypoints: bool = True) -> None:
    """Skeleton edges + keypoint dots + (optional) bp labels."""
    for a, b in SKELETON_EDGES:
        if a in kps and b in kps:
            ax.plot([kps[a][0], kps[b][0]],
                     [kps[a][1], kps[b][1]],
                     color=EDGE_COLOR, linewidth=1.6, zorder=2)
    for bp, (x, y) in kps.items():
        ax.scatter([x], [y], s=110, color=KP_COLORS[bp],
                    edgecolor='black', linewidth=0.6, zorder=4)
        if label_keypoints:
            ax.annotate(bp, (x, y),
                         xytext=(x + 9, y - 9), fontsize=8, color='white',
                         bbox=dict(boxstyle='round,pad=0.18',
                                     facecolor='black', alpha=0.55,
                                     edgecolor='none'),
                         zorder=5)


# ─────────────────────────────────────────────────────────────────────
# Figure 1 — pose / geometric features
# ─────────────────────────────────────────────────────────────────────

def render_pose_schematic(frame: np.ndarray, kps: dict, y_df: pd.DataFrame,
                            cache_row: pd.Series, frame_idx: int,
                            out_path: str):
    h, w = frame.shape[:2]
    fig, ax = plt.subplots(figsize=(11, 8.5), constrained_layout=True)
    ax.imshow(frame, cmap='gray', vmin=0, vmax=255, alpha=0.55)

    # Skeleton edges
    for a, b in SKELETON_EDGES:
        if a in kps and b in kps:
            ax.plot([kps[a][0], kps[b][0]],
                     [kps[a][1], kps[b][1]],
                     color=EDGE_COLOR, linewidth=1.6, zorder=2)

    # Keypoints + labels
    for bp, (x, y) in kps.items():
        ax.scatter([x], [y], s=110, color=KP_COLORS[bp],
                    edgecolor='black', linewidth=0.6, zorder=4)
        ax.annotate(bp, (x, y),
                     xytext=(x + 9, y - 9), fontsize=8, color='white',
                     bbox=dict(boxstyle='round,pad=0.18',
                                 facecolor='black', alpha=0.55,
                                 edgecolor='none'),
                     zorder=5)

    # Example angles (3) — arc at vertex; label parked at an explicit
    # absolute (x, y) location well off the body so labels don't collide.
    # Frame is 760×720 in the 2604 project.
    angle_examples = [
        ('snout',   'neck',     'centroid', 'forebody bend',
            -190,   60, 'left'),     # upper-left outside frame
        ('neck',    'centroid', 'tailbase', 'spine flex',
            w + 190,   60, 'right'), # upper-right outside frame
        ('flpaw',   'centroid', 'frpaw',    'forepaw spread',
            w + 190,  450, 'right'), # mid-right outside frame
    ]
    for a, v, c, label, lx, ly, halign in angle_examples:
        if not all(k in kps for k in (a, v, c)):
            continue
        th1, th2, deg = angle_at_vertex(kps[a], kps[v], kps[c])
        arc = mpatches.Arc(kps[v], 60, 60, angle=0,
                            theta1=min(th1, th2), theta2=max(th1, th2),
                            color='crimson', linewidth=2.0, zorder=6)
        ax.add_patch(arc)
        ax.annotate(f'Ang_{a}-{v}-{c}\n({deg:.0f}°)  {label}',
                     (kps[v][0], kps[v][1]),
                     xytext=(lx, ly), fontsize=7,
                     color='crimson', ha=halign, va='center',
                     arrowprops=dict(arrowstyle='-', color='crimson',
                                      lw=0.7, alpha=0.7),
                     bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='white', alpha=0.92,
                                edgecolor='crimson'),
                     zorder=7)

    # Example distances (3) — double-headed arrow on the body, label at an
    # explicit margin location with a leader line back to the midpoint.
    distance_examples = [
        ('snout',    'tailbase', 'body length',
            -190,  280, 'left'),         # mid-left outside frame
        ('hlpaw',    'hrpaw',    'hind paw spread',
            -190,  580, 'left'),         # lower-left outside frame
        ('flpaw',    'frpaw',    'fore paw spread',
            w + 190,  600, 'right'),     # lower-right outside frame
    ]
    for a, b, label, lx, ly, halign in distance_examples:
        if a not in kps or b not in kps:
            continue
        d = euclid(kps[a], kps[b])
        ax.annotate('', xy=kps[b], xytext=kps[a],
                     arrowprops=dict(arrowstyle='<->', color='navy',
                                       lw=1.6, shrinkA=8, shrinkB=8),
                     zorder=3)
        mx, my = (kps[a][0] + kps[b][0]) / 2, (kps[a][1] + kps[b][1]) / 2
        ax.annotate(f'Dis_{a}-{b}\n({d:.0f} px)  {label}',
                     (mx, my), xytext=(lx, ly),
                     fontsize=7, color='navy', ha=halign, va='center',
                     arrowprops=dict(arrowstyle='-', color='navy',
                                      lw=0.7, alpha=0.7),
                     bbox=dict(boxstyle='round,pad=0.25',
                                facecolor='white', alpha=0.92,
                                edgecolor='navy'),
                     zorder=7)

    # Height reference for centroid
    floor_series = rolling_floor_y(y_df, 'centroid', window=500)
    floor_y = float(floor_series.iloc[frame_idx])
    cx, cy = kps['centroid']
    height_val = max(0.0, floor_y - cy)
    ax.axhline(floor_y, xmin=0, xmax=1,
                color='orange', linestyle='--', linewidth=1.2,
                alpha=0.85, zorder=2)
    ax.annotate('', xy=(cx, cy), xytext=(cx, floor_y),
                 arrowprops=dict(arrowstyle='<->', color='orange',
                                   lw=1.6),
                 zorder=3)
    ax.annotate(f'centroid_Height = {height_val:.0f} px\n'
                  '(distance from rolling-max-y "floor", 500-fr window)',
                 (cx, (cy + floor_y) / 2),
                 xytext=(w / 2, h + 60),
                 fontsize=7, color='darkorange', ha='center', va='top',
                 arrowprops=dict(arrowstyle='-', color='darkorange',
                                   lw=0.7, alpha=0.7),
                 bbox=dict(boxstyle='round,pad=0.25',
                            facecolor='white', alpha=0.92,
                            edgecolor='darkorange'),
                 zorder=7)

    # Extended axes give the margin labels room to sit outside the frame.
    ax.set_xlim(-260, w + 260); ax.set_ylim(h + 110, -40)
    ax.set_xticks([]); ax.set_yticks([])
    # Outline the actual frame so it's clear what's "image" vs "margin".
    ax.add_patch(mpatches.Rectangle((0, 0), w, h, fill=False,
                                      edgecolor='black', linewidth=0.8,
                                      zorder=1))
    ax.set_title(f'Pose features schematic — {frame_idx}', fontsize=12)
    fig.text(0.5, 0.02,
              "9 keypoints, 8 skeleton edges, 36 pairwise distances, "
              "576 angles (every keypoint pair as endpoints × every "
              "keypoint as vertex), 9 heights, plus per-keypoint "
              "velocities (Vel1/2/10), kinematics (Jerk, OnsetSharpness, "
              "PreQuiescence), derivatives. 3 angles + 3 distances + 1 "
              "height reference shown as examples.",
              ha='center', va='bottom', fontsize=8.5,
              wrap=True, color='#222')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 2 — brightness + silhouette features
# ─────────────────────────────────────────────────────────────────────

def render_paw_brightness_schematic(frame: np.ndarray, kps: dict,
                                       cache_row: pd.Series, frame_idx: int,
                                       out_path: str):
    """Two panels (cartoon left, photo right) focused on the classic
    paw-brightness feature-set bodyparts: hlpaw, hrpaw, snout. Each
    panel shows the ROI squares; the photo panel shows Pix_<bp> values
    + a zoomed-in inset of one ROI."""
    h, w = frame.shape[:2]
    fig, axes = plt.subplots(1, 2, figsize=(18, 8),
                              constrained_layout=True,
                              gridspec_kw={'width_ratios': [1, 1.15]})

    for ax, mode in zip(axes, ['cartoon', 'photo']):
        draw_background(ax, frame, mode)
        # Skeleton + keypoints (no labels on cartoon for cleanliness)
        draw_skeleton(ax, kps, label_keypoints=(mode == 'photo'))

        # Paw brightness ROI squares — emphasized with thicker borders
        for bp in PAW_BRIGHTNESS_BPS:
            if bp not in kps:
                continue
            x, y = kps[bp]
            x0 = x - ROI_HALF; y0 = y - ROI_HALF
            ax.add_patch(mpatches.Rectangle(
                (x0, y0), 2 * ROI_HALF, 2 * ROI_HALF,
                linewidth=2.4, edgecolor=KP_COLORS[bp],
                facecolor='none', zorder=5))
            if mode == 'photo':
                pix_val = cache_row.get(f'Pix_{bp}', np.nan)
                lab = f'Pix_{bp}\n= {pix_val:.1f}'
                ax.annotate(lab, (x, y0 + 2 * ROI_HALF),
                             xytext=(x, y0 + 2 * ROI_HALF + 8),
                             fontsize=9, color='black', ha='center',
                             fontweight='bold',
                             bbox=dict(boxstyle='round,pad=0.3',
                                        facecolor=KP_COLORS[bp], alpha=0.7,
                                        edgecolor='black'),
                             zorder=6)

        ax.set_xlim(-30, w + 30)
        ax.set_ylim(h + 30, -30)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title('Cartoon (silhouette body, skeleton, ROI squares)'
                       if mode == 'cartoon'
                       else f'Photo (frame {frame_idx}, Pix_<bp> values)',
                      fontsize=11)

    # Inset on the photo panel: zoomed-in view of the hlpaw ROI to show
    # what "mean of the 20×20 ROI" actually averages over.
    if 'hlpaw' in kps:
        x, y = kps['hlpaw']
        x0 = max(0, int(x) - ROI_HALF)
        y0 = max(0, int(y) - ROI_HALF)
        x1 = min(w, int(x) + ROI_HALF)
        y1 = min(h, int(y) + ROI_HALF)
        roi = frame[y0:y1, x0:x1]
        # Place the inset axes inside the photo panel's lower-left corner.
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        inset = inset_axes(axes[1], width="22%", height="22%",
                            loc='lower left', borderpad=1.0)
        inset.imshow(roi, cmap='gray', vmin=0, vmax=255,
                      interpolation='nearest')
        for s in inset.spines.values():
            s.set(color=KP_COLORS['hlpaw'], linewidth=2.0)
        inset.set_xticks([]); inset.set_yticks([])
        inset.set_title(f'hlpaw ROI (20×20 px)\nmean = {roi.mean():.1f}',
                         fontsize=8, color='black')

    fig.suptitle(f'Paw brightness features — frame {frame_idx}', fontsize=13)
    fig.text(0.5, 0.005,
              "Paw brightness (3 ROIs in the classic feature set: hlpaw, "
              "hrpaw, snout). For each: Pix_<bp> = mean of the 20×20 px "
              "ROI centered on the DLC keypoint. Higher value = paw "
              "pressed harder against the lit floor. ΔBrt = Pix_<bp> − "
              "session_median (animal-invariant). Per-frame derivatives "
              "(|d/dt|, BrightAccel, BrightOnsetPeak, BrightAsymmetry, "
              "SurfaceZ) are computed downstream and feed the classifier.",
              ha='center', va='bottom', fontsize=9, wrap=True, color='#222')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def render_powerpoint_summary(frame: np.ndarray, kps: dict,
                                cache_row: pd.Series, frame_idx: int,
                                out_path: str):
    """Slide-deck cartoon, render_skeleton_video.py style: black canvas,
    bright skeleton + keypoints, paw ROI boxes, one example each of
    angle / distance / silhouette outline. Right-margin text lists
    feature-set totals."""
    h, w = frame.shape[:2]
    fig = plt.figure(figsize=(15, 8.5), facecolor='black')
    ax = fig.add_axes([0.02, 0.04, 0.62, 0.92])
    ax.set_facecolor('black')

    # Skeleton edges (white-ish dim line)
    for a, b in SKELETON_EDGES:
        if a in kps and b in kps:
            ax.plot([kps[a][0], kps[b][0]],
                     [kps[a][1], kps[b][1]],
                     color=EDGE_COLOR, linewidth=2.0, zorder=2)

    # Bright keypoint dots + white-on-dark labels
    for bp, (x, y) in kps.items():
        ax.scatter([x], [y], s=160, color=KP_COLORS[bp],
                    edgecolor='white', linewidth=0.8, zorder=4)
        ax.annotate(bp, (x, y), xytext=(x + 11, y - 11),
                     fontsize=9, color='white', fontweight='bold',
                     bbox=dict(boxstyle='round,pad=0.18',
                                 facecolor='black', alpha=0.55,
                                 edgecolor='none'),
                     zorder=5)

    # Silhouette outer boundary (largest blob, smoothed) — render in dim
    # red so it suggests the body shape without dominating.
    sil_mask = (frame > SILHOUETTE_FLOOR).astype(np.uint8)
    n_lbl, lbl_img, stats, _ = cv2.connectedComponentsWithStats(
        sil_mask, connectivity=8)
    if n_lbl > 1:
        biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        blob_only = (lbl_img == biggest).astype(np.uint8)
        kernel = np.ones((5, 5), np.uint8)
        blob_only = cv2.morphologyEx(blob_only, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(
            blob_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            cnt = cnt.squeeze(1)
            ax.plot(cnt[:, 0], cnt[:, 1], color='#ff5050', linewidth=1.4,
                     alpha=0.85, zorder=3)

    # Example angle — spine flex
    if all(k in kps for k in ('neck', 'centroid', 'tailbase')):
        th1, th2, deg = angle_at_vertex(kps['neck'], kps['centroid'],
                                          kps['tailbase'])
        arc = mpatches.Arc(kps['centroid'], 90, 90, angle=0,
                            theta1=min(th1, th2), theta2=max(th1, th2),
                            color='#ff5577', linewidth=3.0, zorder=6)
        ax.add_patch(arc)

    # Example distance — body length
    if 'snout' in kps and 'tailbase' in kps:
        ax.annotate('', xy=kps['tailbase'], xytext=kps['snout'],
                     arrowprops=dict(arrowstyle='<->',
                                       color='#66ccff', lw=2.4,
                                       shrinkA=12, shrinkB=12),
                     zorder=3)

    # Paw brightness ROI squares
    for bp in PAW_BRIGHTNESS_BPS:
        if bp not in kps:
            continue
        x, y = kps[bp]
        ax.add_patch(mpatches.Rectangle(
            (x - ROI_HALF, y - ROI_HALF),
            2 * ROI_HALF, 2 * ROI_HALF,
            linewidth=2.6, edgecolor=KP_COLORS[bp],
            facecolor='none', zorder=5))

    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Right-side margin — three legend boxes, one per feature category,
    # color-keyed to the on-canvas overlay.
    legend_ax = fig.add_axes([0.66, 0.04, 0.32, 0.92])
    legend_ax.set_facecolor('black')
    legend_ax.set_xticks([]); legend_ax.set_yticks([])
    for s in legend_ax.spines.values():
        s.set_visible(False)
    legend_ax.set_xlim(0, 1); legend_ax.set_ylim(0, 1)

    # (Title now lives only in the figure-level fig.text below — no
    # duplicated header in the legend column.)

    rows = [
        ('#cccccc', '9 keypoints + 8 skeleton edges',
            'tracked positions, the raw structural backbone'),
        ('#66ccff', '36 pairwise distances',
            'body length, paw spread, etc. (e.g. snout↔tailbase)'),
        ('#ff5577', '576 angles',
            'every keypoint pair as endpoints × every keypoint as vertex'),
        ('#ffaa55', '9 height features',
            'per-bodypart distance from session-rolling y "floor"'),
        ('#88ee88', 'paw brightness  Pix_<bp>',
            f'{len(PAW_BRIGHTNESS_BPS)} ROIs ({", ".join(PAW_BRIGHTNESS_BPS)});\n'
            'mean of 20×20 px under each keypoint'),
        ('#bb88ff', 'velocity / kinematics',
            'Vel1 / Vel2 / Vel10, Jerk, OnsetSharpness'),
        ('#ff5050', 'whole-frame silhouette',
            'silhouette_frac, blob_frac, aspect ratio'),
    ]
    y = 0.92
    for color, head, body in rows:
        legend_ax.add_patch(mpatches.Rectangle(
            (0.04, y - 0.013), 0.05, 0.022,
            facecolor=color, edgecolor='none',
            transform=legend_ax.transAxes))
        legend_ax.text(0.12, y, head, color='white',
                        fontsize=11, fontweight='bold',
                        ha='left', va='center')
        legend_ax.text(0.12, y - 0.035, body, color='#cccccc',
                        fontsize=9, ha='left', va='top')
        y -= 0.11

    legend_ax.text(0.5, 0.04,
                    '~700 columns / frame  →  XGBoost classifier',
                    color='#ffaa55', fontsize=11, ha='center',
                    fontweight='bold')

    fig.text(0.5, 0.985,
              'PixelPaws — per-frame feature set',
              ha='center', va='top',
              fontsize=16, fontweight='bold', color='white')
    fig.savefig(out_path, dpi=150, facecolor='black')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 4 — simple 600×600 "skeleton_red" schematic
# ─────────────────────────────────────────────────────────────────────

def render_simple_red_schematic(bgr_full: np.ndarray, kps_full: dict,
                                  cache_row: pd.Series, frame_idx: int,
                                  out_path: str, size: int = 600,
                                  kps_future: dict | None = None,
                                  vel_step: int = 10,
                                  vel_trail: list | None = None):
    """Single-panel 600×600 schematic mirroring the skeleton_red.mp4
    aesthetic — black background, magenta paw-pixel stamps with glow,
    dim skeleton lines, bright keypoint dots — overlaid with the three
    paw-brightness ROI squares (Pix_<bp>), three example angle arcs
    (forebody bend / spine flex / forepaw spread), two example
    distance arrows (body length / hindpaw spread), and — when
    `kps_future` is provided — two example velocity arrows showing
    displacement from `frame_idx` → `frame_idx + vel_step`. Saved at
    exactly size×size px.
    """
    H, W = bgr_full.shape[:2]
    half = size // 2

    # Crop centered on centroid, clamped to frame bounds
    cx0, cy0 = kps_full['centroid']
    x0 = int(max(0, min(W - size, round(cx0) - half)))
    y0 = int(max(0, min(H - size, round(cy0) - half)))
    bgr = bgr_full[y0:y0 + size, x0:x0 + size].copy()

    # Local keypoint coords (subtract crop offset)
    kps = {bp: (x - x0, y - y0) for bp, (x, y) in kps_full.items()}

    # ── Skeleton render (skeleton_red.mp4 style) ────────────────────
    canvas = np.zeros((size, size, 3), dtype=np.float32)

    paw_color_bgr = (200, 0, 200)   # magenta — matches skeleton_red.mp4
    paw_halfs     = {'hrpaw': 40, 'hlpaw': 40, 'frpaw': 15, 'flpaw': 15}
    paw_threshold = 0.40

    for bp, half_p in paw_halfs.items():
        if bp not in kps:
            continue
        x, y = int(round(kps[bp][0])), int(round(kps[bp][1]))
        x1 = max(0, x - half_p); x2 = min(size, x + half_p)
        y1 = max(0, y - half_p); y2 = min(size, y + half_p)
        if x2 <= x1 or y2 <= y1:
            continue
        roi = bgr[y1:y2, x1:x2].astype(np.float32) / 255.0
        mask = roi.max(axis=2) >= paw_threshold
        col_f = np.array(paw_color_bgr, dtype=np.float32) / 255.0
        canvas[y1:y2, x1:x2] += roi * mask[:, :, None] * col_f * 255.0

    # Glow blur (matches render_skeleton_video.py default --glow 0.2)
    glow = cv2.GaussianBlur(canvas, (7, 7), 2.0)
    canvas = np.clip(canvas + glow * 0.2, 0, 255).astype(np.uint8)

    # Skeleton lines (dim grey-green, identical to render_skeleton_video.py)
    SKEL_COLOR_BGR = (100, 110, 100)
    for a, b in SKELETON_EDGES:
        if a in kps and b in kps:
            xa, ya = int(round(kps[a][0])), int(round(kps[a][1]))
            xb, yb = int(round(kps[b][0])), int(round(kps[b][1]))
            cv2.line(canvas, (xa, ya), (xb, yb),
                      SKEL_COLOR_BGR, 1, cv2.LINE_AA)

    # ALL nine DLC keypoints as bright dots (paw dots sit on top of the
    # paw-pixel stamps so the tracked location reads through the magenta
    # mass). Palette mirrors the prior DLC-tracking panel: head/spine in
    # the cyan family, tail in warm orange, paws in magenta with a
    # darker purple-magenta for the hindpaws.
    DLC_BGR = {
        'snout':    (255, 255,  80),
        'neck':     (255, 220, 120),
        'centroid': (220, 200, 140),
        'tailbase': ( 90, 150, 255),
        'tailtip':  (110, 170, 255),
        'frpaw':    (255,  80, 255),
        'flpaw':    (255,  80, 255),
        'hrpaw':    (200,  60, 255),
        'hlpaw':    (200,  60, 255),
    }
    for bp, color in DLC_BGR.items():
        if bp not in kps:
            continue
        x, y = int(round(kps[bp][0])), int(round(kps[bp][1]))
        cv2.circle(canvas, (x, y), 5, color,            -1, cv2.LINE_AA)
        cv2.circle(canvas, (x, y), 5, (255, 255, 255),   1, cv2.LINE_AA)

    # ── Annotations via matplotlib (crisp text + arc) ──────────────
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    fig = plt.figure(figsize=(size / 100, size / 100), dpi=100,
                      facecolor='black')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor('black')
    ax.imshow(rgb, interpolation='nearest')
    ax.set_xlim(0, size); ax.set_ylim(size, 0)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

    # Colour key (consistent across overlays in this panel):
    #   green   = brightness ROIs / Pix_<bp>
    #   pink    = angles / Ang_a-v-c
    #   orange  = distances / Dis_a-b
    #   violet  = velocity / Vel<step>_<bp>  (dashed arrow + ghost dot
    #             at the t+vel_step position)
    BRIGHT_C = '#66ff88'
    ANGLE_C  = '#ff5577'
    DIST_C   = '#ff9933'
    VEL_C    = '#bb88ff'

    # All overlay text uses one font / weight so the panel reads
    # consistently — Arial is widely available on Windows; matplotlib
    # falls back to DejaVu Sans if the preferred face is missing.
    FONT_KW = dict(family=['Arial', 'DejaVu Sans'],
                    weight='bold', size=8)

    def _bbox(color: str):
        return dict(boxstyle='round,pad=0.16',
                     facecolor='black', alpha=0.65,
                     edgecolor=color, linewidth=0.5)

    # Brightness ROI squares — 3 paws (snout, hlpaw, hrpaw)
    for bp in PAW_BRIGHTNESS_BPS:
        if bp not in kps:
            continue
        x, y = kps[bp]
        ax.add_patch(mpatches.Rectangle(
            (x - ROI_HALF, y - ROI_HALF),
            2 * ROI_HALF, 2 * ROI_HALF,
            linewidth=1.6, edgecolor=BRIGHT_C,
            facecolor='none', zorder=5))
        # Symbol-only label — no numeric values (the cache column name
        # is the audience-facing identifier)
        label = f'Pix_{bp}'
        if y - ROI_HALF < 24:
            ly, va = y + ROI_HALF + 4, 'top'
        else:
            ly, va = y - ROI_HALF - 4, 'bottom'
        ax.annotate(label, (x, y), xytext=(x, ly),
                     color=BRIGHT_C, ha='center', va=va,
                     bbox=_bbox(BRIGHT_C), zorder=6,
                     fontfamily=FONT_KW['family'],
                     fontweight=FONT_KW['weight'],
                     fontsize=FONT_KW['size'])

    # Label slots fan around the canvas so leader lines stay short and
    # labels never stack on the centroid. Positions are absolute (size×
    # size canvas, default 600×600) and tuned visually for the default
    # demo frame; re-tune `lx, ly` per spec if rendering a different
    # pose where the body sits elsewhere in the crop.
    #
    # Edge-anchored label slots fan around the canvas so leader lines
    # stay short and labels never stack on the centroid. Positions and
    # alignment are tuned for the default demo frame; re-tune `lx,ly,ha`
    # per spec if rendering a different pose.
    #
    # ── Example angles — three highlighted on the body ──────────────
    angle_specs = [
        # (a, v, c, tag, label_x, label_y, ha)
        ('snout',  'neck',     'centroid', 'forebody bend', 145,  35, 'center'),
        ('neck',   'centroid', 'tailbase', 'spine flex',     10, 285, 'left'),
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
        # Symbol-only label — feature name + descriptive tag, no degrees
        ax.annotate(f'∠ {tag}\nAng_{a}-{v}-{c}',
                     (vx, vy), xytext=(lx, ly),
                     color=ANGLE_C, ha=ha, va='center',
                     arrowprops=dict(arrowstyle='-', color=ANGLE_C,
                                       lw=0.7, alpha=0.85),
                     bbox=_bbox(ANGLE_C), zorder=7,
                     fontfamily=FONT_KW['family'],
                     fontweight=FONT_KW['weight'],
                     fontsize=FONT_KW['size'])

    # ── Example distances — two highlighted on the body ─────────────
    distance_specs = [
        # (a, b, tag, label_x, label_y, ha)
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
        # Symbol-only label — feature name + descriptive tag, no pixel value
        ax.annotate(f'↔ {tag}\nDis_{a}-{b}',
                     (mx, my), xytext=(lx, ly),
                     color=DIST_C, ha=ha, va='center',
                     arrowprops=dict(arrowstyle='-', color=DIST_C,
                                       lw=0.7, alpha=0.85),
                     bbox=_bbox(DIST_C), zorder=7,
                     fontfamily=FONT_KW['family'],
                     fontweight=FONT_KW['weight'],
                     fontsize=FONT_KW['size'])

    # ── Velocity overlay ──────────────────────────────────────────
    # Maps the temporal Vel<step>_<bp> features onto a static schematic.
    # Two render modes, picked by which optional argument is supplied:
    #
    #   vel_trail : list[dict]  — full-trail mode. Continuous polyline
    #     from t-vel_step → t+vel_step through the keypoint's actual
    #     positions (sampled every frame), with marker dots at
    #     t-vel_step / t / t+vel_step and a solid-past + dashed-future
    #     split. Reads as a comet trail through the cropped canvas.
    #
    #   kps_future : dict       — single-arrow mode (legacy). One
    #     dashed arrow t → t+vel_step with a ghost dot at the future
    #     position. Picked up only when vel_trail is None.
    #
    # Two example bodyparts: centroid (whole-body translation) +
    # hlpaw (the more dynamic limb).
    velocity_specs = [
        # (bp, label_dx, label_dy, ha)
        ('centroid', 70,  -30, 'left'),
        ('hlpaw',   -70,   30, 'right'),
    ]

    if vel_trail is not None:
        # vel_trail is a list of length 2*vel_step+1 dicts ordered
        # from frame_idx-vel_step → frame_idx+vel_step. Missing keys
        # are skipped per-bodypart (handles edge frames near 0 / N).
        cur_idx = vel_step  # index of the "current" frame inside vel_trail
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

            # Past + future split through whichever frames we have
            past = [p for p in local[: cur_idx + 1] if p is not None]
            future = [p for p in local[cur_idx:]    if p is not None]
            if past:
                ax.plot([p[0] for p in past], [p[1] for p in past],
                          color=VEL_C, linewidth=1.6, alpha=0.55,
                          zorder=4)
            if future:
                ax.plot([p[0] for p in future], [p[1] for p in future],
                          color=VEL_C, linewidth=1.6, alpha=0.95,
                          linestyle='--', zorder=4)

            # Marker dots at t-vel_step / t / t+vel_step
            markers = [
                (0,                     50, 0.40, 0),     # t-step, faded
                (cur_idx,               110, 0.95, 0.6),   # t,      bright
                (len(local) - 1,        70, 0.55, 0),     # t+step, semi
            ]
            for idx, sz, alpha, edge in markers:
                if 0 <= idx < len(local) and local[idx] is not None:
                    mx_, my_ = local[idx]
                    ax.scatter([mx_], [my_], s=sz, color=VEL_C,
                                 alpha=alpha,
                                 edgecolor='white' if edge else 'none',
                                 linewidth=edge,
                                 zorder=5)

            # Label at the current position
            cur = local[cur_idx]
            if cur is None:
                cur = valid[len(valid) // 2]
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

    elif kps_future is not None:
        future_local = {bp: (x - x0, y - y0)
                          for bp, (x, y) in kps_future.items()}
        for bp, ldx, ldy, ha in velocity_specs:
            if bp not in kps or bp not in future_local:
                continue
            x_a, y_a = kps[bp]
            x_b, y_b = future_local[bp]
            dist = ((x_b - x_a) ** 2 + (y_b - y_a) ** 2) ** 0.5
            if dist < 4:
                continue

            ax.scatter([x_b], [y_b], s=120, color=VEL_C, alpha=0.32,
                         edgecolor='white', linewidth=0.5, zorder=4)
            ax.scatter([x_b], [y_b], s=24,  color=VEL_C, alpha=0.85,
                         zorder=5)
            ax.annotate('', xy=(x_b, y_b), xytext=(x_a, y_a),
                         arrowprops=dict(arrowstyle='->', color=VEL_C,
                                           lw=1.6, linestyle='--',
                                           shrinkA=5, shrinkB=5,
                                           alpha=0.95),
                         zorder=6)
            mx = (x_a + x_b) / 2
            my = (y_a + y_b) / 2
            ax.annotate(f'⇢ Vel{vel_step}_{bp}\n(t → t+{vel_step})',
                         (mx, my), xytext=(mx + ldx, my + ldy),
                         color=VEL_C, ha=ha, va='center',
                         arrowprops=dict(arrowstyle='-', color=VEL_C,
                                           lw=0.7, alpha=0.85),
                         bbox=_bbox(VEL_C), zorder=7,
                         fontfamily=FONT_KW['family'],
                         fontweight=FONT_KW['weight'],
                         fontsize=FONT_KW['size'])

    fig.savefig(out_path, dpi=100, facecolor='black')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 5 — side-by-side raw frame ↔ skeleton-red schematic
# ─────────────────────────────────────────────────────────────────────

def render_red_side_by_side(bgr_full: np.ndarray, kps_full: dict,
                              cache_row: pd.Series, frame_idx: int,
                              red_png: str, out_path: str, size: int = 600):
    """1202×600 PNG: left = raw cropped frame (untouched), right = the
    standalone skeleton-red schematic (paw-pixel stamps + dim skeleton
    + keypoint dots + feature annotations on a black background) read
    from `red_png`. The standalone schematic must already exist."""
    H, W = bgr_full.shape[:2]
    half = size // 2

    # Same crop region as the right panel
    cx0, cy0 = kps_full['centroid']
    x0 = int(max(0, min(W - size, round(cx0) - half)))
    y0 = int(max(0, min(H - size, round(cy0) - half)))
    left = bgr_full[y0:y0 + size, x0:x0 + size].copy()

    right = cv2.imread(red_png)
    if right is None or right.shape[:2] != (size, size):
        sys.exit(f"side-by-side: could not load {red_png} as a "
                  f"{size}×{size} image")

    # 1-px white divider, no headers — the panels speak for themselves
    sep = np.full((size, 2, 3), 255, dtype=np.uint8)
    combo = np.hstack([left, sep, right])
    cv2.imwrite(out_path, combo)


# ─────────────────────────────────────────────────────────────────────
# Figure 6 — cartoon "fake" feature matrix (~700 cols × N frames)
# ─────────────────────────────────────────────────────────────────────

def render_feature_matrix_cartoon(out_path: str,
                                    n_frames: int = 8,
                                    width: int = 1400,
                                    height: int = 600):
    """Illustrative figure showing the per-frame feature matrix that
    feeds XGBoost. Cells are filled with the column-group colour (no
    numeric values) so the figure reads as a *layout cartoon*, not real
    data."""
    # Column-group layout. Counts shrunk for legibility — the goal is a
    # readable cartoon, not a literal 700-column wall. Colours sampled
    # evenly from matplotlib's plasma colormap so the figure reads as a
    # single thematic palette.
    plasma = plt.get_cmap('plasma')
    _names_counts = [
        ('xy',     4),  ('Dis',    6),  ('Ang',   10), ('Height', 3),
        ('Vel',    5),  ('Jerk',   3),  ('Pix',    3), ('ΔBrt',   3),
        ('Sil',    2),  ('lags',   6),  ('…',      3),
    ]
    n_groups = len(_names_counts)
    groups = [
        (name, n, plasma(0.05 + 0.85 * i / max(1, n_groups - 1)))
        for i, (name, n) in enumerate(_names_counts)
    ]
    total_cols = sum(c for _, c, _ in groups)

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                      facecolor='white')
    ax_top = fig.add_axes([0.06, 0.78, 0.74, 0.10])
    ax_mat = fig.add_axes([0.06, 0.12, 0.74, 0.66])
    ax_leg = fig.add_axes([0.82, 0.12, 0.16, 0.76])
    for a in (ax_top, ax_mat, ax_leg):
        a.set_xticks([]); a.set_yticks([])

    # ── Top banner: coloured group bands with names ─────────────────
    ax_top.set_xlim(0, total_cols); ax_top.set_ylim(0, 1)
    x = 0
    for name, n, color in groups:
        ax_top.add_patch(mpatches.Rectangle((x, 0), n, 1,
                                                facecolor=color,
                                                edgecolor='white',
                                                linewidth=0.8))
        ax_top.text(x + n / 2, 0.5, name, ha='center', va='center',
                      fontsize=9, color='black', fontweight='bold')
        x += n
    ax_top.set_title(
        'Per-frame feature matrix — ~700 columns × N frames  '
        '(layout cartoon — counts shrunk for legibility)',
        fontsize=11, fontweight='bold')

    # ── Matrix: solid colour cells per group, no values ─────────────
    ax_mat.set_xlim(0, total_cols); ax_mat.set_ylim(n_frames, 0)
    for r in range(n_frames):
        x = 0
        for _, n, color in groups:
            for c in range(n):
                # Slight alpha jitter purely for visual rhythm — no data
                ax_mat.add_patch(mpatches.Rectangle(
                    (x + c, r), 1, 1,
                    facecolor=color,
                    edgecolor='white', linewidth=0.5,
                    alpha=0.55 + 0.45 * ((r * 7 + c * 3) % 3) / 2))
            x += n
    # Bold separators between groups
    x = 0
    for _, n, _ in groups[:-1]:
        x += n
        ax_mat.axvline(x, color='black', linewidth=0.8, alpha=0.7)
    ax_mat.set_ylabel('frame  →  time', fontsize=10)
    ax_mat.set_xlabel(f'feature column  (1 … {total_cols} shown)',
                        fontsize=10)
    # Row labels: t-3 … t … t+4 — symbolic time index, no data
    half = n_frames // 2
    ax_mat.set_yticks([r + 0.5 for r in range(n_frames)])
    ax_mat.set_yticklabels([f't{i - half:+d}' if i != half else 't'
                                 for i in range(n_frames)], fontsize=8)

    # ── Legend column on the right: group names + symbol meanings ──
    ax_leg.set_xlim(0, 1); ax_leg.set_ylim(0, 1)
    for s in ax_leg.spines.values():
        s.set_visible(False)
    ax_leg.text(0.5, 0.97,
                  '~700 columns → XGBoost',
                  ha='center', va='top',
                  fontsize=11, fontweight='bold')
    # Pull the same plasma sample so the legend swatches match the
    # banner / matrix exactly
    legend_meta = [
        ('xy',     'keypoint coords'),
        ('Dis',    'pairwise distances'),
        ('Ang',    'joint angles'),
        ('Height', 'distance from "floor"'),
        ('Vel',    'velocity'),
        ('Jerk',   'kinematics'),
        ('Pix',    'paw brightness ROI'),
        ('ΔBrt',   'brightness deltas'),
        ('Sil',    'silhouette / blob'),
        ('lags',   'temporal lags'),
        ('…',      'other'),
    ]
    legend_rows = [
        (sym, plasma(0.05 + 0.85 * i / max(1, len(legend_meta) - 1)),
         body)
        for i, (sym, body) in enumerate(legend_meta)
    ]
    yy = 0.90
    for sym, color, body in legend_rows:
        ax_leg.add_patch(mpatches.Rectangle(
            (0.04, yy - 0.022), 0.12, 0.030,
            facecolor=color, edgecolor='black', linewidth=0.4))
        ax_leg.text(0.20, yy - 0.007, sym,
                      ha='left', va='center', fontsize=9,
                      fontweight='bold')
        ax_leg.text(0.20, yy - 0.035, body,
                      ha='left', va='center', fontsize=7.5,
                      color='#444')
        yy -= 0.075
    ax_leg.text(0.5, 0.03,
                  '(cartoon — no real values)',
                  ha='center', va='bottom', fontsize=8,
                  style='italic', color='#666')

    fig.savefig(out_path, dpi=100, facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Figure 7 — feature matrix ↔ cartoon SHAP plot side-by-side
# ─────────────────────────────────────────────────────────────────────

def render_matrix_shap_side_by_side(out_path: str,
                                       width: int = 1100,
                                       height: int = 1400):
    """Slide-ready portrait cartoon (designed to drop onto a
    PowerPoint slide). Top-to-bottom flow:

        [ feature matrix X | y | BORIS logo ]
                       ↓
                 XGBoost + SHAP
                       ↓
              [ SHAP top features ]
                       ↓
            save behavior predictions

    Panel-C-inspired styling: inline text labels (no pills), thin
    arrows with small open heads, BORIS logo sits adjacent to the
    `y` column (no connector arrow — visual proximity does the
    labelling). Same plasma palette as
    `render_feature_matrix_cartoon`."""
    # Same shrunk-count layout + plasma palette as the standalone
    # matrix cartoon, so the two figures are obviously the same data
    # structure with the same colour key.
    plasma = plt.get_cmap('plasma')
    groups = [
        ('xy',     4),  ('Dis',    6),  ('Ang',   10), ('Height', 3),
        ('Vel',    5),  ('Jerk',   3),  ('Pix',    3), ('ΔBrt',   3),
        ('Sil',    2),  ('lags',   6),  ('…',      3),
    ]
    n_groups = len(groups)
    GROUP_C = {
        name: plasma(0.05 + 0.85 * i / max(1, n_groups - 1))
        for i, (name, _) in enumerate(groups)
    }
    total_cols = sum(c for _, c in groups)
    n_frames = 8

    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100,
                      facecolor='white')

    # Portrait layout (figure-fraction coords). Top-to-bottom flow.
    # Matrix and SHAP axes share the same horizontal extent so the
    # two data areas line up vertically. BORIS callout sits adjacent
    # to the `y` column at the top — no connector arrow, just
    # proximity-as-labelling.
    # Matrix and SHAP share x=[0.18, 0.66], BORIS gets its own column
    # at x=[0.70, 0.98] (canvas widened from 900 → 1100 so the logo is
    # well clear of the y-column). Probability-per-frame strip lives
    # under the SHAP plot.
    ax_top    = fig.add_axes([0.18, 0.905, 0.48, 0.020])
    ax_mat    = fig.add_axes([0.18, 0.665, 0.48, 0.240])
    ax_y      = fig.add_axes([0.665, 0.665, 0.018, 0.240])
    boris_ax  = fig.add_axes([0.70, 0.665, 0.28, 0.240])
    ax_shap   = fig.add_axes([0.18, 0.215, 0.48, 0.350])
    ax_pred   = fig.add_axes([0.18, 0.045, 0.48, 0.080])
    for a in (ax_top, ax_mat, ax_y, ax_shap, boris_ax, ax_pred):
        a.set_xticks([]); a.set_yticks([])
    for a in (ax_top, ax_mat, ax_y, ax_shap, boris_ax):
        a.set_xticks([]); a.set_yticks([])

    # ── Top: column-group banner ───────────────────────────────────
    ax_top.set_xlim(0, total_cols); ax_top.set_ylim(0, 1)
    x = 0
    for name, n in groups:
        ax_top.add_patch(mpatches.Rectangle((x, 0), n, 1,
                                                facecolor=GROUP_C[name],
                                                edgecolor='white',
                                                linewidth=0.8))
        ax_top.text(x + n / 2, 0.5, name, ha='center', va='center',
                      fontsize=9, color='black', fontweight='bold')
        x += n
    ax_top.set_title('Feature matrix (X)', fontsize=11,
                       fontweight='bold', pad=6)

    # ── Matrix: solid colour cells, group-coloured ─────────────────
    ax_mat.set_xlim(0, total_cols); ax_mat.set_ylim(n_frames, 0)
    for r in range(n_frames):
        x = 0
        for name, n in groups:
            for c in range(n):
                ax_mat.add_patch(mpatches.Rectangle(
                    (x + c, r), 1, 1,
                    facecolor=GROUP_C[name],
                    edgecolor='white', linewidth=0.5,
                    alpha=0.55 + 0.45 * ((r * 7 + c * 3) % 3) / 2))
            x += n
    x = 0
    for name, n in groups[:-1]:
        x += n
        ax_mat.axvline(x, color='black', linewidth=0.8, alpha=0.7)
    ax_mat.set_xlabel(f'feature column  (1 … {total_cols} shown)',
                        fontsize=10)
    half = n_frames // 2
    ax_mat.set_yticks([r + 0.5 for r in range(n_frames)])
    ax_mat.set_yticklabels([f't{i - half:+d}' if i != half else 't'
                                 for i in range(n_frames)], fontsize=8)
    ax_mat.set_ylabel('frame  →  time', fontsize=10)

    # ── y-column: per-frame BORIS labels (synthetic bout pattern) ──
    # Green = behaviour active, light grey = absent. Synthetic so the
    # cartoon shows a bout starting partway through and ending before
    # the window closes — typical of how BORIS event-coded data looks.
    ax_y.set_xlim(0, 1); ax_y.set_ylim(n_frames, 0)
    label_pattern = [0, 0, 1, 1, 1, 1, 0, 0]
    if len(label_pattern) < n_frames:
        label_pattern += [0] * (n_frames - len(label_pattern))
    for r in range(n_frames):
        col = '#22cc66' if label_pattern[r] else '#dddddd'
        ax_y.add_patch(mpatches.Rectangle((0, r), 1, 1,
                                              facecolor=col,
                                              edgecolor='white',
                                              linewidth=0.5))
    ax_y.set_title('y', fontsize=11, fontweight='bold', pad=6)

    # ── BORIS callout: logo + plain caption in its own column ─────
    # No connector arrow — visual proximity to the `y` column is
    # enough to read as "the y column comes from BORIS labels".
    boris_ax.set_xlim(0, 1); boris_ax.set_ylim(0, 1)
    for s in boris_ax.spines.values():
        s.set_visible(False)
    _logo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                'assets', 'boris_logo.png')
    if os.path.isfile(_logo_path):
        logo_img = plt.imread(_logo_path)
        from matplotlib.offsetbox import OffsetImage, AnnotationBbox
        imagebox = OffsetImage(logo_img, zoom=0.20)
        ab = AnnotationBbox(imagebox, (0.50, 0.65),
                              xycoords='axes fraction', frameon=False,
                              box_alignment=(0.5, 0.5))
        boris_ax.add_artist(ab)
    boris_ax.text(0.50, 0.28, 'BORIS', ha='center', va='center',
                    fontsize=13, fontweight='bold', color='#222',
                    fontfamily=['Arial', 'DejaVu Sans'])
    boris_ax.text(0.50, 0.17, 'manual labels',
                    ha='center', va='center', fontsize=10,
                    color='#444',
                    fontfamily=['Arial', 'DejaVu Sans'])
    boris_ax.text(0.50, 0.07, '(event-coded bouts)',
                    ha='center', va='center', fontsize=8.5,
                    color='#888', fontstyle='italic',
                    fontfamily=['Arial', 'DejaVu Sans'])

    # ── Centre flow: thin arrows + inline "XGBoost + SHAP" label ──
    # Aligned on x=0.42 (= matrix horizontal centre, 0.18 + 0.48/2).
    def _down_arrow(y_top, y_bot, x=0.42, color='#666', lw=1.2):
        a = mpatches.FancyArrowPatch(
            (x, y_top), (x, y_bot),
            transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=12, lw=lw, color=color)
        fig.add_artist(a)

    _down_arrow(y_top=0.640, y_bot=0.620)
    fig.text(0.42, 0.605, 'XGBoost  +  SHAP',
              ha='center', va='center', fontsize=13,
              fontweight='bold', color='#222',
              fontfamily=['Arial', 'DejaVu Sans'])
    _down_arrow(y_top=0.590, y_bot=0.570)

    # ── Bottom: cartoon SHAP horizontal bar chart ─────────────────
    shap_features = [
        ('Ang_neck-centroid-tailbase',   'Ang',     1.00),
        ('Pix_hlpaw',                    'Pix',     0.84),
        ('Height_centroid',              'Height',  0.71),
        ('Ang_snout-neck-centroid',      'Ang',     0.65),
        ('Vel10_centroid',               'Vel',     0.58),
        ('Dis_snout-tailbase',           'Dis',     0.50),
        ('ΔBrt_hlpaw',                   'ΔBrt',    0.43),
        ('Sil_blob_frac',                'Sil',     0.36),
        ('Pix_hrpaw',                    'Pix',     0.28),
        ('Jerk_hlpaw',                   'Jerk',    0.21),
    ]
    n = len(shap_features)
    ys = list(range(n))[::-1]
    bars = [imp for _, _, imp in shap_features]
    cols = [GROUP_C[grp] for _, grp, _ in shap_features]

    ax_shap.set_xlim(0, 1.18)
    # Tight y-limits so bars take the full row with no dead space —
    # ylim = ±0.5 around the integer indices, height=0.95 fills it.
    # ys = [n-1, ..., 0] so highest-importance feature (index 0 in
    # shap_features) sits at the top of the axis.
    ax_shap.set_ylim(-0.5, n - 0.5)
    ax_shap.barh(ys, bars, color=cols, edgecolor='black', linewidth=0.5,
                    height=0.95)
    ax_shap.set_yticks(ys)
    ax_shap.set_yticklabels(
        [name for name, _, _ in shap_features],
        fontsize=9, fontfamily=['Arial', 'DejaVu Sans'])
    ax_shap.tick_params(axis='y', left=False, pad=2)
    for y, (_, grp, imp) in zip(ys, shap_features):
        ax_shap.text(imp + 0.012, y, grp, ha='left', va='center',
                       fontsize=8, color='#444', fontstyle='italic')
    # Real x-axis ticks at evenly-spaced "cartoon importance" values
    ax_shap.set_xticks([0.0, 0.25, 0.50, 0.75, 1.00])
    ax_shap.tick_params(axis='x', labelsize=8, length=3)
    ax_shap.set_xlabel('mean |SHAP|  (cartoon — no real values)',
                          fontsize=10)
    ax_shap.spines['top'].set_visible(False)
    ax_shap.spines['right'].set_visible(False)
    ax_shap.spines['left'].set_visible(False)

    # ── Bottom: arrow + probability-per-frame graph ────────────────
    _down_arrow(y_top=0.155, y_bot=0.135)
    fig.text(0.42, 0.123, 'predicted P(behavior) per frame',
              ha='center', va='center', fontsize=11,
              fontweight='bold', color='#1565d8',
              fontfamily=['Arial', 'DejaVu Sans'])

    # Synthetic probability trace — mostly near zero baseline with a
    # handful of sharp spikes; threshold dashed green; above-threshold
    # frames shaded peach (= predicted bouts). Mirrors the in-app
    # Probability Graph window aesthetic.
    n_pred = 1000
    x_p = np.arange(n_pred)
    rng = np.random.default_rng(7)
    trace = 0.015 + 0.020 * np.abs(rng.standard_normal(n_pred))
    spikes = [
        (110, 0.62,  6), (180, 0.18,  9), (260, 0.38,  7),
        (320, 0.82,  5), (335, 0.92,  4), (430, 0.55, 10),
        (560, 0.96,  4), (575, 0.88,  5),
        (700, 0.30, 11), (820, 0.72,  5),
    ]
    for pos, height, sigma in spikes:
        trace = trace + height * np.exp(-((x_p - pos) ** 2) /
                                          (2 * sigma ** 2))
    trace = np.clip(trace, 0, 1)
    threshold = 0.70

    # Shade above-threshold regions (predicted bouts)
    ax_pred.fill_between(x_p, 0, 1.05,
                            where=trace > threshold,
                            color='#ffb380', alpha=0.55, lw=0,
                            zorder=1)
    # Probability trace
    ax_pred.plot(x_p, trace, color='#1565d8', lw=0.9, zorder=3)
    # Threshold line
    ax_pred.axhline(threshold, color='#22aa66', linestyle='--',
                      lw=1.1, zorder=2)
    ax_pred.text(n_pred * 0.995, threshold + 0.04,
                   f'threshold ({threshold:.2f})',
                   ha='right', va='bottom', fontsize=7,
                   color='#22aa66', fontstyle='italic')

    ax_pred.set_xlim(0, n_pred)
    ax_pred.set_ylim(0, 1.05)
    ax_pred.set_xticks([0, 250, 500, 750, 1000])
    ax_pred.set_yticks([0, 0.5, 1.0])
    ax_pred.tick_params(axis='both', labelsize=7, length=3)
    ax_pred.set_xlabel('frame', fontsize=9)
    ax_pred.set_ylabel('P(behavior)', fontsize=9)
    ax_pred.spines['top'].set_visible(False)
    ax_pred.spines['right'].set_visible(False)

    fig.savefig(out_path, dpi=100, facecolor='white')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',
                     default=r'E:\RSVIDS\Blackbox\2604_DV_DSS')
    ap.add_argument('--session',
                     default='202604_Ai32_61_BL_cropped')
    # Frame default 52282 was tuned for the 2604_DV_DSS demo pose; the
    # 2510_Blackbox_Rimonabant default project for the new
    # skeleton-red figures uses 25930 (well-stretched body + visible
    # centroid + hindpaw motion in the 10-frame window).
    ap.add_argument('--frame', type=int, default=52282)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'analysis_output',
        f'feature_schematic_{datetime.now():%Y%m%d_%H%M%S}')
    os.makedirs(out_dir, exist_ok=True)

    print(f"[1/4] Resolving inputs…")
    video_path = find_video(args.project, args.session)
    dlc_path   = find_dlc_h5(args.project, args.session)
    cache_path = find_cache_pkl(args.project, args.session)
    print(f"      video : {video_path}")
    print(f"      DLC h5: {dlc_path}")
    print(f"      cache : {cache_path}")

    print(f"[2/4] Reading frame {args.frame} + keypoints + cache row…")
    frame     = grab_frame_gray(video_path, args.frame)
    bgr_frame = grab_frame_bgr(video_path,  args.frame)
    x_df, y_df = load_dlc_xy(dlc_path)
    kps = keypoints_at(x_df, y_df, args.frame)
    # Velocity overlay inputs:
    #   - kps_future: just the t+VEL_STEP frame (single-arrow mode)
    #   - vel_trail : 2*VEL_STEP+1 keypoint dicts spanning t-VEL_STEP →
    #                 t+VEL_STEP (multi-frame "comet trail" mode)
    VEL_STEP = 10
    future_idx = min(args.frame + VEL_STEP, len(x_df) - 1)
    kps_future = keypoints_at(x_df, y_df, future_idx)
    vel_trail = []
    for offset in range(-VEL_STEP, VEL_STEP + 1):
        f = args.frame + offset
        if 0 <= f < len(x_df):
            vel_trail.append(keypoints_at(x_df, y_df, f))
        else:
            vel_trail.append({})  # padding for edge frames
    X = pd.read_pickle(cache_path)
    cache_row = X.iloc[args.frame]
    print(f"      frame shape: {frame.shape}")
    print(f"      keypoints  : {len(kps)} / {len(BODYPARTS)} loaded "
          f"({list(kps)})")

    print(f"[3/9] Rendering pose schematic…")
    pose_png = os.path.join(out_dir, 'pose_schematic.png')
    render_pose_schematic(frame, kps, y_df, cache_row,
                            args.frame, pose_png)
    print(f"      → {pose_png}")

    print(f"[4/9] Rendering paw brightness schematic (cartoon + photo)…")
    paw_png = os.path.join(out_dir, 'paw_brightness_schematic.png')
    render_paw_brightness_schematic(frame, kps, cache_row,
                                       args.frame, paw_png)
    print(f"      → {paw_png}")

    print(f"[5/9] Rendering PowerPoint summary cartoon…")
    pp_png = os.path.join(out_dir, 'feature_set_powerpoint.png')
    render_powerpoint_summary(frame, kps, cache_row,
                                args.frame, pp_png)
    print(f"      → {pp_png}")

    print(f"[6/11] Rendering simple 600×600 skeleton-red schematic "
          f"(arrow-style velocity)…")
    red_png = os.path.join(out_dir, 'skeleton_red_schematic.png')
    render_simple_red_schematic(bgr_frame, kps, cache_row,
                                   args.frame, red_png, size=600,
                                   kps_future=kps_future,
                                   vel_step=VEL_STEP)
    print(f"      → {red_png}")

    print(f"[7/11] Rendering trail-style schematic (multi-frame "
          f"comet trail)…")
    red_trail_png = os.path.join(out_dir,
                                    'skeleton_red_schematic_trail.png')
    render_simple_red_schematic(bgr_frame, kps, cache_row,
                                   args.frame, red_trail_png, size=600,
                                   vel_trail=vel_trail,
                                   vel_step=VEL_STEP)
    print(f"      → {red_trail_png}")

    print(f"[8/11] Rendering side-by-side raw ↔ annotated (arrow)…")
    sbs_png = os.path.join(out_dir, 'skeleton_red_side_by_side.png')
    render_red_side_by_side(bgr_frame, kps, cache_row,
                              args.frame, red_png, sbs_png, size=600)
    print(f"      → {sbs_png}")

    print(f"[9/11] Rendering side-by-side raw ↔ annotated (trail)…")
    sbs_trail_png = os.path.join(out_dir,
                                    'skeleton_red_side_by_side_trail.png')
    render_red_side_by_side(bgr_frame, kps, cache_row,
                              args.frame, red_trail_png, sbs_trail_png,
                              size=600)
    print(f"      → {sbs_trail_png}")

    print(f"[10/11] Rendering cartoon feature-matrix illustration…")
    mat_png = os.path.join(out_dir, 'feature_matrix_cartoon.png')
    render_feature_matrix_cartoon(mat_png)
    print(f"      → {mat_png}")

    print(f"[11/11] Rendering matrix ↔ SHAP cartoon side-by-side…")
    shap_png = os.path.join(out_dir, 'feature_matrix_shap_cartoon.png')
    render_matrix_shap_side_by_side(shap_png)
    print(f"      → {shap_png}")

    print(f"\nDone. Outputs → {out_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
