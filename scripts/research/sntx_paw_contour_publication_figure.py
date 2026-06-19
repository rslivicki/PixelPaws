"""
Publication-quality 6-panel figure replicating the GUI Gait & Limb
"Paw Print" + "Mean Contour" panels for the SNTX (n=4) vs Naïve (n=3)
baseline cohort in 2603_SNLT_JG, with SNTX recoloured RED.

Reads 7 `_shapes.npz` files from
  E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline\gait_limb_analysis
each containing per-paw normalized (N, 64, 2) contour arrays (HL, HR).

Applies the same paw-like filter the GUI uses
(`gait_limb_tab.py:9013-9030`):
  solidity      <= 1.00
  aspect_ratio  <= 1.60
  circularity   >= 0.10

Renders SVG (vector) and PNG (preview) and posts both to the #results
Discord channel.

Run with the regular py env:
  PYTHONIOENCODING=utf-8 py -X utf8 -u \
    scripts/research/sntx_paw_contour_publication_figure.py
"""
from __future__ import annotations

import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path

PROJECT = Path(r"E:\RSVIDS\Blackbox\2603_SNLT_JG\Baseline")
GLAB = PROJECT / "gait_limb_analysis"
ANALYSIS = PROJECT / "analysis"

# Treatment lookup: mouse-id string -> 'SNTX' | 'Naive'
KEY_FILE = PROJECT / "key_file.csv"

# Paw-like thresholds (GUI defaults) — used for the mean-contour panels
# and to compute the per-paw counts reported on each panel title.
SOLIDITY_MAX = 1.00
ASPECT_MAX   = 1.60
CIRCULARITY_MIN = 0.10

# STRICT "clean toed paw print" filter — used for picking the single
# representative shape shown in the top two rows. Much tighter than the
# GUI's permissive paw-like default: forces visible toe detail
# (low circularity = high perimeter:area), allows some concavity between
# toes (solidity well below 1), and disallows highly elongated blobs.
STRICT_CIRC_MAX   = 0.30   # circularity < 0.30 -> noticeable toe spread
STRICT_SOLID_MAX  = 0.78
STRICT_ASPECT_MAX = 1.50

GROUP_ORDER = ["SNTX", "Naive"]
GROUP_COLORS = {
    "SNTX":  "#d62728",   # red (was teal in the screenshot)
    "Naive": "#000000",   # black
}

WEBHOOK = (
    ""
)


def step(msg: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def discord_upload(content: str, files: list[Path]) -> None:
    boundary = "----PP" + uuid.uuid4().hex
    body = [
        f"--{boundary}".encode(),
        b'Content-Disposition: form-data; name="payload_json"',
        b'Content-Type: application/json', b"",
        json.dumps({"content": content}).encode(),
    ]
    for i, p in enumerate(files):
        suffix = p.suffix.lower()
        if suffix == ".svg":
            mime = "image/svg+xml"
        elif suffix == ".png":
            mime = "image/png"
        else:
            mime = "application/octet-stream"
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
                 "User-Agent": "PixelPaws-SNTX-ContourFig/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


# --------------------------------------------------------------------- #
# Shape metrics on normalized 64x2 contours
# --------------------------------------------------------------------- #
def shape_metrics(stacked):
    """Vectorised solidity, aspect_ratio, circularity for (N, 64, 2)
    normalized contour points.

    Solidity is scale-invariant for normalized shapes (area / hull_area).
    Aspect ratio and circularity match GUI's `_shape_metrics_batch`.
    """
    import numpy as np
    try:
        from scipy.spatial import ConvexHull
    except Exception as e:
        raise RuntimeError(f"scipy.spatial.ConvexHull required: {e}")

    mins = stacked.min(axis=1); maxs = stacked.max(axis=1)
    wh = maxs - mins
    dim_max = wh.max(axis=1); dim_min = wh.min(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ar = np.where(dim_min > 0, dim_max / dim_min, 999.0)

    closed = np.concatenate([stacked, stacked[:, 0:1, :]], axis=1)
    diffs = np.diff(closed, axis=1)
    perimeter = np.sqrt((diffs ** 2).sum(axis=2)).sum(axis=1)
    x, y = stacked[:, :, 0], stacked[:, :, 1]
    area = 0.5 * np.abs(
        (x * np.roll(y, -1, axis=1)).sum(axis=1)
        - (y * np.roll(x, -1, axis=1)).sum(axis=1)
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        circ = np.where(perimeter > 0,
                        (4 * np.pi * area) / (perimeter ** 2), 0.0)

    solidity = np.zeros(len(stacked))
    for i in range(len(stacked)):
        pts = stacked[i]
        if area[i] <= 0:
            solidity[i] = 0.0; continue
        try:
            hull = ConvexHull(pts)
            hull_area = hull.volume   # 'volume' = area for 2D
            solidity[i] = area[i] / hull_area if hull_area > 0 else 0.0
        except Exception:
            solidity[i] = 0.0
    return solidity, ar, circ


def load_groups():
    """Return {role: {group: (shapes_all, shapes_pawlike, n_pawlike, n_total)}}."""
    import numpy as np
    import pandas as pd

    df = pd.read_csv(KEY_FILE).dropna()
    mouse_to_group = {row["Subject"]: row["Treatment"]
                      for _, row in df.iterrows()
                      if row["Treatment"] in GROUP_ORDER}
    step(f"key file: {len(mouse_to_group)} mice mapped "
         f"(SNTX={sum(v=='SNTX' for v in mouse_to_group.values())}, "
         f"Naive={sum(v=='Naive' for v in mouse_to_group.values())})")

    pooled_all = {r: {g: [] for g in GROUP_ORDER} for r in ("HL", "HR")}

    for p in sorted(GLAB.glob("*_contour_*_shapes.npz")):
        # Extract subject ID. Filenames look like:
        #   260318_JG_9417_Baseline_contour_<hash>_shapes.npz
        name = p.name
        # Walk through the configured mice and find which one is in the filename
        match = next((mid for mid in mouse_to_group if mid in name), None)
        if not match:
            step(f"  ! no key match for {name}"); continue
        group = mouse_to_group[match]
        with np.load(p) as data:
            for role in ("HL", "HR"):
                if role in data:
                    pooled_all[role][group].append(data[role])
        step(f"  {match} -> {group}")

    # Concatenate + paw-like filter
    out = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            chunks = pooled_all[role][g]
            if not chunks:
                continue
            shapes_all = np.concatenate(chunks, axis=0)
            sol, ar, circ = shape_metrics(shapes_all)
            mask = (sol <= SOLIDITY_MAX) & (ar <= ASPECT_MAX) & (circ >= CIRCULARITY_MIN)
            shapes_pawlike = shapes_all[mask]
            out[role][g] = (shapes_all, shapes_pawlike,
                            int(mask.sum()), int(len(shapes_all)))
            step(f"  pooled {role} {g}: paw-like {int(mask.sum())} / "
                 f"{len(shapes_all)} total")
    return out


# --------------------------------------------------------------------- #
# Canonical-orientation alignment
# --------------------------------------------------------------------- #
def pca_canonical_align(pts):
    """Rotate / mirror a (64, 2) contour to a canonical orientation:
       - principal axis vertical (along +Y)
       - "narrow end" (lower per-row X spread) at +Y (so the toes point up)
       - mean of X coordinates non-negative (consistent left/right mirror)
    All transforms are rigid (rotation + axis flips) so shape is preserved.
    """
    import numpy as np
    c = pts.mean(axis=0)
    p = pts - c
    cov = np.cov(p.T)
    _, eigvecs = np.linalg.eigh(cov)
    major = eigvecs[:, -1]  # eigenvector with largest eigenvalue
    # Angle that rotates `major` onto +Y axis
    theta = np.arctan2(major[0], major[1])
    R = np.array([[np.cos(-theta), -np.sin(-theta)],
                  [np.sin(-theta), np.cos(-theta)]])
    a = p @ R.T

    # Flip Y so the narrower (smaller X-extent) end is on +Y (toes up).
    pos = a[a[:, 1] > 0]; neg = a[a[:, 1] < 0]
    if len(pos) > 1 and len(neg) > 1:
        if np.ptp(pos[:, 0]) > np.ptp(neg[:, 0]):
            a[:, 1] *= -1

    # Flip X so left/right asymmetry is on the +X side.
    if np.sum(a[:, 0]) < 0:
        a[:, 0] *= -1
    return a


def align_pool(stacked):
    """Canonically align every shape; return (N, 64, 2)."""
    import numpy as np
    out = np.empty_like(stacked)
    for i in range(len(stacked)):
        out[i] = pca_canonical_align(stacked[i])
    return out


def pick_representative(aligned):
    """Return (index, shape) for the canonically-aligned shape closest
    to the pool mean (lowest sum-squared deviation, post cyclic align).

    Cyclic alignment: for each shape try all 64 starting-point rotations
    and pick the lowest SSE against the mean. The 64 sample points are
    evenly spaced along the perimeter so cyclic shifts are valid here.
    """
    import numpy as np
    mean = aligned.mean(axis=0)
    n_pts = aligned.shape[1]
    best_overall_sse = np.inf
    best_overall_idx = 0
    best_overall_shape = aligned[0]
    for i in range(len(aligned)):
        s = aligned[i]
        best_sse = np.inf; best_shift = 0
        for k in range(n_pts):
            shifted = np.roll(s, k, axis=0)
            sse = float(np.sum((shifted - mean) ** 2))
            if sse < best_sse:
                best_sse = sse; best_shift = k
        if best_sse < best_overall_sse:
            best_overall_sse = best_sse
            best_overall_idx = i
            best_overall_shape = np.roll(s, best_shift, axis=0)
    return best_overall_idx, best_overall_shape


# --------------------------------------------------------------------- #
# Plotting
# --------------------------------------------------------------------- #
def plot_paw_print(ax, shape_pts, color, title, alpha_fill=1.0):
    """Render one filled paw-print outline."""
    import numpy as np
    pts = np.vstack([shape_pts, shape_pts[0:1]])
    ax.fill(pts[:, 0], pts[:, 1], color=color, alpha=alpha_fill,
            edgecolor=color, linewidth=1.4)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title(title, fontsize=10)
    ax.invert_yaxis()


def plot_mean_contour(ax, shapes_dict, title):
    """SD-envelope mean contour overlay, both groups in shapes_dict."""
    import numpy as np
    for group in GROUP_ORDER:
        if group not in shapes_dict:
            continue
        stacked = shapes_dict[group]
        if len(stacked) == 0:
            continue
        mean_shape = stacked.mean(axis=0)
        sd_shape = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean_shape)
        mean_closed = np.vstack([mean_shape, mean_shape[0:1]])
        sd_closed = np.vstack([sd_shape, sd_shape[0:1]])

        radial_sd = np.sqrt(sd_closed[:, 0] ** 2 + sd_closed[:, 1] ** 2)
        centroid = mean_closed[:-1].mean(axis=0)
        directions = mean_closed - centroid
        norms = np.sqrt((directions ** 2).sum(axis=1, keepdims=True))
        norms[norms == 0] = 1
        unit_dirs = directions / norms
        outer = mean_closed + unit_dirs * radial_sd[:, np.newaxis]
        inner = mean_closed - unit_dirs * radial_sd[:, np.newaxis]
        ring_x = np.concatenate([outer[:, 0], inner[::-1, 0]])
        ring_y = np.concatenate([outer[:, 1], inner[::-1, 1]])

        color = GROUP_COLORS[group]
        ax.fill(ring_x, ring_y, alpha=0.18, color=color, linewidth=0)
        ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                color=color, linewidth=2.2, label=f"{group} (n={len(stacked)})")

    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.3)
    ax.axvline(0, color="gray", linewidth=0.4, alpha=0.3)
    ax.set_xlabel("Normalized X")
    ax.set_ylabel("Normalized Y")
    ax.set_title(title, fontweight="bold", fontsize=11)
    ax.invert_yaxis()
    ax.legend(fontsize=9, loc="lower left")


def build_figure(pooled, svg_path: Path, png_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(9, 13))
    gs = fig.add_gridspec(3, 2, hspace=0.40, wspace=0.28,
                          left=0.08, right=0.97, top=0.95, bottom=0.05)

    # For each role, apply a STRICT "clean toed paw print" filter on each
    # group's paw-like pool, then build a joint template (mean across both
    # groups' strict subsets, with cyclic-shift alignment so perimeter
    # starting points line up). Pick the actual strict-filtered shape per
    # group closest to that template -> visually-matched representatives
    # that both show clean toe detail.

    import numpy as np

    def cyclic_align_to(s, template):
        n = s.shape[0]
        best_sse = np.inf; best_shift = 0
        for k in range(n):
            sse = float(np.sum((np.roll(s, k, axis=0) - template) ** 2))
            if sse < best_sse:
                best_sse = sse; best_shift = k
        return np.roll(s, best_shift, axis=0), best_sse

    def strict_filter(stacked):
        """Drop shapes that don't look like clean toed paw prints."""
        sol, ar, circ = shape_metrics(stacked)
        mask = ((sol <= STRICT_SOLID_MAX)
                & (ar  <= STRICT_ASPECT_MAX)
                & (circ <= STRICT_CIRC_MAX)
                & (circ >= CIRCULARITY_MIN))
        return stacked[mask], np.where(mask)[0]

    def joint_template(pools_after_strict):
        stacked = np.concatenate(pools_after_strict, axis=0)
        mean = stacked.mean(axis=0)
        aligned = np.empty_like(stacked)
        for i in range(len(stacked)):
            aligned[i], _ = cyclic_align_to(stacked[i], mean)
        return aligned.mean(axis=0)

    def closest_to_template(pool, template):
        best_sse = np.inf; best_idx = 0; best_shape = pool[0]
        for i in range(len(pool)):
            shaped, sse = cyclic_align_to(pool[i], template)
            if sse < best_sse:
                best_sse = sse; best_idx = i; best_shape = shaped
        return best_idx, best_shape

    representatives = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        # Per-group strict filter
        strict = {}
        strict_indices = {}
        for group in GROUP_ORDER:
            if group not in pooled[role]:
                continue
            _, shapes_pawlike, _, _ = pooled[role][group]
            sub, idxs_in_pawlike = strict_filter(shapes_pawlike)
            strict[group] = sub
            strict_indices[group] = idxs_in_pawlike
            step(f"  strict {role} {group}: {len(sub)} / "
                 f"{len(shapes_pawlike)} paw-like passed (circ<={STRICT_CIRC_MAX}, "
                 f"sol<={STRICT_SOLID_MAX}, ar<={STRICT_ASPECT_MAX})")

        if not all(len(strict[g]) > 0 for g in strict):
            step(f"  ! strict filter emptied one group for {role}; "
                 f"falling back to the GUI paw-like mean as the template")
            template = pooled[role][GROUP_ORDER[0]][1].mean(axis=0)
        else:
            template = joint_template(list(strict.values()))

        for group in GROUP_ORDER:
            if group not in pooled[role]:
                continue
            _, shapes_pawlike, n_pl, n_total = pooled[role][group]
            pool = strict[group] if len(strict[group]) > 0 else shapes_pawlike
            idxs_origin = (strict_indices[group] if len(strict[group]) > 0
                           else np.arange(len(shapes_pawlike)))
            step(f"  matching {role} {group} to template ({len(pool)} candidates)...")
            sub_idx, shape = closest_to_template(pool, template)
            idx_in_pawlike = int(idxs_origin[sub_idx]) + 1   # 1-indexed
            representatives[role][group] = (idx_in_pawlike, shape, n_pl, n_total,
                                            len(pool))
            step(f"    closest: paw-like #{idx_in_pawlike} of {n_pl} "
                 f"(strict pool of {len(pool)})")

    # Row 0: HL paw-print
    role = "HL"
    for col, group in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(gs[0, col])
        idx, rep, n_pl, n_total, n_strict = representatives[role][group]
        plot_paw_print(
            ax, rep, color=GROUP_COLORS[group],
            title=(f"{group} (#{idx} of {n_pl} paw-like / {n_total} total)\n"
                   f"strict-filter pool n={n_strict}"),
            alpha_fill=1.0,
        )

    # Row 1: HR paw-print
    role = "HR"
    for col, group in enumerate(GROUP_ORDER):
        ax = fig.add_subplot(gs[1, col])
        idx, rep, n_pl, n_total, n_strict = representatives[role][group]
        plot_paw_print(
            ax, rep, color=GROUP_COLORS[group],
            title=(f"{group} (#{idx} of {n_pl} paw-like / {n_total} total)\n"
                   f"strict-filter pool n={n_strict}"),
            alpha_fill=1.0,
        )

    # Row 2: mean contour overlay — uses raw paw-like pools (same as the GUI)
    pawlike_dict_HL = {g: pooled["HL"][g][1] for g in GROUP_ORDER if g in pooled["HL"]}
    pawlike_dict_HR = {g: pooled["HR"][g][1] for g in GROUP_ORDER if g in pooled["HR"]}

    ax_hl = fig.add_subplot(gs[2, 0])
    plot_mean_contour(ax_hl, pawlike_dict_HL, "Mean Contour — Hind Left")
    ax_hr = fig.add_subplot(gs[2, 1])
    plot_mean_contour(ax_hr, pawlike_dict_HR, "Mean Contour — Hind Right")

    # Save vector + raster
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, format="png", dpi=240, bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    pooled = load_groups()

    svg_path = ANALYSIS / "sntx_paw_contour_figure.svg"
    png_path = ANALYSIS / "sntx_paw_contour_figure.png"
    step("Building figure...")
    build_figure(pooled, svg_path, png_path)
    step(f"  -> {svg_path}  ({svg_path.stat().st_size/1024:.0f} KB)")
    step(f"  -> {png_path}  ({png_path.stat().st_size/1024:.0f} KB)")

    # Counts table for the post body
    counts = {role: {g: pooled[role][g][2:4] for g in GROUP_ORDER if g in pooled[role]}
              for role in ("HL", "HR")}
    counts_lines = []
    for role in ("HL", "HR"):
        for g in GROUP_ORDER:
            if g in counts[role]:
                n_pl, n_tot = counts[role][g]
                counts_lines.append(f"  {role} {g}: {n_pl} / {n_tot}")
    head = (
        "**SNTX vs Naïve paw-contour figure** (publication, SVG + PNG)\n"
        "Top two rows: representative paw-print outlines per group "
        "(HL row, HR row). Bottom row: mean ± 1 SD radial-envelope "
        "overlay. SNTX in **red**, Naïve in **black**.\n"
        "Paw-like filter applied (solidity ≤ 1.00, aspect ≤ 1.60, "
        "circularity ≥ 0.10). Counts after filter (paw-like / total):\n"
        + "\n".join(counts_lines)
    )
    discord_upload(head, [svg_path, png_path])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
