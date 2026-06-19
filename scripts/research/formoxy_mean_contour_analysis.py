"""
Replicate the GUI Gait & Limb -> "Mean Contour -- Hind Left" plot for
both formalin+oxy cohorts pooled (2512 + 2605).

For each dose group:
  - load all per-session normalized contour shapes (npz files) from
    2512_.../gait_limb_analysis/*_contour_*_shapes.npz and
    2605_.../gait_limb_analysis/*_contour_*_shapes.npz
  - pool by dose (each session contributes <=500 shapes for HL and HR)
  - random-subsample to N_PER_DOSE for visual parity with the GUI
  - mean contour shape over the pool, +/- 1 SD radial envelope

Plotting follows GaitLimbTab._add_contour_shape_tab in gait_limb_tab.py:
mean outline + SD-band filled ring, y-axis inverted (image coordinates).

Output: HL + HR side-by-side panels for the combined cohort, posted to
the #results Discord channel.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from pathlib import Path

COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
NEW_BRT = COHORT_NEW / "gait_limb_analysis"
ANALYSIS = COHORT_NEW / "analysis"

COHORT_OLD = Path(r"E:\RSVIDS\Blackbox\2512_Blackbox_Formalin_Oxy\Left_paws")
OLD_BRT = COHORT_OLD / "gait_limb_analysis"

N_PER_DOSE = 1500       # to match the GUI screenshot legend
RNG_SEED = 17

DOSE_2512 = {1: "Veh", 2: "Veh", 3: "Veh",
             4: "Oxy3", 5: "Oxy3", 6: "Oxy3",
             7: "Oxy10", 8: "Oxy10", 9: "Oxy10",
             10: "Oxy1", 11: "Oxy1", 12: "Oxy1"}
DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]


def dose_2605(n: int) -> str:
    return DOSE_SEQ[(n - 1) % 4]


# Same blue palette as the GUI screenshot (Veh black, Oxy doses light->dark)
DOSE_COLORS = {
    "Veh":   "#000000",
    "Oxy1":  "#a6cee3",
    "Oxy3":  "#1f78b4",
    "Oxy10": "#08306b",
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
        mime = "image/png" if p.suffix == ".png" else "text/csv"
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
                 "User-Agent": "PixelPaws-FormOxy-Contour/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def collect():
    """Return {role: {dose: stacked_shapes_array (N,64,2)}}."""
    import numpy as np

    pooled = {"HL": {d: [] for d in DOSE_SEQ},
              "HR": {d: [] for d in DOSE_SEQ}}

    # 2605
    rx_new = re.compile(r"^2605_FormOxy_S(\d+)_formalin_contour_[0-9a-f]+_shapes\.npz$")
    for p in sorted(NEW_BRT.glob("2605_FormOxy_S*_formalin_contour_*_shapes.npz")):
        m = rx_new.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        dose = dose_2605(subj)
        with np.load(p) as data:
            for role in ("HL", "HR"):
                if role in data:
                    pooled[role][dose].append(data[role])
        step(f"  2605 S{subj} ({dose}): "
             f"HL={len(pooled['HL'][dose][-1]) if pooled['HL'][dose] else 0} "
             f"HR={len(pooled['HR'][dose][-1]) if pooled['HR'][dose] else 0}")

    # 2512
    rx_old = re.compile(r"^2512_FormOxy_S(\d+)_contour_[0-9a-f]+_shapes\.npz$")
    for p in sorted(OLD_BRT.glob("2512_FormOxy_S*_contour_*_shapes.npz")):
        m = rx_old.match(p.name)
        if not m:
            continue
        subj = int(m.group(1))
        dose = DOSE_2512[subj]
        with np.load(p) as data:
            for role in ("HL", "HR"):
                if role in data:
                    pooled[role][dose].append(data[role])
        step(f"  2512 S{subj} ({dose}): "
             f"HL={len(pooled['HL'][dose][-1]) if pooled['HL'][dose] else 0} "
             f"HR={len(pooled['HR'][dose][-1]) if pooled['HR'][dose] else 0}")

    # Concat per dose
    rng = np.random.default_rng(RNG_SEED)
    out = {"HL": {}, "HR": {}}
    for role in ("HL", "HR"):
        for d in DOSE_SEQ:
            chunks = pooled[role][d]
            if not chunks:
                continue
            stacked = np.concatenate(chunks, axis=0)
            if len(stacked) > N_PER_DOSE:
                idx = rng.choice(len(stacked), N_PER_DOSE, replace=False)
                stacked = stacked[idx]
            out[role][d] = stacked
    return out


def plot_mean_contours(per_role: dict, out_png: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paw_label = {"HL": "Hind Left", "HR": "Hind Right"}

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)

    # Track combined axis ranges so both panels share the same x/y scale
    all_pts = []

    for ax, role in zip(axes, ("HL", "HR")):
        ax.set_aspect("equal")
        for dose in DOSE_SEQ:
            if dose not in per_role[role]:
                continue
            stacked = per_role[role][dose]
            if len(stacked) == 0:
                continue
            mean_shape = stacked.mean(axis=0)
            sd_shape = stacked.std(axis=0, ddof=1) if len(stacked) > 1 else np.zeros_like(mean_shape)

            # Close polygon for plotting
            mean_closed = np.vstack([mean_shape, mean_shape[0:1]])
            sd_closed = np.vstack([sd_shape, sd_shape[0:1]])

            color = DOSE_COLORS[dose]
            ax.plot(mean_closed[:, 0], mean_closed[:, 1],
                    color=color, linewidth=2.0,
                    label=f"{dose} (n={len(stacked)})")

            # SD-band ring (matches gait_limb_tab.py:9074-9088)
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
            ax.fill(ring_x, ring_y, alpha=0.15, color=color)
            all_pts.extend([outer, inner])

        ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
        ax.set_title(f"Mean Contour — {paw_label[role]}", fontweight="bold")
        ax.set_xlabel("Normalized X")
        ax.set_ylabel("Normalized Y")
        ax.legend(fontsize=9, loc="upper right")
        ax.invert_yaxis()  # image-coordinate convention (matches the GUI)
        ax.grid(alpha=0.3)

    # Equalize x/y limits across both panels for a fair side-by-side
    if all_pts:
        cat = np.concatenate([np.vstack(all_pts)], axis=0)
        x_min, y_min = cat.min(axis=0)
        x_max, y_max = cat.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min)
        for ax in axes:
            ax.set_xlim(x_min - pad, x_max + pad)
            # invert_yaxis already called -> set y limits in flipped order
            ax.set_ylim(y_max + pad, y_min - pad)

    fig.suptitle(
        "FormOxy: mean paw contour by dose (combined 2512 + 2605)\n"
        "centred + scaled by sqrt(area); ribbon = +/- 1 SD radial",
        fontweight="bold", fontsize=12,
    )
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    step("Collecting contour shapes...")
    per_role = collect()

    # Counts table
    rows = []
    for role in ("HL", "HR"):
        for d in DOSE_SEQ:
            n = len(per_role[role].get(d, []))
            rows.append({"role": role, "dose": d, "n_shapes_used": n})
            step(f"  pooled {role} {d}: n={n}")

    import pandas as pd
    counts_csv = ANALYSIS / "mean_contour_counts.csv"
    pd.DataFrame(rows).to_csv(counts_csv, index=False)

    out_png = ANALYSIS / "mean_contour_HL_HR.png"
    plot_mean_contours(per_role, out_png)
    step(f"Figure: {out_png}")

    head = (
        "**FormOxy mean paw contour (combined 2512 + 2605)**\n"
        f"Hind Left (formalin-injected) and Hind Right (contralateral control), "
        f"per-dose mean +/- 1 SD radial envelope. Up to {N_PER_DOSE} shapes per "
        f"dose (random subsample when more pooled). "
        f"50 px contour ROI, Otsu threshold, normalized centroid + sqrt(area)."
    )
    discord_upload(head, [out_png, counts_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
