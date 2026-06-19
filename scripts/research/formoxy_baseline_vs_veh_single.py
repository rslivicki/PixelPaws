"""Single-subject baseline-vs-formalin (Veh) HL/HR contour-intensity-ratio
timecourse, 5-min bins.

Subject: 2605_S1 (Vehicle, has both baseline + formalin contour CSVs).

The baseline session is plotted at negative time (so t=0 is the formalin
injection moment); formalin runs from 0 -> ~60 min. A vertical line at
t=0 marks injection. No gridlines, clean styling, mean HL/HR ratio per
5-min bin with n_valid frames as transparency for the dot."""
from __future__ import annotations
import json, time, urllib.request, uuid
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COHORT = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
GLAB   = COHORT / "gait_limb_analysis"
OUT    = COHORT / "analysis"

# Two subjects: S1 (Veh) and S4 (Oxy10) in 2605, both have baseline + formalin
SUBJECTS = [
    {"label": "2605_S1 (Veh)",   "color": "#000000", "subject": "S1"},
    {"label": "2605_S4 (Oxy10)", "color": "#08306b", "subject": "S4"},
]

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-BaselineVsVehSingle/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        step(f"  HTTP {r.status}")


def bin_session(csv_path: Path):
    """Per 5-min bin: mean(intensities_HL) / mean(intensities_HR) over
    frames where both contours were detected."""
    df = pd.read_csv(csv_path, usecols=["intensities_HL", "intensities_HR"])
    hl = df["intensities_HL"].astype(float).to_numpy()
    hr = df["intensities_HR"].astype(float).to_numpy()
    n_frames = len(df)
    n_bins = (n_frames + BIN_FRAMES - 1) // BIN_FRAMES
    rows = []
    for i in range(n_bins):
        s = i * BIN_FRAMES
        e = min(s + BIN_FRAMES, n_frames)
        slhl = hl[s:e]; slhr = hr[s:e]
        valid = (slhl > 0) & (slhr > 0) & np.isfinite(slhl) & np.isfinite(slhr)
        n_valid = int(valid.sum())
        if n_valid == 0:
            ratio = np.nan
        else:
            mhl = float(np.mean(slhl[valid])); mhr = float(np.mean(slhr[valid]))
            ratio = mhl / mhr if mhr > 0 else np.nan
        rows.append({"bin_idx": i, "bin_min_start": i * BIN_MIN,
                     "n_valid_frames": n_valid, "ratio_HL_HR": ratio})
    step(f"  {csv_path.name}: {n_frames} frames -> {len(rows)} bins")
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    per_subject = {}
    long = []
    for spec in SUBJECTS:
        subj = spec["subject"]
        base_csv = next(GLAB.glob(f"2605_FormOxy_{subj}_baseline_contour_*.csv"))
        form_csv = next(GLAB.glob(f"2605_FormOxy_{subj}_formalin_contour_*.csv"))
        step(f"{spec['label']}  baseline: {base_csv.name}")
        base_rows = bin_session(base_csv)
        step(f"{spec['label']}  formalin: {form_csv.name}")
        form_rows = bin_session(form_csv)

        n_base = len(base_rows)
        base_x = [(i - n_base) * BIN_MIN for i in range(n_base)]
        base_y = [r["ratio_HL_HR"] for r in base_rows]
        base_n = [r["n_valid_frames"] for r in base_rows]
        form_x = [i * BIN_MIN for i in range(len(form_rows))]
        form_y = [r["ratio_HL_HR"] for r in form_rows]
        form_n = [r["n_valid_frames"] for r in form_rows]

        for x, y, n in zip(base_x, base_y, base_n):
            long.append({"subject": spec["label"], "phase": "baseline",
                          "time_min_start": x, "n_valid_frames": n,
                          "ratio_HL_HR": y})
        for x, y, n in zip(form_x, form_y, form_n):
            long.append({"subject": spec["label"], "phase": "formalin",
                          "time_min_start": x, "n_valid_frames": n,
                          "ratio_HL_HR": y})

        per_subject[spec["label"]] = {
            "color": spec["color"],
            "base_x": base_x, "base_y": base_y,
            "form_x": form_x, "form_y": form_y,
        }

    csv_out = OUT / "2605_baseline_vs_formalin_ratio_tc_S1_S4.csv"
    pd.DataFrame(long).to_csv(csv_out, index=False)
    step(f"wrote {csv_out.name}")

    fig, ax = plt.subplots(figsize=(11.0, 5.6))

    # Reference line at ratio = 1.0
    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1.0, alpha=0.7)
    # Injection marker (single dashed red line shared across subjects)
    ax.axvline(0, color="#d62728", linestyle="--", linewidth=1.2, alpha=0.8,
               label="Injection (t=0)")

    for label, d in per_subject.items():
        col = d["color"]
        # Baseline: lighter / dashed line, hollow markers
        ax.plot(d["base_x"], d["base_y"], color=col, linewidth=1.4,
                marker="o", markersize=5, markerfacecolor="white",
                markeredgecolor=col, linestyle="--", alpha=0.85,
                label=f"{label}  baseline", zorder=2)
        # Formalin: solid line, filled markers
        ax.plot(d["form_x"], d["form_y"], color=col, linewidth=2.0,
                marker="o", markersize=6, markerfacecolor=col,
                markeredgecolor=col, label=f"{label}  formalin", zorder=3)

    ax.set_xlabel("Time relative to formalin injection (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_title("Baseline vs formalin -- single-subject HL/HR contour intensity TC",
                 fontweight="bold")
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="lower right", frameon=True, fontsize=8.5, ncol=1)

    all_x = sum([d["base_x"] + d["form_x"] for d in per_subject.values()], [])
    tick_step = 10
    xmin = (min(all_x) // tick_step) * tick_step
    xmax = ((max(all_x) // tick_step) + 1) * tick_step
    ax.set_xticks(list(range(int(xmin), int(xmax) + 1, tick_step)))
    ax.set_xlim(min(all_x) - 1, max(all_x) + 1)

    fig.tight_layout()
    out_base = OUT / "2605_baseline_vs_formalin_ratio_tc_S1_S4"
    fig.savefig(out_base.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    head = ("**Baseline vs formalin -- single-subject HL/HR contour "
            "intensity ratio TC (Veh + Oxy10)**\n"
            "Two subjects: S1 (Veh, black) and S4 (Oxy10, dark blue). For "
            "each: baseline session at negative time (dashed line + hollow "
            "markers), formalin session at positive time (solid line + "
            "filled markers). Red dashed = formalin injection moment, grey "
            "dotted = ratio 1.0 reference. 5-min bins, `mean(HL)/mean(HR)` "
            "over frames where both contours were detected. No grid.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          csv_out])


if __name__ == "__main__":
    main()
