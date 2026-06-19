"""Render frame-only smoothed precision/recall/F1 threshold curve from
the v2 sweep CSV (no bout curve, cleaner)."""
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
from scipy.signal import savgol_filter

CSV = Path(r"E:/RSVIDS/Blackbox/2510_Blackbox_Rimonabant/Blackbox_videos-selected/"
           r"Classifiers/PixelPaws_Scratching_honest_20260521_160523/plots/"
           r"scratching_honest_threshold_curve_pruned120_v2.csv")
OUT_DIR = CSV.parent

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
                ".svg": "image/svg+xml"}.get(p.suffix.lower(),
                                              "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-FrameOnlySmoothed/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r:
        step(f"  HTTP {r.status}")


def main():
    df = pd.read_csv(CSV)
    thr = df["threshold"].to_numpy()
    p = df["frame_precision"].to_numpy()
    r = df["frame_recall"].to_numpy()
    f = df["frame_f1"].to_numpy()

    win, poly = 21, 3
    sm = lambda v: savgol_filter(v, window_length=win, polyorder=poly)
    f_sm = sm(f)
    p_sm = sm(p)
    r_sm = sm(r)

    best_i = int(np.argmax(f))
    step(f"frame F1 raw best = {f[best_i]:.3f} at thresh = {thr[best_i]:.3f}")
    sm_best_i = int(np.argmax(f_sm))
    step(f"frame F1 smoothed best = {f_sm[sm_best_i]:.3f} at thresh = {thr[sm_best_i]:.3f}")

    fig, ax = plt.subplots(figsize=(10.0, 6.0))
    ax.plot(thr, r_sm, color="#f5a623", linewidth=2.4, label="Recall")
    ax.plot(thr, p_sm, color="#d12c80", linewidth=2.4, label="Precision")
    ax.plot(thr, f_sm, color="#5b2c91", linewidth=2.8, label="F1 Score")

    # Mark F1-optimal threshold (from raw best, but plot peak from smoothed)
    ax.axvline(thr[best_i], color="#888", linestyle=":", linewidth=1.2, alpha=0.8)
    ax.scatter([thr[best_i]], [f[best_i]], color="#5b2c91", s=70,
               edgecolor="white", zorder=5)
    ax.annotate(f"best F1 = {f[best_i]:.2f}\nthresh = {thr[best_i]:.3f}",
                xy=(thr[best_i], f[best_i]),
                xytext=(thr[best_i] + 0.10, f[best_i] - 0.18),
                fontsize=10, ha="left", va="top",
                arrowprops=dict(arrowstyle="->", color="#5b2c91", lw=1.0),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#cccccc"))

    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=True, fontsize=10)
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    out_base = OUT_DIR / "scratching_honest_threshold_curve_pruned120_frame_smoothed"
    fig.savefig(out_base.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    step(f"wrote {out_base}.png / svg / pdf")

    head = ("**Scratching threshold curve -- frame-level only, smoothed**\n"
            "Precision / recall / frame F1 with Savitzky-Golay smoothing "
            f"(window={win}, polyorder={poly}). Honest pooled session-level "
            f"GroupKFold OOF from the pruned-120 model. Frame F1 peaks at "
            f"{f[best_i]:.3f} at thresh = {thr[best_i]:.3f}.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf")])


if __name__ == "__main__":
    main()
