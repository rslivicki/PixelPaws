"""
SHAM vs SNI 5-min-bin HL/HR contour-intensity-ratio timecourse.

For each `*_contour_<hash>.csv` in 2603_SNI_PG / weight_bearing_analysis:
  1. Read `intensities_HL` + `intensities_HR`.
  2. Bin into 5-min windows (5 min x 60 s x 60 fps = 18 000 frames),
     drop frames where either contour was not detected (intensity == 0
     or non-finite).
  3. Per bin, compute `mean(HL) / mean(HR)` (aggregate-first, ratio-
     second -- matches the existing GUI + FormOxy TC convention).
4 sessions -> 2 SHAM + 2 SNI, ~63 min recordings.

Renders:
  * 5-min-bin TC plot, SHAM (green) vs SNI (orange) mean +/- SEM
  * per-session line overlay (thin, group-colour)
  * Mann-Whitney U per-bin marker
  * per-bin summary CSV
"""
from __future__ import annotations
import json, re, sys, time, urllib.request, uuid
from pathlib import Path

REPO = Path(r"E:\PixelPaws"); sys.path.insert(0, str(REPO))

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"]  = 42
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COHORT = Path(r"E:\RSVIDS\Blackbox\2603_SNI_PG")
WBA    = COHORT / "weight_bearing_analysis"
KEY    = COHORT / "key_file.csv"
OUT    = COHORT / "analysis"

FPS = 60
BIN_MIN = 5
BIN_FRAMES = BIN_MIN * 60 * FPS
N_BINS = 12   # 0-60 min

GROUP_ORDER = ["SHAM", "SNI"]
GROUP_COLORS = {"SHAM": "#2ca02c", "SNI": "#ff7f0e"}

WEBHOOK = ("")


def step(m): print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)


def discord_upload(content, files):
    boundary = "----PP" + uuid.uuid4().hex
    body = [f"--{boundary}".encode(),
            b'Content-Disposition: form-data; name="payload_json"',
            b'Content-Type: application/json', b"",
            json.dumps({"content": content}).encode()]
    for i, p in enumerate(files):
        mime = {".png": "image/png", ".svg": "image/svg+xml",
                ".pdf": "application/pdf", ".csv": "text/csv"
                }.get(p.suffix.lower(), "application/octet-stream")
        body += [f"--{boundary}".encode(),
                 f'Content-Disposition: form-data; name="file{i}"; filename="{p.name}"'.encode(),
                 f"Content-Type: {mime}".encode(), b"",
                 p.read_bytes()]
    body.append(f"--{boundary}--".encode())
    req = urllib.request.Request(
        WEBHOOK, data=b"\r\n".join(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "PixelPaws-SNI-ContourRatioTC/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"  HTTP {r.status}")
    except Exception as e:
        step(f"  upload failed: {e}")


def bin_session(df: pd.DataFrame) -> list[dict]:
    out = []
    hl_full = df["intensities_HL"].astype(float).to_numpy()
    hr_full = df["intensities_HR"].astype(float).to_numpy()
    for i in range(N_BINS):
        s = i * BIN_FRAMES
        e = s + BIN_FRAMES
        hl = hl_full[s:e]; hr = hr_full[s:e]
        valid = (hl > 0) & (hr > 0) & np.isfinite(hl) & np.isfinite(hr)
        n_valid = int(valid.sum())
        if n_valid == 0:
            mean_hl = mean_hr = ratio = float("nan")
        else:
            mean_hl = float(np.mean(hl[valid]))
            mean_hr = float(np.mean(hr[valid]))
            ratio = mean_hl / mean_hr if mean_hr > 0 else float("nan")
        out.append({"bin_idx": i, "bin_min_start": i * BIN_MIN,
                    "n_valid_frames": n_valid,
                    "contour_HL_mean": mean_hl,
                    "contour_HR_mean": mean_hr,
                    "ratio_HL_HR": ratio})
    return out


def collect():
    df_key = pd.read_csv(KEY).dropna()
    mouse_to_group = {row["Subject"]: row["Treatment"]
                      for _, row in df_key.iterrows()
                      if row["Treatment"] in GROUP_ORDER}
    rx = re.compile(r"^(?P<subj>[^.]+)_contour_[0-9a-f]+\.csv$")
    rows = []
    for p in sorted(WBA.glob("*_contour_*.csv")):
        m = rx.match(p.name)
        if not m:
            continue
        subj_raw = m.group("subj")
        match = next((mid for mid in mouse_to_group if mid in subj_raw), None)
        if not match:
            step(f"  ! no key match for {p.name}"); continue
        group = mouse_to_group[match]
        step(f"  reading {p.name} -> {group}")
        df = pd.read_csv(p, usecols=["intensities_HL", "intensities_HR"])
        for b in bin_session(df):
            rows.append({"subject": match, "group": group, **b})
    return pd.DataFrame(rows)


def sig_marker(p):
    if not np.isfinite(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return ""


def per_bin_mwu(df):
    from scipy import stats
    pvals = []
    for b in range(N_BINS):
        a = df.loc[(df["bin_idx"] == b) & (df["group"] == "SHAM"),
                   "ratio_HL_HR"].dropna().to_numpy()
        s = df.loc[(df["bin_idx"] == b) & (df["group"] == "SNI"),
                   "ratio_HL_HR"].dropna().to_numpy()
        if len(a) < 2 or len(s) < 2:
            pvals.append(float("nan")); continue
        try:
            _, p = stats.mannwhitneyu(a, s, alternative="two-sided")
            pvals.append(float(p))
        except Exception:
            pvals.append(float("nan"))
    return pvals


def plot_tc(df, out_path: Path, perbin_p):
    x = [i * BIN_MIN for i in range(N_BINS)]
    fig, ax = plt.subplots(figsize=(10.0, 5.6))

    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        c = GROUP_COLORS[g]
        # per-session faint traces
        for subj, sg in sub.groupby("subject"):
            sg = sg.sort_values("bin_idx")
            ax.plot(x[:len(sg)], sg["ratio_HL_HR"].to_numpy(),
                    color=c, alpha=0.25, linewidth=1.0, zorder=1)
        # group mean +/- SEM
        means, sems = [], []
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            means.append(np.mean(vals) if len(vals) else np.nan)
            sems.append(np.std(vals, ddof=1)/np.sqrt(len(vals))
                        if len(vals) > 1 else 0.0)
        ax.errorbar(x, means, yerr=sems, marker="o", color=c,
                    linewidth=2.0, markersize=6, capsize=3,
                    label=f"{g} (n={sub['subject'].nunique()})", zorder=3)

    ax.axhline(1.0, color="#d62728", linestyle="--", linewidth=1.0, alpha=0.7)
    y_top = ax.get_ylim()[1]
    for i, p in enumerate(perbin_p):
        marker = sig_marker(p)
        if marker:
            ax.annotate(marker, xy=(x[i], y_top * 0.985),
                        ha="center", va="top", fontsize=10)

    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Contour intensity ratio HL/HR")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{i*BIN_MIN}-{(i+1)*BIN_MIN}" for i in range(N_BINS)],
                       rotation=0, fontsize=8)
    ax.legend(loc="best", frameon=True, fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    fig.savefig(out_path.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = collect()
    if df.empty:
        step("! no contour CSVs collected"); return 1
    step(f"Collected {len(df)} bin-rows across {df['subject'].nunique()} sessions")
    step(f"groups: " + ", ".join(f"{g}={df.loc[df['group']==g,'subject'].nunique()}"
                                 for g in GROUP_ORDER))

    long_csv = OUT / "sni_contour_intensity_ratio_tc_long.csv"
    df.to_csv(long_csv, index=False)

    perbin_p = per_bin_mwu(df)
    step("Per-bin Mann-Whitney U (SHAM vs SNI):")
    for i, p in enumerate(perbin_p):
        step(f"  bin {i*BIN_MIN:>2}-{(i+1)*BIN_MIN:>2} min: p={p:.3g} {sig_marker(p)}")

    out_base = OUT / "sni_contour_intensity_ratio_tc"
    plot_tc(df, out_base, perbin_p)

    # Compact per-group-per-bin summary
    rows = []
    for g in GROUP_ORDER:
        sub = df[df["group"] == g]
        for b in range(N_BINS):
            vals = sub.loc[sub["bin_idx"] == b, "ratio_HL_HR"].dropna().to_numpy()
            rows.append({
                "group": g, "bin_min_start": b * BIN_MIN, "n": int(len(vals)),
                "mean": round(float(np.mean(vals)), 4) if len(vals) else float("nan"),
                "sem":  round(float(np.std(vals, ddof=1)/np.sqrt(len(vals))), 4)
                        if len(vals) > 1 else 0.0,
                "p_mwu_at_bin": round(perbin_p[b], 4) if np.isfinite(perbin_p[b]) else float("nan"),
                "sig_at_bin": sig_marker(perbin_p[b]),
            })
    summary_csv = OUT / "sni_contour_intensity_ratio_tc_summary.csv"
    pd.DataFrame(rows).to_csv(summary_csv, index=False)

    head = ("**SNI vs SHAM contour intensity ratio HL/HR -- 5-min-bin TC**\n"
            "Mean intensity inside the Otsu-thresholded paw contour, per-bin "
            f"`mean(HL) / mean(HR)`. {BIN_MIN}-min bins, 0-{N_BINS*BIN_MIN} min, "
            "per-bin asterisks = Mann-Whitney U (SHAM vs SNI). Thin traces = "
            "individual sessions; thick markers = group mean +/- SEM. Red dashed "
            "line = symmetric ratio (1.0). PNG + SVG + PDF + per-bin summary CSV.")
    discord_upload(head, [out_base.with_suffix(".png"),
                          out_base.with_suffix(".svg"),
                          out_base.with_suffix(".pdf"),
                          summary_csv, long_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
