"""
Phase-1 vs Phase-2 AUC of Left_licking for the combined FormOxy cohorts.

Formalin-test convention:
  Phase 1 (acute / neurogenic) = 0–10 min post-injection
  Phase 2 (inflammatory)       = 10–60 min post-injection

Reuses the long-form 5-min-bin CSV produced by
`formoxy_licking_timecourse.py`:
  <2605>/analysis/leftlicking_timecourse_5min.csv

Per session AUC = sum of `total_s_licking` across the bins covering each
phase.  Because the bins are uniform 5-min windows the sum is equivalent
to a trapezoid/rectangle-rule integration of the per-frame Left_licking
indicator over the phase, and the units stay in *seconds*.

Output (written under <2605>/analysis/):
  - leftlicking_phase_auc_per_session.csv
  - leftlicking_phase_auc_summary.csv
  - leftlicking_phase_auc_bars.{png,pdf,svg}

Posts the figure + per-session CSV to #results.
"""
from __future__ import annotations

import json
import time
import urllib.request
import uuid
from pathlib import Path

COHORT_NEW = Path(r"E:\RSVIDS\Blackbox\2605_FormOxy_LeftPaws")
ANALYSIS = COHORT_NEW / "analysis"
LONG_CSV = ANALYSIS / "leftlicking_timecourse_5min.csv"

BIN_MIN = 5
PHASE_DEF = {
    "Phase 1 (0–10 min)":  range(0, 2),    # bins 0, 1
    "Phase 2 (10–60 min)": range(2, 12),   # bins 2..11
}
PHASE_DURATION_MIN = {
    "Phase 1 (0–10 min)":  10,
    "Phase 2 (10–60 min)": 50,
}

DOSE_SEQ = ["Veh", "Oxy1", "Oxy3", "Oxy10"]
DOSE_COLORS = {
    "Veh":   "#4A4A4A",
    "Oxy1":  "#90CAF9",
    "Oxy3":  "#1976D2",
    "Oxy10": "#0D47A1",
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
        mime = {".png": "image/png", ".pdf": "application/pdf",
                ".svg": "image/svg+xml", ".csv": "text/csv"}.get(
            p.suffix.lower(), "application/octet-stream"
        )
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
                 "User-Agent": "PixelPaws-FormOxy-PhaseAUC/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            step(f"Discord upload: HTTP {r.status}")
    except Exception as e:
        step(f"Discord upload failed: {e}")


def compute_phase_auc(df):
    """Per-session AUC (seconds licking) for each phase."""
    import pandas as pd
    rows = []
    for (cohort, subj, dose), sub in df.groupby(
        ["cohort", "subject_id", "dose"], sort=False
    ):
        row = {"cohort": cohort, "subject_id": subj, "dose": dose}
        for phase_name, bin_range in PHASE_DEF.items():
            vals = sub.loc[sub["bin_idx"].isin(list(bin_range)),
                           "total_s_licking"].to_numpy()
            row[phase_name] = float(vals.sum()) if len(vals) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def one_way_anova(per_sess, phase_name):
    """Return F, p for a 1-way ANOVA across DOSE_SEQ for a given phase."""
    from scipy import stats
    groups = [per_sess.loc[per_sess["dose"] == d, phase_name].dropna().to_numpy()
              for d in DOSE_SEQ]
    if any(len(g) < 2 for g in groups):
        return float("nan"), float("nan")
    F, p = stats.f_oneway(*groups)
    return float(F), float(p)


def dunnett_vs_veh(per_sess, phase_name):
    """Dunnett's test vs Veh (per dose: t, p). Falls back to Welch's t-test
    with no correction if scikit-posthocs / scipy.stats.dunnett is missing."""
    from scipy import stats
    veh = per_sess.loc[per_sess["dose"] == "Veh", phase_name].dropna().to_numpy()
    out = {}
    if hasattr(stats, "dunnett"):
        samples = [per_sess.loc[per_sess["dose"] == d, phase_name].dropna().to_numpy()
                   for d in DOSE_SEQ if d != "Veh"]
        try:
            res = stats.dunnett(*samples, control=veh)
            for d, t, p in zip(["Oxy1", "Oxy3", "Oxy10"], res.statistic, res.pvalue):
                out[d] = (float(t), float(p))
            return out
        except Exception as e:
            step(f"  Dunnett failed ({e}); falling back to Welch t-tests")
    for d in ["Oxy1", "Oxy3", "Oxy10"]:
        vals = per_sess.loc[per_sess["dose"] == d, phase_name].dropna().to_numpy()
        if len(vals) >= 2 and len(veh) >= 2:
            t, p = stats.ttest_ind(vals, veh, equal_var=False)
            out[d] = (float(t), float(p))
        else:
            out[d] = (float("nan"), float("nan"))
    return out


def plot_phase_bars(per_sess, out_stem: Path):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams["pdf.fonttype"] = 42
    matplotlib.rcParams["ps.fonttype"]  = 42
    matplotlib.rcParams["svg.fonttype"] = "none"
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(0)

    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.6),
                             constrained_layout=True, sharey=False)
    phase_names = list(PHASE_DEF.keys())

    summary_rows = []
    anova_text = []

    has_sex_split = per_sess["cohort"].str.contains(
        r"\(M\)|\(F\)", regex=True).any()

    for ax_idx, (ax, phase_name) in enumerate(zip(axes, phase_names)):
        x_pos = np.arange(len(DOSE_SEQ))
        means, sems, ns = [], [], []
        for j, dose in enumerate(DOSE_SEQ):
            sub = per_sess[(per_sess["dose"] == dose) &
                           per_sess[phase_name].notna()]
            vals = sub[phase_name].astype(float).to_numpy()
            n = len(vals)
            m = float(vals.mean()) if n else float("nan")
            s = float(vals.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            means.append(m); sems.append(s); ns.append(n)
            c = DOSE_COLORS[dose]
            ax.bar(j, m, yerr=s, color=c, alpha=0.55,
                   edgecolor="black", linewidth=0.6,
                   error_kw=dict(ecolor="black", lw=1.0, capsize=4),
                   zorder=2)
            if has_sex_split:
                males = sub[sub["cohort"].str.contains(
                    r"\(M\)", regex=True)][phase_name].astype(float).to_numpy()
                females = sub[sub["cohort"].str.contains(
                    r"\(F\)", regex=True)][phase_name].astype(float).to_numpy()
                if len(males):
                    jitter_m = rng.uniform(-0.13, 0.13, size=len(males))
                    ax.scatter(np.full(len(males), j) + jitter_m, males,
                               c=c, s=28, edgecolors="black",
                               linewidths=0.7, zorder=3)
                if len(females):
                    jitter_f = rng.uniform(-0.13, 0.13, size=len(females))
                    ax.scatter(np.full(len(females), j) + jitter_f, females,
                               facecolors="none", edgecolors=c,
                               s=28, linewidths=1.2, zorder=3)
            else:
                jitter = rng.uniform(-0.13, 0.13, size=n)
                ax.scatter(np.full(n, j) + jitter, vals,
                           c=c, s=28, edgecolors="black",
                           linewidths=0.7, zorder=3)
            summary_rows.append({
                "phase": phase_name, "dose": dose, "n": n,
                "mean_s": round(m, 2) if n else float("nan"),
                "sem_s":  round(s, 2) if n > 1 else 0.0,
            })

        F, p = one_way_anova(per_sess, phase_name)
        anova_text.append(f"{phase_name}: F={F:.2f}, p={p:.3g}")
        dunnett = dunnett_vs_veh(per_sess, phase_name)
        for d, (t, p_d) in dunnett.items():
            anova_text.append(f"  {d} vs Veh: t={t:.2f}, p={p_d:.3g}")

        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"{d}\n(n={n})" for d, n in zip(DOSE_SEQ, ns)],
                           fontsize=9)
        ax.set_ylabel("Total time licking (s)")
        ax.set_title(f"{phase_name}\n(ANOVA F={F:.2f}, p={p:.3g})",
                     fontsize=10, pad=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_ylim(bottom=0)

        # Legend on the first panel only (upper-right is empty: Oxy10≈0)
        if ax_idx == 0 and has_sex_split:
            sex_handles = [
                plt.Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="0.4", markeredgecolor="black",
                           markeredgewidth=0.7, markersize=7,
                           label="Male"),
                plt.Line2D([0], [0], marker="o", color="none",
                           markerfacecolor="none", markeredgecolor="0.4",
                           markeredgewidth=1.2, markersize=7,
                           label="Female"),
            ]
            ax.legend(handles=sex_handles, fontsize=8, loc="upper right",
                      frameon=False)

    out_png = out_stem.with_suffix(".png")
    out_pdf = out_stem.with_suffix(".pdf")
    out_svg = out_stem.with_suffix(".svg")
    fig.savefig(out_png, format="png", dpi=180, bbox_inches="tight")
    fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
    fig.savefig(out_svg, format="svg", bbox_inches="tight")
    plt.close(fig)

    for line in anova_text:
        step(line)

    import pandas as pd
    return (out_png, out_pdf, out_svg, pd.DataFrame(summary_rows), anova_text)


def main() -> int:
    import pandas as pd

    if not LONG_CSV.exists():
        step(f"missing {LONG_CSV} — run formoxy_licking_timecourse.py first")
        return 1
    df = pd.read_csv(LONG_CSV)
    step(f"loaded {len(df):,} rows from {LONG_CSV.name}")

    per_sess = compute_phase_auc(df)
    per_sess_csv = ANALYSIS / "leftlicking_phase_auc_per_session.csv"
    per_sess.to_csv(per_sess_csv, index=False)
    step(f"per-session AUC: {per_sess_csv}")

    base_stem = ANALYSIS / "leftlicking_phase_auc_bars"
    png, pdf, svg, summary_df, anova_text = plot_phase_bars(per_sess, base_stem)
    summary_csv = ANALYSIS / "leftlicking_phase_auc_summary.csv"
    summary_df.to_csv(summary_csv, index=False)
    step(f"summary:        {summary_csv}")
    step(f"figures:        {png.name}, {pdf.name}, {svg.name}")

    msg = (
        "**FormOxy Left_licking — Phase-1 vs Phase-2 AUC** "
        "(formalin-test convention: P1 = 0–10 min acute, "
        "P2 = 10–60 min inflammatory).  Pooled **2512 (M, n=12) + "
        "2605 (F, n=12) = n=6/dose**.  Bars = mean ± SEM, "
        "dots = per-mouse AUC (sum of total-s-licking across bins).\n"
        + "\n".join(f"• {line}" for line in anova_text)
    )
    discord_upload(msg, [png, pdf, svg, per_sess_csv, summary_csv])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
