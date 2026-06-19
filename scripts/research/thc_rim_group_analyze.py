"""
Group-level analysis for the 260512 THC rimonabant cohort.

Reads <PROJECT>/results/<session>_predictions.csv, applies argmax
assignment (PRIORITY = facial_grooming > left_licking > rearing >
still > walking), and emits:

  - per-session state-occupancy CSV (% time per state)
  - per-(chronic, acute, condition) transition matrices
  - figures: immobility, state occupancy, transitions + diffs, bout
    structure, ethograms

Design (2x2 + pre/post):
  chronic  = odd-numbered mouse  -> THC
             even-numbered mouse -> Vehicle
  acute    = cycle of 2 over LIVING mouse sequence (m8 + m9 died).
             Living order: 1,2,3,4,5,6,7,10,11,12,13,14,15.
             -> m1,2=Rim; m3,4=Veh; m5,6=Rim; m7,10=Veh;
                m11,12=Rim; m13,14=Veh; m15=Rim.
  condition = pre-rimonabant     -> 30 min baseline
              post-rimonabant    -> 60 min challenge

Window caps: pre = 30 min, post = 60 min @ 60 fps.

Run:
  PYTHONIOENCODING=utf-8 py -X utf8 scripts/research/thc_rim_group_analyze.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"E:\RSVIDS\Blackbox\260512_THC_Rim_Cohort")
RESULTS_DIR = PROJECT_ROOT / "results"
ANALYSIS_DIR = PROJECT_ROOT / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

FPS = 60
CAP_PRE_MIN = 30
CAP_POST_MIN = 60
PRIORITY = ["Facial_grooming", "Left_licking", "rearing", "still", "walking"]
NOISE_LABEL = "noise"
EXCLUDE_NOISE = True

WEBHOOK = (
    ""
)

CHRONIC_PALETTE = {"THC": "#d62728", "Vehicle": "#1f77b4"}
ACUTE_PALETTE = {"Rim": "#9467bd", "Veh": "#7f7f7f"}


DEAD_MICE = {8, 9}
_LIVING = [n for n in range(1, 16) if n not in DEAD_MICE]


def acute_for_mouse(n: int) -> str:
    """Cycle of 2 over the *living* mouse sequence (8 and 9 died, skipped).

    Living order: 1,2,3,4,5,6,7,10,11,12,13,14,15.
    Pairs in that order alternate Rim,Rim,Veh,Veh,...
    -> m1,2=Rim; m3,4=Veh; m5,6=Rim; m7,10=Veh; m11,12=Rim; m13,14=Veh; m15=Rim.
    """
    if n in DEAD_MICE or n not in _LIVING:
        return "Unknown"
    pos = _LIVING.index(n)  # 0-indexed
    return "Rim" if (pos // 2) % 2 == 0 else "Veh"


def parse_session(name: str):
    """{pre|post}_rimonabant_THC<n>[_YYYYMMDD] -> (n, condition, chronic, acute)."""
    m = re.match(r"^(pre|post)_rimonabant_THC(\d+)(?:_\d+)?$", name)
    if not m:
        return None
    cond = m.group(1)  # 'pre' or 'post'
    n = int(m.group(2))
    chronic = "THC" if (n % 2 == 1) else "Vehicle"
    acute = acute_for_mouse(n)
    return n, cond, chronic, acute


def assign_states(df: pd.DataFrame, priority: list[str]) -> np.ndarray:
    n = len(df)
    states = np.full(n, NOISE_LABEL, dtype=object)
    for behavior in reversed(priority):
        if behavior in df.columns:
            mask = df[behavior].astype(int).to_numpy() == 1
            states[mask] = behavior
    return states


def state_occupancy(states: np.ndarray, all_states: list[str]) -> dict:
    n_total = len(states)
    out = {s: 0.0 for s in all_states}
    out[NOISE_LABEL] = 0.0
    vals, counts = np.unique(states, return_counts=True)
    raw = dict(zip(vals, counts))
    n_noise = int(raw.get(NOISE_LABEL, 0))
    n_labeled = n_total - n_noise
    out[NOISE_LABEL] = n_noise / max(1, n_total)
    denom = n_labeled if (EXCLUDE_NOISE and n_labeled > 0) else max(1, n_total)
    for s in all_states:
        out[s] = raw.get(s, 0) / denom
    return out


def transition_matrix(states: np.ndarray, all_states: list[str]) -> pd.DataFrame:
    if len(states) == 0:
        idx = all_states + ([] if EXCLUDE_NOISE else [NOISE_LABEL])
        return pd.DataFrame(0, index=idx, columns=idx)
    bouts = [states[0]]
    for s in states[1:]:
        if s != bouts[-1]:
            bouts.append(s)
    if EXCLUDE_NOISE:
        bouts = [b for b in bouts if b != NOISE_LABEL]
        idx = all_states
    else:
        idx = all_states + [NOISE_LABEL]
    M = pd.DataFrame(0, index=idx, columns=idx)
    for a, b in zip(bouts[:-1], bouts[1:]):
        if a in M.index and b in M.columns:
            M.loc[a, b] += 1
    return M


def normalise_rows(M: pd.DataFrame) -> pd.DataFrame:
    rs = M.sum(axis=1).replace(0, 1)
    return M.div(rs, axis=0)


def plot_immobility_2x2(occ_df: pd.DataFrame, png: Path) -> None:
    """Four-panel: post-rim still % by chronic x acute, plus pre baselines
    by chronic, plus paired per-mouse pre->post within each chronic-acute cell."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Panel 1: post-rim still % by 4 cells
    ax = axes[0, 0]
    post = occ_df[occ_df["condition"] == "post"]
    cells = [("THC", "Rim"), ("THC", "Veh"), ("Vehicle", "Rim"), ("Vehicle", "Veh")]
    data = [(post[(post["chronic"] == c) & (post["acute"] == a)]["still"].to_numpy() * 100).tolist()
            for c, a in cells]
    labels = [f"{c}\n{a}" for c, a in cells]
    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black"))
    for patch, (c, a) in zip(bp["boxes"], cells):
        face = CHRONIC_PALETTE[c]
        patch.set_facecolor(face)
        patch.set_alpha(0.6 if a == "Rim" else 0.3)
    for i, (cell, ys) in enumerate(zip(cells, data), 1):
        if ys:
            xs = np.random.normal(i, 0.04, len(ys))
            ax.scatter(xs, ys, color="black", s=22, zorder=3)
            ax.text(i, max(ys) + 1, f"n={len(ys)}",
                    ha="center", fontsize=9, color="gray")
    ax.set_ylabel("% time immobile (still)")
    ax.set_title("POST: chronic x acute (primary contrast)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 2: pre still % by chronic only (acute not given yet)
    ax = axes[0, 1]
    pre = occ_df[occ_df["condition"] == "pre"]
    groups = ["THC", "Vehicle"]
    data = [(pre[pre["chronic"] == g]["still"].to_numpy() * 100).tolist() for g in groups]
    bp = ax.boxplot(data, labels=groups, patch_artist=True, widths=0.55,
                    medianprops=dict(color="black"))
    for patch, g in zip(bp["boxes"], groups):
        patch.set_facecolor(CHRONIC_PALETTE[g]); patch.set_alpha(0.6)
    for i, (g, ys) in enumerate(zip(groups, data), 1):
        if ys:
            xs = np.random.normal(i, 0.04, len(ys))
            ax.scatter(xs, ys, color="black", s=22, zorder=3)
            ax.text(i, max(ys) + 1, f"n={len(ys)}",
                    ha="center", fontsize=9, color="gray")
    ax.set_ylabel("% time immobile (still)")
    ax.set_title("PRE: chronic only (acute not yet given)")
    ax.grid(axis="y", alpha=0.3)

    # Panel 3: paired per-mouse pre->post (split by acute condition)
    ax = axes[1, 0]
    for chronic in ["THC", "Vehicle"]:
        for acute in ["Rim", "Veh"]:
            sub = occ_df[(occ_df["chronic"] == chronic) & (occ_df["acute"] == acute)]
            for mouse, mdf in sub.groupby("mouse"):
                pr = mdf[mdf["condition"] == "pre"]["still"]
                po = mdf[mdf["condition"] == "post"]["still"]
                if len(pr) and len(po):
                    ls = "-" if acute == "Rim" else "--"
                    ax.plot([0, 1], [pr.iloc[0] * 100, po.iloc[0] * 100],
                            marker="o", linestyle=ls,
                            color=CHRONIC_PALETTE[chronic],
                            alpha=0.7, label=f"m{mouse} ({chronic}/{acute})")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["pre", "post"])
    ax.set_ylabel("% time immobile (still)")
    ax.set_title("Within-subject: pre -> post (solid=Rim, dashed=Veh)")
    ax.grid(axis="y", alpha=0.3)
    handles = [
        plt.Line2D([0], [0], color=CHRONIC_PALETTE["THC"], marker="o",
                   linestyle="-", label="THC / Rim"),
        plt.Line2D([0], [0], color=CHRONIC_PALETTE["THC"], marker="o",
                   linestyle="--", label="THC / Veh"),
        plt.Line2D([0], [0], color=CHRONIC_PALETTE["Vehicle"], marker="o",
                   linestyle="-", label="Veh / Rim"),
        plt.Line2D([0], [0], color=CHRONIC_PALETTE["Vehicle"], marker="o",
                   linestyle="--", label="Veh / Veh"),
    ]
    ax.legend(handles=handles, loc="best", fontsize=8)

    # Panel 4: per-mouse summary table
    ax = axes[1, 1]; ax.axis("off")
    table_rows = []
    for n in sorted(occ_df["mouse"].unique()):
        sub = occ_df[occ_df["mouse"] == n]
        c = sub["chronic"].iloc[0]
        a = sub["acute"].iloc[0]
        pr = sub[sub["condition"] == "pre"]["still"]
        po = sub[sub["condition"] == "post"]["still"]
        pr_v = f"{pr.iloc[0]*100:.1f}%" if len(pr) else "-"
        po_v = f"{po.iloc[0]*100:.1f}%" if len(po) else "-"
        delta = f"{(po.iloc[0]-pr.iloc[0])*100:+.1f}" if len(pr) and len(po) else "-"
        table_rows.append([f"THC{n}", c, a, pr_v, po_v, delta])
    if table_rows:
        col_lbl = ["mouse", "chronic", "acute", "pre % still", "post % still", "delta"]
        tbl = ax.table(cellText=table_rows, colLabels=col_lbl, loc="center",
                       cellLoc="center", colLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9)
        tbl.scale(1, 1.3)
        for c in range(len(col_lbl)):
            tbl[(0, c)].set_facecolor("#eee"); tbl[(0, c)].set_text_props(weight="bold")
    ax.set_title("Per-mouse summary (window-capped)")

    fig.suptitle("Time immobile — THC chronic x rimonabant challenge cohort",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_state_occupancy_2x2(occ_df: pd.DataFrame, png: Path) -> None:
    states = PRIORITY if EXCLUDE_NOISE else PRIORITY + [NOISE_LABEL]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, cond_label, cond in zip(axes, ["PRE-rim", "POST-rim"], ["pre", "post"]):
        sub = occ_df[occ_df["condition"] == cond]
        x = np.arange(len(states))
        if cond == "pre":
            cells = [("THC", None), ("Vehicle", None)]
        else:
            cells = [("THC", "Rim"), ("THC", "Veh"),
                     ("Vehicle", "Rim"), ("Vehicle", "Veh")]
        w = 0.8 / max(len(cells), 1)
        for i, (chronic, acute) in enumerate(cells):
            if acute is None:
                ss = sub[sub["chronic"] == chronic]
                label = chronic
            else:
                ss = sub[(sub["chronic"] == chronic) & (sub["acute"] == acute)]
                label = f"{chronic}/{acute}"
            means = [ss[s].mean() * 100 if s in ss.columns else 0.0 for s in states]
            sems = [ss[s].sem() * 100 if s in ss.columns else 0.0 for s in states]
            face = CHRONIC_PALETTE[chronic]
            alpha = 0.85 if (acute == "Rim" or acute is None) else 0.4
            ax.bar(x + (i - (len(cells) - 1) / 2) * w, means, w, yerr=sems,
                   label=label, color=face, alpha=alpha, capsize=2)
        ax.set_xticks(x); ax.set_xticklabels(states, rotation=30, ha="right")
        ax.set_ylabel("% time")
        ax.set_title(cond_label)
        ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Per-state occupancy — THC x rimonabant cohort")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def plot_transitions(group_mats: dict, png: Path) -> None:
    """Heatmap grid: rows = (chronic x acute cells), cols = (pre, post).
    With sparse pre/post and 4 cells per condition this is a 4x2 grid."""
    cells_post = [("THC", "Rim"), ("THC", "Veh"), ("Vehicle", "Rim"), ("Vehicle", "Veh")]
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    for r, (chronic, acute) in enumerate(cells_post):
        for c, cond in enumerate(["pre", "post"]):
            ax = axes[r, c]
            M = group_mats.get((chronic, acute, cond))
            if M is None or M.sum().sum() == 0:
                ax.text(0.5, 0.5, "no data", ha="center", va="center")
                ax.set_title(f"{chronic}/{acute} -- {cond}")
                ax.set_xticks([]); ax.set_yticks([])
                continue
            P = normalise_rows(M)
            im = ax.imshow(P.to_numpy(), cmap="inferno", vmin=0, vmax=1)
            ax.set_xticks(range(len(P.columns)))
            ax.set_xticklabels(P.columns, rotation=45, ha="right", fontsize=8)
            ax.set_yticks(range(len(P.index)))
            ax.set_yticklabels(P.index, fontsize=8)
            ax.set_title(f"{chronic}/{acute} -- {cond}", fontsize=10)
            for i in range(P.shape[0]):
                for j in range(P.shape[1]):
                    v = P.iloc[i, j]
                    if v > 0.01:
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                                color="white" if v < 0.5 else "black", fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Transition probabilities (row-normalised)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def collect_bout_stats(per_session_states: dict, all_states: list) -> pd.DataFrame:
    rows = []
    for sess, info in per_session_states.items():
        states = info["states"]
        bouts = []
        if len(states):
            cur = states[0]; start = 0
            for i in range(1, len(states)):
                if states[i] != cur:
                    bouts.append((cur, start, i - 1))
                    cur = states[i]; start = i
            bouts.append((cur, start, len(states) - 1))
        bl = {s: [] for s in all_states + [NOISE_LABEL]}
        for s, a, b in bouts:
            if s in bl:
                bl[s].append(b - a + 1)
        row = {"session": sess, "mouse": info["mouse"],
               "condition": info["condition"], "chronic": info["chronic"],
               "acute": info["acute"]}
        for s in all_states + [NOISE_LABEL]:
            row[f"{s}_bouts"] = len(bl[s])
            row[f"{s}_mean_len"] = float(np.mean(bl[s])) if bl[s] else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def plot_ethograms(per_session_states: dict, all_states: list, png: Path,
                   fps: int = FPS) -> None:
    items = sorted(
        per_session_states.items(),
        key=lambda kv: (kv[1]["chronic"], kv[1]["acute"], kv[1]["condition"], kv[1]["mouse"])
    )
    if not items:
        return
    n_sess = len(items)
    state_to_int = {s: i for i, s in enumerate(all_states + [NOISE_LABEL])}
    max_len = max(len(info["states"]) for _, info in items)
    grid = np.full((n_sess, max_len), state_to_int[NOISE_LABEL], dtype=int)
    labels = []
    for r, (sess, info) in enumerate(items):
        s = info["states"]
        ints = np.array([state_to_int.get(x, state_to_int[NOISE_LABEL]) for x in s])
        grid[r, : len(ints)] = ints
        labels.append(f"{info['chronic'][:3]}/{info['acute']} {info['condition']}  m{info['mouse']}")
    n_states = len(all_states) + 1
    cmap = plt.get_cmap("tab10", n_states)
    fig, ax = plt.subplots(figsize=(15, max(4, n_sess * 0.45)))
    im = ax.imshow(grid, aspect="auto", interpolation="nearest", cmap=cmap,
                   vmin=-0.5, vmax=n_states - 0.5)
    ax.set_yticks(range(n_sess))
    ax.set_yticklabels(labels, fontsize=9)
    n_min = max_len / fps / 60.0
    xt = np.linspace(0, max_len, min(11, int(n_min) + 1))
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{int(t / fps / 60)}" for t in xt])
    ax.set_xlabel("Time (min)")
    ax.set_title("Per-session ethograms (sorted by chronic/acute/condition/mouse)")
    cbar = plt.colorbar(im, ax=ax, ticks=range(n_states), fraction=0.02, pad=0.01)
    cbar.ax.set_yticklabels(all_states + [NOISE_LABEL], fontsize=9)
    fig.tight_layout()
    fig.savefig(png, dpi=140)
    plt.close(fig)


def post_to_discord(text: str, png_paths: list[Path]) -> None:
    import subprocess
    cmd = ["curl", "-s", "-o", os.devnull, "-w", "%{http_code}", "-X", "POST",
           "-A", "PixelPaws-Chain/1.0",
           "-F", f'payload_json={{"content":"{text}"}}']
    for i, p in enumerate(png_paths):
        cmd += ["-F", f"file{i}=@{p}"]
    cmd.append(WEBHOOK)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        print(f"Discord upload: HTTP {r.stdout}")
    except Exception as e:
        print(f"Discord upload failed: {e}")


def main() -> int:
    csvs = sorted(RESULTS_DIR.glob("*_predictions.csv"))
    if not csvs:
        print(f"No prediction CSVs in {RESULTS_DIR}")
        return 1

    rows = []
    group_mats = {}
    per_session_states = {}

    for csv in csvs:
        base = csv.name.replace("_predictions.csv", "")
        parsed = parse_session(base)
        if not parsed:
            print(f"  skip (cannot parse): {base}")
            continue
        n, cond, chronic, acute = parsed
        df = pd.read_csv(csv)

        # Window-cap to the conventional duration before assigning states.
        cap_min = CAP_PRE_MIN if cond == "pre" else CAP_POST_MIN
        cap_frames = cap_min * 60 * FPS
        if len(df) > cap_frames:
            df = df.iloc[:cap_frames].copy()
        states = assign_states(df, PRIORITY)
        occ = state_occupancy(states, PRIORITY)
        occ.update({"session": base, "mouse": n, "condition": cond,
                    "chronic": chronic, "acute": acute,
                    "n_frames": len(states), "duration_min": len(states) / FPS / 60})
        rows.append(occ)
        per_session_states[base] = {
            "states": states, "mouse": n, "condition": cond,
            "chronic": chronic, "acute": acute,
        }
        M = transition_matrix(states, PRIORITY)
        key = (chronic, acute, cond)
        if key in group_mats:
            group_mats[key] = group_mats[key].add(M, fill_value=0)
        else:
            group_mats[key] = M

    occ_df = pd.DataFrame(rows)
    occ_csv = ANALYSIS_DIR / "state_occupancy.csv"
    occ_df.to_csv(occ_csv, index=False)
    print(f"Wrote: {occ_csv}")

    for (chronic, acute, cond), M in group_mats.items():
        out = ANALYSIS_DIR / f"transitions_{chronic}_{acute}_{cond}.csv"
        M.to_csv(out)

    p_immob = ANALYSIS_DIR / "immobility.png"
    p_states = ANALYSIS_DIR / "state_occupancy.png"
    p_trans = ANALYSIS_DIR / "transitions.png"
    p_etho = ANALYSIS_DIR / "ethograms.png"

    plot_immobility_2x2(occ_df, p_immob)
    plot_state_occupancy_2x2(occ_df, p_states)
    plot_transitions(group_mats, p_trans)

    bout_df = collect_bout_stats(per_session_states, PRIORITY)
    bout_df.to_csv(ANALYSIS_DIR / "bout_stats.csv", index=False)
    plot_ethograms(per_session_states, PRIORITY, p_etho)

    print(f"Plots: {p_immob}, {p_states}, {p_trans}, {p_etho}")

    # Discord summary
    lines = [f"THC x rimonabant analysis complete ({len(occ_df)} sessions, "
             f"{occ_df['mouse'].nunique()} mice)."]
    lines.append("% time immobile (mean +/- SD):")
    for cond_label, cond in [("PRE", "pre"), ("POST", "post")]:
        sub = occ_df[occ_df["condition"] == cond]
        if cond == "pre":
            for chronic in ["THC", "Vehicle"]:
                ss = sub[sub["chronic"] == chronic]["still"].to_numpy()
                if len(ss):
                    lines.append(f"  {cond_label} {chronic:8} "
                                 f"{ss.mean()*100:5.1f} +/- {ss.std(ddof=1)*100 if len(ss)>1 else 0:.1f}  (n={len(ss)})")
        else:
            for chronic in ["THC", "Vehicle"]:
                for acute in ["Rim", "Veh"]:
                    ss = sub[(sub["chronic"] == chronic) & (sub["acute"] == acute)]["still"].to_numpy()
                    if len(ss):
                        lines.append(f"  {cond_label} {chronic:8}/{acute:3} "
                                     f"{ss.mean()*100:5.1f} +/- {ss.std(ddof=1)*100 if len(ss)>1 else 0:.1f}  (n={len(ss)})")
    summary = "\\n".join(lines).replace('"', '')
    post_to_discord(summary, [p_immob, p_states, p_trans])
    post_to_discord("Per-session ethograms:", [p_etho])
    return 0


if __name__ == "__main__":
    sys.exit(main())
