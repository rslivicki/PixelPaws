"""Import a native BORIS project (`.boris`) into PixelPaws per-frame `_labels.csv`.

BORIS stores STATE behaviors as alternating START/STOP events carrying a frame
index. This converts each behavior to a per-frame label column aligned 1:1 with
the session's feature cache, using three-valued labels:

    1   reviewed & behavior present
    0   reviewed & behavior absent (a negative the annotator actually observed)
   -1   NOT reviewed / unobserved (excluded from training; AL may query it)

Per-behavior observation window (``per_behavior_tails=True``, the default): a
frame is ``0`` only if a human reviewed it for *that* behavior. After a behavior's
own last event we can't assume that, so the tail is ``-1`` — not an assumed ``0``.
A behavior with no events in a session is entirely ``-1``. This is what makes the
exported data correct "at the outset" so AL/Train don't ingest false negatives.

The core ``export_boris_labels`` is reused by the CLI wrapper
(``scripts/research/export_boris_labels.py``) and by the GUI "Import BORIS
Project" / QC tools.
"""
import os
import sys
import json
import glob
import numpy as np
import pandas as pd

# Repo root so we can import _robust_unpickle for feature-cache row counts.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)


def _robust_unpickle_len(path):
    """Row count of a feature cache (joblib+LZ4 for 8aed1c22, else pickle)."""
    try:
        from active_learning_v2 import _robust_unpickle
        return len(_robust_unpickle(path))
    except Exception:
        return None


def _cache_len(feat_dir, session):
    """Row count of the session's 8aed1c22 feature cache, or None."""
    if not feat_dir:
        return None
    cand = glob.glob(os.path.join(feat_dir, f"{session}*_features_8aed1c22.pkl"))
    exact = [c for c in cand if os.path.basename(c).startswith(session + "_features_")]
    cand = exact or cand
    if not cand:
        return None
    return _robust_unpickle_len(cand[0])


def parse_boris(boris_path):
    """Load a `.boris` JSON project → (behaviors, observations dict)."""
    with open(boris_path, encoding="utf-8", errors="replace") as f:
        proj = json.load(f)
    behaviors = [v.get("code") for v in (proj.get("behaviors_conf") or {}).values()
                 if v.get("code")]
    return behaviors, proj.get("observations", {})


def _event_frame(ev, fps):
    """Frame index of a BORIS event — prefer the stored index, else time*fps."""
    if len(ev) > 5 and isinstance(ev[5], (int, float)):
        return int(ev[5])
    return int(round(float(ev[0]) * fps))


def build_label_frame(events, behaviors, N, fps, per_behavior_tails=True,
                      subjects=None):
    """Build the per-frame label DataFrame for one observation.

    Returns (df, summary) where summary[behavior] = dict(pos, observed_to, unobserved).
    ``observed_to`` is the frame up to which that behavior is treated as reviewed
    (its own last event when per_behavior_tails, else the global last event); frames
    at/after it are ``-1``. A behavior with no events is entirely ``-1``.

    ``subjects`` (optional set) restricts events to those scored under the given BORIS
    subject(s) (``ev[1]``); ``None`` uses all subjects.
    """
    if subjects is not None:
        events = [ev for ev in events if ev[1] in subjects]
    global_last = min(max((_event_frame(ev, fps) for ev in events), default=0), N)
    cols = {}
    summary = {}
    for b in behaviors:
        frames = sorted(_event_frame(ev, fps) for ev in events if ev[2] == b)
        arr = np.zeros(N, dtype=np.int8)
        if not frames:
            # Behavior never scored in this session → fully unobserved.
            arr[:] = -1
            summary[b] = {"pos": 0, "observed_to": 0, "unobserved": int(N)}
            cols[b] = arr
            continue
        last_b = min(max(frames), N)
        observed_to = last_b if per_behavior_tails else global_last
        # Unobserved tail: frames at/after the observation boundary are -1.
        if observed_to < N:
            arr[observed_to:] = -1
        # Fill state bouts (half-open [start, stop)).
        for i in range(len(frames) // 2):
            s, e = frames[2 * i], frames[2 * i + 1]
            if e < s:
                s, e = e, s
            s = max(0, min(s, N)); e = max(0, min(e, N))
            arr[s:e] = 1
        cols[b] = arr
        summary[b] = {"pos": int((arr == 1).sum()),
                      "observed_to": int(observed_to),
                      "unobserved": int((arr == -1).sum())}
    return pd.DataFrame(cols), summary


def export_boris_labels(boris_path, out_dir, feat_dir=None,
                        per_behavior_tails=True, log=print, self_check=True,
                        behaviors=None, sessions=None, subjects=None, merge=True):
    """Write `<session>_labels.csv` for every observation in a BORIS project.

    ``behaviors`` (optional list) restricts the columns written; ``sessions``
    (optional set of session_names) restricts which observations are exported;
    ``subjects`` (optional set) restricts events by BORIS subject. ``None`` = all.

    ``merge`` (default True): if a session's CSV already exists, update/add only the
    selected behavior columns and preserve the rest — so exporting a behavior subset
    never clobbers other behaviors already on disk.

    Returns a list of per-session summary dicts:
        {session, N, source, path, behaviors:{b: {pos, observed_to, unobserved}}, warnings}
    Observations with zero events are skipped (nothing to score).
    When ``self_check`` is True, every written CSV is re-read and compared against
    a fresh re-derivation; any mismatch is logged loudly.
    """
    all_behaviors, observations = parse_boris(boris_path)
    use_behaviors = [b for b in all_behaviors if behaviors is None or b in behaviors]
    sess_filter = set(sessions) if sessions is not None else None
    subj_filter = set(subjects) if subjects is not None else None
    os.makedirs(out_dir, exist_ok=True)
    log(f"BORIS: {os.path.basename(boris_path)} | behaviors: {use_behaviors}")
    log(f"Per-behavior unobserved tails: {'ON' if per_behavior_tails else 'OFF (global)'}")

    results = []
    for obs_key, o in observations.items():
        mi = o.get("media_info", {})
        media = next(iter(mi.get("length", {}).keys()), None)
        if not media:
            log(f"  {obs_key}: no media info — skipped")
            continue
        session = os.path.splitext(os.path.basename(media))[0]
        if sess_filter is not None and session not in sess_filter:
            continue
        fps = float(next(iter(mi.get("fps", {}).values()), 60.0))
        length_s = float(next(iter(mi.get("length", {}).values()), 0.0))

        N = _cache_len(feat_dir, session)
        src = "cache"
        if N is None:
            N = int(round(length_s * fps))
            src = "length*fps (no cache)"
        if N <= 0:
            log(f"  {obs_key}: zero length — skipped")
            continue

        events = o.get("events", [])
        if not events:
            log(f"  {session}: 0 events — skipped (nothing scored)")
            continue

        # Warn on any behavior with an unclosed START (odd number of toggles).
        _ev_for_warn = (events if subj_filter is None
                        else [ev for ev in events if ev[1] in subj_filter])
        warnings = []
        for b in use_behaviors:
            n = sum(1 for ev in _ev_for_warn if ev[2] == b)
            if n % 2:
                warnings.append(f"{b}: unclosed START dropped")

        df, summary = build_label_frame(events, use_behaviors, N, fps,
                                        per_behavior_tails, subjects=subj_filter)
        out = os.path.join(out_dir, f"{session}_labels.csv")
        merged_cols = []
        if merge and os.path.isfile(out):
            try:
                existing = pd.read_csv(out)
                if len(existing) != N:
                    existing = existing.reindex(range(N))   # pad/truncate to cache length
                kept = [c for c in existing.columns if c not in df.columns]
                for c in df.columns:
                    existing[c] = df[c].values               # update/add selected behaviors
                existing.to_csv(out, index=False)
                merged_cols = kept
            except Exception:
                df.to_csv(out, index=False)
        else:
            df.to_csv(out, index=False)

        log(f"  {session}: N={N} ({src})"
            + (f"  [merged; kept {len(merged_cols)} existing col(s): "
               f"{', '.join(merged_cols)}]" if merged_cols else ""))
        for b in use_behaviors:
            s = summary[b]
            if s["pos"] or s["observed_to"] or s["unobserved"] != N:
                log(f"      {b}: {s['pos']:,} pos | observed→{s['observed_to']:,}/{N:,}"
                    f" | {s['unobserved']:,} unobserved(-1)")
        if warnings:
            log("      ! " + "; ".join(warnings))

        results.append({"session": session, "N": N, "source": src, "path": out,
                        "behaviors": summary, "warnings": warnings})

    # Built-in self-check: re-derive and compare every written column.
    if self_check and results:
        ok = _verify_against_boris(boris_path, out_dir, feat_dir, per_behavior_tails,
                                   log, behaviors=use_behaviors, sessions=sess_filter,
                                   subjects=subj_filter)
        log("Self-check: " + ("PASS — CSVs match BORIS." if ok
                              else "FAIL — see mismatches above!"))

    log(f"\nWrote {len(results)} label file(s) → {out_dir}")
    return results


def _verify_against_boris(boris_path, out_dir, feat_dir, per_behavior_tails, log,
                          behaviors=None, sessions=None, subjects=None):
    """Re-derive labels from the .boris and assert each written CSV matches."""
    all_behaviors, observations = parse_boris(boris_path)
    use_behaviors = [b for b in all_behaviors if behaviors is None or b in behaviors]
    sess_filter = set(sessions) if sessions is not None else None
    subj_filter = set(subjects) if subjects is not None else None
    all_ok = True
    for obs_key, o in observations.items():
        mi = o.get("media_info", {})
        media = next(iter(mi.get("length", {}).keys()), None)
        if not media:
            continue
        session = os.path.splitext(os.path.basename(media))[0]
        if sess_filter is not None and session not in sess_filter:
            continue
        csv = os.path.join(out_dir, f"{session}_labels.csv")
        if not os.path.isfile(csv):
            continue
        fps = float(next(iter(mi.get("fps", {}).values()), 60.0))
        length_s = float(next(iter(mi.get("length", {}).values()), 0.0))
        N = _cache_len(feat_dir, session) or int(round(length_s * fps))
        events = o.get("events", [])
        if N <= 0 or not events:
            continue
        ref, _ = build_label_frame(events, use_behaviors, N, fps, per_behavior_tails,
                                   subjects=subj_filter)
        got = pd.read_csv(csv)
        if len(got) != N:
            log(f"      ✗ {session}: {len(got)} rows != {N}"); all_ok = False; continue
        # Only the columns we wrote need to match (merge may keep extra behaviors).
        for c in ref.columns:
            if c not in got.columns:
                log(f"      ✗ {session}: missing column {c}"); all_ok = False; continue
            a = np.where(np.isnan(got[c].values.astype(float)), -1,
                         got[c].values.astype(int))
            if not np.array_equal(a, ref[c].values):
                d = int((a != ref[c].values).sum())
                log(f"      ✗ {session}.{c}: {d} cells differ"); all_ok = False
    return all_ok


# ---------------------------------------------------------------------------
# CLI defaults (SNLT Rearing/Grooming project)
# ---------------------------------------------------------------------------
DEF_BORIS = r"E:\Sync_from_lab\RS_Boris\SNLT_Rearing_Grooming_shaking.boris"
DEF_OUT   = r"E:\Sync_from_lab\Blackbox\2603_SNLT_JG\Baseline\behavior_labels"
DEF_FEAT  = r"E:\Sync_from_lab\Blackbox\2603_SNLT_JG\Baseline\features"


if __name__ == "__main__":
    a = sys.argv
    export_boris_labels(
        a[1] if len(a) > 1 else DEF_BORIS,
        a[2] if len(a) > 2 else DEF_OUT,
        a[3] if len(a) > 3 else DEF_FEAT,
    )
