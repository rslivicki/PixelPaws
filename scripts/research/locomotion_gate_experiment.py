"""Headless: does a STRINGENT 'obvious-walking-only' gate help L_flinching?

Earlier the default walking gate failed (it removed real flinches: 62% of labeled
flinch frames overlap walking, and model FPs were LESS concentrated in walking than
TPs). This sweeps the walking gate from lenient → very stringent (raise the walking
probability threshold + require a long sustained run, optionally + high centroid speed)
to see whether limiting it to OBVIOUS sustained ambulation removes false positives
without deleting real flinches.

Per stringency level (pooled over sessions, observed frames):
  cover%      = % of frames flagged as 'obvious walking' (how much it filters)
  labFlinch%  = % of HUMAN-labeled flinch frames inside the mask  (want LOW)
  FP% / TP%   = % of model false-pos / true-pos flinch frames inside the mask
  gate Δ      = flinch P/R/F1 after zeroing predictions in the mask, vs raw
A level helps only if labFlinch% is LOW and FP% > TP% and ΔP > 0 with small ΔR.

Read-only. Run with Python310.
"""
import sys, glob, os, pickle
import numpy as np, pandas as pd

REPO = r"E:\Sync_from_lab\PixelPaws"
sys.path.insert(0, REPO)
import joblib
from prediction_pipeline import augment_features_post_cache, predict_with_xgboost, apply_smoothing

ARC = r"E:\PixelPaws_Projects\_Archived\2511_Blackbox_Formalin"
LAB = r"E:\Sync_from_lab\RS_Boris\per_frame_labels"
ENC = r"E:\Sync_from_lab\PixelPaws\pixelpaws_global_classifier_encyclopedia\classifiers"
SPEED_COL, SPEED_WIN = "centroid_Vel10", 9
SESSIONS = ["260129_Formalin_3122", "260129_Formalin_3125", "260129_Formalin_3128",
            "260129_Formalin_3305", "260129_Formalin_3304", "260129_Formalin_2802",
            "260129_Formalin_3301"]
# (label, walk_proba_thresh, min_run_frames, speed_pctl_or_None)
LEVELS = [
    ("walk>=.35 run6   (baseline)", 0.35, 6,   None),
    ("walk>=.70 run15",             0.70, 15,  None),
    ("walk>=.90 run30",             0.90, 30,  None),
    ("walk>=.95 run45",             0.95, 45,  None),
    ("walk>=.97 run60",             0.97, 60,  None),
    ("walk>=.90 run30 + spd>p80",   0.90, 30,  80),
    ("walk>=.90 run45 + spd>p90",   0.90, 45,  90),
]

def load_clf(name): return joblib.load(os.path.join(ENC, f"classifier_{name}.pkl"))
def load_feat(p):
    try: d = joblib.load(p)
    except Exception: d = pickle.load(open(p, "rb"))
    if isinstance(d, dict):
        for k in ("X", "features", "df"):
            if k in d: return d[k]
    return d
def find_feat(s):
    h = glob.glob(ARC + rf"\**\{s}_features_8aed1c22.pkl", recursive=True); return h[0] if h else None
def find_h5(s):
    h = [x for x in glob.glob(ARC + rf"\**\{s}*shuffle9*best-190.h5", recursive=True) if not x.endswith("_filtered.h5")]
    return h[0] if h else None
def _load_regen():
    c = [x for x in glob.glob(r"E:\Sync_from_lab\PixelPaws\encyclopedia_runs\L_flinching_gui_run_regen_20260602\classifiers\**\*.pkl", recursive=True) if 'train_set' not in x]
    return joblib.load(c[0]) if c else load_clf("L_flinching")
def proba(X, h5, cd):
    Xa = augment_features_post_cache(X.copy(), cd, cd["clf_model"], h5, log_fn=None)
    return predict_with_xgboost(cd["clf_model"], Xa, calibrator=cd.get("prob_calibrator"), fold_models=cd.get("fold_models"))
def binary(X, h5, cd):
    return apply_smoothing(proba(X, h5, cd), cd, "bout_filters").astype(int)
def runs_min(mask, k):
    m = mask.astype(bool).copy(); pad = np.concatenate([[0], m.astype(int), [0]]); d = np.diff(pad)
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if b - a < k: m[a:b] = False
    return m
def prf(tp, fp, fn):
    p = tp/(tp+fp) if tp+fp else 0; r = tp/(tp+fn) if tp+fn else 0
    return (2*p*r/(p+r) if p+r else 0), p, r

def main():
    walk, flinch = load_clf("walking"), _load_regen()
    sess = []
    for s in SESSIONS:
        fp_, h5 = find_feat(s), find_h5(s); npy = os.path.join(LAB, s + "__L_flinching.npy")
        if not (fp_ and h5 and os.path.isfile(npy)): print(f"skip {s} (missing input)"); continue
        X = load_feat(fp_); y = np.load(npy).astype(int)
        n = min(len(X), len(y)); X = X.iloc[:n].reset_index(drop=True); y = y[:n]
        wp = proba(X, h5, walk)[:n]
        fl = binary(X, h5, flinch)[:n]
        spd = pd.Series(X[SPEED_COL].values[:n] if SPEED_COL in X.columns else np.zeros(n)).rolling(
            SPEED_WIN, center=True, min_periods=1).mean().values
        obs = y >= 0
        sess.append(dict(y=y, fl=fl, wp=wp, spd=spd, obs=obs))
        print(f"loaded {s}: n={n}, labeled flinch={int((y[obs]==1).sum())}")
    # pooled raw flinch metrics
    def pool(key, mask_fn=None):
        tp=fp=fn=0
        for d in sess:
            o=d["obs"]; yt=d["y"]; fl=d["fl"].copy()
            if mask_fn is not None: fl[mask_fn(d)] = 0
            tp+=int(((fl==1)&(yt==1)&o).sum()); fp+=int(((fl==1)&(yt==0)&o).sum()); fn+=int(((fl==0)&(yt==1)&o).sum())
        return prf(tp,fp,fn)
    f0,p0,r0 = pool("raw")
    print(f"\nRAW flinch (regen, pooled weak sessions): F1={f0:.3f} P={p0:.3f} R={r0:.3f}\n")
    hdr=f"{'stringency level':32s} {'cover%':>6} {'labFlinch%':>10} {'FP%':>6} {'TP%':>6}   {'F1':>5} {'P':>5} {'R':>5}  {'ΔP':>6} {'ΔR':>6}"
    print(hdr); print("-"*len(hdr))
    for name, thr, run, spct in LEVELS:
        # precompute pooled speed percentile threshold
        sthr = None
        if spct is not None:
            allspd = np.concatenate([d["spd"][d["obs"]] for d in sess]); sthr = np.percentile(allspd, spct)
        def mk(d):
            mask = d["wp"] >= thr
            if sthr is not None: mask = mask & (d["spd"] > sthr)
            return runs_min(mask, run)
        cov=lab=labtot=fpm=fptot=tpm=tptot=0
        for d in sess:
            o=d["obs"]; yt=d["y"]; fl=d["fl"]; mm=mk(d)
            cov+=int((mm&o).sum())
            lab+=int(((yt==1)&mm&o).sum()); labtot+=int(((yt==1)&o).sum())
            fpm+=int(((fl==1)&(yt==0)&mm&o).sum()); fptot+=int(((fl==1)&(yt==0)&o).sum())
            tpm+=int(((fl==1)&(yt==1)&mm&o).sum()); tptot+=int(((fl==1)&(yt==1)&o).sum())
        nobs=sum(int(d["obs"].sum()) for d in sess)
        fg,pg,rg = pool(name, mk)
        print(f"{name:32s} {100*cov/nobs:6.1f} {100*lab/max(labtot,1):10.1f} "
              f"{100*fpm/max(fptot,1):6.1f} {100*tpm/max(tptot,1):6.1f}   "
              f"{fg:5.2f} {pg:5.2f} {rg:5.2f}  {pg-p0:+6.3f} {rg-r0:+6.3f}")
    print("\nWANT: a level where labFlinch% is LOW (<~15) AND FP% > TP% AND ΔP>0 with small ΔR.")
    return 0

if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
