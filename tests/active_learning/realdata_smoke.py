"""Real-data smoke for AL warm-start.

Warm-starts the engine from a REAL encyclopedia classifier on a REAL 8aed1c22
feature cache, the established way: augment_features_post_cache -> predict_with_xgboost
-> engine candidate selection. Reproduces the validation that caught the augmented-
feature gap (encyclopedia models expect features the base cache lacks). Skips
gracefully if the real fixtures aren't present.
"""
import os, sys, glob, pickle
import numpy as np

_REPO = r"E:\Sync_from_lab\PixelPaws"
sys.path.insert(0, _REPO)

FEAT = r"D:\PixelPaws_Active\2605_SNI_oxy\features\2605_SNIoxy_1_baseline_features_8aed1c22.pkl"
CLF  = r"E:\Sync_from_lab\PixelPaws\pixelpaws_global_classifier_encyclopedia\classifiers\classifier_Scratching.pkl"
VIDS = r"D:\PixelPaws_Active\2605_SNI_oxy\videos"


def main():
    if not (os.path.isfile(FEAT) and os.path.isfile(CLF)):
        print("SKIP: real features/classifier not found."); return 0
    import joblib
    import active_learning_engine as E
    from prediction_pipeline import augment_features_post_cache, predict_with_xgboost
    h5 = [x for x in glob.glob(os.path.join(VIDS, "2605_SNIoxy_1_baseline*shuffle9*best-190.h5"))
          if not x.endswith("_filtered.h5")]
    if not h5:
        print("SKIP: DLC h5 not found."); return 0
    Xdf = pickle.load(open(FEAT, "rb")); cols = list(Xdf.columns)
    cd = joblib.load(CLF); m = cd["clf_model"]
    Xa = augment_features_post_cache(Xdf.copy(), cd, m, h5[0])
    p = predict_with_xgboost(m, Xa, calibrator=(cd.get("prob_calibrator") or cd.get("calibrator")), fold_models=cd.get("fold_models"))
    eng = E.ActiveLearningEngine([{"features": Xdf.values.astype(np.float32),
                                   "labels": np.full(len(Xdf), -1, int)}],
                                 feature_cols=cols, min_bout_frames=5, max_bout_frames=120, min_frame_gap=60)
    batch = eng.score_and_candidates(probas_by_session=[p], batch_size=15, pos_quota_frac=0.4)
    print(f"real warm-start OK: {len(p)} frames scored (mean P={p.mean():.3f}); "
          f"{len(batch)} bouts ({sum(b.pred_pos for b in batch)} pred-positive)")
    assert 0 < len(batch) <= 15
    print("REAL-DATA WARM-START SMOKE: PASS")
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    raise SystemExit(main())
