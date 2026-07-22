"""Reusable PixelPaws pipeline engine — transcode -> DLC -> features.

Generalizes the per-cohort run_pipeline.py into ONE parameterized tool. Stages
stop at feature extraction; classification/analysis stays cohort-specific.

All cohort-agnostic constants (webhooks, DLC config/python, feature hash/cfg,
codec settings, repo path) come from pp_config — nothing is hardcoded here.

Stages (sequential, idempotent — each skips if its output already exists):
  1. transcode : ffmpeg libx265 CRF23 preset slow, -an, -progress videos/.ffprog.txt
  2. dlc       : deeplabcut.analyze_videos via a per-video subprocess (DLC_PYTHON)
  3. features  : prediction_pipeline.PixelPaws_ExtractFeatures -> features/<stem>_features_<hash>.pkl

Sentinels for the tracker:
  <proj>/_stage.json      {stage, video, idx, total, updated}
  <proj>/videos/.ffprog.txt  ffmpeg -progress (live encode fps/speed/out_time)
  <proj>/PIPELINE_DONE    touched at the end of a real run

CLI is the primary interface; run(...) is importable for the tracker/subagent.

Usage:
  pp_pipeline.py --cohort 2606_AOW --proj D:\\PixelPaws_Active\\2606_AOW \\
                 --portal D:\\...\\videos [--sources sources.json] \\
                 [--stages transcode,dlc,features] [--webhook URL] [--dry] \\
                 [--delete-source-after-transcode]
"""
from __future__ import annotations
import argparse, json, os, pickle, subprocess, sys, time, traceback, uuid
from pathlib import Path

# --- import all cohort-agnostic constants from the single source of truth -----
sys.path.insert(0, str(Path(__file__).resolve().parent))      # pp_config beside us
from pp_config import (  # noqa: E402
    PROCESSING_WEBHOOK, FEATURE_HASH, FILTERED_FEATURE_HASH, FEATURE_CFG,
    EXPECTED_FEATURE_COLS, DLC_CONFIG, DLC_PYTHON, SHUFFLE, DLC_BATCH,
    CODEC, CRF, PRESET, REPO, post,
)

ALL_STAGES = ("transcode", "dlc", "features")
MIN_TRANSCODE_BYTES = 1_000_000      # >1 MB sanity floor for a "good" transcode

# --- persistent run log -------------------------------------------------------
# Every step() line is mirrored, append-only with a full date+time stamp, into
# <proj>/_run/pipeline.log so a silent overnight death is diagnosable after the
# fact (stdout from a backgrounded run is otherwise lost). Set once per run().
_LOG_FH = None          # open append handle for <proj>/_run/pipeline.log, or None
_LOG_PATH = None        # its Path, for reporting


def open_run_log(proj: Path):
    """Open (append) <proj>/_run/pipeline.log for the lifetime of this process.
    Best-effort: if it can't be opened, step() simply falls back to stdout-only."""
    global _LOG_FH, _LOG_PATH
    try:
        rd = proj / "_run"
        rd.mkdir(parents=True, exist_ok=True)
        _LOG_PATH = rd / "pipeline.log"
        _LOG_FH = open(_LOG_PATH, "a", encoding="utf-8", errors="replace")
    except Exception as e:
        _LOG_FH, _LOG_PATH = None, None
        print(f"[{time.strftime('%H:%M:%S')}] ! could not open run log: {e}", flush=True)
    return _LOG_PATH


def step(m: str) -> None:
    print(f'[{time.strftime("%H:%M:%S")}] {m}', flush=True)
    if _LOG_FH is not None:
        try:
            _LOG_FH.write(f'[{time.strftime("%Y-%m-%d %H:%M:%S")}] {m}\n')
            _LOG_FH.flush()          # survive a process kill (OS keeps flushed bytes)
        except Exception:
            pass


# --- per-video output resolvers (mirror the template's helpers) ---------------
def _snap_num(name: str) -> int:
    """Snapshot number from a DLC h5 name (e.g. ...best-660... -> 660), else -1."""
    import re as _re
    m = _re.search(r"best-(\d+)", name)
    return int(m.group(1)) if m else -1


def find_h5(video: Path):
    """The raw DLC analyze h5 for a transcoded video, or None. Excludes the DLC
    `_filtered.h5` and our pose-filter `_ppfilt.h5`. When several snapshots exist
    for the same video (e.g. best-460 and best-660), prefer the HIGHEST snapshot
    so downstream uses the most recent model."""
    c = [x for x in video.parent.glob(f"{video.stem}*shuffle{SHUFFLE}*.h5")
         if not x.name.endswith("_filtered.h5") and not x.name.endswith("_ppfilt.h5")]
    if not c:
        return None
    return max(c, key=lambda x: _snap_num(x.name))


def feature_pkl(features_dir: Path, video: Path, feat_hash: str = FEATURE_HASH) -> Path:
    return features_dir / f"{video.stem}_features_{feat_hash}.pkl"


def _duration(p: Path) -> float:
    try:
        r = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "default=nk=1:nw=1", str(p)], capture_output=True, text=True)
        return round(float(r.stdout.strip()), 1)
    except Exception:
        return 0.0


def transcode_output_ok(dst: Path) -> bool:
    """A transcode output is 'good' iff it exists, is non-trivially sized
    (> MIN_TRANSCODE_BYTES), and ffprobe reports a positive duration. Used both
    by the inline --delete-source-after-transcode path and the post-run cleanup
    so the verification is identical in both cases."""
    try:
        if not dst.is_file() or dst.stat().st_size <= MIN_TRANSCODE_BYTES:
            return False
        return _duration(dst) > 0.0
    except Exception:
        return False


def _is_protected_source(src: Path) -> bool:
    """Never delete Syncthing temp/partial files."""
    n = src.name.lower()
    return n.startswith("~syncthing~") or n.endswith(".tmp")


def delete_source_if_safe(ctx: "Ctx", src: Path, dst: Path) -> str:
    """Delete the original portal source ONLY when its transcode output verifies.
    Returns a short status string for reporting. Refuses Syncthing temp files.
    NOTE: the portal is a Syncthing send-receive folder; deletions here may be
    re-announced/re-downloaded by the send-only lab node (caveat surfaced by the
    runner, not handled in code)."""
    if not src.is_file():
        return f"skip(no-source): {src.name}"
    if _is_protected_source(src):
        return f"skip(protected syncthing tmp): {src.name}"
    if not transcode_output_ok(dst):
        return f"skip(transcode-not-verified): {src.name}"
    try:
        freed = src.stat().st_size
        src.unlink()
        ctx._deleted_bytes += freed
        ctx._deleted.append((src, freed))
        return f"DELETED {src.name} (+{freed/1e6:.0f} MB)"
    except Exception as e:
        return f"skip(delete-error {e}): {src.name}"


# --- source resolution --------------------------------------------------------
def resolve_sources(proj: Path, portal: Path | None, sources_json: Path | None):
    """Return a list of (src_path, out_path) pairs.

    out_path is <proj>/videos/<stem>.mp4. If sources_json is given it is a JSON
    list of {"src": ..., "out": ...} (out = output stem/name under videos/).
    Else auto-discover *.mp4 recursively under portal -> videos/<stem>.mp4.
    """
    videos = proj / "videos"
    pairs: list[tuple[Path, Path]] = []
    if sources_json:
        data = json.loads(Path(sources_json).read_text(encoding="utf-8"))
        for ent in data:
            src = Path(ent["src"])
            out = ent["out"]
            name = out if out.lower().endswith(".mp4") else out + ".mp4"
            pairs.append((src, videos / Path(name).name))
        return pairs
    if not portal:
        raise SystemExit("ERROR: provide --portal or --sources")
    for src in sorted(Path(portal).rglob("*.mp4")):
        pairs.append((src, videos / f"{src.stem}.mp4"))
    # de-duplicate output names (keep first), warn on collisions
    seen: dict[str, Path] = {}
    uniq: list[tuple[Path, Path]] = []
    for src, out in pairs:
        if out.name in seen:
            step(f"! WARN duplicate output name {out.name}: keeping {seen[out.name]}, skipping {src}")
            continue
        seen[out.name] = src
        uniq.append((src, out))
    return uniq


# --- stage state --------------------------------------------------------------
class Ctx:
    def __init__(self, proj: Path, cohort: str, webhook: str, dry: bool,
                 delete_source: bool = False, filtered: bool = False):
        self.proj = proj
        self.cohort = cohort
        self.webhook = webhook
        self.dry = dry
        self.delete_source = delete_source
        # When True, stage_features runs pose_filter.filter_pose on the DLC h5 before
        # extraction and writes to the FILTERED_FEATURE_HASH cache (opt-in --filter).
        self.filtered = filtered
        self.feat_hash = FILTERED_FEATURE_HASH if filtered else FEATURE_HASH
        self.videos = proj / "videos"
        self.features = proj / "features"
        self.stage_file = proj / "_stage.json"
        self.prog = self.videos / ".ffprog.txt"
        self.done = proj / "PIPELINE_DONE"
        self.dtag = " _[dry test]_" if dry else ""
        self._deleted: list[tuple[Path, int]] = []   # (src, bytes) actually removed
        self._deleted_bytes = 0

    def set_stage(self, **kw):
        kw["updated"] = time.time()
        try:
            self.stage_file.write_text(json.dumps(kw))
        except Exception as e:
            step(f"! could not write _stage.json: {e}")

    def note(self, msg: str):
        post(msg + self.dtag, self.webhook)


# --- stage 1: transcode -------------------------------------------------------
def stage_transcode(ctx: Ctx, pairs):
    ctx.videos.mkdir(parents=True, exist_ok=True)
    n = len(pairs)
    pending = [(s, o) for (s, o) in pairs if not (o.is_file() and o.stat().st_size > 0) and s.is_file()]
    missing = [s for (s, o) in pairs if not s.is_file() and not o.is_file()]
    ctx.note(f"▶ **[{ctx.cohort}] transcode** — {len(pending)} to encode "
             f"({n - len(pending)} done/missing)."
             + (" _[del-source ON]_" if ctx.delete_source else ""))
    if ctx.dry:
        return
    if missing:
        for s in missing:
            step(f"! source missing, skip: {s}")
    for i, (src, dst) in enumerate(pairs, 1):
        if dst.is_file() and dst.stat().st_size > 0:
            step(f"skip transcode (exists): {dst.name}")
            # Output already present from a prior run; honour delete flag if the
            # output verifies (lets a resume reclaim a source it transcoded before).
            if ctx.delete_source:
                step("  " + delete_source_if_safe(ctx, src, dst))
            continue
        if not src.is_file():
            step(f"! source missing, skip: {src}")
            continue
        ctx.set_stage(stage="transcode", video=dst.stem, idx=i, total=n, duration=_duration(src))
        if ctx.prog.exists():
            try:
                ctx.prog.unlink()
            except Exception:
                pass
        step(f"transcode {i}/{n}: {src.name} -> {dst.name}")
        t0 = time.time()
        # -map_metadata 0 + -movflags use_metadata_tags: preserve the source's
        # CUSTOM container tags through the transcode. PawCapture videos carry
        # spatial-calibration tags (mm_per_pixel, pixelpaws_calibrated,
        # pixelpaws_ref_length_mm/ref_pixels) that the classifier mm-per-pixel
        # path consumes; without these flags ffmpeg silently drops all arbitrary
        # MP4 tags. No-op for sources that have none.
        p = subprocess.run(["ffmpeg", "-y", "-i", str(src),
                            "-map_metadata", "0", "-movflags", "use_metadata_tags",
                            "-c:v", CODEC, "-crf", CRF,
                            "-preset", PRESET, "-an", "-progress", str(ctx.prog),
                            "-stats_period", "2", str(dst)], capture_output=True, text=True)
        if p.returncode != 0:
            step(f"! ffmpeg failed {src.name}: {p.stderr[-400:]}")
            if dst.is_file():
                dst.unlink()
            continue
        step(f"  done {(time.time()-t0)/60:.1f} min, {dst.stat().st_size/1e6:.0f} MB")
        # Inline source reclamation: only after this output verifies good.
        if ctx.delete_source:
            step("  " + delete_source_if_safe(ctx, src, dst))


# --- stage 2: DLC -------------------------------------------------------------
DLC_INNER = """
import sys, json
videos = json.loads(sys.argv[1])
import deeplabcut
deeplabcut.analyze_videos(r"{cfg}", videos, shuffle={sh}, videotype=".mp4",
                          save_as_csv=False, batchsize={b})
"""


def stage_dlc(ctx: Ctx, pairs):
    vids = sorted({o for (_, o) in pairs if o.is_file()})
    pending = [v for v in vids if not find_h5(v)]
    ctx.set_stage(stage="dlc", pending=len(pending), total=len(vids))
    ctx.note(f"▶ **[{ctx.cohort}] DLC pose** — {len(pending)} pending "
             f"(of {len(vids)} transcoded).")
    if not pending:
        step("DLC: all tracked")
        return
    if ctx.dry:
        step(f"[dry] would run DLC on {len(pending)} video(s)")
        return
    step(f"DLC: {len(pending)} video(s) @ batch={DLC_BATCH} (one process per video)")
    # One subprocess PER video so a single crash can't kill the whole batch and
    # silently leave everything untracked. A re-run resumes (find_h5 skips done).
    inner = DLC_INNER.format(cfg=str(DLC_CONFIG), sh=SHUFFLE, b=DLC_BATCH)
    ip = Path(os.environ.get("TEMP", ".")) / f"dlc_{uuid.uuid4().hex}.py"
    ip.write_text(inner)
    dprog = ctx.videos / ".dlcprog.txt"          # per-video DLC tqdm capture -> tracker reads it/s
    # .dlcprog.txt is reset every video (the tracker needs the frame count to
    # restart), so the DLC console output of any single video — including a crash
    # traceback — would otherwise vanish. Mirror each video's full output into an
    # append-only _run/dlc.log that survives both the per-video reset and reruns.
    dlclog = ctx.proj / "_run" / "dlc.log"
    try:
        dlclog.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    failed = []
    try:
        for j, v in enumerate(pending, 1):
            ctx.set_stage(stage="dlc", video=v.stem, idx=j, total=len(pending))
            step(f"DLC {j}/{len(pending)}: {v.name}")
            try:
                dprog.write_text("")             # reset so the frame count restarts per video
            except Exception:
                pass
            t0 = time.time()
            with open(dprog, "w", encoding="utf-8", errors="replace") as pf:
                r = subprocess.run([DLC_PYTHON, str(ip), json.dumps([str(v)])],
                                   cwd=str(REPO), env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                   stdout=pf, stderr=subprocess.STDOUT)
            dt = (time.time() - t0) / 60
            ok = r.returncode == 0 and find_h5(v)
            # Archive this video's captured output into the persistent dlc.log.
            try:
                cap = dprog.read_text(encoding="utf-8", errors="replace")
                with open(dlclog, "a", encoding="utf-8", errors="replace") as lf:
                    lf.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                             f"{v.stem} (exit {r.returncode}, {dt:.1f} min, "
                             f"{'OK' if ok else 'FAIL'}) =====\n{cap}\n")
            except Exception:
                pass
            if not ok:
                step(f"! DLC FAILED: {v.name} (exit {r.returncode}, {dt:.1f} min, "
                     f"h5={'yes' if find_h5(v) else 'NO'})")
                # Surface the tail of the failure into the main pipeline.log too.
                try:
                    tail = dprog.read_text(encoding="utf-8", errors="replace")[-1500:]
                    step(f"  DLC output tail [{v.stem}]:\n{tail}")
                except Exception:
                    pass
                failed.append(v.name)
            else:
                step(f"  DLC ok {v.stem} ({dt:.1f} min)")
    finally:
        ip.unlink(missing_ok=True)
    if failed:
        step(f"⚠ DLC failed on {len(failed)}/{len(pending)} video(s): {failed}")
        ctx.note(f"⚠️ [{ctx.cohort}] DLC failed on {len(failed)}/{len(pending)} "
                 f"(e.g. {failed[:4]}). Re-run to retry — done videos skip.")


# --- stage 3: features --------------------------------------------------------
def _verify_pkl(pkl: Path) -> bool:
    """First produced pkl must have EXPECTED_FEATURE_COLS and non-all-zero
    brightness + optical-flow columns (guards a wrong DLC model)."""
    try:
        with open(pkl, "rb") as f:
            X = pickle.load(f)
    except Exception as e:
        step(f"VERIFY FAIL: could not load {pkl.name}: {e}")
        return False
    cols = list(getattr(X, "columns", []))
    nc = len(cols)
    ok_cols = nc == EXPECTED_FEATURE_COLS
    bri = [c for c in cols if "Bright" in c or c.startswith("Pix_")]
    flo = [c for c in cols if "Flow" in c]
    ok_bri = bool(bri) and bool(X[bri].abs().to_numpy().sum())
    ok_flo = bool(flo) and bool(X[flo].abs().to_numpy().sum())
    step(f"VERIFY {pkl.name}: cols={nc} (expected {EXPECTED_FEATURE_COLS}) "
         f"-> {'OK' if ok_cols else 'FAIL'}; "
         f"brightness({len(bri)}) {'non-zero' if ok_bri else 'ALL-ZERO/MISSING'}; "
         f"optical-flow({len(flo)}) {'non-zero' if ok_flo else 'ALL-ZERO/MISSING'}")
    passed = ok_cols and ok_bri and ok_flo
    step(f"VERIFY {'PASS' if passed else 'FAIL'}")
    return passed


def stage_features(ctx: Ctx, pairs):
    vids = sorted({o for (_, o) in pairs if o.is_file()})
    n = len(vids)
    ctx.features.mkdir(parents=True, exist_ok=True)
    pending = [v for v in vids if not feature_pkl(ctx.features, v, ctx.feat_hash).is_file()]
    ftag = " _[pose-FILTERED]_" if ctx.filtered else ""
    ctx.note(f"▶ **[{ctx.cohort}] features** — {len(pending)} pending (of {n} transcoded).{ftag}")
    if ctx.dry:
        step(f"[dry] would extract features ({ctx.feat_hash}) for {len(pending)} video(s)")
        return
    sys.path.insert(0, str(REPO))
    import contextlib
    from prediction_pipeline import PixelPaws_ExtractFeatures
    _filter_pose = None
    if ctx.filtered:
        from pose_filter import filter_pose as _filter_pose   # jump-gate defaults
        import pandas as _pd
    fprog = ctx.videos / ".featprog.txt"          # extractor tqdm capture -> tracker reads frames/s
    first_new: Path | None = None
    for i, v in enumerate(vids, 1):
        cache = feature_pkl(ctx.features, v, ctx.feat_hash)
        if cache.is_file():
            step(f"features exist: {cache.name}")
            continue
        h5 = find_h5(v)
        if not h5:
            step(f"! no h5 for {v.name}, skip features")
            continue
        ctx.set_stage(stage="features", video=v.stem, idx=i, total=n)
        step(f"features {i}/{n}: {v.stem}{' [filtered]' if ctx.filtered else ''}")
        pose_path = str(h5)
        tmp_h5 = None
        if ctx.filtered:                          # clean the pose into a temp _ppfilt.h5 first
            df = _pd.read_hdf(h5)
            cleaned, _st = _filter_pose(df)
            tmp_h5 = Path(str(h5)[:-3] + "_ppfilt.h5")
            cleaned.to_hdf(tmp_h5, key="df_with_missing", mode="w")
            pose_path = str(tmp_h5)
        try:
            with open(fprog, "w", encoding="utf-8", errors="replace") as _fp, \
                    contextlib.redirect_stdout(_fp), contextlib.redirect_stderr(_fp):
                X = PixelPaws_ExtractFeatures(pose_data_file=pose_path, video_file_path=str(v),
                                              bp_include_list=None, config_yaml_path=None, **FEATURE_CFG)
        finally:
            if tmp_h5 is not None:                # don't leave the temp pose in videos/ (find_h5 excludes it anyway)
                try: tmp_h5.unlink(missing_ok=True)
                except Exception: pass
        tmp = cache.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(X, f)
        os.replace(tmp, cache)            # atomic
        if first_new is None:
            first_new = cache
    # Verify the first pkl PRODUCED this run; if none produced, verify any existing.
    target = first_new
    if target is None:
        existing = [feature_pkl(ctx.features, v, ctx.feat_hash) for v in vids
                    if feature_pkl(ctx.features, v, ctx.feat_hash).is_file()]
        target = existing[0] if existing else None
    if target is not None:
        ok = _verify_pkl(target)
        ctx.note(f"{'✅' if ok else '❌'} [{ctx.cohort}] feature verify "
                 f"{'PASS' if ok else 'FAIL'} on `{target.name}`")
    else:
        step("VERIFY: no feature pkl to check")


# --- orchestration ------------------------------------------------------------
def run(cohort: str, proj, portal=None, sources=None,
        stages=ALL_STAGES, webhook: str = PROCESSING_WEBHOOK, dry: bool = False,
        delete_source: bool = False, filtered: bool = False):
    """Importable entry point. Returns the resolved (src, out) pairs."""
    proj = Path(proj)
    lp = open_run_log(proj)
    ctx = Ctx(proj, cohort, webhook, dry, delete_source=delete_source, filtered=filtered)
    pairs = resolve_sources(proj, portal, sources)
    stages = tuple(s for s in stages if s in ALL_STAGES)
    step("=" * 70)
    step(f"RUN START cohort={cohort} pid={os.getpid()} dry={dry} "
         f"stages={','.join(stages)} sources={len(pairs)} "
         f"del_source={delete_source} filtered={filtered}")
    step(f"  proj={proj}")
    step(f"  DLC python={DLC_PYTHON} | config={DLC_CONFIG} | shuffle={SHUFFLE} "
         f"batch={DLC_BATCH} | feature_hash={ctx.feat_hash}")
    if lp:
        step(f"  log -> {lp}")

    if dry:
        step(f"=== DRY PLAN [{cohort}] — {len(pairs)} source(s), stages={','.join(stages)} ===")
        if delete_source:
            step("  [del-source ON] originals deleted inline after each verified transcode")
        for src, out in pairs:
            acts = []
            if "transcode" in stages:
                done = out.is_file() and out.stat().st_size > 0
                miss = "" if (done or src.is_file()) else " (SRC MISSING)"
                acts.append(f"transcode={'SKIP' if done else 'RUN'}{miss}")
            if "dlc" in stages:
                acts.append(f"dlc={'SKIP' if find_h5(out) else 'RUN'}")
            if "features" in stages:
                acts.append(f"features={'SKIP' if feature_pkl(ctx.features, out, ctx.feat_hash).is_file() else 'RUN'}")
            step(f"  {out.name:42s} | " + "  ".join(acts))
        step("=== DRY PLAN DONE (no compute run) ===")
        return pairs

    if ctx.done.exists():
        try:
            ctx.done.unlink()
        except Exception:
            pass
    ctx.set_stage(stage="starting")
    try:
        if "transcode" in stages:
            step("=== TRANSCODE ===")
            stage_transcode(ctx, pairs)
        if "dlc" in stages:
            step("=== DLC ===")
            stage_dlc(ctx, pairs)
        if "features" in stages:
            step("=== FEATURES ===")
            stage_features(ctx, pairs)
    except BaseException as e:
        # Record the crash before it propagates — the whole point of the run log.
        step(f"!!! RUN CRASHED ({type(e).__name__}): {e}")
        step(traceback.format_exc())
        ctx.set_stage(stage="crashed", error=f"{type(e).__name__}: {e}")
        try:
            ctx.note(f"💥 **[{cohort}] pipeline CRASHED** — {type(e).__name__}: {e}. "
                     f"See `_run/pipeline.log`.")
        except Exception:
            pass
        raise
    ctx.set_stage(stage="done")
    ctx.done.write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
    step("PIPELINE DONE")
    if ctx.delete_source:
        step(f"SOURCE CLEANUP: deleted {len(ctx._deleted)} original(s), "
             f"freed {ctx._deleted_bytes/1e9:.2f} GB")
        ctx.note(f"🧹 [{cohort}] deleted {len(ctx._deleted)} source original(s), "
                 f"freed {ctx._deleted_bytes/1e9:.2f} GB.")
    ctx.note(f"✅ **[{cohort}] pipeline DONE** ({len(pairs)} videos).")
    return pairs


def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Reusable PixelPaws pipeline engine "
                                             "(transcode -> DLC -> features).")
    ap.add_argument("--cohort", required=True, help="cohort name, e.g. 2606_AOW")
    ap.add_argument("--proj", required=True, help="project dir, e.g. D:\\PixelPaws_Active\\2606_AOW")
    ap.add_argument("--portal", help="raw source videos dir (auto-discover *.mp4 recursively)")
    ap.add_argument("--sources", help="optional sources.json: [{src,out}, ...]")
    ap.add_argument("--stages", default="transcode,dlc,features",
                    help="comma list of stages to run (default all three)")
    ap.add_argument("--webhook", default=PROCESSING_WEBHOOK, help="Discord webhook URL")
    ap.add_argument("--dry", action="store_true",
                    help="resolve sources + print RUN/SKIP plan, no compute (does not import DLC)")
    ap.add_argument("--delete-source-after-transcode", dest="delete_source",
                    action="store_true",
                    help="OPT-IN: delete each original source file inline immediately after its "
                         "transcode output is verified (exists, >1MB, ffprobe-valid). Never deletes "
                         "~syncthing~/.tmp files. Default OFF so other cohorts are unaffected.")
    ap.add_argument("--filter", dest="filtered", action="store_true",
                    help="OPT-IN: run pose_filter (jump/teleport gate) on each DLC h5 before feature "
                         "extraction, writing to the FILTERED_FEATURE_HASH cache "
                         "(<stem>_features_8aed1c22f.pkl). Leaves the canonical 8aed1c22 cache untouched.")
    ap.add_argument("--dlc-config", dest="dlc_config",
                    help="OPT-IN: run the DLC stage against this config.yaml instead of pp_config's "
                         "DLC_CONFIG. Use with --shuffle to select a non-default network for one "
                         "cohort without changing the default for every other cohort.")
    ap.add_argument("--shuffle", type=int,
                    help="OPT-IN: DLC shuffle index, overriding pp_config's SHUFFLE. Also selects "
                         "which *shuffle<N>*.h5 find_h5 treats as this run's output, so a cohort "
                         "analyzed under two networks keeps both sets of tracks distinct.")
    return ap.parse_args(argv)


def apply_dlc_override(dlc_config=None, shuffle=None):
    """Rebind the module-level DLC_CONFIG / SHUFFLE used by stage_dlc and find_h5.

    Both are read at call time, so rebinding here is enough. Kept as a function
    (rather than threading them through run()) so the tracker and any importing
    caller get the same override semantics as the CLI."""
    global DLC_CONFIG, SHUFFLE
    if dlc_config:
        p = Path(dlc_config)
        if not p.is_file():
            raise SystemExit(f"--dlc-config not found: {p}")
        DLC_CONFIG = str(p)
    if shuffle is not None:
        SHUFFLE = int(shuffle)


def main(argv=None):
    a = _parse_args(argv)
    apply_dlc_override(a.dlc_config, a.shuffle)
    stages = tuple(s.strip() for s in a.stages.split(",") if s.strip())
    run(cohort=a.cohort, proj=a.proj, portal=a.portal, sources=a.sources,
        stages=stages, webhook=a.webhook, dry=a.dry, delete_source=a.delete_source,
        filtered=a.filtered)


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    main()
