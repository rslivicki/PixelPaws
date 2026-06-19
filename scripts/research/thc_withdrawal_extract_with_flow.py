"""
Re-extract THC withdrawal features with optical flow + 3-element
square_size, matching Habby's config so the same Scratching classifier
will run on either project. Outputs a new hash; old hash files are
preserved for now.

Posts a live Discord progress message that edits per video — same pattern
as the DLC orchestrator (silent edits, no notification spam).
"""
from __future__ import annotations

import json
import os
import pickle
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(r'E:\PixelPaws')
sys.path.insert(0, str(REPO))

PROJECT = Path(r'E:\RSVIDS\Blackbox\260506_RS_THC_Withdrawal')
VIDEO_DIR = PROJECT / 'Videos'
FEATURES_DIR = PROJECT / 'features'
FEATURES_DIR.mkdir(parents=True, exist_ok=True)

WEBHOOK = (
    ""
)
BAR_WIDTH = 24


def step(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def _send(url, payload, method='POST'):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode('utf-8'),
        method=method,
        headers={'Content-Type': 'application/json',
                 'User-Agent': 'PixelPaws-Chain/1.0'},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read()
            return json.loads(body) if body else None
    except Exception as e:
        print(f'  ! Discord {method} failed: {e}', flush=True)
        return None


def discord_create(text):
    resp = _send(WEBHOOK + '?wait=true', {'content': text})
    return resp.get('id') if resp else None


def discord_edit(msg_id, text):
    if not msg_id:
        return
    _send(f'{WEBHOOK}/messages/{msg_id}', {'content': text}, method='PATCH')


def post_to_discord(text):
    _send(WEBHOOK, {'content': text})


def make_bar(cur, total, width=BAR_WIDTH):
    if total <= 0:
        return '░' * width
    filled = max(0, min(width, cur * width // total))
    return '█' * filled + '░' * (width - filled)


def fmt_dur(seconds):
    s = int(seconds)
    if s < 60:
        return f'{s}s'
    m, s = divmod(s, 60)
    if m < 60:
        return f'{m}m{s:02d}s'
    h, m = divmod(m, 60)
    return f'{h}h{m:02d}m'


def progress_text(done, total, t_start, current=None, status=None,
                  cached=0, extracted=0):
    bar = make_bar(done, total)
    pct = 100 * done / total if total else 0
    elapsed = time.time() - t_start
    extracted_done = max(1, extracted) if extracted else 1
    if extracted > 0 and done < total:
        rate = elapsed / extracted_done   # only count extractions for ETA
        remaining = total - done
        # remaining minus the cached count we'd skip — but we don't know
        # in advance which remain are cached, so approximate as all-extract
        eta = rate * remaining
        eta_str = f'~{fmt_dur(eta)}'
    elif done >= total:
        eta_str = 'done'
    else:
        eta_str = '...'
    lines = [
        '**THC features (with optical flow) -- extraction progress**',
        f'`[{bar}] {done}/{total} videos  ({pct:5.1f}%)  '
        f'(cached: {cached}, extracted: {extracted})`',
    ]
    if current:
        lines.append(f'Current: `{current}`{(" -- " + status) if status else ""}')
    lines.append(f'Elapsed: {fmt_dur(elapsed)} | ETA: {eta_str}')
    return '\n'.join(lines)


def main():
    from prediction_pipeline import PixelPaws_ExtractFeatures
    from feature_cache import FeatureCacheManager

    cfg = {
        'bp_include_list': None,
        'bp_pixbrt_list': ['hrpaw', 'hlpaw', 'snout'],
        'square_size': [40, 40, 40],
        'pix_threshold': 0.3,
        'include_optical_flow': True,
        'bp_optflow_list': ['hrpaw', 'hlpaw', 'snout'],
    }
    cfg_hash = FeatureCacheManager.compute_hash(cfg)
    step(f'Target hash: {cfg_hash} (with optical flow)')

    pairs = []
    for v in sorted(VIDEO_DIR.glob('*.mp4')):
        h5 = list(v.parent.glob(f'{v.stem}*shuffle9*filtered.h5'))
        if not h5:
            h5 = list(v.parent.glob(f'{v.stem}*shuffle9*.h5'))
        if h5:
            pairs.append((v, h5[0]))
    step(f'Sessions: {len(pairs)}')

    total = len(pairs)
    t0_all = time.time()
    n_done = 0
    n_cached = 0
    n_extracted = 0
    msg_id = discord_create(progress_text(0, total, t0_all))
    step(f'Discord progress msg id: {msg_id}')

    for v, h5 in pairs:
        base = v.stem
        out = FEATURES_DIR / f'{base}_features_{cfg_hash}.pkl'
        if out.is_file():
            step(f'  skip (exists): {out.name}')
            n_done += 1
            n_cached += 1
            discord_edit(msg_id, progress_text(
                n_done, total, t0_all, current=base, status='cached',
                cached=n_cached, extracted=n_extracted,
            ))
            continue
        step(f'\n=== {base} ===')
        discord_edit(msg_id, progress_text(
            n_done, total, t0_all, current=base, status='extracting...',
            cached=n_cached, extracted=n_extracted,
        ))
        t0 = time.time()
        X = PixelPaws_ExtractFeatures(
            pose_data_file=str(h5),
            video_file_path=str(v),
            bp_include_list=None,
            bp_pixbrt_list=['hrpaw', 'hlpaw', 'snout'],
            square_size=[40, 40, 40],
            pix_threshold=0.3,
            config_yaml_path=None,
            include_optical_flow=True,
            bp_optflow_list=['hrpaw', 'hlpaw', 'snout'],
        )
        with open(out, 'wb') as f:
            pickle.dump(X, f)
        elapsed_v = time.time() - t0
        step(f'  cached ({elapsed_v:.1f}s, X={X.shape}) -> {out.name}')
        n_done += 1
        n_extracted += 1
        discord_edit(msg_id, progress_text(
            n_done, total, t0_all, current=base,
            status=f'extracted ({elapsed_v:.0f}s, shape {X.shape})',
            cached=n_cached, extracted=n_extracted,
        ))

    elapsed = (time.time() - t0_all) / 60
    final = progress_text(n_done, total, t0_all,
                          cached=n_cached, extracted=n_extracted)
    discord_edit(msg_id, final)
    msg = (f'Feature extraction with flow complete. '
           f'{n_done}/{total} cached at hash {cfg_hash} '
           f'({n_extracted} newly extracted, {n_cached} from cache). '
           f'Total {elapsed:.1f} min.')
    step(msg)
    post_to_discord(msg)
    return 0


if __name__ == '__main__':
    sys.exit(main())
