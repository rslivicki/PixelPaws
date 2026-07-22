"""
Recover a moov-less MP4 written by PawCapture (ftyp + mdat, no moov index).

The mdat payload is H.264 in AVCC form (4-byte length-prefixed NAL units),
with SPS/PPS inline before every IDR keyframe. Some files also contain stray
16-byte `mdat` box headers spliced between GOPs, which break a naive
length-walk. We convert the elementary stream to Annex-B (start-code
prefixed), using the inline SPS occurrences as resync anchors: whenever a NAL
length is implausible or would overshoot the next keyframe, we snap forward to
the next SPS. The Annex-B stream is then remuxed by ffmpeg into a fresh MP4
with a valid index.

Usage:
    python recover_moovless_h264.py INPUT.mp4 --dry-run
    python recover_moovless_h264.py INPUT.mp4 | ffmpeg -r 60 -f h264 -i - \
        -c copy -movflags +faststart OUTPUT.mp4
"""
import sys, os, struct, argparse

SPS_SIG = b"\x00\x00\x00\x29\x27\x4d\x00\x1f"  # len=41 + SPS nalbyte 0x27, this rig
START = b"\x00\x00\x00\x01"
FTYP_LEN = 32
MDAT_HDR_LEN = 16  # size(1)=4 + 'mdat'=4 + 64-bit size=8


def find_sps_anchors(path, payload_start, log=sys.stderr):
    """Return sorted list of file offsets where a length-prefixed SPS begins
    (offset points at the 4-byte length, i.e. the NAL boundary)."""
    anchors = []
    CHUNK = 64 * 1024 * 1024
    overlap = len(SPS_SIG)
    with open(path, "rb") as f:
        f.seek(payload_start)
        base = payload_start
        carry = b""
        while True:
            buf = f.read(CHUNK)
            if not buf:
                break
            hay = carry + buf
            i = -1
            while True:
                i = hay.find(SPS_SIG, i + 1)
                if i < 0:
                    break
                anchors.append(base - len(carry) + i)
            carry = hay[-overlap:]
            base += len(buf)
    return anchors


def convert(path, out, dry_run=False, log=sys.stderr):
    size = os.path.getsize(path)
    payload_start = FTYP_LEN + MDAT_HDR_LEN  # 48
    anchors = find_sps_anchors(path, payload_start)
    print(f"# {os.path.basename(path)}: {size:,} bytes, {len(anchors)} SPS keyframes",
          file=log)
    anchor_set = anchors
    n_anchor = len(anchor_set)
    # map for "next anchor strictly greater than pos"
    import bisect

    nals = 0
    resyncs = 0
    skipped = 0
    emitted = 0
    frames_est = 0
    pos = anchors[0] if anchors else payload_start

    with open(path, "rb") as f:
        while pos < size:
            # next keyframe boundary after pos
            j = bisect.bisect_right(anchor_set, pos)
            next_anchor = anchor_set[j] if j < n_anchor else size

            f.seek(pos)
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            ln = struct.unpack(">I", hdr)[0]
            nal_end = pos + 4 + ln
            plausible = (0 < ln <= 50_000_000) and nal_end <= next_anchor
            if not plausible:
                # resync: jump to next anchor (skips stray mdat headers / glitch tails)
                if next_anchor >= size:
                    break
                skipped += next_anchor - pos
                pos = next_anchor
                resyncs += 1
                continue
            nalbyte = f.read(1)[0]
            ntype = nalbyte & 0x1F
            if not dry_run:
                out.write(START)
                # write nalbyte + remaining NAL body
                out.write(bytes([nalbyte]))
                remaining = ln - 1
                while remaining > 0:
                    chunk = f.read(min(1 << 20, remaining))
                    if not chunk:
                        break
                    out.write(chunk)
                    remaining -= len(chunk)
            if ntype in (1, 5):  # VCL slice ~ one per frame here
                frames_est += 1
            emitted += 4 + ln
            nals += 1
            pos = nal_end

    print(f"# NALs emitted: {nals:,}  (~{frames_est:,} frames, ~{frames_est/60.0:.1f}s @60fps)",
          file=log)
    print(f"# resync events: {resyncs}   bytes skipped: {skipped:,}", file=log)
    print(f"# annexb bytes: {emitted:,}", file=log)
    return dict(nals=nals, frames=frames_est, resyncs=resyncs, skipped=skipped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input")
    ap.add_argument("--dry-run", action="store_true",
                    help="analyze only; do not emit Annex-B to stdout")
    args = ap.parse_args()
    out = None if args.dry_run else sys.stdout.buffer
    convert(args.input, out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
