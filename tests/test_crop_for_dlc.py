# -*- coding: utf-8 -*-
"""crop_for_dlc: a start/end trim yields exactly end - start, and PawCapture
calibration tags survive the crop."""
import json
import os
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import crop_for_dlc as cfd  # noqa: E402

needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None,
    reason="ffmpeg/ffprobe not on PATH")


def test_end_before_start_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        cfd.crop_video_ffmpeg("in.mp4", str(tmp_path / "out.mp4"), 0, 0, 10, 10,
                              start_time=20.0, end_time=10.0)


@needs_ffmpeg
def test_trim_duration_and_tags(tmp_path):
    src = tmp_path / "src.mp4"
    subprocess.run(["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
                    "-i", "testsrc=duration=6:size=96x96:rate=30",
                    "-c:v", "libx264", "-preset", "ultrafast",
                    "-metadata", "mm_per_pixel=0.5",
                    "-metadata", "pixelpaws_calibrated=1",
                    "-movflags", "use_metadata_tags", str(src)], check=True)
    out = tmp_path / "out.mp4"
    # 3-5 s: the old "-ss 3 -i src -to 5" ran from 3 s to the end (3 s long)
    cfd.crop_video_ffmpeg(str(src), str(out), 8, 8, 64, 64,
                          start_time=3.0, end_time=5.0)
    fmt = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:format_tags",
         "-of", "json", str(out)], capture_output=True, text=True, check=True).stdout)["format"]
    assert abs(float(fmt["duration"]) - 2.0) < 0.15
    assert fmt["tags"]["mm_per_pixel"] == "0.5"
    assert fmt["tags"]["pixelpaws_calibrated"] == "1"
