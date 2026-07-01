# PawCapture

Multi-camera recording controller for **See3CAM_CU27** USB cameras. Runs 3–4 cameras
side by side, records each to its own hardware-encoded MP4 (GPU where available, CPU
fallback), and writes a session manifest for the downstream **PixelPaws** analysis tool.

Version **0.6.0** · Windows 10/11 (64-bit)

---

## Requirements

- **Windows 10 or 11**, 64-bit
- **3–4 × See3CAM_CU27** USB cameras
  - Use **USB 3.0** ports. For 4 cameras, spread them across separate USB controllers/hubs
    if you can — four 1080p streams is a lot of bus bandwidth.
- A GPU with QSV / NVENC / AMF is used automatically for encoding when present; otherwise
  it falls back to CPU (x264).
- **No Python and no separate FFmpeg install needed** — both builds below are fully bundled.

---

## Install

### Option A — Installer (recommended)

1. Run **`CamSyncPro_Setup_v0.6.0.exe`**.
2. Follow the wizard (default install location is fine).
3. Launch **PawCapture** from the Start menu or desktop shortcut.

### Option B — Portable (no install)

1. Unzip **`PawCapture-0.6.0.zip`** anywhere (e.g. the desktop).
2. Open the unzipped `PawCapture` folder and run **`PawCapture.exe`**.

Nothing is written to system folders in portable mode — the whole app lives in that folder.

---

## First run

1. Plug in the cameras, then start PawCapture.
2. In each camera panel, pick the device from the **Device** dropdown and click **CONNECT**.
   - The dropdown labels cameras by serial; each label opens that exact camera.
3. Set the **Folder** and **File** name per camera (or set one camera's folder and click
   **⇲ all** to apply it to every camera).
4. Pick a **Phase** (Baseline / Post-Drug / etc.) in the top bar if you want phase-tagged
   filenames and subfolders.
5. Click **RECORD ALL** to start every connected camera, **STOP ALL** to end.

Recordings are saved under **`%USERPROFILE%\PawCapture\recordings\YYYY-MM-DD\`** by default,
each session with a `session_*.json` manifest alongside the videos.

---

## Notes & troubleshooting

- **Filenames won't be overwritten.** If a target file already exists, recording is refused
  with a message — change the File name, the suffix, or the phase and try again.
- **Camera shows a tiled / scrambled image:** unplug and replug that camera's USB cable to
  reset its image processor, then reconnect in the app.
- **A camera won't open ("device in use"):** make sure no other app (OBS, Camera, etc.) has
  it open, and that each panel is pointed at a different camera.
- **Logs** are written to `%USERPROFILE%\PawCapture\logs\` (including `crash.log` if the app
  ever exits unexpectedly) — handy when reporting an issue.

---

## Camera settings profiles

Use **SAVE** / **LOAD** in the top bar to store and recall a full multi-camera setup
(devices, resolution/fps, exposure/gain, crop, calibration, folders). Profiles live in
`%USERPROFILE%\PawCapture\profiles\`.
