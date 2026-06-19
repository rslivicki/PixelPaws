"""
frame_rate_dialog.py — Tk dialogs for frame-rate diagnostics + normalization.

Two entry points used by PixelPaws_GUI Tools menu:

  open_diagnose_dialog(parent, project_folder)
  open_normalize_dialog(parent, project_folder)

Both run their work in a worker thread so the UI stays responsive,
streaming progress lines into a scrollable text widget. The normalize
dialog defaults to dry-run; the user must explicitly tick "Apply
changes" to commit.
"""
from __future__ import annotations

import threading
import traceback
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, scrolledtext

try:
    from ui_utils import FONT_FAMILY
except ImportError:
    FONT_FAMILY = 'Segoe UI'


# ============================================================================
# Diagnose dialog
# ============================================================================

def open_diagnose_dialog(parent, project_folder: str) -> None:
    """Run the duplicate-frame diagnostic on a project's videos/."""
    if not project_folder or not Path(project_folder).is_dir():
        messagebox.showerror('Frame Rate Diagnostic',
                             'No project folder is loaded.')
        return

    win = tk.Toplevel(parent)
    win.title('Frame Rate Diagnostic')
    win.geometry('760x520')

    head = ttk.Frame(win, padding=10)
    head.pack(fill='x')
    ttk.Label(head, text=f'Project: {project_folder}',
              font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w')
    ttk.Label(head,
              text=('Scans every video in <project>/videos/ for held/duplicated\n'
                    'frames. A summary JSON is saved to <project>/diagnostics/.'),
              foreground='gray').pack(anchor='w', pady=(4, 0))

    opts = ttk.Frame(win, padding=(10, 4))
    opts.pack(fill='x')
    ttk.Label(opts, text='Max frames per video (0 = full scan):').grid(row=0, column=0, sticky='w')
    max_var = tk.StringVar(value='1500')
    ttk.Entry(opts, textvariable=max_var, width=8).grid(row=0, column=1, sticky='w', padx=4)

    log = scrolledtext.ScrolledText(win, wrap='word', height=18,
                                    font=('Consolas', 9))
    log.pack(fill='both', expand=True, padx=10, pady=(6, 6))

    btns = ttk.Frame(win, padding=10)
    btns.pack(fill='x')
    run_btn = ttk.Button(btns, text='Run Diagnostic')
    run_btn.pack(side='left')
    ttk.Button(btns, text='Close', command=win.destroy).pack(side='right')

    def append(msg: str) -> None:
        log.configure(state='normal')
        log.insert('end', msg + '\n')
        log.see('end')
        log.configure(state='disabled')

    def worker() -> None:
        try:
            try:
                max_frames = int(max_var.get().strip())
            except ValueError:
                max_frames = 0
            mf = max_frames if max_frames > 0 else None

            # Lazy import so the GUI doesn't pay the cv2 cost until needed.
            import sys
            sys.path.insert(0, str(Path(__file__).parent / 'scripts' / 'utilities'))
            try:
                import diagnose_duplicate_frames as diag  # type: ignore
            finally:
                sys.path.pop(0)

            videos = diag._gather_project_videos(Path(project_folder))
            if not videos:
                win.after(0, append, '! No videos found under videos/.')
                return

            win.after(0, append, f'Scanning {len(videos)} video(s)'
                                 + (f' (capped at {mf} frames each)...' if mf else '...'))

            records = []
            for v in videos:
                win.after(0, append, f'- {v.name} ...')
                rec = diag.diagnose_video(v, max_frames=mf)
                records.append(rec)
                if 'error' in rec:
                    win.after(0, append, f'    error: {rec["error"]}')
                    continue
                win.after(0, append,
                          f'    stored_fps={rec["stored_fps"]:.1f}    '
                          f'inferred_true_fps='
                          f'{rec["inferred_true_fps"]}    '
                          f'dup_frac={rec["duplicate_fraction"]:.3f}    '
                          f'pattern={rec["pattern"]}')

            valid = [r for r in records if 'error' not in r]
            if valid:
                from collections import Counter
                patterns = Counter(r['pattern'] for r in valid)
                win.after(0, append, '')
                win.after(0, append, f'Summary: {dict(patterns)}')

            # Persist report
            from datetime import datetime
            import json
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            out_dir = Path(project_folder) / 'diagnostics'
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f'dup_frames_{ts}.json'
            out_path.write_text(json.dumps({
                'generated_at': ts,
                'project': str(project_folder),
                'max_frames': mf,
                'videos': records,
            }, indent=2), encoding='utf-8')
            win.after(0, append, f'Report: {out_path}')
        except Exception:
            win.after(0, append, '! Diagnostic failed:')
            win.after(0, append, traceback.format_exc())
        finally:
            win.after(0, lambda: run_btn.configure(state='normal'))

    def start():
        log.configure(state='normal')
        log.delete('1.0', 'end')
        log.configure(state='disabled')
        run_btn.configure(state='disabled')
        threading.Thread(target=worker, daemon=True).start()

    run_btn.configure(command=start)


# ============================================================================
# Normalize dialog
# ============================================================================

def open_normalize_dialog(parent, project_folder: str) -> None:
    """Trim duplicate frames out of every video + remap labels/DLC h5."""
    if not project_folder or not Path(project_folder).is_dir():
        messagebox.showerror('Normalize Frame Rate',
                             'No project folder is loaded.')
        return

    win = tk.Toplevel(parent)
    win.title('Normalize Project Frame Rate')
    win.geometry('780x600')

    head = ttk.Frame(win, padding=10)
    head.pack(fill='x')
    ttk.Label(head, text=f'Project: {project_folder}',
              font=(FONT_FAMILY, 10, 'bold')).pack(anchor='w')
    ttk.Label(head,
              text=('Trims duplicate/held frames from every video and re-mirrors the\n'
                    'trim into DLC .h5, label CSV, sparse DB, and dense regions.\n'
                    'Originals are moved to <project>/_pre_fps_normalize_<ts>/.\n'
                    'Default is DRY RUN — must explicitly tick "Apply changes" to commit.'),
              foreground='gray').pack(anchor='w', pady=(4, 0))

    opts = ttk.Frame(win, padding=(10, 4))
    opts.pack(fill='x')

    ttk.Label(opts, text='Target output fps:').grid(row=0, column=0, sticky='w', pady=2)
    fps_var = tk.StringVar(value='30.0')
    ttk.Entry(opts, textvariable=fps_var, width=8).grid(row=0, column=1, sticky='w', padx=4)

    ttk.Label(opts, text='Duplicate threshold eps (mean |Δ|):').grid(row=1, column=0, sticky='w', pady=2)
    eps_var = tk.StringVar(value='0.5')
    ttk.Entry(opts, textvariable=eps_var, width=8).grid(row=1, column=1, sticky='w', padx=4)

    apply_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(opts,
                    text='Apply changes (UNCHECK for dry-run preview)',
                    variable=apply_var).grid(row=2, column=0, columnspan=3,
                                              sticky='w', pady=(8, 2))

    log = scrolledtext.ScrolledText(win, wrap='word', height=22,
                                    font=('Consolas', 9))
    log.pack(fill='both', expand=True, padx=10, pady=(6, 6))

    btns = ttk.Frame(win, padding=10)
    btns.pack(fill='x')
    run_btn = ttk.Button(btns, text='Run')
    run_btn.pack(side='left')
    ttk.Button(btns, text='Close', command=win.destroy).pack(side='right')

    def append(msg: str) -> None:
        log.configure(state='normal')
        log.insert('end', msg + '\n')
        log.see('end')
        log.configure(state='disabled')

    def worker(target_fps: float, eps: float, dry_run: bool) -> None:
        try:
            from frame_rate_normalize import normalize_project
            normalize_project(
                project_folder,
                target_fps=target_fps,
                eps=eps,
                dry_run=dry_run,
                progress=lambda m: win.after(0, append, m),
            )
            mode = 'DRY RUN' if dry_run else 'APPLIED'
            win.after(0, append, f'\n[{mode}] complete.')
        except Exception:
            win.after(0, append, '\n! Normalize failed:')
            win.after(0, append, traceback.format_exc())
        finally:
            win.after(0, lambda: run_btn.configure(state='normal'))

    def start():
        try:
            tfps = float(fps_var.get().strip())
            eps = float(eps_var.get().strip())
        except ValueError:
            messagebox.showerror('Normalize Frame Rate',
                                 'target_fps and eps must be numbers.')
            return

        will_apply = apply_var.get()
        if will_apply:
            if not messagebox.askyesno(
                    'Confirm Apply',
                    f'This will rewrite every video, DLC .h5, and labels CSV in:\n\n'
                    f'  {project_folder}\n\n'
                    f'Originals are backed up to _pre_fps_normalize_<ts>/.\n'
                    f'This may take hours depending on project size.\n\n'
                    f'Proceed?'):
                return

        log.configure(state='normal')
        log.delete('1.0', 'end')
        log.configure(state='disabled')
        run_btn.configure(state='disabled')
        threading.Thread(target=worker,
                         args=(tfps, eps, not will_apply),
                         daemon=True).start()

    run_btn.configure(command=start)
