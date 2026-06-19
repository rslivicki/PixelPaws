"""Cleanup: merge the standalone `from ui_utils import FONT_FAMILY` line
that the codemod left behind into the existing multi-line / single-line
`from ui_utils import (...)` import in each file."""

from __future__ import annotations

import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))


def consolidate(src: str) -> str:
    """If the file has both:
        from ui_utils import (a, b)        (or single-line equivalent)
        from ui_utils import FONT_FAMILY
    merge them into one and remove the standalone line."""
    standalone_re = re.compile(
        r'^from ui_utils import FONT_FAMILY\s*$\n?',
        re.MULTILINE,
    )
    if not standalone_re.search(src):
        return src

    # Multi-line tuple form
    multi_re = re.compile(
        r'(from\s+ui_utils\s+import\s*\([^)]*?)(\))',
        re.DOTALL,
    )
    m = multi_re.search(src)
    if m and 'FONT_FAMILY' not in m.group(1):
        before, close = m.group(1), m.group(2)
        merged = before.rstrip().rstrip(',') + ',\n                      FONT_FAMILY' + close
        src = src.replace(m.group(0), merged, 1)
        src = standalone_re.sub('', src, count=1)
        return src

    # Single-line form: from ui_utils import a, b
    single_re = re.compile(r'^from\s+ui_utils\s+import\s+(?!\(|FONT_FAMILY).+$', re.MULTILINE)
    s = single_re.search(src)
    if s and 'FONT_FAMILY' not in s.group(0):
        new_line = s.group(0).rstrip() + ', FONT_FAMILY'
        src = src.replace(s.group(0), new_line, 1)
        src = standalone_re.sub('', src, count=1)
        return src

    return src


def main():
    skip_files = {'ui_utils.py', '_codemod_arial.py', '_consolidate_imports.py'}
    skip_dirs = {'tests', '__pycache__', '_audit_screenshots'}

    n_changed = 0
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs
                   if d not in skip_dirs and not d.startswith('backup_')]
        for f in files:
            if not f.endswith('.py') or f in skip_files:
                continue
            path = os.path.join(root, f)
            try:
                with open(path, encoding='utf-8') as fh:
                    src = fh.read()
            except Exception:
                continue
            new = consolidate(src)
            if new != src:
                with open(path, 'w', encoding='utf-8') as fh:
                    fh.write(new)
                n_changed += 1
                print(f'  merged  {f}')
    print(f'\n{n_changed} files consolidated.')


if __name__ == '__main__':
    main()
