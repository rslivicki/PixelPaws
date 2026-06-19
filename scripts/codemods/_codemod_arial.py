"""One-shot codemod: replace hardcoded ('Arial', N [, style]) font
tuples with (FONT_FAMILY, N [, style]) and ensure FONT_FAMILY is
imported from ui_utils.

Skips:
  - tests/ directory (no GUI code there)
  - backup_*/ directories
  - ui_utils.py itself (it defines FONT_FAMILY)
  - matplotlib font references (rcParams etc.) — they don't use
    ``font=('Arial', ...)`` syntax
"""

from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# Match the literal 'Arial' or "Arial" inside a font=( ... ) tuple.
# We only replace inside tkinter font tuples — so we anchor on font=(
# being on the same line. (All 229 instances follow this pattern.)
ARIAL_RE = re.compile(
    r"(font\s*=\s*\(\s*)(['\"])Arial\2",
    re.DOTALL,
)


def needs_import(src: str) -> bool:
    """True iff the file imports from ui_utils but doesn't yet pull in
    FONT_FAMILY."""
    if 'FONT_FAMILY' in src:
        return False  # already imported (or inlined)
    return ('from ui_utils import' in src) or ('import ui_utils' in src)


def add_font_family_import(src: str) -> str:
    """Append FONT_FAMILY to the existing ``from ui_utils import (...)``
    or single-line form. Conservative: only modify the FIRST such import."""
    # Multi-line tuple form: from ui_utils import (a, b)
    multi = re.search(
        r"(from\s+ui_utils\s+import\s*\([^)]*?)(\))",
        src,
        re.DOTALL,
    )
    if multi:
        before, close = multi.group(1), multi.group(2)
        if 'FONT_FAMILY' not in before:
            insert = before.rstrip().rstrip(',') + ', FONT_FAMILY' + close
            return src.replace(multi.group(0), insert, 1)
        return src
    # Single-line form: from ui_utils import a, b, c
    single = re.search(r"(from\s+ui_utils\s+import\s+)(.+)", src)
    if single:
        names = single.group(2).strip()
        if 'FONT_FAMILY' in names:
            return src
        new_line = f"{single.group(1)}{names}, FONT_FAMILY"
        return src.replace(single.group(0), new_line, 1)
    return src


def add_new_import(src: str) -> str:
    """For files that don't import from ui_utils at all, add a fresh
    one-liner near the top (after the last `import` block)."""
    # Find the last top-level import block
    import_re = re.compile(r"^(from\s+\S+\s+import.*|import\s+\S+.*)$", re.MULTILINE)
    matches = list(import_re.finditer(src))
    if not matches:
        return src
    # Insert just after the LAST import line
    last = matches[-1]
    insert_at = last.end()
    return (
        src[:insert_at]
        + '\nfrom ui_utils import FONT_FAMILY'
        + src[insert_at:]
    )


def process_file(path: str) -> tuple[int, bool]:
    """Returns (n_replacements, import_changed)."""
    with open(path, encoding='utf-8') as f:
        src = f.read()

    matches = list(ARIAL_RE.finditer(src))
    if not matches:
        return 0, False

    new_src, n = ARIAL_RE.subn(r"\1FONT_FAMILY", src)

    import_changed = False
    if 'FONT_FAMILY' not in new_src.split('font=', 1)[0]:
        # FONT_FAMILY is referenced but not imported. Add the import.
        if needs_import(new_src):
            new_src = add_font_family_import(new_src)
        else:
            new_src = add_new_import(new_src)
        import_changed = True

    if new_src != src:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(new_src)
    return n, import_changed


def main():
    skip_files = {'ui_utils.py', '_codemod_arial.py'}
    skip_dirs  = {'tests', '__pycache__', '_audit_screenshots'}
    skip_dir_prefixes = ('backup_',)

    total = 0
    changed_files = 0
    for root, dirs, files in os.walk(ROOT):
        # Prune
        dirs[:] = [
            d for d in dirs
            if d not in skip_dirs and not d.startswith(skip_dir_prefixes)
        ]
        for f in files:
            if not f.endswith('.py') or f in skip_files:
                continue
            path = os.path.join(root, f)
            try:
                n, imp = process_file(path)
            except Exception as e:
                print(f'  ERROR  {f}: {e}')
                continue
            if n:
                total += n
                changed_files += 1
                print(f'  {f:35s}  {n:3d} replacements'
                      f'{"  +import" if imp else ""}')

    print(f'\n{changed_files} files modified, {total} font tuples replaced.')


if __name__ == '__main__':
    main()
