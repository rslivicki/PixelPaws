"""One-shot patcher run after the 2026-05-01 folder restructure.

After moving research scripts into scripts/research/, their existing
``sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))``
calls now point to scripts/research/ instead of E:/PixelPaws/. They
need to inject the project root (parent-of-parent) AND keep their own
directory (so they can still import peer research scripts).

This script edits each moved file IN PLACE. Idempotent — if the new
form is already there, leaves it alone.
"""

from __future__ import annotations
import os
import re

OLD = re.compile(
    r'sys\.path\.insert\(\s*0\s*,\s*os\.path\.dirname\(\s*os\.path\.abspath\(\s*__file__\s*\)\s*\)\s*\)',
)

NEW_BLOCK = (
    "_HERE = os.path.dirname(os.path.abspath(__file__))\n"
    "sys.path.insert(0, _HERE)                                # peer scripts in this folder\n"
    "sys.path.insert(0, os.path.dirname(os.path.dirname(_HERE)))  # project root (E:/PixelPaws)"
)

ROOTS = [
    r'E:/PixelPaws/scripts/research',
    r'E:/PixelPaws/scripts/utilities',
]


def patch_file(path: str) -> bool:
    with open(path, encoding='utf-8') as f:
        src = f.read()

    if '_HERE = os.path.dirname(os.path.abspath(__file__))' in src:
        return False  # already patched

    new_src, n = OLD.subn(NEW_BLOCK, src, count=1)
    if n == 0:
        return False
    with open(path, 'w', encoding='utf-8') as f:
        f.write(new_src)
    return True


def main():
    n = 0
    for root in ROOTS:
        if not os.path.isdir(root):
            continue
        for fname in sorted(os.listdir(root)):
            if not fname.endswith('.py'):
                continue
            path = os.path.join(root, fname)
            if patch_file(path):
                print(f'  patched  {os.path.relpath(path)}')
                n += 1
    print(f'\n{n} file(s) patched.')


if __name__ == '__main__':
    main()
