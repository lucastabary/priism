"""The ``clean`` job: delete paths inside the agent's work folder (the current directory) to free cache space.

``jobs.validate`` already refused absolute paths, ``..`` and the agent's state folder; this check runs
again on resolved paths, so a symlink inside the work folder cannot lead outside it.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def _size(p: Path) -> int:
    if p.is_symlink() or p.is_file():
        return p.lstat().st_size
    return sum(f.lstat().st_size for f in p.rglob("*") if f.is_file() and not f.is_symlink())


def main(paths: list[str]) -> int:
    root = Path.cwd().resolve()
    freed = 0
    for arg in paths:
        p = root / arg
        if not p.exists() and not p.is_symlink():
            print(f"{arg}: not there")
            continue
        target = p.parent.resolve() / p.name  # the entry itself, not where a symlink points
        if root not in target.parents or target.relative_to(root).parts[0] == ".agent":
            print(f"{arg}: outside the work folder, skipped")
            continue
        n = _size(p)
        if p.is_dir() and not p.is_symlink():
            shutil.rmtree(p)
        else:
            p.unlink()
        freed += n
        print(f"{arg}: deleted, {n / 1e9:.2f} GB")
    print(f"freed {freed / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
