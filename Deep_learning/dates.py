"""Alignment of acquisition dates to the time points T1-T61 of the released CSV files.

The reference grid of a year is the most common date list among plants with the largest number of images.
Each image date is snapped to the nearest grid date within one day (ties go to the earlier date, which
reproduces the date corrections of the original preprocessing). T1-T61 are the first 61 grid dates.
"""
import collections
import glob
import os
from datetime import datetime, timedelta

ANGLES = ["SV000", "SV120", "SV240"]
N_T = 61


def date_of(path):
    return datetime.strptime(os.path.splitext(os.path.basename(path))[0][-10:], "%Y-%m-%d")


def plant_files(base):
    out = {}
    for ang in ANGLES:
        for d in glob.glob(os.path.join(base, ang, "*", "*")):
            if os.path.isdir(d):
                out.setdefault(os.path.basename(d), {})[ang] = glob.glob(os.path.join(d, "*.png"))
    return out


def find_root(subdir, candidates=(os.path.join("..", "Images"),)):
    for c in candidates:
        if os.path.isdir(os.path.join(c, subdir)):
            return c
    raise FileNotFoundError(f"{subdir} not found under {candidates}; pass --root")


def reference_grid(plants):
    if not plants:
        raise FileNotFoundError("no plant folders found; check --root")
    lists = [tuple(sorted({date_of(f) for f in p.get("SV000", [])})) for p in plants.values()]
    n_max = max(len(x) for x in lists)
    return list(collections.Counter(x for x in lists if len(x) == n_max).most_common(1)[0][0])


def snap(d, grid_set):
    for c in (d, d - timedelta(days=1), d + timedelta(days=1)):
        if c in grid_set:
            return c
    return None


def align(files, grid):
    """Map one plant/angle file list to T1-T61: list of 61 paths (None where missing)."""
    grid_set = set(grid)
    slot = {}
    for f in sorted(files, key=date_of):
        d = date_of(f)
        g = snap(d, grid_set)
        if g is None:
            continue
        if g not in slot or d == g:
            slot[g] = f
    return [slot.get(g) for g in grid[:N_T]]
