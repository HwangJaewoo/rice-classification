"""Resize the segmented images of all matched plants and store them aligned to T1-T61.

Output data/images_<year>_<size>.npy : uint8 [plants, angles, 61, size, size, 3] (RGB, black padding)
       data/mask_<year>_<size>.npy   : bool  [plants, angles, 61] (False where the image is missing)
       data/meta_<year>_<size>.csv   : row index, plant, cultivar, sample
Requires data/inventory_<year>.csv from inventory.py.

Usage:
    python pack.py [--root <image root>] [--year 2023 2024] [--size 128]
"""
import argparse
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import pandas as pd

from dates import find_root, ANGLES, N_T, align, plant_files, reference_grid


def load_small(path, size):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    s = max(h, w)
    sq = np.zeros((s, s, 3), np.uint8)
    sq[s - h:, (s - w) // 2:(s - w) // 2 + w] = img
    return cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)[:, :, ::-1]


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--subdir", default="segmented")
    ap.add_argument("--year", nargs="*", default=["2023", "2024"])
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--data", default=os.path.join(here, "data"))
    a = ap.parse_args()
    a.root = a.root or find_root(a.subdir)

    for year in a.year:
        inv = pd.read_csv(os.path.join(a.data, f"inventory_{year}.csv"))
        inv = inv[inv.cultivar.notna()].sort_values(["cultivar", "sample"]).reset_index(drop=True)
        plants = plant_files(os.path.join(a.root, a.subdir, year))
        grid = reference_grid(plants)
        P, A, S = len(inv), len(ANGLES), a.size
        out = np.lib.format.open_memmap(os.path.join(a.data, f"images_{year}_{S}.npy"), mode="w+",
                                        dtype=np.uint8, shape=(P, A, N_T, S, S, 3))
        mask = np.zeros((P, A, N_T), bool)
        jobs = []
        for i, r in inv.iterrows():
            for j, ang in enumerate(ANGLES):
                for t, f in enumerate(align(plants[r.plant].get(ang, []), grid)):
                    if f is not None:
                        jobs.append((i, j, t, f))
                        mask[i, j, t] = True

        lock, done, step, t0 = threading.Lock(), [0], max(1, len(jobs) // 5), time.time()

        def work(job):
            i, j, t, f = job
            out[i, j, t] = load_small(f, S)
            with lock:
                done[0] += 1
                if done[0] % step == 0 or done[0] == len(jobs):
                    el = (time.time() - t0) / 60
                    print(f"  {year} [{100 * done[0] // len(jobs):3d}%] elapsed {el:.1f} min, "
                          f"ETA {el * (len(jobs) - done[0]) / done[0]:.1f} min", flush=True)

        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            list(ex.map(work, jobs))
        out.flush()
        del out
        np.save(os.path.join(a.data, f"mask_{year}_{S}.npy"), mask)
        inv[["plant", "cultivar", "sample"]].rename_axis("index").to_csv(
            os.path.join(a.data, f"meta_{year}_{S}.csv"))
        print(f"{year}: {P} plants, {len(jobs)} images, missing {int((~mask).sum())}, array {P}x{A}x{N_T}x{S}x{S}x3")


if __name__ == "__main__":
    main()
