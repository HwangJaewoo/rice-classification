"""Inventory of segmented images and mapping of image folders to the rows of the released CSV files.

Images are aligned to T1-T61 (dates.py). Each plant folder is matched to (Cultivar, Sample_Number) by
comparing vertical_h of its SV000 images at T1, T2 and T61 with vertical_h_<year>.csv.

Usage:
    python inventory.py [--root <image root>] [--year 2023 2024] [--dataset <Preprocessed_dataset/<year>>]
"""
import argparse
import os

import cv2
import numpy as np
import pandas as pd

from dates import find_root, ANGLES, N_T, align, plant_files, reference_grid

CHECK_T = [1, 2, 61]


def vertical_h(path):
    mask = np.any(cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR) != 0, axis=2)
    return mask.shape[0] - np.argwhere(mask)[:, 0].min() - 1


def run(a, year, dataset):
    plants = plant_files(os.path.join(a.root, a.subdir, year))
    grid = reference_grid(plants)
    ref = pd.read_csv(os.path.join(dataset, f"vertical_h_{year}.csv"))
    ref_key = {tuple(int(round(v)) for v in r.iloc[[1 + t for t in CHECK_T]]): (str(r.iloc[0]).strip(), int(r.iloc[1]))
               for _, r in ref.iterrows()}

    rows = []
    for pid in sorted(plants):
        al = {ang: align(plants[pid].get(ang, []), grid) for ang in ANGLES}
        miss = {ang: [t + 1 for t, f in enumerate(al[ang]) if f is None] for ang in ANGLES}
        cultivar, sample = "", np.nan
        s0 = al["SV000"]
        if all(s0[t - 1] is not None for t in CHECK_T):
            key = tuple(int(vertical_h(s0[t - 1])) for t in CHECK_T)
            cultivar, sample = ref_key.get(key, ("", np.nan))
        rows.append({"plant": pid, "cultivar": cultivar, "sample": sample,
                     **{f"n_{ang}": len(plants[pid].get(ang, [])) for ang in ANGLES},
                     **{f"missing_T_{ang}": ";".join(map(str, miss[ang])) for ang in ANGLES}})
    T = pd.DataFrame(rows)

    os.makedirs(a.out, exist_ok=True)
    T.to_csv(os.path.join(a.out, f"inventory_{year}.csv"), index=False)
    pd.DataFrame({"T": [f"T{t + 1}" for t in range(N_T)],
                  "date": [g.strftime("%Y-%m-%d") for g in grid[:N_T]]}).to_csv(
        os.path.join(a.out, f"tgrid_{year}.csv"), index=False)
    ok = T[T.cultivar != ""]
    n_miss = int(sum((ok[f"missing_T_{ang}"] != "").sum() for ang in ANGLES))
    print(f"{year}: {len(T)} plant folders, {len(ok)} matched, {ok.cultivar.nunique()} cultivars, "
          f"{len(ref)} CSV rows, duplicates {int(ok.duplicated(['cultivar', 'sample']).sum())}, "
          f"T1={grid[0]:%Y-%m-%d}, T61={grid[N_T - 1]:%Y-%m-%d}, plant-angles with missing T {n_miss}")


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=None)
    ap.add_argument("--subdir", default="segmented")
    ap.add_argument("--year", nargs="*", default=["2023", "2024"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--out", default=os.path.join(here, "data"))
    a = ap.parse_args()
    a.root = a.root or find_root(a.subdir)
    for year in a.year:
        run(a, year, a.dataset or os.path.join(here, "..", "Preprocessed_dataset", year))


if __name__ == "__main__":
    main()
