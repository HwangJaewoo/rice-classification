"""Check whether the indices computed by extract_indices.py agree with the released values.

Usage (after extract_indices.py):
    python compare_indices.py                         # all six sets (2023 and 2024, angles 0, 120, 240)
    python compare_indices.py --year 2023 --angle 0   # one set
Prints, for every set and index, whether all values agree within the tolerance,
and saves the summary to <results>/summary_<cultivar>_sample<sample>.csv.
"""
import argparse
import os

import numpy as np
import pandas as pd

INDICES = ["r", "g", "b", "h", "s", "v", "y", "cb", "cr", "l_lab", "a_lab", "b_lab",
           "width", "diagonal_h", "area", "vertical_h", "Peri", "FSEP", "a1", "b1", "c1", "d1", "F_std"]
SUFFIX = {"0": "", "120": "_120", "240": "_240"}


def check(results, dataset_root, year, angle, cultivar, sample, tol):
    """Return one row per index for one year and angle."""
    tag = f"{cultivar}_sample{sample}_{year}_angle{angle}"
    ours = pd.read_csv(os.path.join(results, f"indices_{tag}.csv"))
    rows = []
    for name in INDICES:
        ref = pd.read_csv(os.path.join(dataset_root, year, f"{name}_{year}{SUFFIX[angle]}.csv"))
        row = ref[(ref.iloc[:, 0].astype(str).str.strip() == cultivar) & (ref.iloc[:, 1] == sample)]
        released = row.iloc[0, 2:2 + len(ours)].to_numpy(dtype=float)
        computed = ours[name].to_numpy(dtype=float)
        diff = np.abs(computed - released)
        ok = diff <= tol * np.maximum(1.0, np.abs(released))
        rows.append({"year": year, "angle": angle, "index": name, "n": len(ok), "n_same": int(ok.sum()),
                     "max_abs_diff": float(diff.max()), "same": bool(ok.all())})
    return rows


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultivar", default="Samcheonbyeo")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--year", nargs="+", default=["2023", "2024"], choices=["2023", "2024"])
    ap.add_argument("--angle", nargs="+", default=["0", "120", "240"], choices=["0", "120", "240"])
    ap.add_argument("--results", default=os.path.join(here, "results"))
    ap.add_argument("--dataset", default=os.path.join(here, "..", "Preprocessed_dataset"),
                    help="folder that contains 2023/ and 2024/")
    ap.add_argument("--tol", type=float, default=1e-6)
    a = ap.parse_args()

    print(f"{a.cultivar}, plant {a.sample}")
    print(f"criterion: |computed - released| <= {a.tol:g} x max(1, |released|)\n")
    rows = []
    for year in a.year:
        for angle in a.angle:
            r = check(a.results, a.dataset, year, angle, a.cultivar, a.sample, a.tol)
            rows += r
            bad = [x["index"] for x in r if not x["same"]]
            print(f"  {year}, angle {angle:>3}: {len(r) - len(bad)} of {len(r)} indices same"
                  + (f"  (different: {', '.join(bad)})" if bad else ""))
    df = pd.DataFrame(rows)
    out = os.path.join(a.results, f"summary_{a.cultivar}_sample{a.sample}.csv")
    df.to_csv(out, index=False)
    print(f"\n{int(df['same'].sum())} of {len(df)} (set, index) pairs are the same as the released values.")
    print(f"details: {out}")


if __name__ == "__main__":
    main()
