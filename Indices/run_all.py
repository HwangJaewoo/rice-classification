"""Compute the indices for the example images and compare them with the released values.

Usage:
    python run_all.py                      # the released example images (2023 and 2024, angle 0)
    python run_all.py --angle 0 120 240    # other angles, if the images are available
Reads Images/example/preprocessed/<year>/angle<angle>,
writes results/indices_<cultivar>_sample<sample>_<year>_angle<angle>.csv,
then runs the comparison of compare_indices.py.
"""
import argparse
import os
import sys

from extract_indices import extract
import compare_indices


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--cultivar", default="Samcheonbyeo")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--images", default=os.path.join(here, "..", "Images", "example", "preprocessed"),
                    help="folder that contains <year>/angle<angle>/")
    ap.add_argument("--dataset", default=os.path.join(here, "..", "Preprocessed_dataset"),
                    help="folder that contains 2023/ and 2024/")
    ap.add_argument("--year", nargs="+", default=["2023", "2024"], choices=["2023", "2024"])
    ap.add_argument("--angle", nargs="+", default=["0"], choices=["0", "120", "240"])
    ap.add_argument("--out", default=os.path.join(here, "results"))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    for year in a.year:
        for angle in a.angle:
            folder = os.path.join(a.images, year, f"angle{angle}")
            feats = extract(folder)
            tag = f"{a.cultivar}_sample{a.sample}_{year}_angle{angle}"
            feats.to_csv(os.path.join(a.out, f"indices_{tag}.csv"), index=False)
            print(f"{year}, angle {angle:>3}: {len(feats)} images -> indices_{tag}.csv")
    print()
    sys.argv = [sys.argv[0], "--cultivar", a.cultivar, "--sample", str(a.sample), "--results", a.out, "--dataset", a.dataset,
                "--year", *a.year, "--angle", *a.angle]
    compare_indices.main()


if __name__ == "__main__":
    main()