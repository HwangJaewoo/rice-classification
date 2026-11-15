"""Preprocess the raw example images again and compare them pixel by pixel with the released preprocessed images.

Usage:
    cd Image_preprocess
    python reproduce_example.py
Reads the raw images in ../Images/example/raw/<year>/angle0 (original file names)
and the preprocessed images in ../Images/example/preprocessed/<year>/angle0
(file names ending with the acquisition date, YYYY-MM-DD), matched by date.
Each raw image goes through the steps of preprocess.py (illumination correction, cropping,
removal of the blue background, removal of isolated pixels); an image that needs illumination
correction is matched to an uncorrected image of the same plant within +-3 days in the same folder.
Single folders: python reproduce_example.py --year 2023 --angle 120 --raw-dir <raw folder> --pre-dir <preprocessed folder>
Prints, for every year, the number of images that are identical to the released ones, and saves
the details to reproduce_example.csv.
"""
import argparse
import os
import re
import tempfile
import time
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms

import preprocess as pp

ANGLE_TAG = {"0": "SV000", "120": "SV120", "240": "SV240"}
DATE = re.compile(r"(\d{4}-\d{2}-\d{2})$")


def read(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def one(job, tmp):
    date, raw_path, ref_path, rule, pre_path = job
    image = read(raw_path)
    if ref_path:
        image = match_histograms(image, read(ref_path), channel_axis=-1).astype("uint8")
    image = pp.adjust_background(image)[pp.Y_UP:pp.Y_DOWN, pp.X_LEFT:pp.X_RIGHT]
    out = os.path.join(tmp, f"{date}.png")
    cv2.imwrite(out, pp.remove_blue(image))
    pp.remove_isolated_pixels(out)
    new, old = read(out), read(pre_path)
    os.remove(out)
    if new.shape != old.shape:
        return {"date": date, "rule": rule, "identical": False, "n_diff_pixels": -1}
    n = int(np.any(new != old, axis=2).sum())
    return {"date": date, "rule": rule, "identical": n == 0, "n_diff_pixels": n}


def jobs_for(year, raw_dir, pre_dir, tag):
    raws = [os.path.join(raw_dir, f) for f in sorted(os.listdir(raw_dir)) if pp.is_raw_image(f, int(year), tag)]
    pres = {}
    for f in os.listdir(pre_dir):
        m = DATE.search(os.path.splitext(f)[0])
        if m and f.lower().endswith(".png"):
            pres[m.group(1)] = os.path.join(pre_dir, f)
    flagged = [p for p in raws if pp.needs_correction(read(p))]
    normal_index = pp.build_index([p for p in raws if p not in flagged], int(year))
    jobs, missing = [], []
    for p in raws:
        plant, date, _, _ = pp.parse_raw_filename(os.path.basename(p), int(year))
        if date not in pres:
            missing.append(date)
            continue
        ref, rule = (None, "not_flagged")
        if p in flagged:
            ref, rule = pp.find_reference(plant, date, {}, {}, normal_index, None)
            rule = rule or "no_reference_in_folder"
        jobs.append((date, p, ref, rule, pres[date]))
    return jobs, missing


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(here, "..", "Images", "example", "raw"))
    ap.add_argument("--pre", default=os.path.join(here, "..", "Images", "example", "preprocessed"))
    ap.add_argument("--year", nargs="+", default=["2023", "2024"], choices=["2023", "2024"])
    ap.add_argument("--angle", default="0", choices=["0", "120", "240"])
    ap.add_argument("--raw-dir", default=None, help="one folder of raw images (use with --pre-dir and one --year)")
    ap.add_argument("--pre-dir", default=None, help="one folder of preprocessed images of the same plant and angle")
    ap.add_argument("--workers", type=int, default=max(1, min(8, cpu_count() - 1)))
    a = ap.parse_args()
    if (a.raw_dir is None) != (a.pre_dir is None) or (a.raw_dir and len(a.year) != 1):
        raise SystemExit("--raw-dir and --pre-dir must be given together, with a single --year")

    tmp = tempfile.mkdtemp()
    rows = []
    for year in a.year:
        sub = os.path.join(year, f"angle{a.angle}")
        raw_dir = a.raw_dir or os.path.join(a.raw, sub)
        pre_dir = a.pre_dir or os.path.join(a.pre, sub)
        for d, name in ((raw_dir, "raw.zip"), (pre_dir, "preprocessed.zip")):
            if not os.path.isdir(d):
                raise SystemExit(f"Folder not found: {os.path.normpath(d)}\n"
                                 f"Download {name} from the DOI in README.md and unzip it into Images/example/ "
                                 f"(see Images/README.md), or set --raw-dir and --pre-dir.")
        jobs, missing = jobs_for(year, raw_dir, pre_dir, ANGLE_TAG[a.angle])
        if not jobs:
            raise SystemExit(f"No raw {ANGLE_TAG[a.angle]} images of {year} with a matching preprocessed image "
                             f"(raw: {raw_dir}, preprocessed: {pre_dir})")
        res, start, step = [], time.time(), 20
        with Pool(a.workers) as pool:
            for k, r in enumerate(pool.imap_unordered(partial(one, tmp=tmp), jobs), 1):
                res.append(r)
                pct = 100 * k // len(jobs)
                if pct >= step or k == len(jobs):
                    el = time.time() - start
                    print(f"  {year}: {pct:3d}% ({k}/{len(jobs)}), elapsed {el / 60:.1f} min, "
                          f"remaining about {el / k * (len(jobs) - k) / 60:.1f} min", flush=True)
                    step = (pct // 20 + 1) * 20
        res.sort(key=lambda r: r["date"])
        for r in res:
            r["year"] = year
        rows += res
        df = pd.DataFrame(res)
        flagged = df[df.rule != "not_flagged"]
        print(f"{year}, angle {a.angle}: {len(df)} images, {int(df.identical.sum())} identical to the released ones; "
              f"{len(flagged)} needed illumination correction"
              + (f" ({', '.join(f'{d} {r}' for d, r in zip(flagged.date, flagged.rule))})" if len(flagged) else "")
              + (f"; no preprocessed image for {', '.join(missing)}" if missing else ""))
        bad = df[~df.identical]
        for _, r in bad.iterrows():
            print(f"    different: {r.date} ({r.rule}), {r.n_diff_pixels} pixels")
    out = os.path.join(here, "reproduce_example.csv")
    pd.DataFrame(rows)[["year", "date", "rule", "identical", "n_diff_pixels"]].to_csv(out, index=False)
    print(f"details: {out}")


if __name__ == "__main__":
    main()
