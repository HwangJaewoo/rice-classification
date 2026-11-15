"""Image preprocessing: illumination correction and segmentation.

For one year and one viewing angle, every raw image under --input is
1. checked for over-exposure: the median color of a background region is compared with the reference
   background, and the image is corrected when the difference exceeds 10 in any channel;
2. corrected by histogram matching to an uncorrected image of the same plant within +-3 days,
   otherwise of another plant of the same cultivar within +-3 days, otherwise of one fixed plant;
3. segmented: background color normalization, cropping, removal of the blue background, and removal of isolated pixels.

Outputs (in --output):
    segmented/<year>/<angle>/<group>/<plant>/<plant>_VIS_<angle>_<date>.png   final images
    illumination_corrected/<year>/<angle>/...                                corrected images before segmentation
    reference_log_<year>_<angle>.csv                                         reference image of each corrected image
    unreadable_<year>_<angle>.txt                                            files that could not be read

Usage:
    python preprocess.py --year 2023 --angle SV000 --input <raw images> --output <output folder>
"""
import argparse
import csv
import os
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from skimage.exposure import match_histograms

YEARS = (2023, 2024)
ANGLES = ("SV000", "SV120", "SV240")

# Background region (x1, y1, x2, y2) and its reference median color (B, G, R).
BG_ROI = (2170, 920, 2300, 1050)
BG_REF_BGR = np.array([255, 176, 122], dtype=np.float32)
# Illumination correction.
DIFF_THRESHOLD = (10, 10, 10)
SEARCH_OFFSETS = [0, -1, 1, -2, 2, -3, 3]
# Crop window (rows Y_UP:Y_DOWN, columns X_LEFT:X_RIGHT).
Y_UP, Y_DOWN, X_LEFT, X_RIGHT = 1050, 4415, 770, 3700
# Isolated-pixel removal: keep a pixel if its N x N neighbourhood contains at least CRITERIA plant pixels.
OUTLIER_N = 9
OUTLIER_CRITERIA = 7


# ---------------- file names ----------------

def is_raw_image(file_name, year, angle):
    tag = f"VIS_{angle}"
    if year == 2023:
        return tag in file_name
    return file_name.endswith(tag + ".png")


def parse_raw_filename(file_name, year):
    """Return (plant, date, output subfolder <group>/<plant>, output file name)."""
    parts = file_name.split('_')
    if year == 2023:
        plant = '_'.join(parts[3:6])
        date = parts[8].split()[0]
        p = file_name.split(' ')[0].split('_')
        out_subdir = os.path.join(p[4], '_'.join(p[3:6]))
        out_name = '_'.join(p[3:]) + '.png'
    else:
        plant = '_'.join(parts[0:3])
        date = parts[3].split()[0]
        out_subdir = os.path.join(parts[1], '_'.join(parts[0:3]))
        angle = ('_'.join(parts[-2:])).replace('.png', '')
        out_name = f"{plant}_{angle}_{date}.png"
    return plant, date, out_subdir, out_name


def list_files(input_dir, predicate):
    paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if predicate(f):
                paths.append(os.path.join(root, f))
    return sorted(paths, key=lambda p: p.replace('/', '\\'))


# ---------------- illumination correction ----------------

def needs_correction(image):
    x1, y1, x2, y2 = BG_ROI
    roi_median = np.array(np.median(image[y1:y2, x1:x2], axis=(0, 1)), dtype=np.float32)
    return bool(np.any(np.abs(roi_median - BG_REF_BGR) > DIFF_THRESHOLD))


def load_cultivars(path):
    plant_to_cultivar, cultivar_to_plants = {}, defaultdict(list)
    if path and os.path.exists(path):
        for _, row in pd.read_csv(path).iterrows():
            plant_to_cultivar[row["Plant"]] = row["Cultivar"]
            cultivar_to_plants[row["Cultivar"]].append(row["Plant"])
    return plant_to_cultivar, cultivar_to_plants


def build_index(paths, year):
    index = defaultdict(dict)
    for path in paths:
        plant, date, _, _ = parse_raw_filename(os.path.basename(path), year)
        index[plant][date] = path
    return index


def fixed_reference_plant(flagged_index, normal_index):
    required = set()
    for dates in flagged_index.values():
        required.update(dates.keys())
    best_score, best_plant = -1, None
    for plant, dates in normal_index.items():
        if required.issubset(dates.keys()):
            return plant
        score = len(required & set(dates.keys()))
        if score > best_score:
            best_score, best_plant = score, plant
    return best_plant


def find_reference(plant, date, plant_to_cultivar, cultivar_to_plants, normal_index, fixed_plant):
    base = datetime.strptime(date, "%Y-%m-%d")
    days = [(base + timedelta(days=d)).strftime("%Y-%m-%d") for d in SEARCH_OFFSETS]

    if plant in normal_index:
        for d in days:
            if d in normal_index[plant]:
                return normal_index[plant][d], 'same_plant'

    cultivar = plant_to_cultivar.get(plant)
    candidates = sorted(cultivar_to_plants[cultivar]) if cultivar is not None else []
    for d in days:
        for other in candidates:
            if other != plant and other in normal_index and d in normal_index[other]:
                return normal_index[other][d], 'same_cultivar'

    if fixed_plant in normal_index:
        for d in days:
            if d in normal_index[fixed_plant]:
                return normal_index[fixed_plant][d], 'fixed_plant'
    return None, None


def correct_illumination(paths, year, cultivars, out_dir):
    """Return {raw path: corrected image path}, the reference log, and unreadable files."""
    flagged, unreadable = [], []
    for path in paths:
        image = cv2.imread(path)
        if image is None:
            unreadable.append(path)
        elif needs_correction(image):
            flagged.append(path)
    flagged_set, bad = set(flagged), set(unreadable)
    flagged_index = build_index(flagged, year)
    normal_index = build_index([p for p in paths if p not in flagged_set and p not in bad], year)
    fixed_plant = fixed_reference_plant(flagged_index, normal_index)
    plant_to_cultivar, cultivar_to_plants = load_cultivars(cultivars)

    corrected, log = {}, []
    for plant in flagged_index:
        for date, path in flagged_index[plant].items():
            ref_path, rule = find_reference(plant, date, plant_to_cultivar, cultivar_to_plants, normal_index, fixed_plant)
            if ref_path is None:
                log.append((path, '', 'none', ''))
                continue
            matched = match_histograms(cv2.imread(path), cv2.imread(ref_path), channel_axis=-1).astype("uint8")
            _, _, sub, name = parse_raw_filename(os.path.basename(path), year)
            os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
            out_path = os.path.join(out_dir, sub, name)
            cv2.imwrite(out_path, matched)
            corrected[path] = out_path
            log.append((path, ref_path, rule, out_path))
    return corrected, log, unreadable


# ---------------- segmentation ----------------

def adjust_background(image):
    x1, y1, x2, y2 = BG_ROI
    roi_median = np.median(image[y1:y2, x1:x2], axis=(0, 1))
    roi_median = np.where(roi_median == 0, 1, roi_median)
    return np.clip(image.astype(np.float32) * (BG_REF_BGR / roi_median), 0, 255).astype(np.uint8)


def remove_blue(image):
    hsv = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), np.array([100, 0, 0]), np.array([130, 255, 255]))
    bgr = cv2.inRange(image, np.array([150, 0, 0]), np.array([255, 168, 168]))
    ycc = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb), np.array([0, 0, 150], dtype=np.uint8),
                      np.array([255, 240, 240], dtype=np.uint8))
    keep = lambda m: cv2.bitwise_and(image, image, mask=cv2.bitwise_not(m))
    return cv2.bitwise_and(cv2.bitwise_or(keep(bgr), keep(ycc)), keep(hsv))


def remove_isolated_pixels(path):
    img = Image.open(path)
    hsv = np.array(img.convert('HSV'))
    plant = (hsv[:, :, 0] != 0) | (hsv[:, :, 1] != 0) | (hsv[:, :, 2] != 0)
    width, height = img.size
    r = np.arange(-(OUTLIER_N - 1) // 2, (OUTLIER_N - 1) // 2 + 1)
    for y, x in np.argwhere(plant):
        count = 0
        for i in r:
            for j in r:
                yy, xx = y + i, x + j
                if 0 <= yy < height and 0 <= xx < width and plant[yy, xx]:
                    count += 1
            if count >= OUTLIER_CRITERIA:
                break
        if count < OUTLIER_CRITERIA:
            hsv[y, x] = [0, 0, 0]
    Image.fromarray(hsv, 'HSV').convert('RGB').save(path, format='PNG')


def segment_one(raw_path, source_path, year, out_dir):
    image = cv2.imread(source_path)
    if image is None:
        return raw_path
    _, _, sub, name = parse_raw_filename(os.path.basename(raw_path), year)
    image = adjust_background(image)[Y_UP:Y_DOWN, X_LEFT:X_RIGHT]
    os.makedirs(os.path.join(out_dir, sub), exist_ok=True)
    out_path = os.path.join(out_dir, sub, name)
    cv2.imwrite(out_path, remove_blue(image))
    remove_isolated_pixels(out_path)
    return None


# ---------------- main ----------------

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument('--year', type=int, choices=YEARS, required=True)
    ap.add_argument('--angle', choices=ANGLES, required=True)
    ap.add_argument('--input', required=True, help='folder of raw images (searched recursively)')
    ap.add_argument('--output', required=True, help='output folder')
    ap.add_argument('--cultivars', default=None, help='CSV with columns Plant, Cultivar (default: cultivars_<year>.csv here)')
    ap.add_argument('--workers', type=int, default=max(1, min(8, cpu_count() - 1)))
    a = ap.parse_args()
    cultivars = a.cultivars or os.path.join(here, f"cultivars_{a.year}.csv")
    tag = f"{a.year}_{a.angle}"

    paths = list_files(a.input, lambda f: is_raw_image(f, a.year, a.angle))
    if not paths:
        raise SystemExit(f"No {a.angle} images of {a.year} found in {a.input}")
    corrected, log, unreadable = correct_illumination(
        paths, a.year, cultivars, os.path.join(a.output, 'illumination_corrected', str(a.year), a.angle))

    seg_dir = os.path.join(a.output, 'segmented', str(a.year), a.angle)
    bad = set(unreadable)
    jobs = [(p, corrected.get(p, p)) for p in paths if p not in bad]
    with Pool(a.workers) as pool:
        failed = pool.starmap(partial(segment_one, year=a.year, out_dir=seg_dir), jobs)
    unreadable += [f for f in failed if f is not None]

    with open(os.path.join(a.output, f'reference_log_{tag}.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['image', 'reference', 'rule', 'corrected_image'])
        w.writerows(log)
    with open(os.path.join(a.output, f'unreadable_{tag}.txt'), 'w', encoding='utf-8') as fh:
        fh.writelines(p + '\n' for p in unreadable)
    print(f"{len(paths)} images, {len(corrected)} corrected, {len(paths) - len(unreadable)} segmented, "
          f"{len(unreadable)} unreadable")


if __name__ == '__main__':
    main()
