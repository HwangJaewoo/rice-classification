"""Compute the 23 phenotypic indices from segmented images and compare them with the released CSV files.

Usage:
    python extract_indices.py --year 2023 --angle 0 --cultivar Samcheonbyeo --sample 1
By default the images are read from Images/example/preprocessed/<year>/angle<angle>
and compared with Preprocessed_dataset/<year>/<index>_<year><suffix>.csv (suffix '', '_120', '_240').
"""
import argparse
import glob
import os
import re
from datetime import datetime

import cv2
import numpy as np
import pandas as pd

COLOR = ["r", "g", "b", "h", "s", "v", "y", "cb", "cr", "l_lab", "a_lab", "b_lab"]
MORPH = ["width", "diagonal_h", "area", "vertical_h", "Peri", "FSEP", "a1", "b1", "c1", "d1", "F_std"]
INDICES = COLOR + MORPH


def imread(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def time_point(path):
    """T number from a file name such as Samcheonbyeo_sample1_2023_angle0_T01_2023-06-01.png (None if absent)."""
    m = re.search(r"_T(\d+)_", os.path.basename(path))
    return int(m.group(1)) if m else None


def acquisition_date(path):
    return datetime.strptime(os.path.splitext(os.path.basename(path))[0][-10:], "%Y-%m-%d")


def color_and_size(img):
    mask = np.any(img != 0, axis=2)
    px = img[mask]
    q = px.reshape(-1, 1, 3)
    f = {}
    f["b"], f["g"], f["r"] = px.mean(axis=0)
    f["h"], f["s"], f["v"] = cv2.cvtColor(q, cv2.COLOR_BGR2HSV).reshape(-1, 3).mean(axis=0)
    f["l_lab"], f["a_lab"], f["b_lab"] = cv2.cvtColor(q, cv2.COLOR_BGR2LAB).reshape(-1, 3).mean(axis=0)
    f["y"], f["cr"], f["cb"] = cv2.cvtColor(q, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).mean(axis=0)
    idx = np.argwhere(mask)
    H, W = mask.shape
    top = idx[np.argmin(idx[:, 0])]
    f["vertical_h"] = H - idx[:, 0].min() - 1
    f["diagonal_h"] = np.linalg.norm(np.array([H - 1, W // 2]) - top)
    f["width"] = idx[:, 1].max() - idx[:, 1].min() + 1
    f["area"] = int(mask.sum())
    return f


def rgb2gray(img):
    v = 0.298936021293775 * img[:, :, 2].astype(np.float64) + 0.587043074451121 * img[:, :, 1] \
        + 0.114020904255103 * img[:, :, 0]
    return np.clip(np.floor(v + 0.5), 0, 255).astype(np.uint8)


def otsu_level(gray):
    p = np.bincount(gray.ravel(), minlength=256).astype(np.float64)
    p /= p.sum()
    omega = np.cumsum(p)
    mu = np.cumsum(p * np.arange(1, 257))
    with np.errstate(divide="ignore", invalid="ignore"):
        sb = (mu[-1] * omega - mu) ** 2 / (omega * (1 - omega))
    return (np.mean(np.where(sb == np.nanmax(sb))[0] + 1) - 1) / 255.0


def largest_component(bw):
    _, lab, st, _ = cv2.connectedComponentsWithStats(bw.astype(np.uint8), connectivity=8)
    return lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])


def trace_boundary(bw):
    H, W = bw.shape
    pad = np.zeros((H + 2, W + 2), bool)
    pad[1:-1, 1:-1] = bw
    nb = [(-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
    c0 = np.where(bw.any(axis=0))[0][0]
    s = (np.where(bw[:, c0])[0][0] + 1, c0 + 1)
    p, bdir, out, first = s, 6, [s], None
    while True:
        for k in range(1, 9):
            d = (bdir + k) % 8
            q = (p[0] + nb[d][0], p[1] + nb[d][1])
            if pad[q]:
                break
        else:
            break
        prev = (bdir + k - 1) % 8
        bp = (p[0] + nb[prev][0], p[1] + nb[prev][1])
        p = q
        bdir = nb.index((bp[0] - p[0], bp[1] - p[1]))
        if first is None:
            first = (p, bdir)
        elif (p, bdir) == first and out[-1] == s:
            break
        out.append(p)
    B = np.array(out[:-1] if len(out) > 1 else out, dtype=np.float64) - 1
    return np.vstack([B, B[:1]]) if len(out) > 1 else B


def perimeter(B):
    delta = np.diff(B, axis=0) ** 2
    if len(delta) == 0:
        return 0.0
    corner = np.any(np.diff(np.vstack([delta, delta[:1]]), axis=0) != 0, axis=1)
    even = np.any(delta == 0, axis=1)
    return even.sum() * 0.980 + (~even).sum() * 1.406 - corner.sum() * 0.091


def efd_first_harmonic(B):
    co = np.delete(B, np.where(np.all(np.diff(B, axis=0) == 0, axis=1))[0], axis=0)
    x, y = co[:, 1], co[:, 0]
    dx, dy = np.diff(np.append(x, x[0])), np.diff(np.append(y, y[0]))
    dt = np.sqrt(dx ** 2 + dy ** 2)
    dt[dt == 0] = 1e-9
    T = dt.sum()
    tp = np.cumsum(dt)
    tq = np.append(0, tp[:-1])
    k = T / (2 * np.pi ** 2)
    cp = np.cos(2 * np.pi * tp / T) - np.cos(2 * np.pi * tq / T)
    sp = np.sin(2 * np.pi * tp / T) - np.sin(2 * np.pi * tq / T)
    return (k * np.sum(dx / dt * cp), k * np.sum(dx / dt * sp),
            k * np.sum(dy / dt * cp), k * np.sum(dy / dt * sp))


def shape_and_spectrum(img):
    g = rgb2gray(img)
    bw = largest_component(g > otsu_level(g) * 255)
    F = np.fft.fftshift(np.fft.fft2(g.astype(np.float64)))
    E = np.abs(F) ** 2
    M, N = E.shape
    r = max(1, int(np.floor(min(M, N) * 0.1)))
    blk = (slice(M // 2 - r, M // 2 + r + 1), slice(N // 2 - r, N // 2 + r + 1))
    B = trace_boundary(bw)
    a1, b1, c1, d1 = efd_first_harmonic(B)
    return {"FSEP": E[blk].sum() / E.sum(), "Peri": perimeter(B),
            "a1": a1, "b1": b1, "c1": c1, "d1": d1,
            "F_std": np.std(np.real(F[blk]), ddof=1)}


def extract(image_dir):
    files = sorted(glob.glob(os.path.join(image_dir, "*.png")), key=acquisition_date)
    if not files:
        raise SystemExit(f"No PNG images found in {image_dir}; set --images to the folder of segmented images.")
    tps = [time_point(f) for f in files]
    if all(tp is not None for tp in tps) and tps != list(range(1, len(files) + 1)):
        raise SystemExit(f"The T numbers in the file names of {image_dir} do not follow the acquisition dates.")
    if len(files) != 61:
        print(f"warning: {len(files)} images in {image_dir} (expected 61)")
    rows = []
    for t, path in enumerate(files, start=1):
        img = imread(path)
        f = {"T": f"T{t}"}
        f.update(color_and_size(img))
        f.update(shape_and_spectrum(img))
        rows.append(f)
    return pd.DataFrame(rows)[["T"] + INDICES]


SUFFIX = {"0": "", "120": "_120", "240": "_240"}


def compare(features, dataset_dir, year, angle, cultivar, sample, rtol):
    rows = []
    for name in INDICES:
        path = os.path.join(dataset_dir, f"{name}_{year}{SUFFIX[angle]}.csv")
        ref = pd.read_csv(path)
        hit = ref[(ref.iloc[:, 0].astype(str).str.strip() == cultivar) & (ref.iloc[:, 1] == sample)]
        ref_val = hit.iloc[0, 2:].to_numpy(dtype=float)
        val = features[name].to_numpy(dtype=float)
        diff = np.abs(val - ref_val)
        rel = diff / np.maximum(np.abs(ref_val), np.finfo(float).tiny)
        rows.append({"index": name, "n": len(val), "max_abs_diff": diff.max(),
                     "max_rel_diff": rel.max(), "n_mismatch": int((rel > rtol).sum())})
    return pd.DataFrame(rows)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", default="2023", choices=["2023", "2024"])
    ap.add_argument("--angle", default="0", choices=["0", "120", "240"])
    ap.add_argument("--images", default=None, help="default: ../Images/example/preprocessed/<year>/angle<angle>")
    ap.add_argument("--dataset", default=None, help="default: ../Preprocessed_dataset/<year>")
    ap.add_argument("--cultivar", default="Samcheonbyeo")
    ap.add_argument("--sample", type=int, default=1)
    ap.add_argument("--rtol", type=float, default=1e-6)
    ap.add_argument("--out", default=os.path.join(here, "results"))
    a = ap.parse_args()
    images = a.images or os.path.join(here, "..", "Images", "example", "preprocessed", a.year, f"angle{a.angle}")
    dataset = a.dataset or os.path.join(here, "..", "Preprocessed_dataset", a.year)

    os.makedirs(a.out, exist_ok=True)
    tag = f"{a.cultivar}_sample{a.sample}_{a.year}_angle{a.angle}"
    feats = extract(images)
    feats.to_csv(os.path.join(a.out, f"indices_{tag}.csv"), index=False)
    cmp_ = compare(feats, dataset, a.year, a.angle, a.cultivar, a.sample, a.rtol)
    cmp_.to_csv(os.path.join(a.out, f"comparison_{tag}.csv"), index=False)
    n_bad = int(cmp_["n_mismatch"].sum())
    print(f"{a.year}, angle {a.angle}: {len(feats)} images, {len(INDICES)} indices, max relative difference "
          f"{cmp_['max_rel_diff'].max():.2e}, mismatches (rtol={a.rtol:g}): {n_bad}")


if __name__ == "__main__":
    main()
