"""Shared parts of the shallow CNN baseline (data loading, missing days, model, helpers).

Missing days are filled with the nearest available day of the same plant and angle (labels are not used).
Pixel values are scaled by 1/255 (no statistics estimated from data).
"""
import os

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

N_ANGLES, N_T = 3, 61


def fill_missing(mask):
    """Index of the nearest available day for every (plant, angle, day); ties go to the earlier day."""
    P, A, T = mask.shape
    src = np.zeros((P, A, T), np.int64)
    t = np.arange(T)
    for p in range(P):
        for a in range(A):
            ok = np.flatnonzero(mask[p, a])
            if len(ok) == 0:
                raise ValueError(f"plant row {p}, angle {a} has no image")
            d = np.abs(t[:, None] - ok[None, :])
            src[p, a] = ok[np.argmin(d, axis=1)]
    return src


def split(samples, fold):
    test = np.flatnonzero(samples == fold)
    train = np.flatnonzero(samples != fold)
    return train, test


def load_year(data, year, size_in, size):
    meta = pd.read_csv(os.path.join(data, f"meta_{year}_{size_in}.csv"))
    imgs = np.load(os.path.join(data, f"images_{year}_{size_in}.npy"), mmap_mode="r")
    mask = np.load(os.path.join(data, f"mask_{year}_{size_in}.npy"))
    if len(meta) != imgs.shape[0] or mask.shape != imgs.shape[:3]:
        raise ValueError("meta, images and mask do not match")
    src = fill_missing(mask)
    P, A, T = mask.shape
    x = np.empty((P, A, T, size, size, 3), np.uint8)
    for p in range(P):
        for q in range(A):
            for t in range(T):
                f = imgs[p, q, src[p, q, t]]
                x[p, q, t] = f if size == size_in else cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA)
    names = sorted(meta.cultivar.unique())
    labels = meta.cultivar.map({n: i for i, n in enumerate(names)}).to_numpy().copy()
    return torch.from_numpy(x), labels, meta["sample"].to_numpy().astype(int), names, int((~mask).sum())


class ShallowCNN(nn.Module):
    def __init__(self, n_class, in_ch=3 * N_ANGLES, width=(32, 64, 96, 128), t_kernel=5, dropout=0.3):
        super().__init__()
        layers, c = [], in_ch
        for w in width:
            layers += [nn.Conv2d(c, w, 3, padding=1, bias=False), nn.BatchNorm2d(w), nn.ReLU(inplace=True),
                       nn.MaxPool2d(2)]
            c = w
        self.frame = nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.temporal = nn.Sequential(nn.Conv1d(c, c, t_kernel, padding=t_kernel // 2, bias=False),
                                      nn.BatchNorm1d(c), nn.ReLU(inplace=True))
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(c, n_class))

    def forward(self, x):
        B, T = x.shape[:2]
        f = self.frame(x.flatten(0, 1)).view(B, T, -1)
        h = self.temporal(f.transpose(1, 2)).mean(dim=2)
        return self.head(h)


def to_input(xb, device, flip=None):
    B, A, T, H, W, C = xb.shape
    x = xb.to(device, non_blocking=True).permute(0, 2, 1, 5, 3, 4).reshape(B, T, A * C, H, W).float() / 255.0
    if flip is not None:
        x = torch.where(flip.view(B, 1, 1, 1, 1), x.flip(-1), x)
    return x


def macro_f1(y, p, n_class):
    f = []
    for c in range(n_class):
        tp = np.sum((p == c) & (y == c))
        pr = tp / max(np.sum(p == c), 1)
        rc = tp / max(np.sum(y == c), 1)
        f.append(0.0 if pr + rc == 0 else 2 * pr * rc / (pr + rc))
    return float(np.mean(f))


def save_atomic(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def load_all(a, device):
    data = {}
    for year in a.year:
        if not os.path.exists(os.path.join(a.data, f"images_{year}_{a.size_in}.npy")):
            print(f"{year}: data not found in {a.data}, skipped", flush=True)
            continue
        x, labels, samples, names, n_fill = load_year(a.data, year, a.size_in, a.size)
        data[year] = (x.to(device) if device.type == "cuda" else x, labels, samples, names)
        print(f"{year}: {len(labels)} plants, {len(names)} cultivars, filled days {n_fill}", flush=True)
    return data
