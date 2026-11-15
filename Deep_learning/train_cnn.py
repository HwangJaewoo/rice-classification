"""Shallow CNN baseline (Single-, Mixed- and Cross-year validation).

The architecture is in cnn_common.py. The learning rate (--lrs) and the number of epochs (1 to --max_epochs)
are selected by inner cross-validation, using only the training plants of each outer fold:
  Single-year : outer test = plant o of every cultivar; inner validation = each of the other three plants.
  Mixed-year  : outer test = plant o of every cultivar in both years; inner validation = each other plant number.
  Cross-year  : outer test = all plants of the target year; inner validation = each plant number of the source year.
For each learning rate and inner fold, the network is trained for --max_epochs epochs with a cosine schedule
and the validation accuracy is recorded after every epoch. The pair (learning rate, epoch) with the highest mean
validation accuracy is selected, and the final network is trained on all training plants with that learning rate
and the same schedule, stopped at the selected epoch, and evaluated once on the test plants.
Runs are checkpointed after every epoch, and finished runs are skipped when the script is started again.

Outputs (in --out): runs/<tag>.npz (accuracy, macro-F1, predictions, test rows, selected learning rate and epoch,
inner validation curves), cnn_<mode>_runs.csv, cnn_<mode>_summary.csv, and log_<time>.txt.

Usage:
    python train_cnn.py [--mode single mixed cross] [--seeds 1 2 3] [--lrs 3e-4 1e-3 3e-3] [--max_epochs 100]
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from cnn_common import ShallowCNN, load_all, macro_f1, save_atomic, split, to_input


class Tee:
    def __init__(self, path):
        self.f = open(path, "a", encoding="utf-8")
        self.out = sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)
        self.f.flush()

    def flush(self):
        self.out.flush()
        self.f.flush()


def predict(model, xt, te, batch, device, use_amp):
    model.eval()
    out = []
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for b in range(0, len(te), batch):
            out.append(model(to_input(xt[te[b:b + batch]], device)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def train(a, xs, ys, tr, n_class, tag, rseed, device, dirs, stop, lr, val=None, verbose=False):
    """Train on rows tr for `stop` epochs (cosine schedule over a.max_epochs).
    If val = (xv, yv, vi) is given, the validation accuracy after every epoch is returned."""
    ck_path = os.path.join(dirs["ckpt"], tag + ".pt")
    torch.manual_seed(rseed)
    rng = np.random.default_rng(rseed)
    model = ShallowCNN(n_class).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.max_epochs)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    start, hist, curve = 0, [], []
    if os.path.exists(ck_path):
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
        rng.bit_generator.state = ck["np_rng"]; torch.set_rng_state(ck["torch_rng"])
        if use_amp and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state(ck["cuda_rng"])
        start, hist, curve = ck["epoch"], ck["hist"], ck["curve"]
    y_all = torch.from_numpy(np.array(ys)).long()
    t0 = time.time()
    for ep in range(start, stop):
        model.train()
        perm = rng.permutation(tr)
        tot = 0.0
        for b in range(0, len(perm), a.batch):
            idx = perm[b:b + a.batch]
            flip = torch.from_numpy(rng.random(len(idx)) < 0.5).to(device)
            xb = to_input(xs[idx], device, flip)
            yb = y_all[idx].to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss = F.cross_entropy(model(xb), yb, label_smoothing=a.smooth)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            tot += loss.item() * len(idx)
        sched.step()
        hist.append(tot / len(perm))
        if val is not None:
            xv, yv, vi = val
            curve.append(float(np.mean(predict(model, xv, vi, a.batch, device, use_amp) == yv[vi]) * 100))
        save_atomic({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                     "scaler": scaler.state_dict(), "np_rng": rng.bit_generator.state,
                     "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state() if use_amp else None,
                     "epoch": ep + 1, "hist": hist, "curve": curve}, ck_path)
        if verbose and (ep + 1) % max(1, a.max_epochs // 5) == 0:
            msg = f"    epoch {ep + 1}/{stop}, loss {hist[-1]:.3f}"
            if curve:
                msg += f", val {curve[-1]:.1f}%"
            print(msg + f", {time.time() - t0:.0f} s", flush=True)
    return model, np.array(curve), np.array(hist), use_amp


def inner_curve(a, xs, ys, folds, n_class, tag, rseed, device, dirs, lr, verbose):
    """folds: list of (train rows, validation rows). Returns the mean validation curve for learning rate lr."""
    curves = []
    for j, (tr, vi) in enumerate(folds, start=1):
        path = os.path.join(dirs["runs"], f"{tag}_in{j}.npz")
        if os.path.exists(path):
            curves.append(np.load(path)["curve"])
            continue
        _, curve, hist, _ = train(a, xs, ys, tr, n_class, f"{tag}_in{j}", rseed * 10 + j, device, dirs,
                                  a.max_epochs, lr, (xs, ys, vi), verbose and j == 1)
        np.savez(path + ".tmp.npz", curve=curve, loss=hist)
        os.replace(path + ".tmp.npz", path)
        curves.append(curve)
    return np.mean(curves, axis=0)


def outer(a, xs, ys, tr, inner_folds, xt, yt, te, n_class, tag, rseed, device, dirs, verbose):
    path = os.path.join(dirs["runs"], tag + ".npz")
    if os.path.exists(path):
        r = np.load(path)
        return float(r["acc"]), float(r["f1"]), f"{float(r['lr']):g}/{int(r['epochs'])}", True, r["pred"], r["y"]
    curves = np.stack([inner_curve(a, xs, ys, inner_folds, n_class, f"{tag}_lr{k}", rseed, device, dirs, lr,
                                   verbose and k == 0) for k, lr in enumerate(a.lrs)])
    k_star, e_idx = np.unravel_index(int(np.argmax(curves)), curves.shape)
    lr_star, e_star = a.lrs[k_star], int(e_idx) + 1
    curve = curves[k_star]
    model, _, hist, use_amp = train(a, xs, ys, tr, n_class, tag + "_final", rseed, device, dirs, e_star, lr_star)
    pred = predict(model, xt, te, a.batch, device, use_amp)
    y_te = yt[te]
    acc = float(np.mean(pred == y_te) * 100)
    f1 = macro_f1(y_te, pred, n_class)
    torch.save(model.state_dict(), os.path.join(dirs["models"], tag + "_final.pt"))
    np.savez(path + ".tmp.npz", acc=acc, f1=f1, epochs=e_star, lr=lr_star, inner_curves=curves, lrs=np.array(a.lrs),
             pred=pred, y=y_te, test_rows=te, loss=hist)
    os.replace(path + ".tmp.npz", path)
    return acc, f1, f"{lr_star:g}/{e_star}", False, pred, y_te


def report(rows, name, out, keys):
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(out, f"cnn_{name}_runs.csv"), index=False)
    S = R.groupby(keys + ["Seed"]).mean(numeric_only=True).reset_index()
    T = S.groupby(keys).agg(Mean=("Acc", "mean"), SeedSD=("Acc", "std"), MacroF1=("MacroF1", "mean"))
    if "Fold" in R:
        T["FoldSD"] = R.groupby(keys + ["Seed"])["Acc"].std().groupby(keys).mean()
    for c in [c for c in R.columns if c.startswith("Part")]:
        T[c] = S.groupby(keys)[c].mean()
    T.to_csv(os.path.join(out, f"cnn_{name}_summary.csv"))
    print(f"\n[{name}]\n" + T.round(3).to_string(), flush=True)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", nargs="*", default=["single", "mixed", "cross"])
    ap.add_argument("--year", nargs="*", default=["2023", "2024"])
    ap.add_argument("--seeds", nargs="*", type=int, default=[1, 2, 3])
    ap.add_argument("--size_in", type=int, default=128)
    ap.add_argument("--size", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lrs", nargs="*", type=float, default=[3e-4, 1e-3, 3e-3])
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--smooth", type=float, default=0.1)
    ap.add_argument("--data", default=os.path.join(here, "data"))
    ap.add_argument("--out", default=os.path.join(here, "results"))
    a = ap.parse_args()

    dirs = {k: os.path.join(a.out, k) for k in ("runs", "ckpt", "models")}
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    sys.stdout = Tee(os.path.join(a.out, f"log_{time.strftime('%Y%m%d_%H%M%S')}.txt"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device {device} | modes {a.mode} | seeds {a.seeds} | lrs {a.lrs} | {a.size}px | max {a.max_epochs} epochs",
          flush=True)
    data = load_all(a, device)
    T0 = time.time()

    if "single" in a.mode:
        rows = []
        for year, (x, y, smp, names) in data.items():
            for s in a.seeds:
                for o in range(1, 5):
                    tr, te = split(smp, o)
                    inner = [(tr[smp[tr] != p], tr[smp[tr] == p]) for p in sorted(set(smp[tr]))]
                    acc, f1, e, cached, _, _ = outer(a, x, y, tr, inner, x, y, te, len(names),
                                                     f"single_{year}_s{s}_f{o}", 1000 * s + o, device, dirs,
                                                     verbose=(s == a.seeds[0] and o == 1))
                    rows.append({"Year": year, "Seed": s, "Fold": o, "Acc": acc, "MacroF1": f1, "LrEpochs": e})
                    print(f"  single {year} seed {s} fold {o}: {acc:.2f}% (F1 {f1:.3f}, lr/epochs {e})"
                          f"{' [done before]' if cached else ''} | {(time.time() - T0) / 60:.1f} min", flush=True)
        report(rows, "single", a.out, ["Year"])

    if ("mixed" in a.mode or "cross" in a.mode) and len(data) == 2:
        (y1, d1), (y2, d2) = sorted(data.items())
        if d1[3] != d2[3]:
            raise ValueError("cultivar names differ between years")
        n_class = len(d1[3])

        if "mixed" in a.mode:
            x = torch.cat([d1[0], d2[0]])
            y = np.concatenate([d1[1], d2[1]])
            smp = np.concatenate([d1[2], d2[2]])
            yr = np.concatenate([np.full(len(d1[1]), y1), np.full(len(d2[1]), y2)])
            rows = []
            for s in a.seeds:
                for o in range(1, 5):
                    tr, te = split(smp, o)
                    inner = [(tr[smp[tr] != p], tr[smp[tr] == p]) for p in sorted(set(smp[tr]))]
                    acc, f1, e, cached, pred, yte = outer(a, x, y, tr, inner, x, y, te, n_class,
                                                          f"mixed_s{s}_f{o}", 1000 * s + o + 500, device, dirs,
                                                          verbose=(s == a.seeds[0] and o == 1))
                    part = {q: float(np.mean(pred[yr[te] == q] == yte[yr[te] == q]) * 100) for q in (y1, y2)}
                    rows.append({"Seed": s, "Fold": o, "Acc": acc, f"Part{y1}": part[y1], f"Part{y2}": part[y2],
                                 "MacroF1": f1, "LrEpochs": e, "Set": "mixed"})
                    print(f"  mixed seed {s} fold {o}: {acc:.2f}% ({y1} {part[y1]:.2f}, {y2} {part[y2]:.2f}, "
                          f"F1 {f1:.3f}, lr/epochs {e}){' [done before]' if cached else ''}"
                          f" | {(time.time() - T0) / 60:.1f} min", flush=True)
            del x
            report(rows, "mixed", a.out, ["Set"])

        if "cross" in a.mode:
            rows = []
            for src, tgt, ds, dt in [(y1, y2, d1, d2), (y2, y1, d2, d1)]:
                tr, te = np.arange(len(ds[1])), np.arange(len(dt[1]))
                inner = [(tr[ds[2] != p], tr[ds[2] == p]) for p in range(1, 5)]
                for s in a.seeds:
                    acc, f1, e, cached, pred, yte = outer(a, ds[0], ds[1], tr, inner, dt[0], dt[1], te, n_class,
                                                          f"cross_{src}to{tgt}_s{s}", 1000 * s + 9, device, dirs,
                                                          verbose=(s == a.seeds[0]))
                    pa = [float(np.mean(pred[dt[2] == q] == yte[dt[2] == q]) * 100) for q in range(1, 5)]
                    rows.append({"Direction": f"{src}->{tgt}", "Seed": s, "Acc": acc,
                                 "PlantSD": float(np.std(pa, ddof=1)), "MacroF1": f1, "LrEpochs": e})
                    print(f"  cross {src}->{tgt} seed {s}: {acc:.2f}% (F1 {f1:.3f}, lr/epochs {e})"
                          f"{' [done before]' if cached else ''} | {(time.time() - T0) / 60:.1f} min", flush=True)
            report(rows, "cross", a.out, ["Direction"])

    print(f"\nall done | {(time.time() - T0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
