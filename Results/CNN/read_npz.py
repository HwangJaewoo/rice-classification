"""Read the result files of Deep_learning/train_cnn.py.

Usage (from this folder):
    python read_npz.py                      # table of all outer runs (accuracy, macro-F1, selected learning rate and epochs)
    python read_npz.py runs/single_2023_s1_f1.npz
    python read_npz.py --predictions predictions_cnn.csv

Outer runs (runs/<mode>_..._s<seed>[_f<fold>].npz) contain:
    acc, f1        accuracy (%) and macro-F1 on the test plants
    lr, epochs     learning rate and number of epochs selected by inner cross-validation
    lrs            candidate learning rates
    inner_curves   mean inner validation accuracy, candidate learning rate x epoch
    test_rows      rows of the test plants in meta_<year>_128.csv (Mixed: 2023 rows, then 2024 rows)
    y, pred        true and predicted class (index into the alphabetically sorted cultivar names)
    loss           training loss per epoch of the final model
Inner runs (..._lr<k>_in<j>.npz) contain the validation curve (curve) and training loss (loss) of one inner fold.
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd

OUTER = re.compile(r"^(single)_(\d{4})_s(\d+)_f(\d+)$|^(mixed)_s(\d+)_f(\d+)$|^(cross)_(\d{4})to(\d{4})_s(\d+)$")


def parse(name):
    m = OUTER.match(name)
    if not m:
        return None
    g = m.groups()
    if g[0]:
        return {"Scheme": "Single", "Direction": "", "Year": g[1], "Seed": int(g[2]), "Fold": int(g[3])}
    if g[4]:
        return {"Scheme": "Mixed", "Direction": "", "Year": "", "Seed": int(g[5]), "Fold": int(g[6])}
    return {"Scheme": "Cross", "Direction": f"{g[8][2:]}to{g[9][2:]}", "Year": g[9], "Seed": int(g[10]), "Fold": 0}


def show(path):
    r = np.load(path)
    print(f"{os.path.basename(path)}: accuracy {float(r['acc']):.2f}%, macro-F1 {float(r['f1']):.3f}, "
          f"learning rate {float(r['lr']):g}, epochs {int(r['epochs'])}, test plants {len(r['y'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", nargs="?")
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--meta", default=".")
    ap.add_argument("--predictions", default=None, help="write the predictions of all outer runs to this CSV file")
    a = ap.parse_args()

    if a.file:
        show(a.file)
        return

    meta = {y: pd.read_csv(os.path.join(a.meta, f"meta_{y}_128.csv")) for y in ["2023", "2024"]}
    names = sorted(meta["2023"].cultivar.unique())
    rows, preds = [], []
    for path in sorted(glob.glob(os.path.join(a.runs, "*.npz"))):
        info = parse(os.path.splitext(os.path.basename(path))[0])
        if info is None:
            continue
        r = np.load(path)
        rows.append({**info, "Acc": float(r["acc"]), "MacroF1": float(r["f1"]),
                     "LearningRate": float(r["lr"]), "Epochs": int(r["epochs"])})
        if a.predictions:
            if info["Scheme"] == "Mixed":
                m = pd.concat([meta["2023"].assign(Year="2023"), meta["2024"].assign(Year="2024")], ignore_index=True)
            else:
                m = meta[info["Year"]].assign(Year=info["Year"])
            t = m.iloc[r["test_rows"]]
            if list(t.cultivar) != [names[i] for i in r["y"]]:
                raise ValueError(f"{path}: test rows and labels do not match")
            preds.append(pd.DataFrame({
                "Scheme": info["Scheme"], "Method": "CNN", "Direction": info["Direction"], "Seed": info["Seed"],
                "Fold": info["Fold"], "Year": t.Year.to_numpy(), "Cultivar": [names[i] for i in r["y"]],
                "Plant": t["sample"].to_numpy(), "Predicted": [names[i] for i in r["pred"]]}))

    R = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    keys = ["Scheme", "Direction", "Year"]
    S = R.groupby(keys + ["Seed"])[["Acc", "MacroF1"]].mean().groupby(keys).agg(["mean", "std"])
    print("\nMean over seeds (SD between seeds)")
    print(S.round(3).to_string())
    if a.predictions:
        P = pd.concat(preds, ignore_index=True)
        P.to_csv(a.predictions, index=False)
        print(f"\n{len(P)} predictions written to {a.predictions}")


if __name__ == "__main__":
    main()
