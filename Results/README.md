# Results

Results reported in the manuscript.
The files are copied from the working outputs of the scripts (`Classification/results/`, `Deep_learning/results/`).

## MATLAB/

One CSV file per sheet of the summary tables of `Classification/`, named `<table>_<sheet>.csv`.

| Folder | Tables |
|---|---|
| `single/` | `nested_single_<year>_k27` (representative setting), `batch_single_<year>` (27 combinations), `baselines_single_<year>`, `ablation_single_<year>`, `ablation_single_<year>_augsens`, `importance_single_<year>` |
| `mixed/` | `nested_mixed_<comb>_k27` (combination of the yearly F statistics: `geo`, `arith`, `min`, `pooled`), `batch_mixed_geo`, `importance_mixed_geo` (permutation importance of the representative setting) |
| `cross/` | `nested_cross_<direction>_k27` (`source` and `target`), `nested_cross_<direction>_k27_source_minmax`, `nested_cross_<direction>_k27_target_minmax`, `batch_cross_23to24`, `batch_cross_24to23` |
| `analysis/` | `confusion_pairs_k27`, `year_stability_k27`, `index_redundancy` (correlation and effective dimension of the color and morphological features) |

Accuracies are in %, macro-F1 between 0 and 1.
`k27` is the representative setting (Stage 1-3, all indices, angles 0°, 120° and 240°).
The Cross-year modes are explained in `Classification/README.md`.

`predictions_all.csv` contains the prediction for every test sample of the representative settings:

| Column | Content |
|---|---|
| `Scheme` | `Single`, `Mixed`, `Cross` |
| `Method` | classifier (`LDA`, `MNR`, `LOGIT`, `NC`, `SVM`, `RF`, `PLSDA`); for Cross-year, `LDA_<mode>` |
| `Direction` | Cross-year only: `23to24` or `24to23` |
| `Seed` | random seed (1–5) |
| `Fold` | outer fold = number of the test sample (1–4); 0 for Cross-year (all samples of the target year are tested) |
| `Year` | year of the test sample |
| `Cultivar`, `Plant` | true cultivar and sample number |
| `Predicted` | predicted cultivar |

Accuracies, macro-F1, and confusion matrices can be recomputed from this file.

## CNN/

| File | Content |
|---|---|
| `runs/*.npz` | one file per training run of `Deep_learning/train_cnn.py` |
| `cnn_<mode>_runs.csv`, `cnn_<mode>_summary.csv` | accuracy and macro-F1 of every outer run, and their means |
| `predictions_cnn.csv` | prediction for every test sample of the outer runs, in the format of `MATLAB/predictions_all.csv` (`Method` = `CNN`, seeds 1–3) |
| `meta_<year>_128.csv` | image folder (`plant`), cultivar, and sample number (`sample`) of each row of the image arrays |
| `read_npz.py` | prints the results of the npz files, and writes `predictions_cnn.csv` with `--predictions predictions_cnn.csv` |

```
cd Results/CNN
python read_npz.py
```

The contents of the npz files are listed at the top of `read_npz.py`.
