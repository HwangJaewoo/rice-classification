# Classification

Feature extraction, classification, and validation in MATLAB.
The input is `Preprocessed_dataset/`.

## Folders

| Folder | Content |
|---|---|
| `Common/` | settings (`config.m`), data loading, features, ANOVA selection, shrinkage LDA, helpers |
| `Single/` | Single-year validation, baseline classifiers, ablation, feature importance |
| `Mixed/` | Mixed-year validation |
| `Cross/` | Cross-year validation |
| `Analysis/` | confusion analysis, variance components between years, export of all predictions |

## Running

Run each script from its own folder, in this order:

1. `Single/run_single.m`
2. `Mixed/run_mixed.m`
3. `Cross/run_cross.m`
4. `Analysis/run_analysis.m` (uses the saved results of 1–3)

`Single/check_common.m` is a quick test of the data path and the common functions (one fold, no optimization).

Results are written to `results/single/`, `results/mixed/`, `results/cross/`, and `results/analysis/` as `.mat` and `.xlsx` files, with a log of each run.
Finished jobs are saved as checkpoints and skipped when a script is started again;
if a setting is changed, delete or move the `results/` folder first.

## Method

- Features: the index time series are smoothed (moving average over three time points);
  for each stage, six statistics per index and viewing angle, and the differences between the stage means (1620 features for all indices, stages, and angles).
- Classifier: ANOVA F ranking, standardization, and LDA with shrinkage, fitted on the training samples only.
- Validation: nested cross-validation over samples.
  The outer loop tests sample o (o = 1–4) of every cultivar;
  the inner loop selects the number of features and the shrinkage parameter by Bayesian optimization (50 evaluations),
  using only the training samples.
  Every setting is repeated with five random seeds.
- Cross-year: training on all samples of one year and testing on all samples of the other year.
  The features of the test year are standardized in one of four ways (`mode`):

  | Mode | Standardization of the test year |
  |---|---|
  | `source` | with the mean and SD of the training year |
  | `target` | with its own mean and SD (no labels) |
  | `source_minmax` | min-max scaling of the time series (minimum and maximum of the training samples), then as `source` |
  | `target_minmax` | min-max scaling as above, then as `target` |

  The hyperparameters are always selected within the training year.

Settings (stages, indices, angles, optimization) are in `Common/config.m`.

## Requirements

MATLAB R2024b with the Statistics and Machine Learning Toolbox and the Parallel Computing Toolbox.
