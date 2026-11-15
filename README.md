# Classification of rice cultivars using time series of growth images

Code and derived data for the manuscript
"Classification of rice cultivars using time series of growth images" (Computers and Electronics in Agriculture, revised version).

- Archived example images: https://doi.org/10.5281/zenodo.24070543
- License: code under the MIT License (`LICENSE`); data (`Preprocessed_dataset/`, `Results/`) and example images under CC BY 4.0
- Citation: please cite the article (reference to be added on publication) and the archived example images (DOI above)
- Contact: Kyung-Min Kim (kkm@knu.ac.kr), corresponding author

## Contents

| Folder | Content | Language |
|---|---|---|
| `Image_preprocess/` | illumination correction, cropping, segmentation of the original images | Python |
| `Indices/` | extraction of the 23 color and morphological indices stored in the CSV files from segmented images, and comparison with the released CSV files | Python |
| `Preprocessed_dataset/` | time series of the 23 indices for every sample, 2023 and 2024 (derived data); the other four of the 27 indices in the manuscript (ExG, GLI, NGRDI, AR) are computed from them in `Classification/Common/load_store.m` | CSV |
| `Classification/` | features, classifier, nested cross-validation, baselines, ablation, analyses | MATLAB |
| `Deep_learning/` | shallow CNN baseline on the segmented image sequences | Python (PyTorch) |
| `Results/` | summary tables and predictions of every test sample (`MATLAB/`), CNN results (`CNN/`) | CSV, NPZ |
| `Docs/` | data dictionary | |

Because of their size, the original images are not included; the full set is available from the corresponding author upon reasonable request.
Example images of one plant (Samcheonbyeo, sample 1, 0° view, T1–T61, 2023 and 2024) are archived with the DOI above (`raw.zip` and `preprocessed.zip`; see Section 1 below and `Images/README.md`).

## Requirements

- MATLAB R2024b with the Statistics and Machine Learning Toolbox and the Parallel Computing Toolbox.
- Python 3.10 or later: `pip install -r requirements.txt` (PyTorch only for `Deep_learning/`).

## Reproducing the results

All paths are relative to the repository root; results are written to `Classification/results/` and `Deep_learning/results/`.
Every script saves checkpoints and skips finished jobs when it is restarted.
If a setting is changed, use a new results folder, because the checkpoint names do not contain the settings.

### 1. Quick check with the example images

Download `raw.zip` and `preprocessed.zip` from the DOI above and unzip them into `Images/example/raw/` and `Images/example/preprocessed/`, so that the folders are:

```
Images/example/raw/2023/angle0/            61 original images (2023)
Images/example/raw/2024/angle0/            61 original images (2024)
Images/example/preprocessed/2023/angle0/   61 preprocessed images (2023)
Images/example/preprocessed/2024/angle0/   61 preprocessed images (2024)
```

Python, from the repository root (PyTorch is not needed for this check):

```
pip install numpy pandas opencv-python Pillow scikit-image
python Image_preprocess/reproduce_example.py
python Indices/run_all.py
```

- `reproduce_example.py` preprocesses the original images again and prints, for each year, the number of images identical to the preprocessed images.
- `run_all.py` computes the 23 indices of the preprocessed images and prints, for each year and index, whether they agree with `Preprocessed_dataset/`.

MATLAB, from the repository root (no images needed):

```
cd Classification/Single
check_common('2023')
```

Loads `Preprocessed_dataset/2023`, prints the number of features of four settings, and evaluates one fold with fixed hyperparameters (no optimization).

### 2. Classification (MATLAB)

Run from the folder of each script:

| Script | Content |
|---|---|
| `Classification/Single/run_single.m` | Single-year: representative setting, 27 combinations of indices, stages and angles, baseline classifiers, ablation, feature importance, augmentation sensitivity |
| `Classification/Mixed/run_mixed.m` | Mixed-year: representative setting, 27 combinations, combinations of the yearly F statistics |
| `Classification/Cross/run_cross.m` | Cross-year: both directions, source- and target-year standardization, 27 combinations, min-max scaling |
| `Classification/Analysis/run_analysis.m` | confusion analysis, variance components between years, predictions of every test sample (`predictions_all.csv`) |

Settings (stages, indices, angles, Bayesian optimization) are in `Classification/Common/config.m`.

Validation: in Single-year and Mixed-year validation, the outer loop tests sample o (o = 1-4) of every cultivar (of both years in Mixed-year validation),
and the inner loop selects the number of features and the shrinkage parameter by three-fold cross-validation within the training samples.
In Cross-year validation, all samples of the training year are used with four-fold inner cross-validation, and the other year is tested once.
Test samples are never used for the selection.

### 3. Deep-learning baseline (Python)

```
cd Deep_learning
python inventory.py --root ../Images       # matches image folders to the CSV rows, aligns dates to T1-T61
python pack.py --root ../Images            # resized image arrays in data/
python train_cnn.py                        # Single-, Mixed- and Cross-year; learning rate and epochs chosen by inner cross-validation
```

## Data

See `Docs/data_dictionary.md`.
