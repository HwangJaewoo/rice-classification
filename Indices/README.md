# Indices

Extraction of the 23 color and morphological indices from the preprocessed (segmented) images,
and comparison with the values in `Preprocessed_dataset`.

## Scripts

| Script | What it does |
|---|---|
| `run_all.py` | runs the two steps below for the example images (default: 2023 and 2024, 0°; other angles with `--angle 0 120 240`) |
| `extract_indices.py` | computes the 23 indices from the 61 segmented images of one plant, one year, and one angle |
| `compare_indices.py` | checks, for each set and index, whether the computed values are the same as in `Preprocessed_dataset` |

```
cd Indices
python run_all.py
```

Expected layout of the example images (file names `<cultivar>_sample<sample>_<year>_angle<angle>_T<tt>_<YYYY-MM-DD>.png`; the images are ordered by the date):

```
Images/example/preprocessed/<year>/angle0/   61 segmented PNG images, <year> = 2023, 2024 (released: Samcheonbyeo, sample 1)
```

| Option | Default | Meaning |
|---|---|---|
| `--images` | `../Images/example/preprocessed` | folder with `<year>/angle<angle>/` |
| `--dataset` | `../Preprocessed_dataset` | folder with `2023/` and `2024/` (released index CSV files) |
| `--cultivar`, `--sample` | `Samcheonbyeo`, `1` | cultivar name and sample number of the images |
| `--out` | `results` | output folder |

One set only: `python extract_indices.py --year 2024 --angle 0`, then `python compare_indices.py --year 2024 --angle 0` (120° and 240° with `--angle 120` or `--angle 240` if those images are available).

Outputs (in `results`):
- `indices_<cultivar>_sample<sample>_<year>_angle<angle>.csv`: the 23 indices for every image (T1–T61)
- `summary_<cultivar>_sample<sample>.csv`: for every set and index, the number of equal values and the maximum absolute difference

Criterion: |computed − released| ≤ 10⁻⁶ × max(1, |released|).
The released file for each angle is `<index>_<year>.csv` (0°), `<index>_<year>_120.csv`, or `<index>_<year>_240.csv`.

## Indices

Names in the CSV files and the notation in Table 1 of the article:

| Type | CSV name | Article | Computed from |
|---|---|---|---|
| Color (12) | `r`, `g`, `b` | $R, G, B$ | mean over the non-black (plant) pixels in RGB |
| | `h`, `s`, `v` | $H, S, V$ | same, in HSV |
| | `y`, `cb`, `cr` | $Y, Cb, Cr$ | same, in YCbCr |
| | `l_lab`, `a_lab`, `b_lab` | $L, A_{LAB}, B_{LAB}$ | same, in CIELAB |
| Size (4) | `vertical_h`, `width`, `diagonal_h`, `area` | $H_V, W, H_D, A$ | non-black pixels |
| Shape and texture (7) | `Peri` | $P$ | perimeter of the largest connected component of the Otsu-thresholded gray image |
| | `a1`, `b1`, `c1`, `d1` | $a_1, b_1, c_1, d_1$ | first-order elliptic Fourier descriptors of the boundary of that component |
| | `FSEP` | $F$ | share of the energy of the 2-D Fourier spectrum of the gray image in its central block |
| | `F_std` | $\sigma_{FFT}$ | standard deviation of the real part of the central block of that spectrum |

ExG, GLI, NGRDI, and AR are computed from these values in `Classification/` (see `Preprocessed_dataset/README.md`).

## Requirements

Python 3.10 or later, `numpy`, `pandas`, `opencv-python`.
