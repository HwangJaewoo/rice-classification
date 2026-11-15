# Preprocessed_dataset

Time series of the 23 image-derived indices for every sample (derived data).
The indices were computed from the preprocessed (segmented) images;
`Indices/extract_indices.py` computes them from the images of one sample.

## Folder layout

```
Preprocessed_dataset/
  2023/  <index>_2023<angle>.csv   (69 files)
  2024/  <index>_2024<angle>.csv   (69 files)
```

| Part | Values |
|---|---|
| `<index>` | color (12): `r`, `g`, `b`, `h`, `s`, `v`, `y`, `cb`, `cr`, `l_lab`, `a_lab`, `b_lab` |
| | morphology (11): `width`, `diagonal_h`, `area`, `vertical_h`, `Peri`, `FSEP`, `a1`, `b1`, `c1`, `d1`, `F_std` |
| `<angle>` | (empty) = 0° (SV000), `_120` = 120° (SV120), `_240` = 240° (SV240) |

## File format

Each file has one header row and 388 data rows (97 cultivars × 4 samples).
Each cultivar occupies four consecutive rows (samples 1–4),
and the order of the rows is the same in all files and in both years.

| Column | Content |
|---|---|
| 1 | cultivar name |
| 2 | sample number (1–4) |
| 3–63 | index value at time points T1–T61 |

The time points T1–T61 are the first 61 acquisition dates of each year:
2023-06-14 to 2023-10-18 and 2024-06-06 to 2024-10-04.
The definitions of the indices are given in Table 1 of the article and in `Indices/extract_indices.py`.

The remaining four indices of the article (ExG, GLI, NGRDI, and the aspect ratio AR = width / vertical_h)
are computed from these files in MATLAB (`Classification/Common/load_store.m`).

## Missing images

Images are missing at the following time points:

- 2023, T44: sample 3 of Samcheonbyeo (240° only).
- 2024, T52: sample 4 of all 97 cultivars, and sample 3 of 9 cultivars
  (Sodami, Sukwang, Huimangchan, Saeshin, Youngjin, Aromi, Yechan, Chilbo, Unbongbyeo); all three angles.

The index values at these time points were replaced by the mean values of the other plants of the same cultivar on the same date.
