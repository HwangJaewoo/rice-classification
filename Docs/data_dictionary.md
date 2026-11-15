# Data dictionary

## Files

`Preprocessed_dataset/<year>/<index>_<year><angle>.csv`

| Part | Values |
|---|---|
| `<year>` | 2023, 2024 |
| `<index>` | color: r, g, b, h, s, v, y, cb, cr, l_lab, a_lab, b_lab; morphology: width, diagonal_h, area, vertical_h, Peri, FSEP, a1, b1, c1, d1, F_std |
| `<angle>` | empty = 0° (SV000), `_120` = 120° (SV120), `_240` = 240° (SV240) |

ExG, GLI, NGRDI and AR (aspect ratio, width / vertical_h) are not stored; they are computed from r, g, b, width and vertical_h in `Classification/Common/load_store.m`.

## Rows and columns

Each file has 388 rows (97 cultivars x 4 samples): four consecutive rows per cultivar (samples 1-4), with the same cultivar order in all files and both years.

| Column | Content |
|---|---|
| 1 | cultivar name |
| 2 | sample number, 1-4 |
| 3-63 | index value at time points T1-T61 |

## Time points

T1-T61 are the first 61 acquisition dates of each year
(2023: 2023-06-14 to 2023-10-18; 2024: 2024-06-06 to 2024-10-04).

| Year | Sowing | Transplanting | T1 | T61 |
|---|---|---|---|---|
| 2023 | 2023-06-02 | 2023-06-10 | 2023-06-14 | 2023-10-18 |
| 2024 | 2024-05-24 | 2024-06-03 | 2024-06-06 | 2024-10-04 |

Acquisition date of each time point and days after sowing (DAS = date − sowing date).
Images that could not be taken on the scheduled date were taken on the next day; the file names of the example images give the actual date (2024 T35 of the example plant: 2024-08-14).
In 2023, three intervals are four days because of equipment repair.

| T | 2023 | DAS | 2024 | DAS |
|---|---|---|---|---|
| T1 | 2023-06-14 | 12 | 2024-06-06 | 13 |
| T2 | 2023-06-16 | 14 | 2024-06-08 | 15 |
| T3 | 2023-06-18 | 16 | 2024-06-10 | 17 |
| T4 | 2023-06-20 | 18 | 2024-06-12 | 19 |
| T5 | 2023-06-22 | 20 | 2024-06-14 | 21 |
| T6 | 2023-06-24 | 22 | 2024-06-16 | 23 |
| T7 | 2023-06-26 | 24 | 2024-06-18 | 25 |
| T8 | 2023-06-28 | 26 | 2024-06-20 | 27 |
| T9 | 2023-06-30 | 28 | 2024-06-22 | 29 |
| T10 | 2023-07-02 | 30 | 2024-06-24 | 31 |
| T11 | 2023-07-04 | 32 | 2024-06-26 | 33 |
| T12 | 2023-07-06 | 34 | 2024-06-28 | 35 |
| T13 | 2023-07-08 | 36 | 2024-06-30 | 37 |
| T14 | 2023-07-10 | 38 | 2024-07-02 | 39 |
| T15 | 2023-07-12 | 40 | 2024-07-04 | 41 |
| T16 | 2023-07-14 | 42 | 2024-07-06 | 43 |
| T17 | 2023-07-16 | 44 | 2024-07-08 | 45 |
| T18 | 2023-07-18 | 46 | 2024-07-10 | 47 |
| T19 | 2023-07-20 | 48 | 2024-07-12 | 49 |
| T20 | 2023-07-22 | 50 | 2024-07-14 | 51 |
| T21 | 2023-07-24 | 52 | 2024-07-16 | 53 |
| T22 | 2023-07-26 | 54 | 2024-07-18 | 55 |
| T23 | 2023-07-28 | 56 | 2024-07-20 | 57 |
| T24 | 2023-07-30 | 58 | 2024-07-22 | 59 |
| T25 | 2023-08-01 | 60 | 2024-07-24 | 61 |
| T26 | 2023-08-03 | 62 | 2024-07-26 | 63 |
| T27 | 2023-08-05 | 64 | 2024-07-28 | 65 |
| T28 | 2023-08-07 | 66 | 2024-07-30 | 67 |
| T29 | 2023-08-09 | 68 | 2024-08-01 | 69 |
| T30 | 2023-08-11 | 70 | 2024-08-03 | 71 |
| T31 | 2023-08-13 | 72 | 2024-08-05 | 73 |
| T32 | 2023-08-15 | 74 | 2024-08-07 | 75 |
| T33 | 2023-08-17 | 76 | 2024-08-09 | 77 |
| T34 | 2023-08-19 | 78 | 2024-08-11 | 79 |
| T35 | 2023-08-21 | 80 | 2024-08-13 | 81 |
| T36 | 2023-08-25 | 84 | 2024-08-15 | 83 |
| T37 | 2023-08-27 | 86 | 2024-08-17 | 85 |
| T38 | 2023-08-29 | 88 | 2024-08-19 | 87 |
| T39 | 2023-08-31 | 90 | 2024-08-21 | 89 |
| T40 | 2023-09-02 | 92 | 2024-08-23 | 91 |
| T41 | 2023-09-06 | 96 | 2024-08-25 | 93 |
| T42 | 2023-09-10 | 100 | 2024-08-27 | 95 |
| T43 | 2023-09-12 | 102 | 2024-08-29 | 97 |
| T44 | 2023-09-14 | 104 | 2024-08-31 | 99 |
| T45 | 2023-09-16 | 106 | 2024-09-02 | 101 |
| T46 | 2023-09-18 | 108 | 2024-09-04 | 103 |
| T47 | 2023-09-20 | 110 | 2024-09-06 | 105 |
| T48 | 2023-09-22 | 112 | 2024-09-08 | 107 |
| T49 | 2023-09-24 | 114 | 2024-09-10 | 109 |
| T50 | 2023-09-26 | 116 | 2024-09-12 | 111 |
| T51 | 2023-09-28 | 118 | 2024-09-14 | 113 |
| T52 | 2023-09-30 | 120 | 2024-09-16 | 115 |
| T53 | 2023-10-02 | 122 | 2024-09-18 | 117 |
| T54 | 2023-10-04 | 124 | 2024-09-20 | 119 |
| T55 | 2023-10-06 | 126 | 2024-09-22 | 121 |
| T56 | 2023-10-08 | 128 | 2024-09-24 | 123 |
| T57 | 2023-10-10 | 130 | 2024-09-26 | 125 |
| T58 | 2023-10-12 | 132 | 2024-09-28 | 127 |
| T59 | 2023-10-14 | 134 | 2024-09-30 | 129 |
| T60 | 2023-10-16 | 136 | 2024-10-02 | 131 |
| T61 | 2023-10-18 | 138 | 2024-10-04 | 133 |

## Stages used in the analysis

Stage 1 = T1-T25, Stage 2 = T26-T39, Stage 3 = T40-T61 (`Classification/Common/config.m`).

## Missing images

- 2023, T44: sample 3 of Samcheonbyeo (240° only).
- 2024, T52: sample 4 of all 97 cultivars, and sample 3 of 9 cultivars
  (Sodami, Sukwang, Huimangchan, Saeshin, Youngjin, Aromi, Yechan, Chilbo, Unbongbyeo); all three angles.

The index values at these time points were replaced by the mean values of the other plants of the same cultivar on the same date.

## Corrected images

Images corrected by histogram matching, and the reference image of each, are listed in `reference_log_<year>_<angle>.csv` written by `Image_preprocess/preprocess.py`.
