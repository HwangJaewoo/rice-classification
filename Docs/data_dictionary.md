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
| 2023 | 2023-05-02 | 2023-06-02 | 2023-06-14 | 2023-10-18 |
| 2024 | 2024-05-03 | 2024-06-03 | 2024-06-06 | 2024-10-04 |

Acquisition date of each time point and days after sowing (DAS = date − sowing date).
Images that could not be taken on the scheduled date were taken on the next day; the file names of the example images give the actual date (2024 T35 of the example plant: 2024-08-14).
In 2023, three intervals are four days because of equipment repair.

| T | 2023 | DAS | 2024 | DAS |
|---|---|---|---|---|
| T1 | 2023-06-14 | 43 | 2024-06-06 | 34 |
| T2 | 2023-06-16 | 45 | 2024-06-08 | 36 |
| T3 | 2023-06-18 | 47 | 2024-06-10 | 38 |
| T4 | 2023-06-20 | 49 | 2024-06-12 | 40 |
| T5 | 2023-06-22 | 51 | 2024-06-14 | 42 |
| T6 | 2023-06-24 | 53 | 2024-06-16 | 44 |
| T7 | 2023-06-26 | 55 | 2024-06-18 | 46 |
| T8 | 2023-06-28 | 57 | 2024-06-20 | 48 |
| T9 | 2023-06-30 | 59 | 2024-06-22 | 50 |
| T10 | 2023-07-02 | 61 | 2024-06-24 | 52 |
| T11 | 2023-07-04 | 63 | 2024-06-26 | 54 |
| T12 | 2023-07-06 | 65 | 2024-06-28 | 56 |
| T13 | 2023-07-08 | 67 | 2024-06-30 | 58 |
| T14 | 2023-07-10 | 69 | 2024-07-02 | 60 |
| T15 | 2023-07-12 | 71 | 2024-07-04 | 62 |
| T16 | 2023-07-14 | 73 | 2024-07-06 | 64 |
| T17 | 2023-07-16 | 75 | 2024-07-08 | 66 |
| T18 | 2023-07-18 | 77 | 2024-07-10 | 68 |
| T19 | 2023-07-20 | 79 | 2024-07-12 | 70 |
| T20 | 2023-07-22 | 81 | 2024-07-14 | 72 |
| T21 | 2023-07-24 | 83 | 2024-07-16 | 74 |
| T22 | 2023-07-26 | 85 | 2024-07-18 | 76 |
| T23 | 2023-07-28 | 87 | 2024-07-20 | 78 |
| T24 | 2023-07-30 | 89 | 2024-07-22 | 80 |
| T25 | 2023-08-01 | 91 | 2024-07-24 | 82 |
| T26 | 2023-08-03 | 93 | 2024-07-26 | 84 |
| T27 | 2023-08-05 | 95 | 2024-07-28 | 86 |
| T28 | 2023-08-07 | 97 | 2024-07-30 | 88 |
| T29 | 2023-08-09 | 99 | 2024-08-01 | 90 |
| T30 | 2023-08-11 | 101 | 2024-08-03 | 92 |
| T31 | 2023-08-13 | 103 | 2024-08-05 | 94 |
| T32 | 2023-08-15 | 105 | 2024-08-07 | 96 |
| T33 | 2023-08-17 | 107 | 2024-08-09 | 98 |
| T34 | 2023-08-19 | 109 | 2024-08-11 | 100 |
| T35 | 2023-08-21 | 111 | 2024-08-13 | 102 |
| T36 | 2023-08-25 | 115 | 2024-08-15 | 104 |
| T37 | 2023-08-27 | 117 | 2024-08-17 | 106 |
| T38 | 2023-08-29 | 119 | 2024-08-19 | 108 |
| T39 | 2023-08-31 | 121 | 2024-08-21 | 110 |
| T40 | 2023-09-02 | 123 | 2024-08-23 | 112 |
| T41 | 2023-09-06 | 127 | 2024-08-25 | 114 |
| T42 | 2023-09-10 | 131 | 2024-08-27 | 116 |
| T43 | 2023-09-12 | 133 | 2024-08-29 | 118 |
| T44 | 2023-09-14 | 135 | 2024-08-31 | 120 |
| T45 | 2023-09-16 | 137 | 2024-09-02 | 122 |
| T46 | 2023-09-18 | 139 | 2024-09-04 | 124 |
| T47 | 2023-09-20 | 141 | 2024-09-06 | 126 |
| T48 | 2023-09-22 | 143 | 2024-09-08 | 128 |
| T49 | 2023-09-24 | 145 | 2024-09-10 | 130 |
| T50 | 2023-09-26 | 147 | 2024-09-12 | 132 |
| T51 | 2023-09-28 | 149 | 2024-09-14 | 134 |
| T52 | 2023-09-30 | 151 | 2024-09-16 | 136 |
| T53 | 2023-10-02 | 153 | 2024-09-18 | 138 |
| T54 | 2023-10-04 | 155 | 2024-09-20 | 140 |
| T55 | 2023-10-06 | 157 | 2024-09-22 | 142 |
| T56 | 2023-10-08 | 159 | 2024-09-24 | 144 |
| T57 | 2023-10-10 | 161 | 2024-09-26 | 146 |
| T58 | 2023-10-12 | 163 | 2024-09-28 | 148 |
| T59 | 2023-10-14 | 165 | 2024-09-30 | 150 |
| T60 | 2023-10-16 | 167 | 2024-10-02 | 152 |
| T61 | 2023-10-18 | 169 | 2024-10-04 | 154 |

## Stages used in the analysis

Stage 1 = T1-T25, Stage 2 = T26-T39, Stage 3 = T40-T61 (`Classification/Common/config.m`).

## Missing images

- 2023, T44: sample 3 of Samcheonbyeo (240° only).
- 2024, T52: sample 4 of all 97 cultivars, and sample 3 of 9 cultivars
  (Sodami, Sukwang, Huimangchan, Saeshin, Youngjin, Aromi, Yechan, Chilbo, Unbongbyeo); all three angles.

The index values at these time points were replaced by the mean values of the other plants of the same cultivar on the same date.

## Corrected images

Images corrected by histogram matching, and the reference image of each, are listed in `reference_log_<year>_<angle>.csv` written by `Image_preprocess/preprocess.py`.
