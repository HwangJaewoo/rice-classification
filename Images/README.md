Place the images here (not included in the repository):

    Images/segmented/<year>/<angle>/...               segmented images of all plants (output of Image_preprocess/preprocess.py; input of Deep_learning/)
    Images/example/raw/<year>/angle0/                 raw images of the example plant (61 per folder)
    Images/example/preprocessed/<year>/angle0/        segmented images of the example plant (61 per folder; input of Indices/)

The example images (raw and preprocessed) of Samcheonbyeo, sample 1, in 2023 and 2024 from the 0° view (61 images per year and type) are archived at https://doi.org/10.5281/zenodo.24070543 as `raw.zip` and `preprocessed.zip`; unzip them into `Images/example/raw/` and `Images/example/preprocessed/`.
License: CC BY 4.0.
The full set of original images (388 plants per year, three viewing angles, 61 time points) is available from the corresponding author upon reasonable request.

File names:
- raw images: the original names of the imaging system, e.g. `2023_RDA_RICE_A_01_02_VIS_SV000_<YYYY-MM-DD> <hh_mm_ss>.png`;
- preprocessed images: `<cultivar>_sample<sample>_<year>_angle<angle>_T<tt>_<YYYY-MM-DD>.png`, e.g. `Samcheonbyeo_sample1_2023_angle0_T01_2023-06-14.png` (T01–T61 in the order of the acquisition dates).

The SHA-256 checksum of every image is listed in `checksums.txt` of the same record; put it in `Images/example/` after unzipping and run `sha256sum -c checksums.txt` there.

To check the preprocessing, run `python Image_preprocess/reproduce_example.py` from the repository root; it preprocesses `example/raw/` again and compares the result pixel by pixel with `example/preprocessed/` (see `Image_preprocess/README.md`).
