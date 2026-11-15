# Image_preprocess

Illumination correction and segmentation of the original side-view images.
The segmented images are the input of `Indices/` and `Deep_learning/`.

## Usage

```
python preprocess.py --year 2023 --angle SV000 --input <folder of original images> --output <output folder>
```

Run once for each year (2023, 2024) and viewing angle (SV000, SV120, SV240).
The segmented images are written to `<output folder>/segmented/<year>/<angle>/`.

## Steps

1. Illumination correction: an image is corrected when the median color of a background region differs from the reference by more than 10 in any channel.
   It is histogram-matched to an uncorrected image of the same plant within ±3 days;
   if there is none, of another plant of the same cultivar within ±3 days;
   and otherwise of one fixed plant.
2. Segmentation: background color normalization, cropping, removal of the blue background, and removal of isolated pixels.

`cultivars_2023.csv` and `cultivars_2024.csv` give the cultivar of each plant (`Plant`) for the 97 cultivars used in the analysis.

## Requirements

Python 3.10 or later, `numpy`, `opencv-python`, `pandas`, `Pillow`, `scikit-image`.

## Check with the example images

`reproduce_example.py` preprocesses the raw example images again and compares them pixel by pixel with the released preprocessed images:

```
cd Image_preprocess
python reproduce_example.py
```

It reads `Images/example/raw/<year>/angle0` (original file names) and `Images/example/preprocessed/<year>/angle0`, matches the images by date, and prints the number of identical images for each year (details in `reproduce_example.csv`).
An image that needs illumination correction is matched to an uncorrected image of the same plant within ±3 days in the same folder, as in `preprocess.py`.
