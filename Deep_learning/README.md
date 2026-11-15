# Deep_learning

A shallow convolutional neural network (CNN) trained directly on the segmented image sequences,
evaluated with the same Single-, Mixed-, and Cross-year splits as the MATLAB analysis.

## Input

Segmented images from `Image_preprocess/preprocess.py`, in `Images/segmented/<year>/<angle>/<group>/<plant>/*.png` (`<angle>` = `SV000`, `SV120`, `SV240`).
The full image set is not included (see the top-level `README.md`), so these scripts cannot be run with the example images; they are the scripts used to obtain the CNN results in the Supplementary Material (Table S14).

## Steps

Run from this folder:

```
python inventory.py --root ../Images
python pack.py --root ../Images
python train_cnn.py
```

| Script | Content | Output |
|---|---|---|
| `inventory.py` | matches each image folder to its row in `Preprocessed_dataset` and aligns the acquisition dates of each year to T1–T61 | `data/inventory_<year>.csv` (cultivar, sample number, missing time points per angle), `data/tgrid_<year>.csv` (date of each time point) |
| `pack.py` | resizes the images (128 × 128 pixels) and stores them aligned to T1–T61 | `data/images_<year>_128.npy`, `data/mask_<year>_128.npy`, `data/meta_<year>_128.csv` |
| `train_cnn.py` | trains and evaluates the CNN (Single-, Mixed-, and Cross-year, three seeds) | `results/runs/*.npz`, `results/cnn_<mode>_summary.csv`, `results/cnn_<mode>_runs.csv` |

`dates.py` (date alignment) and `cnn_common.py` (data loading and model) are used by the scripts above.

## Model and training

- Input: the three views at each time point stacked into a 9-channel image (64 × 64 pixels), 61 time points.
  A missing image is replaced by the image of the same plant and view at the nearest available time point.
- Network: four convolution blocks (32, 64, 96, 128 channels) applied to every time point,
  a one-dimensional convolution over time, averaging over time, and a linear layer with 97 outputs (about 2.8 × 10⁵ parameters).
- Training: AdamW (weight decay 10⁻⁴), cosine learning-rate schedule, batch of 16 samples, label smoothing 0.1, random horizontal flips.
- The learning rate (3 × 10⁻⁴, 10⁻³, 3 × 10⁻³) and the number of epochs (1–100) are selected by inner cross-validation within the training samples;
  the test samples are used once, after training.

## Requirements

Python 3.10 or later, `numpy`, `pandas`, `opencv-python`, `torch`.

Tested with Python 3.12.6, PyTorch 2.11.0 (CUDA 12.8), and an NVIDIA GeForce RTX 5080 GPU.
