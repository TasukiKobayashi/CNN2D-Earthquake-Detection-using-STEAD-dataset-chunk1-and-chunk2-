# CNN2D — Earthquake Detection (STEAD chunks)

This folder contains scripts to convert STEAD HDF5 waveform data to spectrogram images, visualize dataset statistics, and train a small 2D CNN classifier on the generated spectrogram images.

## Quick overview

- `1.image_creation.py` — generate PNG spectrograms from HDF5 traces (multiprocessing).
- `2.data_visualization.py` — plots for histograms, correlation matrix, categorical distributions, and waveform + spectrogram examples.
- `3.cnn_class.py` — prepares image dataset, trains a small PyTorch CNN for classification, evaluates and saves model & plots.

## Where models are saved

Trained classification models are saved to `./models/classification/classification.pth` (fixed filename). Each run will overwrite that file. If you want unique filenames, edit the save logic in `3.cnn_class.py`.

## Prerequisites

- Python 3.8+
- On Windows use `cmd.exe` for the commands below.

Recommended workflow (cmd.exe):

```bat
python -m venv .venv
.\.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Run steps (basic)

1) Generate spectrogram images (creates `image/` folder):

```bat
python 1.image_creation.py
```

- Edit the top-of-file paths in `1.image_creation.py` if your HDF5/CSV files are stored elsewhere (variables: `dst1_hdf5`, `dst1_csv`, `dst2_hdf5`, `dst2_csv`, `img_save_path`).

2) Explore dataset visualizations:

```bat
python 2.data_visualization.py
```

- The script writes plots to a path configured inside the file (variable: `SAVE_PATH`). Update it if needed.

3) Train the classification CNN and evaluate:

```bat
python 3.cnn_class.py
```

- `3.cnn_class.py` expects the image folder path in the `dir` variable near the bottom of the file. Update it to match the `img_save_path` used by `1.image_creation.py`.
- Training will save the model to `./models/classification/classification.pth` and generate evaluation plots (confusion matrix, accuracy/loss).

## Tips & notes

- Multiprocessing on Windows: `1.image_creation.py` calls `set_start_method('spawn')` — this is correct for Windows.
- If you get memory issues during image generation, reduce the `processes` argument or batch size in `process_chunk()`.
- If you want timestamped model filenames (no overwrite), change the save block in `classification_cnn()` inside `3.cnn_class.py`.

## Files in this folder

- `1.image_creation.py` — spectrogram generation.
- `2.data_visualization.py` — EDA and plotting.
- `3.cnn_class.py` — dataset prep, training, evaluation.
- `requirements.txt` — dependencies (in parent folder).
- `models/` — model checkpoints (created at runtime).

## Dataset

Replace the placeholder below with the actual dataset link you used:

- STEAD or your prepared HDF5/CSV dataset: (https://github.com/smousavi05/STEAD)


