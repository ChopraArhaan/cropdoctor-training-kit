# CropDoctor: model training kit

Trains the crop-disease classifier and exports the three files the Android app needs.
Four steps, four commands.

| File | What it does |
|---|---|
| `prepare_data.py` | finds the datasets, maps folder names to class ids, removes duplicates, makes a leak-free split → `data/manifest.csv` |
| `train.py` | fine-tunes a pretrained backbone, calibrates it, prints the test report |
| `export.py` | converts to `model.tflite` + `labels.txt` + `model_meta.json` |
| `taxonomy.py`, `config.yaml` | class list and settings |

## 1. Setup (once)

Python 3.10 or 3.11 and an NVIDIA GPU.

```bash
python -m venv .venv
.venv\Scripts\activate                 # Linux/Mac: source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA version
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"     # must print True
```

## 2. Prepare data

Put datasets in `data/raw/<name>/` (or let Kaggle / Hugging Face download them, see `sources` in `config.yaml`).
Your own phone photos go in `data/my_photos/<Crop disease>/` and non-leaf photos in `data/negatives/`.

```bash
python prepare_data.py
```

Check the output: `[drop crop]` lines are crops removed because they only have healthy or only disease classes
(they would give wrong answers). `data/unmapped.txt` lists folders the taxonomy did not recognise.
`field: true` in `config.yaml` is only for real phone/field photos, not plain-background lab photos.

## 3. Train

```bash
python train.py
```

About 15 to 30 minutes on a mid-range GPU. It prints `~N min left` after every epoch and ends with the test report.
**Quote the FIELD accuracy** (with the number of field images) to judges: it is the honest number.

Want a better model and have time? `python train.py --epochs 8 --samples 20000`.
Another backbone: `python train.py --model convnextv2_nano.fcmae_ft_in22k_in1k`.
Windows data-loader crash? `python train.py --workers 0`.

It fine-tunes a pretrained ImageNet backbone, so there is no need to train from zero.

## 4. Export for the app

On Windows, use `export_tflite.py` with `onnx2tf` installed. On Linux or Google Colab, use `export.py` with `ai-edge-torch`.
For a remote export, copy `export.py`, `config.yaml`, `runs/<model>/best.pt` and `runs/<model>/metrics.json`
(the manifest and images are optional; the parity check is skipped automatically if the images are missing).

```bash
# Windows
python -m pip install onnx2tf
python export_tflite.py

# Linux / Google Colab
pip install ai-edge-torch ai-edge-litert ai-edge-quantizer
python export.py                       # add --fp16 for a smaller file
```

It checks the converted model against PyTorch and prints `tflite agrees with PyTorch on N/N images`.
If it warns, do not ship that file.

Copy `model.tflite`, `labels.txt` and `model_meta.json` from `export/<model>/` into
`CropDoctor/app/src/main/assets/` (or pass `--assets <that folder>`), then rebuild the app.

## Notes

- Classes in `taxonomy.py` must have matching entries in the app's `knowledge_base.json`; otherwise the app shows a
  safe "consult your KVK" message for them.
- Photos are centre-cropped to a square and resized, in training and in the app. Do not change one without the other.
- Public datasets are mostly lab photos. Even 30 of your own field photos per class help a lot.
