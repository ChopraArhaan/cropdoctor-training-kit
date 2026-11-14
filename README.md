# CropDoctor: model training kit

Self-contained. Copy this whole `training/` folder to the training machine. It needs nothing from the app.

Pipeline: **prepare data → (bake-off) → train → evaluate → export → copy 3 files into the app.**

## 0. What is different from a plain "fine-tune and report accuracy" pipeline

| Problem | What this kit does |
|---|---|
| Lab photos (plain backgrounds) look nothing like a phone photo in a field | Real field photos are kept first, sampled 2x in training, and over-represented in val/test. Accuracy is reported **separately on field photos**. |
| Copies/augmented duplicates leak into validation and inflate accuracy | Images are grouped by an orientation-invariant perceptual hash; a whole group goes to ONE split. |
| Softmax is over-confident | Temperature scaling on the validation set. ECE is reported before/after. The temperature is baked into the exported model, so the app's percentages are honest. |
| Model guesses a crop for a keyboard or a wall | A `Background___not_a_leaf` class from your own negatives + textures (DTD) + pets/objects. |
| "Which backbone?" is a guess | `bakeoff.py` trains all candidates and ranks them on field accuracy. |
| The `.tflite` silently differs from PyTorch | `export_litert.py` checks top-1 agreement on real images and warns. |

## 1. Setup

Python 3.10 or 3.11, an NVIDIA GPU with 8 GB+ VRAM is plenty (a 12 GB card trains any candidate here).

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/Mac:  source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121   # match your CUDA, see pytorch.org
pip install -r requirements-train.txt
python -c "import torch; print(torch.cuda.is_available())"   # must print True
```

The export step uses a **separate** environment (see `requirements-export.txt`) because `ai-edge-torch` pins its
own torch version. On Windows it may not install; use WSL2, Linux, or Colab for step 5 only.

## 2. Datasets

`config.yaml → sources` lists them. Kaggle sources need credentials once:

```bash
# https://www.kaggle.com/settings -> Create New Token, then:
export KAGGLE_USERNAME=...   export KAGGLE_KEY=...        # Windows PowerShell: $env:KAGGLE_USERNAME="..."
```

If a download fails, put the extracted dataset in `data/raw/<source name>/` yourself; the script uses it as-is.

Strongly recommended, because it is the biggest accuracy lever:
* **Your own photos.** Put labelled phone photos in `data/my_photos/<class folder>/*.jpg` (folder names such as
  `Tomato late blight` are understood). Even 30 per class of real field photos helps a lot.
* **Your own negatives.** 100+ photos of things that are *not* leaves (soil, hands, walls, sacks, tools, screens,
  other plants) in `data/negatives/`.

## 3. Prepare data

```bash
python taxonomy.py                 # lists every class the taxonomy knows about
python prepare_data.py             # downloads, maps, de-duplicates, splits  -> data/manifest.csv
```

Read `data/unmapped.txt`. It lists dataset folders that did not match the taxonomy. If one is a class you care
about, add a rule in `taxonomy.py` (and a matching entry in the app's `knowledge_base.json`).

What `prepare_data.py` guards against, so the numbers you report are honest:

* **Field vs lab flag.** `field: true` is only for real phone/field photos. A big lab-style collection marked
  `field: true` makes every class look like "field" and the field accuracy meaningless. Check the printed
  per-class lines and `is_field` in `data/manifest.csv`.
* **Source balance.** `data.max_per_source_per_class` stops one huge dataset from filling a class on its own.
* **One-sided crops are dropped** (`[drop crop]` in the output): a crop with only a healthy class would call a
  diseased leaf healthy; a crop with only disease classes would call a healthy leaf diseased. Add the missing side
  to keep the crop.
* **Local datasets.** `kind: local` paths may use `~` and `$ENV_VARS`; a missing folder is reported, not ignored.

## 4. Choose a backbone and train

**Short on time?** Skip the bake-off and run the time-boxed training (about 15 to 30 minutes on a mid-range GPU):

```bash
python train.py --model mobilenetv4_conv_medium.e500_r256_in1k --fast
python evaluate.py --run runs/mobilenetv4_conv_medium.e500_r256_in1k
```

`--fast` = 1 head epoch + 5 full epochs of 12,000 images each, validated on a 3,000-image subset. Watch the
`~N min left` estimate printed every epoch. If it is far above your budget, stop and lower
`--samples-per-epoch`. If you have more time, raise it, for example
`--samples-per-epoch 30000 --epochs-full 8`. Useful flags: `--workers 4` (or `0` if Windows data-loader workers
crash), `--epochs-head`, `--epochs-full`, `--val-subset`.

If it prints `[warn] CUDA is not available`, stop: CPU training is far too slow. Reinstall the CUDA build of PyTorch.


```bash
python bakeoff.py                  # quick screening of every candidate (~5-10 min each), writes runs/bakeoff.md
python bakeoff.py --only mobilenetv4_conv_medium.e500_r256_in1k convnextv2_nano.fcmae_ft_in22k_in1k   # just two
python train.py --model mobilenetv4_conv_medium.e500_r256_in1k      # full run of the winner
python evaluate.py --run runs/mobilenetv4_conv_medium.e500_r256_in1k
python evaluate.py --run runs/<name> --folder data/my_test_photos   # photos the model never saw, ideal sanity check
```

Pick on **field accuracy + macro-F1** first. If two models are close, take the smaller/faster one; the app's
Settings → *Benchmark model* measures real phone latency (CPU vs GPU delegate).

Rough expectations (not promises): high-90s on lab photos; noticeably lower on field photos. That gap is real. Quote
the **field** number to judges, with the test-set size.

## 5. Export for Android

In the export environment:

```bash
python export_litert.py --run runs/<name>            # float32
python export_litert.py --run runs/<name> --fp16     # smaller; if the recipe is unavailable it falls back to fp32 and says so
```

It prints a parity line like `top-1 agreement 40/40`. Do not ship if it warns.

## 6. Hand over to the app

Copy these three files from `export/<name>/` into `CropDoctor/app/src/main/assets/`:

```
model.tflite   labels.txt   model_meta.json
```

Rebuild the app. Nothing else changes. The app reads the class list, input size, temperature and thresholds from
those files. If you added classes to `taxonomy.py`, add them to `app/src/main/assets/knowledge_base.json` too
(otherwise the app shows a safe "consult your KVK" message for them).

## Files

| File | Purpose |
|---|---|
| `config.yaml` | all settings |
| `taxonomy.py` | canonical classes + folder-name matching rules |
| `prepare_data.py` | download, map, dedup, leak-free split, manifest |
| `common.py` | dataset, transforms (center-square crop matches the app), metrics, calibration |
| `train.py` | two-stage fine-tune with EMA, mixup/cutmix, class-balanced field-weighted sampling |
| `bakeoff.py` | compare candidates |
| `evaluate.py` | held-out test or your own photo folder |
| `export_litert.py` | PyTorch → LiteRT with baked-in preprocessing, calibration, parity check |

## Notes and honest caveats

* `mobilenetv4_*` and the other timm names in `config.yaml` need `timm>=1.0.9`. If a name is not found, run
  `python -c "import timm; print(timm.list_models('mobilenetv4*', pretrained=True))"`.
* Not every op is supported by every phone GPU. If the app's benchmark shows the GPU delegate failing or slower than
  CPU, it falls back to CPU automatically; consider the smaller candidate.
* Desktop CPU latency printed by training is only for ranking. Phone numbers come from the app.
