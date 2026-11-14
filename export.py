"""
Convert the trained model into the three files the Android app needs.

    python export.py                                   # uses the newest run in runs/
    python export.py --run runs/<model> --fp16         # smaller file
    python export.py --assets ../CropDoctor/app/src/main/assets    # also copy the files into the app

Run it in an environment that has ai-edge-torch (see README). That package is Linux-first: on Windows use WSL2
or Google Colab for this one step. Everything it needs is in runs/<model>/best.pt.

Output (export/<model>/): model.tflite, labels.txt, model_meta.json
  model.tflite: input float32 [1,S,S,3] NHWC pixels 0..255 (normalisation baked in), output [1,N] calibrated softmax.
It checks the .tflite against PyTorch on real validation images and warns if they disagree.
"""
import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from PIL import Image

BACKGROUND = "Background___not_a_leaf"


class Deployable(torch.nn.Module):
    """NHWC 0..255 input -> normalise -> network -> softmax(logits / temperature)."""

    def __init__(self, net, mean, std, temperature):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)
        self.t = float(temperature)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        return torch.softmax(self.net((x - self.mean) / self.std) / self.t, dim=1)


def center_square(im):
    w, h = im.size
    s = min(w, h)
    left, top = (w - s) // 2, (h - s) // 2
    return im.crop((left, top, left + s, top + s))


def try_fp16(src: Path, dst: Path) -> bool:
    """Weight-only fp16 with ai-edge-quantizer; falls back to float32 if the recipe is unavailable."""
    try:
        from ai_edge_quantizer import quantizer, recipe
        names = [n for n in dir(recipe) if "fp16" in n.lower() and not n.startswith("_")]
        if not names:
            return False
        qt = quantizer.Quantizer(str(src))
        qt.load_quantization_recipe(getattr(recipe, names[0])())
        qt.quantize().export_model(str(dst), overwrite=True)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[fp16] skipped ({e}); keeping float32")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="runs/<model> folder (default: newest)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--assets", help="app assets folder to copy the three files into")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    runs = Path(cfg["paths"]["runs_dir"])
    run = Path(a.run) if a.run else max((p.parent for p in runs.glob("*/best.pt")), key=lambda p: (p / "best.pt").stat().st_mtime)
    print(f"[export] run: {run}")

    try:
        import ai_edge_torch
        from ai_edge_litert.interpreter import Interpreter
    except ImportError as e:
        raise SystemExit(f"{e}\nInstall the export tools first:  pip install ai-edge-torch ai-edge-litert ai-edge-quantizer\n"
                         "On Windows this usually needs WSL2 or Google Colab (see README).")

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    classes, S = ck["classes"], ck["img_size"]
    net = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapped = Deployable(net, ck["mean"], ck["std"], ck["temperature"]).eval()

    out = Path(cfg["paths"]["export_dir"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    print("[convert] this can take a few minutes...")
    edge = ai_edge_torch.convert(wrapped, (torch.rand(1, S, S, 3) * 255.0,))
    final = out / "model.tflite"
    edge.export(str(final))
    fp16 = False
    if a.fp16:
        fp16 = try_fp16(final, out / "model_fp16.tflite")
        if fp16:
            shutil.copy(out / "model_fp16.tflite", final)

    # parity check on real validation images
    rows = []
    if Path(cfg["paths"]["manifest"]).exists():
        with open(cfg["paths"]["manifest"], newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r["split"] == "val" and Path(r["path"]).exists()]
    rows = rows[:: max(1, len(rows) // 40)][:40]
    if not rows:
        print("[check] skipped: the validation images listed in the manifest are not on this machine")
    it = Interpreter(model_path=str(final))
    it.allocate_tensors()
    i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
    agree, worst = 0, 0.0
    for r in rows:
        im = center_square(Image.open(r["path"]).convert("RGB")).resize((S, S), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32))[None]
        with torch.no_grad():
            ref = wrapped(x).numpy()[0]
        it.set_tensor(i_d["index"], x.numpy())
        it.invoke()
        got = it.get_tensor(o_d["index"])[0]
        agree += int(ref.argmax() == got.argmax())
        worst = max(worst, float(np.abs(ref - got).max()))
    if rows:
        print(f"[check] tflite agrees with PyTorch on {agree}/{len(rows)} images (max probability difference {worst:.4f})")
    if rows and agree < len(rows) * 0.95:
        print("[WARNING] the export disagrees with PyTorch. Do NOT ship it. Re-run without --fp16.")

    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    thr = (m["val"].get("confident_threshold") or {}).get("threshold", 0.6)
    (out / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    meta = {
        "name": "CropDoctor classifier", "architecture": ck["model"], "input_size": S, "input_layout": "NHWC",
        "input_range": "0-255 float32, normalisation baked in", "output": "softmax probabilities (calibrated)",
        "num_classes": len(classes), "background_class": BACKGROUND, "temperature": ck["temperature"],
        "confident_threshold": thr, "unsure_threshold": round(max(0.25, thr - 0.2), 3),
        "precision": "fp16 weights" if fp16 else "fp32", "model_size_mb": round(final.stat().st_size / 1e6, 2),
        "test_accuracy": round(m["test"]["acc"], 4),
        "test_field_accuracy": None if m["test"]["acc_field"] is None else round(m["test"]["acc_field"], 4),
        "test_macro_f1": round(m["test"]["macro_f1"], 4), "test_images": m["test"]["n"],
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if a.assets:
        dst = Path(a.assets)
        dst.mkdir(parents=True, exist_ok=True)
        for n in ("model.tflite", "labels.txt", "model_meta.json"):
            shutil.copy(out / n, dst / n)
        print(f"[copied] model.tflite, labels.txt, model_meta.json -> {dst}")
    print(f"\n[done] {final} ({meta['model_size_mb']} MB). Copy the 3 files in {out} into CropDoctor/app/src/main/assets/")


if __name__ == "__main__":
    main()
