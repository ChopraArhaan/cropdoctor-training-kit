"""
Step 3: export the trained PyTorch model to a LiteRT (.tflite) bundle for the Android app.

    python export_litert.py --run runs/mobilenetv4_conv_medium.e500_r256_in1k            # float32 (safest)
    python export_litert.py --run runs/<name> --fp16                                      # smaller file, GPU friendly

Run this in the EXPORT virtualenv (requirements-export.txt).

Model contract with the app (app/src/main/assets/):
  model.tflite       input  : float32 [1, S, S, 3] NHWC, pixel values 0..255   (normalisation is baked in)
                     output : float32 [1, N] softmax probabilities, already temperature-calibrated
  labels.txt         N lines, class ids in output order  (<Crop>___<disease_id>)
  model_meta.json    input size, temperature, thresholds, accuracy numbers shown in the app's About screen

The script verifies the .tflite against PyTorch on real validation images before it declares success.
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import timm
import torch
from PIL import Image

from common import CenterSquare, load_cfg, read_manifest
from taxonomy import BACKGROUND


class Deployable(torch.nn.Module):
    """NHWC 0..255 float input -> normalise -> network -> softmax(logits / T)."""

    def __init__(self, net, mean, std, temperature):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)
        self.t = float(temperature)

    def forward(self, x):                       # x: [1,S,S,3]
        x = x.permute(0, 3, 1, 2)
        x = (x - self.mean) / self.std
        return torch.softmax(self.net(x) / self.t, dim=1)


def try_fp16(src: Path, dst: Path) -> bool:
    """Weight-only fp16 via ai-edge-quantizer. The recipe name differs between versions, so search for it."""
    try:
        from ai_edge_quantizer import quantizer, recipe
        names = [n for n in dir(recipe) if "fp16" in n.lower() and not n.startswith("_")]
        if not names:
            print("[fp16] no fp16 recipe in this ai-edge-quantizer version; keeping float32")
            return False
        qt = quantizer.Quantizer(str(src))
        qt.load_quantization_recipe(getattr(recipe, names[0])())
        qt.quantize().export_model(str(dst), overwrite=True)
        print(f"[fp16] used recipe {names[0]}")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[fp16] failed ({e}); keeping float32")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--samples", type=int, default=40, help="validation images used for the parity check")
    a = ap.parse_args()

    import ai_edge_torch
    from ai_edge_litert.interpreter import Interpreter

    cfg = load_cfg(a.config)
    run = Path(a.run)
    ck = torch.load(run / "best.pt", map_location="cpu")
    classes, S = ck["classes"], ck["img_size"]
    net = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapped = Deployable(net, ck["mean"], ck["std"], ck["temperature"]).eval()

    out = Path(cfg["paths"]["export_dir"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    sample = torch.rand(1, S, S, 3) * 255.0
    print("[convert] tracing + converting (this can take a few minutes)...")
    edge = ai_edge_torch.convert(wrapped, (sample,))
    f32 = out / "model_fp32.tflite"
    edge.export(str(f32))
    final = out / "model.tflite"
    shutil.copy(f32, final)
    if a.fp16 and try_fp16(f32, out / "model_fp16.tflite"):
        shutil.copy(out / "model_fp16.tflite", final)

    # ---- parity check: real images through PyTorch and through the .tflite
    rows = [r for r in read_manifest(cfg["paths"]["manifest"]) if r["split"] == "val"]
    rows = rows[:: max(1, len(rows) // a.samples)][: a.samples]
    interp = Interpreter(model_path=str(final))
    interp.allocate_tensors()
    i_d, o_d = interp.get_input_details()[0], interp.get_output_details()[0]
    agree, max_diff, cs = 0, 0.0, CenterSquare()
    for r in rows:
        im = cs(Image.open(r["path"]).convert("RGB")).resize((S, S), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(im, dtype=np.float32))[None]
        with torch.no_grad():
            ref = wrapped(x).numpy()[0]
        interp.set_tensor(i_d["index"], x.numpy())
        interp.invoke()
        got = interp.get_tensor(o_d["index"])[0]
        agree += int(ref.argmax() == got.argmax())
        max_diff = max(max_diff, float(np.abs(ref - got).max()))
    print(f"[parity] top-1 agreement {agree}/{len(rows)}, max |prob diff| {max_diff:.4f}")
    if agree < len(rows) * 0.95:
        print("[WARNING] tflite disagrees with PyTorch on more than 5% of images. Do NOT ship this export; try float32.")

    # ---- bundle
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    (out / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    thr = (m["val"].get("confident_threshold") or {}).get("threshold", 0.6)
    meta = {
        "name": "CropDoctor classifier",
        "architecture": ck["model"],
        "input_size": S,
        "input_layout": "NHWC",
        "input_range": "0-255 float32, normalisation baked in",
        "output": "softmax probabilities (temperature calibrated)",
        "num_classes": len(classes),
        "background_class": BACKGROUND,
        "temperature": ck["temperature"],
        "confident_threshold": thr,
        "unsure_threshold": round(max(0.25, thr - 0.2), 3),
        "precision": "fp16 weights" if (a.fp16 and (out / "model_fp16.tflite").exists()) else "fp32",
        "model_size_mb": round(final.stat().st_size / 1e6, 2),
        "test_accuracy": round(m["test"]["acc"], 4),
        "test_field_accuracy": None if m["test"]["acc_field"] is None else round(m["test"]["acc_field"], 4),
        "test_macro_f1": round(m["test"]["macro_f1"], 4),
        "test_images": m["test"]["n"],
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\n[done] {final}  ({meta['model_size_mb']} MB)")
    print("Copy model.tflite, labels.txt and model_meta.json from", out, "into  CropDoctor/app/src/main/assets/")


if __name__ == "__main__":
    main()
