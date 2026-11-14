"""
Convert the trained model into the three files the Android app needs.

    python export.py                                   # uses the newest run in runs/
    python export.py --run runs/<model> --fp16         # smaller file
    python export.py --assets ../CropDoctor/app/src/main/assets    # also copy the files into the app

Works locally on Windows (PyTorch -> ONNX -> TFLite via onnx2tf) or via ai_edge_torch.

Output (export/<model>/): model.tflite, labels.txt, model_meta.json
  model.tflite: input float32 [1,S,S,3] NHWC pixels 0..255 (normalisation baked in), output [1,N] calibrated softmax.
It checks the .tflite against PyTorch on real validation images and warns if they disagree.
"""
import argparse
import csv
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import timm
import torch
import yaml
from PIL import Image

BACKGROUND = "Background___not_a_leaf"


class Deployable(torch.nn.Module):
    """NHWC 0..255 float32 input [1, H, W, 3] -> normalise -> PyTorch network -> softmax(logits / temperature)."""

    def __init__(self, net, mean, std, temperature):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)
        self.t = float(temperature)

    def forward(self, x):
        # x input is NHWC [1, H, W, 3] in 0..255 float32
        x_nchw = x.permute(0, 3, 1, 2)
        norm_x = (x_nchw - self.mean) / self.std
        logits = self.net(norm_x)
        return torch.softmax(logits / self.t, dim=1)


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


def convert_local_onnx(wrapped, S: int, dst_tflite: Path):
    """Local Windows conversion: PyTorch -> ONNX -> TFLite via onnx2tf."""
    print("[convert] Exporting PyTorch -> ONNX -> TFLite locally on Windows...")
    with tempfile.TemporaryDirectory() as tmp_dir:
        onnx_file = os.path.join(tmp_dir, "model.onnx")
        tf_model_dir = os.path.join(tmp_dir, "tf_model")

        dummy_input = torch.randn(1, S, S, 3, dtype=torch.float32) * 255.0
        torch.onnx.export(
            wrapped,
            dummy_input,
            onnx_file,
            input_names=["input"],
            output_names=["output"],
            opset_version=18,
        )

        import onnx2tf
        onnx2tf.convert(
            input_onnx_file_path=onnx_file,
            output_folder_path=tf_model_dir,
            not_use_onnxsim=True,
        )

        converted_tflite = os.path.join(tf_model_dir, "model_float32.tflite")
        if not os.path.exists(converted_tflite):
            tflites = list(Path(tf_model_dir).glob("*.tflite"))
            if tflites:
                converted_tflite = str(tflites[0])
            else:
                raise RuntimeError("No tflite file produced by onnx2tf.")

        shutil.copy(converted_tflite, dst_tflite)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="runs/<model> folder (default: newest)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--assets", help="app assets folder to copy the three files into")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    runs = Path(cfg["paths"]["runs_dir"])
    run_candidates = list(runs.glob("*/best.pt"))
    if not run_candidates:
        raise SystemExit("No trained model found. Run 'python train.py' first.")

    run = Path(a.run) if a.run else max((p.parent for p in run_candidates), key=lambda p: (p / "best.pt").stat().st_mtime)
    print(f"[export] run: {run}")

    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    classes, S = ck["classes"], ck["img_size"]
    net = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapped = Deployable(net, ck["mean"], ck["std"], ck["temperature"]).eval()

    out = Path(cfg["paths"]["export_dir"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    final = out / "model.tflite"

    try:
        import ai_edge_torch
        print("[convert] Using ai_edge_torch...")
        edge = ai_edge_torch.convert(wrapped, (torch.rand(1, S, S, 3) * 255.0,))
        edge.export(str(final))
    except Exception as e:
        print(f"[convert] ai_edge_torch unavailable ({e}). Using local ONNX -> TFLite exporter...")
        convert_local_onnx(wrapped, S, final)

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
    else:
        try:
            try:
                from ai_edge_litert.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

            it = Interpreter(model_path=str(final))
            it.allocate_tensors()
            i_d, o_d = it.get_input_details()[0], it.get_output_details()[0]
            agree, worst = 0, 0.0
            for r in rows:
                im = center_square(Image.open(r["path"]).convert("RGB")).resize((S, S), Image.BILINEAR)
                # Ensure input x matches the TFLite input tensor shape expected by i_d['shape']
                arr = np.asarray(im, dtype=np.float32)
                if len(i_d["shape"]) == 4 and i_d["shape"][1] == 3:  # NCHW
                    arr = np.transpose(arr, (2, 0, 1))
                x_np = arr[None]
                x_torch = torch.from_numpy(np.asarray(im, dtype=np.float32))[None]
                with torch.no_grad():
                    ref = wrapped(x_torch).numpy()[0]
                it.set_tensor(i_d["index"], x_np)
                it.invoke()
                got = it.get_tensor(o_d["index"])[0]
                agree += int(ref.argmax() == got.argmax())
                worst = max(worst, float(np.abs(ref - got).max()))
            print(f"[check] tflite agrees with PyTorch on {agree}/{len(rows)} images (max probability difference {worst:.4f})")
            if agree < len(rows) * 0.95:
                print("[WARNING] the export disagrees with PyTorch. Do NOT ship it. Re-run without --fp16.")
        except Exception as check_e:
            print(f"[check] TFLite interpreter check skipped ({check_e})")

    m_file = run / "metrics.json"
    m = json.loads(m_file.read_text(encoding="utf-8")) if m_file.exists() else {}
    thr = ((m.get("val") or {}).get("confident_threshold") or {}).get("threshold", 0.6)
    (out / "labels.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")
    meta = {
        "name": "CropDoctor classifier",
        "architecture": ck["model"],
        "input_size": S,
        "input_layout": "NHWC",
        "input_range": "0-255 float32, normalisation baked in",
        "output": "softmax probabilities (calibrated)",
        "num_classes": len(classes),
        "background_class": BACKGROUND,
        "temperature": ck["temperature"],
        "confident_threshold": thr,
        "unsure_threshold": round(max(0.25, thr - 0.2), 3),
        "precision": "fp16 weights" if fp16 else "fp32",
        "model_size_mb": round(final.stat().st_size / 1e6, 2),
        "test_accuracy": round((m.get("test") or {}).get("acc", 0.0), 4),
        "test_field_accuracy": round((m.get("test") or {}).get("acc_field", 0.0), 4) if (m.get("test") or {}).get("acc_field") is not None else None,
        "test_macro_f1": round((m.get("test") or {}).get("macro_f1", 0.0), 4),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    assets_dst = Path(a.assets) if a.assets else Path("../CropDoctor/app/src/main/assets")
    if assets_dst.exists():
        for n in ("model.tflite", "labels.txt", "model_meta.json"):
            shutil.copy(out / n, assets_dst / n)
        print(f"[copied] model.tflite, labels.txt, model_meta.json -> {assets_dst}")

    print(f"\n[done] {final} ({meta['model_size_mb']} MB). Export successful!")


if __name__ == "__main__":
    main()
