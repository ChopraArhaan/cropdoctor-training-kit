"""
Windows-compatible TFLite exporter for CropDoctor PyTorch checkpoints (best.pt -> model.tflite).

Exports:
  1. model.tflite
  2. labels.txt
  3. model_meta.json
"""
import argparse
import json
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import timm
import torch
import yaml

BACKGROUND = "Background___not_a_leaf"


class Deployable(torch.nn.Module):
    """NHWC 0..255 float32 input -> normalise -> PyTorch network -> softmax(logits / temperature)."""

    def __init__(self, net, mean, std, temperature):
        super().__init__()
        self.net = net
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1) * 255.0)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1) * 255.0)
        self.t = float(temperature)

    def forward(self, x):
        # input x is [B, H, W, C] in 0..255 float32
        x = x.permute(0, 3, 1, 2)
        logits = self.net((x - self.mean) / self.std)
        return torch.softmax(logits / self.t, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="runs/<model> folder")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--assets", help="app assets folder to copy files to")
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    runs = Path(cfg["paths"]["runs_dir"])
    if a.run:
        run = Path(a.run)
    else:
        run_candidates = list(runs.glob("*/best.pt"))
        if not run_candidates:
            raise SystemExit("No best.pt checkpoint found in runs/.")
        run = max((p.parent for p in run_candidates), key=lambda p: (p / "best.pt").stat().st_mtime)

    print(f"[export] Selected run: {run}")
    ck = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
    classes, S = ck["classes"], ck["img_size"]
    print(f"[model] {ck['model']} | {len(classes)} classes | img_size: {S}")

    net = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
    net.load_state_dict(ck["state_dict"])
    net.eval()
    wrapped = Deployable(net, ck["mean"], ck["std"], ck["temperature"]).eval()

    out = Path(cfg["paths"]["export_dir"]) / run.name
    out.mkdir(parents=True, exist_ok=True)
    tflite_path = out / "model.tflite"

    print("[convert] Exporting PyTorch -> ONNX -> TFLite...")
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
            opset_version=14,
        )
        print("  + ONNX model exported successfully.")

        try:
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
            shutil.copy(converted_tflite, tflite_path)
            print(f"  + TFLite model exported via onnx2tf to: {tflite_path}")
        except Exception as e:
            print(f"  - onnx2tf conversion failed: {e}. Falling back to TensorFlow SavedModel converter...")
            import tensorflow as tf
            import onnx
            from onnx_tf.backend import prepare
            onnx_model = onnx.load(onnx_file)
            tf_rep = prepare(onnx_model)
            tf_rep.export_graph(tf_model_dir)
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_dir)
            tflite_model = converter.convert()
            tflite_path.write_bytes(tflite_model)
            print(f"  + TFLite model exported via TF SavedModel to: {tflite_path}")

    # Write labels.txt and model_meta.json
    labels_content = "\n".join(classes) + "\n"
    (out / "labels.txt").write_text(labels_content, encoding="utf-8")

    m_path = run / "metrics.json"
    m = json.loads(m_path.read_text(encoding="utf-8")) if m_path.exists() else {}
    thr = ((m.get("val") or {}).get("confident_threshold") or {}).get("threshold", 0.6)

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
        "precision": "fp32",
        "model_size_mb": round(tflite_path.stat().st_size / 1e6, 2),
        "test_accuracy": round((m.get("test") or {}).get("acc", 0.0), 4),
        "test_field_accuracy": round((m.get("test") or {}).get("acc_field", 0.0), 4) if (m.get("test") or {}).get("acc_field") is not None else None,
        "test_macro_f1": round((m.get("test") or {}).get("macro_f1", 0.0), 4),
    }
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # Auto-copy assets if requested or if Android app assets folder exists
    assets_dir = Path(a.assets) if a.assets else Path("../CropDoctor/app/src/main/assets")
    if assets_dir.exists():
        for name in ("model.tflite", "labels.txt", "model_meta.json"):
            shutil.copy(out / name, assets_dir / name)
        print(f"[copied] Exported files copied directly to Android assets: {assets_dir}")

    print(f"\n[SUCCESS] TFLite export complete! File: {tflite_path} ({meta['model_size_mb']} MB)")


if __name__ == "__main__":
    main()
