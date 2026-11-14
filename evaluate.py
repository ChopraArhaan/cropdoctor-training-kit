"""
Evaluate a trained run on the held-out TEST split, or on any folder of your own phone photos.

    python evaluate.py --run runs/mobilenetv4_conv_medium.e500_r256_in1k
    python evaluate.py --run runs/<name> --folder data/my_test_photos     # sub-folders named like the classes

Prints accuracy, per-class recall (worst first) and the most confused pairs.
"""
import argparse
import json
from pathlib import Path

import timm
import torch
from torch.utils.data import DataLoader

from common import ManifestDataset, load_cfg, make_transforms, predict_logits, read_manifest, summarize
from taxonomy import canonical

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--config", default="config.yaml")
ap.add_argument("--folder")
a = ap.parse_args()

cfg = load_cfg(a.config)
ck = torch.load(Path(a.run) / "best.pt", map_location="cpu")
classes = ck["classes"]
c2i = {c: i for i, c in enumerate(classes)}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = timm.create_model(ck["model"], pretrained=False, num_classes=len(classes))
model.load_state_dict(ck["state_dict"])
model.to(device).eval()

if a.folder:
    rows = []
    for d in Path(a.folder).iterdir():
        lab = canonical(d.name) or d.name
        if lab not in c2i:
            print("skip folder (not a known class):", d.name)
            continue
        rows += [{"path": str(p), "label": lab, "is_field": True} for p in d.rglob("*")
                 if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
else:
    rows = [r for r in read_manifest(cfg["paths"]["manifest"]) if r["split"] == "test"]

dl = DataLoader(ManifestDataset(rows, c2i, make_transforms(ck["img_size"], ck["mean"], ck["std"], False)),
                batch_size=64, num_workers=4)
logits, labels = predict_logits(model, dl, device)
m = summarize(logits, labels, [r["is_field"] for r in rows], classes, ck["temperature"])
print(json.dumps({k: v for k, v in m.items() if k not in ("per_class", "most_confused")}, indent=2))
worst = sorted(((v["recall"], k, v["n"]) for k, v in m["per_class"].items() if v["recall"] is not None))[:12]
print("\nweakest classes (recall, class, n):")
for r, k, n in worst:
    print(f"  {r:.2f}  {k}  (n={n})")
print("\nmost confused:")
for c in m["most_confused"][:10]:
    print(f"  {c['n']:>3}  {c['true']}  ->  {c['pred']}")
