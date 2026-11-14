"""Shared helpers: manifest dataset, transforms, evaluation, calibration."""
import csv
import math
import random
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageFile
from sklearn.metrics import f1_score
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def load_cfg(path="config.yaml"):
    return yaml.safe_load(open(path, encoding="utf-8"))


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_field"] = r["is_field"] == "1"
    return rows


class CenterSquare:
    """Center-crop to a square. The Android app does exactly the same before resizing, so keep them in sync."""

    def __call__(self, im):
        w, h = im.size
        s = min(w, h)
        l, t = (w - s) // 2, (h - s) // 2
        return im.crop((l, t, l + s, t + s))


def make_transforms(img, mean, std, train):
    norm = T.Normalize(mean, std)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img, scale=(0.35, 1.0), ratio=(0.75, 1.33)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.RandomPerspective(0.25, p=1.0)], p=0.3),
            T.TrivialAugmentWide(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.05),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),   # phone shake / focus miss
            T.ToTensor(), norm,
            T.RandomErasing(p=0.25, scale=(0.02, 0.15)),
        ])
    return T.Compose([CenterSquare(), T.Resize((img, img), interpolation=T.InterpolationMode.BILINEAR),
                      T.ToTensor(), norm])


class ManifestDataset(Dataset):
    def __init__(self, rows, class_to_idx, transform):
        self.rows, self.c2i, self.tf = rows, class_to_idx, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(r["path"]).convert("RGB")
        except Exception:  # noqa: BLE001
            im = Image.new("RGB", (256, 256), (127, 127, 127))
        return self.tf(im), self.c2i[r["label"]]


# ----------------------------------------------------------------------------- evaluation
@torch.no_grad()
def predict_logits(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            outs.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def fit_temperature(logits, labels):
    """Temperature scaling: one scalar T that makes softmax(logits / T) honest. Fit on validation data."""
    T_ = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T_], lr=0.05, max_iter=200)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(logits / T_.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T_.clamp(min=0.05).item())


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e


def summarize(logits, labels, is_field, class_names, T_=1.0):
    probs = torch.softmax(logits / T_, 1)
    pred = probs.argmax(1)
    y, p = labels.numpy(), pred.numpy()
    top3 = (probs.topk(min(3, probs.shape[1]), 1).indices == labels[:, None]).any(1).float().mean().item()
    field = np.array(is_field, dtype=bool)
    out = {
        "n": int(len(y)),
        "acc": float((p == y).mean()),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "top3": top3,
        "acc_field": float((p[field] == y[field]).mean()) if field.any() else None,
        "n_field": int(field.sum()),
        "ece": ece(probs, labels),
    }
    conf = probs.max(1).values.numpy()
    # smallest confidence threshold giving >=90% precision on what the app would show as "confident"
    rec = None
    for t in np.linspace(0.30, 0.95, 27):
        m = conf >= t
        if m.sum() >= 20 and (p[m] == y[m]).mean() >= 0.90:
            rec = {"threshold": round(float(t), 3), "coverage": float(m.mean()),
                   "precision": float((p[m] == y[m]).mean())}
            break
    out["confident_threshold"] = rec
    # most confused pairs
    conf_pairs = {}
    for t, q in zip(y, p):
        if t != q:
            conf_pairs[(class_names[t], class_names[q])] = conf_pairs.get((class_names[t], class_names[q]), 0) + 1
    out["most_confused"] = [{"true": a, "pred": b, "n": n}
                            for (a, b), n in sorted(conf_pairs.items(), key=lambda kv: -kv[1])[:15]]
    per = {}
    for i, c in enumerate(class_names):
        m = y == i
        per[c] = {"n": int(m.sum()), "recall": float((p[m] == i).mean()) if m.any() else None}
    out["per_class"] = per
    return out


def cosine_lr(step, total, warmup, base):
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
