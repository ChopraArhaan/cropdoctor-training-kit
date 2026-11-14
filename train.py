"""
Train the CropDoctor classifier. One script: train -> calibrate -> test report.

    python train.py                       # default: ~15-30 min on a mid-range GPU
    python train.py --epochs 8 --samples 20000     # longer / better
    python train.py --model convnextv2_nano.fcmae_ft_in22k_in1k   # another backbone

Needs data/manifest.csv from prepare_data.py. It fine-tunes an ImageNet-pretrained backbone (downloaded
automatically by timm), so it converges in a few epochs: the heavy lifting was already done by the pretrained weights.
Output: runs/<model>/best.pt, labels.txt, metrics.json. Then run export.py.
"""
import argparse
import csv
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torchvision.transforms as T
import yaml
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True
DEFAULT_MODEL = "mobilenetv4_conv_medium.e500_r256_in1k"


# ------------------------------------------------------------------ data
class CenterSquare:
    """Centre-crop to a square. The Android app does exactly this before resizing, so keep it in sync."""

    def __call__(self, im):
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        return im.crop((left, top, left + s, top + s))


def make_transform(size, mean, std, train):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.4, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.ColorJitter(0.3, 0.3, 0.3, 0.03),
            T.RandomApply([T.GaussianBlur(5, (0.1, 2.0))], p=0.2),   # phone shake / missed focus
            T.ToTensor(), T.Normalize(mean, std),
        ])
    return T.Compose([CenterSquare(), T.Resize((size, size)), T.ToTensor(), T.Normalize(mean, std)])


class Rows(Dataset):
    def __init__(self, rows, class_index, transform):
        self.rows, self.idx, self.tf = rows, class_index, transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        try:
            im = Image.open(r["path"])
            im.draft("RGB", (512, 512))      # JPEG: decode big phone photos at reduced size (much faster)
            im = im.convert("RGB")
        except Exception:                    # unreadable file: a grey image instead of a crash
            im = Image.new("RGB", (256, 256), (127, 127, 127))
        return self.tf(im), self.idx[r["label"]]


def read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["is_field"] = r["is_field"] == "1"
    return rows


# ------------------------------------------------------------------ metrics
@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outs, ys = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            outs.append(model(x).float().cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def macro_f1(y, p, n):
    f1 = []
    for c in range(n):
        tp = int(((p == c) & (y == c)).sum())
        fp = int(((p == c) & (y != c)).sum())
        fn = int(((p != c) & (y == c)).sum())
        if tp + fp + fn:
            f1.append(2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f1)) if f1 else 0.0


def fit_temperature(logits, labels):
    """One scalar T so that softmax(logits / T) is honest about its confidence. Fit on validation data."""
    t = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([t], lr=0.05, max_iter=200)
    ce = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = ce(logits / t.clamp(min=0.05), labels)
        loss.backward()
        return loss

    opt.step(closure)
    return float(t.clamp(min=0.05).item())


def ece(probs, labels, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == labels).float()
    edges = torch.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return total


def report(logits, labels, is_field, names, temp=1.0):
    probs = torch.softmax(logits / temp, 1)
    y, p = labels.numpy(), probs.argmax(1).numpy()
    conf = probs.max(1).values.numpy()
    field = np.array(is_field, dtype=bool)
    out = {
        "n": int(len(y)),
        "acc": float((p == y).mean()),
        "macro_f1": macro_f1(y, p, len(names)),
        "top3": float((probs.topk(min(3, probs.shape[1]), 1).indices == labels[:, None]).any(1).float().mean()),
        "acc_field": float((p[field] == y[field]).mean()) if field.any() else None,
        "n_field": int(field.sum()),
        "ece": ece(probs, labels),
        "confident_threshold": None,
    }
    for t in np.linspace(0.30, 0.95, 27):       # lowest threshold whose "confident" answers are >= 90% right
        m = conf >= t
        if m.sum() >= 20 and (p[m] == y[m]).mean() >= 0.90:
            out["confident_threshold"] = {"threshold": round(float(t), 3), "coverage": float(m.mean())}
            break
    recalls = []
    for i, c in enumerate(names):
        m = y == i
        if m.sum() >= 10:
            recalls.append((float((p[m] == i).mean()), c, int(m.sum())))
    out["weakest_classes"] = [{"class": c, "recall": r, "n": n} for r, c, n in sorted(recalls)[:8]]
    return out


# ------------------------------------------------------------------ training
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--epochs", type=int, default=6, help="total epochs (the first one trains only the new head)")
    ap.add_argument("--samples", type=int, default=12000, help="images drawn per epoch")
    ap.add_argument("--workers", type=int, help="data-loader workers (use 0 if Windows workers crash)")
    ap.add_argument("--batch", type=int)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    tc = cfg["train"]
    batch = a.batch or tc["batch"]
    workers = tc["workers"] if a.workers is None else a.workers
    size = cfg["img_size"]
    random.seed(cfg["seed"]); np.random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise SystemExit("CUDA is not available, so training would take many hours. Fix the PyTorch CUDA install "
                         "(see README), or run this on Google Colab with a GPU.")

    rows = read_manifest(cfg["paths"]["manifest"])
    names = sorted({r["label"] for r in rows})
    idx = {c: i for i, c in enumerate(names)}
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}
    print(f"[data] {len(names)} classes | train {len(split['train'])} | val {len(split['val'])} | "
          f"test {len(split['test'])} | field images in test: {sum(r['is_field'] for r in split['test'])}")

    model = timm.create_model(a.model, pretrained=True, num_classes=len(names), drop_rate=0.2)
    dc = timm.data.resolve_model_data_config(model)
    mean, std = list(dc["mean"]), list(dc["std"])
    model.to(device).to(memory_format=torch.channels_last)
    head = list(model.get_classifier().parameters())

    freq = {}
    for r in split["train"]:
        freq[r["label"]] = freq.get(r["label"], 0) + 1
    w = [(1 / math.sqrt(freq[r["label"]])) * (tc["field_weight"] if r["is_field"] else 1.0) for r in split["train"]]
    sampler = WeightedRandomSampler(w, num_samples=a.samples, replacement=True)

    def loader(rs, train):
        kw = dict(num_workers=workers, pin_memory=True, persistent_workers=workers > 0)
        ds = Rows(rs, idx, make_transform(size, mean, std, train))
        if train:
            return DataLoader(ds, batch_size=batch, sampler=sampler, drop_last=True, **kw)
        return DataLoader(ds, batch_size=batch * 2, shuffle=False, **kw)

    train_dl = loader(split["train"], True)
    val_dl, test_dl = loader(split["val"], False), loader(split["test"], False)
    sub = random.Random(cfg["seed"]).sample(split["val"], min(3000, len(split["val"])))   # quick per-epoch check
    sub_dl, sub_field = loader(sub, False), [r["is_field"] for r in sub]

    try:
        scaler = torch.amp.GradScaler("cuda")
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler()
    crit = torch.nn.CrossEntropyLoss(label_smoothing=tc["label_smoothing"])

    steps = len(train_dl)
    full_steps = max(1, (a.epochs - 1) * steps)
    print(f"[train] {a.model} | {steps} steps/epoch x {a.epochs} epochs | batch {batch} | workers {workers}")

    out_dir = Path(cfg["paths"]["runs_dir"]) / a.model
    out_dir.mkdir(parents=True, exist_ok=True)
    best_f1, best_state, step, t0 = -1.0, None, 0, time.time()

    for epoch in range(a.epochs):
        head_only = epoch == 0 and a.epochs > 1
        for p in model.parameters():
            p.requires_grad = not head_only
        for p in head:
            p.requires_grad = True
        if epoch <= 1:   # new optimiser when the set of trainable weights changes
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                    lr=tc["lr_head"] if head_only else tc["lr_full"], weight_decay=tc["weight_decay"])
        model.train()
        running = 0.0
        for i, (x, y) in enumerate(tqdm(train_dl, desc=f"epoch {epoch + 1}/{a.epochs}" + (" (head)" if head_only else ""), leave=False)):
            if head_only:
                lr = tc["lr_head"]
            else:
                k = step - (steps if a.epochs > 1 else 0)   # steps taken in the full phase
                lr = tc["lr_full"] * min(1.0, (k + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, k / full_steps)))
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            with torch.autocast("cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            step += 1
            running += loss.item()

        lg, lb = predict(model, sub_dl, device)
        yv, pv = lb.numpy(), lg.argmax(1).numpy()
        f1 = macro_f1(yv, pv, len(names))
        fld = np.array(sub_field)
        fa = float((pv[fld] == yv[fld]).mean()) if fld.any() else float("nan")
        done = epoch + 1
        eta = (time.time() - t0) / done * (a.epochs - done) / 60
        print(f"epoch {done}/{a.epochs}: loss {running / max(1, steps):.3f} | val acc {(pv == yv).mean():.4f} | "
              f"field acc {fa:.4f} | macro-F1 {f1:.4f} | ~{eta:.0f} min left", flush=True)
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    v_logits, v_labels = predict(model, val_dl, device)
    temp = fit_temperature(v_logits, v_labels)
    val_m = report(v_logits, v_labels, [r["is_field"] for r in split["val"]], names, temp)
    t_logits, t_labels = predict(model, test_dl, device)
    test_m = report(t_logits, t_labels, [r["is_field"] for r in split["test"]], names, temp)
    raw_ece = report(t_logits, t_labels, [r["is_field"] for r in split["test"]], names, 1.0)["ece"]

    torch.save({"state_dict": best_state, "classes": names, "model": a.model, "mean": mean, "std": std,
                "img_size": size, "temperature": temp}, out_dir / "best.pt")
    (out_dir / "labels.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(
        {"model": a.model, "classes": len(names), "temperature": temp, "val": val_m, "test": test_m,
         "test_ece_before_calibration": raw_ece, "train_minutes": (time.time() - t0) / 60}, indent=2), encoding="utf-8")

    fa = test_m["acc_field"]
    print("\n=== TEST RESULTS (images the model never saw) ===")
    print(f"overall accuracy : {test_m['acc']:.4f}   ({test_m['n']} images)")
    print(f"FIELD accuracy   : {'n/a' if fa is None else f'{fa:.4f}'}   ({test_m['n_field']} field images)  <- quote this one")
    print(f"macro-F1 {test_m['macro_f1']:.4f} | top-3 {test_m['top3']:.4f} | "
          f"calibration error {test_m['ece']:.4f} (was {raw_ece:.4f} before calibration) | T={temp:.3f}")
    print("weakest classes:")
    for c in test_m["weakest_classes"]:
        print(f"  recall {c['recall']:.2f}  {c['class']}  (n={c['n']})")
    print(f"\nsaved {out_dir / 'best.pt'}   total {(time.time() - t0) / 60:.1f} min\nNext: python export.py")


if __name__ == "__main__":
    main()
