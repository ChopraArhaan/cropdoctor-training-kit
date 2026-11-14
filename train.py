"""
Step 2: train one model.

    python train.py --model mobilenetv4_conv_medium.e500_r256_in1k
    python train.py --model convnextv2_nano.fcmae_ft_in22k_in1k --epochs-full 20

Writes runs/<model>/{best.pt, metrics.json, labels.txt}.
Selection metric = macro-F1 on the validation split (which over-samples real field photos).
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
from timm.data import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV3
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from common import (ManifestDataset, cosine_lr, fit_temperature, load_cfg, make_transforms, predict_logits,
                    read_manifest, seed_all, summarize)


def build_loaders(cfg, mean, std, class_to_idx):
    rows = read_manifest(cfg["paths"]["manifest"])
    img, tc = cfg["img_size"], cfg["train"]
    split = {s: [r for r in rows if r["split"] == s] for s in ("train", "val", "test")}

    # sampling weights: sqrt-inverse class frequency, field photos count more
    freq = {}
    for r in split["train"]:
        freq[r["label"]] = freq.get(r["label"], 0) + 1
    w = [(1.0 / np.sqrt(freq[r["label"]])) * (tc["field_weight"] if r["is_field"] else 1.0) for r in split["train"]]
    sampler = WeightedRandomSampler(w, num_samples=len(w), replacement=True)

    kw = dict(num_workers=tc["workers"], pin_memory=True, persistent_workers=tc["workers"] > 0)
    train = DataLoader(ManifestDataset(split["train"], class_to_idx, make_transforms(img, mean, std, True)),
                       batch_size=tc["batch"], sampler=sampler, drop_last=True, **kw)
    mk = lambda rs: DataLoader(ManifestDataset(rs, class_to_idx, make_transforms(img, mean, std, False)),  # noqa: E731
                               batch_size=tc["batch"] * 2, shuffle=False, **kw)
    return train, mk(split["val"]), mk(split["test"]), split


def run(cfg, model_name, epochs_head=None, epochs_full=None, tag=None):
    seed_all(cfg["seed"])
    tc = cfg["train"]
    epochs_head = tc["epochs_head"] if epochs_head is None else epochs_head
    epochs_full = tc["epochs_full"] if epochs_full is None else epochs_full
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    rows = read_manifest(cfg["paths"]["manifest"])
    class_names = sorted({r["label"] for r in rows})
    c2i = {c: i for i, c in enumerate(class_names)}

    model = timm.create_model(model_name, pretrained=True, num_classes=len(class_names), drop_rate=0.2)
    data_cfg = timm.data.resolve_model_data_config(model)
    mean, std = list(data_cfg["mean"]), list(data_cfg["std"])
    model.to(device).to(memory_format=torch.channels_last)
    ema = ModelEmaV3(model, decay=tc["ema_decay"])

    train_dl, val_dl, test_dl, split = build_loaders(cfg, mean, std, c2i)
    val_field = [r["is_field"] for r in split["val"]]
    test_field = [r["is_field"] for r in split["test"]]
    mixup = Mixup(mixup_alpha=tc["mixup"], cutmix_alpha=tc["cutmix"], label_smoothing=tc["label_smoothing"],
                  num_classes=len(class_names))
    crit = SoftTargetCrossEntropy()
    scaler = torch.amp.GradScaler(enabled=device.type == "cuda")

    out_dir = Path(cfg["paths"]["runs_dir"]) / (tag or model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "labels.txt").write_text("\n".join(class_names) + "\n", encoding="utf-8")

    best_f1, best_state, bad, t0 = -1.0, None, 0, time.time()
    total_epochs = epochs_head + epochs_full
    steps_per_epoch = len(train_dl)
    step = 0
    head_params = [p for n, p in model.named_parameters() if n.startswith(("head", "classifier", "fc"))]

    for epoch in range(total_epochs):
        frozen = epoch < epochs_head
        for p in model.parameters():
            p.requires_grad = not frozen
        for p in head_params:
            p.requires_grad = True
        base_lr = tc["lr_head"] if frozen else tc["lr_full"]
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=base_lr,
                                weight_decay=tc["weight_decay"])
        model.train()
        bar = tqdm(train_dl, desc=f"epoch {epoch + 1}/{total_epochs} {'(head)' if frozen else ''}", leave=False)
        run_loss = 0.0
        for i, (x, y) in enumerate(bar):
            lr = cosine_lr(step, total_epochs * steps_per_epoch, steps_per_epoch, base_lr)
            for g in opt.param_groups:
                g["lr"] = lr
            x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            x, ys = mixup(x, y)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                loss = crit(model(x), ys)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model, step=step)
            step += 1
            run_loss += loss.item()
            bar.set_postfix(loss=f"{run_loss / (i + 1):.3f}")

        logits, labels = predict_logits(ema.module, val_dl, device)
        m = summarize(logits, labels, val_field, class_names)
        print(f"epoch {epoch + 1:>2}: loss {run_loss / steps_per_epoch:.3f} | val acc {m['acc']:.4f} "
              f"| val field acc {m['acc_field']} | macro-F1 {m['macro_f1']:.4f} | {time.time() - t0:.0f}s")
        if m["macro_f1"] > best_f1 and not frozen:
            best_f1, bad = m["macro_f1"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
        elif not frozen:
            bad += 1
            if bad >= tc["patience"]:
                print("early stop")
                break

    if best_state is None:  # only head epochs requested
        best_state = {k: v.detach().cpu().clone() for k, v in ema.module.state_dict().items()}
    ema.module.load_state_dict(best_state)
    net = ema.module

    # calibrate on VAL, report on held-out TEST
    v_logits, v_labels = predict_logits(net, val_dl, device)
    temperature = fit_temperature(v_logits, v_labels)
    val_m = summarize(v_logits, v_labels, val_field, class_names, temperature)
    t_logits, t_labels = predict_logits(net, test_dl, device)
    test_m = summarize(t_logits, t_labels, test_field, class_names, temperature)
    test_raw = summarize(t_logits, t_labels, test_field, class_names, 1.0)

    # CPU latency (desktop CPU, only for ranking candidates: measure the real phone with the app's Benchmark screen)
    net_cpu = timm.create_model(model_name, pretrained=False, num_classes=len(class_names)).eval()
    net_cpu.load_state_dict(best_state)
    xin = torch.randn(1, 3, cfg["img_size"], cfg["img_size"])
    with torch.no_grad():
        for _ in range(3):
            net_cpu(xin)
        t1 = time.time()
        for _ in range(15):
            net_cpu(xin)
    cpu_ms = (time.time() - t1) / 15 * 1000
    params_m = sum(p.numel() for p in net_cpu.parameters()) / 1e6

    metrics = {
        "model": model_name, "img_size": cfg["img_size"], "classes": len(class_names),
        "mean": mean, "std": std, "temperature": temperature, "params_m": params_m, "cpu_ms": cpu_ms,
        "val": val_m, "test": test_m, "test_ece_before_calibration": test_raw["ece"],
        "train_seconds": time.time() - t0,
    }
    torch.save({"state_dict": best_state, "classes": class_names, "model": model_name, "mean": mean, "std": std,
                "img_size": cfg["img_size"], "temperature": temperature}, out_dir / "best.pt")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== TEST (held-out, field-weighted) ===")
    print(f"acc {test_m['acc']:.4f} | field acc {test_m['acc_field']} | macro-F1 {test_m['macro_f1']:.4f} | "
          f"top3 {test_m['top3']:.4f} | ECE {test_m['ece']:.4f} (was {test_raw['ece']:.4f} before calibration) | T={temperature:.3f}")
    print("most confused:")
    for c in test_m["most_confused"][:10]:
        print(f"  {c['n']:>3}  {c['true']}  ->  {c['pred']}")
    print(f"saved {out_dir / 'best.pt'}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--model", required=True)
    ap.add_argument("--epochs-head", type=int)
    ap.add_argument("--epochs-full", type=int)
    ap.add_argument("--tag")
    a = ap.parse_args()
    run(load_cfg(a.config), a.model, a.epochs_head, a.epochs_full, a.tag)
