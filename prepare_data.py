"""
Step 1: download datasets, map folder names into the CropDoctor taxonomy, build a leak-free manifest.

    python prepare_data.py              # everything
    python prepare_data.py --no-download   # reuse what is already in data/raw

Output: data/manifest.csv  (path,label,source,is_field,split)  and  data/unmapped.txt

No images are copied: the manifest points at the original files, which saves tens of GB.

Leak-free splitting: near-duplicate images (same photo, flipped/rotated copies, augmented copies shipped inside
some datasets) are grouped by perceptual hash and a whole group goes to ONE split. A random split would leak
copies into validation and inflate accuracy.
"""
import argparse
import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import imagehash
import yaml
from PIL import Image, ImageFile
from tqdm import tqdm

from taxonomy import BACKGROUND, canonical, one_sided_crops

ImageFile.LOAD_TRUNCATED_IMAGES = True
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_WORDS = {"train", "valid", "val", "test", "images", "dataset", "data"}


# ----------------------------------------------------------------------------- download
def fetch_kaggle(src, raw_dir: Path) -> Path | None:
    try:
        import kagglehub
        p = kagglehub.dataset_download(src["handle"])
        print(f"[kaggle] {src['name']} -> {p}")
        return Path(p)
    except Exception as e:  # noqa: BLE001
        print(f"[kaggle SKIP] {src['name']}: {e}\n   (set KAGGLE_USERNAME / KAGGLE_KEY, or download manually into {raw_dir / src['name']})")
        return None


def fetch_hf(src, raw_dir: Path) -> Path | None:
    out = raw_dir / src["name"]
    if out.exists() and any(out.iterdir()):
        print(f"[hf cache] {src['name']} -> {out}")
        return out
    try:
        from datasets import load_dataset
        ds = load_dataset(src["handle"], split=src.get("split", "train"))
        label_key = next((k for k in ("label", "label_name", "category", "disease") if k in ds.features), None)
        image_key = next((k for k in ("image", "img", "photo") if k in ds.features), None)
        if not label_key or not image_key:
            raise RuntimeError(f"cannot find image/label columns in {list(ds.features)}")
        names = getattr(ds.features[label_key], "names", None)
        for i, row in enumerate(tqdm(ds, desc=src["name"])):
            lab = row[label_key]
            lab = names[lab] if names is not None and isinstance(lab, int) else str(lab)
            d = out / lab.replace("/", "_")
            d.mkdir(parents=True, exist_ok=True)
            row[image_key].convert("RGB").save(d / f"{i}.jpg", quality=92)
        return out
    except Exception as e:  # noqa: BLE001
        print(f"[hf SKIP] {src['name']}: {e}")
        return None


def locate_sources(cfg, download: bool) -> dict:
    raw = Path(cfg["paths"]["raw_dir"])
    raw.mkdir(parents=True, exist_ok=True)
    found = {}
    for src in cfg["sources"]:
        name = src["name"]
        if src["kind"] == "local":
            p = Path(os.path.expandvars(os.path.expanduser(str(src["path"]))))
            if p.exists():
                found[name] = (p, src)
            else:
                print(f"[local SKIP] {name}: folder not found: {p}")
            continue
        manual = raw / name
        if manual.exists() and any(manual.iterdir()):
            found[name] = (manual, src)
            continue
        if not download:
            continue
        p = fetch_kaggle(src, raw) if src["kind"] == "kaggle" else fetch_hf(src, raw)
        if p:
            found[name] = (p, src)
    return found


# ----------------------------------------------------------------------------- scan + map
def scan_source(root: Path, src: dict, unmapped: set):
    """Yield (path, label) for every image under root whose parent folder maps into the taxonomy."""
    by_folder = defaultdict(list)
    for dirpath, _, files in os.walk(root):
        for f in files:
            if os.path.splitext(f)[1].lower() in IMG_EXT:
                by_folder[dirpath].append(os.path.join(dirpath, f))
    for dirpath, files in by_folder.items():
        name = os.path.basename(dirpath)
        if name.lower() in SPLIT_WORDS or len(files) < 3:
            continue
        label = canonical(name, src.get("crop"))
        if label is None:
            unmapped.add(f"{src['name']}: {name} ({len(files)} images)")
            continue
        for p in files:
            yield p, label


def negatives(cfg, raw_dir: Path):
    """Collect 'not a leaf' images: user folder + DTD + Oxford Pets."""
    out, ncfg = [], cfg["negatives"]
    nd = Path(cfg["paths"]["negatives_dir"])
    if nd.exists():
        out += [str(p) for p in nd.rglob("*") if p.suffix.lower() in IMG_EXT]
        print(f"[negatives] {len(out)} from {nd}")
    try:
        import torchvision
        if ncfg.get("use_dtd"):
            torchvision.datasets.DTD(str(raw_dir / "tv"), split="train", download=True)
            out += [str(p) for p in (raw_dir / "tv" / "dtd").rglob("*.jpg")]
        if ncfg.get("use_pets"):
            torchvision.datasets.OxfordIIITPet(str(raw_dir / "tv"), download=True)
            out += [str(p) for p in (raw_dir / "tv" / "oxford-iiit-pet" / "images").glob("*.jpg")]
    except Exception as e:  # noqa: BLE001
        print(f"[negatives] torchvision negatives skipped: {e}")
    random.shuffle(out)
    return out[: ncfg["max_total"]]


# ----------------------------------------------------------------------------- dedup groups
def dihedral_hash(path: str) -> int | None:
    """Orientation-invariant perceptual hash: min over the 8 flips/rotations."""
    try:
        im = Image.open(path).convert("L")
        im.draft("L", (128, 128))
        best = None
        for t in (None, Image.FLIP_LEFT_RIGHT):
            base = im if t is None else im.transpose(t)
            for rot in (0, 90, 180, 270):
                v = int(str(imagehash.phash(base.rotate(rot, expand=True))), 16)
                best = v if best is None or v < best else best
        return best
    except Exception:  # noqa: BLE001
        return None


def group_near_duplicates(paths: list[str], thresh: int) -> list[list[str]]:
    hashes = [(p, dihedral_hash(p)) for p in paths]
    hashes = [(p, h) for p, h in hashes if h is not None]
    parent = list(range(len(hashes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(hashes)):
        hi = hashes[i][1]
        for j in range(i + 1, len(hashes)):
            if bin(hi ^ hashes[j][1]).count("1") <= thresh:
                parent[find(j)] = find(i)
    groups = defaultdict(list)
    for i, (p, _) in enumerate(hashes):
        groups[find(i)].append(p)
    return list(groups.values())


def pick_split(is_field: bool, cfg) -> str:
    probs = cfg["data"]["split_field" if is_field else "split_lab"]
    return random.choices(["train", "val", "test"], weights=probs)[0]


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--no-download", action="store_true")
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config, encoding="utf-8"))
    random.seed(cfg["seed"])
    dcfg = cfg["data"]

    found = locate_sources(cfg, download=not a.no_download)
    if not found:
        raise SystemExit("No datasets found. See README.md, section 'Datasets'.")

    unmapped, per_class = set(), defaultdict(lambda: {"field": [], "lab": []})
    for name, (root, src) in found.items():
        n = 0
        for path, label in scan_source(root, src, unmapped):
            per_class[label]["field" if src.get("field") else "lab"].append((path, name))
            n += 1
        print(f"[scan] {name}: {n} images mapped")

    Path("data").mkdir(exist_ok=True)
    Path("data/unmapped.txt").write_text("\n".join(sorted(unmapped)), encoding="utf-8")
    print(f"[scan] {len(unmapped)} folders outside the taxonomy -> data/unmapped.txt")

    # Cap each source per class so one huge dataset cannot crowd out the others (and cannot flood the cap with
    # its own photos, which would also distort the field/lab balance).
    per_src = dcfg.get("max_per_source_per_class")
    if per_src:
        for d in per_class.values():
            for kind in ("field", "lab"):
                random.shuffle(d[kind])
                seen = defaultdict(int)
                kept = []
                for p_, s_ in d[kind]:
                    seen[s_] += 1
                    if seen[s_] <= per_src:
                        kept.append((p_, s_))
                d[kind] = kept

    # Drop crops that have only a healthy class or only disease classes: they would mislead (see taxonomy.py).
    counts = {lab: len(d["field"]) + len(d["lab"]) for lab, d in per_class.items()}
    for crop, why in sorted(one_sided_crops(counts, dcfg["min_per_class"]).items()):
        print(f"[drop crop] {crop}: {why}. Add the missing side to the datasets to keep it.")
        for lab in [k for k in per_class if k.startswith(crop + "___")]:
            del per_class[lab]

    rows = []
    for label, d in sorted(per_class.items()):
        random.shuffle(d["field"])
        random.shuffle(d["lab"])
        cap = dcfg["max_per_class"]
        chosen = d["field"][:cap] + d["lab"][: max(0, cap - len(d["field"][:cap]))]
        if len(chosen) < dcfg["min_per_class"]:
            print(f"[drop] {label}: only {len(chosen)} images")
            continue
        field_set = {p for p, _ in d["field"]}
        src_of = dict(chosen)
        groups = group_near_duplicates([p for p, _ in chosen], dcfg["dedup_hamming"])
        # first group always goes to train so every class is trainable; one group to val/test if there are enough
        random.shuffle(groups)
        for gi, g in enumerate(groups):
            is_field = any(x in field_set for x in g)
            split = "train" if gi == 0 else pick_split(is_field, cfg)
            for p in g:
                rows.append((p, label, src_of[p], int(p in field_set), split))
        print(f"[class] {label:<42} {len(chosen):>5} images in {len(groups):>5} groups")

    neg = negatives(cfg, Path(cfg["paths"]["raw_dir"]))
    groups_neg = [neg[i:i + 1] for i in range(len(neg))]
    for p in neg:
        rows.append((p, BACKGROUND, "negatives", 0, random.choices(["train", "val", "test"], weights=[0.8, 0.1, 0.1])[0]))
    print(f"[class] {BACKGROUND:<42} {len(neg):>5} images")

    with open(cfg["paths"]["manifest"], "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "source", "is_field", "split"])
        w.writerows(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[r[4]] += 1
    print(f"[done] {len(rows)} rows -> {cfg['paths']['manifest']}   splits: {dict(counts)}")


if __name__ == "__main__":
    main()
