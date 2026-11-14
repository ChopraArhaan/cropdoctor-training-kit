"""
Canonical class taxonomy for CropDoctor.

Every training image is mapped to a class id of the form  <Crop>___<disease_id>.
The Android app's knowledge base (app/src/main/assets/knowledge_base.json) is keyed by the SAME ids,
so do not rename ids here without renaming them there.

Anything that does not match a rule is written to data/unmapped.txt and skipped, so you can see what the
datasets contain that the taxonomy does not cover yet.
"""
import re

BACKGROUND = "Background___not_a_leaf"

# crop -> keywords that identify the crop inside a folder name
CROP_KEYWORDS = {
    "Tomato": r"tomato",
    "Potato": r"potato",
    "Corn": r"corn|maize",
    "Rice": r"rice|paddy",
    "Sugarcane": r"sugar ?cane",
    "Cotton": r"cotton",
    "Pepper": r"pepper|capsicum|chilli|chili",
    "Grape": r"grape",
    "Apple": r"apple",
    "Blueberry": r"blueberry",
    "Cherry": r"cherry",
    "Citrus": r"citrus|orange|lemon|lime",
    "Peach": r"peach",
    "Raspberry": r"raspberry",
    "Soybean": r"soybean|soyabean",
    "Squash": r"squash",
    "Strawberry": r"strawberry",
}

# crop -> ordered list of (disease_id, regex). First match wins, so put specific rules first.
DISEASES = {
    "Tomato": [
        ("early_blight", r"early blight"),
        ("late_blight", r"late blight"),
        ("bacterial_spot", r"bacterial spot"),
        ("leaf_mold", r"mold|mould"),
        ("septoria_leaf_spot", r"septoria"),
        ("spider_mites", r"spider|mite"),
        ("target_spot", r"target spot"),
        ("yellow_leaf_curl_virus", r"yellow leaf curl|yellow virus|ylcv|yellow"),
        ("mosaic_virus", r"mosaic"),
    ],
    "Potato": [
        ("early_blight", r"early blight"),
        ("late_blight", r"late blight"),
    ],
    "Corn": [
        ("gray_leaf_spot", r"gray|grey|cercospora"),
        ("common_rust", r"rust"),
        ("northern_leaf_blight", r"northern|blight"),
    ],
    "Rice": [
        ("bacterial_leaf_blight", r"bacterial leaf blight|bacterial blight|blb"),
        ("bacterial_leaf_streak", r"streak"),
        ("bacterial_panicle_blight", r"panicle"),
        ("brown_spot", r"brown spot"),
        ("leaf_blast", r"blast"),
        ("leaf_smut", r"smut"),
        ("tungro", r"tungro"),
        ("hispa", r"hispa"),
        ("dead_heart", r"dead ?heart"),
        ("downy_mildew", r"downy"),
    ],
    "Sugarcane": [
        ("red_rot", r"red ?rot"),
        ("mosaic", r"mosaic"),
        ("rust", r"rust"),
        ("yellow_leaf", r"yellow"),
    ],
    "Cotton": [
        ("alternaria_leaf_spot", r"alternaria"),
        ("curl_virus", r"curl"),
        ("wilt", r"fusarium|verticillium|wilt"),
        ("bacterial_blight", r"bacterial"),
    ],
    "Pepper": [
        ("bacterial_spot", r"bacterial spot|leaf spot"),
    ],
    "Grape": [
        ("black_rot", r"black rot"),
        ("esca", r"esca|black measles"),
        ("leaf_blight", r"leaf blight|isariopsis"),
    ],
    "Apple": [
        ("scab", r"scab"),
        ("black_rot", r"black rot"),
        ("cedar_apple_rust", r"cedar|rust"),
    ],
    "Blueberry": [],
    "Cherry": [
        ("powdery_mildew", r"powdery|mildew"),
    ],
    "Citrus": [
        ("citrus_greening", r"haunglongbing|greening|hlb"),
    ],
    "Peach": [
        ("bacterial_spot", r"bacterial spot|bacterial|spot"),
    ],
    "Raspberry": [],
    "Soybean": [],
    "Squash": [
        ("powdery_mildew", r"powdery|mildew"),
    ],
    "Strawberry": [
        ("leaf_scorch", r"scorch|leaf scorch"),
    ],
}

_FILLER = re.compile(r"\b(leaf|leaves|plant|crop|disease|diseased|image|images|bell|\(|\))\b")


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def canonical(folder_name: str, crop_hint: str | None = None) -> str | None:
    """Map a dataset folder name to '<Crop>___<disease_id>' or None if it is outside the taxonomy."""
    s = _norm(folder_name)
    crop = None
    for c, pat in CROP_KEYWORDS.items():
        if re.search(pat, s):
            crop = c
            break
    if crop is None:
        crop = crop_hint
    if crop is None or crop not in DISEASES:
        return None

    for disease_id, pat in DISEASES[crop]:
        if re.search(pat, s):
            return f"{crop}___{disease_id}"

    if re.search(r"healthy|normal", s):
        return f"{crop}___healthy"

    # Names like "Tomato leaf" / "Apple leaf" (PlantDoc) carry no disease words: healthy.
    leftover = CROP_KEYWORDS[crop]
    leftover = re.sub(leftover, " ", s)
    leftover = _FILLER.sub(" ", leftover)
    leftover = re.sub(r"[^a-z]+", " ", leftover).strip()
    if leftover == "":
        return f"{crop}___healthy"
    return None


def all_expected_classes() -> list[str]:
    out = [BACKGROUND]
    for crop, ds in DISEASES.items():
        out.append(f"{crop}___healthy")
        out.extend(f"{crop}___{d}" for d, _ in ds)
    return sorted(out)


def one_sided_crops(counts: dict, min_images: int = 1) -> dict:
    """
    Crops that cannot be classified safely because they have only a healthy class or only disease classes.

    A crop with only 'healthy' would label a diseased leaf healthy; a crop with only diseases would call every
    healthy leaf diseased. Both mislead a farmer, so prepare_data.py drops such crops (and says so).
    `counts` maps class id -> number of images. Returns {crop: reason}.
    """
    by_crop: dict = {}
    for cid, n in counts.items():
        if cid == BACKGROUND or n < min_images or "___" not in cid:
            continue
        crop, disease = cid.split("___", 1)
        side = by_crop.setdefault(crop, {"healthy": False, "disease": False})
        side["healthy" if disease == "healthy" else "disease"] = True
    out = {}
    for crop, side in by_crop.items():
        if side["healthy"] and not side["disease"]:
            out[crop] = "has a healthy class but no disease class"
        elif side["disease"] and not side["healthy"]:
            out[crop] = "has disease classes but no healthy class"
    return out


if __name__ == "__main__":
    classes = all_expected_classes()
    print(len(classes), "classes in the taxonomy")
    for c in classes:
        print(" ", c)
