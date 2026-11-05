"""Download SH17, extract it safely, and write a resized copy with per-image metadata."""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path, PurePosixPath

import requests
from PIL import Image, ImageOps

from .common import DATA, EXTRACT, NAMES, RAW, SH17_KAGGLE, read_labels, save_json

MAX_SIDE = 1280   # originals are Pexels full-size photos (often 4-6k px); 1280 keeps small PPE visible
IMG_EXT = {".jpg", ".jpeg", ".png", ".webp"}


def download(url: str, dest: Path) -> Path:
    """Resumable HTTP download (Kaggle's public API redirects to a signed storage URL)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    done = dest.stat().st_size if dest.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with requests.get(url, headers=headers, stream=True, timeout=60) as r:
        if r.status_code == 416:      # already complete
            return dest
        r.raise_for_status()
        mode = "ab" if r.status_code == 206 else "wb"
        with open(dest, mode) as f:
            for chunk in r.iter_content(1 << 22):
                f.write(chunk)
    return dest


def fetch() -> Path:
    archive = RAW / "sh17_archive.zip"
    if not (archive.exists() and zipfile.is_zipfile(archive)):
        print(f"downloading SH17 (~14.2 GB) -> {archive}")
        download(SH17_KAGGLE, archive)
    return archive


def safe_extract(archive: Path, out: Path) -> None:
    """Extract images, YOLO labels, Pexels metadata and split lists; refuse unsafe member names."""
    if (out / ".done").exists():
        return
    with zipfile.ZipFile(archive) as z:
        for m in z.infolist():
            p = PurePosixPath(m.filename)
            if m.is_dir() or p.is_absolute() or ".." in p.parts:
                continue
            ext = p.suffix.lower()
            if ext in IMG_EXT or (ext == ".txt" and "voc_labels" not in p.parts) or (ext == ".json" and "meta-data" in p.parts):
                target = out.joinpath(*p.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(m) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
    (out / ".done").write_text("ok")


def _resize_one(args):
    src, dst = args
    with Image.open(src) as im:
        orient = im.getexif().get(0x0112, 1)
        im.draft("RGB", (MAX_SIDE, MAX_SIDE))          # fast JPEG downscale on decode
        im = ImageOps.exif_transpose(im)               # match cv2.imread, which also applies EXIF
        ow, oh = im.size
        im = im.convert("RGB")
        im.thumbnail((MAX_SIDE, MAX_SIDE), Image.Resampling.LANCZOS)
        im.save(dst, "JPEG", quality=92)
        return src.stem, orient, ow, oh, im.size[0], im.size[1]


def pexels_meta(path: Path) -> dict:
    """Pexels API record shipped with SH17: photo id, photographer id/name and the page title."""
    if not path.exists():
        return {"pexels_id": -1, "photographer_id": -1, "photographer": "", "title": ""}
    d = json.loads(path.read_text(encoding="utf-8"))
    pid = int(d.get("id", -1))
    title = re.sub(rf"-?{pid}/?$", "", d.get("url", "").rstrip("/").split("/")[-1])
    return {"pexels_id": pid, "photographer_id": int(d.get("photographer_id", -1)),
            "photographer": d.get("photographer", ""), "title": title}


def prepare(workers: int = 12) -> Path:
    archive = fetch()
    safe_extract(archive, EXTRACT)
    meta_path = DATA / "meta.json"
    if meta_path.exists():
        return meta_path

    images = {p.stem: p for p in EXTRACT.rglob("*") if p.suffix.lower() in IMG_EXT and "images" in p.parts}
    labels = {p.stem: p for p in EXTRACT.rglob("*.txt") if "labels" in p.parts}
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    (DATA / "labels").mkdir(parents=True, exist_ok=True)

    jobs = [(p, DATA / "images" / f"{s}.jpg") for s, p in sorted(images.items())]
    with ProcessPoolExecutor(workers) as ex:
        sizes = {r[0]: r[1:] for r in ex.map(_resize_one, jobs, chunksize=16)}

    rows, bad = [], 0
    for stem in sorted(images):
        lab = labels.get(stem)
        boxes = read_labels(lab) if lab else []
        ok = [b for b in boxes if 0 <= b[0] < len(NAMES) and 0 < b[3] <= 1 and 0 < b[4] <= 1]
        bad += len(boxes) - len(ok)
        (DATA / "labels" / f"{stem}.txt").write_text(
            "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in ok))
        orient, ow, oh, w, h = sizes[stem]
        counts = [0] * len(NAMES)
        for b in ok:
            counts[b[0]] += 1
        rows.append(dict(stem=stem, **pexels_meta(EXTRACT / "meta-data" / f"{stem}.json"), exif=orient,
                         orig_w=ow, orig_h=oh, w=w, h=h, counts=counts))

    official = {}
    for name in ("train_files.txt", "val_files.txt"):
        for f in EXTRACT.rglob(name):
            official.update({Path(line.strip()).stem: name.split("_")[0]
                             for line in f.read_text().splitlines() if line.strip()})
    for r in rows:
        r["official_split"] = official.get(r["stem"], "")

    save_json({"n_images": len(rows), "dropped_boxes": bad, "names": NAMES, "images": rows}, meta_path)
    print(f"prepared {len(rows)} images, dropped {bad} invalid boxes")
    return meta_path
