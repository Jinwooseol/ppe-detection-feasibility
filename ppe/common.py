"""Shared paths, class names and small helpers for the PPE feasibility pipeline."""
from __future__ import annotations

import json
import os
from pathlib import Path

# Code, results and the PDF live in the repo. Heavy files (venv, raw data, training runs)
# live in a work directory outside OneDrive so they are not synced to the cloud.
REPO = Path(__file__).resolve().parents[1]
WORK = Path(os.environ.get("PPE_WORK", Path.home() / "ml-work" / "ppe-detection-feasibility"))

RAW = WORK / "data" / "raw"            # downloaded archive (untrusted, nothing is executed from here)
EXTRACT = WORK / "data" / "sh17_raw"   # extracted archive
DATA = WORK / "data" / "sh17"          # resized images + YOLO labels used for training
SPLITS = WORK / "splits"
RUNS = WORK / "runs"

RESULTS = REPO / "results"
FIGURES = RESULTS / "figures"
MODELS = REPO / "models"
DOCS = REPO / "docs"

SH17_KAGGLE = "https://www.kaggle.com/api/v1/datasets/download/mugheesahmad/sh17-dataset-for-ppe-detection"

# Class order of the SH17 release (sh17.yaml).
NAMES = [
    "person", "ear", "ear-mufs", "face", "face-guard", "face-mask", "foot", "tool", "glasses",
    "gloves", "helmet", "hands", "head", "medical-suit", "shoes", "safety-suit", "safety-vest",
]
CID = {n: i for i, n in enumerate(NAMES)}

# Pairs a safety team would care about: the protected and the unprotected body part look alike
# at a distance, and a wrong call in either direction is an alert error.
LOOKALIKE = [
    ("helmet", "head"),
    ("gloves", "hands"),
    ("shoes", "foot"),
    ("ear-mufs", "ear"),
    ("face-mask", "face"),
    ("face-guard", "face"),
    ("safety-vest", "safety-suit"),
    ("safety-suit", "medical-suit"),
]

SEED = 0


def ensure_dirs() -> None:
    for d in (RAW, EXTRACT, DATA, SPLITS, RUNS, RESULTS, FIGURES, MODELS, DOCS):
        d.mkdir(parents=True, exist_ok=True)


WORK_TOKEN = "$PPE_WORK"


def portable(obj):
    """Write paths inside the work directory as $PPE_WORK/... so results carry no local paths."""
    if isinstance(obj, str):
        for prefix in (str(WORK), WORK.as_posix()):
            if obj.startswith(prefix):
                return WORK_TOKEN + obj[len(prefix):].replace("\\", "/")
        return obj
    if isinstance(obj, dict):
        return {k: portable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v) for v in obj]
    return obj


def local(p) -> Path:
    """Inverse of portable() for one path."""
    s = str(p).replace("\\", "/")
    return WORK / s[len(WORK_TOKEN) + 1:] if s.startswith(WORK_TOKEN) else Path(p)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(portable(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_labels(path: Path):
    """YOLO label file -> list of (cls, cx, cy, w, h), all normalised."""
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) == 5:
                rows.append((int(parts[0]), *map(float, parts[1:])))
    return rows


def data_yaml(split: str) -> Path:
    return SPLITS / split / "data.yaml"


def run_dir(split: str, model: str) -> Path:
    return RUNS / f"{split}_{model}"
