"""Fine-tune a COCO-pretrained YOLO26 detector on one split.

Training runs in a child process: after a CUDA fault the CUDA context is unusable, so a retry
needs a fresh process. A retry waits a minute, resumes from last.pt, and the number of attempts is
recorded in done.json. On the Windows development PC the faults (cuDNN execution failure, CUDA
out of memory) appeared only while other PyTorch processes ran next to training; each dataloader
worker maps the CUDA libraries, and together they exhausted the page-file-backed commit limit.
Run nothing else that imports torch while training.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from .common import REPO, RUNS, SEED, WORK, data_yaml, load_json, run_dir, save_json


def _fit(split: str, weights: str, epochs: int, imgsz: int, batch: int, workers: int, resume: bool) -> None:
    import ppe
    ppe.quiet_ultralytics()
    from ultralytics import YOLO
    if resume:
        YOLO(str(run_dir(split, Path(weights).stem) / "weights" / "last.pt")).train(resume=True)
        return
    YOLO(weights).train(
        data=str(data_yaml(split)), epochs=epochs, imgsz=imgsz, batch=batch, workers=workers,
        seed=SEED, deterministic=True, patience=max(15, epochs // 4), close_mosaic=10, cos_lr=True,
        project=str(RUNS), name=f"{split}_{Path(weights).stem}", exist_ok=True, plots=True, amp=True,
    )


def train(split: str, weights: Path, epochs: int, imgsz: int = 640, batch: int = 16,
          workers: int = 8, retries: int = 2) -> Path:
    d = run_dir(split, weights.stem)
    best, last, done = d / "weights" / "best.pt", d / "weights" / "last.pt", d / "done.json"
    if done.exists():
        return best
    if d.exists():
        shutil.rmtree(d)                 # a partial run from an earlier, interrupted invocation
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1"}
    attempts, resume = 0, False
    while True:
        attempts += 1
        cmd = [sys.executable, "-m", "ppe.train", split, str(weights), str(epochs), str(imgsz), str(batch), str(workers)]
        rc = subprocess.call(cmd + (["--resume"] if resume else []), cwd=WORK, env=env)
        if rc == 0:
            break
        if attempts > retries or not last.exists():
            raise RuntimeError(f"training {split} failed after {attempts} attempt(s)")
        print(f"training {split} exited with {rc}; resuming from {last} in 60 s")
        time.sleep(60)
        resume = True
    save_json({"attempts": attempts, "resumed": attempts > 1, "epochs": epochs, "imgsz": imgsz,
               "batch": batch, "weights": weights.name}, done)
    return best


def training_info(split: str, model: str) -> dict:
    d = run_dir(split, model)
    info = load_json(d / "done.json") if (d / "done.json").exists() else {}
    csv = d / "results.csv"
    if csv.exists():
        rows = [r.split(",") for r in csv.read_text().strip().splitlines()]
        head, body = rows[0], rows[1:]
        info["epochs_run"] = len(body)
        info["train_minutes"] = round(float(body[-1][head.index("time")]) / 60, 1)
        k = head.index("metrics/mAP50-95(B)")
        info["best_epoch"] = 1 + max(range(len(body)), key=lambda i: float(body[i][k]))
    return info


if __name__ == "__main__":
    a = sys.argv[1:]
    _fit(a[0], a[1], int(a[2]), int(a[3]), int(a[4]), int(a[5]), "--resume" in a)
