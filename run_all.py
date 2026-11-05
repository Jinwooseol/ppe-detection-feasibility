"""Reproduce everything: data -> site-condition split -> fine-tune -> evaluate -> TensorRT -> README/PDF.

    python run_all.py                      # full run (each finished step is skipped on re-run)
    python run_all.py --only report        # rebuild figures, README tables and the PDF

Heavy files go to %PPE_WORK% (default: ~/ml-work/ppe-detection-feasibility); results, figures,
the trained weights and the PDF are written into this repository.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import ppe
from ppe.common import WORK, ensure_dirs

STEPS = ["data", "split", "train", "evaluate", "deploy", "report"]


def weights_path(name: str) -> Path:
    from ultralytics.utils.downloads import attempt_download_asset
    p = WORK / "weights" / f"{name}.pt"
    p.parent.mkdir(parents=True, exist_ok=True)
    return Path(attempt_download_asset(str(p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo26s")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--only", choices=STEPS, nargs="*", help="run only these steps")
    ap.add_argument("--no-random-baseline", action="store_true", help="skip the leaky random-split model")
    a = ap.parse_args()
    steps = a.only or STEPS

    ensure_dirs()
    os.chdir(WORK)                      # stray downloads (fonts, AMP check weights) land in the work dir
    ppe.quiet_ultralytics()
    base = weights_path(a.model)
    splits = ["group"] + ([] if a.no_random_baseline else ["random"])

    from ppe import data, deploy, evaluate, report, split, train
    if "data" in steps:
        data.prepare()
    if "split" in steps:
        print(split.make_splits())
    best = {}
    for s in splits:
        best[s] = train.train(s, base, a.epochs, batch=a.batch, workers=a.workers) if "train" in steps \
            else WORK / "runs" / f"{s}_{a.model}" / "weights" / "best.pt"
    if "evaluate" in steps:
        for s in splits:
            evaluate.evaluate(best[s], s)
    if "deploy" in steps:
        deploy.deploy(best["group"], "group")
    if "report" in steps:
        report.build()


if __name__ == "__main__":
    main()
