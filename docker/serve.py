"""Containerised PPE inference: build a TensorRT engine for this GPU once, then run a source.

    python serve.py --source /data/in --out /data/out          # folder of images or a video file
    python serve.py --source rtsp://camera/stream --out /data/out --vid-stride 6

Writes one JSON line per frame with the detections that pass each class's operating-point
threshold (results/group/operating_points.csv, main budget), and prints the measured FPS.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

os.environ.setdefault("YOLO_AUTOINSTALL", "False")


def load_thresholds(path: Path, point: str) -> dict[str, float]:
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if row["point"] == point and row["threshold"] not in ("", "None"):
                out[row["class"]] = float(row["threshold"])
    return out


def engine_for_this_gpu(pt: Path, cache: Path, imgsz: int, half: bool) -> Path:
    """TensorRT engines are tied to the GPU and TensorRT version, so build on first start."""
    import torch
    from ultralytics import YOLO
    tag = torch.cuda.get_device_name(0).replace(" ", "_")
    engine = cache / f"{pt.stem}_{tag}_{imgsz}_{'fp16' if half else 'fp32'}.engine"
    if not engine.exists():
        cache.mkdir(parents=True, exist_ok=True)
        built = Path(YOLO(str(pt)).export(format="engine", imgsz=imgsz, quantize=16 if half else None, device=0))
        built.replace(engine)
    return engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="/app/models/ppe_yolo26s_sh17_group.pt")
    ap.add_argument("--thresholds", default="/app/operating_points.csv")
    ap.add_argument("--point", default="budget_20", help="operating point column, e.g. budget_5 / budget_20 / budget_50")
    ap.add_argument("--source", required=True)
    ap.add_argument("--out", default="/data/out")
    ap.add_argument("--cache", default="/app/cache")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--vid-stride", type=int, default=1, help="analyse every n-th video frame")
    ap.add_argument("--no-trt", action="store_true", help="run the PyTorch weights directly")
    a = ap.parse_args()

    from ultralytics import YOLO
    thr = load_thresholds(Path(a.thresholds), a.point)
    weights = Path(a.weights) if a.no_trt else engine_for_this_gpu(Path(a.weights), Path(a.cache), a.imgsz, True)
    model = YOLO(str(weights), task="detect")
    floor = min(thr.values()) if thr else 0.25

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n, t0 = 0, time.perf_counter()
    with open(out / "detections.jsonl", "w") as f:
        for r in model.predict(a.source, imgsz=a.imgsz, conf=floor, quantize=16, stream=True,
                               vid_stride=a.vid_stride, verbose=False):
            dets = []
            for (x0, y0, x1, y1), c, s in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
                name = r.names[int(c)]
                if s >= thr.get(name, 0.25):
                    dets.append({"class": name, "conf": round(s, 3), "xyxy": [round(v, 1) for v in (x0, y0, x1, y1)]})
            f.write(json.dumps({"frame": n, "source": r.path, "detections": dets}) + "\n")
            n += 1
    dt = time.perf_counter() - t0
    print(f"{n} frames in {dt:.1f} s -> {n / max(dt, 1e-9):.1f} FPS end to end ({weights.name})")


if __name__ == "__main__":
    main()
