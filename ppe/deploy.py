"""Export to ONNX / TensorRT and measure speed and accuracy of each format on the same test split."""
from __future__ import annotations

import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

from .common import MODELS, REPO, RESULTS, RUNS, SPLITS, WORK, data_yaml, load_json, save_json

IMGSZ = 640


def _export(src_pt: Path, sub: str, **kw) -> Path:
    d = RUNS / "deploy" / sub
    d.mkdir(parents=True, exist_ok=True)
    pt = d / "best.pt"
    if not pt.exists():
        shutil.copy2(src_pt, pt)
    target = pt.with_suffix(".engine" if kw["format"] == "engine" else ".onnx")
    failed = d / "export_failed.txt"      # delete it to retry a failed build
    if failed.exists():
        raise RuntimeError(failed.read_text())
    if not target.exists():
        _child("export", {"pt": str(pt), "kw": kw})
        if not target.exists():
            if not failed.exists():
                failed.write_text("export produced no file")
            raise RuntimeError(failed.read_text())
    return target


def _child(task: str, payload: dict) -> int:
    """Run one export or one format's benchmark in a fresh process.

    Measured in one long-lived process, GPU memory piled up across formats (PyTorch's cache, the
    TensorRT contexts, cuDNN workspaces) until all 12 GB were taken and timings fell apart, so
    every format starts from an empty GPU.
    """
    env = {**os.environ, "PYTHONPATH": str(REPO), "PYTHONUTF8": "1"}
    return subprocess.call([sys.executable, "-m", "ppe.deploy", task, json.dumps(payload)], cwd=WORK, env=env)


def _export_child(pt: str, kw: dict) -> None:
    import ppe
    ppe.quiet_ultralytics()
    from ultralytics import YOLO
    try:
        YOLO(pt).export(imgsz=IMGSZ, device=0, **kw)
    except Exception as e:
        (Path(pt).parent / "export_failed.txt").write_text(f"{type(e).__name__}: {e}")
        raise


def _bench_child(p: dict) -> None:
    import ppe
    ppe.quiet_ultralytics()
    w = Path(p["weights"])
    images = (SPLITS / p["split"] / "test.txt").read_text().split()[:p["n_images"]]
    row = {"file": w.name, "end_to_end_b1": end_to_end(w, images, p["half"]),
           "model_only_b1": model_throughput(w, 1, p["half"])}
    if p.get("b8"):
        row["model_only_b8"] = model_throughput(Path(p["b8"]), 8, p["half"])
    row["accuracy"] = accuracy(w, p["split"], p["half"], p["val_batch"], f"deploy_val_{p['name']}")  # last: it fills GPU memory
    save_json(row, Path(p["out"]))


def gpu_info() -> dict:
    q = "name,driver_version,power.limit,power.max_limit,clocks.max.sm,temperature.gpu"
    out = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip()
    info = dict(zip(q.split(","), [x.strip() for x in out.split(",")]))
    try:
        ps = subprocess.run(["powershell", "-NoProfile", "-Command",
                             "(Get-CimInstance Win32_Battery).BatteryStatus"], capture_output=True, text=True)
        info["on_ac_power"] = ps.stdout.strip() == "2"
    except OSError:
        pass
    import tensorrt
    import torch
    info.update(torch=torch.__version__, cuda=torch.version.cuda, tensorrt=tensorrt.__version__)
    return info


def model_throughput(weights: Path, batch: int, half: bool, rounds: int = 5, n_iter: int = 100) -> dict:
    """Model-only latency on a fixed GPU tensor (no JPEG decode, no post-processing).

    PyTorch eager at batch 1 is bound by Python and kernel-launch overhead and swung 2x between
    runs on the laptop, so we time several rounds and keep the median round and the range.
    """
    import torch
    from ultralytics.nn.autobackend import AutoBackend
    dev = torch.device("cuda:0")
    m = AutoBackend(str(weights), device=dev, fp16=half, verbose=False)
    m.eval()
    x = torch.rand(batch, 3, IMGSZ, IMGSZ, device=dev)
    fp16 = getattr(getattr(m, "backend", m), "fp16", half)
    x = x.half() if fp16 else x            # an engine's input dtype decides, not the request
    means = []
    with torch.inference_mode():
        for _ in range(100):
            m(x)
        torch.cuda.synchronize()
        for _ in range(rounds):
            times = []
            for _ in range(n_iter):
                t0 = time.perf_counter()
                m(x)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
            means.append(1000 * statistics.mean(times))
    med = statistics.median(means)
    return {"batch": batch, "latency_ms_mean": round(med, 3),
            "latency_ms_round_range": [round(min(means), 3), round(max(means), 3)],
            "images_per_s": round(batch * 1000 / med, 1)}


def end_to_end(weights: Path, images: list[str], half: bool) -> dict:
    """Ultralytics predict on real test JPEGs at batch 1.

    total_ms = letterbox + inference + post-processing as Ultralytics reports them (frame already
    decoded, as with a video stream); pipeline_fps also counts reading and decoding the JPEG.
    """
    from ultralytics import YOLO
    q = 16 if half else None
    m = YOLO(str(weights), task="detect")
    kw = dict(imgsz=IMGSZ, quantize=q, rect=False, stream=True, verbose=False, conf=0.25)  # fixed 640x640 like the engines
    # PyTorch eager on this laptop ran 50-100x slower for the first rounds of some processes, then
    # settled; warm up until two rounds of 50 frames agree within 20 %.
    prev, warmup = None, 0
    while warmup < 12:
        warmup += 1
        cur = statistics.median(r.speed["inference"] for r in m.predict(images[:50], **kw))
        if prev is not None and abs(cur - prev) <= 0.2 * prev:
            break
        prev = cur
    rounds = []
    for _ in range(3):                    # keep the median of three rounds
        t0 = time.perf_counter()
        sp = [r.speed for r in m.predict(images, **kw)]
        wall = time.perf_counter() - t0
        mean = {k: statistics.mean(s[k] for s in sp) for k in ("preprocess", "inference", "postprocess")}
        rounds.append((sum(mean.values()), mean, len(sp) / wall))
    total, mean, pipe = sorted(rounds, key=lambda r: r[0])[1]
    return {**{f"{k}_ms": round(v, 2) for k, v in mean.items()}, "total_ms": round(total, 2),
            "fps": round(1000 / total, 1), "pipeline_fps_incl_jpeg_decode": round(pipe, 1),
            "warmup_rounds": warmup, "round_totals_ms": [round(r[0], 2) for r in rounds]}


def accuracy(weights: Path, split: str, half: bool, batch: int, name: str) -> dict:
    from ultralytics import YOLO
    m = YOLO(str(weights), task="detect").val(data=str(data_yaml(split)), split="test", imgsz=IMGSZ,
                                             batch=batch, quantize=16 if half else None, conf=0.001, iou=0.7, plots=False,
                                             project=str(RUNS), name=name, exist_ok=True, verbose=False)
    return {"mAP50": round(float(m.box.map50), 4), "mAP50_95": round(float(m.box.map), 4)}


def deploy(best_pt: Path, split: str = "group", n_images: int = 300) -> dict:
    out = RESULTS / "deployment.json"
    if out.exists():
        return load_json(out)
    onnx = _export(best_pt, "onnx", format="onnx", simplify=True)
    # (name, export kwargs, FP16 input). A format that cannot be built here is recorded, not fatal.
    variants = [
        ("tensorrt_fp32", dict(format="engine", workspace=4), False),
        ("tensorrt_fp16", dict(format="engine", quantize=16, workspace=4), True),
        ("tensorrt_int8", dict(format="engine", quantize=8, data=str(data_yaml(split)), workspace=4), False),
    ]
    engines, failed = {}, {}
    for name, kw, half in variants:
        try:
            engines[name] = (_export(best_pt, name, **kw), half)
        except Exception as e:  # e.g. TensorRT 11 needs NVIDIA ModelOpt for FP16 / INT8
            failed[name] = f"{type(e).__name__}: {e}"
    b8 = None
    if "tensorrt_fp16" in engines:
        try:
            b8 = _export(best_pt, "tensorrt_fp16_b8", format="engine", quantize=16, batch=8, workspace=4)
        except Exception as e:
            failed["tensorrt_fp16_b8"] = f"{type(e).__name__}: {e}"

    rows = {}
    runs = [("pytorch_fp32", best_pt, False, 16, None), ("pytorch_fp16", best_pt, True, 16, best_pt)]
    runs += [(name, w, half, 1, b8 if name == "tensorrt_fp16" else None) for name, (w, half) in engines.items()]
    for name, w, half, vb, w8 in runs:
        tmp = RUNS / "deploy" / f"bench_{name}.json"
        tmp.unlink(missing_ok=True)
        _child("bench", {"name": name, "weights": str(w), "half": half, "val_batch": vb, "split": split,
                         "n_images": n_images, "b8": str(w8) if w8 else None, "out": str(tmp)})
        if not tmp.exists():
            raise RuntimeError(f"benchmark of {name} failed; see the log above")
        rows[name] = load_json(tmp)
    result = {"imgsz": IMGSZ, "gpu": gpu_info(), "formats": rows, "not_built": failed,
              "n_latency_images": n_images, "onnx_mb": round(onnx.stat().st_size / 1e6, 1)}
    MODELS.mkdir(exist_ok=True)
    shutil.copy2(best_pt, MODELS / "ppe_yolo26s_sh17_group.pt")
    shutil.copy2(onnx, MODELS / "ppe_yolo26s_sh17_group.onnx")
    save_json(result, out)
    return result


if __name__ == "__main__":
    task, payload = sys.argv[1], json.loads(sys.argv[2])
    if task == "export":
        _export_child(payload["pt"], payload["kw"])
    else:
        _bench_child(payload)
