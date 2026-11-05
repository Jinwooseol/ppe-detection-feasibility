"""Per-class evaluation, error analysis and alert-budget operating points.

Ultralytics' own validator gives the reference mAP. On top of it we keep every prediction
(conf >= 0.001), match it to ground truth ourselves (IoU >= 0.5, greedy by confidence) and
derive what a safety team needs: per-class precision / recall at a chosen threshold, false
alerts per 1,000 frames, and *why* each false alert or miss happened.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .common import CID, DATA, LOOKALIKE, NAMES, RESULTS, RUNS, SPLITS, data_yaml, load_json, local, portable, read_labels, save_json

IOU_MATCH = 0.5
BUDGETS = (5, 20, 50)          # false alerts per 1,000 frames
MAIN_BUDGET = 20
SIZE_EDGES = (16, 32, 96)      # longest side in pixels at the 640 network input


# ---------------------------------------------------------------- geometry
def xywhn_to_xyxy(b):
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-12)


def size_bucket(box_xyxy, w, h, imgsz=640):
    side = max((box_xyxy[2] - box_xyxy[0]) * w, (box_xyxy[3] - box_xyxy[1]) * h) * imgsz / max(w, h)
    return ["tiny", "small", "medium", "large"][int(np.searchsorted(SIZE_EDGES, side, side="right"))]


# ---------------------------------------------------------------- inference
def load_val_predictions(path: Path, images: list[Path], sizes):
    """Read the validator's own predictions, so our analysis and its mAP use the same boxes."""
    import json
    by_file = {}
    for d in json.loads(path.read_text()):
        by_file.setdefault(d["file_name"], []).append(d)
    preds = []
    for p, (w, h) in zip(images, sizes):
        rows = by_file.get(p.name, [])
        b = np.array([d["bbox"] for d in rows], np.float64).reshape(-1, 4)
        xyxy = np.c_[b[:, 0] / w, b[:, 1] / h, (b[:, 0] + b[:, 2]) / w, (b[:, 1] + b[:, 3]) / h]
        preds.append((xyxy, np.array([d["category_id"] - 1 for d in rows], int),   # ids are 1-based
                      np.array([d["score"] for d in rows], np.float64)))
    return preds


def ultralytics_val(weights: Path, split: str, name: str, imgsz=640, batch=16):
    from ultralytics import YOLO
    m = YOLO(str(weights)).val(data=str(data_yaml(split)), split="test", imgsz=imgsz, batch=batch,
                               conf=0.001, iou=0.7, plots=True, save_json=True, project=str(RUNS),
                               name=name, exist_ok=True, verbose=False)
    per = {}
    for k, c in enumerate(m.box.ap_class_index):
        per[NAMES[int(c)]] = {"P": float(m.box.p[k]), "R": float(m.box.r[k]),
                              "mAP50": float(m.box.ap50[k]), "mAP50_95": float(m.box.ap[k])}
    return {"mAP50": float(m.box.map50), "mAP50_95": float(m.box.map), "P": float(m.box.mp),
            "R": float(m.box.mr), "per_class": per, "speed_ms": dict(m.speed),
            "save_dir": str(m.save_dir)}


# ---------------------------------------------------------------- matching
def match(gts, preds, sizes):
    """Return one record per prediction and per ground-truth box.

    pred record: img, cls, conf, tp, kind-of-error inputs (best IoU with same class, with an
    already matched same-class GT, best other-class GT and its class).
    """
    P, G = [], []
    for i, ((gb, gc), (pb, pc, ps), (w, h)) in enumerate(zip(gts, preds, sizes)):
        ious = iou_matrix(pb, gb)
        matched = np.full(len(gb), -1)
        order = np.argsort(-ps, kind="stable")
        tp = np.zeros(len(pb), bool)
        for j in order:
            same = np.where(gc == pc[j])[0]
            if len(same):
                free = same[matched[same] < 0]
                if len(free):
                    k = free[np.argmax(ious[j, free])]
                    if ious[j, k] >= IOU_MATCH:
                        matched[k] = j
                        tp[j] = True
        for j in range(len(pb)):
            same = gc == pc[j]
            other = ~same
            best_same = ious[j, same].max() if same.any() else 0.0
            if other.any():
                ko = np.where(other)[0][np.argmax(ious[j, other])]
                best_other, other_cls = ious[j, ko], int(gc[ko])
            else:
                best_other, other_cls = 0.0, -1
            P.append((i, int(pc[j]), float(ps[j]), bool(tp[j]), float(best_same), float(best_other), other_cls))
        gg = iou_matrix(gb, gb)
        for k in range(len(gb)):
            # what else was predicted on this box (for misses). Classes that are also labelled here
            # (a helmet on top of a head box) are correct detections, not confusions.
            here = set(gc[gg[k] >= IOU_MATCH].tolist())
            cand = np.where(~np.isin(pc, list(here)) & (ious[:, k] >= IOU_MATCH))[0] if len(pb) else np.array([], int)
            alt = [(int(pc[j]), float(ps[j]), pb[j].round(5).tolist()) for j in cand]
            loc = [float(ps[j]) for j in np.where((pc == gc[k]) & (ious[:, k] >= 0.1) & (ious[:, k] < IOU_MATCH))[0]] if len(pb) else []
            G.append(dict(img=i, cls=int(gc[k]), box=gb[k].round(5).tolist(), size=size_bucket(gb[k], w, h),
                          tp_conf=float(ps[matched[k]]) if matched[k] >= 0 else None, alt=alt, loc=loc))
    return P, G


def curve(P, n_gt, n_img, c):
    rows = sorted((p for p in P if p[1] == c), key=lambda p: -p[2])
    conf = np.array([p[2] for p in rows])
    tp = np.cumsum([p[3] for p in rows]) if rows else np.zeros(0)
    fp = np.cumsum([not p[3] for p in rows]) if rows else np.zeros(0)
    rec = tp / max(n_gt, 1)
    prec = tp / np.maximum(tp + fp, 1)
    return conf, rec, prec, fp / n_img * 1000


def ap101(rec, prec):
    if len(rec) == 0:
        return 0.0
    mrec = np.r_[0, rec, 1]
    mpre = np.flip(np.maximum.accumulate(np.flip(np.r_[1, prec, 0])))
    x = np.linspace(0, 1, 101)
    return float(np.trapezoid(np.interp(x, mrec, mpre), x))


def at_threshold(conf, rec, prec, fpk, t):
    k = np.searchsorted(-conf, -t, side="right") - 1      # last index with conf >= t
    if k < 0:
        return dict(thr=t, P=1.0, R=0.0, FP_per_1k=0.0)
    return dict(thr=float(t), P=float(prec[k]), R=float(rec[k]), FP_per_1k=float(fpk[k]))


def operating_points(conf, rec, prec, fpk):
    out = {}
    for b in BUDGETS:
        ok = np.where(fpk <= b)[0]
        out[f"budget_{b}"] = at_threshold(conf, rec, prec, fpk, conf[ok[-1]]) if len(ok) else dict(thr=None, P=None, R=0.0, FP_per_1k=0.0)
    if len(conf):
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        out["max_f1"] = at_threshold(conf, rec, prec, fpk, conf[int(np.argmax(f1))])
        out["conf_0.25"] = at_threshold(conf, rec, prec, fpk, 0.25)
    return out


def error_breakdown(P, G, thr, n_img, images):
    """FP and FN reasons per class at per-class thresholds, plus every look-alike miss as a case."""
    fp = {n: {"duplicate": 0, "wrong_class": 0, "localisation": 0, "background": 0, "confused_with": {}} for n in NAMES}
    for i, c, s, tp, best_same, best_other, ocls in P:
        t = thr.get(c)
        if t is None or s < t or tp:
            continue
        d = fp[NAMES[c]]
        if best_same >= IOU_MATCH:
            d["duplicate"] += 1
        elif best_other >= IOU_MATCH:
            d["wrong_class"] += 1
            d["confused_with"][NAMES[ocls]] = d["confused_with"].get(NAMES[ocls], 0) + 1
        elif max(best_same, best_other) >= 0.1:
            d["localisation"] += 1
        else:
            d["background"] += 1
    fn = {n: {"detected": 0, "as_other_class": 0, "poor_box": 0, "missed": 0, "predicted_as": {},
              "recall_by_size": {}} for n in NAMES}
    size_tot, cases = {}, []
    for g in G:
        c, t = g["cls"], thr.get(g["cls"])
        d = fn[NAMES[c]]
        key = (c, g["size"])
        size_tot[key] = size_tot.get(key, [0, 0])
        size_tot[key][0] += 1
        if t is not None and g["tp_conf"] is not None and g["tp_conf"] >= t:
            d["detected"] += 1
            size_tot[key][1] += 1
            continue
        alt = [x for x in g["alt"] if thr.get(x[0]) is not None and x[1] >= thr[x[0]]]
        if alt:
            a, s, box = max(alt, key=lambda x: x[1])
            d["as_other_class"] += 1
            d["predicted_as"][NAMES[a]] = d["predicted_as"].get(NAMES[a], 0) + 1
            cases.append({"label": NAMES[c], "predicted": NAMES[a], "conf": round(s, 3),
                          "image": str(images[g["img"]]), "gt": g["box"], "pred": box})
        elif t is not None and any(s >= t for s in g["loc"]):
            d["poor_box"] += 1
        else:
            d["missed"] += 1
    for (c, sz), (n, k) in size_tot.items():
        fn[NAMES[c]]["recall_by_size"][sz] = {"n": n, "recall": round(k / n, 4)}
    for n in NAMES:
        fp[n]["per_1k_frames"] = round(sum(v for k, v in fp[n].items() if isinstance(v, int)) / n_img * 1000, 2)
    return fp, fn, cases


def test_set(split: str):
    """Test images, their sizes and ground truth (xyxy normalised, class ids)."""
    from PIL import Image
    images = [Path(p) for p in (SPLITS / split / "test.txt").read_text().split()]
    sizes = [Image.open(p).size for p in images]
    gts = []
    for p in images:
        lab = read_labels(DATA / "labels" / f"{p.stem}.txt")
        gts.append((xywhn_to_xyxy([b[1:] for b in lab]), np.array([b[0] for b in lab], int)))
    return images, sizes, gts


def confusion_cases(result: dict, gt_name: str, pred_name: str):
    """True `gt_name` boxes missed at the operating point and called `pred_name` instead, with no
    `pred_name` box labelled there: exactly the cases counted in the look-alike table."""
    return [c for c in result["confusions"] if c["label"] == gt_name and c["predicted"] == pred_name]


def lookalike_table(fn, n_gt):
    rows = []
    for a, b in LOOKALIKE:
        for x, y in ((a, b), (b, a)):
            k = fn[x]["predicted_as"].get(y, 0)
            rows.append(dict(gt=x, predicted_as=y, count=k, n_gt=n_gt[x], rate=round(k / max(n_gt[x], 1), 4)))
    return rows


# ---------------------------------------------------------------- main
def evaluate(weights: Path, split: str, tag: str | None = None) -> dict:
    tag = tag or split
    out_dir = RESULTS / tag
    out = out_dir / "metrics.json"
    if out.exists():
        return load_json(out)
    images, sizes, gts = test_set(split)
    n_img = len(images)
    n_gt = {n: int(sum((gc == c).sum() for _, gc in gts)) for c, n in enumerate(NAMES)}

    ref_path = out_dir / "ultralytics_val.json"     # the only step that needs the GPU
    if ref_path.exists() and load_json(ref_path)["weights"] == portable(str(weights)):
        ref = load_json(ref_path)
    else:
        ref = {**ultralytics_val(weights, split, f"{tag}_test_val"), "weights": str(weights)}
        save_json(ref, ref_path)
    preds = load_val_predictions(local(ref["save_dir"]) / "predictions.json", images, sizes)
    P, G = match(gts, preds, sizes)

    per_class, ops, curves = {}, {}, {}
    for c, n in enumerate(NAMES):
        conf, rec, prec, fpk = curve(P, n_gt[n], n_img, c)
        ops[n] = operating_points(conf, rec, prec, fpk)
        per_class[n] = {"n_gt": n_gt[n], "n_img": int(sum((gc == c).any() for _, gc in gts)),
                        "AP50_own": round(ap101(rec, prec), 4), **{k: round(v, 4) for k, v in ref["per_class"].get(n, {}).items()}}
        # keep a thinned curve for plots
        keep = np.unique(np.linspace(0, max(len(conf) - 1, 0), min(len(conf), 400)).astype(int)) if len(conf) else []
        curves[n] = {"conf": conf[keep].round(4).tolist(), "R": rec[keep].round(4).tolist(),
                     "P": prec[keep].round(4).tolist(), "FP_per_1k": fpk[keep].round(2).tolist()}

    thr = {CID[n]: ops[n][f"budget_{MAIN_BUDGET}"]["thr"] for n in NAMES}
    fp, fn, cases = error_breakdown(P, G, thr, n_img, images)
    result = {
        "split": split, "weights": str(weights), "n_test_images": n_img, "reference": ref,
        "per_class": per_class, "operating_points": ops, "main_budget_per_1k": MAIN_BUDGET,
        "false_positives": fp, "misses": fn, "lookalike": lookalike_table(fn, n_gt), "confusions": cases,
    }
    save_json(result, out)
    save_json(curves, out_dir / "curves.json")
    _write_csv(result, out_dir)
    return result


def _write_csv(r, d: Path):
    import csv
    with open(d / "per_class.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "test_images", "instances", "P_ultra", "R_ultra", "mAP50", "mAP50_95",
                    f"thr@{MAIN_BUDGET}/1k", f"P@{MAIN_BUDGET}/1k", f"R@{MAIN_BUDGET}/1k"])
        for n in NAMES:
            pc, op = r["per_class"][n], r["operating_points"][n][f"budget_{MAIN_BUDGET}"]
            w.writerow([n, pc["n_img"], pc["n_gt"], pc.get("P"), pc.get("R"), pc.get("mAP50"), pc.get("mAP50_95"),
                        op["thr"], op["P"], op["R"]])
    with open(d / "operating_points.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class", "point", "threshold", "precision", "recall", "false_alerts_per_1k_frames"])
        for n in NAMES:
            for k, op in r["operating_points"][n].items():
                w.writerow([n, k, op["thr"], op["P"], op["R"], op["FP_per_1k"]])
