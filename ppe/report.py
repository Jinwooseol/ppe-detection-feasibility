"""Build figures, markdown result tables (injected into README.md) and the one-page PDF."""
from __future__ import annotations

import re
import shutil
from pathlib import Path

from .common import FIGURES, LOOKALIKE, NAMES, REPO, RESULTS, RUNS, SPLITS, load_json, local
from .evaluate import BUDGETS, MAIN_BUDGET

BEGIN, END = "<!-- results:begin -->", "<!-- results:end -->"
KEY_CLASSES = ["helmet", "head", "gloves", "hands", "shoes", "foot", "safety-vest", "person"]


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}"


def num(x, d=2):
    return "–" if x is None else f"{x:.{d}f}"


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def split_md(s):
    g, r = s["group"], s["random"]
    rows = [
        ["images (train / val / test)", f"{g['train']['images']:,} / {g['val']['images']:,} / {g['test']['images']:,}",
         f"{r['train']['images']:,} / {r['val']['images']:,} / {r['test']['images']:,}"],
        ["photographers in test", f"{g['test']['photographers']}", f"{r['test']['photographers']}"],
        ["largest photographer's share of test", pct(g['test']['largest_photographer_share']) + "%",
         pct(r['test']['largest_photographer_share']) + "%"],
        ["test images whose photographer is also in train", pct(g["test_images_whose_photographer_is_in_train"]) + "%",
         pct(r["test_images_whose_photographer_is_in_train"]) + "%"],
        ["test images with a near-duplicate in train", pct(g["test_images_with_near_duplicate_in_train"]) + "%",
         pct(r["test_images_with_near_duplicate_in_train"]) + "%"],
    ]
    return table(["", "held-out photographers (used)", "random images (baseline)"], rows)


def headline_md(g, r):
    rows = [["held-out photographers", g["n_test_images"], num(g["reference"]["mAP50"], 3), num(g["reference"]["mAP50_95"], 3)]]
    if r:
        rows.append(["random images (leaky)", r["n_test_images"], num(r["reference"]["mAP50"], 3), num(r["reference"]["mAP50_95"], 3)])
    return table(["test set", "images", "mAP@0.5", "mAP@0.5:0.95"], rows)


def per_class_md(g, r):
    rows = []
    for n in sorted(NAMES, key=lambda n: -g["per_class"][n]["n_gt"]):
        pc, op = g["per_class"][n], g["operating_points"][n][f"budget_{MAIN_BUDGET}"]
        rmap = r["per_class"][n].get("mAP50") if r else None
        rows.append([n, pc["n_img"], pc["n_gt"], num(pc.get("mAP50"), 3), num(pc.get("mAP50_95"), 3),
                     num(rmap, 3), num(op["thr"]), pct(op["P"]), pct(op["R"])])
    return table(["class", "test images", "instances", "mAP@0.5", "mAP@0.5:0.95", "mAP@0.5 random split",
                  f"threshold @ {MAIN_BUDGET}/1k", "precision %", "recall %"], rows)


def lookalike_md(g):
    rows = []
    for row in g["lookalike"]:
        if row["count"]:
            rows.append([row["gt"], row["predicted_as"], row["count"], row["n_gt"], pct(row["rate"]) + "%"])
    return table(["true class", "predicted as", "boxes", "true boxes in test", "share of true boxes"], rows)


AUDIT_KINDS = ["label_missing", "label_rule_edge", "model_wrong", "unclear"]


def audit_tally(g) -> dict:
    """Match the manual audit (results/audit_verdicts.json) to the current confusion cases."""
    path = RESULTS / "audit_verdicts.json"
    if not path.exists():
        return {}
    a = load_json(path)
    verdict = {(c["image"], tuple(round(v, 3) for v in c["gt"])): c["verdict"] for c in a["cases"]}
    out = {}
    for gt, pred in [(x, y) for p in (("helmet", "head"), ("gloves", "hands"), ("shoes", "foot")) for x, y in (p, p[::-1])]:
        cases = [c for c in g["confusions"] if c["label"] == gt and c["predicted"] == pred]
        if cases:
            got = [verdict.get((Path(c["image"]).stem, tuple(round(v, 3) for v in c["gt"]))) for c in cases]
            out[(gt, pred)] = {"cases": len(cases), **{k: sum(v == k for v in got) for k in AUDIT_KINDS},
                               "not_audited": sum(v is None for v in got)}
    return out


def audit_md(tally):
    if not tally:
        return "_No manual audit recorded._"
    rows = [[f"{gt} → {pred}", t["cases"], *[t[k] for k in AUDIT_KINDS], t["not_audited"]]
            for (gt, pred), t in tally.items()]
    return table(["label → predicted", "cases", "item visible, not labelled", "edge of the label rule",
                  "model wrong", "unclear", "not audited"], rows)


def miss_md(g):
    rows = []
    for n in KEY_CLASSES:
        m = g["misses"][n]
        tot = m["detected"] + m["as_other_class"] + m["poor_box"] + m["missed"]
        if not tot:
            continue
        rs = m["recall_by_size"]
        size = " / ".join(f"{pct(rs[k]['recall'])}" if k in rs else "–" for k in ("tiny", "small", "medium", "large"))
        rows.append([n, tot, pct(m["detected"] / tot), pct(m["as_other_class"] / tot), pct(m["poor_box"] / tot),
                     pct(m["missed"] / tot), size])
    return table(["class", "true boxes", "found %", "called look-alike %", "poor box %", "not detected %",
                  "recall % by size: tiny / small / medium / large"], rows)


def ops_md(g):
    rows = []
    for n in KEY_CLASSES:
        ops = g["operating_points"][n]
        cells = [n]
        for b in BUDGETS:
            o = ops[f"budget_{b}"]
            cells.append("–" if o["thr"] is None else f"{num(o['thr'])} · P {pct(o['P'])} · R {pct(o['R'])}")
        o = ops.get("conf_0.25", {})
        cells.append(f"P {pct(o.get('P'))} · R {pct(o.get('R'))} · {num(o.get('FP_per_1k'), 0)}/1k" if o else "–")
        rows.append(cells)
    return table(["class"] + [f"≤ {b} false alerts / 1k frames" for b in BUDGETS] + ["default conf 0.25"], rows)


def deploy_md(d):
    label = {"pytorch_fp32": "PyTorch FP32", "pytorch_fp16": "PyTorch FP16", "tensorrt_fp32": "TensorRT FP32",
             "tensorrt_fp16": "TensorRT FP16", "tensorrt_int8": "TensorRT INT8"}
    rows = []
    for k, v in d["formats"].items():
        e, m = v["end_to_end_b1"], v["model_only_b1"]
        b8 = v.get("model_only_b8", {}).get("images_per_s")
        rt = e.get("round_totals_ms") or [e["total_ms"]]
        if max(rt) > 1.5 * min(rt):        # rounds disagree: show the spread, not a number
            frame, fps, pipe = f"unstable {min(rt):.1f}–{max(rt):.0f}", "–", "–"
        else:
            frame, fps, pipe = num(e["total_ms"], 1), num(e["fps"], 0), num(e.get("pipeline_fps_incl_jpeg_decode"), 0)
        rows.append([label[k], num(v["accuracy"]["mAP50"], 3), num(v["accuracy"]["mAP50_95"], 3),
                     num(m["latency_ms_mean"]), num(m["images_per_s"], 0), "–" if b8 is None else num(b8, 0),
                     frame, fps, pipe])
    md = table(["format", "mAP@0.5", "mAP@0.5:0.95", "model latency ms (b1)", "model images/s (b1)",
                "model images/s (b8)", "per-frame ms (b1)", "FPS (b1)", "FPS incl. JPEG decode"], rows)
    if d.get("not_built"):
        md += "\n\nNot built in this run: " + "; ".join(
            f"{label.get(k, k)} ({v.splitlines()[0][:120]})" for k, v in d["not_built"].items())
    return md


def training_md(results: dict) -> str:
    from .train import training_info
    rows = []
    for name, res in results.items():
        if not res:
            continue
        run = Path(res["weights"]).parents[1]
        split, model = run.name.split("_", 1)
        t = training_info(split, model)
        rows.append([name, t.get("epochs_run"), t.get("best_epoch"), t.get("train_minutes"),
                     "yes" if t.get("resumed") else "no"])
    return table(["model", "epochs", "best epoch (val mAP@0.5:0.95)", "train minutes", "resumed after a fault"], rows)


def build() -> None:
    from . import figures, pdf
    g = load_json(RESULTS / "group" / "metrics.json")
    r = load_json(RESULTS / "random" / "metrics.json") if (RESULTS / "random" / "metrics.json").exists() else None
    s = load_json(SPLITS / "split_summary.json")
    d = load_json(RESULTS / "deployment.json") if (RESULTS / "deployment.json").exists() else None
    curves = load_json(RESULTS / "group" / "curves.json")
    shutil.copy2(SPLITS / "split_summary.json", RESULTS / "split_summary.json")
    cm = local(g["reference"]["save_dir"]) / "confusion_matrix_normalized.png"
    if cm.exists():
        shutil.copy2(cm, FIGURES / "confusion_matrix_normalized.png")
    figures.make_all(g, r, curves)
    tally = audit_tally(g)

    gpu = d["gpu"] if d else {}
    blocks = [
        "### Split", split_md(s),
        "### Training (YOLO26s from COCO weights, 640 px, batch 16, seed 0)",
        training_md({"held-out photographers": g, "random images": r}),
        "### Headline", headline_md(g, r),
        "### Per class (held-out photographers)", per_class_md(g, r),
        "### Look-alike confusions at the operating point", lookalike_md(g),
        "### Manual audit of the look-alike cases (one reviewer, sheets in `results/audit/`)", audit_md(tally),
        "### Where the misses come from", miss_md(g),
        "### Operating points (threshold · precision · recall)", ops_md(g),
    ]
    if d:
        blocks += [f"### Deployment ({gpu.get('name')}, TensorRT {gpu.get('tensorrt')}, {d['imgsz']} px)", deploy_md(d)]
    md = "\n\n".join(blocks)
    (RESULTS / "RESULTS.md").write_text(md + "\n", encoding="utf-8")
    readme = REPO / "README.md"
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        if BEGIN in txt and END in txt:
            txt = re.sub(re.escape(BEGIN) + ".*?" + re.escape(END), lambda _: f"{BEGIN}\n{md}\n{END}", txt, flags=re.S)
            readme.write_text(txt, encoding="utf-8")
    pdf.build(g, r, s, d, tally)
