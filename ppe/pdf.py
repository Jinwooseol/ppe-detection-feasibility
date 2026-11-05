"""One-page feasibility study outline (shipyard PPE / fall detection), filled with pilot numbers."""
from __future__ import annotations

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .common import DOCS, FIGURES
from .evaluate import MAIN_BUDGET

INK, INK2, RULE, BOX, ACCENT = (colors.HexColor(c) for c in ("#0b0b0b", "#52514e", "#d9d8d3", "#f1f0ec", "#2a78d6"))


def _styles():
    base = dict(fontName="Helvetica", textColor=INK, fontSize=7.9, leading=9.9)
    return {
        "title": ParagraphStyle("t", **{**base, "fontName": "Helvetica-Bold", "fontSize": 14, "leading": 17}),
        "sub": ParagraphStyle("s", **{**base, "textColor": INK2, "fontSize": 8.2, "leading": 10.5}),
        "h": ParagraphStyle("h", **{**base, "fontName": "Helvetica-Bold", "fontSize": 9, "leading": 11, "textColor": ACCENT}),
        "b": ParagraphStyle("b", **base),
        "ev": ParagraphStyle("e", **{**base, "fontSize": 7.4, "leading": 9.2, "textColor": INK2}),
        "foot": ParagraphStyle("f", **{**base, "fontSize": 6.8, "leading": 8.4, "textColor": INK2}),
    }


def _pct(x):
    return f"{100 * x:.0f}%"


def build(g: dict, r: dict | None, s: dict, d: dict | None, tally: dict | None = None) -> None:
    st = _styles()
    tally = tally or {}
    op = lambda n: g["operating_points"][n][f"budget_{MAIN_BUDGET}"]
    gm, rm = g["reference"]["mAP50"], (r["reference"]["mAP50"] if r else None)
    test = s["group"]["test"]
    leak = s["random"]["test_images_whose_photographer_is_in_train"]
    hh, fs = tally.get(("head", "helmet")), tally.get(("foot", "shoes"))
    rules_ev = ("In SH17 a helmet is labelled on top of a head box, while gloves and bare hands are exclusive. "
                + (f"{fs['label_rule_edge']} of the {fs['cases']} 'bare foot called shoes' errors were socked feet, "
                   "which SH17 labels as foot: " if fs else "")
                + "the rule book decides what counts as an error.")
    site_ev = f"Test: {test['images']:,} photos from {test['photographers']} photographers never seen in training. "
    if rm is not None:
        gap = {n: r["per_class"][n]["mAP50"] - g["per_class"][n]["mAP50"] for n in g["per_class"]}
        top = sorted(gap, key=lambda n: -gap[n])[:3]
        common = [n for n in g["per_class"] if g["per_class"][n]["n_gt"] > 1500]
        site_ev += (f"A random image split of the same size, where {_pct(leak)} of test photos share a photographer "
                    f"with training, scored {(rm - gm) * 100:.1f} points higher mAP@0.5 ({rm:.3f} vs {gm:.3f}). "
                    f"The gain sits in rare classes ({', '.join(f'{n} +{gap[n] * 100:.0f}' for n in top)}); "
                    f"the {len(common)} most frequent classes moved by at most "
                    f"{max(abs(gap[n]) for n in common) * 100:.1f} points.")
    op_ev = (f"At ≤ {MAIN_BUDGET} false alerts per 1,000 frames: bare hands recall {_pct(op('hands')['R'])} "
             f"(precision {_pct(op('hands')['P'])}), helmet recall {_pct(op('helmet')['R'])} "
             f"(precision {_pct(op('helmet')['P'])}). "
             + (f"Of the {hh['cases']} helmets detected on heads labelled bare, {hh['label_missing']} were real hard "
                "hats nobody labelled, so a label audit moves the operating point. " if hh else "")
             + "At one analysed frame every 10 s, an 8-hour shift is ~2,900 frames per camera.")
    if d:
        trt = d["formats"]["tensorrt_fp16"]
        b8 = trt.get("model_only_b8", {}).get("images_per_s", 0)
        hw = (f"TensorRT FP16 on a laptop RTX 5070 Ti, 640 px: {trt['model_only_b1']['latency_ms_mean']:.1f} ms "
              f"model latency, {trt['end_to_end_b1']['fps']:.0f} FPS per frame with resize and post-processing at "
              f"batch 1, mAP@0.5 {trt['accuracy']['mAP50']:.3f} (PyTorch FP32 "
              f"{d['formats']['pytorch_fp32']['accuracy']['mAP50']:.3f}). {b8:,.0f} images/s at batch 8: an upper "
              f"bound of ~{b8 / 5:.0f} streams at 5 analysed FPS, before video decoding.")
    else:
        hw = "TensorRT numbers not measured yet."

    sections = [
        ("1. Decision",
         "Go / no-go for a paid pilot on <b>N fixed cameras</b> (e.g. block-assembly hall, dry dock, paint shop). "
         "Success criteria are written down before any data is seen: per-event recall at an agreed false-alert "
         "budget, alert latency, and hardware cost per camera.",
         f"Rehearsed on public data: SH17, {s['n_images']:,} photos of people at work, 17 PPE and body-part "
         f"classes, {s['n_photographers']} photographers, with a fine-tuned YOLO26s. No shipyard data was used."),
        ("2. Data request",
         "Per camera: view, height and distance; day / night / rain / welding glare / fog. Typical work and the rare "
         "events (falls, work at height, missing harness). A <b>class list with written label rules</b>, "
         "e.g. helmet = hard hat worn on the head (held in hand is not worn); fallen = torso on the ground for over "
         "2 s (kneeling is not fallen). 2–4 weeks of footage, with the site's privacy (PDPA) handling agreed.",
         rules_ev),
        ("3. Site-condition test set",
         "Held out <b>by camera and by date</b>, never by random frame. Report per-class precision and recall, and "
         "the look-alike pairs that matter: kneeling vs fallen, helmet vs cap / hood / bare head, gloves vs bare "
         "hands, harness vs vest straps.",
         site_ev),
        ("4. Baselines",
         "Prompted ZERO out of the box first, then fine-tuned on site data; report both on the same site test set, "
         "so the customer sees what the fine-tuning buys and what it costs in labelling.",
         f"Fine-tuned only: mAP@0.5 {gm:.2f}, mAP@0.5:0.95 {g['reference']['mAP50_95']:.2f} on held-out "
         "photographers. A zero-shot baseline was not run here."),
        ("5. Operating point",
         "An <b>alert budget per camera per shift</b>, agreed with the safety team, because a missed fall and a "
         "false alarm cost different things. Thresholds are set per class from that budget, then confirmed over "
         "k of n frames before an alert is sent.",
         op_ev),
        ("6. Hardware and latency",
         "Edge (Jetson Orin class) or on-premise server (L4 / A10 class); frames per second per stream, streams "
         "per GPU, alert latency end to end. The engine is built per GPU, so the container builds it on first start.",
         hw),
        ("7. Risks and next steps",
         "Data gaps (night, rain, real falls are rare) → generated data for rare hazards, but a <b>real-only</b> "
         "test set. Label drift between annotators → a written rule book and a double-labelled audit sample. "
         "Timeline: week 1 data and rules · weeks 2–3 baseline and fine-tune · week 4 site test and operating "
         "point · week 5 go / no-go review. From the site: camera access, sample footage, a safety-team contact.",
         "Gaps of this pilot: no falls, no video, no night or rain conditions; stock photos, not CCTV."),
    ]

    def block(h, body, ev):
        return [Paragraph(h, st["h"]), Spacer(0, 1.2 * mm), Paragraph(body, st["b"]), Spacer(0, 1.4 * mm),
                Table([[Paragraph("<b>Pilot evidence</b> · " + ev, st["ev"])]], colWidths=[86 * mm],
                      style=TableStyle([("BACKGROUND", (0, 0), (-1, -1), BOX),
                                        ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                                        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)])),
                Spacer(0, 3.2 * mm)]

    left = [f for s in sections[:4] for f in block(*s)]
    right = [f for s in sections[4:] for f in block(*s)]
    right.append(Paragraph(
        "<b>Reproduce:</b> <font face='Courier'>python run_all.py</font> rebuilds the split, model, metrics, "
        "TensorRT engines and this page. Data: SH17 (CC BY-NC-SA 4.0, Pexels photos). Code: AGPL-3.0, built on "
        "Ultralytics YOLO26s.", st["foot"]))
    grid = Table([[left, right]], colWidths=[91 * mm, 91 * mm],
                 style=TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                   ("RIGHTPADDING", (0, 0), (-1, -1), 5 * mm)]))
    fig = FIGURES / "alert_budget.png"
    band = []
    if fig.exists():
        w, h = ImageReader(str(fig)).getSize()
        width = 182 * mm
        band = [Spacer(0, 1 * mm), Paragraph("Pilot: recall against false alerts per 1,000 frames for the look-alike "
                                             "pairs (held-out photographers). The dot is the operating point.", st["ev"]),
                Spacer(0, 1 * mm), Image(str(fig), width=width, height=width * h / w)]
    story = [
        Paragraph("Feasibility study outline — PPE and fall detection in shipyard work areas", st["title"]),
        Spacer(0, 1.5 * mm),
        Paragraph("Jinwoo Seol · personal example, November 2026 · independent work, not affiliated with any vendor "
                  "or shipyard. Grey boxes show numbers from a public-data pilot, not from a shipyard.", st["sub"]),
        Spacer(0, 2 * mm),
        Table([[""]], colWidths=[182 * mm], rowHeights=[0.6], style=TableStyle([("LINEABOVE", (0, 0), (-1, -1), 0.6, RULE)])),
        Spacer(0, 3 * mm),
        grid,
        *band,
    ]
    DOCS.mkdir(parents=True, exist_ok=True)
    out = DOCS / "feasibility_outline.pdf"
    SimpleDocTemplate(str(out), pagesize=A4, leftMargin=14 * mm, rightMargin=14 * mm, topMargin=13 * mm,
                      bottomMargin=11 * mm, title="Feasibility study outline – shipyard PPE / fall detection",
                      author="Jinwoo Seol").build(story)
