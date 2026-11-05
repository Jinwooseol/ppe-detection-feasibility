"""Static figures for the README (light surface, reference palette, thin marks, direct labels).

Nothing here runs the model: every figure is drawn from results/*.json and the validator's saved
predictions, so figures always show the same boxes the metrics were computed from.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from .common import FIGURES, NAMES, RESULTS, local, save_json
from .evaluate import BUDGETS, MAIN_BUDGET, confusion_cases

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
PAIRS = [("helmet", "head"), ("gloves", "hands"), ("shoes", "foot")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "legend.frameon": False, "figure.dpi": 150,
})


def split_gap(group: dict, random: dict, path: Path) -> None:
    """Per-class mAP50 on the leaky random split vs the held-out-photographer split."""
    order = sorted(NAMES, key=lambda n: group["per_class"][n].get("mAP50", 0))
    g = [group["per_class"][n].get("mAP50", 0) for n in order]
    r = [random["per_class"][n].get("mAP50", 0) for n in order]
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.barh(y + 0.2, r, 0.38, color=S2, label="random image split (leaky)", edgecolor=SURFACE, linewidth=1)
    ax.barh(y - 0.2, g, 0.38, color=S1, label="held-out photographers", edgecolor=SURFACE, linewidth=1)
    for yi, gi, ri in zip(y, g, r):
        diff = round((ri - gi) * 100)
        ax.text(max(gi, ri) + 0.012, yi, f"{diff:+d}" if diff else "0", va="center", fontsize=8, color=INK2)
    ax.set_yticks(y, order)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("mAP@0.5 on each split's own test set   (label: random minus held-out, points)")
    ax.grid(axis="y", visible=False)
    ax.set_title("A random image split overstates accuracy on unseen sources", loc="left")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def budget_curves(curves: dict, ops: dict, pairs, path: Path) -> None:
    """Recall vs false alerts per 1,000 frames for protected / unprotected look-alike pairs."""
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.5), sharey=True)
    for ax, (a, b) in zip(np.atleast_1d(axes), pairs):
        pts = {}
        for n, col in ((a, S1), (b, S2)):
            c = curves[n]
            ax.plot(np.maximum(np.array(c["FP_per_1k"]), 0.1), c["R"], color=col, lw=2, label=n)
            op = ops[n][f"budget_{MAIN_BUDGET}"]
            if op["thr"] is not None:
                pts[n] = (max(op["FP_per_1k"], 0.1), op["R"], op["thr"], col)
        hi = max(pts, key=lambda n: pts[n][1]) if pts else None
        for n, (x, yv, thr, col) in pts.items():
            ax.plot(x, yv, "o", ms=7, color=col, mec=SURFACE, mew=2, zorder=5)
            dy = 8 if n == hi else -14
            ax.annotate(f"R {yv:.2f} @ conf {thr:.2f}", (x, yv), textcoords="offset points",
                        xytext=(-6, dy), ha="right", fontsize=8, color=INK2, zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.9))
        for bgt in BUDGETS:
            ax.axvline(bgt, color=INK2 if bgt == MAIN_BUDGET else GRID, lw=1, ls="--", zorder=0)
        ax.set_xscale("log")
        ax.set_xlim(0.5, 2000)
        ax.set_ylim(0, 1)
        ax.set_title(f"{a} vs {b}", loc="left")
        ax.set_xlabel("false alerts per 1,000 frames")
        ax.legend(loc="upper left", fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel("recall")
    fig.suptitle(f"Operating point: highest recall within {MAIN_BUDGET} false alerts / 1,000 frames "
                 f"(dashed: {', '.join(map(str, BUDGETS))})", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def error_bars(fp: dict, fn: dict, n_gt: dict, path: Path) -> None:
    """Why the false alerts and the misses happen, per class, at the main operating point."""
    miss_keys = ("as_other_class", "poor_box", "missed")
    order = sorted(NAMES, key=lambda n: -sum(fn[n][k] for k in miss_keys) / max(n_gt[n], 1))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 6.2), sharey=True)
    y = np.arange(len(order))
    for ax, data, keys, title, scale in (
        (a1, fp, [("wrong_class", "another class labelled there", S2), ("localisation", "poor box", S4),
                  ("duplicate", "duplicate", S3), ("background", "nothing labelled there", S1)],
         "False alerts (count)", None),
        (a2, fn, [("as_other_class", "called a look-alike class", S2), ("poor_box", "poor box", S4),
                  ("missed", "not detected", S1)], "Missed objects (% of true boxes)", n_gt),
    ):
        left = np.zeros(len(order))
        for key, label, col in keys:
            v = np.array([data[n][key] for n in order], float)
            if scale:
                v = 100 * v / np.maximum([scale[n] for n in order], 1)
            ax.barh(y, v, left=left, color=col, label=label, edgecolor=SURFACE, linewidth=1.5, height=0.7)
            left += v
        ax.set_title(title, loc="left")
        ax.grid(axis="y", visible=False)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncol=2, fontsize=8)
    a2.set_xlim(0, 100)
    a1.set_yticks(y, order)
    a1.invert_yaxis()
    fig.suptitle(f"Error causes at each class's operating point ({MAIN_BUDGET} false alerts / 1,000 frames); "
                 "classes sorted by miss rate", x=0.01, ha="left", fontsize=9.5, color=INK2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _crop(src, g, q, pad=0.8):
    im = Image.open(local(src))
    W, H = im.size
    x0, y0, x1, y1 = np.array(g) * [W, H, W, H]
    p = max(x1 - x0, y1 - y0) * pad
    box = (int(max(0, x0 - p)), int(max(0, y0 - p)), int(min(W, x1 + p)), int(min(H, y1 + p)))
    rects = [(np.array(b) * [W, H, W, H]) - [box[0], box[1], box[0], box[1]] for b in (g, q)]
    return im.crop(box), rects


def confusion_gallery(result: dict, pairs, path: Path, per_pair: int = 6) -> None:
    """Crops where the model called one look-alike class the other (label green, prediction orange)."""
    rows = []
    for gt_name, pred_name in pairs:
        cases = sorted(confusion_cases(result, gt_name, pred_name), key=lambda c: -c["conf"])
        if cases:
            rows.append((gt_name, pred_name, cases[:per_pair], len(cases)))
    if not rows:
        return
    fig, axes = plt.subplots(len(rows), per_pair, figsize=(1.6 * per_pair + 1.0, 1.9 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (gt_name, pred_name, cases, total) in enumerate(rows):
        for c in range(per_pair):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(cases):
                continue
            im, (g, q) = _crop(cases[c]["image"], cases[c]["gt"], cases[c]["pred"])
            ax.imshow(im)
            for (bx0, by0, bx1, by1), col in ((g, S3), (q, S2)):
                ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=col, lw=1.5))
            ax.set_title(f"{cases[c]['conf']:.2f}", fontsize=7, color=INK2)
        axes[r, 0].text(-0.1, 0.5, f"label: {gt_name}\npredicted: {pred_name}\n({total} cases)",
                        transform=axes[r, 0].transAxes, ha="right", va="center", fontsize=8)
    fig.suptitle("Look-alike confusions on held-out photographers, most confident first "
                 "(green: label, orange: prediction, title: confidence)", x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def audit_sheets(result: dict, pairs, out_dir: Path, cols: int = 8) -> dict:
    """Every confusion case of a pair on one numbered sheet, for a manual label audit."""
    out_dir.mkdir(parents=True, exist_ok=True)
    index = {}
    for gt_name, pred_name in pairs:
        cases = sorted(confusion_cases(result, gt_name, pred_name), key=lambda c: -c["conf"])
        key = f"{gt_name}_as_{pred_name}"
        index[key] = cases
        if not cases:
            continue
        rows = (len(cases) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(1.7 * cols, 1.9 * rows))
        for k, ax in enumerate(np.atleast_1d(axes).ravel()):
            ax.axis("off")
            if k >= len(cases):
                continue
            im, (g, q) = _crop(cases[k]["image"], cases[k]["gt"], cases[k]["pred"], pad=1.2)
            ax.imshow(im)
            for (bx0, by0, bx1, by1), col in ((g, S3), (q, S2)):
                ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0, fill=False, ec=col, lw=1.2))
            ax.set_title(f"#{k + 1}  {cases[k]['conf']:.2f}", fontsize=7, color=INK2)
        fig.suptitle(f"Audit: label {gt_name}, predicted {pred_name} ({len(cases)} cases)", x=0.01, ha="left", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{key}.png", dpi=110)
        plt.close(fig)
    save_json(index, out_dir / "cases.json")
    return index


def make_all(group: dict, random: dict | None, curves: dict) -> list[Path]:
    FIGURES.mkdir(parents=True, exist_ok=True)
    out = []
    if random:
        split_gap(group, random, FIGURES / "split_gap.png")
        out.append(FIGURES / "split_gap.png")
    n_gt = {n: group["per_class"][n]["n_gt"] for n in NAMES}
    budget_curves(curves, group["operating_points"], PAIRS, FIGURES / "alert_budget.png")
    error_bars(group["false_positives"], group["misses"], n_gt, FIGURES / "error_causes.png")
    both_ways = [(a, b) for p in PAIRS for (a, b) in (p, p[::-1])]
    confusion_gallery(group, both_ways, FIGURES / "lookalike_examples.png")
    audit_sheets(group, both_ways, RESULTS / "audit")
    out += [FIGURES / f for f in ("alert_budget.png", "error_causes.png", "lookalike_examples.png")]
    return out
