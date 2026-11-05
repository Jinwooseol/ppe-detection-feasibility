"""Site-condition style split for SH17: hold out whole photographers.

SH17 has no camera or site ids, but every image ships with its Pexels record, including the
photographer id. A photographer is the closest thing this data has to "one camera at one site":
the same places, people, light and lens recur across their photos. We hold out whole
photographers, after merging any pixel-level near-duplicates across photographers (re-uploads)
into the same group. Very large sources (> MAX_HELDOUT_SHARE of all images) stay in training so
the test set is not dominated by one studio. A random image-level split of the same size is kept
as the leaky baseline.

Near-duplicates use a 64-bit DCT perceptual hash. Detector-backbone embeddings were tried first
and rejected: they rate different photos of similar scenes as near-identical.
"""
from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .common import DATA, NAMES, SEED, SPLITS, WORK, load_json, save_json

FRACTIONS = {"train": 0.7, "val": 0.1, "test": 0.2}
DUP_HAMMING = 6            # pHash bits; at <= 6 every pair checked by eye was the same photo
MAX_HELDOUT_SHARE = 0.03   # sources bigger than this stay in train


def phash(paths: list[Path], cache: Path) -> np.ndarray:
    """64-bit DCT perceptual hash per image (uint64)."""
    if cache.exists():
        h = np.load(cache)
        if len(h) == len(paths):
            return h
    n = 32
    k = np.arange(n)
    dct = np.cos(np.pi * (2 * k[None, :] + 1) * k[:, None] / (2 * n))
    bits = []
    for p in paths:
        a = np.asarray(Image.open(p).convert("L").resize((n, n), Image.Resampling.LANCZOS), np.float64)
        low = (dct @ a @ dct.T)[:8, :8].ravel()
        bits.append(low > np.median(low[1:]))
    h = np.packbits(np.array(bits), axis=1).view(np.uint64).ravel()
    np.save(cache, h)
    return h


def hamming(h: np.ndarray) -> np.ndarray:
    d = np.zeros((len(h), len(h)), np.uint8)
    for s in range(0, len(h), 1024):
        d[s:s + 1024] = np.bitwise_count(h[s:s + 1024, None] ^ h[None, :])
    np.fill_diagonal(d, 64)
    return d


class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.p[max(a, b)] = min(a, b)
            return True
        return False


def build_groups(rows, dist):
    """Group = photographer, merged across photographers by pixel-level near-duplicates."""
    n = len(rows)
    dsu = DSU(n)
    first = {}
    for k, r in enumerate(rows):
        key = r["photographer_id"] if r["photographer_id"] >= 0 else f"img{k}"
        if key in first:
            dsu.union(first[key], k)
        else:
            first[key] = k
    dup_pairs = cross = 0
    for i, j in zip(*np.where(np.triu(dist <= DUP_HAMMING, 1))):
        dup_pairs += 1
        cross += dsu.union(int(i), int(j))
    roots = [dsu.find(k) for k in range(n)]
    remap = {r: g for g, r in enumerate(sorted(set(roots)))}
    return [remap[r] for r in roots], {"near_duplicate_pairs": dup_pairs, "photographer_merges": cross}


def stratified_group_split(rows, groups):
    """Greedy: biggest groups first, each to the split whose class shares are furthest below target."""
    k = len(NAMES)
    gcount, gsize = {}, Counter(groups)
    for r, g in zip(rows, groups):
        gcount.setdefault(g, np.zeros(k))
        gcount[g] += np.minimum(np.array(r["counts"]), 1)   # images containing the class
    total = sum(gcount.values())
    n_img = len(rows)
    names = list(FRACTIONS)
    target = np.array([FRACTIONS[s] for s in names])
    cur = np.zeros((len(names), k))
    cur_n = np.zeros(len(names))

    rng = random.Random(SEED)
    gids = list(gcount)
    rng.shuffle(gids)
    gids.sort(key=lambda g: -gsize[g])
    assign = {}
    for g in gids:
        allowed = [0] if gsize[g] > MAX_HELDOUT_SHARE * n_img else range(len(names))
        best, best_cost = 0, None
        for s in allowed:
            c, cn = cur.copy(), cur_n.copy()
            c[s] += gcount[g]
            cn[s] += gsize[g]
            share = c / np.maximum(total, 1)
            cost = ((share - target[:, None]) ** 2).sum() + 4 * ((cn / n_img - target) ** 2).sum()
            if best_cost is None or cost < best_cost:
                best, best_cost = s, cost
        assign[g] = names[best]
        cur[best] += gcount[g]
        cur_n[best] += gsize[g]
    return [assign[g] for g in groups]


def random_split(rows, sizes):
    idx = list(range(len(rows)))
    random.Random(SEED).shuffle(idx)
    out, start = [None] * len(rows), 0
    for s in FRACTIONS:
        for k in idx[start:start + sizes[s]]:
            out[k] = s
        start += sizes[s]
    return out


def write_split(name, rows, split):
    d = SPLITS / name
    d.mkdir(parents=True, exist_ok=True)
    for s in FRACTIONS:
        files = [str(DATA / "images" / f"{r['stem']}.jpg") for r, x in zip(rows, split) if x == s]
        (d / f"{s}.txt").write_text("\n".join(files) + "\n")
    yaml = [f"path: {DATA.as_posix()}"] + [f"{s}: {(d / f'{s}.txt').as_posix()}" for s in FRACTIONS]
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(NAMES)]
    (d / "data.yaml").write_text("\n".join(yaml) + "\n")


def summarise(rows, split, groups, dist):
    out = {}
    for s in FRACTIONS:
        idx = [k for k, x in enumerate(split) if x == s]
        inst = np.sum([rows[k]["counts"] for k in idx], axis=0)
        img = np.sum([np.minimum(rows[k]["counts"], 1) for k in idx], axis=0)
        phot = Counter(rows[k]["photographer_id"] for k in idx)
        out[s] = {"images": len(idx), "groups": len({groups[k] for k in idx}), "photographers": len(phot),
                  "largest_photographer_share": round(phot.most_common(1)[0][1] / len(idx), 4),
                  "instances": {n: int(v) for n, v in zip(NAMES, inst)},
                  "images_with_class": {n: int(v) for n, v in zip(NAMES, img)}}
    tr = np.array([x == "train" for x in split])
    te = np.where([x == "test" for x in split])[0]
    train_phot = {rows[k]["photographer_id"] for k in np.where(tr)[0]}
    nn = dist[np.ix_(te, np.where(tr)[0])].min(axis=1) if len(te) and tr.any() else np.zeros(0)
    out["test_images_whose_photographer_is_in_train"] = round(
        float(np.mean([rows[k]["photographer_id"] in train_phot for k in te])), 4)
    out["test_images_with_near_duplicate_in_train"] = round(float(np.mean(nn <= DUP_HAMMING)), 4)
    return out


def make_splits() -> dict:
    summary_path = SPLITS / "split_summary.json"
    if summary_path.exists() and (SPLITS / "group" / "data.yaml").exists():
        return load_json(summary_path)
    rows = load_json(DATA / "meta.json")["images"]
    paths = [DATA / "images" / f"{r['stem']}.jpg" for r in rows]
    dist = hamming(phash(paths, WORK / "data" / "phash64.npy"))
    groups, links = build_groups(rows, dist)
    sizes = Counter(groups)

    group_split = stratified_group_split(rows, groups)
    n = Counter(group_split)
    rand_split = random_split(rows, {s: n[s] for s in FRACTIONS})
    write_split("group", rows, group_split)
    write_split("random", rows, rand_split)

    summary = {
        "params": {"dup_hamming": DUP_HAMMING, "max_heldout_share": MAX_HELDOUT_SHARE, "fractions": FRACTIONS, "seed": SEED},
        "n_images": len(rows), "n_photographers": len({r["photographer_id"] for r in rows}),
        "n_groups": len(sizes), "largest_groups": [v for _, v in sizes.most_common(10)], "links": links,
        "group": summarise(rows, group_split, groups, dist),
        "random": summarise(rows, rand_split, groups, dist),
    }
    save_json({"stem": [r["stem"] for r in rows], "group": groups, "group_split": group_split,
               "random_split": rand_split}, SPLITS / "assignments.json")
    save_json(summary, summary_path)
    return summary
