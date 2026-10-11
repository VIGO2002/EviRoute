#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import argparse
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter


class TopKMILScorer(nn.Module):
    def __init__(self, feat_dim=4096, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_patches(self, x):
        return self.scorer(x).squeeze(-1)


def get_first_key(d, keys):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of keys {keys} found. Existing keys: {list(d.keys())}")


def load_payload(features_pt):
    d = torch.load(features_pt, map_location="cpu")
    patch_features = get_first_key(d, ["patch_features", "patch_feats", "patch_tokens"])
    labels = get_first_key(d, ["labels", "label", "y"]).long()

    paths = d.get("paths", d.get("path", [""] * len(labels)))
    sources = d.get("sources", d.get("source", ["unknown"] * len(labels)))
    case_names = d.get("case_names", d.get("case_name", [""] * len(labels)))

    if isinstance(paths, np.ndarray):
        paths = paths.tolist()
    if isinstance(sources, np.ndarray):
        sources = sources.tolist()
    if isinstance(case_names, np.ndarray):
        case_names = case_names.tolist()

    return {
        "patch_features": patch_features,
        "labels": labels,
        "paths": paths,
        "sources": sources,
        "case_names": case_names,
    }


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    args = ckpt.get("args", {})

    model = TopKMILScorer(
        feat_dim=int(args.get("feat_dim", 4096)),
        hidden_dim=int(args.get("hidden_dim", 256)),
        dropout=float(args.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mu = ckpt["mu"].float().to(device)
    std = ckpt["std"].float().to(device)

    return model, mu, std


@torch.no_grad()
def get_scores(model, mu, std, patch_features, device):
    x = patch_features.unsqueeze(0).float().to(device)
    x = (x - mu.view(1, 1, -1)) / std.view(1, 1, -1)
    scores = model.score_patches(x).squeeze(0).detach().cpu().numpy()
    return scores


def ring_id(patch_id, grid=14):
    r = patch_id // grid
    c = patch_id % grid
    if r == 0 or r == grid - 1 or c == 0 or c == grid - 1:
        return "edge1"
    if r <= 1 or r >= grid - 2 or c <= 1 or c >= grid - 2:
        return "edge2_only"
    return "interior"


def sample_ring_matched(topk_ids, rng, grid=14):
    all_ids = list(range(grid * grid))
    topk_set = set(topk_ids)

    ring_to_pool = {
        "edge1": [],
        "edge2_only": [],
        "interior": [],
    }

    for pid in all_ids:
        if pid in topk_set:
            continue
        ring_to_pool[ring_id(pid, grid)].append(pid)

    counts = Counter(ring_id(pid, grid) for pid in topk_ids)

    sampled = []
    used = set()

    for ring, n in counts.items():
        pool = [p for p in ring_to_pool[ring] if p not in used]
        rng.shuffle(pool)
        take = pool[:n]
        sampled.extend(take)
        used.update(take)

    if len(sampled) < len(topk_ids):
        remain = [p for p in all_ids if p not in used and p not in topk_set]
        rng.shuffle(remain)
        sampled.extend(remain[: len(topk_ids) - len(sampled)])

    return sampled[: len(topk_ids)]


def perturb_image_224(img, patch_ids, perturb="local_mean", blur_radius=6.0, grid=14):
    img = img.convert("RGB").resize((224, 224), Image.BICUBIC)

    if perturb == "clean":
        return img

    arr = np.array(img).copy()
    cell = 224 // grid

    if perturb == "blur":
        blurred = np.array(img.filter(ImageFilter.GaussianBlur(radius=blur_radius)))
        for pid in patch_ids:
            r = pid // grid
            c = pid % grid
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            arr[y0:y1, x0:x1] = blurred[y0:y1, x0:x1]

    elif perturb == "local_mean":
        for pid in patch_ids:
            r = pid // grid
            c = pid % grid
            y0, y1 = r * cell, (r + 1) * cell
            x0, x1 = c * cell, (c + 1) * cell
            region = arr[y0:y1, x0:x1]
            mean = region.reshape(-1, 3).mean(axis=0)
            arr[y0:y1, x0:x1] = mean

    else:
        raise ValueError(f"Unknown perturb: {perturb}")

    return Image.fromarray(arr)


def safe_name(s):
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(s))


def read_index_csv(index_csv):
    rows = []
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_pt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--num_random", type=int, default=20)
    ap.add_argument("--perturb", default="local_mean", choices=["local_mean", "blur"])
    ap.add_argument("--blur_radius", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    img_dir = os.path.join(args.out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    payload = load_payload(args.features_pt)
    model, mu, std = load_model(args.checkpoint, device)
    index_rows = read_index_csv(args.index_csv)

    csv_rows = []
    meta_rows = []

    for row_i, row in enumerate(index_rows):
        orig_idx = int(row["idx"])
        label = int(payload["labels"][orig_idx].item())

        if label != 1:
            continue

        src = payload["sources"][orig_idx]
        case_name = payload["case_names"][orig_idx] if orig_idx < len(payload["case_names"]) else str(orig_idx)
        img_path = payload["paths"][orig_idx]

        if not os.path.exists(img_path):
            print("[Skip missing]", img_path)
            continue

        patch_features = payload["patch_features"][orig_idx]
        scores = get_scores(model, mu, std, patch_features, device)
        topk_ids = np.argsort(-scores)[: args.topk].tolist()

        img = Image.open(img_path).convert("RGB")

        variants = []
        variants.append(("clean", 0, []))
        variants.append(("topk", 0, topk_ids))

        for rep in range(args.num_random):
            rand_ids = sample_ring_matched(topk_ids, rng, grid=14)
            variants.append(("rand", rep, rand_ids))

        for variant, rep, patch_ids in variants:
            if variant == "clean":
                out_img = perturb_image_224(img, [], perturb="clean")
            else:
                out_img = perturb_image_224(
                    img,
                    patch_ids,
                    perturb=args.perturb,
                    blur_radius=args.blur_radius,
                    grid=14,
                )

            out_name = (
                f"{row_i:03d}_orig{orig_idx}_{safe_name(src)}_"
                f"{variant}{rep:02d}_{args.perturb}.png"
            )
            out_path = os.path.join(img_dir, out_name)
            out_img.save(out_path)

            csv_rows.append({
                "path": out_path,
                "label": 1,
                "split": "test",
                "source": f"{src}__{variant}",
                "case_name": f"{case_name}__{variant}{rep:02d}",
                "orig_idx": orig_idx,
                "variant": variant,
                "rep": rep,
            })

            meta_rows.append({
                "row_id": len(meta_rows),
                "path": out_path,
                "orig_idx": orig_idx,
                "label": 1,
                "source": src,
                "case_name": case_name,
                "variant": variant,
                "rep": rep,
                "topk_ids": topk_ids,
                "perturb_ids": patch_ids,
                "perturb": args.perturb if variant != "clean" else "clean",
            })

        print(f"[Done] {row_i+1}/{len(index_rows)} orig_idx={orig_idx} source={src}")

    out_csv = os.path.join(args.out_dir, "perturbations.csv")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["path", "label", "split", "source", "case_name", "orig_idx", "variant", "rep"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    out_meta = os.path.join(args.out_dir, "metadata.jsonl")
    with open(out_meta, "w", encoding="utf-8") as f:
        for m in meta_rows:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    print("Saved CSV:", out_csv)
    print("Saved metadata:", out_meta)
    print("Total images:", len(csv_rows))


if __name__ == "__main__":
    main()
