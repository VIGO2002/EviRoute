#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import argparse
from collections import defaultdict, Counter

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
def get_patch_scores(model, mu, std, patch_features, device):
    x = patch_features.unsqueeze(0).float().to(device)
    x = (x - mu.view(1, 1, -1)) / std.view(1, 1, -1)
    scores = model.score_patches(x).squeeze(0).detach().cpu().numpy()
    return scores


def read_index_csv(index_csv):
    rows = []
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


def ring_id(pid, grid=14):
    r = pid // grid
    c = pid % grid
    if r == 0 or r == grid - 1 or c == 0 or c == grid - 1:
        return "edge1"
    if r <= 1 or r >= grid - 2 or c <= 1 or c >= grid - 2:
        return "edge2_only"
    return "interior"


def select_red_blue(scores, topk=10, grid=14):
    """
    red: global top-k high fake-support patches.
    blue: same ring composition, lowest-score patches from corresponding rings.
    """
    all_ids = list(range(grid * grid))
    red_ids = np.argsort(-scores)[:topk].tolist()
    red_set = set(red_ids)

    counts = Counter(ring_id(pid, grid) for pid in red_ids)

    blue_ids = []
    used = set(red_ids)

    for ring, n in counts.items():
        candidates = [pid for pid in all_ids if pid not in used and ring_id(pid, grid) == ring]
        candidates = sorted(candidates, key=lambda p: scores[p])
        take = candidates[:n]
        blue_ids.extend(take)
        used.update(take)

    if len(blue_ids) < len(red_ids):
        candidates = [pid for pid in all_ids if pid not in used]
        candidates = sorted(candidates, key=lambda p: scores[p])
        blue_ids.extend(candidates[: len(red_ids) - len(blue_ids)])

    return red_ids, blue_ids[:len(red_ids)]


def patch_slice(pid, grid=14, size=224):
    cell = size // grid
    r, c = pid // grid, pid % grid
    y0, y1 = r * cell, (r + 1) * cell
    x0, x1 = c * cell, (c + 1) * cell
    return y0, y1, x0, x1


def gray_image(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    return 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]


def fft_hf_ratio(patch):
    x = patch.astype(np.float32)
    x = x - x.mean()
    f = np.fft.fftshift(np.fft.fft2(x))
    e = np.abs(f) ** 2

    h, w = x.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rr = rr / (rr.max() + 1e-8)

    mask = rr >= 0.35
    return float(e[mask].sum() / (e.sum() + 1e-8))


def percentile_ranks(values):
    """
    Return percentile rank in [0, 1] for each patch within an image.
    """
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def clipped_abs_robust_z(values, clip=5.0):
    values = np.asarray(values, dtype=np.float64)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad + 1e-8
    z = np.abs(values - med) / scale
    return np.clip(z, 0.0, clip)


def compute_patch_metrics(img):
    img = img.convert("RGB").resize((224, 224), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    gray = gray_image(img)

    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")).astype(np.float32) / 255.0
    blur_gray = 0.299 * blur[:, :, 0] + 0.587 * blur[:, :, 1] + 0.114 * blur[:, :, 2]
    residual = gray - blur_gray

    # simple finite differences
    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    grad_energy_map = gx ** 2 + gy ** 2

    # Laplacian using shifted neighbors
    pad = np.pad(gray, ((1, 1), (1, 1)), mode="reflect")
    lap = (
        pad[1:-1, :-2] + pad[1:-1, 2:] +
        pad[:-2, 1:-1] + pad[2:, 1:-1] -
        4.0 * gray
    )

    metrics = {
        "lap_var": [],
        "grad_energy": [],
        "residual_abs": [],
        "fft_hf_ratio": [],
        "color_var": [],
    }

    for pid in range(196):
        y0, y1, x0, x1 = patch_slice(pid)

        p_gray = gray[y0:y1, x0:x1]
        p_lap = lap[y0:y1, x0:x1]
        p_grad = grad_energy_map[y0:y1, x0:x1]
        p_res = residual[y0:y1, x0:x1]
        p_rgb = rgb[y0:y1, x0:x1, :]

        metrics["lap_var"].append(float(np.var(p_lap)))
        metrics["grad_energy"].append(float(np.mean(p_grad)))
        metrics["residual_abs"].append(float(np.mean(np.abs(p_res))))
        metrics["fft_hf_ratio"].append(fft_hf_ratio(p_gray))
        metrics["color_var"].append(float(np.var(p_rgb.reshape(-1, 3), axis=0).mean()))

    return {k: np.asarray(v, dtype=np.float64) for k, v in metrics.items()}


def bootstrap_ci_mean(vals, n_boot=1000, seed=123):
    vals = np.asarray(vals, dtype=np.float64)
    if len(vals) == 0:
        return [None, None]
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(vals, size=len(vals), replace=True)
        means.append(sample.mean())
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def summarize(records, metric_names):
    out = {"n": len(records), "metrics": {}}
    for m in metric_names:
        raw_delta = np.asarray([r[f"{m}_raw_delta"] for r in records], dtype=np.float64)
        pct_delta = np.asarray([r[f"{m}_pct_delta"] for r in records], dtype=np.float64)
        z_delta = np.asarray([r[f"{m}_zclip_delta"] for r in records], dtype=np.float64)

        out["metrics"][m] = {
            "mean_raw_delta": float(raw_delta.mean()),
            "median_raw_delta": float(np.median(raw_delta)),
            "red_gt_blue_raw_rate": float((raw_delta > 0).mean()),

            "mean_pct_delta": float(pct_delta.mean()),
            "median_pct_delta": float(np.median(pct_delta)),
            "red_gt_blue_pct_rate": float((pct_delta > 0).mean()),
            "mean_pct_delta_ci95": bootstrap_ci_mean(pct_delta),

            "mean_zclip_delta": float(z_delta.mean()),
            "median_zclip_delta": float(np.median(z_delta)),
            "red_gt_blue_zclip_rate": float((z_delta > 0).mean()),
        }

    # 综合 percentile abnormality
    comp = np.asarray([r["composite_pct_delta"] for r in records], dtype=np.float64)
    out["composite_pct"] = {
        "mean_delta": float(comp.mean()),
        "median_delta": float(np.median(comp)),
        "red_gt_blue_rate": float((comp > 0).mean()),
        "mean_delta_ci95": bootstrap_ci_mean(comp),
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_pt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    payload = load_payload(args.features_pt)
    model, mu, std = load_model(args.checkpoint, device)
    rows = read_index_csv(args.index_csv)

    metric_names = ["lap_var", "grad_energy", "residual_abs", "fft_hf_ratio", "color_var"]

    records = []
    by_source = defaultdict(list)

    for row_i, row in enumerate(rows):
        idx = int(row["idx"])
        label = int(payload["labels"][idx].item())
        src = payload["sources"][idx]
        case_name = payload["case_names"][idx] if idx < len(payload["case_names"]) else str(idx)
        path = payload["paths"][idx]

        if not os.path.exists(path):
            print("[Skip missing]", path)
            continue

        scores = get_patch_scores(model, mu, std, payload["patch_features"][idx].float(), device)
        red_ids, blue_ids = select_red_blue(scores, topk=args.topk)

        img = Image.open(path).convert("RGB")
        patch_metrics = compute_patch_metrics(img)

        rec = {
            "idx": idx,
            "label": label,
            "source": src,
            "case_name": case_name,
            "path": path,
            "red_ids": red_ids,
            "blue_ids": blue_ids,
            "red_score_mean": float(np.mean(scores[red_ids])),
            "blue_score_mean": float(np.mean(scores[blue_ids])),
            "score_gap": float(np.mean(scores[red_ids]) - np.mean(scores[blue_ids])),
        }

        pct_deltas = []

        for m in metric_names:
            vals = patch_metrics[m]
            pct = percentile_ranks(vals)
            zclip = clipped_abs_robust_z(vals, clip=5.0)

            red_raw = float(vals[red_ids].mean())
            blue_raw = float(vals[blue_ids].mean())
            red_pct = float(pct[red_ids].mean())
            blue_pct = float(pct[blue_ids].mean())
            red_z = float(zclip[red_ids].mean())
            blue_z = float(zclip[blue_ids].mean())

            rec[f"{m}_red_raw"] = red_raw
            rec[f"{m}_blue_raw"] = blue_raw
            rec[f"{m}_raw_delta"] = red_raw - blue_raw

            rec[f"{m}_red_pct"] = red_pct
            rec[f"{m}_blue_pct"] = blue_pct
            rec[f"{m}_pct_delta"] = red_pct - blue_pct

            rec[f"{m}_red_zclip"] = red_z
            rec[f"{m}_blue_zclip"] = blue_z
            rec[f"{m}_zclip_delta"] = red_z - blue_z

            pct_deltas.append(red_pct - blue_pct)

        rec["composite_pct_delta"] = float(np.mean(pct_deltas))

        records.append(rec)
        by_source[src].append(rec)

        if (row_i + 1) % 20 == 0:
            print(f"[Done] {row_i+1}/{len(rows)}")

    out = {
        "features_pt": args.features_pt,
        "checkpoint": args.checkpoint,
        "index_csv": args.index_csv,
        "comparison": "heatmap_red_vs_same_ring_blue",
        "red_definition": "top-k highest v2b fake-support scores",
        "blue_definition": "same-ring lowest v2b fake-support scores",
        "topk": args.topk,
        "overall": summarize(records, metric_names),
        "by_source": {src: summarize(rs, metric_names) for src, rs in sorted(by_source.items())},
        "per_sample": records,
    }

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    cols = list(records[0].keys()) if records else []
    with open(args.output_csv, "w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for r in records:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, list):
                    v = "|".join(map(str, v))
                elif isinstance(v, str):
                    v = v.replace(",", "_")
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print("Saved JSON:", args.output_json)
    print("Saved CSV:", args.output_csv)
    print(json.dumps(out["overall"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
