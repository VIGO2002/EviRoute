#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from PIL import Image, ImageFilter


def load_metadata(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_metadata(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[int(r["orig_idx"])].append(r)
    return grouped


def gray_image(img):
    arr = np.asarray(img.convert("RGB")).astype(np.float32) / 255.0
    gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    return gray


def conv2d_same(x, k):
    kh, kw = k.shape
    ph, pw = kh // 2, kw // 2
    xp = np.pad(x, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(x, dtype=np.float32)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            out[i, j] = np.sum(xp[i:i+kh, j:j+kw] * k)
    return out


def patch_slice(pid, grid=14, size=224):
    cell = size // grid
    r, c = pid // grid, pid % grid
    y0, y1 = r * cell, (r + 1) * cell
    x0, x1 = c * cell, (c + 1) * cell
    return y0, y1, x0, x1


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


def compute_patch_metrics(img):
    img = img.convert("RGB").resize((224, 224), Image.BICUBIC)
    rgb = np.asarray(img).astype(np.float32) / 255.0
    gray = gray_image(img)

    blur = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=1.0)).convert("RGB")).astype(np.float32) / 255.0
    blur_gray = 0.299 * blur[:, :, 0] + 0.587 * blur[:, :, 1] + 0.114 * blur[:, :, 2]

    residual = gray - blur_gray

    k_lap = np.array([[0, 1, 0],
                      [1, -4, 1],
                      [0, 1, 0]], dtype=np.float32)
    lap = conv2d_same(gray, k_lap)

    gy = np.zeros_like(gray)
    gx = np.zeros_like(gray)
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    grad_energy_map = gx ** 2 + gy ** 2

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


def robust_abs_z(values):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    scale = 1.4826 * mad + 1e-8
    return np.abs(values - med) / scale


def summarize_records(records, metric_names):
    out = {"n": len(records), "metrics": {}}

    for m in metric_names:
        raw_delta = np.asarray([r[f"{m}_raw_delta"] for r in records], dtype=np.float64)
        atyp_delta = np.asarray([r[f"{m}_atyp_delta"] for r in records], dtype=np.float64)

        out["metrics"][m] = {
            "mean_raw_delta": float(raw_delta.mean()),
            "median_raw_delta": float(np.median(raw_delta)),
            "raw_topk_gt_random_rate": float((raw_delta > 0).mean()),

            "mean_atyp_delta": float(atyp_delta.mean()),
            "median_atyp_delta": float(np.median(atyp_delta)),
            "atyp_topk_gt_random_rate": float((atyp_delta > 0).mean()),
        }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata_jsonl", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--output_csv", required=True)
    args = ap.parse_args()

    rows = load_metadata(args.metadata_jsonl)
    grouped = group_metadata(rows)

    records = []
    metric_names = ["lap_var", "grad_energy", "residual_abs", "fft_hf_ratio", "color_var"]

    for orig_idx, rs in sorted(grouped.items()):
        clean_rows = [r for r in rs if r["variant"] == "clean"]
        topk_rows = [r for r in rs if r["variant"] == "topk"]
        rand_rows = [r for r in rs if r["variant"] == "rand"]

        if len(clean_rows) != 1 or len(topk_rows) != 1 or len(rand_rows) == 0:
            continue

        clean_path = clean_rows[0]["path"]
        source = clean_rows[0]["source"]
        case_name = clean_rows[0].get("case_name", str(orig_idx))

        img = Image.open(clean_path).convert("RGB")
        patch_metrics = compute_patch_metrics(img)

        topk_ids = topk_rows[0]["topk_ids"]

        rec = {
            "orig_idx": orig_idx,
            "source": source,
            "case_name": case_name,
            "clean_path": clean_path,
        }

        for m in metric_names:
            vals = patch_metrics[m]
            atyp = robust_abs_z(vals)

            topk_raw = float(vals[topk_ids].mean())
            topk_atyp = float(atyp[topk_ids].mean())

            rand_raw_vals = []
            rand_atyp_vals = []

            for rr in rand_rows:
                ids = rr["perturb_ids"]
                rand_raw_vals.append(float(vals[ids].mean()))
                rand_atyp_vals.append(float(atyp[ids].mean()))

            rand_raw = float(np.mean(rand_raw_vals))
            rand_atyp = float(np.mean(rand_atyp_vals))

            rec[f"{m}_topk_raw"] = topk_raw
            rec[f"{m}_rand_raw"] = rand_raw
            rec[f"{m}_raw_delta"] = topk_raw - rand_raw

            rec[f"{m}_topk_atyp"] = topk_atyp
            rec[f"{m}_rand_atyp"] = rand_atyp
            rec[f"{m}_atyp_delta"] = topk_atyp - rand_atyp

        records.append(rec)

        if len(records) % 20 == 0:
            print(f"[Done] {len(records)} samples")

    overall = summarize_records(records, metric_names)

    by_source = {}
    for src in sorted(set(r["source"] for r in records)):
        src_records = [r for r in records if r["source"] == src]
        by_source[src] = summarize_records(src_records, metric_names)

    out = {
        "metadata_jsonl": args.metadata_jsonl,
        "comparison": "v2b_topk_vs_ring_matched_random",
        "stat_definition": {
            "raw_delta": "mean(metric(topk patches)) - mean(metric(random patches))",
            "atyp_delta": "mean(abs robust z-score of topk patches) - mean(abs robust z-score of random patches)",
        },
        "overall": overall,
        "by_source": by_source,
        "per_sample": records,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    with open(args.output_csv, "w", encoding="utf-8") as f:
        cols = list(records[0].keys()) if records else []
        f.write(",".join(cols) + "\n")
        for r in records:
            vals = []
            for c in cols:
                v = r[c]
                if isinstance(v, str):
                    v = v.replace(",", "_")
                vals.append(str(v))
            f.write(",".join(vals) + "\n")

    print("Saved JSON:", args.output_json)
    print("Saved CSV:", args.output_csv)
    print(json.dumps(overall, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
