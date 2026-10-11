#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_evidence_region_faithfulness_v1.py

目标：
验证 System 1 框出的 candidate evidence region 是否不是随机框。

核心实验：
1. Deletion test:
   删除 / 模糊 System 1 选中区域，看 fake_prob 下降是否大于随机同面积区域。

2. Retention test:
   只保留 System 1 选中区域，其余区域模糊，看 fake_prob 是否高于随机同面积区域。

输出：
- faithfulness_results.csv
- faithfulness_summary.csv
- failed_cases.jsonl

注意：
- 第一版只做 top-1 evidence region。
- 默认使用 original_224.png，因为 bbox 通常是 224 坐标，避免 raw/model 坐标混乱。
- 遮挡方式使用 Gaussian blur，不用黑块，避免引入强 OOD artifact。
"""

import os
import sys
import json
import glob
import random
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter

import torch
from torchvision import transforms


# ============================================================
# Basic utils
# ============================================================

VALID_EXTS = (".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG", ".webp", ".WEBP")


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def write_jsonl(path: str, rows: List[Dict[str, Any]]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def safe_float(x, default=0.0):
    try:
        if x is None:
            return default
        return float(x)
    except Exception:
        return default


def safe_int(x, default=0):
    try:
        if x is None:
            return default
        return int(x)
    except Exception:
        return default


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def iou_xyxy(a: List[int], b: List[int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def clamp_bbox_xyxy(bbox: List[float], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, round(float(x1)))))
    y1 = int(max(0, min(h - 1, round(float(y1)))))
    x2 = int(max(x1 + 1, min(w, round(float(x2)))))
    y2 = int(max(y1 + 1, min(h, round(float(y2)))))
    return [x1, y1, x2, y2]


# ============================================================
# Find files / regions
# ============================================================

def find_original_224(case_dir: str) -> Optional[str]:
    xs = sorted(glob.glob(os.path.join(case_dir, "*_original_224.png")), key=natural_key)
    return xs[0] if xs else None


def find_evidence_full(case_dir: str) -> Optional[str]:
    xs = sorted(glob.glob(os.path.join(case_dir, "*_evidence_full.json")), key=natural_key)
    return xs[0] if xs else None


def load_report_json_from_phasec_row(row: pd.Series) -> Optional[Dict[str, Any]]:
    p = str(row.get("report_json", ""))
    if p and os.path.exists(p):
        return read_json(p)
    return None


def parse_bbox_from_region(region: Dict[str, Any]) -> Optional[List[float]]:
    """
    尽量兼容不同字段：
    bbox / bbox_224 / bbox_model / bbox_xyxy / box / rect
    """
    keys = [
        "bbox_224",
        "bbox_model",
        "bbox",
        "bbox_xyxy",
        "xyxy",
        "box",
        "rect",
        "crop_box",
        "patch_bbox",
        "patch_box",
    ]

    def dict_to_bbox(d):
        for ks in [["x1", "y1", "x2", "y2"], ["left", "top", "right", "bottom"]]:
            if all(k in d for k in ks):
                return [d[ks[0]], d[ks[1]], d[ks[2]], d[ks[3]]]
        if all(k in d for k in ["x", "y", "w", "h"]):
            x, y, w, h = d["x"], d["y"], d["w"], d["h"]
            return [x, y, x + w, y + h]
        return None

    for k in keys:
        if k not in region:
            continue
        v = region[k]

        if isinstance(v, (list, tuple)) and len(v) == 4:
            x1, y1, a, b = [float(x) for x in v]
            # 优先当 xyxy；如果明显像 xywh，也兼容
            if a > x1 and b > y1:
                return [x1, y1, a, b]
            if a > 0 and b > 0:
                return [x1, y1, x1 + a, y1 + b]

        if isinstance(v, dict):
            bb = dict_to_bbox(v)
            if bb is not None:
                return bb

    bb = dict_to_bbox(region)
    return bb


def choose_top_region(report_obj: Optional[Dict[str, Any]], evidence_obj: Optional[Dict[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    优先从 Phase C user_report 的 focus_regions 取；
    如果没有，则从 evidence_full 的 validated_evidence_objects / proposal_objects 取。
    """
    candidates = []

    if isinstance(report_obj, dict):
        regs = report_obj.get("focus_regions", [])
        if isinstance(regs, list):
            for r in regs:
                if isinstance(r, dict) and parse_bbox_from_region(r) is not None:
                    candidates.append((r, "phasec_focus_regions"))

    if not candidates and isinstance(evidence_obj, dict):
        for key in ["validated_evidence_objects", "proposal_objects", "regions"]:
            regs = evidence_obj.get(key, [])
            if isinstance(regs, list):
                for r in regs:
                    if isinstance(r, dict) and parse_bbox_from_region(r) is not None:
                        candidates.append((r, f"evidence_full:{key}"))
                if candidates:
                    break

    if not candidates:
        return None, "none"

    def score_region(item):
        r, _src = item
        tier = str(r.get("tier", r.get("status", "")))
        tier_score = 2 if "High" in tier or "高" in tier else 1
        s1 = safe_float(r.get("score_max", r.get("best_score_max", 0.0)))
        s2 = safe_float(r.get("contrast_ratio", r.get("best_contrast_ratio", 0.0)))
        s3 = safe_float(r.get("token_contrast", r.get("best_token_contrast", 0.0)))
        return (tier_score, s1, s2, s3)

    candidates = sorted(candidates, key=score_region, reverse=True)
    return candidates[0]


# ============================================================
# Image perturbations
# ============================================================

def blur_inside_bbox(img: Image.Image, bbox: List[int], blur_radius: float = 12.0) -> Image.Image:
    """
    删除实验：只模糊 bbox 内区域。
    """
    out = img.convert("RGB").copy()
    blurred = out.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    x1, y1, x2, y2 = bbox
    crop = blurred.crop((x1, y1, x2, y2))
    out.paste(crop, (x1, y1))
    return out


def keep_only_bbox(img: Image.Image, bbox: List[int], blur_radius: float = 12.0) -> Image.Image:
    """
    保留实验：全图模糊，只把 bbox 内原图贴回去。
    """
    raw = img.convert("RGB")
    out = raw.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    x1, y1, x2, y2 = bbox
    crop = raw.crop((x1, y1, x2, y2))
    out.paste(crop, (x1, y1))
    return out


def center_box_like(bbox: List[int], w: int, h: int) -> List[int]:
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]
    cx = w // 2
    cy = h // 2
    x1 = int(cx - bw / 2)
    y1 = int(cy - bh / 2)
    return clamp_bbox_xyxy([x1, y1, x1 + bw, y1 + bh], w, h)


def sample_random_boxes_like(
    bbox: List[int],
    w: int,
    h: int,
    n: int,
    rng: random.Random,
    max_iou_with_selected: float = 0.30,
    max_tries: int = 500,
) -> List[List[int]]:
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    boxes = []
    tries = 0

    if bw >= w or bh >= h:
        return boxes

    while len(boxes) < n and tries < max_tries:
        tries += 1
        x1 = rng.randint(0, max(0, w - bw))
        y1 = rng.randint(0, max(0, h - bh))
        b = [x1, y1, x1 + bw, y1 + bh]

        if iou_xyxy(b, bbox) > max_iou_with_selected:
            continue

        # 防止随机框之间全重叠，稍微做一点去重
        if any(iou_xyxy(b, old) > 0.80 for old in boxes):
            continue

        boxes.append(b)

    return boxes


# ============================================================
# System 1 scoring
# ============================================================

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
    ])


def load_system1_from_local_code(code_dir: str, ckpt_path: str, device: str):
    """
    尽量复用你当前工程里的 test_system1_protocol.py。
    """
    sys.path.insert(0, code_dir)
    sys.path.insert(0, ".")

    from test_system1_protocol import DinoV3Model, LinearProbe, REAL_CLASS_IDX, FAKE_CLASS_IDX

    model = DinoV3Model(model_name="dinov3_vit_7b", pool_type="patch_avg").to(device)
    model.eval()

    num_prefix = getattr(model.backbone, "num_prefix_tokens", 5)

    ckpt = torch.load(ckpt_path, map_location=device)
    official_fisher_indices = ckpt.get("token_indices", None)
    probe_state = ckpt.get("probe_state_dict", ckpt.get("model_state_dict", None))

    if official_fisher_indices is None:
        raise RuntimeError("checkpoint missing token_indices")
    if probe_state is None:
        raise RuntimeError("checkpoint missing probe_state_dict/model_state_dict")

    linear_probe = LinearProbe(
        input_dim=probe_state["fc.weight"].shape[1],
        num_classes=probe_state["fc.weight"].shape[0],
    ).to(device)
    linear_probe.load_state_dict(probe_state, strict=True)
    linear_probe.eval()

    return model, linear_probe, official_fisher_indices, num_prefix, REAL_CLASS_IDX, FAKE_CLASS_IDX


@torch.no_grad()
def score_images(
    images: List[Image.Image],
    model,
    linear_probe,
    official_fisher_indices,
    fake_class_idx: int,
    transform,
    device: str,
    batch_size: int = 4,
) -> List[float]:
    """
    返回每张图的 fake_prob。
    """
    probs_out = []

    for st in range(0, len(images), batch_size):
        batch_imgs = images[st:st + batch_size]
        xs = []
        for im in batch_imgs:
            xs.append(transform(im.convert("RGB")))
        x = torch.stack(xs, dim=0).to(device)

        outputs = model(x, return_protocol=True)
        token_sequence = outputs["token_sequence"]

        selected_tokens = token_sequence[:, official_fisher_indices, :]
        logits = linear_probe(selected_tokens)
        probs = torch.softmax(logits, dim=-1)[:, fake_class_idx]

        probs_out.extend([float(v) for v in probs.detach().cpu().tolist()])

    return probs_out


# ============================================================
# Main experiment
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasec_reports", required=True)
    ap.add_argument("--cases_root", default="aigi_now_eval/probe_genimage/cases_unique")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--code_dir", default="xai_code_pack_clean_v1")
    ap.add_argument("--ckpt_path", default="checkpoints/AIGCDetectionBenchmark/linear_probe.pth")

    ap.add_argument("--only_fake", action="store_true")
    ap.add_argument("--only_with_regions", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0)

    ap.add_argument("--num_random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--blur_radius", type=float, default=12.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()

    ensure_dir(args.output_dir)

    rng = random.Random(args.seed)
    transform = build_transform()

    print(f"[*] Loading System 1 on {args.device}")
    model, linear_probe, official_fisher_indices, num_prefix, real_cls, fake_cls = load_system1_from_local_code(
        code_dir=args.code_dir,
        ckpt_path=args.ckpt_path,
        device=args.device,
    )

    df = pd.read_csv(args.phasec_reports, encoding="utf-8-sig")

    if args.only_fake and "display_verdict" in df.columns:
        df = df[df["display_verdict"].astype(str) == "疑似人工智能生成"].copy()

    if args.only_with_regions and "num_focus_regions" in df.columns:
        df = df[df["num_focus_regions"].fillna(0).astype(int) > 0].copy()

    if args.max_cases and args.max_cases > 0:
        df = df.head(args.max_cases).copy()

    print(f"[*] Candidate rows: {len(df)}")

    rows = []
    failures = []

    for idx, row in df.iterrows():
        case_name = str(row["case_name"])
        case_dir = os.path.join(args.cases_root, case_name)

        try:
            img_path = find_original_224(case_dir)
            if img_path is None:
                raise FileNotFoundError(f"original_224 not found: {case_dir}")

            img = Image.open(img_path).convert("RGB")
            w, h = img.size

            report_obj = load_report_json_from_phasec_row(row)

            evidence_path = find_evidence_full(case_dir)
            evidence_obj = read_json(evidence_path) if evidence_path else None

            region, region_source = choose_top_region(report_obj, evidence_obj)
            if region is None:
                raise RuntimeError("no valid region found")

            bbox_raw = parse_bbox_from_region(region)
            if bbox_raw is None:
                raise RuntimeError("cannot parse bbox from top region")

            bbox = clamp_bbox_xyxy(bbox_raw, w, h)
            bw = bbox[2] - bbox[0]
            bh = bbox[3] - bbox[1]
            area_ratio = (bw * bh) / float(w * h)

            random_boxes = sample_random_boxes_like(
                bbox=bbox,
                w=w,
                h=h,
                n=args.num_random,
                rng=rng,
                max_iou_with_selected=0.30,
            )

            if len(random_boxes) == 0:
                raise RuntimeError("failed to sample random boxes")

            center_bbox = center_box_like(bbox, w, h)

            # 组装待评分图像
            full_img = img
            remove_selected_img = blur_inside_bbox(img, bbox, blur_radius=args.blur_radius)
            keep_selected_img = keep_only_bbox(img, bbox, blur_radius=args.blur_radius)

            remove_random_imgs = [
                blur_inside_bbox(img, rb, blur_radius=args.blur_radius)
                for rb in random_boxes
            ]
            keep_random_imgs = [
                keep_only_bbox(img, rb, blur_radius=args.blur_radius)
                for rb in random_boxes
            ]

            remove_center_img = blur_inside_bbox(img, center_bbox, blur_radius=args.blur_radius)
            keep_center_img = keep_only_bbox(img, center_bbox, blur_radius=args.blur_radius)

            all_imgs = (
                [full_img, remove_selected_img, keep_selected_img, remove_center_img, keep_center_img]
                + remove_random_imgs
                + keep_random_imgs
            )

            scores = score_images(
                all_imgs,
                model=model,
                linear_probe=linear_probe,
                official_fisher_indices=official_fisher_indices,
                fake_class_idx=fake_cls,
                transform=transform,
                device=args.device,
                batch_size=args.batch_size,
            )

            fake_prob_full = scores[0]
            fake_prob_remove_selected = scores[1]
            fake_prob_keep_selected = scores[2]
            fake_prob_remove_center = scores[3]
            fake_prob_keep_center = scores[4]

            rand_start = 5
            rand_mid = rand_start + len(remove_random_imgs)
            fake_prob_remove_random = scores[rand_start:rand_mid]
            fake_prob_keep_random = scores[rand_mid:rand_mid + len(keep_random_imgs)]

            drop_selected = fake_prob_full - fake_prob_remove_selected
            drop_random_mean = fake_prob_full - float(np.mean(fake_prob_remove_random))
            drop_random_std = float(np.std(fake_prob_remove_random))
            drop_center = fake_prob_full - fake_prob_remove_center

            keep_random_mean = float(np.mean(fake_prob_keep_random))
            keep_random_std = float(np.std(fake_prob_keep_random))

            drop_advantage = drop_selected - drop_random_mean
            keep_advantage = fake_prob_keep_selected - keep_random_mean

            row_out = {
                "case_name": case_name,
                "case_dir": case_dir,
                "image_path": img_path,
                "region_source": region_source,

                "display_verdict": row.get("display_verdict", ""),
                "source_status": row.get("source_status", ""),
                "evidence_support_level": row.get("evidence_support_level", ""),
                "basis_mode": row.get("basis_mode", ""),

                "selected_bbox": json.dumps(bbox, ensure_ascii=False),
                "selected_w": bw,
                "selected_h": bh,
                "selected_area_ratio": area_ratio,
                "num_random": len(random_boxes),

                "fake_prob_full": fake_prob_full,

                "fake_prob_remove_selected": fake_prob_remove_selected,
                "drop_selected": drop_selected,

                "fake_prob_remove_random_mean": float(np.mean(fake_prob_remove_random)),
                "fake_prob_remove_random_std": drop_random_std,
                "drop_random_mean": drop_random_mean,

                "fake_prob_remove_center": fake_prob_remove_center,
                "drop_center": drop_center,

                "drop_advantage": drop_advantage,
                "drop_advantage_over_center": drop_selected - drop_center,

                "fake_prob_keep_selected": fake_prob_keep_selected,
                "fake_prob_keep_random_mean": keep_random_mean,
                "fake_prob_keep_random_std": keep_random_std,
                "fake_prob_keep_center": fake_prob_keep_center,

                "keep_advantage": keep_advantage,
                "keep_advantage_over_center": fake_prob_keep_selected - fake_prob_keep_center,

                "region_score_max": safe_float(region.get("score_max", region.get("best_score_max", 0.0))),
                "region_contrast_ratio": safe_float(region.get("contrast_ratio", region.get("best_contrast_ratio", 0.0))),
                "region_token_contrast": safe_float(region.get("token_contrast", region.get("best_token_contrast", 0.0))),
            }

            rows.append(row_out)

            print(
                f"[{len(rows):03d}] {case_name} | "
                f"full={fake_prob_full:.4f} | "
                f"drop_sel={drop_selected:.4f} | "
                f"drop_rand={drop_random_mean:.4f} | "
                f"keep_sel={fake_prob_keep_selected:.4f} | "
                f"keep_rand={keep_random_mean:.4f}"
            )

        except Exception as e:
            failures.append({
                "case_name": case_name,
                "case_dir": case_dir,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[!] failed {case_name}: {repr(e)}")

    out_csv = os.path.join(args.output_dir, "faithfulness_results.csv")
    fail_path = os.path.join(args.output_dir, "failed_cases.jsonl")
    summary_csv = os.path.join(args.output_dir, "faithfulness_summary.csv")

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    write_jsonl(fail_path, failures)

    if rows:
        res = pd.DataFrame(rows)
        summary = {
            "num_success": len(res),
            "num_failed": len(failures),

            "mean_fake_prob_full": res["fake_prob_full"].mean(),

            "mean_drop_selected": res["drop_selected"].mean(),
            "mean_drop_random": res["drop_random_mean"].mean(),
            "mean_drop_center": res["drop_center"].mean(),
            "mean_drop_advantage": res["drop_advantage"].mean(),
            "drop_advantage_positive_rate": float((res["drop_advantage"] > 0).mean()),

            "mean_keep_selected": res["fake_prob_keep_selected"].mean(),
            "mean_keep_random": res["fake_prob_keep_random_mean"].mean(),
            "mean_keep_center": res["fake_prob_keep_center"].mean(),
            "mean_keep_advantage": res["keep_advantage"].mean(),
            "keep_advantage_positive_rate": float((res["keep_advantage"] > 0).mean()),
        }

        pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

        print("\n[+] Summary")
        for k, v in summary.items():
            print(f"{k}: {v}")

    print("\n[+] Finished")
    print("[*] results:", out_csv)
    print("[*] summary:", summary_csv)
    print("[*] failed :", fail_path)


if __name__ == "__main__":
    main()