#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_evidence_region_feature_faithfulness_v2.py

Feature-space faithfulness validation for System 1 evidence regions.

目标：
验证 System 1 框出的 bbox 是否在 DINO/token feature space 中
对 LinearProbe 的 fake-real logit margin 产生反事实影响。

核心思想：
1. 原图只前向 DINOv3 一次，得到 token_sequence。
2. 找到 selected bbox 覆盖的 patch tokens。
3. 将这些 patch tokens 替换为全图 patch token mean。
4. 用完全相同的 LinearProbe scoring path 重新计算：
   logit_margin = fake_logit - real_logit
5. 与随机同面积 bbox 对照比较。

输出：
- feature_faithfulness_v2_results.csv
- feature_faithfulness_v2_summary.csv
- failed_cases.jsonl

重点指标：
- margin_drop_selected
- margin_drop_random_mean
- margin_drop_advantage
- used_token_coverage
- used_token_hit_rate
"""

import os
import sys
import json
import glob
import math
import random
import argparse
import traceback
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torchvision import transforms


# ============================================================
# Basic utils
# ============================================================

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


def natural_key(s: str):
    import re
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(s))]


def clamp_bbox_xyxy(bbox: List[float], w: int, h: int) -> List[int]:
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, round(float(x1)))))
    y1 = int(max(0, min(h - 1, round(float(y1)))))
    x2 = int(max(x1 + 1, min(w, round(float(x2)))))
    y2 = int(max(y1 + 1, min(h, round(float(y2)))))
    return [x1, y1, x2, y2]


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


# ============================================================
# File / region loading
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
# Token grid / bbox mapping
# ============================================================

def infer_patch_grid(num_patch_tokens: int) -> Tuple[int, int]:
    g = int(round(math.sqrt(num_patch_tokens)))
    if g * g != num_patch_tokens:
        raise ValueError(f"Cannot infer square grid from {num_patch_tokens} patch tokens")
    return g, g


def bbox_to_patch_abs_indices(
    bbox: List[int],
    img_w: int,
    img_h: int,
    gh: int,
    gw: int,
    num_prefix: int,
) -> List[int]:
    """
    bbox 是 224 image space 中的 xyxy。
    返回 token_sequence 中的 absolute indices，即 num_prefix + patch_idx。
    """
    x1, y1, x2, y2 = bbox
    out = []

    for yy in range(gh):
        for xx in range(gw):
            cx = (xx + 0.5) / gw * img_w
            cy = (yy + 0.5) / gh * img_h
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                patch_idx = yy * gw + xx
                out.append(num_prefix + patch_idx)

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
    max_tries: int = 1000,
) -> List[List[int]]:
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    if bw >= w or bh >= h:
        return []

    boxes = []
    tries = 0

    while len(boxes) < n and tries < max_tries:
        tries += 1
        x1 = rng.randint(0, max(0, w - bw))
        y1 = rng.randint(0, max(0, h - bh))
        b = [x1, y1, x1 + bw, y1 + bh]

        if iou_xyxy(b, bbox) > max_iou_with_selected:
            continue
        if any(iou_xyxy(b, old) > 0.80 for old in boxes):
            continue

        boxes.append(b)

    return boxes


# ============================================================
# System 1 loading / scoring
# ============================================================

def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
    ])


def load_system1_from_local_code(code_dir: str, ckpt_path: str, device: str):
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

    official_fisher_indices = [int(x) for x in official_fisher_indices]

    linear_probe = LinearProbe(
        input_dim=probe_state["fc.weight"].shape[1],
        num_classes=probe_state["fc.weight"].shape[0],
    ).to(device)
    linear_probe.load_state_dict(probe_state, strict=True)
    linear_probe.eval()

    return model, linear_probe, official_fisher_indices, num_prefix, REAL_CLASS_IDX, FAKE_CLASS_IDX


@torch.no_grad()
def extract_token_sequence(img: Image.Image, model, transform, device: str) -> torch.Tensor:
    x = transform(img.convert("RGB")).unsqueeze(0).to(device)
    outputs = model(x, return_protocol=True)
    token_sequence = outputs["token_sequence"]
    if token_sequence.ndim != 3 or token_sequence.shape[0] != 1:
        raise RuntimeError(f"Unexpected token_sequence shape: {tuple(token_sequence.shape)}")
    return token_sequence


@torch.no_grad()
def score_from_token_sequence(
    token_sequence: torch.Tensor,
    linear_probe,
    official_fisher_indices: List[int],
    real_class_idx: int,
    fake_class_idx: int,
) -> Dict[str, float]:
    """
    严格复用 System 1 scoring path:
    selected_tokens = token_sequence[:, official_fisher_indices, :]
    logits = linear_probe(selected_tokens)
    """
    idx = torch.as_tensor(official_fisher_indices, dtype=torch.long, device=token_sequence.device)
    selected_tokens = token_sequence[:, idx, :]
    logits = linear_probe(selected_tokens)

    probs = torch.softmax(logits, dim=-1)[0]
    fake_prob = float(probs[fake_class_idx].detach().cpu().item())
    real_prob = float(probs[real_class_idx].detach().cpu().item())

    fake_logit = float(logits[0, fake_class_idx].detach().cpu().item())
    real_logit = float(logits[0, real_class_idx].detach().cpu().item())
    margin = fake_logit - real_logit

    return {
        "fake_prob": fake_prob,
        "real_prob": real_prob,
        "fake_logit": fake_logit,
        "real_logit": real_logit,
        "logit_margin": margin,
    }


def replace_tokens_with_global_patch_mean(
    token_sequence: torch.Tensor,
    abs_indices_to_replace: List[int],
    num_prefix: int,
) -> torch.Tensor:
    """
    将指定 token 替换成全图 patch token mean。
    注意：只替换 token_sequence 中的对应 index。
    """
    t2 = token_sequence.clone()

    if len(abs_indices_to_replace) == 0:
        return t2

    n_total = token_sequence.shape[1]
    patch_tokens = token_sequence[:, num_prefix:, :]
    replace_vec = patch_tokens.mean(dim=1, keepdim=False)[0]

    valid = [i for i in abs_indices_to_replace if 0 <= i < n_total]
    if not valid:
        return t2

    idx = torch.as_tensor(valid, dtype=torch.long, device=token_sequence.device)
    t2[0, idx, :] = replace_vec
    return t2


def compute_token_contribution_scores(
    token_sequence: torch.Tensor,
    linear_probe,
    real_class_idx: int,
    fake_class_idx: int,
    num_prefix: int,
) -> np.ndarray:
    """
    返回 196 个 patch token 的 fake-real 线性方向分数。
    这是 token-level 诊断使用，不改变主 scoring path。
    """
    patch_tokens = token_sequence[:, num_prefix:, :][0]
    w_real = linear_probe.fc.weight[real_class_idx]
    w_fake = linear_probe.fc.weight[fake_class_idx]
    delta_w = torch.nn.functional.normalize(w_fake - w_real, dim=0)
    patch_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)
    scores = torch.matmul(patch_norm, delta_w)
    return scores.detach().cpu().numpy()


# ============================================================
# Per-case experiment
# ============================================================

def run_one_case(
    row: pd.Series,
    args,
    model,
    linear_probe,
    official_fisher_indices,
    num_prefix,
    real_cls,
    fake_cls,
    transform,
    rng: random.Random,
) -> Dict[str, Any]:
    case_name = str(row["case_name"])
    case_dir = os.path.join(args.cases_root, case_name)

    img_path = find_original_224(case_dir)
    if img_path is None:
        raise FileNotFoundError(f"original_224 not found: {case_dir}")

    img = Image.open(img_path).convert("RGB")
    img_w, img_h = img.size

    report_obj = load_report_json_from_phasec_row(row)

    evidence_path = find_evidence_full(case_dir)
    evidence_obj = read_json(evidence_path) if evidence_path else None

    region, region_source = choose_top_region(report_obj, evidence_obj)
    if region is None:
        raise RuntimeError("no valid evidence region found")

    bbox_raw = parse_bbox_from_region(region)
    if bbox_raw is None:
        raise RuntimeError("cannot parse bbox from region")

    bbox = clamp_bbox_xyxy(bbox_raw, img_w, img_h)

    token_sequence = extract_token_sequence(img, model, transform, args.device)

    n_total = token_sequence.shape[1]
    num_patch_tokens = n_total - num_prefix
    gh, gw = infer_patch_grid(num_patch_tokens)

    full_score = score_from_token_sequence(
        token_sequence,
        linear_probe,
        official_fisher_indices,
        real_cls,
        fake_cls,
    )

    selected_abs = bbox_to_patch_abs_indices(
        bbox=bbox,
        img_w=img_w,
        img_h=img_h,
        gh=gh,
        gw=gw,
        num_prefix=num_prefix,
    )

    official_set = set(int(x) for x in official_fisher_indices)
    selected_set = set(int(x) for x in selected_abs)
    selected_used = sorted(list(selected_set & official_set))

    t_selected = replace_tokens_with_global_patch_mean(
        token_sequence,
        abs_indices_to_replace=selected_abs,
        num_prefix=num_prefix,
    )

    selected_score = score_from_token_sequence(
        t_selected,
        linear_probe,
        official_fisher_indices,
        real_cls,
        fake_cls,
    )

    # random boxes
    random_boxes = sample_random_boxes_like(
        bbox=bbox,
        w=img_w,
        h=img_h,
        n=args.num_random,
        rng=rng,
        max_iou_with_selected=0.30,
    )
    if not random_boxes:
        raise RuntimeError("failed to sample random boxes")

    rand_scores = []
    rand_num_tokens = []
    rand_num_used = []

    for rb in random_boxes:
        rb_abs = bbox_to_patch_abs_indices(
            bbox=rb,
            img_w=img_w,
            img_h=img_h,
            gh=gh,
            gw=gw,
            num_prefix=num_prefix,
        )
        rb_set = set(int(x) for x in rb_abs)
        rb_used = sorted(list(rb_set & official_set))

        t_rand = replace_tokens_with_global_patch_mean(
            token_sequence,
            abs_indices_to_replace=rb_abs,
            num_prefix=num_prefix,
        )
        sc = score_from_token_sequence(
            t_rand,
            linear_probe,
            official_fisher_indices,
            real_cls,
            fake_cls,
        )
        rand_scores.append(sc)
        rand_num_tokens.append(len(rb_abs))
        rand_num_used.append(len(rb_used))

    # center box
    center_bbox = center_box_like(bbox, img_w, img_h)
    center_abs = bbox_to_patch_abs_indices(
        bbox=center_bbox,
        img_w=img_w,
        img_h=img_h,
        gh=gh,
        gw=gw,
        num_prefix=num_prefix,
    )
    center_used = sorted(list(set(center_abs) & official_set))

    t_center = replace_tokens_with_global_patch_mean(
        token_sequence,
        abs_indices_to_replace=center_abs,
        num_prefix=num_prefix,
    )
    center_score = score_from_token_sequence(
        t_center,
        linear_probe,
        official_fisher_indices,
        real_cls,
        fake_cls,
    )

    # token-level diagnostic
    contrib_scores = compute_token_contribution_scores(
        token_sequence,
        linear_probe,
        real_cls,
        fake_cls,
        num_prefix,
    )
    selected_patch_indices = [i - num_prefix for i in selected_abs if i >= num_prefix]
    selected_patch_indices = [i for i in selected_patch_indices if 0 <= i < len(contrib_scores)]

    if selected_patch_indices:
        selected_token_score_mean = float(np.mean(contrib_scores[selected_patch_indices]))
        selected_token_score_max = float(np.max(contrib_scores[selected_patch_indices]))
    else:
        selected_token_score_mean = float("nan")
        selected_token_score_max = float("nan")

    # margin/prob drops
    margin_full = full_score["logit_margin"]
    prob_full = full_score["fake_prob"]

    margin_selected = selected_score["logit_margin"]
    prob_selected = selected_score["fake_prob"]

    rand_margins = np.array([s["logit_margin"] for s in rand_scores], dtype=np.float32)
    rand_probs = np.array([s["fake_prob"] for s in rand_scores], dtype=np.float32)

    center_margin = center_score["logit_margin"]
    center_prob = center_score["fake_prob"]

    margin_drop_selected = margin_full - margin_selected
    margin_drop_random_mean = float(np.mean(margin_full - rand_margins))
    margin_drop_random_std = float(np.std(margin_full - rand_margins))
    margin_drop_center = margin_full - center_margin

    prob_drop_selected = prob_full - prob_selected
    prob_drop_random_mean = float(np.mean(prob_full - rand_probs))
    prob_drop_random_std = float(np.std(prob_full - rand_probs))
    prob_drop_center = prob_full - center_prob

    num_tokens_in_bbox = len(selected_abs)
    num_used_tokens_in_bbox = len(selected_used)

    used_token_coverage = (
        num_used_tokens_in_bbox / max(num_tokens_in_bbox, 1)
    )
    used_token_hit_rate = (
        num_used_tokens_in_bbox / max(len(official_fisher_indices), 1)
    )

    rand_used_mean = float(np.mean(rand_num_used))
    rand_tokens_mean = float(np.mean(rand_num_tokens))

    return {
        "case_name": case_name,
        "case_dir": case_dir,
        "image_path": img_path,
        "region_source": region_source,

        "display_verdict": row.get("display_verdict", ""),
        "source_status": row.get("source_status", ""),
        "evidence_support_level": row.get("evidence_support_level", ""),
        "basis_mode": row.get("basis_mode", ""),

        "selected_bbox": json.dumps(bbox, ensure_ascii=False),
        "selected_area_ratio": ((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])) / float(img_w * img_h),

        "num_patch_tokens": num_patch_tokens,
        "grid_h": gh,
        "grid_w": gw,
        "num_official_fisher_indices": len(official_fisher_indices),

        "num_tokens_in_bbox": num_tokens_in_bbox,
        "num_used_tokens_in_bbox": num_used_tokens_in_bbox,
        "used_token_coverage": used_token_coverage,
        "used_token_hit_rate": used_token_hit_rate,

        "random_num_tokens_mean": rand_tokens_mean,
        "random_num_used_tokens_mean": rand_used_mean,

        "fake_prob_full": prob_full,
        "real_prob_full": full_score["real_prob"],
        "fake_logit_full": full_score["fake_logit"],
        "real_logit_full": full_score["real_logit"],
        "margin_full": margin_full,

        "fake_prob_after_selected_replace": prob_selected,
        "margin_after_selected_replace": margin_selected,

        "prob_drop_selected": prob_drop_selected,
        "prob_drop_random_mean": prob_drop_random_mean,
        "prob_drop_random_std": prob_drop_random_std,
        "prob_drop_center": prob_drop_center,
        "prob_drop_advantage": prob_drop_selected - prob_drop_random_mean,

        "margin_drop_selected": margin_drop_selected,
        "margin_drop_random_mean": margin_drop_random_mean,
        "margin_drop_random_std": margin_drop_random_std,
        "margin_drop_center": margin_drop_center,
        "margin_drop_advantage": margin_drop_selected - margin_drop_random_mean,
        "margin_drop_advantage_over_center": margin_drop_selected - margin_drop_center,

        "fake_prob_after_random_replace_mean": float(np.mean(rand_probs)),
        "margin_after_random_replace_mean": float(np.mean(rand_margins)),

        "fake_prob_after_center_replace": center_prob,
        "margin_after_center_replace": center_margin,

        "selected_token_score_mean": selected_token_score_mean,
        "selected_token_score_max": selected_token_score_max,

        "region_score_max": safe_float(region.get("score_max", region.get("best_score_max", 0.0))),
        "region_contrast_ratio": safe_float(region.get("contrast_ratio", region.get("best_contrast_ratio", 0.0))),
        "region_token_contrast": safe_float(region.get("token_contrast", region.get("best_token_contrast", 0.0))),
    }


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phasec_reports", required=True)
    ap.add_argument("--cases_root", default="aigi_now_eval/probe_genimage/cases_unique")
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--code_dir", default="xai_code_pack_clean_v1")
    ap.add_argument("--ckpt_path", default="checkpoints/AIGCDetectionBenchmark/linear_probe.pth")

    ap.add_argument("--only_gt_fake", action="store_true")
    ap.add_argument("--only_with_regions", action="store_true")
    ap.add_argument("--max_cases", type=int, default=0)

    ap.add_argument("--num_random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
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

    print(f"[*] num_prefix = {num_prefix}")
    print(f"[*] num official_fisher_indices = {len(official_fisher_indices)}")

    df = pd.read_csv(args.phasec_reports, encoding="utf-8-sig")

    if args.only_gt_fake:
        df = df[df["case_name"].astype(str).str.contains("__Fake__")].copy()

    if args.only_with_regions and "num_focus_regions" in df.columns:
        df = df[df["num_focus_regions"].fillna(0).astype(int) > 0].copy()

    if args.max_cases and args.max_cases > 0:
        df = df.head(args.max_cases).copy()

    print(f"[*] Candidate rows: {len(df)}")

    rows = []
    failures = []

    for _, row in df.iterrows():
        case_name = str(row["case_name"])

        try:
            out = run_one_case(
                row=row,
                args=args,
                model=model,
                linear_probe=linear_probe,
                official_fisher_indices=official_fisher_indices,
                num_prefix=num_prefix,
                real_cls=real_cls,
                fake_cls=fake_cls,
                transform=transform,
                rng=rng,
            )
            rows.append(out)

            print(
                f"[{len(rows):03d}] {case_name} | "
                f"prob={out['fake_prob_full']:.4f} | "
                f"margin={out['margin_full']:.3f} | "
                f"m_drop_sel={out['margin_drop_selected']:.3f} | "
                f"m_drop_rand={out['margin_drop_random_mean']:.3f} | "
                f"adv={out['margin_drop_advantage']:.3f} | "
                f"used={out['num_used_tokens_in_bbox']}/{out['num_tokens_in_bbox']}"
            )

        except Exception as e:
            failures.append({
                "case_name": case_name,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[!] failed {case_name}: {repr(e)}")

    result_csv = os.path.join(args.output_dir, "feature_faithfulness_v2_results.csv")
    summary_csv = os.path.join(args.output_dir, "feature_faithfulness_v2_summary.csv")
    failed_jsonl = os.path.join(args.output_dir, "failed_cases.jsonl")

    pd.DataFrame(rows).to_csv(result_csv, index=False, encoding="utf-8-sig")
    write_jsonl(failed_jsonl, failures)

    if rows:
        res = pd.DataFrame(rows)

        summary = {
            "num_success": len(res),
            "num_failed": len(failures),

            "mean_fake_prob_full": res["fake_prob_full"].mean(),
            "mean_margin_full": res["margin_full"].mean(),

            "mean_margin_drop_selected": res["margin_drop_selected"].mean(),
            "mean_margin_drop_random": res["margin_drop_random_mean"].mean(),
            "mean_margin_drop_center": res["margin_drop_center"].mean(),
            "mean_margin_drop_advantage": res["margin_drop_advantage"].mean(),
            "margin_drop_advantage_positive_rate": float((res["margin_drop_advantage"] > 0).mean()),

            "mean_prob_drop_selected": res["prob_drop_selected"].mean(),
            "mean_prob_drop_random": res["prob_drop_random_mean"].mean(),
            "mean_prob_drop_advantage": res["prob_drop_advantage"].mean(),
            "prob_drop_advantage_positive_rate": float((res["prob_drop_advantage"] > 0).mean()),

            "mean_num_tokens_in_bbox": res["num_tokens_in_bbox"].mean(),
            "mean_num_used_tokens_in_bbox": res["num_used_tokens_in_bbox"].mean(),
            "mean_used_token_coverage": res["used_token_coverage"].mean(),
            "mean_used_token_hit_rate": res["used_token_hit_rate"].mean(),
        }

        pd.DataFrame([summary]).to_csv(summary_csv, index=False, encoding="utf-8-sig")

        print("\n[+] Summary")
        for k, v in summary.items():
            print(f"{k}: {v}")

    print("\n[+] Finished")
    print("[*] result :", result_csv)
    print("[*] summary:", summary_csv)
    print("[*] failed :", failed_jsonl)


if __name__ == "__main__":
    main()