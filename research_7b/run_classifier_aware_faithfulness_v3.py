#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run_classifier_aware_faithfulness_v3.py

Classifier-aware feature-space faithfulness validation.

目标：
从 LinearProbe 实际使用的 official_fisher_indices 中选择高可疑 patch tokens，
构造 classifier-aware evidence tokens / bbox，
然后做 feature-space counterfactual replacement。

与 v2 的区别：
- v2 验证已有 bbox 是否覆盖 used tokens；
- v3 主动从 used tokens 中选择 high-suspicious tokens，生成 classifier-aware evidence region。

核心指标：
- margin_drop_selected
- margin_drop_random_box_mean
- margin_drop_random_used_tokens_mean
- margin_advantage_over_random_box
- margin_advantage_over_random_used_tokens
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
# File loading
# ============================================================

def find_original_224(case_dir: str) -> Optional[str]:
    xs = sorted(glob.glob(os.path.join(case_dir, "*_original_224.png")), key=natural_key)
    return xs[0] if xs else None


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
    t2 = token_sequence.clone()

    if len(abs_indices_to_replace) == 0:
        return t2

    n_total = token_sequence.shape[1]
    patch_tokens = token_sequence[:, num_prefix:, :]
    replace_vec = patch_tokens.mean(dim=1, keepdim=False)[0]

    valid = [int(i) for i in abs_indices_to_replace if 0 <= int(i) < n_total]
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
    patch_tokens = token_sequence[:, num_prefix:, :][0]

    w_real = linear_probe.fc.weight[real_class_idx]
    w_fake = linear_probe.fc.weight[fake_class_idx]

    delta_w = torch.nn.functional.normalize(w_fake - w_real, dim=0)
    patch_norm = torch.nn.functional.normalize(patch_tokens, dim=-1)

    scores = torch.matmul(patch_norm, delta_w)
    return scores.detach().cpu().numpy()


# ============================================================
# Token index / bbox helpers
# ============================================================

def infer_patch_grid(num_patch_tokens: int) -> Tuple[int, int]:
    g = int(round(math.sqrt(num_patch_tokens)))
    if g * g != num_patch_tokens:
        raise ValueError(f"Cannot infer square grid from {num_patch_tokens}")
    return g, g


def normalize_official_indices_to_abs(
    official_fisher_indices: List[int],
    num_prefix: int,
    n_total: int,
) -> List[int]:
    """
    优先认为 official_fisher_indices 是 token_sequence absolute indices。
    如果没有任何 patch token 命中，则兼容 patch-relative indices。
    """
    num_patch = n_total - num_prefix

    abs_patch = [
        int(i) for i in official_fisher_indices
        if num_prefix <= int(i) < n_total
    ]

    if len(abs_patch) > 0:
        return sorted(set(abs_patch))

    # fallback: treat as patch-relative 0~195
    rel = [
        int(i) + num_prefix for i in official_fisher_indices
        if 0 <= int(i) < num_patch
    ]

    return sorted(set(rel))


def patch_abs_to_xy(abs_idx: int, num_prefix: int, gw: int) -> Tuple[int, int]:
    patch_idx = int(abs_idx) - num_prefix
    y = patch_idx // gw
    x = patch_idx % gw
    return x, y


def patch_abs_indices_to_bbox(
    abs_indices: List[int],
    img_w: int,
    img_h: int,
    gh: int,
    gw: int,
    num_prefix: int,
    margin_cells: int = 0,
) -> List[int]:
    xs = []
    ys = []

    for abs_idx in abs_indices:
        x, y = patch_abs_to_xy(abs_idx, num_prefix, gw)
        xs.append(x)
        ys.append(y)

    if not xs:
        return [0, 0, img_w, img_h]

    x1_cell = max(0, min(xs) - margin_cells)
    y1_cell = max(0, min(ys) - margin_cells)
    x2_cell = min(gw - 1, max(xs) + margin_cells)
    y2_cell = min(gh - 1, max(ys) + margin_cells)

    cell_w = img_w / gw
    cell_h = img_h / gh

    x1 = x1_cell * cell_w
    y1 = y1_cell * cell_h
    x2 = (x2_cell + 1) * cell_w
    y2 = (y2_cell + 1) * cell_h

    return clamp_bbox_xyxy([x1, y1, x2, y2], img_w, img_h)


def bbox_to_patch_abs_indices(
    bbox: List[int],
    img_w: int,
    img_h: int,
    gh: int,
    gw: int,
    num_prefix: int,
) -> List[int]:
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


def choose_classifier_aware_tokens(
    official_abs_patch_indices: List[int],
    contribution_scores: np.ndarray,
    num_prefix: int,
    top_k: int,
    only_positive: bool = True,
) -> List[int]:
    """
    从 official Fisher patch tokens 中选 fake-real contribution score 最高的 tokens。
    """
    scored = []

    for abs_idx in official_abs_patch_indices:
        rel = int(abs_idx) - num_prefix
        if 0 <= rel < len(contribution_scores):
            s = float(contribution_scores[rel])
            scored.append((abs_idx, s))

    if not scored:
        return []

    scored = sorted(scored, key=lambda x: x[1], reverse=True)

    if only_positive:
        pos = [(idx, s) for idx, s in scored if s > 0]
        if len(pos) > 0:
            scored = pos

    k = min(top_k, len(scored))
    return [idx for idx, _ in scored[:k]]


def sample_random_official_token_sets(
    official_abs_patch_indices: List[int],
    selected_abs: List[int],
    k: int,
    n: int,
    rng: random.Random,
) -> List[List[int]]:
    selected_set = set(int(x) for x in selected_abs)
    pool = [int(x) for x in official_abs_patch_indices if int(x) not in selected_set]

    if len(pool) < k:
        pool = [int(x) for x in official_abs_patch_indices]

    if len(pool) == 0:
        return []

    out = []
    for _ in range(n):
        if len(pool) >= k:
            out.append(rng.sample(pool, k))
        else:
            out.append([rng.choice(pool) for _ in range(k)])
    return out


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

    token_sequence = extract_token_sequence(img, model, transform, args.device)

    n_total = token_sequence.shape[1]
    num_patch_tokens = n_total - num_prefix
    gh, gw = infer_patch_grid(num_patch_tokens)

    official_abs_patch_indices = normalize_official_indices_to_abs(
        official_fisher_indices=official_fisher_indices,
        num_prefix=num_prefix,
        n_total=n_total,
    )

    if len(official_abs_patch_indices) == 0:
        raise RuntimeError("no official patch indices found")

    full_score = score_from_token_sequence(
        token_sequence=token_sequence,
        linear_probe=linear_probe,
        official_fisher_indices=official_fisher_indices,
        real_class_idx=real_cls,
        fake_class_idx=fake_cls,
    )

    contribution_scores = compute_token_contribution_scores(
        token_sequence=token_sequence,
        linear_probe=linear_probe,
        real_class_idx=real_cls,
        fake_class_idx=fake_cls,
        num_prefix=num_prefix,
    )

    selected_abs = choose_classifier_aware_tokens(
        official_abs_patch_indices=official_abs_patch_indices,
        contribution_scores=contribution_scores,
        num_prefix=num_prefix,
        top_k=args.top_used_tokens,
        only_positive=args.only_positive,
    )

    if len(selected_abs) == 0:
        raise RuntimeError("no selected classifier-aware tokens")

    selected_bbox = patch_abs_indices_to_bbox(
        abs_indices=selected_abs,
        img_w=img_w,
        img_h=img_h,
        gh=gh,
        gw=gw,
        num_prefix=num_prefix,
        margin_cells=args.margin_cells,
    )

    selected_bbox_abs = bbox_to_patch_abs_indices(
        bbox=selected_bbox,
        img_w=img_w,
        img_h=img_h,
        gh=gh,
        gw=gw,
        num_prefix=num_prefix,
    )

    official_set = set(int(x) for x in official_abs_patch_indices)
    selected_bbox_used = sorted(list(set(selected_bbox_abs) & official_set))

    # For faithfulness, replace the selected classifier-aware tokens directly.
    t_selected = replace_tokens_with_global_patch_mean(
        token_sequence=token_sequence,
        abs_indices_to_replace=selected_abs,
        num_prefix=num_prefix,
    )

    selected_score = score_from_token_sequence(
        token_sequence=t_selected,
        linear_probe=linear_probe,
        official_fisher_indices=official_fisher_indices,
        real_class_idx=real_cls,
        fake_class_idx=fake_cls,
    )

    # Baseline 1: random area-matched bbox
    random_boxes = sample_random_boxes_like(
        bbox=selected_bbox,
        w=img_w,
        h=img_h,
        n=args.num_random,
        rng=rng,
        max_iou_with_selected=0.30,
    )

    random_box_scores = []
    random_box_num_used = []

    for rb in random_boxes:
        rb_abs = bbox_to_patch_abs_indices(
            bbox=rb,
            img_w=img_w,
            img_h=img_h,
            gh=gh,
            gw=gw,
            num_prefix=num_prefix,
        )
        rb_used = sorted(list(set(rb_abs) & official_set))
        random_box_num_used.append(len(rb_used))

        t_rb = replace_tokens_with_global_patch_mean(
            token_sequence=token_sequence,
            abs_indices_to_replace=rb_abs,
            num_prefix=num_prefix,
        )
        sc = score_from_token_sequence(
            token_sequence=t_rb,
            linear_probe=linear_probe,
            official_fisher_indices=official_fisher_indices,
            real_class_idx=real_cls,
            fake_class_idx=fake_cls,
        )
        random_box_scores.append(sc)

    # Baseline 2: random Fisher-token-matched sets
    random_used_token_sets = sample_random_official_token_sets(
        official_abs_patch_indices=official_abs_patch_indices,
        selected_abs=selected_abs,
        k=len(selected_abs),
        n=args.num_random,
        rng=rng,
    )

    random_used_scores = []
    for token_set in random_used_token_sets:
        t_ru = replace_tokens_with_global_patch_mean(
            token_sequence=token_sequence,
            abs_indices_to_replace=token_set,
            num_prefix=num_prefix,
        )
        sc = score_from_token_sequence(
            token_sequence=t_ru,
            linear_probe=linear_probe,
            official_fisher_indices=official_fisher_indices,
            real_class_idx=real_cls,
            fake_class_idx=fake_cls,
        )
        random_used_scores.append(sc)

    margin_full = full_score["logit_margin"]
    prob_full = full_score["fake_prob"]

    margin_selected = selected_score["logit_margin"]
    prob_selected = selected_score["fake_prob"]

    random_box_margins = np.array([s["logit_margin"] for s in random_box_scores], dtype=np.float32)
    random_box_probs = np.array([s["fake_prob"] for s in random_box_scores], dtype=np.float32)

    random_used_margins = np.array([s["logit_margin"] for s in random_used_scores], dtype=np.float32)
    random_used_probs = np.array([s["fake_prob"] for s in random_used_scores], dtype=np.float32)

    margin_drop_selected = margin_full - margin_selected
    prob_drop_selected = prob_full - prob_selected

    margin_drop_random_box = margin_full - random_box_margins
    prob_drop_random_box = prob_full - random_box_probs

    margin_drop_random_used = margin_full - random_used_margins
    prob_drop_random_used = prob_full - random_used_probs

    selected_rel = [int(x) - num_prefix for x in selected_abs]
    selected_scores = [
        float(contribution_scores[r])
        for r in selected_rel
        if 0 <= r < len(contribution_scores)
    ]

    return {
        "case_name": case_name,
        "case_dir": case_dir,
        "image_path": img_path,

        "display_verdict": row.get("display_verdict", ""),
        "source_status": row.get("source_status", ""),
        "evidence_support_level": row.get("evidence_support_level", ""),
        "basis_mode": row.get("basis_mode", ""),

        "fake_prob_full": prob_full,
        "real_prob_full": full_score["real_prob"],
        "fake_logit_full": full_score["fake_logit"],
        "real_logit_full": full_score["real_logit"],
        "margin_full": margin_full,

        "num_patch_tokens": num_patch_tokens,
        "grid_h": gh,
        "grid_w": gw,
        "num_official_fisher_indices_raw": len(official_fisher_indices),
        "num_official_patch_indices": len(official_abs_patch_indices),

        "selected_abs_tokens": json.dumps(selected_abs, ensure_ascii=False),
        "selected_bbox": json.dumps(selected_bbox, ensure_ascii=False),
        "selected_num_tokens": len(selected_abs),
        "selected_num_bbox_tokens": len(selected_bbox_abs),
        "selected_num_bbox_used_tokens": len(selected_bbox_used),
        "selected_bbox_used_hit_rate": len(selected_bbox_used) / max(len(official_abs_patch_indices), 1),

        "selected_token_score_mean": float(np.mean(selected_scores)) if selected_scores else float("nan"),
        "selected_token_score_max": float(np.max(selected_scores)) if selected_scores else float("nan"),
        "selected_token_score_min": float(np.min(selected_scores)) if selected_scores else float("nan"),

        "fake_prob_after_selected_replace": prob_selected,
        "margin_after_selected_replace": margin_selected,

        "prob_drop_selected": prob_drop_selected,
        "margin_drop_selected": margin_drop_selected,

        "fake_prob_after_random_box_replace_mean": float(np.mean(random_box_probs)) if len(random_box_probs) else float("nan"),
        "margin_after_random_box_replace_mean": float(np.mean(random_box_margins)) if len(random_box_margins) else float("nan"),
        "prob_drop_random_box_mean": float(np.mean(prob_drop_random_box)) if len(prob_drop_random_box) else float("nan"),
        "margin_drop_random_box_mean": float(np.mean(margin_drop_random_box)) if len(margin_drop_random_box) else float("nan"),
        "margin_drop_random_box_std": float(np.std(margin_drop_random_box)) if len(margin_drop_random_box) else float("nan"),
        "random_box_num_used_tokens_mean": float(np.mean(random_box_num_used)) if random_box_num_used else float("nan"),

        "fake_prob_after_random_used_replace_mean": float(np.mean(random_used_probs)) if len(random_used_probs) else float("nan"),
        "margin_after_random_used_replace_mean": float(np.mean(random_used_margins)) if len(random_used_margins) else float("nan"),
        "prob_drop_random_used_mean": float(np.mean(prob_drop_random_used)) if len(prob_drop_random_used) else float("nan"),
        "margin_drop_random_used_mean": float(np.mean(margin_drop_random_used)) if len(margin_drop_random_used) else float("nan"),
        "margin_drop_random_used_std": float(np.std(margin_drop_random_used)) if len(margin_drop_random_used) else float("nan"),

        "margin_advantage_over_random_box": margin_drop_selected - float(np.mean(margin_drop_random_box)) if len(margin_drop_random_box) else float("nan"),
        "margin_advantage_over_random_used": margin_drop_selected - float(np.mean(margin_drop_random_used)) if len(margin_drop_random_used) else float("nan"),

        "prob_advantage_over_random_box": prob_drop_selected - float(np.mean(prob_drop_random_box)) if len(prob_drop_random_box) else float("nan"),
        "prob_advantage_over_random_used": prob_drop_selected - float(np.mean(prob_drop_random_used)) if len(prob_drop_random_used) else float("nan"),

        "num_random": args.num_random,
        "top_used_tokens": args.top_used_tokens,
        "margin_cells": args.margin_cells,
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

    ap.add_argument("--top_used_tokens", type=int, default=3)
    ap.add_argument("--margin_cells", type=int, default=0)
    ap.add_argument("--num_random", type=int, default=20)
    ap.add_argument("--only_positive", action="store_true")

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
    print(f"[*] raw official_fisher_indices = {len(official_fisher_indices)}")

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
                f"drop_sel={out['margin_drop_selected']:.3f} | "
                f"drop_box={out['margin_drop_random_box_mean']:.3f} | "
                f"drop_used={out['margin_drop_random_used_mean']:.3f} | "
                f"adv_box={out['margin_advantage_over_random_box']:.3f} | "
                f"adv_used={out['margin_advantage_over_random_used']:.3f}"
            )

        except Exception as e:
            failures.append({
                "case_name": case_name,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            })
            print(f"[!] failed {case_name}: {repr(e)}")

    result_csv = os.path.join(args.output_dir, "classifier_aware_faithfulness_v3_results.csv")
    summary_csv = os.path.join(args.output_dir, "classifier_aware_faithfulness_v3_summary.csv")
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
            "mean_margin_drop_random_box": res["margin_drop_random_box_mean"].mean(),
            "mean_margin_drop_random_used": res["margin_drop_random_used_mean"].mean(),

            "mean_margin_advantage_over_random_box": res["margin_advantage_over_random_box"].mean(),
            "margin_advantage_over_random_box_positive_rate": float((res["margin_advantage_over_random_box"] > 0).mean()),

            "mean_margin_advantage_over_random_used": res["margin_advantage_over_random_used"].mean(),
            "margin_advantage_over_random_used_positive_rate": float((res["margin_advantage_over_random_used"] > 0).mean()),

            "mean_prob_drop_selected": res["prob_drop_selected"].mean(),
            "mean_prob_advantage_over_random_box": res["prob_advantage_over_random_box"].mean(),
            "mean_prob_advantage_over_random_used": res["prob_advantage_over_random_used"].mean(),

            "mean_selected_num_tokens": res["selected_num_tokens"].mean(),
            "mean_selected_num_bbox_tokens": res["selected_num_bbox_tokens"].mean(),
            "mean_selected_num_bbox_used_tokens": res["selected_num_bbox_used_tokens"].mean(),
            "mean_selected_bbox_used_hit_rate": res["selected_bbox_used_hit_rate"].mean(),

            "top_used_tokens": args.top_used_tokens,
            "margin_cells": args.margin_cells,
            "num_random": args.num_random,
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