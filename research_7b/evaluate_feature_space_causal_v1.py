#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import csv
import json
import math
import argparse
import random
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.nn as nn


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

    pool = args.get("pool", "topk")
    topk = int(args.get("topk", 10))
    lse_tau = float(args.get("lse_tau", 1.0))

    return {
        "model": model,
        "mu": mu,
        "std": std,
        "pool": pool,
        "topk": topk,
        "lse_tau": lse_tau,
    }


def pool_scores(scores, pool="topk", topk=10, lse_tau=1.0):
    if pool == "topk":
        k = min(topk, scores.shape[1])
        return torch.topk(scores, k=k, dim=1).values.mean(dim=1)

    if pool == "lse":
        return lse_tau * torch.logsumexp(scores / lse_tau, dim=1) - lse_tau * math.log(scores.shape[1])

    if pool == "mean":
        return scores.mean(dim=1)

    raise ValueError(pool)


@torch.no_grad()
def get_logit_and_scores(model_pack, raw_patch_features, device):
    x = raw_patch_features.unsqueeze(0).float().to(device)
    x = (x - model_pack["mu"].view(1, 1, -1)) / model_pack["std"].view(1, 1, -1)

    scores = model_pack["model"].score_patches(x)
    logit = pool_scores(
        scores,
        pool=model_pack["pool"],
        topk=model_pack["topk"],
        lse_tau=model_pack["lse_tau"],
    )

    return float(logit.item()), scores.squeeze(0).detach().cpu().numpy(), x.squeeze(0)


@torch.no_grad()
def get_logit_from_normalized(model_pack, x_norm):
    scores = model_pack["model"].score_patches(x_norm.unsqueeze(0))
    logit = pool_scores(
        scores,
        pool=model_pack["pool"],
        topk=model_pack["topk"],
        lse_tau=model_pack["lse_tau"],
    )
    return float(logit.item())


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


def read_index_csv(index_csv):
    rows = []
    with open(index_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def summarize(records):
    adv_logit = np.array([r["advantage_logit"] for r in records], dtype=np.float64)
    adv_prob = np.array([r["advantage_prob"] for r in records], dtype=np.float64)
    drop_topk = np.array([r["drop_topk_logit"] for r in records], dtype=np.float64)
    drop_rand = np.array([r["drop_rand_mean_logit"] for r in records], dtype=np.float64)
    gt = np.array([r["topk_gt_randmean"] for r in records], dtype=np.float64)
    pair = np.array([r["pairwise_gt_random_rate"] for r in records], dtype=np.float64)

    return {
        "n": int(len(records)),
        "mean_drop_topk_logit": float(drop_topk.mean()),
        "mean_drop_random_logit": float(drop_rand.mean()),
        "mean_advantage_logit": float(adv_logit.mean()),
        "median_advantage_logit": float(np.median(adv_logit)),
        "topk_gt_randmean_rate": float(gt.mean()),
        "pairwise_gt_random_rate": float(pair.mean()),
        "mean_advantage_prob": float(adv_prob.mean()),
        "median_advantage_prob": float(np.median(adv_prob)),
    }


def evaluate_model_with_v2b_topk(
    eval_name,
    eval_model_pack,
    select_model_pack,
    payload,
    indices,
    rng,
    topk=10,
    num_random=20,
    grid=14,
    device="cuda",
):
    records = []
    by_source = defaultdict(list)

    for orig_idx in indices:
        label = int(payload["labels"][orig_idx].item())
        if label != 1:
            continue

        raw_patch = payload["patch_features"][orig_idx].float()

        # 用 v2b-lite 选择 top-k
        _, select_scores, _ = get_logit_and_scores(select_model_pack, raw_patch, device)
        topk_ids = np.argsort(-select_scores)[:topk].tolist()

        # 用 eval model 评估 drop
        clean_logit, _, x_norm = get_logit_and_scores(eval_model_pack, raw_patch, device)
        clean_prob = sigmoid(clean_logit)

        x_topk = x_norm.clone()
        x_topk[topk_ids, :] = 0.0
        topk_logit = get_logit_from_normalized(eval_model_pack, x_topk)
        topk_prob = sigmoid(topk_logit)

        drop_topk_logit = clean_logit - topk_logit
        drop_topk_prob = clean_prob - topk_prob

        rand_drops_logit = []
        rand_drops_prob = []

        for _ in range(num_random):
            rand_ids = sample_ring_matched(topk_ids, rng, grid=grid)
            x_rand = x_norm.clone()
            x_rand[rand_ids, :] = 0.0

            rand_logit = get_logit_from_normalized(eval_model_pack, x_rand)
            rand_prob = sigmoid(rand_logit)

            rand_drops_logit.append(clean_logit - rand_logit)
            rand_drops_prob.append(clean_prob - rand_prob)

        rand_drops_logit = np.array(rand_drops_logit, dtype=np.float64)
        rand_drops_prob = np.array(rand_drops_prob, dtype=np.float64)

        rec = {
            "orig_idx": int(orig_idx),
            "source": payload["sources"][orig_idx],
            "case_name": payload["case_names"][orig_idx] if orig_idx < len(payload["case_names"]) else str(orig_idx),
            "clean_logit": float(clean_logit),
            "clean_prob": float(clean_prob),
            "drop_topk_logit": float(drop_topk_logit),
            "drop_rand_mean_logit": float(rand_drops_logit.mean()),
            "drop_rand_std_logit": float(rand_drops_logit.std()),
            "advantage_logit": float(drop_topk_logit - rand_drops_logit.mean()),
            "topk_gt_randmean": bool(drop_topk_logit > rand_drops_logit.mean()),
            "pairwise_gt_random_rate": float((drop_topk_logit > rand_drops_logit).mean()),
            "drop_topk_prob": float(drop_topk_prob),
            "drop_rand_mean_prob": float(rand_drops_prob.mean()),
            "advantage_prob": float(drop_topk_prob - rand_drops_prob.mean()),
            "topk_ids": topk_ids,
        }

        records.append(rec)
        by_source[rec["source"]].append(rec)

    return {
        "model_name": eval_name,
        "overall": summarize(records),
        "by_source": {src: summarize(rs) for src, rs in sorted(by_source.items())},
        "per_sample": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_pt", required=True)
    ap.add_argument("--index_csv", required=True)
    ap.add_argument("--checkpoint_v2b", required=True)
    ap.add_argument("--checkpoint_naive", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--num_random", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rng = random.Random(args.seed)

    payload = load_payload(args.features_pt)
    rows = read_index_csv(args.index_csv)
    indices = [int(r["idx"]) for r in rows if int(r.get("label", 1)) == 1]

    v2b = load_model(args.checkpoint_v2b, device)
    naive = load_model(args.checkpoint_naive, device)

    out = {
        "features_pt": args.features_pt,
        "index_csv": args.index_csv,
        "topk": args.topk,
        "num_random": args.num_random,
        "ablation": "normalized_feature_to_zero_train_mean",
        "topk_selector": "v2b_lite",
        "models": {}
    }

    out["models"]["v2b_lite"] = evaluate_model_with_v2b_topk(
        "v2b_lite",
        v2b,
        v2b,
        payload,
        indices,
        rng,
        topk=args.topk,
        num_random=args.num_random,
        device=device,
    )

    # 重置随机种子，保证 naive 用同一套 random sampling 逻辑
    rng = random.Random(args.seed)

    out["models"]["naive_topk"] = evaluate_model_with_v2b_topk(
        "naive_topk",
        naive,
        v2b,
        payload,
        indices,
        rng,
        topk=args.topk,
        num_random=args.num_random,
        device=device,
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved:", args.output_json)
    print(json.dumps({
        "v2b_lite": out["models"]["v2b_lite"]["overall"],
        "naive_topk": out["models"]["naive_topk"]["overall"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
