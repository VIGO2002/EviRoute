#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from collections import defaultdict

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

    if isinstance(paths, np.ndarray):
        paths = paths.tolist()
    if isinstance(sources, np.ndarray):
        sources = sources.tolist()

    return {
        "patch_features": patch_features,
        "labels": labels,
        "paths": paths,
        "sources": sources,
    }


def load_metadata(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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
        vals = torch.topk(scores, k=k, dim=1).values
        return vals.mean(dim=1)

    if pool == "lse":
        return lse_tau * torch.logsumexp(scores / lse_tau, dim=1) - lse_tau * math.log(scores.shape[1])

    if pool == "mean":
        return scores.mean(dim=1)

    raise ValueError(pool)


@torch.no_grad()
def compute_logits(model_pack, patch_features, device, batch_size=64):
    model = model_pack["model"]
    mu = model_pack["mu"]
    std = model_pack["std"]

    outs = []
    n = patch_features.shape[0]

    for i in range(0, n, batch_size):
        x = patch_features[i:i+batch_size].float().to(device)
        x = (x - mu.view(1, 1, -1)) / std.view(1, 1, -1)

        scores = model.score_patches(x)
        logits = pool_scores(
            scores,
            pool=model_pack["pool"],
            topk=model_pack["topk"],
            lse_tau=model_pack["lse_tau"],
        )
        outs.append(logits.cpu())

    return torch.cat(outs, dim=0).numpy()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def summarize_records(records):
    adv = np.array([r["advantage_logit"] for r in records], dtype=np.float64)
    topk = np.array([r["drop_topk_logit"] for r in records], dtype=np.float64)
    rand = np.array([r["drop_rand_mean_logit"] for r in records], dtype=np.float64)
    gt_mean = np.array([r["topk_gt_randmean"] for r in records], dtype=np.float64)
    pair = np.array([r["pairwise_gt_random_rate"] for r in records], dtype=np.float64)

    out = {
        "n": int(len(records)),
        "mean_drop_topk_logit": float(topk.mean()),
        "mean_drop_random_logit": float(rand.mean()),
        "mean_advantage_logit": float(adv.mean()),
        "median_advantage_logit": float(np.median(adv)),
        "topk_gt_randmean_rate": float(gt_mean.mean()),
        "pairwise_gt_random_rate": float(pair.mean()),
    }

    prob_adv = np.array([r["advantage_prob"] for r in records], dtype=np.float64)
    out.update({
        "mean_advantage_prob": float(prob_adv.mean()),
        "median_advantage_prob": float(np.median(prob_adv)),
    })

    return out


def evaluate_one_model(name, model_pack, orig_payload, pert_payload, metadata, device, batch_size):
    orig_logits_all = compute_logits(model_pack, orig_payload["patch_features"], device, batch_size=batch_size)
    pert_logits_all = compute_logits(model_pack, pert_payload["patch_features"], device, batch_size=batch_size)

    if len(pert_logits_all) != len(metadata):
        raise RuntimeError(f"Perturbed feature count {len(pert_logits_all)} != metadata count {len(metadata)}")

    grouped = defaultdict(list)

    for i, m in enumerate(metadata):
        orig_idx = int(m["orig_idx"])
        variant = m["variant"]

        orig_logit = float(orig_logits_all[orig_idx])
        pert_logit = float(pert_logits_all[i])

        orig_prob = float(sigmoid(orig_logit))
        pert_prob = float(sigmoid(pert_logit))

        row = {
            "metadata_row": i,
            "orig_idx": orig_idx,
            "source": m["source"],
            "variant": variant,
            "rep": int(m["rep"]),
            "orig_logit": orig_logit,
            "pert_logit": pert_logit,
            "drop_logit": orig_logit - pert_logit,
            "orig_prob": orig_prob,
            "pert_prob": pert_prob,
            "drop_prob": orig_prob - pert_prob,
        }
        grouped[orig_idx].append(row)

    records = []
    by_source_records = defaultdict(list)

    for orig_idx, rows in grouped.items():
        topk_rows = [r for r in rows if r["variant"] == "topk"]
        rand_rows = [r for r in rows if r["variant"] == "rand"]

        if len(topk_rows) != 1 or len(rand_rows) == 0:
            continue

        t = topk_rows[0]
        rand_drops = np.array([r["drop_logit"] for r in rand_rows], dtype=np.float64)
        rand_prob_drops = np.array([r["drop_prob"] for r in rand_rows], dtype=np.float64)

        rec = {
            "orig_idx": orig_idx,
            "source": t["source"],
            "drop_topk_logit": float(t["drop_logit"]),
            "drop_rand_mean_logit": float(rand_drops.mean()),
            "drop_rand_std_logit": float(rand_drops.std()),
            "advantage_logit": float(t["drop_logit"] - rand_drops.mean()),
            "topk_gt_randmean": bool(t["drop_logit"] > rand_drops.mean()),
            "pairwise_gt_random_rate": float((t["drop_logit"] > rand_drops).mean()),
            "drop_topk_prob": float(t["drop_prob"]),
            "drop_rand_mean_prob": float(rand_prob_drops.mean()),
            "advantage_prob": float(t["drop_prob"] - rand_prob_drops.mean()),
            "orig_prob": float(t["orig_prob"]),
        }

        records.append(rec)
        by_source_records[t["source"]].append(rec)

    out = {
        "model_name": name,
        "overall": summarize_records(records),
        "by_source": {
            src: summarize_records(rs)
            for src, rs in sorted(by_source_records.items())
        },
        "per_sample": records,
    }

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--orig_features_pt", required=True)
    ap.add_argument("--pert_features_pt", required=True)
    ap.add_argument("--metadata_jsonl", required=True)
    ap.add_argument("--checkpoint_v2b", required=True)
    ap.add_argument("--checkpoint_naive", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    orig_payload = load_payload(args.orig_features_pt)
    pert_payload = load_payload(args.pert_features_pt)
    metadata = load_metadata(args.metadata_jsonl)

    v2b = load_model(args.checkpoint_v2b, device)
    naive = load_model(args.checkpoint_naive, device)

    out = {
        "orig_features_pt": args.orig_features_pt,
        "pert_features_pt": args.pert_features_pt,
        "metadata_jsonl": args.metadata_jsonl,
        "models": {}
    }

    out["models"]["v2b_lite"] = evaluate_one_model(
        "v2b_lite",
        v2b,
        orig_payload,
        pert_payload,
        metadata,
        device,
        args.batch_size,
    )

    out["models"]["naive_topk"] = evaluate_one_model(
        "naive_topk",
        naive,
        orig_payload,
        pert_payload,
        metadata,
        device,
        args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved:", args.output_json)
    print(json.dumps({
        "v2b_lite": out["models"]["v2b_lite"]["overall"],
        "naive_topk": out["models"]["naive_topk"]["overall"],
    }, indent=2))


if __name__ == "__main__":
    main()
