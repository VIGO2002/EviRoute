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

    return {
        "model": model,
        "mu": mu,
        "std": std,
        "pool": args.get("pool", "topk"),
        "topk": int(args.get("topk", 10)),
        "lse_tau": float(args.get("lse_tau", 1.0)),
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


def summarize(records):
    if len(records) == 0:
        return {
            "n": 0,
            "mean_drop_topk_logit": None,
            "mean_drop_random_logit": None,
            "mean_advantage_logit": None,
            "median_advantage_logit": None,
            "topk_gt_randmean_rate": None,
            "pairwise_gt_random_rate": None,
            "mean_advantage_prob": None,
            "median_advantage_prob": None,
            "mean_clean_prob": None,
        }

    adv = np.array([r["advantage_logit"] for r in records], dtype=np.float64)
    topk = np.array([r["drop_topk_logit"] for r in records], dtype=np.float64)
    rand = np.array([r["drop_rand_mean_logit"] for r in records], dtype=np.float64)
    gt = np.array([r["topk_gt_randmean"] for r in records], dtype=np.float64)
    pair = np.array([r["pairwise_gt_random_rate"] for r in records], dtype=np.float64)
    adv_prob = np.array([r["advantage_prob"] for r in records], dtype=np.float64)
    clean_prob = np.array([r["clean_prob"] for r in records], dtype=np.float64)

    return {
        "n": int(len(records)),
        "mean_drop_topk_logit": float(topk.mean()),
        "mean_drop_random_logit": float(rand.mean()),
        "mean_advantage_logit": float(adv.mean()),
        "median_advantage_logit": float(np.median(adv)),
        "topk_gt_randmean_rate": float(gt.mean()),
        "pairwise_gt_random_rate": float(pair.mean()),
        "mean_advantage_prob": float(adv_prob.mean()),
        "median_advantage_prob": float(np.median(adv_prob)),
        "mean_clean_prob": float(clean_prob.mean()),
    }


def evaluate_one_model(name, model_pack, payload, metadata, device, batch_size):
    logits_all = compute_logits(model_pack, payload["patch_features"], device, batch_size=batch_size)

    if len(logits_all) != len(metadata):
        raise RuntimeError(f"features count {len(logits_all)} != metadata count {len(metadata)}")

    grouped = defaultdict(list)

    for i, m in enumerate(metadata):
        logit = float(logits_all[i])
        prob = float(sigmoid(logit))

        row = {
            "metadata_row": i,
            "orig_idx": int(m["orig_idx"]),
            "source": m["source"],
            "variant": m["variant"],
            "rep": int(m["rep"]),
            "logit": logit,
            "prob": prob,
        }

        grouped[int(m["orig_idx"])].append(row)

    records = []
    by_source = defaultdict(list)

    for orig_idx, rows in grouped.items():
        clean_rows = [r for r in rows if r["variant"] == "clean"]
        topk_rows = [r for r in rows if r["variant"] == "topk"]
        rand_rows = [r for r in rows if r["variant"] == "rand"]

        if len(clean_rows) != 1 or len(topk_rows) != 1 or len(rand_rows) == 0:
            continue

        clean = clean_rows[0]
        topk = topk_rows[0]

        clean_logit = clean["logit"]
        clean_prob = clean["prob"]

        drop_topk_logit = clean_logit - topk["logit"]
        drop_topk_prob = clean_prob - topk["prob"]

        rand_drops_logit = np.array([clean_logit - r["logit"] for r in rand_rows], dtype=np.float64)
        rand_drops_prob = np.array([clean_prob - r["prob"] for r in rand_rows], dtype=np.float64)

        rec = {
            "orig_idx": orig_idx,
            "source": clean["source"],
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
        }

        records.append(rec)
        by_source[rec["source"]].append(rec)

    high05 = [r for r in records if r["clean_prob"] >= 0.5]
    high03 = [r for r in records if r["clean_prob"] >= 0.3]
    low05 = [r for r in records if r["clean_prob"] < 0.5]

    return {
        "model_name": name,
        "overall_all": summarize(records),
        "overall_high_conf_05": summarize(high05),
        "overall_high_conf_03": summarize(high03),
        "overall_low_conf_05": summarize(low05),
        "by_source": {
            src: summarize(rs)
            for src, rs in sorted(by_source.items())
        },
        "per_sample": records,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pert_features_pt", required=True)
    ap.add_argument("--metadata_jsonl", required=True)
    ap.add_argument("--checkpoint_v2b", required=True)
    ap.add_argument("--checkpoint_naive", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    payload = load_payload(args.pert_features_pt)
    metadata = load_metadata(args.metadata_jsonl)

    v2b = load_model(args.checkpoint_v2b, device)
    naive = load_model(args.checkpoint_naive, device)

    out = {
        "pert_features_pt": args.pert_features_pt,
        "metadata_jsonl": args.metadata_jsonl,
        "baseline": "clean_224",
        "drop": "clean_logit - perturbed_logit",
        "models": {},
    }

    out["models"]["v2b_lite"] = evaluate_one_model(
        "v2b_lite",
        v2b,
        payload,
        metadata,
        device,
        args.batch_size,
    )

    out["models"]["naive_topk"] = evaluate_one_model(
        "naive_topk",
        naive,
        payload,
        metadata,
        device,
        args.batch_size,
    )

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved:", args.output_json)

    for name, obj in out["models"].items():
        print("\n==", name, "==")
        print("ALL")
        print(json.dumps(obj["overall_all"], indent=2, ensure_ascii=False))
        print("HIGH >= 0.5")
        print(json.dumps(obj["overall_high_conf_05"], indent=2, ensure_ascii=False))
        print("HIGH >= 0.3")
        print(json.dumps(obj["overall_high_conf_03"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
