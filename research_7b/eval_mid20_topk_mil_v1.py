#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from sklearn.metrics import roc_auc_score, average_precision_score
except Exception:
    roc_auc_score = None
    average_precision_score = None


def get_first_key(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of keys {keys} found. Existing keys: {list(d.keys())}")


def load_payload(features_pt: str):
    d = torch.load(features_pt, map_location="cpu")

    patch_features = get_first_key(d, ["patch_features", "patch_feats", "patch_tokens"])
    labels = get_first_key(d, ["labels", "label", "y"]).long()

    sources = d.get("sources", d.get("source", ["unknown"] * len(labels)))
    paths = d.get("paths", d.get("path", [""] * len(labels)))
    case_names = d.get("case_names", d.get("case_name", [""] * len(labels)))

    if isinstance(sources, np.ndarray):
        sources = sources.tolist()
    if isinstance(paths, np.ndarray):
        paths = paths.tolist()
    if isinstance(case_names, np.ndarray):
        case_names = case_names.tolist()

    return {
        "patch_features": patch_features,
        "labels": labels,
        "sources": sources,
        "paths": paths,
        "case_names": case_names,
    }


class PatchFeatureDataset(Dataset):
    def __init__(self, payload, indices=None):
        self.patch_features = payload["patch_features"]
        self.labels = payload["labels"]
        if indices is None:
            self.indices = list(range(len(self.labels)))
        else:
            self.indices = list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        idx = self.indices[i]
        return self.patch_features[idx], self.labels[idx]


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

    def pool_scores(self, scores, pool="topk", topk=10, lse_tau=1.0):
        if pool == "topk":
            k = min(topk, scores.shape[1])
            vals = torch.topk(scores, k=k, dim=1).values
            return vals.mean(dim=1)
        if pool == "lse":
            return lse_tau * torch.logsumexp(scores / lse_tau, dim=1) - lse_tau * math.log(scores.shape[1])
        if pool == "mean":
            return scores.mean(dim=1)
        raise ValueError(pool)

    def forward(self, x, pool="topk", topk=10, lse_tau=1.0):
        scores = self.score_patches(x)
        image_logit = self.pool_scores(scores, pool=pool, topk=topk, lse_tau=lse_tau)
        return image_logit, scores


def normalize_patches(x, mu, std):
    return (x.float() - mu.view(1, 1, -1)) / std.view(1, 1, -1)


def classification_metrics(labels, logits):
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    logits_np = logits.detach().cpu().numpy()
    probs_np = 1.0 / (1.0 + np.exp(-logits_np))
    preds_np = (probs_np >= 0.5).astype(np.int64)

    out = {
        "n": int(len(labels_np)),
        "acc": float((preds_np == labels_np).mean()),
        "mean_prob_fake_real": float(probs_np[labels_np == 0].mean()) if np.any(labels_np == 0) else None,
        "mean_prob_fake_fake": float(probs_np[labels_np == 1].mean()) if np.any(labels_np == 1) else None,
    }

    if len(np.unique(labels_np)) == 2 and roc_auc_score is not None:
        out["auc"] = float(roc_auc_score(labels_np, probs_np))
        out["ap"] = float(average_precision_score(labels_np, probs_np))
    else:
        out["auc"] = None
        out["ap"] = None

    return out


def edge_masks(grid=14):
    yy, xx = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    edge1 = (yy == 0) | (yy == grid - 1) | (xx == 0) | (xx == grid - 1)
    edge2 = (yy <= 1) | (yy >= grid - 2) | (xx <= 1) | (xx >= grid - 2)
    return edge1.reshape(-1), edge2.reshape(-1)


def normalized_entropy(indices, num_positions=196):
    if len(indices) == 0:
        return None
    counts = np.bincount(indices, minlength=num_positions).astype(np.float64)
    p = counts / counts.sum()
    p = p[p > 0]
    ent = -np.sum(p * np.log(p))
    return float(ent / np.log(num_positions))


def edge_metrics_for_group(score_maps, labels, group, grid=14):
    labels_np = labels.detach().cpu().numpy().astype(np.int64)
    scores = score_maps.detach().cpu().numpy()

    if group == "fake":
        idx = np.where(labels_np == 1)[0]
        support = scores[idx]
    elif group == "real":
        idx = np.where(labels_np == 0)[0]
        support = -scores[idx]
    elif group == "all":
        idx = np.arange(len(labels_np))
        sign = np.where(labels_np == 1, 1.0, -1.0)[:, None]
        support = scores * sign
    else:
        raise ValueError(group)

    if len(idx) == 0:
        return None

    e1, e2 = edge_masks(grid)
    top1 = np.argmax(support, axis=1)
    top3 = np.argsort(-support, axis=1)[:, :3]
    top10 = np.argsort(-support, axis=1)[:, :10]

    return {
        "n": int(len(idx)),
        "top1_edge1_rate": float(e1[top1].mean()),
        "top1_edge2_rate": float(e2[top1].mean()),
        "top3_edge1_fraction": float(e1[top3].mean()),
        "top3_edge2_fraction": float(e2[top3].mean()),
        "top10_edge1_fraction": float(e1[top10].mean()),
        "top10_edge2_fraction": float(e2[top10].mean()),
        "top1_position_entropy": normalized_entropy(top1, num_positions=grid * grid),
    }


@torch.no_grad()
def evaluate_indices(model, payload, indices, mu, std, args, device):
    model.eval()

    ds = PatchFeatureDataset(payload, indices=indices)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_logits = []
    all_scores = []
    all_labels = []

    for patches, labels in loader:
        patches = patches.to(device)
        labels = labels.to(device)

        patches = normalize_patches(patches, mu.to(device), std.to(device))
        logits, scores = model(
            patches,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
        )

        all_logits.append(logits.cpu())
        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return {
        "classification": classification_metrics(labels, logits),
        "edge_metrics": {
            "fake": edge_metrics_for_group(scores, labels, "fake", grid=args.grid),
            "real": edge_metrics_for_group(scores, labels, "real", grid=args.grid),
            "all": edge_metrics_for_group(scores, labels, "all", grid=args.grid),
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--features_pt", required=True)
    ap.add_argument("--output_json", required=True)

    ap.add_argument("--pool", default=None, choices=[None, "topk", "lse", "mean"])
    ap.add_argument("--topk", type=int, default=None)
    ap.add_argument("--lse_tau", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    feat_dim = int(ckpt_args.get("feat_dim", 4096))
    hidden_dim = int(ckpt_args.get("hidden_dim", 256))
    dropout = float(ckpt_args.get("dropout", 0.1))

    if args.pool is None:
        args.pool = ckpt_args.get("pool", "topk")
    if args.topk is None:
        args.topk = int(ckpt_args.get("topk", 10))
    if args.lse_tau is None:
        args.lse_tau = float(ckpt_args.get("lse_tau", 1.0))

    model = TopKMILScorer(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    mu = ckpt["mu"].float()
    std = ckpt["std"].float()

    payload = load_payload(args.features_pt)

    all_indices = list(range(len(payload["labels"])))
    overall = evaluate_indices(model, payload, all_indices, mu, std, args, device)

    by_source = {}
    sources = payload["sources"]
    for src in sorted(set(sources)):
        idx = [i for i, s in enumerate(sources) if s == src]
        by_source[src] = evaluate_indices(model, payload, idx, mu, std, args, device)

    out = {
        "checkpoint": args.checkpoint,
        "features_pt": args.features_pt,
        "pool": args.pool,
        "topk": args.topk,
        "overall": overall,
        "by_source": by_source,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("Saved:", args.output_json)
    print(json.dumps(out["overall"], indent=2))


if __name__ == "__main__":
    main()
