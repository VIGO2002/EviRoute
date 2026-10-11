#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
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


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def auc_ap_acc(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s).astype(float)

    try:
        from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
        auc = float(roc_auc_score(y, s))
        ap = float(average_precision_score(y, s))
        pred = (s >= 0.5).astype(int)
        acc = float(accuracy_score(y, pred))
        return auc, ap, acc
    except Exception as e:
        print("[Warn] sklearn metric failed:", e)
        pred = (s >= 0.5).astype(int)
        acc = float((pred == y).mean())
        return None, None, acc


def ring_flags(pid, grid=14):
    r = pid // grid
    c = pid % grid
    edge1 = (r == 0 or r == grid - 1 or c == 0 or c == grid - 1)
    edge2 = (r <= 1 or r >= grid - 2 or c <= 1 or c >= grid - 2)
    return edge1, edge2


def summarize_group(rows):
    y = [r["label"] for r in rows]
    p = [r["prob"] for r in rows]

    auc, ap, acc = auc_ap_acc(y, p)

    fake_rows = [r for r in rows if r["label"] == 1]
    real_rows = [r for r in rows if r["label"] == 0]

    def edge_rate(sub, key):
        if len(sub) == 0:
            return None
        return float(np.mean([r[key] for r in sub]))

    return {
        "n": len(rows),
        "n_real": len(real_rows),
        "n_fake": len(fake_rows),
        "auc": auc,
        "ap": ap,
        "acc_thr05": acc,
        "fake_top1_edge1": edge_rate(fake_rows, "top1_edge1"),
        "fake_top1_edge2": edge_rate(fake_rows, "top1_edge2"),
        "real_top1_edge1": edge_rate(real_rows, "top1_edge1"),
        "real_top1_edge2": edge_rate(real_rows, "top1_edge2"),
        "mean_prob_real": float(np.mean([r["prob"] for r in real_rows])) if real_rows else None,
        "mean_prob_fake": float(np.mean([r["prob"] for r in fake_rows])) if fake_rows else None,
    }


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_pt", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output_json", required=True)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    data = torch.load(args.features_pt, map_location="cpu")
    patch_features = get_first_key(data, ["patch_features", "patch_feats", "patch_tokens"])
    labels = get_first_key(data, ["labels", "label", "y"]).long()

    paths = data.get("paths", data.get("path", [""] * len(labels)))
    sources = data.get("sources", data.get("source", ["unknown"] * len(labels)))

    if isinstance(paths, np.ndarray):
        paths = paths.tolist()
    if isinstance(sources, np.ndarray):
        sources = sources.tolist()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ckpt_args = ckpt.get("args", {})

    model = TopKMILScorer(
        feat_dim=int(ckpt_args.get("feat_dim", patch_features.shape[-1])),
        hidden_dim=int(ckpt_args.get("hidden_dim", 256)),
        dropout=float(ckpt_args.get("dropout", 0.1)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    mu = ckpt["mu"].float().to(device)
    std = ckpt["std"].float().to(device)

    records = []
    n = len(labels)

    for st in range(0, n, args.batch_size):
        ed = min(n, st + args.batch_size)

        x = patch_features[st:ed].float().to(device)
        x = (x - mu.view(1, 1, -1)) / std.view(1, 1, -1)

        scores = model.score_patches(x).detach().cpu().numpy()
        topk_idx = np.argsort(-scores, axis=1)[:, :args.topk]
        logits = np.take_along_axis(scores, topk_idx, axis=1).mean(axis=1)
        probs = sigmoid(logits)

        top1 = topk_idx[:, 0]

        for i in range(ed - st):
            pid = int(top1[i])
            e1, e2 = ring_flags(pid)
            records.append({
                "idx": st + i,
                "path": paths[st + i],
                "source": sources[st + i],
                "label": int(labels[st + i].item()),
                "logit": float(logits[i]),
                "prob": float(probs[i]),
                "top1_pid": pid,
                "top1_edge1": bool(e1),
                "top1_edge2": bool(e2),
            })

        print(f"[Done] {ed}/{n}")

    by_source = defaultdict(list)
    for r in records:
        by_source[r["source"]].append(r)

    out = {
        "features_pt": args.features_pt,
        "checkpoint": args.checkpoint,
        "topk": args.topk,
        "overall": summarize_group(records),
        "by_source": {src: summarize_group(rs) for src, rs in sorted(by_source.items())},
        "per_sample": records,
    }

    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print("[Saved]", args.output_json)
    print(json.dumps(out["overall"], indent=2, ensure_ascii=False))
    print(json.dumps(out["by_source"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
