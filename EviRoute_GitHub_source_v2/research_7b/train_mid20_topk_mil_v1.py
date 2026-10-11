#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import math
import argparse
import random
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


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_first_key(d: Dict[str, Any], keys: List[str]):
    for k in keys:
        if k in d:
            return d[k]
    raise KeyError(f"None of keys {keys} found. Existing keys: {list(d.keys())}")


def load_split(feature_dir: str, split: str):
    path = os.path.join(feature_dir, f"{split}.pt")
    d = torch.load(path, map_location="cpu")

    patch_features = get_first_key(d, ["patch_features", "patch_feats", "patch_tokens"])
    mean_features = get_first_key(d, ["mean_features", "features", "x"])
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
        "mean_features": mean_features,
        "labels": labels,
        "paths": paths,
        "sources": sources,
        "case_names": case_names,
    }


class PatchFeatureDataset(Dataset):
    def __init__(self, payload: Dict[str, Any]):
        self.patch_features = payload["patch_features"]
        self.labels = payload["labels"]

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
        return self.patch_features[idx], self.labels[idx]


class TopKMILScorer(nn.Module):
    def __init__(
        self,
        feat_dim: int = 4096,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_patches(self, x):
        # x: [B, P, D]
        return self.scorer(x).squeeze(-1)  # [B, P]

    def pool_scores(
        self,
        scores,
        pool: str = "topk",
        topk: int = 10,
        lse_tau: float = 1.0,
        token_dropout: float = 0.0,
    ):
        # scores: [B, P]
        if self.training and token_dropout > 0:
            B, P = scores.shape
            mask = torch.rand(B, P, device=scores.device) < token_dropout

            # avoid masking all tokens
            all_masked = mask.all(dim=1)
            if all_masked.any():
                mask[all_masked, 0] = False

            scores = scores.masked_fill(mask, -1e4)

        if pool == "topk":
            k = min(topk, scores.shape[1])
            vals = torch.topk(scores, k=k, dim=1).values
            return vals.mean(dim=1)

        if pool == "lse":
            # normalized log-sum-exp pooling
            return lse_tau * torch.logsumexp(scores / lse_tau, dim=1) - lse_tau * math.log(scores.shape[1])

        if pool == "mean":
            return scores.mean(dim=1)

        raise ValueError(f"Unknown pool: {pool}")

    def forward(
        self,
        x,
        pool: str = "topk",
        topk: int = 10,
        lse_tau: float = 1.0,
        token_dropout: float = 0.0,
    ):
        scores = self.score_patches(x)
        image_logit = self.pool_scores(
            scores,
            pool=pool,
            topk=topk,
            lse_tau=lse_tau,
            token_dropout=token_dropout,
        )
        return image_logit, scores


def compute_norm_from_train(train_payload):
    # Use train mean_features for stable feature normalization.
    x = train_payload["mean_features"].float()
    mu = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mu, std


def normalize_patches(x, mu, std):
    # x: [B, P, D]
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


def edge_metrics_for_group(score_maps, labels, group: str, grid=14):
    # score_maps: [N, 196], higher = fake evidence
    # For fake group, support = score.
    # For real group, support = -score.
    # For all group, support = score for fake, -score for real.
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
def evaluate_payload(model, payload, mu, std, args, device):
    model.eval()

    ds = PatchFeatureDataset(payload)
    loader = DataLoader(
        ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_logits = []
    all_scores = []
    all_labels = []

    for patches, labels in loader:
        patches = patches.to(device, non_blocking=False)
        labels = labels.to(device, non_blocking=False)

        patches = normalize_patches(patches, mu.to(device), std.to(device))
        logits, scores = model(
            patches,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
            token_dropout=0.0,
        )

        all_logits.append(logits.cpu())
        all_scores.append(scores.cpu())
        all_labels.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    scores = torch.cat(all_scores, dim=0)
    labels = torch.cat(all_labels, dim=0)

    out = {
        "classification": classification_metrics(labels, logits),
        "edge_metrics": {
            "fake": edge_metrics_for_group(scores, labels, "fake", grid=args.grid),
            "real": edge_metrics_for_group(scores, labels, "real", grid=args.grid),
            "all": edge_metrics_for_group(scores, labels, "all", grid=args.grid),
        },
    }
    return out


def train_one_epoch(model, loader, mu, std, optimizer, criterion, args, device):
    model.train()
    total_loss = 0.0
    total_n = 0

    for patches, labels in loader:
        patches = patches.to(device, non_blocking=False)
        labels = labels.float().to(device, non_blocking=False)

        patches = normalize_patches(patches, mu.to(device), std.to(device))

        logits, _ = model(
            patches,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
            token_dropout=args.token_dropout,
        )

        loss = criterion(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = labels.shape[0]
        total_loss += float(loss.item()) * bs
        total_n += bs

    return total_loss / max(total_n, 1)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--feature_dir", required=True)
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--feat_dim", type=int, default=4096)
    ap.add_argument("--hidden_dim", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)

    ap.add_argument("--pool", type=str, default="topk", choices=["topk", "lse", "mean"])
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--lse_tau", type=float, default=1.0)
    ap.add_argument("--token_dropout", type=float, default=0.15)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)

    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("Loading features...")
    train_payload = load_split(args.feature_dir, "train")
    val_payload = load_split(args.feature_dir, "val")
    test_payload = load_split(args.feature_dir, "test")

    print("train patch:", tuple(train_payload["patch_features"].shape))
    print("val patch:", tuple(val_payload["patch_features"].shape))
    print("test patch:", tuple(test_payload["patch_features"].shape))

    mu, std = compute_norm_from_train(train_payload)

    train_ds = PatchFeatureDataset(train_payload)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    model = TopKMILScorer(
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.BCEWithLogitsLoss()

    best = {
        "epoch": -1,
        "val_auc": -1,
        "state_dict": None,
        "summary": None,
    }

    history = []

    for epoch in range(1, args.epochs + 1):
        loss = train_one_epoch(
            model,
            train_loader,
            mu,
            std,
            optimizer,
            criterion,
            args,
            device,
        )

        val_out = evaluate_payload(model, val_payload, mu, std, args, device)
        train_cls = None

        val_auc = val_out["classification"].get("auc")
        if val_auc is None:
            val_auc = val_out["classification"]["acc"]

        record = {
            "epoch": epoch,
            "loss": float(loss),
            "val_acc": val_out["classification"]["acc"],
            "val_auc": val_out["classification"]["auc"],
            "val_ap": val_out["classification"]["ap"],
            "val_fake_top1_edge1": val_out["edge_metrics"]["fake"]["top1_edge1_rate"],
            "val_fake_top1_edge2": val_out["edge_metrics"]["fake"]["top1_edge2_rate"],
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={loss:.6f} "
            f"val_acc={record['val_acc']:.4f} "
            f"val_auc={record['val_auc']:.4f} "
            f"val_fake_e1={record['val_fake_top1_edge1']:.4f} "
            f"val_fake_e2={record['val_fake_top1_edge2']:.4f}"
        )

        if val_auc > best["val_auc"]:
            best["epoch"] = epoch
            best["val_auc"] = val_auc
            best["state_dict"] = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            ckpt = {
                "model_state_dict": best["state_dict"],
                "mu": mu.cpu(),
                "std": std.cpu(),
                "args": vars(args),
                "best_epoch": epoch,
                "best_val_auc": float(val_auc),
            }
            torch.save(ckpt, os.path.join(args.output_dir, "mid20_topk_mil_best.pt"))

    print("Loading best model...")
    model.load_state_dict(best["state_dict"])

    train_out = evaluate_payload(model, train_payload, mu, std, args, device)
    val_out = evaluate_payload(model, val_payload, mu, std, args, device)
    test_out = evaluate_payload(model, test_payload, mu, std, args, device)

    summary = {
        "config": vars(args),
        "best_epoch": best["epoch"],
        "metrics": {
            "train": train_out["classification"],
            "val": val_out["classification"],
            "test": test_out["classification"],
        },
        "edge_metrics": {
            "train": train_out["edge_metrics"],
            "val": val_out["edge_metrics"],
            "test": test_out["edge_metrics"],
        },
        "history": history,
    }

    with open(os.path.join(args.output_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", os.path.join(args.output_dir, "summary.json"))
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
