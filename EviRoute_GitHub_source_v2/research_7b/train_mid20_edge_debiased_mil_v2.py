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
    def __init__(self, payload):
        self.patch_features = payload["patch_features"]
        self.labels = payload["labels"]

    def __len__(self):
        return int(self.labels.shape[0])

    def __getitem__(self, idx):
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
        return self.scorer(x).squeeze(-1)  # [B, 196]


def make_edge_masks(grid=14, device="cpu"):
    yy, xx = torch.meshgrid(
        torch.arange(grid, device=device),
        torch.arange(grid, device=device),
        indexing="ij",
    )

    edge1 = (yy == 0) | (yy == grid - 1) | (xx == 0) | (xx == grid - 1)
    edge2 = (yy <= 1) | (yy >= grid - 2) | (xx <= 1) | (xx >= grid - 2)
    edge2_only = edge2 & (~edge1)

    return edge1.reshape(-1), edge2.reshape(-1), edge2_only.reshape(-1)


def apply_dropout_masks(
    scores,
    token_dropout=0.0,
    edge1_dropout=0.0,
    edge2_only_dropout=0.0,
    grid=14,
):
    """
    scores: [B, P]
    edge1 和 edge2_only 分开 mask，避免最外圈被重复 mask。
    """
    B, P = scores.shape
    device = scores.device

    final_mask = torch.zeros(B, P, dtype=torch.bool, device=device)

    if token_dropout > 0:
        final_mask |= (torch.rand(B, P, device=device) < token_dropout)

    edge1, _, edge2_only = make_edge_masks(grid=grid, device=device)

    if edge1_dropout > 0:
        m = torch.rand(B, P, device=device) < edge1_dropout
        final_mask |= (m & edge1.view(1, -1))

    if edge2_only_dropout > 0:
        m = torch.rand(B, P, device=device) < edge2_only_dropout
        final_mask |= (m & edge2_only.view(1, -1))

    # 避免某张图所有 token 都被 mask
    all_masked = final_mask.all(dim=1)
    if all_masked.any():
        final_mask[all_masked, :] = False

    return scores.masked_fill(final_mask, -1e4)


def pool_scores(scores, pool="topk", topk=10, lse_tau=1.0):
    if pool == "topk":
        k = min(topk, scores.shape[1])
        vals = torch.topk(scores, k=k, dim=1).values
        return vals.mean(dim=1)

    if pool == "lse":
        return lse_tau * torch.logsumexp(scores / lse_tau, dim=1) - lse_tau * math.log(scores.shape[1])

    if pool == "mean":
        return scores.mean(dim=1)

    raise ValueError(f"Unknown pool: {pool}")


def compute_norm_from_train(train_payload):
    x = train_payload["mean_features"].float()
    mu = x.mean(dim=0)
    std = x.std(dim=0).clamp_min(1e-6)
    return mu, std


def normalize_patches(x, mu, std):
    return (x.float() - mu.view(1, 1, -1)) / std.view(1, 1, -1)


def edge_mass_loss(
    raw_scores,
    labels,
    grid=14,
    tau=1.0,
    lambda_edge1=0.0,
    rho_edge1=0.35,
    lambda_edge2=0.0,
    rho_edge2=0.60,
):
    """
    只对 fake samples 做 edge mass penalty。
    raw_scores 不经过 dropout，用来约束测试时真正输出的 heatmap。
    """
    if lambda_edge1 <= 0 and lambda_edge2 <= 0:
        return raw_scores.new_tensor(0.0)

    fake_mask = labels > 0.5
    if fake_mask.sum() == 0:
        return raw_scores.new_tensor(0.0)

    scores_f = raw_scores[fake_mask]
    p = torch.softmax(scores_f / tau, dim=1)

    edge1, edge2, _ = make_edge_masks(grid=grid, device=raw_scores.device)

    loss = raw_scores.new_tensor(0.0)

    if lambda_edge1 > 0:
        mass1 = p[:, edge1].sum(dim=1)
        loss1 = torch.relu(mass1 - rho_edge1).pow(2).mean()
        loss = loss + lambda_edge1 * loss1

    if lambda_edge2 > 0:
        mass2 = p[:, edge2].sum(dim=1)
        loss2 = torch.relu(mass2 - rho_edge2).pow(2).mean()
        loss = loss + lambda_edge2 * loss2

    return loss


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


def edge_masks_np(grid=14):
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


def edge_metrics_for_group(score_maps, labels, group="fake", grid=14):
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

    e1, e2 = edge_masks_np(grid)
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
        patches = patches.to(device)
        labels = labels.to(device)

        patches = normalize_patches(patches, mu.to(device), std.to(device))
        raw_scores = model.score_patches(patches)
        logits = pool_scores(
            raw_scores,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
        )

        all_logits.append(logits.cpu())
        all_scores.append(raw_scores.cpu())
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


def train_one_epoch(model, loader, mu, std, optimizer, bce_loss, args, device):
    model.train()

    total_loss = 0.0
    total_cls = 0.0
    total_edge = 0.0
    total_n = 0

    for patches, labels in loader:
        patches = patches.to(device)
        labels_float = labels.float().to(device)

        patches = normalize_patches(patches, mu.to(device), std.to(device))

        raw_scores = model.score_patches(patches)

        masked_scores = apply_dropout_masks(
            raw_scores,
            token_dropout=args.token_dropout,
            edge1_dropout=args.edge1_dropout,
            edge2_only_dropout=args.edge2_only_dropout,
            grid=args.grid,
        )

        logits = pool_scores(
            masked_scores,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
        )

        loss_cls = bce_loss(logits, labels_float)

        loss_edge = edge_mass_loss(
            raw_scores=raw_scores,
            labels=labels_float,
            grid=args.grid,
            tau=args.edge_tau,
            lambda_edge1=args.lambda_edge1,
            rho_edge1=args.rho_edge1,
            lambda_edge2=args.lambda_edge2,
            rho_edge2=args.rho_edge2,
        )

        loss = loss_cls + loss_edge

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        bs = labels.shape[0]
        total_loss += float(loss.item()) * bs
        total_cls += float(loss_cls.item()) * bs
        total_edge += float(loss_edge.item()) * bs
        total_n += bs

    return {
        "loss": total_loss / max(total_n, 1),
        "loss_cls": total_cls / max(total_n, 1),
        "loss_edge": total_edge / max(total_n, 1),
    }


def checkpoint_score(val_out, args):
    cls = val_out["classification"]
    fake_edge = val_out["edge_metrics"]["fake"]

    val_auc = cls.get("auc")
    if val_auc is None:
        val_auc = cls["acc"]

    e1 = fake_edge["top1_edge1_rate"]
    e2 = fake_edge["top1_edge2_rate"]

    # 如果 AUC 已经足够高，优先选边缘更低的模型。
    if val_auc >= args.min_auc_for_edge_select:
        score = 10.0 + val_auc - args.select_edge1_weight * e1 - args.select_edge2_weight * e2
    else:
        score = val_auc - args.low_auc_penalty * (args.min_auc_for_edge_select - val_auc)

    return float(score)


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
    ap.add_argument("--edge1_dropout", type=float, default=0.5)
    ap.add_argument("--edge2_only_dropout", type=float, default=0.25)

    ap.add_argument("--lambda_edge1", type=float, default=0.0)
    ap.add_argument("--rho_edge1", type=float, default=0.35)
    ap.add_argument("--lambda_edge2", type=float, default=0.0)
    ap.add_argument("--rho_edge2", type=float, default=0.60)
    ap.add_argument("--edge_tau", type=float, default=1.0)

    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--eval_batch_size", type=int, default=32)

    ap.add_argument("--grid", type=int, default=14)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--min_auc_for_edge_select", type=float, default=0.995)
    ap.add_argument("--select_edge1_weight", type=float, default=1.0)
    ap.add_argument("--select_edge2_weight", type=float, default=0.5)
    ap.add_argument("--low_auc_penalty", type=float, default=5.0)

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

    train_loader = DataLoader(
        PatchFeatureDataset(train_payload),
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

    bce_loss = nn.BCEWithLogitsLoss()

    best_score = -1e9
    best_epoch = -1
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        loss_info = train_one_epoch(
            model,
            train_loader,
            mu,
            std,
            optimizer,
            bce_loss,
            args,
            device,
        )

        val_out = evaluate_payload(model, val_payload, mu, std, args, device)
        score = checkpoint_score(val_out, args)

        val_cls = val_out["classification"]
        val_fake = val_out["edge_metrics"]["fake"]

        record = {
            "epoch": epoch,
            **loss_info,
            "val_acc": val_cls["acc"],
            "val_auc": val_cls["auc"],
            "val_ap": val_cls["ap"],
            "val_fake_top1_edge1": val_fake["top1_edge1_rate"],
            "val_fake_top1_edge2": val_fake["top1_edge2_rate"],
            "select_score": score,
        }
        history.append(record)

        print(
            f"[Epoch {epoch:03d}] "
            f"loss={loss_info['loss']:.6f} "
            f"cls={loss_info['loss_cls']:.6f} "
            f"edge={loss_info['loss_edge']:.6f} "
            f"val_acc={val_cls['acc']:.4f} "
            f"val_auc={val_cls['auc']:.4f} "
            f"fake_e1={val_fake['top1_edge1_rate']:.4f} "
            f"fake_e2={val_fake['top1_edge2_rate']:.4f} "
            f"score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

            ckpt = {
                "model_state_dict": best_state,
                "mu": mu.cpu(),
                "std": std.cpu(),
                "args": vars(args),
                "best_epoch": best_epoch,
                "best_score": best_score,
            }
            torch.save(ckpt, os.path.join(args.output_dir, "mid20_edge_debiased_mil_best.pt"))
            # 为了兼容旧 eval 脚本，也额外保存一个旧文件名
            torch.save(ckpt, os.path.join(args.output_dir, "mid20_topk_mil_best.pt"))

    print("Loading best model...")
    model.load_state_dict(best_state)

    train_out = evaluate_payload(model, train_payload, mu, std, args, device)
    val_out = evaluate_payload(model, val_payload, mu, std, args, device)
    test_out = evaluate_payload(model, test_payload, mu, std, args, device)

    summary = {
        "config": vars(args),
        "best_epoch": best_epoch,
        "best_score": best_score,
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

    out_path = os.path.join(args.output_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("Saved:", out_path)
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
