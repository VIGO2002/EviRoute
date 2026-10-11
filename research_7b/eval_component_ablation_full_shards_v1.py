#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
)


GRID = 14
NUM_PATCHES = GRID * GRID


class TopKMILScorer(nn.Module):
    def __init__(
        self,
        feat_dim=4096,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def score_patches(self, x):
        return self.scorer(x).squeeze(-1)


def safe_torch_load(path):
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location="cpu",
        )


def get_first_key(payload, keys):
    for key in keys:
        if key in payload:
            return payload[key]

    raise KeyError(
        f"None of {keys} found; "
        f"existing keys={list(payload.keys())}"
    )


def parse_model_specs(specs, device):
    models = {}

    for spec in specs:
        if "=" not in spec:
            raise ValueError(
                f"Invalid --model specification: {spec}; "
                f"expected NAME=/path/to/checkpoint.pt"
            )

        name, checkpoint_path = spec.split("=", 1)
        checkpoint_path = Path(checkpoint_path)

        if name in models:
            raise ValueError(
                f"Duplicated model name: {name}"
            )

        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)

        checkpoint = safe_torch_load(checkpoint_path)

        checkpoint_args = checkpoint.get("args", {})
        if hasattr(checkpoint_args, "__dict__"):
            checkpoint_args = vars(checkpoint_args)

        state = checkpoint.get(
            "model_state_dict",
            checkpoint.get("state_dict"),
        )

        if state is None:
            raise KeyError(
                f"No model state in {checkpoint_path}"
            )

        model = TopKMILScorer(
            feat_dim=int(
                checkpoint_args.get("feat_dim", 4096)
            ),
            hidden_dim=int(
                checkpoint_args.get("hidden_dim", 256)
            ),
            dropout=float(
                checkpoint_args.get("dropout", 0.1)
            ),
        ).to(device)

        model.load_state_dict(state, strict=True)
        model.eval()

        mu = checkpoint["mu"].float().to(device)
        std = checkpoint["std"].float().to(device)

        models[name] = {
            "model": model,
            "mu": mu,
            "std": std,
            "checkpoint": str(checkpoint_path),
            "best_epoch": checkpoint.get("best_epoch"),
            "best_score": checkpoint.get(
                "best_score",
                checkpoint.get("best_val_auc"),
            ),
        }

        print(
            f"[Loaded] {name}: "
            f"{checkpoint_path}"
        )

    return models


def discover_feature_files(
    feature_root,
    dataset,
):
    feature_root = Path(feature_root)

    if dataset == "genimage":
        files = sorted(
            feature_root.glob("*/*/val.pt")
        )
    elif dataset == "aiginow":
        files = sorted(
            feature_root.glob("*/*/*/test.pt")
        )
    else:
        raise ValueError(dataset)

    return files


def parse_group_from_path(
    feature_root,
    feature_path,
    dataset,
):
    relative = feature_path.relative_to(feature_root)

    if dataset == "genimage":
        source = relative.parts[0]
        mode = "genimage"
        shard = relative.parts[1]
    else:
        source = relative.parts[0]
        mode = relative.parts[1]
        shard = relative.parts[2]

    return source, mode, shard


def sigmoid_np(values):
    values = np.clip(
        np.asarray(values, dtype=np.float64),
        -60.0,
        60.0,
    )
    return 1.0 / (1.0 + np.exp(-values))


def best_accuracy(labels, scores):
    """
    精确计算所有可能阈值下的最高准确率。
    预测规则为 score >= threshold 判为 fake。
    """
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)

    n = len(labels)
    if n == 0:
        return np.nan, np.nan

    order = np.argsort(
        -scores,
        kind="mergesort",
    )

    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # threshold > max(score)：全部预测为 real
    initial_correct = int(
        (labels == 0).sum()
    )

    changes = np.where(
        sorted_labels == 1,
        1,
        -1,
    )

    cumulative_correct = (
        initial_correct
        + np.cumsum(changes)
    )

    group_ends = np.r_[
        sorted_scores[:-1]
        != sorted_scores[1:],
        True,
    ]

    candidate_correct = (
        cumulative_correct[group_ends]
    )
    candidate_thresholds = (
        sorted_scores[group_ends]
    )

    all_correct = np.r_[
        initial_correct,
        candidate_correct,
    ]

    best_index = int(np.argmax(all_correct))
    best_correct = int(all_correct[best_index])

    if best_index == 0:
        threshold = float("inf")
    else:
        threshold = float(
            candidate_thresholds[best_index - 1]
        )

    return (
        float(best_correct / n),
        threshold,
    )


def normalized_entropy(indices):
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if len(indices) == 0:
        return np.nan

    counts = np.bincount(
        indices,
        minlength=NUM_PATCHES,
    ).astype(np.float64)

    probabilities = counts / counts.sum()
    probabilities = probabilities[
        probabilities > 0
    ]

    entropy = -np.sum(
        probabilities
        * np.log(probabilities)
    )

    return float(
        entropy / np.log(NUM_PATCHES)
    )


def summarize_subset(
    labels,
    logits,
    top1,
    top10_edge1_fraction,
    top10_edge2_fraction,
    edge1_mask,
    edge2_mask,
):
    labels = np.asarray(labels, dtype=np.int64)
    logits = np.asarray(logits, dtype=np.float64)
    top1 = np.asarray(top1, dtype=np.int64)

    probabilities = sigmoid_np(logits)
    predictions = (logits >= 0).astype(np.int64)

    n = len(labels)
    real_mask = labels == 0
    fake_mask = labels == 1

    if len(np.unique(labels)) == 2:
        auc = float(
            roc_auc_score(labels, logits)
        )
        ap = float(
            average_precision_score(
                labels,
                logits,
            )
        )
    else:
        auc = np.nan
        ap = np.nan

    best_acc, best_threshold_logit = (
        best_accuracy(labels, logits)
    )

    def safe_mean(values):
        values = np.asarray(values)

        if len(values) == 0:
            return np.nan

        return float(np.mean(values))

    return {
        "n": int(n),
        "n_real": int(real_mask.sum()),
        "n_fake": int(fake_mask.sum()),

        "auc": auc,
        "ap": ap,
        "acc_thr05": float(
            np.mean(predictions == labels)
        ),
        "best_acc": best_acc,
        "best_threshold_logit":
            best_threshold_logit,

        "mean_prob_real": safe_mean(
            probabilities[real_mask]
        ),
        "mean_prob_fake": safe_mean(
            probabilities[fake_mask]
        ),

        "fake_top1_edge1": safe_mean(
            edge1_mask[top1[fake_mask]]
        ),
        "fake_top1_edge2": safe_mean(
            edge2_mask[top1[fake_mask]]
        ),
        "real_top1_edge1": safe_mean(
            edge1_mask[top1[real_mask]]
        ),
        "real_top1_edge2": safe_mean(
            edge2_mask[top1[real_mask]]
        ),

        "fake_top10_edge1_fraction":
            safe_mean(
                top10_edge1_fraction[
                    fake_mask
                ]
            ),
        "fake_top10_edge2_fraction":
            safe_mean(
                top10_edge2_fraction[
                    fake_mask
                ]
            ),
        "real_top10_edge1_fraction":
            safe_mean(
                top10_edge1_fraction[
                    real_mask
                ]
            ),
        "real_top10_edge2_fraction":
            safe_mean(
                top10_edge2_fraction[
                    real_mask
                ]
            ),

        "fake_top1_position_entropy":
            normalized_entropy(
                top1[fake_mask]
            ),
        "real_top1_position_entropy":
            normalized_entropy(
                top1[real_mask]
            ),
    }


def nanmean_rows(rows, key):
    values = [
        row[key]
        for row in rows
        if key in row
        and row[key] is not None
        and np.isfinite(row[key])
    ]

    if not values:
        return np.nan

    return float(np.mean(values))


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["genimage", "aiginow"],
    )
    parser.add_argument(
        "--feature_root",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Repeatable NAME=/path/checkpoint.pt"
        ),
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--max_files",
        type=int,
        default=0,
        help="0 means all files",
    )

    args = parser.parse_args()

    feature_root = Path(args.feature_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )

    feature_files = discover_feature_files(
        feature_root,
        args.dataset,
    )

    if args.max_files > 0:
        feature_files = feature_files[
            :args.max_files
        ]

    if not feature_files:
        raise FileNotFoundError(
            f"No feature files under "
            f"{feature_root}"
        )

    print(
        f"Dataset: {args.dataset}\n"
        f"Feature files: {len(feature_files)}"
    )

    models = parse_model_specs(
        args.model,
        device,
    )

    edge1 = np.zeros(
        NUM_PATCHES,
        dtype=bool,
    )
    edge2 = np.zeros(
        NUM_PATCHES,
        dtype=bool,
    )

    for pid in range(NUM_PATCHES):
        row = pid // GRID
        col = pid % GRID

        edge1[pid] = (
            row == 0
            or row == GRID - 1
            or col == 0
            or col == GRID - 1
        )

        edge2[pid] = (
            row <= 1
            or row >= GRID - 2
            or col <= 1
            or col >= GRID - 2
        )

    edge1_t = torch.from_numpy(
        edge1
    ).to(device)
    edge2_t = torch.from_numpy(
        edge2
    ).to(device)

    shared_labels = []
    shared_sources = []
    shared_modes = []
    shared_shards = []
    shared_local_indices = []

    outputs = {
        name: {
            "logits": [],
            "top1": [],
            "top10_edge1_fraction": [],
            "top10_edge2_fraction": [],
        }
        for name in models
    }

    total_seen = 0

    for file_id, feature_path in enumerate(
        feature_files,
        start=1,
    ):
        source, mode, shard = (
            parse_group_from_path(
                feature_root,
                feature_path,
                args.dataset,
            )
        )

        payload = safe_torch_load(
            feature_path
        )

        patch_features = get_first_key(
            payload,
            [
                "patch_features",
                "patch_feats",
                "patch_tokens",
            ],
        )

        labels_t = get_first_key(
            payload,
            ["labels", "label", "y"],
        ).long()

        labels_np = (
            labels_t.cpu().numpy().astype(
                np.int64
            )
        )

        n = len(labels_np)

        shared_labels.append(labels_np)
        shared_sources.extend(
            [source] * n
        )
        shared_modes.extend(
            [mode] * n
        )
        shared_shards.extend(
            [shard] * n
        )
        shared_local_indices.extend(
            range(n)
        )

        for start in range(
            0,
            n,
            args.batch_size,
        ):
            end = min(
                start + args.batch_size,
                n,
            )

            x = (
                patch_features[start:end]
                .float()
                .to(device)
            )

            labels_batch = torch.from_numpy(
                labels_np[start:end]
            ).to(device)

            fake_batch = (
                labels_batch > 0
            ).view(-1, 1)

            for name, bundle in models.items():
                normalized = (
                    x
                    - bundle["mu"].view(
                        1, 1, -1
                    )
                ) / bundle["std"].view(
                    1, 1, -1
                )

                scores = bundle[
                    "model"
                ].score_patches(normalized)

                positive_topk = torch.topk(
                    scores,
                    k=args.topk,
                    dim=1,
                )

                negative_topk = torch.topk(
                    -scores,
                    k=args.topk,
                    dim=1,
                )

                # 图像分类始终使用最高 fake-support
                logits = (
                    positive_topk.values.mean(
                        dim=1
                    )
                )

                # 边缘分析使用类别对齐证据：
                # fake 取最高 score，real 取最低 score
                aligned_topk = torch.where(
                    fake_batch,
                    positive_topk.indices,
                    negative_topk.indices,
                )

                top1 = aligned_topk[:, 0]

                top10_edge1_fraction = (
                    edge1_t[aligned_topk]
                    .float()
                    .mean(dim=1)
                )

                top10_edge2_fraction = (
                    edge2_t[aligned_topk]
                    .float()
                    .mean(dim=1)
                )

                outputs[name]["logits"].append(
                    logits.cpu().numpy()
                )
                outputs[name]["top1"].append(
                    top1.cpu().numpy()
                )
                outputs[name][
                    "top10_edge1_fraction"
                ].append(
                    top10_edge1_fraction
                    .cpu()
                    .numpy()
                )
                outputs[name][
                    "top10_edge2_fraction"
                ].append(
                    top10_edge2_fraction
                    .cpu()
                    .numpy()
                )

                del normalized, scores

            del x

        total_seen += n

        print(
            f"[{file_id:02d}/"
            f"{len(feature_files):02d}] "
            f"{source}/{mode}/{shard}: "
            f"n={n}, total={total_seen}"
        )

        del payload, patch_features, labels_t

    labels = np.concatenate(
        shared_labels
    )
    sources = np.asarray(
        shared_sources,
        dtype=object,
    )
    modes = np.asarray(
        shared_modes,
        dtype=object,
    )
    shards = np.asarray(
        shared_shards,
        dtype=object,
    )
    local_indices = np.asarray(
        shared_local_indices,
        dtype=np.int64,
    )

    summary_rows = []
    group_rows = []

    per_sample = pd.DataFrame({
        "dataset": args.dataset,
        "source": sources,
        "mode": modes,
        "shard": shards,
        "local_idx": local_indices,
        "label": labels,
    })

    for name, bundle in models.items():
        logits = np.concatenate(
            outputs[name]["logits"]
        )
        top1 = np.concatenate(
            outputs[name]["top1"]
        )
        top10_e1 = np.concatenate(
            outputs[name][
                "top10_edge1_fraction"
            ]
        )
        top10_e2 = np.concatenate(
            outputs[name][
                "top10_edge2_fraction"
            ]
        )

        if not (
            len(logits)
            == len(labels)
            == len(top1)
        ):
            raise RuntimeError(
                f"Length mismatch for {name}"
            )

        global_metrics = summarize_subset(
            labels,
            logits,
            top1,
            top10_e1,
            top10_e2,
            edge1,
            edge2,
        )

        source_metrics = []

        for source in sorted(set(sources)):
            mask = sources == source

            metrics = summarize_subset(
                labels[mask],
                logits[mask],
                top1[mask],
                top10_e1[mask],
                top10_e2[mask],
                edge1,
                edge2,
            )

            metrics.update({
                "model": name,
                "scope_type": "source",
                "source": source,
                "mode": "ALL",
            })

            source_metrics.append(metrics)
            group_rows.append(metrics)

        mode_metrics = []

        for mode in sorted(set(modes)):
            mask = modes == mode

            metrics = summarize_subset(
                labels[mask],
                logits[mask],
                top1[mask],
                top10_e1[mask],
                top10_e2[mask],
                edge1,
                edge2,
            )

            metrics.update({
                "model": name,
                "scope_type": "mode",
                "source": "ALL",
                "mode": mode,
            })

            mode_metrics.append(metrics)
            group_rows.append(metrics)

        stratum_metrics = []

        strata = sorted(
            set(zip(sources, modes))
        )

        for source, mode in strata:
            mask = (
                (sources == source)
                & (modes == mode)
            )

            metrics = summarize_subset(
                labels[mask],
                logits[mask],
                top1[mask],
                top10_e1[mask],
                top10_e2[mask],
                edge1,
                edge2,
            )

            metrics.update({
                "model": name,
                "scope_type":
                    "source_mode",
                "source": source,
                "mode": mode,
            })

            stratum_metrics.append(metrics)
            group_rows.append(metrics)

        summary_row = {
            "dataset": args.dataset,
            "model": name,
            "checkpoint":
                bundle["checkpoint"],
            "best_epoch":
                bundle["best_epoch"],
            "training_best_score":
                bundle["best_score"],
            **global_metrics,

            "source_macro_auc":
                nanmean_rows(
                    source_metrics,
                    "auc",
                ),
            "source_macro_ap":
                nanmean_rows(
                    source_metrics,
                    "ap",
                ),
            "source_macro_acc_thr05":
                nanmean_rows(
                    source_metrics,
                    "acc_thr05",
                ),
            "source_macro_best_acc":
                nanmean_rows(
                    source_metrics,
                    "best_acc",
                ),
            "source_macro_fake_top1_edge1":
                nanmean_rows(
                    source_metrics,
                    "fake_top1_edge1",
                ),
            "source_macro_fake_top1_edge2":
                nanmean_rows(
                    source_metrics,
                    "fake_top1_edge2",
                ),

            "stratum_macro_auc":
                nanmean_rows(
                    stratum_metrics,
                    "auc",
                ),
            "stratum_macro_ap":
                nanmean_rows(
                    stratum_metrics,
                    "ap",
                ),
            "stratum_macro_fake_top1_edge1":
                nanmean_rows(
                    stratum_metrics,
                    "fake_top1_edge1",
                ),
            "stratum_macro_fake_top1_edge2":
                nanmean_rows(
                    stratum_metrics,
                    "fake_top1_edge2",
                ),
        }

        summary_rows.append(summary_row)

        per_sample[f"{name}_logit"] = logits
        per_sample[f"{name}_prob"] = (
            sigmoid_np(logits)
        )
        per_sample[
            f"{name}_class_aligned_top1"
        ] = top1
        per_sample[
            f"{name}_top1_edge1"
        ] = edge1[top1]
        per_sample[
            f"{name}_top1_edge2"
        ] = edge2[top1]
        per_sample[
            f"{name}_top10_edge1_fraction"
        ] = top10_e1
        per_sample[
            f"{name}_top10_edge2_fraction"
        ] = top10_e2

        print(
            f"\n[{args.dataset}] {name}: "
            f"AUC={global_metrics['auc']:.6f}, "
            f"AP={global_metrics['ap']:.6f}, "
            f"Acc={global_metrics['acc_thr05']:.6f}, "
            f"BestAcc={global_metrics['best_acc']:.6f}, "
            f"FakeE1={global_metrics['fake_top1_edge1']:.6f}, "
            f"FakeE2={global_metrics['fake_top1_edge2']:.6f}"
        )

    summary_df = pd.DataFrame(
        summary_rows
    )

    group_df = pd.DataFrame(
        group_rows
    )

    summary_csv = (
        output_dir
        / f"{args.dataset}_ablation_summary.csv"
    )
    group_csv = (
        output_dir
        / f"{args.dataset}_ablation_by_group.csv"
    )
    per_sample_csv = (
        output_dir
        / f"{args.dataset}_ablation_per_sample.csv.gz"
    )
    summary_json = (
        output_dir
        / f"{args.dataset}_ablation_summary.json"
    )

    summary_df.to_csv(
        summary_csv,
        index=False,
    )

    group_df.to_csv(
        group_csv,
        index=False,
    )

    per_sample.to_csv(
        per_sample_csv,
        index=False,
        compression="gzip",
    )

    with open(
        summary_json,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "dataset": args.dataset,
                "feature_root":
                    str(feature_root),
                "num_feature_files":
                    len(feature_files),
                "num_samples":
                    int(len(labels)),
                "topk": args.topk,
                "summary":
                    summary_df.to_dict(
                        orient="records"
                    ),
                "by_group":
                    group_df.to_dict(
                        orient="records"
                    ),
            },
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=True,
        )

    print("\n[Saved]", summary_csv)
    print("[Saved]", group_csv)
    print("[Saved]", per_sample_csv)
    print("[Saved]", summary_json)

    display_columns = [
        "dataset",
        "model",
        "n",
        "auc",
        "ap",
        "acc_thr05",
        "best_acc",
        "fake_top1_edge1",
        "fake_top1_edge2",
        "source_macro_auc",
        "source_macro_ap",
    ]

    print("\n========== Main ablation table ==========")
    print(
        summary_df[
            display_columns
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
