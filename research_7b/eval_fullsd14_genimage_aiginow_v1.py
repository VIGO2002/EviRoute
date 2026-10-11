#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unified full evaluation for the full-SD1.4-trained EviRoute checkpoint.

Datasets
--------
GenImage:
    features/genimage_full_mid20_shards/
        {source}/{shard}/val.pt

AIGI-Now:
    features/aiginow_full_mid20_shards/
        {source}/{pixel|semantic}/{shard}/test.pt

Outputs
-------
For each benchmark:
- micro overall AUC / AP / ACC
- macro-source AUC / AP / ACC
- per-source metrics
- AIGI-Now additionally:
    per-mode
    per-source-mode
    macro-source-mode
- fake evidence:
    Top-1 B1 / B2
    Top-10 B1 / B2
- compact per-sample CSV.gz

Important
---------
Prediction path exactly follows the frozen checkpoint:
    checkpoint-specific mu/std
    patch scorer
    clean raw scores
    Top-K mean pooling

No training-time dropout is used during evaluation.
"""

import argparse
import gc
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)


# ============================================================
# Model
# ============================================================

class TopKMILScorer(nn.Module):

    def __init__(
        self,
        feat_dim=4096,
        hidden_dim=256,
        dropout=0.1,
    ):
        super().__init__()

        self.scorer = nn.Sequential(
            nn.Linear(
                feat_dim,
                hidden_dim,
            ),
            nn.GELU(),
            nn.Dropout(
                dropout,
            ),
            nn.Linear(
                hidden_dim,
                1,
            ),
        )

    def score_patches(
        self,
        x,
    ):
        return (
            self.scorer(x)
            .squeeze(-1)
        )


def pool_scores(
    scores,
    pool="topk",
    topk=10,
    lse_tau=1.0,
):

    if pool == "topk":

        k = min(
            topk,
            scores.shape[1],
        )

        vals = torch.topk(
            scores,
            k=k,
            dim=1,
        ).values

        return vals.mean(
            dim=1
        )

    if pool == "lse":

        return (
            lse_tau
            * torch.logsumexp(
                scores / lse_tau,
                dim=1,
            )
            - lse_tau
            * math.log(
                scores.shape[1]
            )
        )

    if pool == "mean":
        return scores.mean(
            dim=1
        )

    raise ValueError(
        f"Unknown pool: {pool}"
    )


# ============================================================
# Feature loading
# ============================================================

def get_first_key(
    d,
    keys,
):
    for k in keys:
        if k in d:
            return d[k]

    raise KeyError(
        f"None of keys {keys} found. "
        f"Existing keys: {list(d.keys())}"
    )


def load_payload(
    path,
):

    d = torch.load(
        path,
        map_location="cpu",
    )

    patch_features = get_first_key(
        d,
        [
            "patch_features",
            "patch_feats",
            "patch_tokens",
        ],
    )

    labels = get_first_key(
        d,
        [
            "labels",
            "label",
            "y",
        ],
    ).long()

    paths = d.get(
        "paths",
        d.get(
            "path",
            [""] * len(labels),
        ),
    )

    case_names = d.get(
        "case_names",
        d.get(
            "case_name",
            [""] * len(labels),
        ),
    )

    return {
        "patch_features":
            patch_features,

        "labels":
            labels,

        "paths":
            paths,

        "case_names":
            case_names,
    }


# ============================================================
# Spatial masks
# ============================================================

def build_edge_masks(
    grid,
    device,
):

    yy, xx = torch.meshgrid(
        torch.arange(
            grid,
            device=device,
        ),
        torch.arange(
            grid,
            device=device,
        ),
        indexing="ij",
    )

    b1 = (
        (yy == 0)
        | (yy == grid - 1)
        | (xx == 0)
        | (xx == grid - 1)
    )

    b2 = (
        (yy <= 1)
        | (yy >= grid - 2)
        | (xx <= 1)
        | (xx >= grid - 2)
    )

    return (
        b1.reshape(-1),
        b2.reshape(-1),
    )


# ============================================================
# One shard inference
# ============================================================

@torch.no_grad()
def evaluate_shard(
    model,
    payload,
    mu,
    std,
    device,
    pool,
    topk,
    lse_tau,
    batch_size,
    grid,
):

    model.eval()

    patches_all = payload[
        "patch_features"
    ]

    labels_all = payload[
        "labels"
    ]


    if (
        patches_all.ndim != 3
        or patches_all.shape[1] != 196
        or patches_all.shape[2] != mu.numel()
    ):
        raise RuntimeError(
            "Unexpected patch feature shape: "
            f"{tuple(patches_all.shape)}"
        )


    mu = mu.to(
        device
    )

    std = std.to(
        device
    )

    b1_mask, b2_mask = (
        build_edge_masks(
            grid,
            device,
        )
    )


    records = []

    n = int(
        labels_all.shape[0]
    )


    for start in range(
        0,
        n,
        batch_size,
    ):

        end = min(
            start + batch_size,
            n,
        )

        patches = (
            patches_all[
                start:end
            ]
            .to(
                device,
                non_blocking=True,
            )
        )

        labels = (
            labels_all[
                start:end
            ]
            .to(
                device,
                non_blocking=True,
            )
        )


        # Exact checkpoint-specific normalization.
        patches = (
            patches.float()
            - mu.view(
                1,
                1,
                -1,
            )
        ) / std.view(
            1,
            1,
            -1,
        )


        raw_scores = (
            model.score_patches(
                patches
            )
        )


        # ----------------------------------------------------
        # Image-level fake logit
        # ----------------------------------------------------

        logits = pool_scores(
            raw_scores,
            pool=pool,
            topk=topk,
            lse_tau=lse_tau,
        )


        probs = torch.sigmoid(
            logits
        )

        preds = (
            logits >= 0
        ).long()


        # ----------------------------------------------------
        # Class-aligned evidence support
        #
        # Fake: +score
        # Real: -score
        # ----------------------------------------------------

        sign = torch.where(
            labels == 1,
            torch.ones_like(
                labels,
                dtype=torch.float32,
            ),
            -torch.ones_like(
                labels,
                dtype=torch.float32,
            ),
        )

        support = (
            raw_scores
            * sign.view(
                -1,
                1,
            )
        )


        k_evidence = min(
            10,
            support.shape[1],
        )

        top10_idx = torch.topk(
            support,
            k=k_evidence,
            dim=1,
        ).indices


        top1_idx = (
            top10_idx[:, 0]
        )


        top1_b1 = (
            b1_mask[
                top1_idx
            ]
        )

        top1_b2 = (
            b2_mask[
                top1_idx
            ]
        )


        top10_b1 = (
            b1_mask[
                top10_idx
            ]
            .float()
            .mean(
                dim=1
            )
        )

        top10_b2 = (
            b2_mask[
                top10_idx
            ]
            .float()
            .mean(
                dim=1
            )
        )


        # ----------------------------------------------------
        # Bring compact results back to CPU
        # ----------------------------------------------------

        logits_np = (
            logits.cpu()
            .numpy()
        )

        probs_np = (
            probs.cpu()
            .numpy()
        )

        preds_np = (
            preds.cpu()
            .numpy()
        )

        labels_np = (
            labels.cpu()
            .numpy()
        )

        top1_np = (
            top1_idx.cpu()
            .numpy()
        )

        top1_b1_np = (
            top1_b1.cpu()
            .numpy()
        )

        top1_b2_np = (
            top1_b2.cpu()
            .numpy()
        )

        top10_b1_np = (
            top10_b1.cpu()
            .numpy()
        )

        top10_b2_np = (
            top10_b2.cpu()
            .numpy()
        )


        for i in range(
            end - start
        ):

            records.append({
                "local_idx":
                    int(
                        start + i
                    ),

                "label":
                    int(
                        labels_np[i]
                    ),

                "logit":
                    float(
                        logits_np[i]
                    ),

                "prob":
                    float(
                        probs_np[i]
                    ),

                "pred":
                    int(
                        preds_np[i]
                    ),

                "class_aligned_top1":
                    int(
                        top1_np[i]
                    ),

                "top1_B1":
                    bool(
                        top1_b1_np[i]
                    ),

                "top1_B2":
                    bool(
                        top1_b2_np[i]
                    ),

                "top10_B1_fraction":
                    float(
                        top10_b1_np[i]
                    ),

                "top10_B2_fraction":
                    float(
                        top10_b2_np[i]
                    ),
            })


        del patches
        del labels
        del raw_scores
        del support
        del logits


    return pd.DataFrame(
        records
    )


# ============================================================
# Metrics
# ============================================================

def classification_metrics(
    df,
):

    labels = (
        df["label"]
        .to_numpy(
            dtype=np.int64
        )
    )

    logits = (
        df["logit"]
        .to_numpy(
            dtype=np.float64
        )
    )

    preds = (
        df["pred"]
        .to_numpy(
            dtype=np.int64
        )
    )


    out = {
        "n":
            int(
                len(df)
            ),

        "real":
            int(
                (labels == 0).sum()
            ),

        "fake":
            int(
                (labels == 1).sum()
            ),

        "acc":
            float(
                (
                    preds
                    == labels
                ).mean()
            ),
    }


    if len(
        np.unique(
            labels
        )
    ) == 2:

        # AUC/AP can use logits directly.
        # They are ranking metrics and this avoids numerical
        # saturation from very large sigmoid logits.
        out["auc"] = float(
            roc_auc_score(
                labels,
                logits,
            )
        )

        out["ap"] = float(
            average_precision_score(
                labels,
                logits,
            )
        )

    else:

        out["auc"] = None
        out["ap"] = None


    fake = df[
        df["label"] == 1
    ]

    real = df[
        df["label"] == 0
    ]


    if len(fake):

        out[
            "fake_top1_B1"
        ] = float(
            fake[
                "top1_B1"
            ].mean()
        )

        out[
            "fake_top1_B2"
        ] = float(
            fake[
                "top1_B2"
            ].mean()
        )

        out[
            "fake_top10_B1"
        ] = float(
            fake[
                "top10_B1_fraction"
            ].mean()
        )

        out[
            "fake_top10_B2"
        ] = float(
            fake[
                "top10_B2_fraction"
            ].mean()
        )

    else:

        out[
            "fake_top1_B1"
        ] = None

        out[
            "fake_top1_B2"
        ] = None

        out[
            "fake_top10_B1"
        ] = None

        out[
            "fake_top10_B2"
        ] = None


    if len(real):

        out[
            "real_top1_B1"
        ] = float(
            real[
                "top1_B1"
            ].mean()
        )

        out[
            "real_top1_B2"
        ] = float(
            real[
                "top1_B2"
            ].mean()
        )

    else:

        out[
            "real_top1_B1"
        ] = None

        out[
            "real_top1_B2"
        ] = None


    return out


def group_metrics(
    df,
    columns,
):

    out = {}

    grouped = df.groupby(
        columns,
        sort=True,
        dropna=False,
    )

    for key, part in grouped:

        if not isinstance(
            key,
            tuple,
        ):
            key = (
                key,
            )

        name = "/".join(
            str(v)
            for v in key
        )

        out[name] = (
            classification_metrics(
                part
            )
        )

    return out


def macro_metrics(
    group_dict,
):

    keys = [
        "auc",
        "ap",
        "acc",
        "fake_top1_B1",
        "fake_top1_B2",
        "fake_top10_B1",
        "fake_top10_B2",
    ]

    out = {
        "num_groups":
            int(
                len(
                    group_dict
                )
            )
    }


    for metric in keys:

        vals = [
            x.get(
                metric
            )
            for x in group_dict.values()
            if x.get(
                metric
            ) is not None
        ]

        out[metric] = (
            float(
                np.mean(
                    vals
                )
            )
            if vals
            else None
        )


    return out


# ============================================================
# Dataset evaluation
# ============================================================

def evaluate_genimage(
    root,
    model,
    mu,
    std,
    args,
):

    root = Path(
        root
    )

    files = sorted(
        root.glob(
            "*/shard*/val.pt"
        )
    )


    if not files:
        raise RuntimeError(
            f"No GenImage val.pt found under {root}"
        )


    print()
    print("=" * 100)
    print("GENIMAGE FULL")
    print("=" * 100)

    print(
        "shards:",
        len(files),
    )


    frames = []


    for idx, pt in enumerate(
        files,
        start=1,
    ):

        rel = pt.relative_to(
            root
        )

        source = rel.parts[0]
        shard = rel.parts[1]


        print(
            f"[GenImage "
            f"{idx:03d}/{len(files):03d}] "
            f"{source}/{shard}",
            flush=True,
        )


        payload = load_payload(
            pt
        )


        df = evaluate_shard(
            model=model,
            payload=payload,
            mu=mu,
            std=std,
            device=args.device_obj,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
            batch_size=args.batch_size,
            grid=args.grid,
        )


        df.insert(
            0,
            "dataset",
            "GenImage",
        )

        df.insert(
            1,
            "source",
            source,
        )

        df.insert(
            2,
            "mode",
            "",
        )

        df.insert(
            3,
            "shard",
            shard,
        )


        paths = payload.get(
            "paths",
            [],
        )

        if paths is not None and len(
            paths
        ) == len(df):

            df[
                "path"
            ] = [
                str(x)
                for x in paths
            ]


        frames.append(
            df
        )


        del payload
        del df

        gc.collect()


    all_df = pd.concat(
        frames,
        ignore_index=True,
    )


    overall = (
        classification_metrics(
            all_df
        )
    )

    by_source = group_metrics(
        all_df,
        [
            "source",
        ],
    )

    macro_source = macro_metrics(
        by_source
    )


    return (
        all_df,
        {
            "overall":
                overall,

            "macro_source":
                macro_source,

            "by_source":
                by_source,

            "num_shards":
                int(
                    len(files)
                ),
        },
    )


def evaluate_aiginow(
    root,
    model,
    mu,
    std,
    args,
):

    root = Path(
        root
    )

    files = sorted(
        root.glob(
            "*/*/shard*/test.pt"
        )
    )


    if not files:
        raise RuntimeError(
            f"No AIGI-Now test.pt found under {root}"
        )


    print()
    print("=" * 100)
    print("AIGI-NOW FULL")
    print("=" * 100)

    print(
        "shards:",
        len(files),
    )


    frames = []


    for idx, pt in enumerate(
        files,
        start=1,
    ):

        rel = pt.relative_to(
            root
        )

        source = rel.parts[0]
        mode = rel.parts[1]
        shard = rel.parts[2]


        print(
            f"[AIGI "
            f"{idx:02d}/{len(files):02d}] "
            f"{source}/{mode}/{shard}",
            flush=True,
        )


        payload = load_payload(
            pt
        )


        df = evaluate_shard(
            model=model,
            payload=payload,
            mu=mu,
            std=std,
            device=args.device_obj,
            pool=args.pool,
            topk=args.topk,
            lse_tau=args.lse_tau,
            batch_size=args.batch_size,
            grid=args.grid,
        )


        df.insert(
            0,
            "dataset",
            "AIGI-Now",
        )

        df.insert(
            1,
            "source",
            source,
        )

        df.insert(
            2,
            "mode",
            mode,
        )

        df.insert(
            3,
            "shard",
            shard,
        )


        paths = payload.get(
            "paths",
            [],
        )

        if paths is not None and len(
            paths
        ) == len(df):

            df[
                "path"
            ] = [
                str(x)
                for x in paths
            ]


        frames.append(
            df
        )


        del payload
        del df

        gc.collect()


    all_df = pd.concat(
        frames,
        ignore_index=True,
    )


    overall = (
        classification_metrics(
            all_df
        )
    )


    by_source = group_metrics(
        all_df,
        [
            "source",
        ],
    )


    by_mode = group_metrics(
        all_df,
        [
            "mode",
        ],
    )


    by_source_mode = (
        group_metrics(
            all_df,
            [
                "source",
                "mode",
            ],
        )
    )


    macro_source = (
        macro_metrics(
            by_source
        )
    )


    macro_source_mode = (
        macro_metrics(
            by_source_mode
        )
    )


    return (
        all_df,
        {
            "overall":
                overall,

            "macro_source":
                macro_source,

            "macro_source_mode":
                macro_source_mode,

            "by_source":
                by_source,

            "by_mode":
                by_mode,

            "by_source_mode":
                by_source_mode,

            "num_shards":
                int(
                    len(files)
                ),
        },
    )


# ============================================================
# Human-readable tables
# ============================================================

def pct(
    value,
):
    if value is None:
        return np.nan

    return (
        100.0
        * float(
            value
        )
    )


def build_table(
    dataset_name,
    result,
):

    rows = []


    overall = result[
        "overall"
    ]

    rows.append({
        "dataset":
            dataset_name,

        "group":
            "MICRO overall",

        "n":
            overall["n"],

        "AUC_%":
            pct(
                overall["auc"]
            ),

        "AP_%":
            pct(
                overall["ap"]
            ),

        "ACC_%":
            pct(
                overall["acc"]
            ),

        "fake_B1_top1_%":
            pct(
                overall.get(
                    "fake_top1_B1"
                )
            ),

        "fake_B2_top1_%":
            pct(
                overall.get(
                    "fake_top1_B2"
                )
            ),

        "fake_B1_top10_%":
            pct(
                overall.get(
                    "fake_top10_B1"
                )
            ),

        "fake_B2_top10_%":
            pct(
                overall.get(
                    "fake_top10_B2"
                )
            ),
    })


    macro = result[
        "macro_source"
    ]

    rows.append({
        "dataset":
            dataset_name,

        "group":
            "MACRO source",

        "n":
            macro[
                "num_groups"
            ],

        "AUC_%":
            pct(
                macro["auc"]
            ),

        "AP_%":
            pct(
                macro["ap"]
            ),

        "ACC_%":
            pct(
                macro["acc"]
            ),

        "fake_B1_top1_%":
            pct(
                macro.get(
                    "fake_top1_B1"
                )
            ),

        "fake_B2_top1_%":
            pct(
                macro.get(
                    "fake_top1_B2"
                )
            ),

        "fake_B1_top10_%":
            pct(
                macro.get(
                    "fake_top10_B1"
                )
            ),

        "fake_B2_top10_%":
            pct(
                macro.get(
                    "fake_top10_B2"
                )
            ),
    })


    if (
        dataset_name
        == "AIGI-Now"
    ):

        macro_sm = result[
            "macro_source_mode"
        ]

        rows.append({
            "dataset":
                dataset_name,

            "group":
                "MACRO source-mode",

            "n":
                macro_sm[
                    "num_groups"
                ],

            "AUC_%":
                pct(
                    macro_sm["auc"]
                ),

            "AP_%":
                pct(
                    macro_sm["ap"]
                ),

            "ACC_%":
                pct(
                    macro_sm["acc"]
                ),

            "fake_B1_top1_%":
                pct(
                    macro_sm.get(
                        "fake_top1_B1"
                    )
                ),

            "fake_B2_top1_%":
                pct(
                    macro_sm.get(
                        "fake_top1_B2"
                    )
                ),

            "fake_B1_top10_%":
                pct(
                    macro_sm.get(
                        "fake_top10_B1"
                    )
                ),

            "fake_B2_top10_%":
                pct(
                    macro_sm.get(
                        "fake_top10_B2"
                    )
                ),
        })


    for source, m in result[
        "by_source"
    ].items():

        rows.append({
            "dataset":
                dataset_name,

            "group":
                f"source:{source}",

            "n":
                m["n"],

            "AUC_%":
                pct(
                    m["auc"]
                ),

            "AP_%":
                pct(
                    m["ap"]
                ),

            "ACC_%":
                pct(
                    m["acc"]
                ),

            "fake_B1_top1_%":
                pct(
                    m.get(
                        "fake_top1_B1"
                    )
                ),

            "fake_B2_top1_%":
                pct(
                    m.get(
                        "fake_top1_B2"
                    )
                ),

            "fake_B1_top10_%":
                pct(
                    m.get(
                        "fake_top10_B1"
                    )
                ),

            "fake_B2_top10_%":
                pct(
                    m.get(
                        "fake_top10_B2"
                    )
                ),
        })


    return pd.DataFrame(
        rows
    )


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()


    ap.add_argument(
        "--checkpoint",
        required=True,
    )

    ap.add_argument(
        "--genimage_root",
        default=(
            "features/"
            "genimage_full_mid20_shards"
        ),
    )

    ap.add_argument(
        "--aiginow_root",
        default=(
            "features/"
            "aiginow_full_mid20_shards"
        ),
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    ap.add_argument(
        "--grid",
        type=int,
        default=14,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )


    args = ap.parse_args()


    # ========================================================
    # Output
    # ========================================================

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    # ========================================================
    # Device / checkpoint
    # ========================================================

    args.device_obj = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 100)
    print("LOADING CHECKPOINT")
    print("=" * 100)

    print(
        args.checkpoint
    )


    ckpt = torch.load(
        args.checkpoint,
        map_location="cpu",
    )


    ckpt_args = ckpt.get(
        "args",
        {},
    )


    feat_dim = int(
        ckpt_args.get(
            "feat_dim",
            4096,
        )
    )

    hidden_dim = int(
        ckpt_args.get(
            "hidden_dim",
            256,
        )
    )

    dropout = float(
        ckpt_args.get(
            "dropout",
            0.1,
        )
    )


    args.pool = ckpt_args.get(
        "pool",
        "topk",
    )

    args.topk = int(
        ckpt_args.get(
            "topk",
            10,
        )
    )

    args.lse_tau = float(
        ckpt_args.get(
            "lse_tau",
            1.0,
        )
    )


    print(
        "best_epoch:",
        ckpt.get(
            "best_epoch"
        ),
    )

    print(
        "best_score:",
        ckpt.get(
            "best_score"
        ),
    )

    print(
        "feat_dim:",
        feat_dim,
    )

    print(
        "pool:",
        args.pool,
    )

    print(
        "topk:",
        args.topk,
    )


    if feat_dim != 4096:
        raise RuntimeError(
            "Expected ViT-7B feat_dim=4096, "
            f"got {feat_dim}"
        )


    # ========================================================
    # Model
    # ========================================================

    model = TopKMILScorer(
        feat_dim=feat_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(
        args.device_obj
    )


    model.load_state_dict(
        ckpt[
            "model_state_dict"
        ]
    )


    model.eval()


    mu = (
        ckpt[
            "mu"
        ]
        .float()
    )

    std = (
        ckpt[
            "std"
        ]
        .float()
    )


    if (
        tuple(
            mu.shape
        )
        != (
            4096,
        )
    ):
        raise RuntimeError(
            f"Bad mu shape: {tuple(mu.shape)}"
        )


    if (
        tuple(
            std.shape
        )
        != (
            4096,
        )
    ):
        raise RuntimeError(
            f"Bad std shape: {tuple(std.shape)}"
        )


    # ========================================================
    # GenImage
    # ========================================================

    gen_df, gen_result = (
        evaluate_genimage(
            args.genimage_root,
            model,
            mu,
            std,
            args,
        )
    )


    gen_df.to_csv(
        output_dir
        / "genimage_full_per_sample.csv.gz",
        index=False,
        compression="gzip",
    )


    with (
        output_dir
        / "genimage_full_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            gen_result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # Release compact DF later only after table building.
    gc.collect()


    # ========================================================
    # AIGI-Now
    # ========================================================

    aigi_df, aigi_result = (
        evaluate_aiginow(
            args.aiginow_root,
            model,
            mu,
            std,
            args,
        )
    )


    aigi_df.to_csv(
        output_dir
        / "aiginow_full_per_sample.csv.gz",
        index=False,
        compression="gzip",
    )


    with (
        output_dir
        / "aiginow_full_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            aigi_result,
            f,
            indent=2,
            ensure_ascii=False,
        )


    # ========================================================
    # Combined readable summary
    # ========================================================

    table = pd.concat(
        [
            build_table(
                "GenImage",
                gen_result,
            ),
            build_table(
                "AIGI-Now",
                aigi_result,
            ),
        ],
        ignore_index=True,
    )


    table.to_csv(
        output_dir
        / "combined_metrics.csv",
        index=False,
    )


    # Main compact result table.
    main_rows = []

    for name, result in [
        (
            "GenImage",
            gen_result,
        ),
        (
            "AIGI-Now",
            aigi_result,
        ),
    ]:

        m = result[
            "overall"
        ]

        main_rows.append({
            "dataset":
                name,

            "n":
                m["n"],

            "AUC_%":
                pct(
                    m["auc"]
                ),

            "AP_%":
                pct(
                    m["ap"]
                ),

            "ACC_%":
                pct(
                    m["acc"]
                ),

            "fake_B1_top1_%":
                pct(
                    m[
                        "fake_top1_B1"
                    ]
                ),

            "fake_B2_top1_%":
                pct(
                    m[
                        "fake_top1_B2"
                    ]
                ),

            "fake_B1_top10_%":
                pct(
                    m[
                        "fake_top10_B1"
                    ]
                ),

            "fake_B2_top10_%":
                pct(
                    m[
                        "fake_top10_B2"
                    ]
                ),
        })


    main_table = pd.DataFrame(
        main_rows
    )


    main_table.to_csv(
        output_dir
        / "main_results.csv",
        index=False,
    )


    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 100)
    print("FULL SD1.4 EVIRoute — MAIN RESULTS")
    print("=" * 100)

    print(
        main_table.to_string(
            index=False,
            float_format=lambda x: (
                f"{x:.2f}"
            ),
        )
    )


    print()
    print("=" * 100)
    print("MACRO RESULTS")
    print("=" * 100)

    for name, result in [
        (
            "GenImage",
            gen_result,
        ),
        (
            "AIGI-Now",
            aigi_result,
        ),
    ]:

        m = result[
            "macro_source"
        ]

        print(
            f"{name:10s} "
            f"Macro AUC={pct(m['auc']):.2f}%  "
            f"AP={pct(m['ap']):.2f}%  "
            f"ACC={pct(m['acc']):.2f}%"
        )


    print()
    print(
        "Saved to:",
        output_dir,
    )


if __name__ == "__main__":
    main()
