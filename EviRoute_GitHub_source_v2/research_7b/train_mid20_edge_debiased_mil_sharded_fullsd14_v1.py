#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Sharded full-SD1.4 trainer for EviRoute.

Scientific goal:
- Replace only the original smoke2k training split by the
  complete SD1.4 training split.
- Keep the original validation/test sets.
- Reuse the exact original:
    TopKMILScorer
    normalization definition
    dropout mechanism
    top-k pooling
    edge-mass loss
    evaluation
    checkpoint selection

Memory strategy:
- one train shard in RAM at a time
- global mu/std computed from all train mean_features
- val/test remain the original fixed payloads
"""

import argparse
import gc
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_mid20_edge_debiased_mil_v2 as base


# ============================================================
# Utilities
# ============================================================

def atomic_torch_save(obj, path):
    path = Path(path)
    tmp = Path(str(path) + ".tmp")

    if tmp.exists():
        tmp.unlink()

    torch.save(obj, tmp)
    os.replace(tmp, path)


def cpu_state_dict(model):
    return {
        k: v.detach().cpu()
        for k, v in model.state_dict().items()
    }


def discover_shards(root):
    root = Path(root)

    shards = sorted(
        root.glob("shard*/train.pt")
    )

    if not shards:
        raise RuntimeError(
            f"No shard*/train.pt found under {root}"
        )

    return shards


# ============================================================
# Full-train normalization
# ============================================================

def compute_or_load_global_norm(
    shards,
    cache_path,
    feat_dim,
):
    """
    Match the original normalization definition:

        x = train_payload["mean_features"].float()
        mu = x.mean(dim=0)
        std = x.std(dim=0).clamp_min(1e-6)

    We retain only mean_features from each shard.
    patch_features are released immediately.
    """

    cache_path = Path(cache_path)

    if cache_path.exists():

        print(
            f"[Norm] Loading cache: {cache_path}",
            flush=True,
        )

        d = torch.load(
            cache_path,
            map_location="cpu",
        )

        mu = d["mu"].float()
        std = d["std"].float()
        n = int(d["n"])

        if tuple(mu.shape) != (feat_dim,):
            raise RuntimeError(
                f"Cached mu shape invalid: {tuple(mu.shape)}"
            )

        if tuple(std.shape) != (feat_dim,):
            raise RuntimeError(
                f"Cached std shape invalid: {tuple(std.shape)}"
            )

        print(
            f"[Norm] n={n}",
            flush=True,
        )

        return mu, std, n


    print()
    print("=" * 90)
    print("COMPUTING GLOBAL FULL-TRAIN NORMALIZATION")
    print("=" * 90)

    mean_chunks = []
    total_n = 0

    t0 = time.perf_counter()

    for i, p in enumerate(
        shards,
        start=1,
    ):

        d = torch.load(
            p,
            map_location="cpu",
        )

        if "mean_features" not in d:
            raise RuntimeError(
                f"{p}: mean_features missing"
            )

        x = d[
            "mean_features"
        ].float()

        if (
            x.ndim != 2
            or x.shape[1] != feat_dim
        ):
            raise RuntimeError(
                f"{p}: invalid mean_features "
                f"{tuple(x.shape)}"
            )

        # clone so that the retained tensor is independent
        # from the large shard payload.
        mean_chunks.append(
            x.clone()
        )

        total_n += int(
            x.shape[0]
        )

        del x
        del d

        gc.collect()

        if (
            i % 10 == 0
            or i == len(shards)
        ):
            elapsed = (
                time.perf_counter()
                - t0
            )

            print(
                f"[Norm] "
                f"{i}/{len(shards)} shards, "
                f"n={total_n}, "
                f"{elapsed / 60:.1f} min",
                flush=True,
            )


    print(
        "[Norm] Concatenating mean features...",
        flush=True,
    )

    all_mean = torch.cat(
        mean_chunks,
        dim=0,
    )

    del mean_chunks
    gc.collect()


    # EXACT same mathematical definition
    # as original trainer.
    mu = all_mean.mean(
        dim=0
    )

    std = (
        all_mean.std(
            dim=0
        )
        .clamp_min(
            1e-6
        )
    )


    if tuple(mu.shape) != (feat_dim,):
        raise RuntimeError(
            f"mu shape invalid: {tuple(mu.shape)}"
        )

    if tuple(std.shape) != (feat_dim,):
        raise RuntimeError(
            f"std shape invalid: {tuple(std.shape)}"
        )


    del all_mean
    gc.collect()


    cache = {
        "mu": mu.cpu(),
        "std": std.cpu(),
        "n": int(total_n),
        "num_shards": int(
            len(shards)
        ),
    }

    atomic_torch_save(
        cache,
        cache_path,
    )

    print(
        f"[Norm] Saved: {cache_path}",
        flush=True,
    )

    print(
        f"[Norm] total n={total_n}",
        flush=True,
    )

    return (
        mu,
        std,
        total_n,
    )


# ============================================================
# Sharded epoch
# ============================================================

def train_sharded_epoch(
    model,
    shard_paths,
    mu,
    std,
    optimizer,
    bce_loss,
    args,
    device,
    epoch,
):

    epoch_start = time.perf_counter()

    # Deterministic shard-order shuffle.
    order = list(
        shard_paths
    )

    rng = random.Random(
        args.seed
        + epoch
    )

    rng.shuffle(
        order
    )


    total_loss = 0.0
    total_cls = 0.0
    total_edge = 0.0
    total_n = 0


    for shard_idx, shard_path in enumerate(
        order,
        start=1,
    ):

        shard_start = (
            time.perf_counter()
        )

        payload = torch.load(
            shard_path,
            map_location="cpu",
        )


        patches = payload[
            "patch_features"
        ]

        labels = payload[
            "labels"
        ]


        if (
            patches.ndim != 3
            or patches.shape[1] != 196
            or patches.shape[2] != args.feat_dim
        ):
            raise RuntimeError(
                f"{shard_path}: "
                f"bad patch shape "
                f"{tuple(patches.shape)}"
            )


        n = int(
            labels.shape[0]
        )


        dataset = (
            base.PatchFeatureDataset(
                payload
            )
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False,
        )


        info = base.train_one_epoch(
            model=model,
            loader=loader,
            mu=mu,
            std=std,
            optimizer=optimizer,
            bce_loss=bce_loss,
            args=args,
            device=device,
        )


        total_loss += (
            info["loss"]
            * n
        )

        total_cls += (
            info["loss_cls"]
            * n
        )

        total_edge += (
            info["loss_edge"]
            * n
        )

        total_n += n


        del loader
        del dataset
        del patches
        del labels
        del payload

        gc.collect()


        shard_seconds = (
            time.perf_counter()
            - shard_start
        )


        if (
            shard_idx % 10 == 0
            or shard_idx == len(order)
        ):

            epoch_seconds = (
                time.perf_counter()
                - epoch_start
            )

            print(
                f"  [Epoch {epoch:03d}] "
                f"shard "
                f"{shard_idx:03d}/"
                f"{len(order):03d} "
                f"n={total_n} "
                f"last={shard_seconds:.1f}s "
                f"elapsed={epoch_seconds / 60:.1f}min",
                flush=True,
            )


    epoch_seconds = (
        time.perf_counter()
        - epoch_start
    )


    return {
        "loss":
            total_loss
            / max(
                total_n,
                1,
            ),

        "loss_cls":
            total_cls
            / max(
                total_n,
                1,
            ),

        "loss_edge":
            total_edge
            / max(
                total_n,
                1,
            ),

        "train_n":
            int(
                total_n
            ),

        "epoch_seconds":
            float(
                epoch_seconds
            ),
    }


# ============================================================
# Main
# ============================================================

def main():

    ap = argparse.ArgumentParser()


    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    ap.add_argument(
        "--train_shard_root",
        required=True,
    )

    ap.add_argument(
        "--eval_feature_dir",
        required=True,
    )

    ap.add_argument(
        "--output_dir",
        required=True,
    )

    ap.add_argument(
        "--norm_cache",
        default=None,
    )

    ap.add_argument(
        "--expected_shards",
        type=int,
        default=162,
    )


    # --------------------------------------------------------
    # Original model hyperparameters
    # --------------------------------------------------------

    ap.add_argument(
        "--feat_dim",
        type=int,
        default=4096,
    )

    ap.add_argument(
        "--hidden_dim",
        type=int,
        default=256,
    )

    ap.add_argument(
        "--dropout",
        type=float,
        default=0.1,
    )


    # --------------------------------------------------------
    # Original MIL
    # --------------------------------------------------------

    ap.add_argument(
        "--pool",
        type=str,
        default="topk",
        choices=[
            "topk",
            "lse",
            "mean",
        ],
    )

    ap.add_argument(
        "--topk",
        type=int,
        default=10,
    )

    ap.add_argument(
        "--lse_tau",
        type=float,
        default=1.0,
    )


    # --------------------------------------------------------
    # Original dropout
    # --------------------------------------------------------

    ap.add_argument(
        "--token_dropout",
        type=float,
        default=0.15,
    )

    ap.add_argument(
        "--edge1_dropout",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--edge2_only_dropout",
        type=float,
        default=0.25,
    )


    # --------------------------------------------------------
    # Original edge penalty
    # --------------------------------------------------------

    ap.add_argument(
        "--lambda_edge1",
        type=float,
        default=0.01,
    )

    ap.add_argument(
        "--rho_edge1",
        type=float,
        default=0.4,
    )

    ap.add_argument(
        "--lambda_edge2",
        type=float,
        default=0.0,
    )

    ap.add_argument(
        "--rho_edge2",
        type=float,
        default=0.6,
    )

    ap.add_argument(
        "--edge_tau",
        type=float,
        default=1.0,
    )


    # --------------------------------------------------------
    # Optimization
    # --------------------------------------------------------

    ap.add_argument(
        "--epochs",
        type=int,
        default=1,
    )

    ap.add_argument(
        "--lr",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--weight_decay",
        type=float,
        default=1e-4,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--eval_batch_size",
        type=int,
        default=32,
    )


    # --------------------------------------------------------
    # Misc
    # --------------------------------------------------------

    ap.add_argument(
        "--grid",
        type=int,
        default=14,
    )

    ap.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    ap.add_argument(
        "--device",
        default="cuda",
    )


    # --------------------------------------------------------
    # Original checkpoint selection
    # --------------------------------------------------------

    ap.add_argument(
        "--min_auc_for_edge_select",
        type=float,
        default=0.995,
    )

    ap.add_argument(
        "--select_edge1_weight",
        type=float,
        default=1.0,
    )

    ap.add_argument(
        "--select_edge2_weight",
        type=float,
        default=0.5,
    )

    ap.add_argument(
        "--low_auc_penalty",
        type=float,
        default=5.0,
    )


    args = ap.parse_args()


    # ========================================================
    # Setup
    # ========================================================

    base.set_seed(
        args.seed
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    device = torch.device(
        args.device
        if torch.cuda.is_available()
        else "cpu"
    )


    shard_paths = discover_shards(
        args.train_shard_root
    )


    print("=" * 90)
    print("FULL SD1.4 SHARDED EVIRoute TRAINING")
    print("=" * 90)

    print(
        "train shards:",
        len(shard_paths),
    )

    print(
        "train root:",
        args.train_shard_root,
    )

    print(
        "eval dir:",
        args.eval_feature_dir,
    )

    print(
        "epochs:",
        args.epochs,
    )

    print(
        "device:",
        device,
    )


    if (
        args.expected_shards > 0
        and len(shard_paths)
        != args.expected_shards
    ):
        raise RuntimeError(
            f"Expected "
            f"{args.expected_shards} shards, "
            f"found {len(shard_paths)}"
        )


    # ========================================================
    # Load fixed original val/test
    # ========================================================

    print()
    print(
        "[*] Loading original validation/test...",
        flush=True,
    )


    val_payload = base.load_split(
        args.eval_feature_dir,
        "val",
    )

    test_payload = base.load_split(
        args.eval_feature_dir,
        "test",
    )


    print(
        "val patch:",
        tuple(
            val_payload[
                "patch_features"
            ].shape
        ),
    )

    print(
        "test patch:",
        tuple(
            test_payload[
                "patch_features"
            ].shape
        ),
    )


    # ========================================================
    # Global full-train normalization
    # ========================================================

    if args.norm_cache is None:
        norm_cache = (
            Path(
                args.train_shard_root
            )
            / "global_full_train_norm.pt"
        )
    else:
        norm_cache = Path(
            args.norm_cache
        )


    mu, std, norm_n = (
        compute_or_load_global_norm(
            shard_paths,
            norm_cache,
            args.feat_dim,
        )
    )


    if norm_n != 323997:
        print(
            f"[Warning] normalization n={norm_n}, "
            f"expected current freeze n=323997",
            flush=True,
        )


    # Keep normalization on GPU.
    # Original functions call .to(device), which is then a no-op.
    mu_device = mu.to(
        device
    )

    std_device = std.to(
        device
    )


    # ========================================================
    # Model / optimizer
    # ========================================================

    model = base.TopKMILScorer(
        feat_dim=args.feat_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(
        device
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )


    bce_loss = (
        nn.BCEWithLogitsLoss()
    )


    best_score = -1e9
    best_epoch = -1

    history = []


    # ========================================================
    # Training
    # ========================================================

    total_start = (
        time.perf_counter()
    )


    for epoch in range(
        1,
        args.epochs + 1,
    ):

        print()
        print("=" * 90)
        print(
            f"EPOCH {epoch}/{args.epochs}"
        )
        print("=" * 90)


        loss_info = (
            train_sharded_epoch(
                model=model,
                shard_paths=shard_paths,
                mu=mu_device,
                std=std_device,
                optimizer=optimizer,
                bce_loss=bce_loss,
                args=args,
                device=device,
                epoch=epoch,
            )
        )


        # ----------------------------------------------------
        # Original validation
        # ----------------------------------------------------

        val_out = base.evaluate_payload(
            model,
            val_payload,
            mu_device,
            std_device,
            args,
            device,
        )


        score = (
            base.checkpoint_score(
                val_out,
                args,
            )
        )


        val_cls = (
            val_out[
                "classification"
            ]
        )

        val_fake = (
            val_out[
                "edge_metrics"
            ][
                "fake"
            ]
        )


        record = {
            "epoch":
                int(
                    epoch
                ),

            **loss_info,

            "val_acc":
                val_cls[
                    "acc"
                ],

            "val_auc":
                val_cls[
                    "auc"
                ],

            "val_ap":
                val_cls[
                    "ap"
                ],

            "val_fake_top1_edge1":
                val_fake[
                    "top1_edge1_rate"
                ],

            "val_fake_top1_edge2":
                val_fake[
                    "top1_edge2_rate"
                ],

            "select_score":
                float(
                    score
                ),
        }


        history.append(
            record
        )


        print()
        print(
            f"[Epoch {epoch:03d}] "
            f"loss={loss_info['loss']:.6f} "
            f"cls={loss_info['loss_cls']:.6f} "
            f"edge={loss_info['loss_edge']:.6f} "
            f"val_acc={val_cls['acc']:.4f} "
            f"val_auc={val_cls['auc']:.6f} "
            f"val_ap={val_cls['ap']:.6f} "
            f"fake_e1="
            f"{val_fake['top1_edge1_rate']:.4f} "
            f"fake_e2="
            f"{val_fake['top1_edge2_rate']:.4f} "
            f"score={score:.6f} "
            f"time={loss_info['epoch_seconds']/60:.1f}min",
            flush=True,
        )


        # ----------------------------------------------------
        # Save history every epoch
        # ----------------------------------------------------

        with (
            output_dir
            / "training_history.json"
        ).open(
            "w",
            encoding="utf-8",
        ) as f:

            json.dump(
                history,
                f,
                indent=2,
            )


        # ----------------------------------------------------
        # Metadata compatible with old checkpoints
        # ----------------------------------------------------

        saved_args = vars(
            args
        ).copy()

        # compatibility field for older evaluation scripts
        saved_args[
            "feature_dir"
        ] = (
            args.eval_feature_dir
        )


        # ----------------------------------------------------
        # Original checkpoint rule
        # ----------------------------------------------------

        if score > best_score:

            best_score = float(
                score
            )

            best_epoch = int(
                epoch
            )


            ckpt = {
                "model_state_dict":
                    cpu_state_dict(
                        model
                    ),

                "mu":
                    mu.cpu(),

                "std":
                    std.cpu(),

                "args":
                    saved_args,

                "best_epoch":
                    best_epoch,

                "best_score":
                    best_score,

                "full_train_n":
                    int(
                        norm_n
                    ),

                "num_train_shards":
                    int(
                        len(
                            shard_paths
                        )
                    ),

                "train_shard_root":
                    str(
                        args.train_shard_root
                    ),

                "eval_feature_dir":
                    str(
                        args.eval_feature_dir
                    ),
            }


            atomic_torch_save(
                ckpt,
                output_dir
                / "mid20_edge_debiased_mil_best.pt",
            )


            # old-name compatibility
            atomic_torch_save(
                ckpt,
                output_dir
                / "mid20_topk_mil_best.pt",
            )


            print(
                f"[Best] epoch={best_epoch} "
                f"score={best_score:.6f}",
                flush=True,
            )


        # ----------------------------------------------------
        # Save last checkpoint for recovery/inspection
        # ----------------------------------------------------

        last_ckpt = {
            "model_state_dict":
                cpu_state_dict(
                    model
                ),

            "mu":
                mu.cpu(),

            "std":
                std.cpu(),

            "args":
                saved_args,

            "epoch":
                int(
                    epoch
                ),

            "best_epoch":
                int(
                    best_epoch
                ),

            "best_score":
                float(
                    best_score
                ),

            "history":
                history,
        }


        atomic_torch_save(
            last_ckpt,
            output_dir
            / "mid20_edge_debiased_mil_last.pt",
        )


    # ========================================================
    # Final test evaluation using BEST checkpoint
    # ========================================================

    best_path = (
        output_dir
        / "mid20_edge_debiased_mil_best.pt"
    )


    best_ckpt = torch.load(
        best_path,
        map_location="cpu",
    )


    model.load_state_dict(
        best_ckpt[
            "model_state_dict"
        ]
    )


    test_out = (
        base.evaluate_payload(
            model,
            test_payload,
            mu_device,
            std_device,
            args,
            device,
        )
    )


    total_seconds = (
        time.perf_counter()
        - total_start
    )


    summary = {
        "best_epoch":
            int(
                best_epoch
            ),

        "best_score":
            float(
                best_score
            ),

        "full_train_n":
            int(
                norm_n
            ),

        "num_train_shards":
            int(
                len(
                    shard_paths
                )
            ),

        "epochs":
            int(
                args.epochs
            ),

        "total_seconds":
            float(
                total_seconds
            ),

        "test":
            test_out,
    }


    with (
        output_dir
        / "summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            summary,
            f,
            indent=2,
        )


    print()
    print("=" * 90)
    print("TRAINING COMPLETE")
    print("=" * 90)

    print(
        "best_epoch:",
        best_epoch,
    )

    print(
        "best_score:",
        best_score,
    )

    print(
        "full_train_n:",
        norm_n,
    )

    print(
        "total time:",
        f"{total_seconds / 3600:.2f} h",
    )

    print(
        "best checkpoint:",
        best_path,
    )

    print(
        "summary:",
        output_dir
        / "summary.json",
    )


if __name__ == "__main__":
    main()
