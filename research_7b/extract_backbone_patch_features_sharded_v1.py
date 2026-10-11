#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Memory-safe sharded Mid20 feature extraction.

Key properties:
1. Load DINOv3 exactly ONCE.
2. Process shard CSVs sequentially in the same Python process.
3. Save one train.pt per shard.
4. Release CPU feature payload after every shard.
5. Resume automatically by skipping completed train.pt files.
6. Reuse the ORIGINAL extraction implementation:
   CSVDataset / collate_fn / extract_split / DinoV3Model.

Thus image preprocessing, DINOv3 forward, Mid20 hook,
patch slicing, and feature dtype remain unchanged.
"""

import argparse
import gc
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

import extract_backbone_patch_features_v1 as base


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--csv_dir",
        required=True,
    )

    ap.add_argument(
        "--output_root",
        required=True,
    )

    # REQUIRED deliberately:
    # do not accidentally fall back to ViT-B.
    ap.add_argument(
        "--model_name",
        required=True,
    )

    ap.add_argument(
        "--img_size",
        type=int,
        default=224,
    )

    ap.add_argument(
        "--mid_block",
        type=int,
        default=20,
    )

    ap.add_argument(
        "--batch_size",
        type=int,
        default=16,
    )

    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )

    ap.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = ap.parse_args()

    csv_dir = Path(args.csv_dir)
    output_root = Path(args.output_root)

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = sorted(
        csv_dir.glob("shard[0-9][0-9][0-9].csv")
    )

    if not csv_files:
        raise RuntimeError(
            f"No shard CSVs found in {csv_dir}"
        )

    print("=" * 100, flush=True)
    print("SHARDED DINOv3 FEATURE EXTRACTION", flush=True)
    print("=" * 100, flush=True)

    print(
        "CSV shards :",
        len(csv_files),
        flush=True,
    )

    print(
        "model_name :",
        args.model_name,
        flush=True,
    )

    print(
        "img_size   :",
        args.img_size,
        flush=True,
    )

    print(
        "mid_block  :",
        args.mid_block,
        flush=True,
    )

    print(
        "batch_size :",
        args.batch_size,
        flush=True,
    )

    print(
        "workers    :",
        args.num_workers,
        flush=True,
    )

    print(
        "device     :",
        args.device,
        flush=True,
    )


    # ========================================================
    # Determine pending shards BEFORE loading 7B model
    # ========================================================

    pending = []

    for csv_path in csv_files:

        shard = csv_path.stem

        output_path = (
            output_root
            / shard
            / "train.pt"
        )

        if (
            output_path.exists()
            and output_path.stat().st_size > 1_000_000
        ):
            print(
                f"[Already complete] {shard}",
                flush=True,
            )
            continue

        pending.append(
            csv_path
        )

    print()
    print(
        f"Completed : {len(csv_files) - len(pending)}",
        flush=True,
    )

    print(
        f"Pending   : {len(pending)}",
        flush=True,
    )

    if not pending:
        print("\nNothing to do.", flush=True)
        return


    # ========================================================
    # LOAD DINOv3 ONCE
    # ========================================================

    print()
    print("=" * 100, flush=True)
    print("LOADING DINOv3 ONCE", flush=True)
    print("=" * 100, flush=True)

    model = base.DinoV3Model(
        model_name=args.model_name,
        img_size=args.img_size,
        pool_type="patch_avg",
    ).to(args.device)

    # Same metadata convention as original extractor.
    model.model_name = args.model_name

    model.eval()

    backbone = model.backbone

    print(
        "num_blocks:",
        len(backbone.blocks),
        flush=True,
    )

    print(
        "feature_dim:",
        getattr(
            backbone,
            "num_features",
            None,
        ),
        flush=True,
    )

    print(
        "num_prefix_tokens:",
        getattr(
            backbone,
            "num_prefix_tokens",
            None,
        ),
        flush=True,
    )

    print(
        "patch_grid:",
        getattr(
            backbone.patch_embed,
            "grid_size",
            None,
        ),
        flush=True,
    )


    # Strong safety checks for the frozen EviRoute setup.
    feature_dim = int(
        getattr(
            backbone,
            "num_features",
            -1,
        )
    )

    if feature_dim != 4096:
        raise RuntimeError(
            f"Expected DINOv3 ViT-7B feature_dim=4096, "
            f"got {feature_dim}. "
            f"STOPPING to avoid extracting with the wrong backbone."
        )

    if len(backbone.blocks) <= args.mid_block:
        raise RuntimeError(
            f"mid_block={args.mid_block} invalid for "
            f"{len(backbone.blocks)} blocks."
        )

    patch_grid = getattr(
        backbone.patch_embed,
        "grid_size",
        None,
    )

    if tuple(patch_grid) != (14, 14):
        raise RuntimeError(
            f"Expected 14x14 patch grid, got {patch_grid}"
        )

    print()
    print(
        "[OK] Backbone sanity checks passed.",
        flush=True,
    )

    print(
        "[OK] DINOv3 will remain resident on GPU "
        "for all remaining shards.",
        flush=True,
    )


    # ========================================================
    # Sequential shard extraction
    # ========================================================

    total_pending = len(pending)

    for pending_index, csv_path in enumerate(
        pending,
        start=1,
    ):

        shard = csv_path.stem

        shard_dir = (
            output_root
            / shard
        )

        shard_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = (
            shard_dir
            / "train.pt"
        )

        # Atomic-save temporary path:
        # if process dies during torch.save, no fake complete
        # train.pt is left behind.
        temp_path = (
            shard_dir
            / "train.pt.tmp"
        )

        if temp_path.exists():
            temp_path.unlink()

        print()
        print("=" * 100, flush=True)

        print(
            f"[{pending_index}/{total_pending}] "
            f"{shard}",
            flush=True,
        )

        print(
            "CSV:",
            csv_path,
            flush=True,
        )

        print("=" * 100, flush=True)


        # ----------------------------------------------------
        # Load only this shard CSV
        # ----------------------------------------------------

        frame = pd.read_csv(
            csv_path
        )

        required_columns = {
            "path",
            "label",
            "source",
            "case_name",
            "split",
        }

        missing = (
            required_columns
            - set(frame.columns)
        )

        if missing:
            raise KeyError(
                f"{csv_path}: missing columns "
                f"{sorted(missing)}"
            )

        subset = (
            frame[
                frame["split"] == "train"
            ]
            .reset_index(drop=True)
        )

        if len(subset) == 0:
            raise RuntimeError(
                f"{csv_path}: no train samples"
            )

        print(
            "samples:",
            len(subset),
            flush=True,
        )

        print(
            "fake:",
            int(
                (subset["label"] == 1).sum()
            ),
            flush=True,
        )

        print(
            "real:",
            int(
                (subset["label"] == 0).sum()
            ),
            flush=True,
        )


        # ----------------------------------------------------
        # Same Dataset / collate / extraction as original
        # ----------------------------------------------------

        dataset = base.CSVDataset(
            subset
        )

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=base.collate_fn,
        )

        payload = base.extract_split(
            model=model,
            loader=loader,
            mid_block=args.mid_block,
            device=args.device,
            save_patch=True,
        )


        # ----------------------------------------------------
        # Strong output validation
        # ----------------------------------------------------

        patches = payload[
            "patch_features"
        ]

        means = payload[
            "mean_features"
        ]

        labels = payload[
            "labels"
        ]

        expected_n = len(subset)

        if tuple(
            patches.shape
        ) != (
            expected_n,
            196,
            4096,
        ):
            raise RuntimeError(
                f"{shard}: unexpected patch shape "
                f"{tuple(patches.shape)}"
            )

        if tuple(
            means.shape
        ) != (
            expected_n,
            4096,
        ):
            raise RuntimeError(
                f"{shard}: unexpected mean shape "
                f"{tuple(means.shape)}"
            )

        if int(
            labels.shape[0]
        ) != expected_n:
            raise RuntimeError(
                f"{shard}: label count mismatch"
            )

        if patches.dtype != torch.float16:
            raise RuntimeError(
                f"{shard}: expected patch_features float16, "
                f"got {patches.dtype}"
            )


        # ----------------------------------------------------
        # Atomic save
        # ----------------------------------------------------

        print(
            "[*] Saving:",
            output_path,
            flush=True,
        )

        torch.save(
            payload,
            temp_path,
        )

        os.replace(
            temp_path,
            output_path,
        )

        size_gib = (
            output_path.stat().st_size
            / (1024 ** 3)
        )

        print(
            f"[Saved] {output_path} "
            f"({size_gib:.2f} GiB)",
            flush=True,
        )


        # ----------------------------------------------------
        # Release THIS shard only.
        #
        # Important:
        # model/backbone are NOT deleted.
        # ----------------------------------------------------

        del payload
        del patches
        del means
        del labels

        del loader
        del dataset
        del subset
        del frame

        gc.collect()

        # Keep the 7B model resident.
        # empty_cache only releases unused CUDA cache;
        # model parameters remain allocated.
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(
            f"[Released CPU buffers] {shard}",
            flush=True,
        )


    print()
    print("=" * 100, flush=True)
    print("ALL SHARDS COMPLETE", flush=True)
    print("=" * 100, flush=True)


if __name__ == "__main__":
    main()
