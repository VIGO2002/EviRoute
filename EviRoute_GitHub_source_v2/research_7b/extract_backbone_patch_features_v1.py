#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from models.dinov3_models import DinoV3Model
from run_input_space_edge_diagnostic_v1 import (
    load_raw_image,
)


class CSVDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(
            drop=True
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]

        image = load_raw_image(
            row["path"]
        )

        return {
            "image": image,
            "label": int(row["label"]),
            "path": str(row["path"]),
            "case_name": str(row["case_name"]),
            "source": str(row["source"]),
        }


def collate_fn(batch):
    images = torch.stack(
        [item["image"] for item in batch],
        dim=0,
    )

    labels = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long,
    )

    paths = [
        item["path"]
        for item in batch
    ]
    case_names = [
        item["case_name"]
        for item in batch
    ]
    sources = [
        item["source"]
        for item in batch
    ]

    return (
        images,
        labels,
        paths,
        case_names,
        sources,
    )


def unwrap_tokens(output):
    if isinstance(output, torch.Tensor):
        return output

    if isinstance(output, (tuple, list)):
        for item in output:
            if (
                isinstance(item, torch.Tensor)
                and item.ndim == 3
            ):
                return item

    if isinstance(output, dict):
        for key in [
            "x",
            "tokens",
            "last_hidden_state",
        ]:
            value = output.get(key)

            if (
                isinstance(value, torch.Tensor)
                and value.ndim == 3
            ):
                return value

    raise RuntimeError(
        "Unable to obtain token tensor "
        f"[B,N,D] from output type "
        f"{type(output)}"
    )


@torch.inference_mode()
def extract_split(
    model,
    loader,
    mid_block,
    device,
    save_patch,
):
    backbone = model.backbone

    num_blocks = len(backbone.blocks)

    if not 0 <= mid_block < num_blocks:
        raise ValueError(
            f"mid_block={mid_block} is invalid "
            f"for num_blocks={num_blocks}"
        )

    patch_start = int(
        getattr(
            backbone,
            "num_prefix_tokens",
            5,
        )
    )

    grid_size = getattr(
        backbone.patch_embed,
        "grid_size",
        None,
    )

    if grid_size is None:
        raise RuntimeError(
            "backbone.patch_embed.grid_size "
            "is unavailable"
        )

    grid_h = int(grid_size[0])
    grid_w = int(grid_size[1])
    num_patches = grid_h * grid_w

    if (grid_h, grid_w) != (14, 14):
        raise RuntimeError(
            f"Expected 14x14 patch grid, "
            f"got {grid_h}x{grid_w}"
        )

    if patch_start != 5:
        raise RuntimeError(
            f"Expected five prefix tokens, "
            f"got {patch_start}"
        )

    mean_features = []
    patch_features = []
    labels_all = []
    paths_all = []
    case_names_all = []
    sources_all = []

    for batch_index, batch in enumerate(
        loader
    ):
        (
            images,
            labels,
            paths,
            case_names,
            sources,
        ) = batch

        images = images.to(
            device,
            non_blocking=True,
        )

        activations = {}

        def hook_fn(module, inputs, output):
            activations["mid"] = output

        handle = (
            backbone.blocks[mid_block]
            .register_forward_hook(hook_fn)
        )

        try:
            # 与原 7B 提取脚本一致：
            # 运行完整 forward_features，
            # 但读取指定 block 的原始输出。
            backbone.forward_features(images)
        finally:
            handle.remove()

        if "mid" not in activations:
            raise RuntimeError(
                "The intermediate hook did not "
                "capture an activation"
            )

        tokens = unwrap_tokens(
            activations["mid"]
        )

        expected_tokens = (
            patch_start + num_patches
        )

        if tokens.shape[1] < expected_tokens:
            raise RuntimeError(
                f"Expected at least "
                f"{expected_tokens} tokens, "
                f"got {tuple(tokens.shape)}"
            )

        patches = tokens[
            :,
            patch_start:
            patch_start + num_patches,
            :,
        ]

        if patches.shape[1] != 196:
            raise RuntimeError(
                f"Expected 196 patch tokens, "
                f"got {tuple(patches.shape)}"
            )

        pooled = patches.mean(dim=1)

        mean_features.append(
            pooled.detach().cpu().float()
        )
        labels_all.append(
            labels.detach().cpu()
        )

        paths_all.extend(paths)
        case_names_all.extend(case_names)
        sources_all.extend(sources)

        if save_patch:
            patch_features.append(
                patches.detach().cpu().half()
            )

        if (
            (batch_index + 1) % 20 == 0
            or batch_index + 1 == len(loader)
        ):
            print(
                f"  batch "
                f"{batch_index + 1}/"
                f"{len(loader)}"
            )

    payload = {
        "mean_features": torch.cat(
            mean_features,
            dim=0,
        ),
        "labels": torch.cat(
            labels_all,
            dim=0,
        ),
        "paths": paths_all,
        "case_names": case_names_all,
        "sources": sources_all,
        "layer": "mid",
        "mid_block": int(mid_block),
        "patch_start": int(patch_start),
        "num_patches": int(num_patches),
        "model_name": model.model_name
        if hasattr(model, "model_name")
        else None,
        "num_blocks": int(num_blocks),
        "feature_dim": int(
            patches.shape[-1]
        ),
        "patch_grid": [
            grid_h,
            grid_w,
        ],
    }

    if save_patch:
        payload["patch_features"] = (
            torch.cat(
                patch_features,
                dim=0,
            )
        )

    return payload


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--csv",
        required=True,
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="dinov3_vitb16",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=224,
    )
    parser.add_argument(
        "--mid_block",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--save_patch_splits",
        default="train,val,test",
    )
    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = parser.parse_args()

    output_dir = Path(
        args.output_dir
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    frame = pd.read_csv(args.csv)

    required_columns = {
        "path",
        "label",
        "source",
        "case_name",
        "split",
    }

    missing_columns = (
        required_columns
        - set(frame.columns)
    )

    if missing_columns:
        raise KeyError(
            f"CSV missing columns: "
            f"{sorted(missing_columns)}"
        )

    save_patch_splits = {
        value.strip()
        for value
        in args.save_patch_splits.split(",")
        if value.strip()
    }

    print("[*] Loading backbone")
    print("model_name:", args.model_name)
    print("img_size:", args.img_size)
    print("mid_block:", args.mid_block)

    model = DinoV3Model(
        model_name=args.model_name,
        img_size=args.img_size,
        pool_type="patch_avg",
    ).to(args.device)

    # 便于记录元数据。
    model.model_name = args.model_name
    model.eval()

    backbone = model.backbone

    print(
        "num_blocks:",
        len(backbone.blocks),
    )
    print(
        "feature_dim:",
        getattr(
            backbone,
            "num_features",
            None,
        ),
    )
    print(
        "num_prefix_tokens:",
        getattr(
            backbone,
            "num_prefix_tokens",
            None,
        ),
    )
    print(
        "patch_grid:",
        getattr(
            backbone.patch_embed,
            "grid_size",
            None,
        ),
    )

    for split in [
        "train",
        "val",
        "test",
    ]:
        subset = frame[
            frame["split"] == split
        ].reset_index(drop=True)

        if len(subset) == 0:
            print(
                f"[Skip] empty split: "
                f"{split}"
            )
            continue

        print(
            f"\n[*] Extracting "
            f"split={split}, "
            f"n={len(subset)}"
        )

        dataset = CSVDataset(subset)

        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        payload = extract_split(
            model=model,
            loader=loader,
            mid_block=args.mid_block,
            device=args.device,
            save_patch=(
                split in save_patch_splits
            ),
        )

        output_path = (
            output_dir / f"{split}.pt"
        )

        torch.save(
            payload,
            output_path,
        )

        print("[Saved]", output_path)
        print(
            "mean_features:",
            tuple(
                payload[
                    "mean_features"
                ].shape
            ),
            payload[
                "mean_features"
            ].dtype,
        )

        if "patch_features" in payload:
            print(
                "patch_features:",
                tuple(
                    payload[
                        "patch_features"
                    ].shape
                ),
                payload[
                    "patch_features"
                ].dtype,
            )

    print("\n[Done]")


if __name__ == "__main__":
    main()
