#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from run_input_space_edge_diagnostic_v1 import (
    load_protocol_module,
    load_raw_image,
)


PATCH_START = 5
NUM_PATCHES = 196


class CSVDataset(Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = load_raw_image(row["path"])
        return {
            "image": image,
            "label": int(row["label"]),
            "path": row["path"],
            "case_name": str(row["case_name"]),
            "source": str(row["source"]),
        }


def collate_fn(batch):
    images = torch.stack([x["image"] for x in batch], dim=0)
    labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    paths = [x["path"] for x in batch]
    case_names = [x["case_name"] for x in batch]
    sources = [x["source"] for x in batch]
    return images, labels, paths, case_names, sources


@torch.inference_mode()
def extract_split(
    model,
    loader,
    layer,
    mid_block,
    device,
    save_patch,
):
    mean_feats = []
    labels_all = []
    paths_all = []
    case_names_all = []
    sources_all = []
    patch_feats = []

    backbone = model.backbone

    for batch_idx, (images, labels, paths, case_names, sources) in enumerate(loader):
        images = images.to(device, non_blocking=True)

        acts = {}

        if layer == "mid":
            def hook_fn(module, inp, out):
                acts["mid"] = out

            handle = backbone.blocks[mid_block].register_forward_hook(hook_fn)

        tokens = backbone.forward_features(images)

        if layer == "mid":
            handle.remove()
            tokens = acts["mid"]

        if isinstance(tokens, dict):
            # 预留兼容。当前你的输出是 tensor。
            for key in ["x", "last_hidden_state", "tokens"]:
                if key in tokens:
                    tokens = tokens[key]
                    break

        if not isinstance(tokens, torch.Tensor):
            raise RuntimeError(f"Unexpected token type: {type(tokens)}")

        patches = tokens[:, PATCH_START:PATCH_START + NUM_PATCHES, :]

        if patches.shape[1] != NUM_PATCHES:
            raise RuntimeError(f"Expected 196 patch tokens, got {patches.shape}")

        pooled = patches.mean(dim=1)

        mean_feats.append(pooled.detach().cpu().float())
        labels_all.append(labels.detach().cpu())
        paths_all.extend(paths)
        case_names_all.extend(case_names)
        sources_all.extend(sources)

        if save_patch:
            patch_feats.append(patches.detach().cpu().half())

        if (batch_idx + 1) % 20 == 0:
            print(f"  batch {batch_idx + 1}/{len(loader)}")

    payload = {
        "mean_features": torch.cat(mean_feats, dim=0),
        "labels": torch.cat(labels_all, dim=0),
        "paths": paths_all,
        "case_names": case_names_all,
        "sources": sources_all,
        "layer": layer,
        "mid_block": mid_block,
        "patch_start": PATCH_START,
        "num_patches": NUM_PATCHES,
    }

    if save_patch:
        payload["patch_features"] = torch.cat(patch_feats, dim=0)

    return payload


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--protocol_file", default="repro_pack/fgts_core/test_system1_protocol.py")
    ap.add_argument("--layer", choices=["final", "mid"], required=True)
    ap.add_argument("--mid_block", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--save_patch_splits", default="val,test")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    save_patch_splits = {
        x.strip() for x in args.save_patch_splits.split(",") if x.strip()
    }

    df = pd.read_csv(args.csv)

    protocol = load_protocol_module(args.protocol_file)
    DinoV3Model = protocol.DinoV3Model

    print("[*] Loading DINOv3...")
    model = DinoV3Model(
        model_name="dinov3_vit_7b",
        pool_type="patch_avg",
    ).to(args.device)
    model.eval()

    for split in ["train", "val", "test"]:
        sub = df[df["split"] == split].reset_index(drop=True)
        if len(sub) == 0:
            print(f"[!] skip empty split: {split}")
            continue

        print(f"\n[*] Extracting split={split}, n={len(sub)}, layer={args.layer}")

        dataset = CSVDataset(sub)
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
            layer=args.layer,
            mid_block=args.mid_block,
            device=args.device,
            save_patch=(split in save_patch_splits),
        )

        out_path = os.path.join(args.output_dir, f"{split}.pt")
        torch.save(payload, out_path)
        print("saved:", out_path)

    print("\n[Done]")


if __name__ == "__main__":
    main()
