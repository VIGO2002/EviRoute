#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import glob
import json
import random
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import functional as TF


IMAGENET_MEAN = torch.tensor(
    [0.485, 0.456, 0.406], dtype=torch.float32
).view(1, 3, 1, 1)

IMAGENET_STD = torch.tensor(
    [0.229, 0.224, 0.225], dtype=torch.float32
).view(1, 3, 1, 1)


def load_protocol_module(protocol_file: str):
    protocol_file = os.path.abspath(protocol_file)
    protocol_dir = os.path.dirname(protocol_file)

    # 使 test_system1_protocol.py 的本地依赖可以正常导入
    for p in [
        os.getcwd(),
        protocol_dir,
        os.path.dirname(protocol_dir),
        os.path.dirname(os.path.dirname(protocol_dir)),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    spec = importlib.util.spec_from_file_location(
        "fgts_protocol_for_edge_diagnostic",
        protocol_file
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import protocol file: {protocol_file}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def infer_gt(case_name: str) -> str:
    s = str(case_name)
    if "__Fake__" in s:
        return "Fake"
    if "__Real__" in s:
        return "Real"
    return "Unknown"


def find_case_image(cases_root: str, case_name: str) -> str:
    case_dir = os.path.join(cases_root, case_name)

    patterns = [
        "*_original_224.png",
        "*original*.png",
        "*.png",
        "*.jpg",
        "*.jpeg",
        "*.webp",
    ]

    for pattern in patterns:
        paths = sorted(glob.glob(os.path.join(case_dir, pattern)))
        if paths:
            return paths[0]

    raise FileNotFoundError(f"No image found for case: {case_name}")


def load_raw_image(path: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = img.resize((224, 224), Image.BICUBIC)
    return TF.to_tensor(img)  # [3,224,224], range [0,1]


def patch_ids_to_coords(patch_ids, grid_size=14):
    return [(int(pid) // grid_size, int(pid) % grid_size)
            for pid in patch_ids]


def perturb_patches(
    image: torch.Tensor,
    patch_ids,
    mode: str,
    grid_size: int = 14,
) -> torch.Tensor:
    """
    image: [3,H,W], unnormalized, range [0,1]
    """
    out = image.clone()
    _, h, w = image.shape

    ph = h // grid_size
    pw = w // grid_size

    if mode == "mean":
        source = image.mean(dim=(1, 2), keepdim=True).expand_as(image)

    elif mode == "blur":
        # 先模糊整图，再只复制指定patch
        source = TF.gaussian_blur(
            image,
            kernel_size=[17, 17],
            sigma=[5.0, 5.0],
        )

    else:
        raise ValueError(f"Unknown perturbation mode: {mode}")

    for pid in patch_ids:
        pid = int(pid)
        row, col = divmod(pid, grid_size)

        y0 = row * ph
        y1 = h if row == grid_size - 1 else (row + 1) * ph
        x0 = col * pw
        x1 = w if col == grid_size - 1 else (col + 1) * pw

        out[:, y0:y1, x0:x1] = source[:, y0:y1, x0:x1]

    return out


def make_center_patch_ids(
    num_patches: int,
    fisher_patch_ids,
    grid_size: int = 14,
):
    fisher_set = set(int(x) for x in fisher_patch_ids)
    center = (grid_size - 1) / 2.0

    candidates = []
    for pid in range(grid_size * grid_size):
        if pid in fisher_set:
            continue

        row, col = divmod(pid, grid_size)
        distance = (row - center) ** 2 + (col - center) ** 2
        candidates.append((distance, pid))

    candidates.sort(key=lambda x: (x[0], x[1]))
    return [pid for _, pid in candidates[:num_patches]]


def normalize_batch(batch: torch.Tensor, device: str) -> torch.Tensor:
    mean = IMAGENET_MEAN.to(device)
    std = IMAGENET_STD.to(device)
    return (batch.to(device) - mean) / std


@torch.inference_mode()
def infer_batch(
    raw_images,
    model,
    probe,
    fisher_abs_indices,
    real_idx,
    fake_idx,
    device,
    batch_size,
):
    margins = []
    fake_probs = []

    fisher_tensor = torch.tensor(
        fisher_abs_indices,
        dtype=torch.long,
        device=device,
    )

    for start in range(0, len(raw_images), batch_size):
        batch = torch.stack(
            raw_images[start:start + batch_size],
            dim=0,
        )

        inputs = normalize_batch(batch, device)

        outputs = model(inputs, return_protocol=True)
        token_sequence = outputs["token_sequence"]

        selected_tokens = token_sequence.index_select(
            dim=1,
            index=fisher_tensor,
        )

        logits = probe(selected_tokens)
        probs = torch.softmax(logits, dim=-1)

        margin = logits[:, fake_idx] - logits[:, real_idx]

        margins.extend(margin.detach().cpu().tolist())
        fake_probs.extend(
            probs[:, fake_idx].detach().cpu().tolist()
        )

    return np.asarray(margins), np.asarray(fake_probs)


def summarize_group(df: pd.DataFrame, name: str):
    if len(df) == 0:
        return {
            "group": name,
            "n": 0,
        }

    return {
        "group": name,
        "n": int(len(df)),
        "mean_baseline_margin": float(df["baseline_margin"].mean()),
        "mean_fisher_anchor_drop": float(
            df["fisher_anchor_drop"].mean()
        ),
        "mean_center_drop": float(df["center_drop"].mean()),
        "mean_random_drop": float(df["random_drop_mean"].mean()),
        "mean_fisher_adv_over_random": float(
            df["fisher_adv_over_random"].mean()
        ),
        "mean_fisher_adv_over_center": float(
            df["fisher_adv_over_center"].mean()
        ),
        "fisher_gt_random_rate": float(
            (df["fisher_adv_over_random"] > 0).mean()
        ),
        "fisher_gt_center_rate": float(
            (df["fisher_adv_over_center"] > 0).mean()
        ),
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input_csv",
        required=True,
    )
    parser.add_argument(
        "--cases_root",
        default="aigi_now_eval/probe_genimage/cases_unique",
    )
    parser.add_argument(
        "--protocol_file",
        default="repro_pack/fgts_core/test_system1_protocol.py",
    )
    parser.add_argument(
        "--ckpt_path",
        default="checkpoints/AIGCDetectionBenchmark/linear_probe.pth",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
    )
    parser.add_argument(
        "--num_fake",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--num_real",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--fake_margin_min",
        type=float,
        default=4.0,
    )
    parser.add_argument(
        "--num_random",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--mode",
        choices=["mean", "blur"],
        default="mean",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protocol = load_protocol_module(args.protocol_file)

    DinoV3Model = protocol.DinoV3Model
    LinearProbe = protocol.LinearProbe

    real_idx = int(getattr(protocol, "REAL_CLASS_IDX", 0))
    fake_idx = int(getattr(protocol, "FAKE_CLASS_IDX", 1))

    print("[*] Loading DINOv3...")
    model = DinoV3Model(
        model_name="dinov3_vit_7b",
        pool_type="patch_avg",
    ).to(args.device)
    model.eval()

    num_prefix = int(
        getattr(model.backbone, "num_prefix_tokens", 5)
    )

    print("[*] Loading LinearProbe...")
    ckpt = torch.load(args.ckpt_path, map_location=args.device)

    fisher_abs_indices = [
        int(x) for x in ckpt["token_indices"]
    ]

    probe_state = ckpt.get(
        "probe_state_dict",
        ckpt.get("model_state_dict"),
    )
    if probe_state is None:
        raise RuntimeError("Checkpoint missing probe state dict")

    input_dim = probe_state["fc.weight"].shape[1]
    num_classes = probe_state["fc.weight"].shape[0]

    probe = LinearProbe(
        input_dim=input_dim,
        num_classes=num_classes,
    ).to(args.device)
    probe.load_state_dict(probe_state, strict=True)
    probe.eval()

    # absolute token index -> patch-relative index
    fisher_patch_ids = [
        idx - num_prefix
        for idx in fisher_abs_indices
        if num_prefix <= idx < num_prefix + 14 * 14
    ]

    if len(fisher_patch_ids) != len(fisher_abs_indices):
        raise RuntimeError(
            "Some Fisher indices are not valid patch tokens: "
            f"{fisher_abs_indices}"
        )

    center_patch_ids = make_center_patch_ids(
        len(fisher_patch_ids),
        fisher_patch_ids,
        grid_size=14,
    )

    print("\n== token layout ==")
    print("num_prefix:", num_prefix)
    print("fisher absolute indices:", fisher_abs_indices)
    print("fisher patch ids:", fisher_patch_ids)
    print(
        "fisher coordinates:",
        patch_ids_to_coords(fisher_patch_ids),
    )
    print(
        "center patch ids:",
        center_patch_ids,
    )
    print(
        "center coordinates:",
        patch_ids_to_coords(center_patch_ids),
    )

    df = pd.read_csv(args.input_csv)

    if "gt" not in df.columns:
        df["gt"] = df["case_name"].apply(infer_gt)

    # 优先使用已有margin筛选强fake
    fake_df = df[df["gt"] == "Fake"].copy()
    if "margin_full" in fake_df.columns:
        fake_df = fake_df[
            fake_df["margin_full"] > args.fake_margin_min
        ]

    real_df = df[df["gt"] == "Real"].copy()

    fake_n = min(args.num_fake, len(fake_df))
    real_n = min(args.num_real, len(real_df))

    fake_sample = fake_df.sample(
        n=fake_n,
        random_state=args.seed,
    )
    real_sample = real_df.sample(
        n=real_n,
        random_state=args.seed,
    )

    selected_df = pd.concat(
        [fake_sample, real_sample],
        ignore_index=True,
    )

    selected_df = selected_df.sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    all_patch_ids = list(range(14 * 14))
    random_pool = [
        pid for pid in all_patch_ids
        if pid not in set(fisher_patch_ids)
    ]

    rows = []

    for row_idx, row in selected_df.iterrows():
        case_name = str(row["case_name"])
        gt = str(row["gt"])

        try:
            img_path = find_case_image(
                args.cases_root,
                case_name,
            )
            original = load_raw_image(img_path)

            variants = [original]

            fisher_variant = perturb_patches(
                original,
                fisher_patch_ids,
                mode=args.mode,
            )
            center_variant = perturb_patches(
                original,
                center_patch_ids,
                mode=args.mode,
            )

            variants.append(fisher_variant)
            variants.append(center_variant)

            rng = random.Random(args.seed + row_idx * 1009)
            random_sets = []

            for _ in range(args.num_random):
                patch_ids = rng.sample(
                    random_pool,
                    k=len(fisher_patch_ids),
                )
                random_sets.append(patch_ids)
                variants.append(
                    perturb_patches(
                        original,
                        patch_ids,
                        mode=args.mode,
                    )
                )

            margins, fake_probs = infer_batch(
                variants,
                model,
                probe,
                fisher_abs_indices,
                real_idx,
                fake_idx,
                args.device,
                args.batch_size,
            )

            baseline_margin = float(margins[0])
            fisher_margin = float(margins[1])
            center_margin = float(margins[2])
            random_margins = margins[3:]

            fisher_drop = baseline_margin - fisher_margin
            center_drop = baseline_margin - center_margin
            random_drops = baseline_margin - random_margins

            random_drop_mean = float(np.mean(random_drops))
            random_drop_std = float(np.std(random_drops))

            out_row = {
                "case_name": case_name,
                "gt": gt,
                "image_path": img_path,
                "mode": args.mode,
                "baseline_fake_prob": float(fake_probs[0]),
                "baseline_margin": baseline_margin,
                "fisher_anchor_margin": fisher_margin,
                "center_margin": center_margin,
                "fisher_anchor_drop": fisher_drop,
                "center_drop": center_drop,
                "random_drop_mean": random_drop_mean,
                "random_drop_std": random_drop_std,
                "fisher_adv_over_random": (
                    fisher_drop - random_drop_mean
                ),
                "fisher_adv_over_center": (
                    fisher_drop - center_drop
                ),
                "random_drops_json": json.dumps(
                    [float(x) for x in random_drops]
                ),
                "fisher_patch_ids_json": json.dumps(
                    fisher_patch_ids
                ),
                "center_patch_ids_json": json.dumps(
                    center_patch_ids
                ),
            }

            rows.append(out_row)

            print(
                f"[{row_idx+1:03d}/{len(selected_df):03d}] "
                f"{gt:4s} "
                f"base={baseline_margin:+.4f} "
                f"fisher_drop={fisher_drop:+.4f} "
                f"center_drop={center_drop:+.4f} "
                f"random={random_drop_mean:+.4f}"
            )

        except Exception as exc:
            print(f"[!] failed {case_name}: {exc}")

    result_df = pd.DataFrame(rows)

    result_csv = os.path.join(
        args.output_dir,
        "input_space_edge_diagnostic_results.csv",
    )
    result_df.to_csv(
        result_csv,
        index=False,
        encoding="utf-8-sig",
    )

    strong_fake = result_df[
        (result_df["gt"] == "Fake")
        & (result_df["baseline_fake_prob"] >= 0.5)
        & (result_df["baseline_margin"] > args.fake_margin_min)
    ]

    summaries = [
        summarize_group(result_df, "all"),
        summarize_group(
            result_df[result_df["gt"] == "Fake"],
            "fake",
        ),
        summarize_group(
            result_df[result_df["gt"] == "Real"],
            "real",
        ),
        summarize_group(
            strong_fake,
            "strong_fake",
        ),
    ]

    summary_json = os.path.join(
        args.output_dir,
        "input_space_edge_diagnostic_summary.json",
    )
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "num_prefix": num_prefix,
                "fisher_abs_indices": fisher_abs_indices,
                "fisher_patch_ids": fisher_patch_ids,
                "fisher_coordinates": patch_ids_to_coords(
                    fisher_patch_ids
                ),
                "center_patch_ids": center_patch_ids,
                "center_coordinates": patch_ids_to_coords(
                    center_patch_ids
                ),
                "groups": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n== summary ==")
    for item in summaries:
        print(json.dumps(item, indent=2, ensure_ascii=False))

    print("\nsaved:", result_csv)
    print("saved:", summary_json)


if __name__ == "__main__":
    main()
