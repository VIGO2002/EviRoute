#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
from torchvision.transforms import functional as TF

from run_input_space_edge_diagnostic_v1 import (
    load_protocol_module,
    infer_gt,
    find_case_image,
    load_raw_image,
    infer_batch,
)


IMAGE_SIZE = 224
PATCH_SIZE = 16
GRID = IMAGE_SIZE // PATCH_SIZE


def parse_int_list(text):
    return [
        int(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]


def make_rect_mask(y0, y1, x0, x1, device):
    mask = torch.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.bool,
        device=device,
    )
    mask[y0:y1, x0:x1] = True
    return mask


def make_border_masks(width, device):
    center_y0 = (IMAGE_SIZE - width) // 2
    center_x0 = (IMAGE_SIZE - width) // 2

    top = make_rect_mask(
        0, width, 0, IMAGE_SIZE, device
    )
    bottom = make_rect_mask(
        IMAGE_SIZE - width,
        IMAGE_SIZE,
        0,
        IMAGE_SIZE,
        device,
    )
    left = make_rect_mask(
        0, IMAGE_SIZE, 0, width, device
    )
    right = make_rect_mask(
        0,
        IMAGE_SIZE,
        IMAGE_SIZE - width,
        IMAGE_SIZE,
        device,
    )

    all_border = top | bottom | left | right

    # 与单个水平边带形状、面积一致的中心控制。
    center_h = make_rect_mask(
        center_y0,
        center_y0 + width,
        0,
        IMAGE_SIZE,
        device,
    )

    # 与单个垂直边带形状、面积一致的中心控制。
    center_v = make_rect_mask(
        0,
        IMAGE_SIZE,
        center_x0,
        center_x0 + width,
        device,
    )

    return {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
        "all_border": all_border,
        "center_h": center_h,
        "center_v": center_v,
    }


def make_random_interior_patch_mask(
    width,
    rng,
    device,
):
    """
    从不属于 width 边缘环的内部 patch 中，
    随机选择与外边缘环相同数量的 16×16 patch。
    width 只支持16或32。
    """
    ring = width // PATCH_SIZE

    border_patch_ids = []
    interior_patch_ids = []

    for row in range(GRID):
        for col in range(GRID):
            patch_id = row * GRID + col

            is_border = (
                row < ring
                or row >= GRID - ring
                or col < ring
                or col >= GRID - ring
            )

            if is_border:
                border_patch_ids.append(patch_id)
            else:
                interior_patch_ids.append(patch_id)

    if len(border_patch_ids) > len(interior_patch_ids):
        raise ValueError(
            f"Cannot create matched interior control "
            f"for width={width}: "
            f"border={len(border_patch_ids)}, "
            f"interior={len(interior_patch_ids)}"
        )

    selected = rng.sample(
        interior_patch_ids,
        len(border_patch_ids),
    )

    mask = torch.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=torch.bool,
        device=device,
    )

    for patch_id in selected:
        row, col = divmod(patch_id, GRID)

        y0 = row * PATCH_SIZE
        y1 = y0 + PATCH_SIZE
        x0 = col * PATCH_SIZE
        x1 = x0 + PATCH_SIZE

        mask[y0:y1, x0:x1] = True

    return mask


def blur_masked_region(image, mask):
    blurred = TF.gaussian_blur(
        image,
        kernel_size=[17, 17],
        sigma=[5.0, 5.0],
    )

    output = image.clone()
    output[:, mask] = blurred[:, mask]
    return output


def prediction_from_margin(margin):
    return int(margin > 0)


def summarize(df, width, gt):
    sub = df[df["width"] == width]

    if gt != "All":
        sub = sub[sub["gt"] == gt]

    if len(sub) == 0:
        return {
            "width": int(width),
            "group": gt,
            "n": 0,
        }

    fields = [
        "top_support_drop",
        "bottom_support_drop",
        "left_support_drop",
        "right_support_drop",
        "all_border_support_drop",
        "center_h_support_drop",
        "center_v_support_drop",
        "random_interior_support_drop_mean",
        "all_border_adv_over_random",
        "top_adv_over_center_h",
        "bottom_adv_over_center_h",
        "left_adv_over_center_v",
        "right_adv_over_center_v",
    ]

    result = {
        "width": int(width),
        "group": gt,
        "n": int(len(sub)),
        "all_border_prediction_flip_rate":
            float(sub["all_border_prediction_flip"].mean()),
        "random_interior_prediction_flip_rate":
            float(
                sub[
                    "random_interior_prediction_flip_rate"
                ].mean()
            ),
        "all_border_gt_random_rate":
            float(
                (
                    sub["all_border_adv_over_random"] > 0
                ).mean()
            ),
    }

    for field in fields:
        result[f"mean_{field}"] = float(
            sub[field].mean()
        )

    return result


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--input_csv", required=True)

    ap.add_argument(
        "--cases_root",
        default="aigi_now_eval/probe_genimage/cases_unique",
    )
    ap.add_argument(
        "--protocol_file",
        default="repro_pack/fgts_core/test_system1_protocol.py",
    )
    ap.add_argument(
        "--ckpt_path",
        default="checkpoints/AIGCDetectionBenchmark/linear_probe.pth",
    )
    ap.add_argument("--output_dir", required=True)

    ap.add_argument("--widths", default="16,32")
    ap.add_argument("--num_fake", type=int, default=20)
    ap.add_argument("--num_real", type=int, default=20)
    ap.add_argument("--fake_margin_min", type=float, default=4.0)

    ap.add_argument(
        "--num_random",
        type=int,
        default=20,
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2)

    ap.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    widths = parse_int_list(args.widths)

    for width in widths:
        if width not in (16, 32):
            raise ValueError(
                "For matched controls, widths must be 16 or 32"
            )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    protocol = load_protocol_module(
        args.protocol_file
    )

    DinoV3Model = protocol.DinoV3Model
    LinearProbe = protocol.LinearProbe

    real_idx = int(
        getattr(protocol, "REAL_CLASS_IDX", 0)
    )
    fake_idx = int(
        getattr(protocol, "FAKE_CLASS_IDX", 1)
    )

    print("[*] Loading DINOv3...")
    model = DinoV3Model(
        model_name="dinov3_vit_7b",
        pool_type="patch_avg",
    ).to(args.device)
    model.eval()

    print("[*] Loading LinearProbe...")
    checkpoint = torch.load(
        args.ckpt_path,
        map_location=args.device,
    )

    fisher_abs_indices = [
        int(x)
        for x in checkpoint["token_indices"]
    ]

    state = checkpoint.get(
        "probe_state_dict",
        checkpoint.get("model_state_dict"),
    )

    if state is None:
        raise RuntimeError(
            "Checkpoint missing probe weights"
        )

    probe = LinearProbe(
        input_dim=state["fc.weight"].shape[1],
        num_classes=state["fc.weight"].shape[0],
    ).to(args.device)

    probe.load_state_dict(state, strict=True)
    probe.eval()

    data = pd.read_csv(args.input_csv)

    if "gt" not in data.columns:
        data["gt"] = data["case_name"].apply(
            infer_gt
        )

    fake_data = data[
        data["gt"] == "Fake"
    ].copy()

    if "margin_full" in fake_data.columns:
        fake_data = fake_data[
            fake_data["margin_full"]
            > args.fake_margin_min
        ]

    real_data = data[
        data["gt"] == "Real"
    ].copy()

    fake_sample = fake_data.sample(
        n=min(args.num_fake, len(fake_data)),
        random_state=args.seed,
    )

    real_sample = real_data.sample(
        n=min(args.num_real, len(real_data)),
        random_state=args.seed,
    )

    selected = pd.concat(
        [fake_sample, real_sample],
        ignore_index=True,
    ).sample(
        frac=1.0,
        random_state=args.seed,
    ).reset_index(drop=True)

    selected.to_csv(
        os.path.join(
            args.output_dir,
            "selected_cases.csv",
        ),
        index=False,
        encoding="utf-8-sig",
    )

    rows = []

    for sample_idx, row in selected.iterrows():
        case_name = str(row["case_name"])
        gt = str(row["gt"])

        try:
            image_path = find_case_image(
                args.cases_root,
                case_name,
            )
            original = load_raw_image(image_path)

            for width in widths:
                masks = make_border_masks(
                    width,
                    original.device,
                )

                variant_names = [
                    "original",
                    "top",
                    "bottom",
                    "left",
                    "right",
                    "all_border",
                    "center_h",
                    "center_v",
                ]

                variants = [original]

                for name in variant_names[1:]:
                    variants.append(
                        blur_masked_region(
                            original,
                            masks[name],
                        )
                    )

                rng = random.Random(
                    args.seed
                    + sample_idx * 1009
                    + width * 37
                )

                random_masks = []

                for _ in range(args.num_random):
                    random_mask = (
                        make_random_interior_patch_mask(
                            width,
                            rng,
                            original.device,
                        )
                    )

                    random_masks.append(random_mask)

                    variants.append(
                        blur_masked_region(
                            original,
                            random_mask,
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

                original_margin = float(margins[0])

                margin_by_name = {
                    name: float(margins[index])
                    for index, name
                    in enumerate(variant_names)
                }

                random_margins = np.asarray(
                    margins[len(variant_names):],
                    dtype=np.float64,
                )

                class_sign = (
                    1.0 if gt == "Fake" else -1.0
                )

                def support_drop(margin):
                    return class_sign * (
                        original_margin - margin
                    )

                drops = {
                    name: support_drop(
                        margin_by_name[name]
                    )
                    for name in variant_names[1:]
                }

                random_drops = class_sign * (
                    original_margin
                    - random_margins
                )

                original_prediction = (
                    prediction_from_margin(
                        original_margin
                    )
                )

                random_flip_rate = float(
                    np.mean(
                        [
                            prediction_from_margin(x)
                            != original_prediction
                            for x in random_margins
                        ]
                    )
                )

                output = {
                    "case_name": case_name,
                    "gt": gt,
                    "width": int(width),

                    "original_margin":
                        original_margin,

                    "original_fake_prob":
                        float(fake_probs[0]),

                    "top_support_drop":
                        float(drops["top"]),

                    "bottom_support_drop":
                        float(drops["bottom"]),

                    "left_support_drop":
                        float(drops["left"]),

                    "right_support_drop":
                        float(drops["right"]),

                    "all_border_support_drop":
                        float(drops["all_border"]),

                    "center_h_support_drop":
                        float(drops["center_h"]),

                    "center_v_support_drop":
                        float(drops["center_v"]),

                    "random_interior_support_drop_mean":
                        float(random_drops.mean()),

                    "random_interior_support_drop_std":
                        float(random_drops.std()),

                    "all_border_adv_over_random":
                        float(
                            drops["all_border"]
                            - random_drops.mean()
                        ),

                    "top_adv_over_center_h":
                        float(
                            drops["top"]
                            - drops["center_h"]
                        ),

                    "bottom_adv_over_center_h":
                        float(
                            drops["bottom"]
                            - drops["center_h"]
                        ),

                    "left_adv_over_center_v":
                        float(
                            drops["left"]
                            - drops["center_v"]
                        ),

                    "right_adv_over_center_v":
                        float(
                            drops["right"]
                            - drops["center_v"]
                        ),

                    "all_border_prediction_flip":
                        bool(
                            prediction_from_margin(
                                margin_by_name[
                                    "all_border"
                                ]
                            )
                            != original_prediction
                        ),

                    "random_interior_prediction_flip_rate":
                        random_flip_rate,

                    "random_drops_json":
                        json.dumps(
                            [
                                float(x)
                                for x in random_drops
                            ]
                        ),
                }

                rows.append(output)

                print(
                    f"[{sample_idx+1:03d}/"
                    f"{len(selected):03d}] "
                    f"{gt:4s} "
                    f"width={width:2d} "
                    f"border_drop="
                    f"{drops['all_border']:+.3f} "
                    f"random="
                    f"{random_drops.mean():+.3f} "
                    f"adv="
                    f"{output['all_border_adv_over_random']:+.3f}"
                )

        except Exception as exc:
            print(
                f"[!] failed {case_name}: {exc}"
            )

    result = pd.DataFrame(rows)

    result_path = os.path.join(
        args.output_dir,
        "border_band_ablation_results.csv",
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    summaries = []

    for width in widths:
        for group in ["Fake", "Real", "All"]:
            summaries.append(
                summarize(
                    result,
                    width,
                    group,
                )
            )

    summary_path = os.path.join(
        args.output_dir,
        "border_band_ablation_summary.json",
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "widths": widths,
                "num_random": args.num_random,
                "groups": summaries,
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    print("\n== summary ==")
    print(
        json.dumps(
            summaries,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nsaved:", result_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
