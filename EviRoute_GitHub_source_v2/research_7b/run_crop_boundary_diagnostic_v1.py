#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import json
import random
import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_input_space_edge_diagnostic_v1 import (
    load_protocol_module,
    find_case_image,
    load_raw_image,
    infer_batch,
)

from run_coarse_input_causal_map_v1 import (
    GRID,
    perturb_block,
)


IMAGE_SIZE = 224
BLOCK_SIZE = IMAGE_SIZE // GRID


def safe_name(text):
    return re.sub(r"[^0-9A-Za-z._-]+", "_", str(text))


def parse_int_list(text):
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text):
    return [x.strip() for x in str(text).split(",") if x.strip()]


def safe_corr(a, b):
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)

    valid = np.isfinite(a) & np.isfinite(b)
    a = a[valid]
    b = b[valid]

    if len(a) < 3:
        return np.nan

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return np.nan

    return float(np.corrcoef(a, b)[0, 1])


def upsample_map(map_7x7):
    return np.repeat(
        np.repeat(map_7x7, BLOCK_SIZE, axis=0),
        BLOCK_SIZE,
        axis=1,
    )


def resize_numpy_map(array, height, width):
    tensor = torch.tensor(
        array,
        dtype=torch.float32,
    )[None, None]

    resized = F.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )

    return resized[0, 0].numpy()


def crop_resize(image, crop_px):
    cropped = image[
        :,
        crop_px:IMAGE_SIZE - crop_px,
        crop_px:IMAGE_SIZE - crop_px,
    ]

    return TF.resize(
        cropped,
        [IMAGE_SIZE, IMAGE_SIZE],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )


def crop_reflect_pad(image, crop_px):
    cropped = image[
        :,
        crop_px:IMAGE_SIZE - crop_px,
        crop_px:IMAGE_SIZE - crop_px,
    ].unsqueeze(0)

    restored = F.pad(
        cropped,
        (crop_px, crop_px, crop_px, crop_px),
        mode="reflect",
    )

    return restored[0]


def map_crop_resize_back_to_original(transformed_map, crop_px):
    """
    transformed_map 是裁剪后重新 resize 到 224×224 的因果图。
    将其重新缩回原始裁剪区域，并放回原图坐标系。
    """
    transformed_pixel_map = upsample_map(transformed_map)

    crop_size = IMAGE_SIZE - 2 * crop_px
    mapped_crop = resize_numpy_map(
        transformed_pixel_map,
        crop_size,
        crop_size,
    )

    mapped_full = np.full(
        (IMAGE_SIZE, IMAGE_SIZE),
        np.nan,
        dtype=np.float32,
    )

    mapped_full[
        crop_px:IMAGE_SIZE - crop_px,
        crop_px:IMAGE_SIZE - crop_px,
    ] = mapped_crop

    return mapped_full


def grid_edge_metrics(heatmap):
    top_order = np.argsort(heatmap.reshape(-1))[::-1]

    top1 = int(top_order[0])
    top1_row, top1_col = divmod(top1, GRID)

    top3 = []
    for idx in top_order[:3]:
        row, col = divmod(int(idx), GRID)
        top3.append((row, col))

    def is_edge(row, col):
        return (
            row == 0
            or row == GRID - 1
            or col == 0
            or col == GRID - 1
        )

    edge_mask = np.zeros((GRID, GRID), dtype=bool)
    for row in range(GRID):
        for col in range(GRID):
            edge_mask[row, col] = is_edge(row, col)

    positive = np.maximum(heatmap, 0.0)
    positive_sum = float(positive.sum())

    if positive_sum > 1e-12:
        positive_edge_mass_fraction = float(
            positive[edge_mask].sum() / positive_sum
        )
    else:
        positive_edge_mass_fraction = 0.0

    return {
        "top1_row": int(top1_row),
        "top1_col": int(top1_col),
        "top1_is_edge": bool(is_edge(top1_row, top1_col)),
        "top3_edge_fraction": float(
            np.mean([is_edge(r, c) for r, c in top3])
        ),
        "positive_edge_mass_fraction":
            positive_edge_mass_fraction,
    }


def exact_border_mass_fraction(pixel_map, border_width):
    positive = np.maximum(
        np.nan_to_num(pixel_map, nan=0.0),
        0.0,
    )

    total = float(positive.sum())
    if total <= 1e-12:
        return 0.0

    mask = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=bool,
    )

    mask[:border_width, :] = True
    mask[-border_width:, :] = True
    mask[:, :border_width] = True
    mask[:, -border_width:] = True

    return float(positive[mask].sum() / total)


def save_tensor_image(image_tensor, path):
    TF.to_pil_image(
        image_tensor.detach().cpu().clamp(0, 1)
    ).save(path)


def save_overlay(image_tensor, pixel_map, path, title):
    image = (
        image_tensor.detach()
        .cpu()
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )

    heatmap = np.maximum(
        np.nan_to_num(pixel_map, nan=0.0),
        0.0,
    )

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image)

    if float(heatmap.max()) > 0:
        ax.imshow(
            heatmap,
            cmap="jet",
            alpha=0.45,
            vmin=0,
            vmax=float(heatmap.max()),
        )

    ax.set_title(title)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.inference_mode()
def compute_causal_map(
    image,
    gt,
    mode,
    model,
    probe,
    fisher_abs_indices,
    real_idx,
    fake_idx,
    device,
    batch_size,
):
    variants = [image]

    for row in range(GRID):
        for col in range(GRID):
            variants.append(
                perturb_block(
                    image,
                    row,
                    col,
                    mode=mode,
                )
            )

    margins, fake_probs = infer_batch(
        variants,
        model,
        probe,
        fisher_abs_indices,
        real_idx,
        fake_idx,
        device,
        batch_size,
    )

    baseline_margin = float(margins[0])
    perturbed_margins = margins[1:].reshape(GRID, GRID)

    class_sign = 1.0 if gt == "Fake" else -1.0

    support_drop_map = class_sign * (
        baseline_margin - perturbed_margins
    )

    return (
        support_drop_map,
        baseline_margin,
        float(fake_probs[0]),
    )


def summarize_group(df, variant, crop_px, gt):
    sub = df[
        (df["variant"] == variant)
        & (df["crop_px"] == crop_px)
    ]

    if gt != "All":
        sub = sub[sub["gt"] == gt]

    if len(sub) == 0:
        return {
            "variant": variant,
            "crop_px": int(crop_px),
            "group": gt,
            "n": 0,
        }

    return {
        "variant": variant,
        "crop_px": int(crop_px),
        "group": gt,
        "n": int(len(sub)),

        "prediction_consistency":
            float(sub["prediction_consistent"].mean()),

        "mean_abs_margin_change":
            float(sub["margin_abs_change"].mean()),

        "mean_content_aligned_corr":
            float(sub["content_aligned_corr"].mean()),

        "mean_frame_aligned_corr":
            float(sub["frame_aligned_corr"].mean()),

        "mean_alignment_advantage":
            float(sub["alignment_advantage"].mean()),

        "content_corr_higher_rate":
            float(
                (
                    sub["content_aligned_corr"]
                    > sub["frame_aligned_corr"]
                ).mean()
            ),

        "top1_edge_rate":
            float(sub["top1_is_edge"].mean()),

        "mean_top3_edge_fraction":
            float(sub["top3_edge_fraction"].mean()),

        "mean_positive_edge_mass_fraction":
            float(
                sub["positive_edge_mass_fraction"].mean()
            ),

        "mean_exact_border_mass_fraction":
            float(
                sub["exact_border_mass_fraction"].mean()
            ),

        "mean_edge_mass_change":
            float(sub["edge_mass_change"].mean()),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--baseline_results",
        required=True,
        help="Previous blur coarse causal result CSV",
    )
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

    ap.add_argument(
        "--crop_sizes",
        default="32",
        help="Example: 16,32,48",
    )
    ap.add_argument(
        "--variants",
        default="crop_resize,reflect_pad",
    )
    ap.add_argument(
        "--mode",
        choices=["blur", "local_mean"],
        default="blur",
    )

    ap.add_argument("--num_fake", type=int, default=10)
    ap.add_argument("--num_real", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=2)

    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )

    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    crop_sizes = parse_int_list(args.crop_sizes)
    variants = parse_str_list(args.variants)

    allowed = {"crop_resize", "reflect_pad"}
    unknown = set(variants) - allowed
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}")

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

    print("[*] Loading LinearProbe...")
    ckpt = torch.load(
        args.ckpt_path,
        map_location=args.device,
    )

    fisher_abs_indices = [
        int(x) for x in ckpt["token_indices"]
    ]

    state = ckpt.get(
        "probe_state_dict",
        ckpt.get("model_state_dict"),
    )

    if state is None:
        raise RuntimeError("Checkpoint missing probe weights")

    probe = LinearProbe(
        input_dim=state["fc.weight"].shape[1],
        num_classes=state["fc.weight"].shape[0],
    ).to(args.device)

    probe.load_state_dict(state, strict=True)
    probe.eval()

    baseline = pd.read_csv(args.baseline_results)

    fake_rows = baseline[baseline["gt"] == "Fake"]
    real_rows = baseline[baseline["gt"] == "Real"]

    fake_sample = fake_rows.sample(
        n=min(args.num_fake, len(fake_rows)),
        random_state=args.seed,
    )

    real_sample = real_rows.sample(
        n=min(args.num_real, len(real_rows)),
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
        os.path.join(args.output_dir, "selected_cases.csv"),
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

            baseline_map_path = os.path.join(
                str(row["case_output_dir"]),
                "support_drop_map.npy",
            )

            if not os.path.exists(baseline_map_path):
                raise FileNotFoundError(
                    f"Missing baseline map: {baseline_map_path}"
                )

            original_map = np.load(baseline_map_path)
            original_pixel_map = upsample_map(original_map)
            original_metrics = grid_edge_metrics(original_map)

            case_root = os.path.join(
                args.output_dir,
                "cases",
                safe_name(case_name),
            )
            os.makedirs(case_root, exist_ok=True)

            for crop_px in crop_sizes:
                if crop_px <= 0 or crop_px >= IMAGE_SIZE // 2:
                    raise ValueError(
                        f"Invalid crop size: {crop_px}"
                    )

                for variant in variants:
                    if variant == "crop_resize":
                        transformed = crop_resize(
                            original,
                            crop_px,
                        )
                    elif variant == "reflect_pad":
                        transformed = crop_reflect_pad(
                            original,
                            crop_px,
                        )
                    else:
                        raise ValueError(variant)

                    (
                        transformed_map,
                        transformed_margin,
                        transformed_fake_prob,
                    ) = compute_causal_map(
                        transformed,
                        gt,
                        args.mode,
                        model,
                        probe,
                        fisher_abs_indices,
                        real_idx,
                        fake_idx,
                        args.device,
                        args.batch_size,
                    )

                    transformed_pixel_map = upsample_map(
                        transformed_map
                    )

                    transformed_metrics = grid_edge_metrics(
                        transformed_map
                    )

                    if variant == "crop_resize":
                        mapped_back = (
                            map_crop_resize_back_to_original(
                                transformed_map,
                                crop_px,
                            )
                        )

                        center_slice = (
                            slice(crop_px, IMAGE_SIZE - crop_px),
                            slice(crop_px, IMAGE_SIZE - crop_px),
                        )

                        # 内容坐标对齐：将裁剪图热图映射回原图。
                        content_corr = safe_corr(
                            original_pixel_map[center_slice],
                            mapped_back[center_slice],
                        )

                        # 画布坐标对齐：不进行几何映射。
                        frame_corr = safe_corr(
                            original_pixel_map[center_slice],
                            transformed_pixel_map[center_slice],
                        )

                    else:
                        # reflect pad 中心内容坐标没有缩放。
                        mapped_back = transformed_pixel_map

                        center_slice = (
                            slice(crop_px, IMAGE_SIZE - crop_px),
                            slice(crop_px, IMAGE_SIZE - crop_px),
                        )

                        content_corr = safe_corr(
                            original_pixel_map[center_slice],
                            transformed_pixel_map[center_slice],
                        )

                        frame_corr = safe_corr(
                            original_pixel_map,
                            transformed_pixel_map,
                        )

                    baseline_margin = float(
                        row["baseline_margin"]
                    )

                    prediction_original = baseline_margin > 0
                    prediction_transformed = (
                        transformed_margin > 0
                    )

                    exact_border_fraction = (
                        exact_border_mass_fraction(
                            transformed_pixel_map,
                            crop_px,
                        )
                    )

                    output_dir = os.path.join(
                        case_root,
                        f"{variant}_crop{crop_px}",
                    )
                    os.makedirs(output_dir, exist_ok=True)

                    np.save(
                        os.path.join(
                            output_dir,
                            "transformed_support_drop_map.npy",
                        ),
                        transformed_map,
                    )

                    pd.DataFrame(
                        transformed_map
                    ).to_csv(
                        os.path.join(
                            output_dir,
                            "transformed_support_drop_map.csv",
                        ),
                        index=False,
                        header=False,
                    )

                    save_tensor_image(
                        transformed,
                        os.path.join(
                            output_dir,
                            "transformed_image.png",
                        ),
                    )

                    save_overlay(
                        transformed,
                        transformed_pixel_map,
                        os.path.join(
                            output_dir,
                            "transformed_causal_overlay.png",
                        ),
                        (
                            f"{gt} | {variant} | crop={crop_px} | "
                            f"margin={transformed_margin:.3f}"
                        ),
                    )

                    save_overlay(
                        original,
                        mapped_back,
                        os.path.join(
                            output_dir,
                            "mapped_back_overlay.png",
                        ),
                        (
                            f"Mapped to original | "
                            f"{variant} | crop={crop_px}"
                        ),
                    )

                    out_row = {
                        "case_name": case_name,
                        "gt": gt,
                        "variant": variant,
                        "crop_px": int(crop_px),
                        "mode": args.mode,

                        "original_margin": baseline_margin,
                        "transformed_margin":
                            float(transformed_margin),

                        "transformed_fake_prob":
                            float(transformed_fake_prob),

                        "margin_abs_change":
                            abs(
                                transformed_margin
                                - baseline_margin
                            ),

                        "prediction_consistent":
                            prediction_original
                            == prediction_transformed,

                        "content_aligned_corr":
                            float(content_corr),

                        "frame_aligned_corr":
                            float(frame_corr),

                        "alignment_advantage":
                            float(content_corr - frame_corr),

                        "original_top1_is_edge":
                            original_metrics["top1_is_edge"],

                        "original_positive_edge_mass_fraction":
                            original_metrics[
                                "positive_edge_mass_fraction"
                            ],

                        "top1_row":
                            transformed_metrics["top1_row"],

                        "top1_col":
                            transformed_metrics["top1_col"],

                        "top1_is_edge":
                            transformed_metrics["top1_is_edge"],

                        "top3_edge_fraction":
                            transformed_metrics[
                                "top3_edge_fraction"
                            ],

                        "positive_edge_mass_fraction":
                            transformed_metrics[
                                "positive_edge_mass_fraction"
                            ],

                        "exact_border_mass_fraction":
                            exact_border_fraction,

                        "edge_mass_change":
                            transformed_metrics[
                                "positive_edge_mass_fraction"
                            ]
                            - original_metrics[
                                "positive_edge_mass_fraction"
                            ],

                        "case_output_dir": output_dir,
                    }

                    rows.append(out_row)

                    print(
                        f"[{sample_idx+1:02d}/{len(selected):02d}] "
                        f"{gt:4s} "
                        f"{variant:11s} "
                        f"crop={crop_px:2d} "
                        f"margin={baseline_margin:+.2f}"
                        f"->{transformed_margin:+.2f} "
                        f"content={content_corr:+.3f} "
                        f"frame={frame_corr:+.3f} "
                        f"edge={transformed_metrics['top1_is_edge']}"
                    )

        except Exception as exc:
            print(f"[!] failed {case_name}: {exc}")

    result = pd.DataFrame(rows)

    result_path = os.path.join(
        args.output_dir,
        "crop_boundary_diagnostic_results.csv",
    )

    result.to_csv(
        result_path,
        index=False,
        encoding="utf-8-sig",
    )

    summaries = []

    for variant in variants:
        for crop_px in crop_sizes:
            for group in ["Fake", "Real", "All"]:
                summaries.append(
                    summarize_group(
                        result,
                        variant,
                        crop_px,
                        group,
                    )
                )

    summary_path = os.path.join(
        args.output_dir,
        "crop_boundary_diagnostic_summary.json",
    )

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": args.mode,
                "crop_sizes": crop_sizes,
                "variants": variants,
                "groups": summaries,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n== summary ==")
    print(json.dumps(
        summaries,
        indent=2,
        ensure_ascii=False,
    ))

    print("\nsaved:", result_path)
    print("saved:", summary_path)


if __name__ == "__main__":
    main()
